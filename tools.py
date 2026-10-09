"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import datetime
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree  # (arXiv answers with Atom XML)

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()   # EXA_API_KEY stays on the host: only these tools read it

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SUMMARY_CHARS = 600
PAGE_CHARS = 12_000
ARXIV_GAP_S = 3.0
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_LOCK = threading.Lock()   # researchers run in parallel threads: the 3 s gap must hold across all of them
_arxiv_last_call = 0.0


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again (at most `attempts` calls in total).

    Wait = the server's Retry-After when given, else base * 2**attempt plus random jitter; never more than `cap`.
    The last failure is re-raised without sleeping. Any other exception propagates immediately.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, exc.retry_after)
            else:
                delay = min(cap, base * 2 ** attempt + random.uniform(0, base))
            time.sleep(max(0.0, delay))


def _retry_after(response):
    try:
        return float(response.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None   # absent, or an HTTP date: fall back to exponential backoff


def _request(method, url, **kwargs):
    """One HTTP call. Temporary failures (network, 429, 5xx) become RetryableError; other HTTP errors raise."""
    try:
        response = httpx.request(method, url, timeout=60, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {url}", _retry_after(response))
    response.raise_for_status()
    return response


def _clean(text):
    return " ".join(str(text or "").split())


def _clamp(value, low, high):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return low


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 2: arXiv ----
def _arxiv_terms(query):
    """LLM-written query -> plain search terms: no field prefixes (all:, ti:), no boolean operators, no quotes."""
    query = re.sub(r"\b(?:all|ti|abs|au|cat|co|jr|rn|id):", " ", query or "", flags=re.I)
    terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", query) if t not in {"AND", "OR", "NOT", "ANDNOT"}]
    return terms[:8]


def _arxiv_get(params):
    global _arxiv_last_call
    with _ARXIV_LOCK:
        wait = _arxiv_last_call + ARXIV_GAP_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _arxiv_last_call = time.monotonic()
    return _request("GET", ARXIV_URL, params=params)


def _arxiv_records(xml_text):
    records = []
    for entry in xml.etree.ElementTree.fromstring(xml_text).findall("a:entry", _ATOM):
        raw_id = entry.findtext("a:id", "", _ATOM).rsplit("/abs/", 1)[-1]
        paper_id = re.sub(r"v\d+$", "", raw_id.strip())
        if not paper_id:
            continue
        records.append({
            "id": paper_id,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "published": entry.findtext("a:published", "", _ATOM)[:10],
            "title": _clean(entry.findtext("a:title", "", _ATOM)),
            "summary": _clean(entry.findtext("a:summary", "", _ATOM))[:SUMMARY_CHARS],
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few plain keywords (e.g. "world model video prediction"), newest first.
    Returns a JSON list of {id, url, published, title, summary}; url is https://arxiv.org/abs/<id>.
    Use 2-5 keywords: every keyword must appear in the paper, so long queries return NO RESULTS."""
    try:
        terms = _arxiv_terms(query)
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
                  "sortOrder": "descending", "start": 0, "max_results": _clamp(max_results, 1, 30)}
        # arXiv rate-limits per IP (a whole class behind one network): long cap, more attempts
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        records = _arxiv_records(response.text)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # a tool never raises: the agent reads the error and switches source
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_records(items):
    records = []
    for item in items if isinstance(items, list) else []:
        paper = item.get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(paper.get("ai_summary") or paper.get("summary") or item.get("summary"))[:SUMMARY_CHARS],
            "upvotes": paper.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "",
            "stars": paper.get("githubStars") or 0,
        })
    return records


def _hf_daily_day(day):
    params = {"limit": 100, **({"date": day} if day else {})}
    return _hf_records(with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json())


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "", days: int = 1) -> str:
    """Hugging Face Daily Papers = what is trending in AI research right now (recent papers with community upvotes).
    Returns a JSON list of {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes;
    url is https://huggingface.co/papers/<id>. `date` is YYYY-MM-DD (empty = latest day).
    `days` (1-21) scans that many days back from `date` (or from today), e.g. days=14 for the last two weeks.
    `keyword` keeps papers whose title/summary contain ALL its words (use 1-2 short words, e.g. "world model");
    there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        days = _clamp(days, 1, 21)
        if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.strip()):
            return "ERROR: date must be YYYY-MM-DD"
        if days == 1:
            day_list = [date.strip()]
        else:
            end = datetime.date.fromisoformat(date.strip()) if date else datetime.datetime.now(datetime.UTC).date()
            day_list = [(end - datetime.timedelta(days=i)).isoformat() for i in range(days)]
        by_id = {}
        for day in day_list:
            for record in _hf_daily_day(day):
                by_id.setdefault(record["id"], record)
        records = list(by_id.values())
        words = keyword.lower().split()
        if words:
            records = [r for r in records if all(w in f"{r['title']} {r['summary']}".lower() for w in words)]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        records = records[:_clamp(limit, 1, 100)]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (semantic search over arXiv papers indexed by Hugging Face).
    Returns a JSON list of {id, url, published, title, summary, upvotes, github, stars};
    url is https://huggingface.co/papers/<id>. Good for finding well-known and popular papers on a topic."""
    try:
        if not (query or "").strip():
            return "NO RESULTS"
        params = {"q": query.strip(), "limit": _clamp(limit, 1, 50)}
        records = _hf_records(with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params)).json())
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _exa_key():
    return (os.getenv("EXA_API_KEY") or "").strip()


def _redact(text):
    key = _exa_key()
    text = re.sub(r"(exaApiKey=)[^&\s'\"]+", r"\1***", str(text))
    return text.replace(key, "***") if key else text


def _looks_rate_limited(meta, text):
    """The free tier may answer HTTP 200 with a rate-limit notice as the 'content' and a flag in result._meta."""
    if re.search(r"rate.?limit", json.dumps(meta or {}), re.I):
        return True
    return len(text) < 1000 and "exa" in text.lower() and re.search(r"rate.?limit", text, re.I) is not None


def _sse_message(response):
    """The answer is server-sent events (`data: {...}` lines) or, on some errors, a plain JSON body."""
    for line in response.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    return response.json()


def _exa_call(name, arguments):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if _exa_key():
        headers["Authorization"] = f"Bearer {_exa_key()}"   # header, not URL: keeps the key out of exception text
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}

    def once():
        message = _sse_message(_request("POST", EXA_URL, json=body, headers=headers))
        if "error" in message:
            error_text = str(message["error"].get("message", message["error"]))
            if re.search(r"rate.?limit", error_text, re.I):
                raise RetryableError(f"Exa rate limited: {error_text[:200]}")
            raise RuntimeError(f"Exa error: {error_text[:300]}")
        result = message.get("result") or {}
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text").strip()
        if _looks_rate_limited(result.get("_meta"), text):
            raise RetryableError("Exa rate limited (HTTP 200 with a rate-limit notice)")
        if result.get("isError"):
            raise RuntimeError(f"Exa tool error: {text[:300]}")
        return text

    return with_retry(once, attempts=6, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa): blogs, surveys, project pages, docs, news. Describe the ideal page in natural language
    (e.g. "survey paper comparing world model architectures"). `objective` says what facts to pull out.
    Returns clean text of the top results, each with Title, URL and Published date."""
    try:
        if not (query or "").strip():
            return "NO RESULTS"
        arguments = {"query": query.strip(), "objective": (objective or "").strip() or f"Find pages about: {query.strip()}",
                     "numResults": _clamp(num_results, 1, 10)}
        text = _exa_call("web_search_exa", arguments)
        return text[:PAGE_CHARS] if text else "NO RESULTS"
    except Exception as exc:
        return _redact(_error(exc))


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page or a blog post) as markdown, given its URL.
    Long pages are truncated to ~12000 characters."""
    try:
        if not re.match(r"https?://", (url or "").strip()):
            return "ERROR: url must start with http:// or https://"
        text = _exa_call("web_fetch_exa", {"urls": [url.strip()]})
        return text[:PAGE_CHARS] if text else "NO RESULTS"
    except Exception as exc:
        return _redact(_error(exc))


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
