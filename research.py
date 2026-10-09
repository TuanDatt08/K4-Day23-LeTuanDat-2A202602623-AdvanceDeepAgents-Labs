"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from langgraph.errors import GraphRecursionError

from agents import (FINALIZER_PATH, RECURSION_LIMIT, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR,
                    build_lead_agent)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[\W_]+", "-", (topic or "").lower(), flags=re.ASCII).strip("-")
    return slug[:60].strip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Topic: {topic}\n\n"
            f"Produce the cited survey report on this topic by following every step of your instructions: plan, "
            f"delegate the sub-questions to researchers in parallel, verify their notes, merge {SOURCES_PATH}, write "
            f"{REPORT_PATH}, run the finalizer and the validator until it prints OK, then spot-check citations.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    calls, tokens = Counter(), Counter()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(sorted(calls.items())), "tokens": {"input": tokens["input"], "output": tokens["output"]}}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    Raises RuntimeError and writes NOTHING when the run did not produce a usable report."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report, raw_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report or not report.decode("utf-8", errors="replace").strip():
        raise RuntimeError(f"the agent produced no report ({REPORT_PATH} is missing or empty)")
    if raw_sources is None:
        raise RuntimeError(f"the agent produced no {SOURCES_PATH}")
    try:
        sources = json.loads(raw_sources.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON list")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")})}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = reports_dir / slugify(topic)
    # bytes exactly as downloaded from the sandbox (RUBRIC: submitted files = the sandbox output)
    Path(f"{stem}.sources.json").write_bytes(raw_sources)
    Path(f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = Path(f"{stem}.md")
    report_path.write_bytes(report)
    return report_path


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "unknown")


def run_agent(agent, topic, start):
    """agent.invoke, but streamed so every tool call (lead and subagents) is printed as it happens.
    Returns the lead's final state."""
    seen, final = {}, None
    stream = agent.stream({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                          config={"recursion_limit": RECURSION_LIMIT}, stream_mode="values", subgraphs=True)
    for namespace, state in stream:
        messages = state.get("messages", [])
        for message in messages[seen.get(namespace, 0):]:
            for call in getattr(message, "tool_calls", None) or []:
                args = call.get("args", {})
                detail = (f"{args.get('subagent_type')}: {args.get('description', '')}" if call["name"] == "task"
                          else json.dumps(args, ensure_ascii=False))
                who = "lead" if not namespace else "  sub"
                print(f"[{time.monotonic() - start:6.0f}s] {who} {call['name']} {detail[:110]}", flush=True)
        seen[namespace] = len(messages)
        if not namespace:
            final = state
    if final is None:
        raise RuntimeError("the agent produced no output")
    return final


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = make_model()
    start = time.monotonic()
    with open_sandbox() as backend:   # the sandbox is always stopped and removed, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        try:
            result = run_agent(agent, topic, start)
            report_path = save_outputs(backend, topic, result["messages"], time.monotonic() - start, _model_name(model))
        except (RuntimeError, GraphRecursionError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
