"""Offline checks (no network, no LLM):   python test_lab.py   (or pytest test_lab.py)"""
import json
import tempfile
from pathlib import Path
from unittest import mock

import research
import tools
from check_citations import check
from finalize_citations import finalize

SOURCES = [{"n": 1, "url": "https://arxiv.org/abs/1"}, {"n": 2, "url": "https://huggingface.co/papers/2"}]
REPORT = ("# T\nA [1, 2]. `[7]` [3](https://x)\n\n## References\n"
          "[1] A. arxiv. https://arxiv.org/abs/1 (2025-01-01)\n[2] B. hf-search. https://huggingface.co/papers/2 (n.d.)\n")


def test_with_retry_backoff_cap_and_last_attempt():
    calls = []

    def flaky():
        calls.append(1)
        raise tools.RetryableError("busy")

    with mock.patch.object(tools.time, "sleep") as sleep:
        try:
            tools.with_retry(flaky, attempts=4, base=1.0, cap=3.0)
        except tools.RetryableError:
            pass
        else:
            raise AssertionError("must re-raise after the last attempt")
    assert len(calls) == 4 and sleep.call_count == 3            # no sleep after the last attempt
    assert all(0 < c.args[0] <= 3.0 for c in sleep.call_args_list)


def test_with_retry_honours_retry_after_and_ignores_other_errors():
    attempts = iter([tools.RetryableError("429", retry_after=7), "ok"])

    def fn():
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    with mock.patch.object(tools.time, "sleep") as sleep:
        assert tools.with_retry(fn, cap=60) == "ok"
        sleep.assert_called_once_with(7)
        try:
            tools.with_retry(lambda: 1 / 0)
        except ZeroDivisionError:
            pass
        assert sleep.call_count == 1                              # a bug is not retried


def test_tools_never_raise_and_redact_key():
    assert tools.arxiv_search.invoke({"query": "\"::\" AND OR"}) == "NO RESULTS"
    assert tools._arxiv_terms('all:"world model" AND ti:robot') == ["world", "model", "robot"]
    with mock.patch.dict("os.environ", {"EXA_API_KEY": "secret123"}), \
            mock.patch.object(tools, "_exa_call", side_effect=RuntimeError("bad url ?exaApiKey=secret123")):
        out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("ERROR:") and "secret123" not in out


def test_slugify():
    assert research.slugify("Survey about World Model!") == "survey-about-world-model"
    assert research.slugify("../../x") == "x"
    assert research.slugify("  ") == "topic"
    assert len(research.slugify("a" * 200)) == 60


def test_check_citations():
    assert check(REPORT, SOURCES) == []
    assert check(REPORT, []) == ["no sources in sources.json"]
    assert any("exactly one URL" in p for p in check(REPORT.replace("(n.d.)", "https://y.org"), SOURCES))
    assert any("References" in p for p in check("body [1][2]", SOURCES))
    assert any("[3] cited" in p for p in check(REPORT.replace("A [1, 2]", "A [1, 2][3]"), SOURCES))
    report, sources, problems = finalize("# T\nx [2] y [1]\n", SOURCES)
    assert not problems and check(report, sources) == []


def test_save_outputs_writes_nothing_on_failure():
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(research, "download", return_value={research.REPORT_PATH: b"", research.SOURCES_PATH: b"[]"}):
        try:
            research.save_outputs(None, "t", [], 1.0, "m", reports_dir=tmp)
        except RuntimeError:
            pass
        else:
            raise AssertionError("an empty report must fail")
        assert not list(Path(tmp).iterdir())
    files = {research.REPORT_PATH: REPORT.encode(),
             research.SOURCES_PATH: json.dumps([dict(SOURCES[0], source="arxiv")]).encode()}
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(research, "download", return_value=files):
        path = research.save_outputs(None, "My Topic", [], 1.23, "m", reports_dir=tmp)
        meta = json.loads((Path(tmp) / "my-topic.meta.json").read_text())
        assert path.name == "my-topic.md" and meta["source_families"] == ["arxiv"] and meta["elapsed_s"] == 1.2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
