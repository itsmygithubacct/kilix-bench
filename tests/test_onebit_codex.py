"""The codex backend against a fake ``codex`` executable on PATH.

The fake reads the prompt on stdin and answers 0 when it holds "2 + 2 = 4"
or "disk full", else 1, with the event stream and usage numbers of a real
``codex exec --json`` call (taken from a one-item smoke run). FAKE_CODEX_MODE
makes it fail like a spent usage limit, a signed-out CLI or a rejected
schema. No network.
"""

import contextlib
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kilix_bench.onebit import contract, runner, scoring
from kilix_bench.onebit.backends import (
    REGISTRY, CodexBackend, parse_events, parse_schema_reply)
from kilix_bench.onebit.backends.codex import SCHEMA, classify

# A real event stream (one item, Luna 6 low, 2026-11-18).
REAL_EVENTS = "\n".join([
    '{"type":"thread.started","thread_id":"01a1ede6-6f18-7ec2-acf2-37d36b3eecd3"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
    '"text":"0"}}',
    '{"type":"turn.completed","usage":{"input_tokens":15513,'
    '"cached_input_tokens":11392,"cache_write_input_tokens":0,'
    '"output_tokens":5,"reasoning_output_tokens":0}}',
])

FAKE = r'''
import json, os, sys
argv = sys.argv[1:]
if argv == ["--version"]:
    print("codex-cli 9.9.9-fake")
    sys.exit(0)
mode = os.environ.get("FAKE_CODEX_MODE", "")
log = os.environ["FAKE_CODEX_LOG"]
prompt = sys.stdin.read()
with open(log, "a") as handle:
    handle.write(json.dumps({"argv": argv, "cwd_entries": os.listdir("."),
                             "codex_home": os.environ.get("CODEX_HOME"),
                             "prompt": prompt}) + "\n")
with open(log) as handle:
    calls = sum(1 for _ in handle)
out = argv[argv.index("-o") + 1]
schema = argv[argv.index("--output-schema") + 1] if "--output-schema" in argv else None

def fail(message):
    print(json.dumps({"type": "thread.started", "thread_id": "t"}))
    print(json.dumps({"type": "error", "message": message}))
    print(json.dumps({"type": "turn.failed", "error": {"message": message}}))
    sys.exit(1)

if mode.startswith("limit_after:") and calls > int(mode.split(":")[1]):
    fail("You've hit your usage limit. Upgrade or try again later.")
if mode == "auth":
    fail("401 Unauthorized: your session expired, please log in again")
if mode == "reject_schema" and schema:
    fail("Invalid schema for response_format 'codex_output_schema'")
answer = "0" if ("2 + 2 = 4" in prompt or "disk full" in prompt) else "1"
text = json.dumps({"answer": answer}, separators=(",", ":")) if schema else answer
if mode == "chatty":
    text = "The answer is " + answer
with open(out, "w") as handle:
    handle.write(text)
print(json.dumps({"type": "thread.started", "thread_id": "t"}))
print(json.dumps({"type": "turn.started"}))
print(json.dumps({"type": "item.completed", "item": {
    "id": "item_0", "type": "agent_message", "text": text}}))
print(json.dumps({"type": "turn.completed", "usage": {
    "input_tokens": 15374, "cached_input_tokens": 11392,
    "cache_write_input_tokens": 0,
    "output_tokens": 15 if schema else 5, "reasoning_output_tokens": 0}}))
'''

ITEMS = [
    {"id": "t-a", "band": "grade-01", "domain": "maths", "form": "question",
     "prompt": "Is 2 + 2 = 4?", "answer": 0, "polarity": "plain",
     "rationale": "r", "split": "dev", "source": "original"},
    {"id": "t-b", "band": "grade-01", "domain": "maths", "form": "question",
     "prompt": "Is 2 + 2 = 5?", "answer": 1, "polarity": "plain",
     "rationale": "r", "split": "dev", "source": "original"},
    {"id": "t-c", "band": "computer-use", "domain": "logs", "form": "verdict",
     "context": "ERROR disk full", "prompt": "Verdict on this job.",
     "answer": 1, "polarity": "plain", "rationale": "r", "split": "dev",
     "source": "original"},
]
PROFILE = 'model = "fake-luna"\nmodel_reasoning_effort = "low"\n'


class FakeCodex:
    """A temporary codex home, a fake codex on PATH, and its call log."""

    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.bin = root / "bin"
        self.bin.mkdir()
        script = self.bin / "codex"
        script.write_text(f"#!{sys.executable}\n{FAKE}", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        self.home = root / "codex-home"
        self.home.mkdir()
        (self.home / "onebit-test.config.toml").write_text(
            PROFILE, encoding="utf-8")
        (self.home / "AGENTS.md").write_text("# prefs\n", encoding="utf-8")
        self.log = root / "calls.jsonl"
        self.cache = root / "cache"
        self.env = dict(os.environ,
                        PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                        FAKE_CODEX_LOG=str(self.log))

    def backend(self, mode="", jobs=1):
        return CodexBackend("onebit-test", str(self.home), jobs=jobs,
                            timeout=30, env=dict(self.env,
                                                 FAKE_CODEX_MODE=mode))

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in
                self.log.read_text(encoding="utf-8").splitlines()]

    def close(self):
        self.temporary.cleanup()


class EventParseTest(unittest.TestCase):
    def test_real_event_stream(self):
        events = parse_events(REAL_EVENTS)
        self.assertEqual(events["messages"], ["0"])
        self.assertEqual(events["errors"], [])
        self.assertEqual(events["usage"], {
            "input_tokens": 15513, "cached_input_tokens": 11392,
            "output_tokens": 5, "reasoning_output_tokens": 0,
            "cache_write_input_tokens": 0})

    def test_failed_turn(self):
        events = parse_events(
            '{"type":"error","message":"stream disconnected"}\n'
            'not json\n'
            '{"type":"turn.failed","error":{"message":"stream disconnected"}}')
        self.assertEqual(events["errors"], ["stream disconnected"] * 2)
        self.assertIsNone(events["usage"])

    def test_schema_reply(self):
        self.assertEqual(parse_schema_reply('{"answer":"0"}'), 0)
        self.assertEqual(parse_schema_reply(' {"answer": "1"}\n'), 1)
        for text in ('{"answer":0}', '{"answer":"0","why":"x"}', "0", "",
                     '{"answer":"yes"}', '["0"]', None):
            with self.subTest(text=text):
                self.assertIsNone(parse_schema_reply(text))

    def test_causes(self):
        self.assertEqual(classify("You've hit your usage limit."), "limit")
        self.assertEqual(classify("exceeded retry limit, last status: 429 "
                                  "Too Many Requests"), "limit")
        self.assertEqual(classify("401 Unauthorized"), "auth")
        self.assertEqual(classify("stream disconnected before completion"),
                         "network")
        self.assertEqual(classify("model refused"), "error")

    def test_registered(self):
        self.assertEqual(sorted(REGISTRY),
                         ["chat", "codex", "command", "needle"])


class CodexBackendTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeCodex()

    def tearDown(self):
        self.fake.close()

    def test_raw_call(self):
        backend = self.fake.backend()
        chat = contract.messages(ITEMS[0])
        answer = backend.ask(chat, "raw")
        self.assertEqual(answer["reply"], "0")
        self.assertEqual(backend.parse(answer, "raw"), 0)
        self.assertEqual(answer["usage"]["input_tokens"], 15374)
        self.assertEqual(answer["usage"]["cached_input_tokens"], 11392)
        self.assertEqual(answer["usage"]["output_tokens"], 5)
        self.assertEqual(answer["usage"]["reasoning_output_tokens"], 0)
        (call,) = self.fake.calls()
        argv = call["argv"]
        self.assertEqual(argv[:3], ["exec", "-p", "onebit-test"])
        for flag in ("--ephemeral", "--skip-git-repo-check", "--ignore-rules",
                     "--json"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("-s") + 1], "read-only")
        self.assertEqual(argv[argv.index("-c") + 1], "mcp_servers={}")
        self.assertEqual(argv[-1], "-")
        self.assertNotIn("--output-schema", argv)
        self.assertEqual(call["cwd_entries"], [])
        self.assertEqual(call["codex_home"], str(self.fake.home))
        # The same text every backend gets: system prompt, blank line, item.
        self.assertEqual(call["prompt"],
                         contract.system_prompt() + "\n\nIs 2 + 2 = 4?")

    def test_constrained_call_sends_the_schema(self):
        backend = self.fake.backend()
        answer = backend.ask(contract.messages(ITEMS[1]), "constrained")
        self.assertEqual(answer["reply"], '{"answer":"1"}')
        self.assertEqual(backend.parse(answer, "constrained"), 1)
        self.assertEqual(backend.parse(answer, "raw"), None)
        self.assertEqual(answer["usage"]["output_tokens"], 15)
        self.assertIn("--output-schema", self.fake.calls()[0]["argv"])
        self.assertEqual(SCHEMA["required"], ["answer"])
        self.assertFalse(SCHEMA["additionalProperties"])

    def test_chatty_reply_is_a_format_failure(self):
        backend = self.fake.backend("chatty")
        answer = backend.ask(contract.messages(ITEMS[0]), "raw")
        self.assertIsNone(backend.parse(answer, "raw"))

    def test_describe(self):
        info = self.fake.backend().describe()
        self.assertEqual(info["model"], "fake-luna")
        self.assertEqual(info["reasoning_effort"], "low")
        self.assertEqual(info["codex_version"], "codex-cli 9.9.9-fake")
        self.assertEqual(info["profile_sha256"],
                         hashlib.sha256(PROFILE.encode()).hexdigest())
        self.assertEqual(info["codex_home_agents_md_sha256"],
                         hashlib.sha256(b"# prefs\n").hexdigest())

    def test_signed_out_is_an_auth_error(self):
        with self.assertRaises(runner.BackendError) as caught:
            self.fake.backend("auth").ask(contract.messages(ITEMS[0]), "raw")
        self.assertEqual(caught.exception.kind, "auth")


class CodexRunTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeCodex()

    def tearDown(self):
        self.fake.close()

    def run_fake(self, mode="", jobs=1):
        backend = self.fake.backend(mode, jobs)
        with mock.patch.object(runner.itemset, "load",
                               return_value=[dict(row) for row in ITEMS]):
            return runner.run({"split": "dev", "sizer_command": "off",
                               "cache_dir": str(self.fake.cache)},
                              backend=backend)

    def test_both_arms_and_cost(self):
        payload = self.run_fake(jobs=2)
        raw = payload["arms"]["raw"]
        self.assertEqual([row["parsed"] for row in raw["records"]], [0, 1, 0])
        self.assertEqual(raw["jobs"], 2)
        cost = raw["summary"]["cost"]
        # 3 items x (15374 input + 5 output); 2 of 3 correct.
        self.assertEqual(cost["total_tokens"], 46137)
        self.assertEqual(cost["tokens_per_item"], 15379)
        self.assertEqual(cost["tokens_per_correct"], 23068.5)
        self.assertAlmostEqual(cost["cached_share"], 11392 / 15374)
        self.assertEqual(cost["output_tokens_per_item"], 5)
        constrained = payload["arms"]["constrained"]["summary"]["cost"]
        self.assertEqual(constrained["tokens_per_item"], 15389)
        self.assertEqual(payload["run_totals"]["usage"]["input_tokens"],
                         6 * 15374)
        self.assertEqual(payload["run_totals"]["usage"]["output_tokens"],
                         3 * 5 + 3 * 15)
        self.assertEqual(payload["model"]["model"], "fake-luna")
        text = runner.render(payload)
        self.assertIn("tok/item", text)
        self.assertIn("15,379", text)
        self.assertIn("run totals:", text)

    def test_limit_stops_cleanly_and_resume_pays_once(self):
        with self.assertRaises(runner.RunStopped) as caught:
            self.run_fake("limit_after:2")
        stopped = caught.exception
        self.assertEqual((stopped.arm, stopped.answered, stopped.total,
                          stopped.failed), ("raw", 2, 3, 1))
        self.assertEqual(stopped.error.kind, "limit")
        self.assertIn("run the same command again", stopped.message())
        self.assertEqual(len(self.fake.calls()), 3)
        # Resumed: the two cached raw answers are not asked again.
        payload = self.run_fake()
        raw = payload["arms"]["raw"]
        self.assertEqual(raw["from_cache"], 2)
        self.assertEqual([row["parsed"] for row in raw["records"]], [0, 1, 0])
        self.assertEqual(len(self.fake.calls()), 3 + 1 + 3)
        # A finished run asks nothing again.
        self.run_fake()
        self.assertEqual(len(self.fake.calls()), 7)

    def test_limit_error_is_never_a_format_failure(self):
        backend = self.fake.backend("limit_after:0")
        with mock.patch.object(runner.itemset, "load",
                               return_value=[dict(row) for row in ITEMS]):
            with self.assertRaises(runner.RunStopped):
                runner.run({"split": "dev", "sizer_command": "off"},
                           backend=backend)

    def test_rejected_schema_marks_the_arm_unsupported(self):
        payload = self.run_fake("reject_schema")
        self.assertTrue(payload["arms"]["raw"]["supported"])
        constrained = payload["arms"]["constrained"]
        self.assertFalse(constrained["supported"])
        self.assertIn("schema", constrained["reason"])


class CodexCommandLineTest(unittest.TestCase):
    """The whole command: a stopped run writes no result file."""

    def setUp(self):
        self.fake = FakeCodex()
        self.bench = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.fake.close()
        self.bench.cleanup()

    def command(self, mode):
        from kilix_bench.onebit import cli
        env = dict(self.fake.env, FAKE_CODEX_MODE=mode,
                   KILIX_BENCH_HOME=self.bench.name)
        argv = ["--backend", "codex", "--codex-profile", "onebit-test",
                "--codex-home", str(self.fake.home), "--limit", "3",
                "--jobs", "1", "--sizer-command", "off", "--label", "fake"]
        with mock.patch.dict(os.environ, env, clear=True), \
                contextlib.redirect_stdout(io.StringIO()):
            return cli.run_command(argv)

    def results(self):
        directory = Path(self.bench.name) / "results"
        return sorted(path.name for path in directory.glob("*.json")) \
            if directory.exists() else []

    def test_stopped_run_writes_nothing_then_resumes(self):
        with self.assertRaises(SystemExit) as caught:
            self.command("limit_after:1")
        self.assertEqual(caught.exception.code, runner.STOPPED_EXIT)
        self.assertEqual(self.results(), [])
        self.assertEqual(self.command(""), 0)
        names = self.results()
        self.assertIn("onebit-fake-dev-latest.json", names)
        saved = json.loads((Path(self.bench.name) / "results" /
                            "onebit-fake-dev-latest.json").read_text())
        self.assertEqual(saved["arms"]["raw"]["from_cache"], 1)
        self.assertEqual(saved["backend"], "codex")
        self.assertNotIn(str(self.fake.home), json.dumps(saved))


class CostTest(unittest.TestCase):
    def test_hand_numbers(self):
        rows = [
            {"answer": 0, "parsed": 0, "latency_ms": 1000.0,
             "usage": {"input_tokens": 100, "cached_input_tokens": 50,
                       "output_tokens": 2, "reasoning_output_tokens": 1}},
            {"answer": 1, "parsed": 0, "latency_ms": 3000.0,
             "usage": {"input_tokens": 300, "cached_input_tokens": 150,
                       "output_tokens": 6, "reasoning_output_tokens": 3}},
            {"answer": 1, "parsed": None, "skipped": "too long"},
        ]
        cost = scoring.cost(rows)
        self.assertEqual(cost["items"], 2)
        self.assertEqual(cost["total_tokens"], 408)
        self.assertEqual(cost["tokens_per_item"], 204)
        self.assertEqual(cost["tokens_per_correct"], 408)
        self.assertEqual(cost["cached_share"], 0.5)
        self.assertEqual(cost["reasoning_tokens_per_item"], 2)
        self.assertEqual(cost["seconds_per_item"], 2.0)
        self.assertIsNone(scoring.cost([{"answer": 0, "parsed": 0}]))
        lines = scoring.cost_table(cost)
        self.assertIn("tok/correct", lines[0])
        self.assertIn("408", lines[1])
        self.assertIn("50%", lines[1])


if __name__ == "__main__":
    unittest.main()
