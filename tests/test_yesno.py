#!/usr/bin/env python3
"""Exercise native provider commands, durable runs and exact reply scoring offline."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
WORKER = PROJECT / "bin/kilix-yesno-worker"
sys.path.insert(0, str(PROJECT/"src"))
from kilix_bench.yesno import machine, worker
from kilix_bench.yesno.models import gguf_artifact, gguf_snapshot

QUESTIONS = [
    dict(id="first", band="maths", prompt="Is two even?", answer=0, rationale="Two is even."),
    dict(id="second", band="reading", prompt="Is the light on?", context="The light is off.",
         answer=1, rationale="The light is off."),
    dict(id="third", band="logic", prompt="Is none zero?", answer=0, rationale="None means zero."),
    dict(id="fourth", band="logic", prompt="Are there two?", answer=1, rationale="There are three."),
]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class BenchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.log = self.root / "calls"
        for name in ("codex", "grok", "claude"):
            binary = self.root / name
            binary.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys, time
args = sys.argv[1:]
provider = pathlib.Path(sys.argv[0]).name
if args[:1] == ['app-server']:
    for line in sys.stdin:
        call = json.loads(line)
        if call['method'] == 'model/list':
            print(json.dumps({'id': 3, 'result': {'data': [
                {'model': 'example-base', 'isDefault': True}, {'model': 'example-pro'},
                {'model': 'hidden-id', 'hidden': True}]}}), flush=True)
            print(json.dumps({'id': 2, 'result': {'config': {'model': 'example-pro'}}}), flush=True)
            break
    sys.exit()
if args == ['models']:
    print('Default model: example-fast')
    print('* example-fast (default)')
    print('- example-large')
    sys.exit()
if '--input-format' in args:
    call = json.loads(sys.stdin.readline())
    assert call['request'] == {'subtype': 'initialize'}
    print(json.dumps({'type': 'control_response', 'response': {'request_id': call['request_id'],
          'response': {'models': [{'value': 'default'}, {'value': 'opus'}, {'value': 'sonnet'}]}}}))
    sys.exit()
prompt = args[args.index('--single')+1] if provider == 'grok' else sys.stdin.read()
with open(os.environ['BENCH_CALLS'], 'a') as out:
    print(json.dumps([provider, args, prompt]), file=out)
time.sleep(float(os.environ.get('BENCH_DELAY', '0')))
if 'two even' in prompt: reply = 'YES'
elif 'light' in prompt: reply = 'YES'
elif 'none' in prompt: reply = 'NO, because...'
else: reply = 'NO'
reply = os.environ.get('BENCH_REPLY', reply)
if provider == 'codex':
    pathlib.Path(args[args.index('-o')+1]).write_text(reply)
    print('unrelated progress on stdout')
elif provider == 'claude' and os.environ.get('BENCH_RESOLVED_MODEL'):
    print(json.dumps({'result': reply, 'modelUsage': {os.environ['BENCH_RESOLVED_MODEL']: {}}}))
else:
    print(reply)
""")
            binary.chmod(0o700)
        package = self.root / "fixture/src/kilix_bench/onebit"
        (package / "backends").mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "backends/__init__.py").touch()
        (package / "items.py").write_text('def load(split):\n assert split=="dev"\n return ' + repr(QUESTIONS))
        (package / "contract.py").write_text("def user_message(item):\n return item.get('context','')+'\\n'+item['prompt']\n")
        (package / "backends/chat.py").write_text("class ChatBackend: pass\n")
        self.env = {**os.environ, "PUZZLLM_YESNO_BENCH": str(self.root/"fixture"),
                    "BENCH_CALLS": str(self.log), "PYTHONDONTWRITEBYTECODE": "1",
                    "HOME": str(self.root/"home"), "CODEX_HOME": str(self.root/"home/.codex"),
                    "GROK_HOME": str(self.root/"home/.grok"), "CLAUDE_CONFIG_DIR": str(self.root/"home/.claude"),
                    "ANTHROPIC_MODEL": ""}
        for name in ("codex", "grok", "claude"):
            self.env["PUZZLLM_" + name.upper()] = str(self.root/name)

    def process(self, *args, delay=0):
        proc = subprocess.Popen([WORKER, *args], env={**self.env, "BENCH_DELAY": str(delay)},
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def close():
            if proc.poll() is None:
                proc.terminate()
            proc.communicate(timeout=8)
        self.addCleanup(close)
        return proc

    def wait_for(self, condition):
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.025)
        self.fail("worker did not reach expected state")

    def reports(self):
        return list((self.state/"bench/yesno").glob("*.jsonl"))

    def result(self, *args):
        result = subprocess.run([WORKER, *args], env=self.env, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=10, check=True)
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_hosted_model_catalogs_use_native_ids_and_caches(self):
        def names(provider):
            return [r["model"] for r in self.result("models", provider)]
        self.assertEqual(names("codex"), ["example-pro", "example-base"])
        self.assertEqual(names("grok"), ["example-fast", "example-large"])
        settings = Path(self.env["CLAUDE_CONFIG_DIR"])
        settings.mkdir(parents=True)
        (settings/"settings.json").write_text(json.dumps({"model": "opus", "modelSettings": {"exact-version": {}}}))
        (settings/".claude.json").write_text(json.dumps({"additionalModelOptionsCache": [{"value": "long-context"}]}))
        self.assertEqual(names("claude"), ["opus", "sonnet", "long-context", "exact-version"])
        for provider, content, expected in (
                ("codex", [{"slug": "cache-pro", "visibility": "list"}, {"slug": "hidden", "visibility": "hide"}],
                 ["cache-pro"]),
                ("grok", {"cache-fast": {"info": {}}, "hidden": {"info": {"hidden": True}}}, ["cache-fast"])):
            home = Path(self.env[provider.upper()+"_HOME"])
            home.mkdir(parents=True)
            (home/"models_cache.json").write_text(json.dumps({"models": content, "unused": "private fixture"}))
            self.assertEqual(names(provider), expected)
        self.assertFalse(self.log.exists())

    def test_selected_and_configured_models_share_their_concrete_identity(self):
        for provider, concrete, alias in (("codex", "example-pro", "example-pro"),
                                          ("grok", "example-fast", "example-fast"),
                                          ("claude", "example-concrete", "opus")):
            self.env["BENCH_RESOLVED_MODEL"] = concrete
            for choice in ("", alias):
                self.result("run", self.state, provider, choice)
            board = self.result("records", self.state)
            model_rows = [r for r in board if r["event"] == "standing" and r["provider"] == provider]
            self.assertEqual(len(model_rows), 1)
            self.assertEqual((model_rows[0]["model"], model_rows[0]["runs"], model_rows[0]["avg_correct"]),
                             (concrete, 2, 2.0))
            calls = [args for name, args, _ in rows(self.log) if name == provider]
            if provider == "claude":
                self.assertNotIn("--model", calls[0])
                calls = calls[1:4]
            self.assertTrue(all(args[args.index("--model")+1] == concrete for args in calls))

    def test_rankings_rebuild_each_model_once_and_retain_every_attempt(self):
        folder = self.state / "bench/yesno"
        folder.mkdir(parents=True)
        def journal(n, provider, model, total, grades, end="end"):
            path = folder / f"20260101T0000{n:02}.000000Z-00000000.jsonl"
            events = [dict(event="begin", provider=provider, model=model, file=path.name, total=total)]
            events += [dict(event="answer", n=i, verdict=v) for i, v in enumerate(grades, 1)]
            if end:
                events.append(dict(event=end))
            path.write_text("".join(json.dumps(e)+"\n" for e in events))
            return path
        c, w = "correct", "wrong"
        first = journal(1, "ollama", "library/small", 4, [c,c,c,w])
        journal(2, "ollama", "small:latest", 2, [w,w])
        journal(3, "ollama", "small", 4, ["invalid"], "stopped")
        journal(4, "ollama", "small", 4, [], "error")
        journal(5, "ollama", "small", 4, [c,c], None)
        live = journal(6, "ollama", "small", 4, [c], None)
        journal(7, "codex", "tie", 8, [c,c,c,w,w,w,w,w])
        journal(8, "codex", "top", 1, [c])
        journal(9, "codex", "", 1, [c])
        journal(10, "ollama", "small", 4, [c])  # incomplete terminal record
        bad = journal(11, "grok", "damaged", 1, [], None)
        with bad.open("a") as out:
            out.write("invalid line\n")
        broken = journal(12, "grok", ["invalid id"], 0, [])
        self.assertTrue(broken.exists())
        with live.open("a") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX)
            board = self.result("records", self.state)
            self.assertEqual(board, self.result("records", self.state))
        ranks = [r for r in board if r["event"] == "standing"]
        self.assertEqual([(r["rank"], r["model"]) for r in ranks if r["rank"]],
                         [(1, "top"), (2, "tie"), (2, "small:latest")])
        small = next(r for r in ranks if r["model"] == "small:latest")
        self.assertEqual((small["runs"], small["finished"], small["avg_correct"], small["avg_percent"]),
                         (7, 2, 1.5, 37.5))
        self.assertEqual(sum(r["runs"] for r in ranks), 12)
        self.assertEqual((small["invalid_answers"], small["invalid_counted_runs"]), (1, 7))
        self.assertEqual({r["status"] for r in board if r["event"] == "saved_run" and r["model"] == small["model"]},
                         {"Complete", "Failed", "Interrupted", "Running", "Stopped"})
        self.assertTrue(any(not r["model"] and r["rank"] == 0 and r["provider"] == "codex" for r in ranks))
        self.assertEqual(self.result("report", self.state, first.name), rows(first))
        path = self.root/"models"/"one.gguf"
        path.parent.mkdir()
        path.touch()
        link = self.root/"alias.gguf"
        link.symlink_to(path)
        for i, name in enumerate((link, path, self.root/"other"/path.name), 13):
            journal(i, "llama.cpp", str(name), 1, [c])
        gguf = [r for r in self.result("records", self.state) if r["event"] == "standing" and r["provider"] == "llama.cpp"]
        self.assertEqual({r["model"]: r["runs"] for r in gguf}, {str(path): 2, str(self.root/"other"/path.name): 1})

    def test_all_hosted_adapters_use_one_fresh_prompt_and_strict_grades(self):
        for provider in ("codex", "grok", "claude"):
            with self.subTest(provider=provider):
                output = subprocess.run([WORKER, "run", self.state, provider, "chosen-model"],
                                        env=self.env, stdin=subprocess.DEVNULL, capture_output=True,
                                        text=True, check=True, timeout=12).stdout
                events = [json.loads(line) for line in output.splitlines()]
                self.assertEqual(events[-1]["event"], "end")
                self.assertEqual([e["verdict"] for e in events if e["event"] == "answer"],
                                 ["correct", "wrong", "invalid", "correct"])
                calls = [row for row in rows(self.log) if row[0] == provider]
                self.assertEqual(len(calls), len(QUESTIONS))
                for (_, args, prompt), item in zip(calls, QUESTIONS):
                    self.assertEqual(args[args.index("--model") + 1], "chosen-model")
                    self.assertIn("Respond with only YES or NO.", "\n".join([*args, prompt]))
                    self.assertIn(item["prompt"], prompt)
                    self.assertNotIn("rationale", prompt)
                    if provider == "grok":
                        self.assertEqual(args[args.index("--tools") + 1], "read_file")
                        denied = args[args.index("--disallowed-tools") + 1].split(",")
                        self.assertTrue({"read_file", "search_tool", "use_tool"}.issubset(denied))
                        self.assertNotIn("--max-turns", args)
                    if "context" in item:
                        self.assertIn(item["context"], prompt)
        self.assertEqual(len(self.reports()), 3)
        self.assertTrue(all(not (p.stat().st_mode & 0o077) for p in self.reports()))
        saved = [r for r in rows(self.reports()[0]) if r["event"] == "answer"]
        self.assertEqual(saved[2]["reply"], "NO, because...")
        self.assertEqual(saved[1]["context"], QUESTIONS[1]["context"])

    def test_hosted_timeout_retries_the_same_question_and_effort(self):
        notices = []
        with patch.dict(os.environ, {**self.env, "BENCH_DELAY": "1"}), \
                patch.object(worker, "HOSTED_TIMEOUTS", (0.6, 3)), \
                worker.answerer("codex", "selected", self.root, None, notices.append, effort="low") as ask:
            self.assertEqual(ask("Is two even?"), "YES")
        first, retry = rows(self.log)
        self.assertEqual(first, retry)
        self.assertIn('model_reasoning_effort="low"', first[1])
        self.assertEqual(len(notices), 1)
        self.assertIn("Retrying this question", notices[0])

    def test_repeated_hosted_timeout_is_a_provider_error(self):
        notices = []
        with patch.dict(os.environ, {**self.env, "BENCH_DELAY": "30"}), \
                patch.object(worker, "HOSTED_TIMEOUTS", (0.6, 0.8)), \
                worker.answerer("codex", "selected", self.root, None, notices.append) as ask:
            with self.assertRaisesRegex(RuntimeError, "no answer after 2 timed-out attempts"):
                ask("Is two even?")
        self.assertEqual(len(rows(self.log)), 2)
        self.assertEqual(len(notices), 1)

    def test_lowercase_reply_is_a_format_failure(self):
        self.env["BENCH_REPLY"] = "yes"
        result, _ = self.process("run", self.state, "claude").communicate(timeout=5)
        events = [json.loads(line) for line in result.splitlines()]
        self.assertEqual([r["verdict"] for r in events if r["event"] == "answer"], ["invalid"] * 4)

    def test_local_kilix_models_and_native_reply_contract(self):
        from kilix_bench.yesno.local import needle_reply
        self.assertEqual([e["model"] for e in self.result("models","kilix")], ["needle2","needle3","lfm2"])
        good = {"function_calls": [{"name": "answer", "arguments": {"value": "NO"}}]}
        self.assertEqual(needle_reply(good).text, "NO")
        for calls in ([], good["function_calls"]*2, [{"name":"open","arguments":{"value":"YES"}}],
                      [{"name":"answer","arguments":{"value":"yes"}}]):
            reply = needle_reply({"function_calls":calls})
            self.assertNotIn(reply.text, ("YES", "NO"))
            self.assertEqual(reply.evidence["needle_reply"]["function_calls"], calls)

    def test_effort_picker_respects_the_model_catalog_and_pins_configured_effort(self):
        home = Path(self.env["GROK_HOME"])
        home.mkdir(parents=True)
        (home/"config.toml").write_text('[models]\ndefault="selected"\ndefault_reasoning_effort="high"\n')
        (home/"models_cache.json").write_text(json.dumps({"models":{"selected":{"info":{
            "reasoning_efforts":[{"value":"low"},{"value":"high"}]}}}}))
        self.assertEqual([e["model"] for e in self.result("efforts","grok","selected")], ["low","high"])
        self.result("run", self.state, "grok", "selected")
        board=self.result("records",self.state)
        self.assertEqual(next(r for r in board if r["event"]=="standing")["effort"],"high")
        self.assertTrue(all(args[args.index("--reasoning-effort")+1]=="high" for _,args,_ in rows(self.log)))
        result = subprocess.run([WORKER, "run", self.state, "grok", "selected", "xhigh"], env=self.env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.reports()), 1)

    def test_worker_saves_versioned_rankings_and_restores_without_journals(self):
        self.result("run", self.state, "codex", "chosen-model")
        path = self.state/"bench/yesno/rankings.json"
        original = path.read_bytes()
        board = json.loads(original)
        self.assertEqual((board["schema_version"], board["benchmark"], board["split"], board["total_runs"]),
                         (3, "OneBit", "dev", 1))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertIn(b'\n  "models": [\n', original)
        group, = board["models"]
        self.assertEqual((group["provider"], group["model"], group["rank"], group["finished"], group["avg_percent"]),
                         ("codex", "chosen-model", 1, 1, 50.0))
        run, = group["history"]
        self.assertEqual((run["wrong"], run["invalid"], group["invalid_answers"]), (1, 1, 1))
        self.assertEqual(run["backend_scope"], "provider CLI")
        self.assertIsInstance(run["runner"]["cpu"], dict)
        self.assertIsInstance(run["runner"]["gpus"], list)
        self.assertIn("peak_ram_used_bytes", run["runner_usage"])
        self.assertEqual(run["timing"]["timed_answers"], 4)
        self.assertGreaterEqual(run["timing"]["wall_seconds"], run["timing"]["setup_seconds"])
        self.assertTrue(run["timing"]["ended_at"].endswith("Z"))
        self.assertEqual(run["file"], self.reports()[0].name)
        self.assertRegex(run["started"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$")
        self.assertTrue(run["journal_available"])
        restored = self.root/"restored"
        recovered = restored/"bench/yesno/rankings.json"
        recovered.parent.mkdir(parents=True)
        recovered.write_bytes(original)
        self.result("run", restored, "codex", "chosen-model")
        group, = json.loads(recovered.read_bytes())["models"]
        self.assertEqual((group["runs"], group["finished"], group["avg_correct"]), (2, 2, 2.0))
        previous = next(r for r in group["history"] if r["file"] == run["file"])
        self.assertFalse(previous["journal_available"])
        self.assertEqual(previous["status"], "Complete")
        self.assertEqual((previous["runner"], previous["timing"]), (run["runner"], run["timing"]))
        self.assertIn("Complete (summary only)", [r.get("status") for r in self.result("records", restored)])
        # A restored legacy copy of the same journal can have fewer measurements.
        legacy = [r for r in rows(self.reports()[0]) if r["event"] not in ("machine", "ready")]
        for event in legacy:
            for key in ("wall_seconds", "ended_at", "runner_usage", "seconds", "backend_scope"):
                event.pop(key, None)
        (recovered.parent/run["file"]).write_text("".join(json.dumps(r)+"\n" for r in legacy))
        self.result("records", restored)
        history = json.loads(recovered.read_bytes())["models"][0]["history"]
        kept = next(r for r in history if r["file"] == run["file"])
        self.assertEqual((kept["runner"], kept["timing"], kept["runner_usage"]),
                         (run["runner"], run["timing"], run["runner_usage"]))

    def test_inventory_and_system_wide_samples_on_mixed_graphics(self):
        proc, sysroot = self.root/"proc", self.root/"sys"
        proc.mkdir()
        (proc/"cpuinfo").write_text("model name: Test CPU\nphysical id: 0\ncore id: 0\n\n"
                                   "model name: Test CPU\nphysical id: 0\ncore id: 1\n")
        (proc/"stat").write_text("cpu 10 0 10 80 0 0 0 0\n")
        (proc/"meminfo").write_text("MemTotal: 1024 kB\nMemAvailable: 768 kB\n")
        slot = "0000:02:00.0"
        device = sysroot/"bus/pci/devices"/slot
        device.mkdir(parents=True)
        for name, value in (("class", "0x030000"), ("product_name", "Fixture GPU"), ("vendor", "0x1002"),
                            ("device", "0xabcd"), ("gpu_busy_percent", "20"), ("mem_info_vram_total", "4096"),
                            ("mem_info_vram_used", "256")):
            (device/name).write_text(value)
        drm = sysroot/"class/drm/card0"
        drm.mkdir(parents=True)
        (drm/"device").symlink_to(device)
        def probe(argv):
            if argv[0] == "nvidia-smi":
                return '00000000:01:00.0, NVIDIA Fixture, 8192, 25, 512\n'
            return ""
        with patch.object(machine, "command", side_effect=probe), patch.object(machine.os, "cpu_count", return_value=4):
            monitor = machine.RunnerMonitor(proc, sysroot)
            try:
                (proc/"stat").write_text("cpu 30 0 20 150 0 0 0 0\n")
                (proc/"meminfo").write_text("MemTotal: 1024 kB\nMemAvailable: 128 kB\n")
                (device/"gpu_busy_percent").write_text("60")
                (device/"mem_info_vram_used").write_text("1024")
            finally:
                usage = monitor.finish()
        profile = monitor.profile
        self.assertEqual((profile["cpu"]["model"], profile["cpu"]["physical_cores"], profile["cpu"]["logical_cpus"]),
                         ("Test CPU", 2, 4))
        self.assertEqual(profile["ram_bytes"], 1024**2)
        self.assertEqual([(g["model"], g["vram_bytes"]) for g in profile["gpus"]],
                         [("NVIDIA Fixture", 8192*1024**2), ("Fixture GPU", 4096)])
        self.assertEqual((usage["cpu_busy_percent"], usage["peak_ram_used_bytes"]), (30, 896*1024))
        self.assertEqual([(g["mean_busy_percent"], g["peak_vram_used_bytes"]) for g in usage["gpus"]],
                         [(25, 512*1024**2), (40, 1024)])
        with patch.object(machine, "command", return_value=""):
            empty = machine.RunnerMonitor(self.root/"missing-proc", self.root/"missing-sys")
            self.assertEqual(empty.profile["gpus"], [])
            self.assertIsNone(empty.finish()["cpu_busy_percent"])
        self.assertEqual(machine.backend_scope("ollama", "127.0.0.1:11434"), "loopback Ollama")
        remote = machine.backend_scope("ollama", "http://remote.example")
        self.assertTrue(remote.startswith("remote Ollama ["))
        self.assertEqual(remote, machine.backend_scope("ollama", "http://person@remote.example:11434/"))
        self.assertNotEqual(remote, machine.backend_scope("ollama", "http://second.example"))

    def test_cross_machine_comparison_uses_content_and_deduplicates_archives(self):
        def report(state, ordinal, runner, selected, latency, artifact):
            folder = state/"bench/yesno"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/f"20260901T02000{ordinal}.000000Z-12341234.jsonl"
            profile = dict(id=runner, label=runner, cpu={"model": "CPU "+runner, "logical_cpus": 8},
                           ram_bytes=8*1024**3, gpus=[dict(slot="card0", model="GPU "+runner, vram_bytes=1024**3)])
            events = [dict(event="begin", provider="llama.cpp", model=selected, total=2, backend_scope="local llama.cpp"),
                      dict(event="machine", runner=profile), dict(event="ready", setup_seconds=1.5),
                      dict(event="answer", n=1, verdict="correct", seconds=latency),
                      dict(event="answer", n=2, verdict="wrong", seconds=2*latency),
                      dict(event="end", wall_seconds=3*latency+2, ended_at="2026-09-01T02:02:00Z"),
                      dict(event="artifact", id=artifact)]
            path.write_text("".join(json.dumps(e)+"\n" for e in events))
            return path
        a, b = self.root/"a", self.root/"b"
        cache = self.root/"digests"
        weight = self.root/"weights.gguf"
        weight.write_bytes(b"example weights")
        snapshot = gguf_snapshot(weight)
        digest = gguf_artifact(snapshot, cache, "machine-a")
        self.assertEqual(digest, "sha256:"+hashlib.sha256(weight.read_bytes()).hexdigest())
        weight.write_bytes(b"different contents")
        self.assertIsNone(gguf_artifact(snapshot, cache, "machine-a"))
        for n in (1, 2):
            report(a, n, "alpha", "/first/location/model.gguf", n, digest)
        other = report(b, 3, "bravo", "/other/place/model.gguf", 4, digest)
        self.result("records", a)
        self.result("records", b)
        files = [d/"bench/yesno/rankings.json" for d in (a,b,a)]
        before = [f.read_bytes() for f in files]
        result = subprocess.run([WORKER,"compare",*files], env=self.env, text=True, capture_output=True, check=True)
        table = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
        by_runner = {r["Machine ID"]: r for r in table}
        self.assertEqual(len(table), 2)
        for name, runs, wall, seconds in (("alpha", "2", 6.5, 2.25), ("bravo", "1", 14, 6)):
            row = by_runner[name]
            self.assertEqual((row["Runs"], row["Finished"], float(row["Mean wall s"]), float(row["Mean s/answer"])),
                             (runs, runs, wall, seconds))
            self.assertIn("GPU "+name, row["GPU"])
        self.assertEqual([f.read_bytes() for f in files], before)
        (a/"bench/yesno"/other.name).write_bytes(other.read_bytes())
        self.result("records", a)
        group, = json.loads(files[0].read_bytes())["models"]
        self.assertEqual((group["runs"], group["avg_percent"], group["artifact_id"]), (3, 50, digest))
        self.assertEqual(len(group["machine_comparisons"]), 2)
        self.assertEqual({r["model_path"] for r in group["history"]}, {"/first/location/model.gguf","/other/place/model.gguf"})

    def test_legacy_timing_migrates_without_assigning_the_readers_hardware(self):
        folder = self.state/"bench/yesno"
        folder.mkdir(parents=True)
        path = folder/"old.jsonl"
        events = [dict(event="begin",provider="codex",model="fixture",total=3),
                  dict(event="answer",n=1,verdict="correct",seconds=0),
                  dict(event="answer",n=1,verdict="correct",seconds=0),
                  dict(event="answer",n=2,verdict="wrong",seconds=2),
                  dict(event="answer",n=3,verdict="correct"), dict(event="end")]
        path.write_text("".join(json.dumps(e)+"\n" for e in events))
        board = folder/"rankings.json"
        board.write_text(json.dumps(dict(schema_version=1,benchmark="OneBit",split="dev",total_runs=0,models=[])))
        self.result("records",self.state)
        document = json.loads(board.read_bytes())
        r, = document["models"][0]["history"]
        self.assertEqual(document["schema_version"], 3)
        self.assertIsNone(r["runner"])
        self.assertIsNone(r["runner_usage"])
        self.assertIsNone(r["timing"]["wall_seconds"])
        self.assertEqual((r["timing"]["timed_answers"], r["timing"]["answer_seconds"], r["timing"]["mean_answer_seconds"],
                          r["timing"]["p95_answer_seconds"], r["correct"]), (2, 2, 1, 2, 2))

    def test_snapshot_repair_and_failed_atomic_replace_preserve_previous_data(self):
        self.result("run", self.state, "codex", "one")
        folder = self.state/"bench/yesno"
        path = folder/"rankings.json"
        original = path.read_bytes()
        same = path.stat().st_ino
        self.result("records", self.state)
        self.assertEqual((path.stat().st_ino, path.read_bytes()), (same, original))
        self.reports()[0].rename(folder/"archived")
        with patch("kilix_bench.yesno.records.os.replace", side_effect=OSError("fixture interrupted write")):
            with self.assertRaises(OSError):
                worker.save_rankings(folder)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(folder.glob(".rankings-*")), [])
        (folder/"archived").rename(folder/json.loads(original)["models"][0]["history"][0]["file"])
        path.write_bytes(b"part of an invalid file")
        self.result("latest", self.state)
        self.assertEqual(path.read_bytes(), original)
        backup, = folder.glob("rankings-damaged-*.json")
        self.assertEqual(backup.read_bytes(), b"part of an invalid file")
        newer = b'{"schema_version":22}\n'
        path.write_bytes(newer)
        events = self.result("records", self.state)
        self.assertEqual(path.read_bytes(), newer)
        self.assertEqual(events[0]["runs"], 1)
        self.assertIn("Unrecognized", events[-1]["text"])

    def test_detaching_rejoining_and_concurrent_start_keep_one_run(self):
        watch = self.process("run", self.state, "grok", delay=0.45)
        self.wait_for(lambda: self.log.exists())
        watch.terminate()
        watch.communicate(timeout=4)
        join = self.process("latest", self.state)
        other = self.process("run", self.state, "codex", "another-model")
        for p in (join, other):
            out, _ = p.communicate(timeout=9)
            events = [json.loads(line) for line in out.splitlines()]
            self.assertEqual(events[-1]["event"], "end")
            self.assertEqual(sum(e["event"] == "answer" for e in events), 4)
        self.assertEqual(len(self.reports()), 1)
        self.assertEqual([provider for provider, _, _ in rows(self.log)], ["grok"] * 4)
        group, = json.loads((self.state/"bench/yesno/rankings.json").read_bytes())["models"]
        self.assertEqual((group["runs"], group["finished"], group["history"][0]["status"]), (1, 1, "Complete"))

    def test_rankings_finish_on_disk_with_every_monitor_closed(self):
        path = self.state/"bench/yesno/rankings.json"
        watch = self.process("run", self.state, "grok", "named", delay=0.2)
        self.wait_for(self.log.exists)
        watch.terminate()
        watch.communicate(timeout=4)
        self.wait_for(lambda: json.loads(path.read_bytes())["models"][0]["finished"] == 1)
        data = json.loads(path.read_bytes())
        self.assertEqual((data["total_runs"], data["models"][0]["model"], data["models"][0]["avg_correct"]),
                         (1, "named", 2.0))

    def test_stop_is_explicit_and_survives_replay(self):
        watch = self.process("run", self.state, "grok", delay=30)
        self.wait_for(self.log.exists)
        result, _ = watch.communicate("stop\n", timeout=7)
        self.assertEqual(json.loads(result.splitlines()[-1])["event"], "stopped")
        group, = json.loads((self.state/"bench/yesno/rankings.json").read_bytes())["models"]
        self.assertEqual((group["runs"], group["rank"], group["history"][0]["status"]), (1, None, "Stopped"))
        result, _ = self.process("latest", self.state).communicate(timeout=4)
        self.assertEqual(json.loads(result.splitlines()[-1])["event"], "stopped")
        self.assertEqual(len(rows(self.log)), 1)

    def test_game_launcher_finds_user_installed_clis_without_a_login_shell(self):
        home = self.root / "home"
        for provider, location in (("grok", ".grok/bin"), ("claude", ".local/bin")):
            folder = home / location
            folder.mkdir(parents=True)
            (folder/provider).symlink_to(self.root/provider)
            with patch.dict(os.environ, HOME=str(home), PATH=os.defpath,
                            BENCH_CALLS=str(self.log), **{"PUZZLLM_"+provider.upper(): ""}):
                answer = worker.hosted_answer(provider, "", "Is two even?", self.root)
                self.assertEqual(answer.strip(), "YES")

    def test_provider_error_stops_without_scoring_the_item(self):
        self.env["PUZZLLM_CODEX"] = str(self.root / "missing")
        proc = self.process("run", self.state, "codex")
        result, _ = proc.communicate(timeout=5)
        events = [json.loads(line) for line in result.splitlines()]
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(events[-1]["event"], "error")
        self.assertFalse(any(e["event"] == "answer" for e in events))

    def test_invalid_text_survives_replay_and_incomplete_legacy_counts(self):
        raw = "   \t\n"
        self.env["BENCH_REPLY"] = raw
        replies = [r for r in self.result("run", self.state, "codex", "example", "high") if r["event"] == "answer"]
        self.assertEqual([(r["reply"], r["verdict"]) for r in replies], [("(whitespace only)", "invalid")] * 4)
        path, = self.reports()
        saved = [r for r in rows(path) if r["event"] == "answer"]
        self.assertEqual([(r["raw_reply"], r["reply"]) for r in saved], [(raw, "")] * 4)
        board = path.parent/"rankings.json"
        doc = json.loads(board.read_bytes())
        group, = doc["models"]
        self.assertEqual((group["invalid_answers"], group["history"][0]["wrong"]), (4, 0))
        legacy = dict(group["history"][0], file="earlier.jsonl")
        for key in ("invalid", "wrong"):
            del legacy[key]
        group["history"].append(legacy)
        doc["total_runs"] += 1
        board.write_text(json.dumps(doc))
        self.result("records", self.state)
        group, = json.loads(board.read_bytes())["models"]
        self.assertIsNone(next(r for r in group["history"] if r["file"] == "earlier.jsonl")["invalid"])
        self.assertEqual((group["invalid_answers"], group["invalid_counted_runs"], group["runs"]), (4, 1, 2))
        machine_row, = group["machine_comparisons"]
        self.assertEqual((machine_row["invalid_answers"], machine_row["invalid_counted_runs"]), (4, 1))

    def test_effort_is_passed_to_every_trial_and_groups_the_saved_scores(self):
        for provider in ("codex", "grok", "claude"):
            for effort in ("low", "high", "low"):
                self.result("run", self.state, provider, "chosen", effort)
            calls = [args for name, args, _ in rows(self.log) if name == provider]
            for args, level in zip(calls, ["low"]*4 + ["high"]*4 + ["low"]*4, strict=True):
                if provider == "codex":
                    self.assertIn(("-c", 'model_reasoning_effort="'+level+'"'), list(zip(args, args[1:])))
                else:
                    flag = "--reasoning-effort" if provider == "grok" else "--effort"
                    self.assertEqual(args[args.index(flag)+1], level)
        board = self.state/"bench/yesno/rankings.json"
        groups = json.loads(board.read_bytes())["models"]
        self.assertEqual(len(groups), 6)
        for group in groups:
            count = 2 if group["effort"] == "low" else 1
            self.assertEqual((group["runs"], group["invalid_answers"], group["avg_percent"]), (count, count, 50))
        result = subprocess.run([WORKER,"compare",board,board], env=self.env, text=True, capture_output=True, check=True)
        table = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
        self.assertEqual({(r["Provider"], r["Effort"], r["Invalid"]) for r in table},
                         {(p, e, n) for p in ("codex", "grok", "claude") for e,n in (("low","2"),("high","1"))})

    def test_decision_model_uses_native_binary_questions_and_preserves_probabilities(self):
        calls = []
        class DecisionServer(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, body))
                if self.path == "/api/show":
                    data = {"capabilities": ["decision"]}
                elif self.path == "/v1/systemone":
                    n = sum(path == self.path for path, _ in calls) - 1
                    data = {"answers": {"answer": {"type": "noul", "noul": [0.9, 0.1, 0.5, "bad"][n]}}}
                else:
                    self.send_error(400, "does not support chat")
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
        with ThreadingHTTPServer(("127.0.0.1", 0), DecisionServer) as server:
            self.env["OLLAMA_HOST"] = "127.0.0.1:" + str(server.server_port)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                self.result("run", self.state, "ollama", "autotrust-jev")
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual([path for path, _ in calls], ["/api/show"] + ["/v1/systemone"]*4)
        for (_, body), item in zip(calls[1:], QUESTIONS, strict=True):
            self.assertEqual(body["questions"], {"answer": {"type": "noul", "instructions": item["prompt"]}})
            self.assertIn(worker.INSTRUCTION, body["state"])
            if "context" in item:
                self.assertIn(item["context"], body["state"])
            self.assertNotIn("rationale", body)
        answers = [r for r in rows(self.reports()[0]) if r["event"] == "answer"]
        self.assertEqual([r["verdict"] for r in answers], ["correct"]*3 + ["invalid"])
        self.assertEqual([r["decision"]["answer"]["noul"] for r in answers], [0.9, 0.1, 0.5, "bad"])

    def test_local_catalogs_include_installed_tags_and_nested_ggufs(self):
        posts, prompts = [], []
        class Server(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"models":[{"name":"example:cloud"},{"name":"small:v2"}]}')
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                posts.append((self.path, data["model"]))
                missing = self.path == "/api/show" and data["model"] != "present:v1" and ("/api/pull", data["model"]) not in posts
                self.send_response(404 if missing else 200)
                self.end_headers()
                if self.path == "/api/chat":
                    prompts.append(data)
                    self.wfile.write(b'{"message":{"thinking":"private reasoning","content":"NO"}}')
                else:
                    self.wfile.write(b'{"status":"success"}\n')
        with ThreadingHTTPServer(("127.0.0.1", 0), Server) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                with patch.dict(os.environ, OLLAMA_HOST="127.0.0.1:" + str(server.server_port)):
                    self.assertEqual(worker.local_models("ollama"), ["example:cloud", "small:v2"])
                    worker.ensure_ollama("someone/custom:v3", lambda _text: None)
                    with worker.answerer("ollama", "present:v1", self.root, None, lambda _text: None) as ask:
                        self.assertEqual(ask("Is two odd?"), "NO")
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(posts, [("/api/show", "someone/custom:v3"), ("/api/pull", "someone/custom:v3"),
                                 ("/api/show", "someone/custom:v3"), ("/api/show", "present:v1"), ("/api/chat", "present:v1")])
        self.assertIs(prompts[0]["think"], False)
        self.assertEqual(prompts[0]["messages"], [
            {"role": "system", "content": "Respond with only YES or NO."},
            {"role": "user", "content": "Is two odd?"}])
        folder = self.root / "models/subdir"
        folder.mkdir(parents=True)
        files = [folder/f"choice {i}.GGUF" for i in range(35)]
        for path in files:
            path.touch()
        with patch.dict(os.environ, KILIX_LLM_MODELS_DIR=str(folder.parent), PUZZLLM_MODEL=""):
            self.assertEqual(set(worker.local_models("llama.cpp")), set(map(str, files)))


if __name__ == "__main__":
    unittest.main()
