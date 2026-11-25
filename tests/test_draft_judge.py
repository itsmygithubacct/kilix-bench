"""draft-judge workflow with llama-server and HTTP replaced by fakes."""

import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kilix_bench.draft_judge import cli, tui, workflow


QUESTION = "Is this reply fully correct?\n\n"
DEVICES = """Available devices:
  Vulkan0: Intel(R) UHD Graphics (1 MiB, 1 MiB free)
  Vulkan1: llvmpipe (LLVM 15) (1 MiB, 1 MiB free)
  Vulkan2: Microsoft Basic Render Driver (1 MiB, 1 MiB free)
  Vulkan3: Quadro RTX 3000 (1 MiB, 1 MiB free)
"""


def materialize(spec):
    path = spec
    if path.endswith(":cpu") or path.endswith(":gpu"):
        path = path.rsplit(":", 1)[0]
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        file.write_bytes(b"gguf")


def rows(path):
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class Response:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakePipe:
    def __init__(self, output):
        self.output = output
        self.returncode = 0
        self.pid = 0

    def communicate(self, timeout=None):
        return self.output, ""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class FakeProc:
    def __init__(self, command, runtime):
        self.command = command
        self.runtime = runtime
        self.returncode = None
        runtime.pid += 1
        self.pid = runtime.pid

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15
            self.runtime.events.append(("stop", self.command))

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class Runtime:
    """Stand-in for llama-server processes and their HTTP API."""

    def __init__(self, replies, scores=None, fail=(), devices=None, die=()):
        self.replies = list(replies)
        self.scores = dict(scores or {})
        self.fail = set(fail)
        self.devices = devices
        self.die = {name.casefold() for name in die}
        self.events = []
        self.gens = []
        self.judges = []
        self.listed = False
        self.port_model = {}
        self.pid = 900000
        self._patches = None

    def __enter__(self):
        self._patches = [
            patch("kilix_bench.draft_judge.workflow.subprocess.Popen", self.popen),
            patch("kilix_bench.draft_judge.workflow.urlopen", self.urlopen),
        ]
        for item in self._patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in reversed(self._patches):
            item.stop()
        return False

    def popen(self, cmd, stdin=None, stdout=None, stderr=None, text=False, **kwargs):
        cmd = [str(part) for part in cmd]
        if "--list-devices" in cmd:
            self.listed = True
            listing = self.devices if self.devices is not None else (
                "  Vulkan0: Test Adapter (1 MiB, 1 MiB free)\n")
            return FakePipe(listing)
        port = cmd[cmd.index("--port") + 1]
        model = Path(cmd[cmd.index("-m") + 1]).name
        self.port_model[str(port)] = model
        self.events.append(("start", cmd))
        proc = FakeProc(cmd, self)
        if model.casefold() in self.die:
            proc.returncode = 1
        return proc

    def urlopen(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/health"):
            return Response(b"ok")
        body = json.loads(request.data.decode())
        if url.endswith("/v1/systemone"):
            self.judges.append(("systemone", url, body))
            question = next(iter(body["questions"].values()))
            if not question["instructions"].startswith(QUESTION):
                raise AssertionError(question["instructions"])
            draft = question["instructions"][len(QUESTION):]
            return Response(json.dumps({
                "answers": {next(iter(body["questions"])): {"noul": self.scores[draft]}},
            }).encode())
        if body.get("logprobs") or body.get("max_tokens") == 1:
            self.judges.append(("logprobs", url, body))
            content = body["messages"][0]["content"]
            matches = [text for text in self.scores if text in content]
            if len(matches) != 1:
                raise AssertionError((matches, content))
            score = self.scores[matches[0]]
            payload = {
                "choices": [{
                    "message": {"content": "Yes"},
                    "logprobs": {"content": [{
                        "token": "Yes",
                        "logprob": math.log(score),
                        "top_logprobs": [
                            {"token": "Yes", "logprob": math.log(score)},
                            {"token": " No", "logprob": math.log(1 - score)},
                        ],
                    }]},
                }],
                "usage": {"completion_tokens": 1},
            }
            return Response(json.dumps(payload).encode())
        port = str(urllib.parse.urlparse(url).port)
        attempt = len(self.gens)
        self.gens.append({
            "model": self.port_model.get(port),
            "body": body,
            "attempt": attempt,
        })
        if attempt in self.fail:
            raise urllib.error.URLError("connection refused")
        text = self.replies[attempt]
        return Response(json.dumps({
            "choices": [{"message": {"content": text}}],
            "usage": {"completion_tokens": 4 + attempt},
        }).encode())


class DraftJudgeTest(unittest.TestCase):
    def run_cli(self, runtime, args, env=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        values = {"DRAFT_JUDGE_LLAMA_SERVER": ""}
        if env:
            values.update(env)
        with runtime, patch.dict(os.environ, values), \
                patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            code = cli.main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def assert_layout(self, directory):
        required = {
            "config.json", "vanilla.txt", "vanilla.json", "drafts.jsonl",
            "judge.jsonl", "report.txt", "resources.jsonl", "timing.json",
        }
        self.assertTrue(required <= set(os.listdir(directory)))
        timing = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
        for key in ("vanilla", "drafts", "judge"):
            self.assertGreaterEqual(timing[key], 0)
        sample_text = (directory / "resources.jsonl").read_text(encoding="utf-8").strip()
        self.assertTrue(sample_text)
        for line in sample_text.splitlines():
            row = json.loads(line)
            self.assertIn(row["phase"], {"load", "vanilla", "drafts", "judge"})
            self.assertIsInstance(row["rss"], int)
            self.assertTrue(row["gpu_memory_mib"] is None or isinstance(row["gpu_memory_mib"], int))
            self.assertGreaterEqual(row["seconds"], 0)

    def assert_server_flags(self, command):
        self.assertEqual(command[0], "/fake/llama-server")
        self.assertIn("--jinja", command)
        self.assertEqual(command[command.index("--reasoning") + 1], "off")
        self.assertEqual(command[command.index("-fa") + 1], "on")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")
        self.assertEqual(command[command.index("-c") + 1], "8192")
        self.assertEqual(command[command.index("-np") + 1], "1")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("mmproj", " ".join(command))

    def command(self, tmp, drafts, judge, prompt="Say hello", drafts_n=None, schedule=None,
                code=False, gpu_device=None, prompt_file=None):
        out = Path(tmp) / "out"
        for item in [*drafts, judge]:
            materialize(item)
        argv = ["run", "--judge", judge, "--output", str(out), "--llama-server", "/fake/llama-server"]
        for item in drafts:
            argv.extend(["--draft", item])
        if prompt_file is None:
            argv.extend(["--prompt", prompt])
        else:
            argv.extend(["--prompt-file", prompt_file])
        if drafts_n is not None:
            argv.extend(["--drafts", str(drafts_n)])
        if schedule is not None:
            argv.extend(["--schedule", schedule])
        if code:
            argv.append("--code")
        if gpu_device:
            argv.extend(["--gpu-device", gpu_device])
        return argv, out

    def test_search_order_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            known = root / "known-server"
            known.write_text("x", encoding="utf-8")
            with patch.object(workflow, "KNOWN_LLAMA_SERVER", known):
                with patch.dict(os.environ, {"DRAFT_JUDGE_LLAMA_SERVER": str(root / "from-env")}):
                    self.assertEqual(
                        workflow.find_llama_server(str(root / "from-flag")), str(root / "from-env"))
                with patch.dict(os.environ, {"DRAFT_JUDGE_LLAMA_SERVER": ""}):
                    self.assertEqual(
                        workflow.find_llama_server(str(root / "from-flag")), str(root / "from-flag"))
                    self.assertEqual(workflow.find_llama_server(None), str(known))
                with patch.object(workflow, "KNOWN_LLAMA_SERVER", root / "missing"), \
                        patch.dict(os.environ, {"DRAFT_JUDGE_LLAMA_SERVER": ""}), \
                        patch("kilix_bench.draft_judge.workflow.shutil.which",
                              return_value="/usr/bin/llama-server"):
                    self.assertEqual(workflow.find_llama_server(None), "/usr/bin/llama-server")
        omitted = workflow.parse_spec("/tmp/model.gguf")
        self.assertEqual(omitted.device, "gpu")
        self.assertEqual(omitted.name, "model.gguf")
        cpu = workflow.parse_spec("/tmp/weird:name.gguf:cpu")
        self.assertEqual(cpu.device, "cpu")
        self.assertEqual(cpu.name, "weird:name.gguf")
        self.assertEqual(workflow.pick_device(DEVICES), "Vulkan3")
        self.assertEqual(
            workflow.pick_device("  Vulkan0: intel uhd\n  CUDA0: Example GPU\n"), "CUDA0")

    def test_help_names_the_environment_variable(self):
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(cli.main(["--help"]), 0)
        text = stdout.getvalue()
        self.assertIn("DRAFT_JUDGE_LLAMA_SERVER", text)
        lowered = text.casefold()
        self.assertNotIn("localhost", lowered)
        self.assertNotIn("127.0.0.1", text)
        self.assertNotIn("/home/", text)

    def test_menu_requires_a_terminal(self):
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stderr", io.StringIO()):
            self.assertEqual(cli.main([]), 2)
            self.assertEqual(cli.main(["tui"]), 2)

    def test_prompt_is_required(self):
        with patch("sys.stderr", io.StringIO()):
            self.assertEqual(cli.main([
                "run", "--draft", "a.gguf:cpu", "--judge", "b.gguf:cpu",
                "--llama-server", "/fake/llama-server",
            ]), 2)
            self.assertEqual(cli.main([
                "run", "--draft", "a.gguf:cpu", "--judge", "b.gguf:cpu",
                "--prompt", "x", "--prompt-file", "y", "--llama-server", "/fake/llama-server",
            ]), 2)
            self.assertEqual(cli.main([
                "run", "--draft", "a.gguf:cpu", "--judge", "b.gguf:cpu",
                "--prompt", "x", "--drafts", "0", "--llama-server", "/fake/llama-server",
            ]), 2)

    def test_round_robin_greedy_and_sampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.gguf", root / "b.gguf"
            judge = root / "j.gguf"
            argv, out = self.command(
                root, [f"{first}:cpu", f"{second}:cpu"], f"{judge}:cpu")
            runtime = Runtime(
                ["VANILLA", "D0", "D1", "D2", "D3"],
                {"D0": 0.25, "D1": 0.5, "D2": 0.25, "D3": 0.25})
            code, stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            self.assert_layout(out)
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["schedule"], "rotate")
            self.assertEqual(config["draft_count"], 4)
            self.assertFalse(config["code"])
            drafts = rows(out / "drafts.jsonl")
            self.assertEqual([row["model"] for row in drafts], ["a.gguf", "b.gguf", "a.gguf", "b.gguf"])
            self.assertEqual([row["temp"] for row in drafts], [0, 0.8, 0.8, 0.8])
            self.assertEqual([item["model"] for item in runtime.gens],
                             ["a.gguf", "a.gguf", "b.gguf", "a.gguf", "b.gguf"])
            bodies = [item["body"] for item in runtime.gens]
            self.assertEqual([body["temperature"] for body in bodies], [0, 0, 0.8, 0.8, 0.8])
            self.assertEqual([body["seed"] for body in bodies], [1000, 1000, 1001, 1002, 1003])
            self.assertTrue(all(body["max_tokens"] == 768 for body in bodies))
            self.assertTrue(all(
                body["chat_template_kwargs"] == {"enable_thinking": False} for body in bodies))
            self.assertEqual(bodies[0]["messages"], [{"role": "user", "content": "Say hello"}])
            self.assertIn(str(out), stdout)

    def test_rotate_stops_a_server_before_the_next_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, out = self.command(
                root,
                [f"{root / 'a.gguf'}:cpu", f"{root / 'b.gguf'}:cpu"],
                f"{root / 'c.gguf'}:cpu",
                drafts_n=2, schedule="rotate")
            runtime = Runtime(["VANILLA", "D0", "D1"], {"D0": 0.25, "D1": 0.5})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            kinds = [kind for kind, _command in runtime.events]
            self.assertEqual(kinds, ["start", "stop", "start", "stop", "start", "stop"])
            for kind, command in runtime.events:
                if kind == "start":
                    self.assert_server_flags(command)
                    self.assertEqual(command[command.index("-ngl") + 1], "0")
                    self.assertEqual(command[command.index("--device") + 1], "none")
            self.assertFalse(runtime.listed)
            self.assertTrue((out / "report.txt").is_file())

    def test_together_keeps_servers_until_the_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, _out = self.command(
                root,
                [f"{root / 'a.gguf'}:cpu", f"{root / 'b.gguf'}:cpu"],
                f"{root / 'c.gguf'}:cpu",
                drafts_n=2, schedule="together")
            runtime = Runtime(["VANILLA", "D0", "D1"], {"D0": 0.25, "D1": 0.5})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            kinds = [kind for kind, _command in runtime.events]
            self.assertEqual(kinds[:3], ["start", "start", "start"])
            self.assertNotIn("stop", kinds[:3])

    def test_d3_uses_systemone_and_a_normal_gguf_uses_logprobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d3_argv, d3_out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'D3-judge.gguf'}:cpu", drafts_n=1)
            d3 = Runtime(["VANILLA", "DRAFT"], {"DRAFT": 0.5})
            code, _stdout, _stderr = self.run_cli(d3, d3_argv)
            self.assertEqual(code, 0)
            self.assertEqual([kind for kind, _url, _body in d3.judges], ["systemone"])
            _kind, url, body = d3.judges[0]
            self.assertTrue(url.endswith("/v1/systemone"))
            self.assertEqual(body["state"], {"task": "Say hello"})
            self.assertEqual(body["questions"]["d0"]["type"], "noul")
            self.assertEqual(body["questions"]["d0"]["instructions"], QUESTION + "DRAFT")
            self.assertEqual(rows(d3_out / "judge.jsonl")[0]["score"], 0.5)
            judge_cmd = next(
                command for kind, command in d3.events
                if kind == "start" and command[command.index("-m") + 1].endswith("D3-judge.gguf"))
            self.assertEqual(judge_cmd[judge_cmd.index("-b") + 1], "8192")
            self.assertEqual(judge_cmd[judge_cmd.index("-ub") + 1], "8192")
            draft_cmd = next(
                command for kind, command in d3.events
                if kind == "start" and command[command.index("-m") + 1].endswith("a.gguf"))
            self.assertNotIn("-b", draft_cmd)
            self.assertNotIn("-ub", draft_cmd)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'qwen-judge.gguf'}:cpu", drafts_n=1)
            runtime = Runtime(["VANILLA", "DRAFT"], {"DRAFT": 0.75})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            self.assertEqual([kind for kind, _url, _body in runtime.judges], ["logprobs"])
            _kind, url, body = runtime.judges[0]
            self.assertTrue(url.endswith("/v1/chat/completions"))
            self.assertNotIn("systemone", url)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["max_tokens"], 1)
            self.assertIs(body["logprobs"], True)
            self.assertEqual(body["top_logprobs"], 20)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            content = body["messages"][0]["content"]
            self.assertIn("Task:\nSay hello\n\n", content)
            self.assertIn(QUESTION + "DRAFT", content)
            self.assertIn("Answer with only Yes or No.", content)
            self.assertAlmostEqual(rows(out / "judge.jsonl")[0]["score"], 0.75, places=5)

    def test_tie_keeps_the_earliest_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'D3-judge.gguf'}:cpu", drafts_n=3)
            runtime = Runtime(
                ["VANILLA", "LOW", "HIGH-A", "HIGH-B"],
                {"LOW": 0.25, "HIGH-A": 0.5, "HIGH-B": 0.5})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            report = (out / "report.txt").read_text(encoding="utf-8")
            self.assertIn("chosen\nHIGH-A\n", report)
            self.assertNotIn("HIGH-B", report)
            scores = [row["score"] for row in rows(out / "judge.jsonl")]
            self.assertEqual(scores, [0.25, 0.5, 0.5])

    def test_vanilla_is_saved_when_a_later_draft_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'D3-judge.gguf'}:cpu", drafts_n=2)
            runtime = Runtime(
                ["VANILLA", "EARLY", "LATER"], {"EARLY": 0.25, "LATER": 0.75})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            self.assertEqual((out / "vanilla.txt").read_text(encoding="utf-8"), "VANILLA")
            vanilla = json.loads((out / "vanilla.json").read_text(encoding="utf-8"))
            self.assertTrue(vanilla["ok"])
            self.assertEqual(vanilla["text"], "VANILLA")
            self.assertEqual(vanilla["tokens"], 4)
            report = (out / "report.txt").read_text(encoding="utf-8")
            self.assertLess(report.index("VANILLA"), report.index("LATER"))
            self.assertIn("chosen\nLATER\n", report)
            self.assertEqual(rows(out / "drafts.jsonl")[1]["text"], "LATER")

    def test_failed_draft_is_recorded_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'D3-judge.gguf'}:cpu", drafts_n=3)
            # Attempts: 0 vanilla, 1 draft 0, 2 draft 1, 3 draft 2.
            runtime = Runtime(
                ["VANILLA", "KEEP", "UNUSED", "LATER"],
                {"KEEP": 0.25, "LATER": 0.75},
                fail={2})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            drafts = rows(out / "drafts.jsonl")
            self.assertEqual([row["ok"] for row in drafts], [True, False, True])
            self.assertIn("connection refused", drafts[1]["error"])
            self.assertEqual(drafts[1]["text"], "")
            judged = rows(out / "judge.jsonl")
            self.assertEqual([row["index"] for row in judged], [0, 2])
            self.assertTrue(all(row["ok"] for row in judged))
            self.assertEqual(len(runtime.judges), 2)
            instructions = [
                next(iter(body["questions"].values()))["instructions"]
                for _kind, _url, body in runtime.judges
            ]
            self.assertEqual(instructions, [QUESTION + "KEEP", QUESTION + "LATER"])

    def test_exit_one_when_nothing_can_be_judged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'j.gguf'}:cpu", drafts_n=2)
            runtime = Runtime(["VANILLA", "D0", "D1"], fail={0, 1, 2})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 1)
            self.assertEqual(runtime.judges, [])
            self.assertEqual((out / "judge.jsonl").read_text(encoding="utf-8"), "")
            vanilla = json.loads((out / "vanilla.json").read_text(encoding="utf-8"))
            self.assertFalse(vanilla["ok"])
            self.assertTrue(all(not row["ok"] for row in rows(out / "drafts.jsonl")))
            self.assertIn("(no reply could be judged)", (out / "report.txt").read_text(encoding="utf-8"))
            self.assert_layout(out)

    def test_code_prompt_and_prompt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, _out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'j.gguf'}:cpu",
                prompt="def f():", drafts_n=1, code=True)
            runtime = Runtime(["VANILLA", "DRAFT"], {"DRAFT": 0.5})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            messages = runtime.gens[0]["body"]["messages"]
            self.assertEqual(messages[0], {
                "role": "system",
                "content": "You are a careful Python programmer.",
            })
            self.assertEqual(
                messages[1]["content"],
                "Complete the following Python function. Reply with the complete function "
                "(signature and body) in one ```python block and nothing else.\n\ndef f():")
            self.assertEqual(runtime.gens[1]["body"]["messages"], messages)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_file = root / "prompt.txt"
            prompt_file.write_text("from the file", encoding="utf-8")
            argv, _out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'j.gguf'}:cpu",
                drafts_n=1, prompt_file=str(prompt_file))
            runtime = Runtime(["VANILLA", "DRAFT"], {"DRAFT": 0.5})
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            self.assertEqual(
                runtime.gens[0]["body"]["messages"],
                [{"role": "user", "content": "from the file"}])

    def test_gpu_device_skips_software_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, _out = self.command(
                root, [str(root / "a.gguf")], f"{root / 'j.gguf'}:cpu", drafts_n=1)
            runtime = Runtime(["VANILLA", "DRAFT"], {"DRAFT": 0.5}, devices=DEVICES)
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            self.assertTrue(runtime.listed)
            gpu = [command for kind, command in runtime.events
                   if kind == "start" and "Vulkan3" in command]
            self.assertEqual(len(gpu), 1)
            self.assertEqual(gpu[0][gpu[0].index("-ngl") + 1], "99")
            self.assertEqual(gpu[0][gpu[0].index("--device") + 1], "Vulkan3")

    def test_gpu_device_override_skips_the_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, _out = self.command(
                root, [str(root / "a.gguf")], f"{root / 'j.gguf'}:cpu",
                drafts_n=1, gpu_device="Vulkan9")
            runtime = Runtime(["VANILLA", "DRAFT"], {"DRAFT": 0.5}, devices=DEVICES)
            code, _stdout, _stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 0)
            self.assertFalse(runtime.listed)
            gpu = [command for kind, command in runtime.events
                   if kind == "start" and command[command.index("--device") + 1] == "Vulkan9"]
            self.assertEqual(len(gpu), 1)
            self.assertEqual(gpu[0][gpu[0].index("-ngl") + 1], "99")

    def test_model_list_skips_the_ollama_blob_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "models"
            good.mkdir()
            (good / "draft-judge-visible.gguf").write_bytes(b"gguf")
            blobs = root / "ollama" / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "draft-judge-hidden.gguf").write_bytes(b"gguf")
            found = {path.name for path in workflow.discover_models([good, blobs])}
            self.assertIn("draft-judge-visible.gguf", found)
            self.assertNotIn("draft-judge-hidden.gguf", found)

    def test_tui_walks_the_choices_and_prints_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            draft = root / "draft.gguf"
            judge = root / "judge.gguf"
            draft.write_bytes(b"gguf")
            judge.write_bytes(b"gguf")
            answers = iter([
                str(draft),
                "cpu",
                "",
                str(judge),
                "cpu",
                "2",
                "rotate",
                "",
                "Say hello",
                "n",
                "",
            ])

            def fake_input(prompt=""):
                try:
                    return next(answers)
                except StopIteration as exc:
                    raise AssertionError(prompt) from exc

            runtime = Runtime(["VANILLA", "AAA", "BBB"], {"AAA": 0.25, "BBB": 0.75})
            stdout = io.StringIO()
            with runtime, patch.dict(os.environ, {
                "DRAFT_JUDGE_LLAMA_SERVER": "",
                "XDG_STATE_HOME": str(state),
            }), patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", fake_input), patch("sys.stdout", stdout):
                code = cli.main([
                    "tui", "--model-dir", str(root / "empty"),
                    "--llama-server", "/fake/llama-server",
                ])
            self.assertEqual(code, 0)
            text = stdout.getvalue()
            self.assertIn("VANILLA", text)
            self.assertIn("BBB", text)
            self.assertLess(text.index("VANILLA"), text.index("BBB"))
            runs = list((state / "draft-judge" / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertIn(str(runs[0]), text)
            self.assertEqual((runs[0] / "vanilla.txt").read_text(encoding="utf-8"), "VANILLA")
            self.assertIn("chosen\nBBB\n", (runs[0] / "report.txt").read_text(encoding="utf-8"))

    def test_missing_model_file_stops_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = Runtime([])
            code, _stdout, stderr = self.run_cli(runtime, [
                "run", "--draft", str(root / "missing.gguf"),
                "--judge", str(root / "also-missing.gguf"),
                "--prompt", "hello", "--llama-server", "/fake/llama-server",
                "--output", str(root / "out"),
            ])
            self.assertEqual(code, 2)
            self.assertEqual(runtime.events, [])
            self.assertIn("model file not found", stderr)
            self.assertFalse((root / "out").exists())

    def test_judge_that_exits_does_not_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, out = self.command(
                root, [f"{root / 'a.gguf'}:cpu"], f"{root / 'c.gguf'}:cpu",
                drafts_n=4, schedule="together")
            runtime = Runtime(["VANILLA"], die={"c.gguf"})
            code, _stdout, stderr = self.run_cli(runtime, argv)
            self.assertEqual(code, 1)
            self.assertEqual(runtime.gens, [])
            report = (out / "report.txt").read_text(encoding="utf-8")
            self.assertIn("result: judge failed to load", report)
            self.assertIn("llama-server exited (1)", report)
            self.assertIn("judge failed to load", stderr)

    def test_log_excerpt_keeps_the_server_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                "0.00.121 I srv  llama_server: initializing ...\n"
                "0.00.124 E gguf_init_from_file: failed to open GGUF file\n"
                "0.00.127 E srv  load_model: failed to load model\n",
                encoding="utf-8")
            excerpt = workflow.log_excerpt(path)
            self.assertIn("failed to open GGUF file", excerpt)
            self.assertIn("failed to load model", excerpt)
            self.assertNotIn("initializing", excerpt)

    def test_http_error_includes_the_server_message(self):
        body = json.dumps({
            "error": {"message": "input (870 tokens) is too large to process"},
        }).encode()
        exc = urllib.error.HTTPError(
            "http://unit.test/v1/systemone", 500, "err", {}, io.BytesIO(body))
        self.assertEqual(
            workflow.error_text(exc),
            "HTTP 500: input (870 tokens) is too large to process")

    def test_short_name_matches_ignore_punctuation(self):
        models = [
            Path("Qwen3.5-0.8B-Q8_0.gguf"),
            Path("Qwen3.5-4B-Q4_K_M.gguf"),
            Path("Qwen3.5-4B-Q6_K.gguf"),
            Path("d3-2b-Q8_0.gguf"),
            Path("d3-2b-Q8_0-extra.gguf"),
        ]
        one = tui.match_models(models, "d3:2b")
        self.assertEqual([path.name for _index, path in one], [
            "d3-2b-Q8_0.gguf", "d3-2b-Q8_0-extra.gguf"])
        exact = tui.match_models(models, "d3-2b-Q8_0.gguf")
        self.assertEqual([path.name for _index, path in exact], ["d3-2b-Q8_0.gguf"])
        small = tui.match_models(models, "0.8")
        self.assertEqual([path.name for _index, path in small], ["Qwen3.5-0.8B-Q8_0.gguf"])
        self.assertEqual(tui.strip_wrapping_quotes('"what is the meaning of life?"'),
                         "what is the meaning of life?")
        self.assertEqual(tui.split_device("d3:2b cpu"), ("d3:2b", "cpu"))
        self.assertEqual(tui.split_device("4:gpu"), ("4", "gpu"))

    def test_menu_short_name_sets_the_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            names = [
                "Qwen3.5-0.8B-Q8_0.gguf",
                "Qwen3.5-4B-Q4_K_M.gguf",
                "d3-2b-Q8_0.gguf",
            ]
            models = []
            for name in names:
                path = root / name
                path.write_bytes(b"gguf")
                models.append(path.resolve())
            answers = iter([
                "0.8 cpu",
                "",
                "d3:2b",
                "",
                "1",
                "rotate",
                "",
                '"what is the meaning of life?"',
                "n",
                "",
            ])

            def fake_input(prompt=""):
                try:
                    return next(answers)
                except StopIteration as exc:
                    raise AssertionError(prompt) from exc

            runtime = Runtime(["VANILLA", "ONLY"], {"ONLY": 0.8})
            stdout = io.StringIO()
            with runtime, patch.dict(os.environ, {
                "DRAFT_JUDGE_LLAMA_SERVER": "",
                "XDG_STATE_HOME": str(state),
            }), patch("sys.stdin.isatty", return_value=True), \
                    patch("kilix_bench.draft_judge.tui.discover_models", return_value=models), \
                    patch("builtins.input", fake_input), patch("sys.stdout", stdout):
                code = cli.main([
                    "tui", "--llama-server", "/fake/llama-server",
                ])
            self.assertEqual(code, 0)
            runs = list((state / "draft-judge" / "runs").iterdir())
            config = json.loads((runs[0] / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["draft_models"][0]["file"], "Qwen3.5-0.8B-Q8_0.gguf")
            self.assertEqual(config["draft_models"][0]["device"], "cpu")
            self.assertEqual(config["judge"]["file"], "d3-2b-Q8_0.gguf")
            self.assertEqual(config["judge"]["device"], "gpu")
            self.assertEqual(config["prompt"], "what is the meaning of life?")
            self.assertIn("Matched", stdout.getvalue())
            self.assertEqual([kind for kind, _url, _body in runtime.judges], ["systemone"])

    def test_menu_can_go_back_and_change_one_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["a.gguf", "qwen-4b-q4.gguf", "qwen-4b-q6.gguf", "d3-judge.gguf"]
            models = []
            for name in names:
                path = root / name
                path.write_bytes(b"gguf")
                models.append(path.resolve())
            answers = iter([
                "4b",
                "2 cpu",
                "1",
                "b",
                "b",
                "3 gpu",
                "",
                "d3",
                "",
                "2",
                "together",
                "",
                "Say hello",
                "n",
                "4",
                "rotate",
                "5",
                "",
                "",
            ])

            def fake_input(prompt=""):
                try:
                    return next(answers)
                except StopIteration as exc:
                    raise AssertionError(prompt) from exc

            seen = {}

            def fake_execute(config):
                seen["config"] = config
                return 0, root, "report\n"

            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, {"DRAFT_JUDGE_LLAMA_SERVER": ""}), \
                    patch("sys.stdin.isatty", return_value=True), \
                    patch("kilix_bench.draft_judge.tui.discover_models", return_value=models), \
                    patch("kilix_bench.draft_judge.tui.execute", fake_execute), \
                    patch("builtins.input", fake_input), \
                    patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = cli.main(["tui", "--llama-server", "/fake/llama-server"])
            self.assertEqual(code, 0)
            text = stdout.getvalue()
            self.assertIn("Several models match", text)
            self.assertIn("Removed qwen-4b-q4.gguf.", text)
            config = seen["config"]
            self.assertEqual([spec.name for spec in config["draft_models"]], ["qwen-4b-q6.gguf"])
            self.assertEqual(config["draft_models"][0].device, "gpu")
            self.assertEqual(config["judge"].name, "d3-judge.gguf")
            self.assertEqual(config["judge"].device, "gpu")
            self.assertEqual(config["draft_count"], 2)
            self.assertEqual(config["schedule"], "rotate")
            self.assertEqual(config["prompt"], "Say hello")
            self.assertFalse(config["code"])

    def test_menu_quit_does_not_run(self):
        answers = iter(["q"])

        def fake_input(prompt=""):
            return next(answers)

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.stdin.isatty", return_value=True), \
                patch("kilix_bench.draft_judge.tui.discover_models", return_value=[]), \
                patch("kilix_bench.draft_judge.tui.execute") as execute, \
                patch("builtins.input", fake_input), \
                patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            code = cli.main(["tui"])
        self.assertEqual(code, 1)
        self.assertIn("cancelled", stderr.getvalue())
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
