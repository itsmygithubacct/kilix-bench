"""The OneBit runner with fake backends: parsing, arms, records, results."""

import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from kilix_bench.onebit import contract, runner
from kilix_bench.onebit.backends import (
    Backend, ChatBackend, CommandBackend, chat_usage, first_token_logprobs)
from kilix_bench.report import write_run


class ParseTest(unittest.TestCase):
    def test_exact_digits(self):
        self.assertEqual(contract.parse_reply("0"), 0)
        self.assertEqual(contract.parse_reply("1"), 1)

    def test_whitespace_is_stripped(self):
        self.assertEqual(contract.parse_reply(" 0\n"), 0)
        self.assertEqual(contract.parse_reply("\t1  "), 1)

    def test_anything_else_is_a_format_failure(self):
        for reply in ("0.", "1.", "Yes", "yes", "No", "", "   ", "01", "10",
                      "0 1", "`0`", "'1'", "<think>\n</think>0", "Answer: 0",
                      "zero", "０", None):
            with self.subTest(reply=reply):
                self.assertIsNone(contract.parse_reply(reply))


class ContractTest(unittest.TestCase):
    def test_message_with_information(self):
        item = {"context": "exit 1", "prompt": "Did it pass?", "answer": 1}
        chat = contract.messages(item)
        self.assertEqual(chat[0]["role"], "system")
        self.assertEqual(
            chat[1]["content"],
            "<information>\nexit 1\n</information>\n\nDid it pass?")

    def test_message_without_information(self):
        chat = contract.messages({"prompt": "Is 2 + 2 = 4?", "answer": 0})
        self.assertEqual(chat[1]["content"], "Is 2 + 2 = 4?")

    def test_system_prompt_states_the_contract(self):
        text = contract.system_prompt()
        self.assertIn("0 means yes, 1 means no", text)
        self.assertIn("Did the build fail?", text)
        self.assertEqual(len(contract.system_prompt_sha256()), 64)

    def test_item_hash_changes_with_the_key(self):
        item = {"prompt": "Is 2 + 2 = 4?", "answer": 0}
        flipped = dict(item, answer=1)
        self.assertNotEqual(contract.item_sha256(item),
                            contract.item_sha256(flipped))


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


class FakeBackend(Backend):
    """Answers from a table; the constrained arm is reported unsupported."""

    kind = "chat"

    def __init__(self, replies):
        self.replies = replies
        self.asked = []

    def describe(self):
        return {"flavor": "fake", "model": "fake-model"}

    def supports(self, arm):
        if arm == "raw":
            return True, "no grammar"
        return False, "fake has no grammar"

    def ask(self, messages, arm):
        self.asked.append(messages[-1]["content"])
        return {"reply": self.replies.pop(0), "latency_ms": 1.5,
                "logprob_0": -0.1, "logprob_1": -2.3}

    def settings(self):
        return {"temperature": 0, "max_tokens": 4}


class RunTest(unittest.TestCase):
    def run_fake(self, replies, **spec):
        backend = FakeBackend(list(replies))
        with mock.patch.object(runner.itemset, "load",
                               return_value=[dict(row) for row in ITEMS]):
            payload = runner.run(
                {"split": "dev", "sizer_command": "off", **spec},
                backend=backend)
        return payload, backend

    def test_records_and_summary(self):
        payload, backend = self.run_fake(["0", "Yes", " 1\n"])
        raw = payload["arms"]["raw"]
        self.assertTrue(raw["supported"])
        replies = [(row["id"], row["reply"], row["parsed"])
                   for row in raw["records"]]
        self.assertEqual(replies, [("t-a", "0", 0), ("t-b", "Yes", None),
                                   ("t-c", " 1\n", 1)])
        self.assertEqual(raw["records"][0]["latency_ms"], 1.5)
        self.assertEqual(raw["records"][0]["logprob_1"], -2.3)
        overall = raw["summary"]["overall"]
        self.assertEqual(overall["correct"], 2)
        self.assertEqual(overall["format_failures"], 1)
        self.assertEqual(backend.asked[2],
                         "<information>\nERROR disk full\n</information>"
                         "\n\nVerdict on this job.")

    def test_unsupported_arm_is_recorded(self):
        payload, _backend = self.run_fake(["0", "1", "1"])
        self.assertEqual(payload["arms"]["constrained"],
                         {"supported": False, "reason": "fake has no grammar"})

    def test_record_carries_provenance(self):
        payload, _backend = self.run_fake(["0", "1", "1"])
        self.assertEqual(payload["contract"]["system_prompt_sha256"],
                         contract.system_prompt_sha256())
        self.assertEqual(payload["items"]["split"], "dev")
        self.assertEqual(len(payload["items"]["set_sha256"]), 64)
        self.assertTrue(payload["items"]["files_sha256"])
        self.assertIn("canary", payload["items"])

    def test_long_items_are_skipped_and_counted(self):
        payload, backend = self.run_fake([], max_input_tokens=1)
        overall = payload["arms"]["raw"]["summary"]["overall"]
        self.assertEqual(overall["skipped"], 3)
        self.assertEqual(backend.asked, [])
        self.assertEqual(payload["input_limit"]["source"], "option")

    def test_render_names_chance(self):
        payload, _backend = self.run_fake(["0", "1", "1"])
        text = runner.render(payload)
        self.assertIn("chance 50.0%", text)
        self.assertIn("arm constrained: unsupported", text)


class UsageTest(unittest.TestCase):
    def test_chat_usage_in_the_shared_shape(self):
        self.assertEqual(
            chat_usage({"prompt_tokens": 200, "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 150}}),
            {"input_tokens": 200, "cached_input_tokens": 150,
             "output_tokens": 1, "reasoning_output_tokens": None})
        self.assertIsNone(chat_usage(None))
        self.assertIsNone(chat_usage({"total_tokens": 3}))


class LogprobTest(unittest.TestCase):
    def test_first_token_logprobs(self):
        choice = {"logprobs": {"content": [{
            "token": "0", "logprob": -0.05,
            "top_logprobs": [{"token": "0", "logprob": -0.05},
                             {"token": "1", "logprob": -3.0}]}]}}
        self.assertEqual(first_token_logprobs(choice), (-0.05, -3.0))
        self.assertEqual(first_token_logprobs({}), (None, None))


class _Server(BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *args):
        pass

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/props":
            self._send({"default_generation_settings": {},
                        "model_path": "/models/deep/Qwen-test-Q8_0.gguf"})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _Server.bodies.append(body)
        reply = "1" if "grammar" in body else "0.\n"
        self._send({"choices": [{
            "message": {"content": reply}, "finish_reason": "stop",
            "logprobs": {"content": [{"token": reply[0], "logprob": -0.2,
                                      "top_logprobs": []}]}}]})


class ChatBackendTest(unittest.TestCase):
    def setUp(self):
        _Server.bodies = []
        self.server = HTTPServer(("127.0.0.1", 0), _Server)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_llama_server_arms(self):
        backend = ChatBackend(self.url + "/v1/chat/completions", "m",
                              max_tokens=3)
        self.assertEqual(backend.flavor, "llama-server")
        self.assertEqual(backend.describe()["model_file"],
                         "Qwen-test-Q8_0.gguf")
        chat = contract.messages(ITEMS[0])
        raw = backend.ask(chat, "raw")
        constrained = backend.ask(chat, "constrained")
        self.assertIsNone(contract.parse_reply(raw["reply"]))
        self.assertEqual(contract.parse_reply(constrained["reply"]), 1)
        first, second = _Server.bodies
        self.assertNotIn("grammar", first)
        self.assertEqual(second["grammar"], 'root ::= "0" | "1"')
        self.assertEqual(first["temperature"], 0)
        self.assertEqual(first["max_tokens"], 3)
        self.assertEqual(first["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertEqual(backend.settings()["thinking_control"], "sent")

    def test_other_servers_have_no_constrained_arm(self):
        backend = ChatBackend(self.url, "m", flavor="openai")
        supported, why = backend.supports("constrained")
        self.assertFalse(supported)
        self.assertIn("openai", why)


class CommandBackendTest(unittest.TestCase):
    def backend(self, code):
        script = ("import sys; text = sys.stdin.read(); "
                  f"sys.exit({code} if 'information' in text else 9)")
        return CommandBackend([sys.executable, "-c", script])

    def test_exit_status_is_the_answer(self):
        chat = contract.messages(ITEMS[2])
        for code, parsed in ((0, 0), (1, 1), (3, None)):
            with self.subTest(code=code):
                answer = self.backend(code).ask(chat, "command")
                self.assertEqual(contract.parse_reply(answer["reply"]), parsed)
                self.assertEqual(answer["exit_status"], code)
        failure = self.backend(3).ask(chat, "command")
        self.assertEqual(failure["tool_failure"], "exit 3")


class ResultFileTest(unittest.TestCase):
    def test_written_privately_and_paths_refused(self):
        payload, _backend = RunTest.run_fake(RunTest(), ["0", "1", "1"])
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {"KILIX_BENCH_HOME": temporary}):
                stamped, latest, text = write_run(
                    payload, slug="onebit-fake-dev", render=runner.render)
                self.assertTrue(stamped.name.endswith("-onebit-fake-dev.json"))
                saved = json.loads(latest.read_text(encoding="utf-8"))
                self.assertEqual(saved["suite"], "onebit")
                self.assertIn("OneBit", text.read_text(encoding="utf-8"))
                mode = stat.S_IMODE(Path(stamped).stat().st_mode)
                self.assertEqual(mode & 0o077, 0)
                leaky = dict(payload, endpoint="/" + "srv/secret-place/x")
                with self.assertRaises(SystemExit):
                    write_run(leaky, forbidden=("/srv/secret-place",),
                              slug="onebit-leak", render=runner.render)


if __name__ == "__main__":
    unittest.main()
