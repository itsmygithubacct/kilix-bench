"""The native answer actions, verified model selection, and independent turns."""
import base64
from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kilix_bench.yesno import local, needle2, worker


class NeedleTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.scratch = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.assets = Mock()
        self.identity = Mock()

    def image(self, path, digest):
        file = self.stack.enter_context(path.open("rb"))
        return nullcontext(SimpleNamespace(fd=file.fileno(), sha256=digest))

    def test_native_actions_and_invalid_replies(self):
        choices = {"answer_yes": "YES", "answer_no": "NO"}
        self.assertEqual({tool["name"] for tool in local.needle_tools()}, set(choices))
        for name, word in choices.items():
            envelope = {"function_calls": [{"name": name, "arguments": {}}], "raw_text": "raw model output"}
            reply = local.needle_reply(envelope)
            self.assertEqual(reply.text, word)
            self.assertIs(reply.evidence["needle_reply"], envelope)
        bad = [
            {"function_calls": []},
            {"function_calls": [{"name": "answer_yes", "arguments": {"value": "NO"}}]},
            {"function_calls": [{"name": "answer_no", "arguments": {}},
                                {"name": "answer_yes", "arguments": {}}]},
            {"function_calls": [{"name": "unrelated", "arguments": {}}]},
            {"function_calls": "YES", "raw_text": "NO"},
        ]
        for marker in ({"type": "error"}, {"error": "truncated"}, {"success": False},
                       {"reason": "runtime_failure"}):
            bad.append(dict(envelope, **marker, raw_text="YES"))
            bad.append(dict(marker, function_calls=[], raw_text="NO"))
        for result in bad:
            with self.subTest(result=result):
                parsed = local.needle_reply(result)
                self.assertNotIn(parsed.text.strip(), ("YES", "NO"))
                self.assertEqual(parsed.evidence["needle_reply"], result)
        self.assertEqual(local.needle_reply({"raw_text": "NO"}).text, "NO")

    def test_incomplete_utf8_in_the_explanation_preserves_the_vote_and_original_bytes(self):
        output = (b'{"success":true,"function_calls":[{"name":"answer_no","arguments":{}}],'
                  b'"reasoning":"truncated \xe2\x80"}')
        native = needle2.decode_reply(output)
        self.assertEqual(local.needle_reply(native).text, "NO")
        self.assertEqual(native["reasoning"], "truncated \ufffd")
        self.assertEqual(base64.b64decode(native["native_output_base64"]), output)
        for damaged in (output.replace(b"true", b"false"), output.replace(b"answer_no", b"answer_\xfe"),
                        output.replace(b'"success"', b'"su\xffccess"'), output[:-1]):
            result = needle2.decode_reply(damaged)
            self.assertNotIn(local.needle_reply(result).text.strip(), ("YES", "NO"))
            self.assertEqual(base64.b64decode(result["native_output_base64"]), damaged)
            json.dumps(result, ensure_ascii=False).encode("utf-8")

    def test_stock_uses_exact_prompt_and_preserves_malformed_native_output(self):
        exe = self.scratch / "stock"
        # The fixture checks the actual argv and schema outside the parent process.
        exe.write_text("""#!/usr/bin/python3
import json, pathlib, sys
args = sys.argv[1:]
assert args[2:5] == ["--max", "256", "--prompt"] and len(args) == 6
tools = json.loads(pathlib.Path(args[1]).read_text())
assert len(tools) == 2 and all(t['parameters']['properties'] == {} for t in tools)
if args[5] == 'bad':
    print('{broken')
elif args[5] == 'bad-shape':
    print('[]')
else:
    print(json.dumps({'function_calls':[{'name':'answer_yes','arguments':{}}], 'input':args[5]}))
""")
        exe.chmod(0o700)
        self.assets.from_installed.return_value = self.image(exe, "stock-digest")
        with needle2.open_needle2(self.assets, SimpleNamespace(selected=lambda: None), "root",
                                  local.needle_tools(), self.scratch, self.identity) as ask:
            text = 'Line one\n"Quoted" path C:\\data\\x and π'
            bad = ask("bad")
            self.assertIn("{broken", bad["raw_text"])
            self.assertNotIn(local.needle_reply(bad).text.strip(), ("YES", "NO"))
            self.assertEqual(ask("bad-shape")["type"], "error")
            self.assertEqual(ask(text)["input"], text)
        self.assets.from_installed.assert_called_once_with("root")
        self.identity.assert_called_once_with("stock-digest")

    @unittest.skipUnless(shutil.which("cc"), "a C compiler is needed for the native ABI fixture")
    def test_selected_library_gets_verified_weights_and_fresh_state_after_a_bad_reply(self):
        source = self.scratch / "native.c"
        library = self.scratch / "libtest.so"
        source.write_text(r"""
#include <stdio.h>
#include <string.h>
int needle_load(const char *data, unsigned long long n) {
    return n == 5 && memcmp(data, "TUNED", 5) == 0 ? 0 : -1;
}
int needle_init(const char *unused, const char *tools, const char *options) {
    return strstr(tools, "answer_yes") && strstr(tools, "answer_no");
}
int needle_complete(const char *input, int limit, char *out, int capacity) {
    static int calls = 0;
    if (++calls != 1 || limit != 256) return -1;
    if (!strcmp(input, "malformed")) return snprintf(out, capacity, "{broken native JSON");
    if (!strcmp(input, "engine error")) return -1;
    if (!strcmp(input, "yes\ncontext")) return snprintf(out, capacity,
        "{\"function_calls\":[{\"name\":\"answer_yes\",\"arguments\":{}}]}");
    return snprintf(out, capacity, "{\"function_calls\":[{\"name\":\"answer_no\",\"arguments\":{}}]}");
}
""")
        subprocess.run(["cc", "-shared", "-fPIC", "-o", library, source], check=True)
        weights = self.scratch / "selected"
        weights.write_bytes(b"TUNED")
        self.assets.installed_library.return_value = self.image(library, "library")
        self.assets.load_verified.return_value = self.image(weights, "weight-digest")
        selection = SimpleNamespace(selected=lambda: {"weights": str(weights), "sha256": "weight-digest"})
        with needle2.open_needle2(self.assets, selection, "root", local.needle_tools(),
                                  self.scratch, self.identity) as ask:
            self.assertEqual(local.needle_reply(ask("yes\ncontext")).text, "YES")
            self.assertIn("{broken native JSON", ask("malformed")["raw_text"])
            self.assertEqual(local.needle_reply(ask("no")).text, "NO")
            with self.assertRaisesRegex(RuntimeError, "could not complete"):
                ask("engine error")
        self.identity.assert_called_once_with("weight-digest")
        self.assets.load_verified.assert_called_once_with(weights, "weight-digest", 5)
        self.assets.from_installed.assert_not_called()

    def test_cached_generation_three_delivers_context_and_uses_both_verified_images(self):
        runtime = Mock()
        entry = Mock(return_value=nullcontext(runtime))
        runtime.complete.side_effect = [
            {"function_calls": [{"name": "answer_yes", "arguments": {}}]},
            {"function_calls": [{"name": "answer_no", "arguments": {}}]},
        ]
        lib = SimpleNamespace(sha256="library")
        weights = SimpleNamespace(sha256="weights")
        members = {"libneedle3-cached.so": lib, "weights.cact": weights}
        release = {"files": {name: {"sha256": image.sha256} for name, image in members.items()}}
        self.assets.installed_member.side_effect = lambda asset, name, root: nullcontext(members[name])
        modules = {"asset": self.assets, "libengine": SimpleNamespace(LibEngine=entry),
                   "compact_model": SimpleNamespace(require_cached_platform=Mock()),
                   "commands_needle3": SimpleNamespace(ASSET_ID="selected", selection=lambda: release)}
        (self.scratch / "asset.py").touch()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"PUZZLLM_KILIX_NEEDLE": str(self.scratch)}):
            with local.kilix_answerer("needle3", worker.INSTRUCTION, self.identity, self.scratch) as ask:
                text = 'An item with "quotes",\na backslash \\ and two lines.\nIs it correct?'
                self.assertEqual(ask(text).text, "YES")
                self.assertIn(text, runtime.complete.call_args.args[0])
                self.assertEqual(ask("Another question?").text, "NO")
            self.assertEqual(runtime.reset.call_count, 2)
            self.identity.assert_called_once_with("weights")
            entry.assert_called_once_with(lib, local.needle_tools(), weights,
                                          generation=3, backend="rust3", preparation_timeout=90)
            weights.sha256 = "unexpected"
            with self.assertRaisesRegex(ValueError, "differs from"):
                with local.kilix_answerer("needle3", worker.INSTRUCTION, self.identity, self.scratch):
                    self.fail("opened an unverified model")


if __name__ == "__main__":
    unittest.main()
