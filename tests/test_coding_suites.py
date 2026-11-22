import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from kilix_bench.__main__ import main
from kilix_bench.coding_tasks import tasks
from kilix_bench import terminal_bench


class CodingSuites(unittest.TestCase):
    @patch("kilix_bench.local_models.subprocess.run")
    def test_coding_tasks_and_limits(self, run):
        run.return_value.returncode = 0
        for suite, names in (("cruxeval", "kilix_cruxeval_input,kilix_cruxeval_output"),
                             ("humaneval-plus", "kilix_humaneval_plus")):
            self.assertEqual(main(["--suite", suite, "--backend", "gguf", "--model-args",
                                   "base_url=http://localhost:8000", "--output", "scores"]), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--tasks") + 1], names)
            self.assertNotIn("--limit", command)
            self.assertTrue(Path(command[command.index("--include_path") + 1]).is_dir())
            self.assertIn("--log_samples", command)
            if suite == "humaneval-plus":
                self.assertEqual(command[command.index("--model") + 1], "kilix-gguf-code")
                self.assertIn("kilix_gguf_code", command)

    @unittest.skipUnless(shutil.which("bwrap"), "requires Linux bubblewrap")
    def test_crux_checks_alternate_inputs_and_literal_outputs(self):
        doc = {"code": "def f(n):\n    return str(abs(n))", "input": "3", "output": "'3'"}
        self.assertEqual(tasks.score_input(doc, ["-3"]), {"pass@1": 1})
        self.assertEqual(tasks.score_input(doc, ["4"]), {"pass@1": 0})
        self.assertEqual(tasks.score_output(doc, ['"3"']), {"pass@1": 1})
        self.assertEqual(tasks.score_output(doc, ["3"]), {"pass@1": 0})

    @unittest.skipUnless(importlib.util.find_spec("lm_eval"), "requires the optional lm-eval harness")
    def test_gguf_code_retains_python_indentation(self):
        from kilix_bench.coding_tasks.kilix_gguf_code import GGUFCode

        model = GGUFCode(base_url="http://localhost:8080", parallel=1)
        body = "\n    if values:\n        return values[0]\n    return None\n"
        with patch.object(model, "gguf_completion", return_value={"choices": [{"text": body}]}):
            self.assertEqual(model._generate_one(("def first(values):\n", {"until": ["\ndef"]})), body)

    def test_terminal_rewards_must_cover_every_selected_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "result.json").write_text(json.dumps({"finished_at": "done"}))
            trial = job / "trial-1"
            trial.mkdir()
            row = {"task_name": "terminal-bench/fix-git", "finished_at": "done", "reward_valid": True,
                   "verifier_result": {"rewards": {"reward": 0}}}
            (trial / "result.json").write_text(json.dumps(row))
            self.assertEqual(terminal_bench.collect(job, ["fix-git"]), {"fix-git": 0})
            with self.assertRaises(ValueError):
                terminal_bench.collect(job, ["fix-git", "log-summary-date-ranges"])
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                terminal_bench.collect(job, ["log-summary-date-ranges"])
            row["reward_valid"] = False
            row["exception_info"] = {"exception_type": "DockerError"}
            (trial / "result.json").write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "DockerError"):
                terminal_bench.collect(job, ["fix-git"])

    @patch("kilix_bench.terminal_bench.run_job")
    def test_terminal_cli_selects_real_tasks(self, run):
        self.assertEqual(main(["--suite", "terminal-bench", "--endpoint", "http://localhost:8080",
                               "--limit", "2"]), 0)
        self.assertEqual(run.call_args.args[3], list(terminal_bench.TASKS[:2]))

    def test_terminal_rejects_non_profile_tasks(self):
        with self.assertRaises(ValueError):
            terminal_bench.run_job("model", None, "jobs", ["unknown-task"])

    @patch("kilix_bench.terminal_bench.collect", return_value={"fix-git": 1})
    @patch("kilix_bench.terminal_bench.shutil.which", return_value="/bin/true")
    def test_terminal_passes_exact_served_model_and_token_budget(self, _launcher, _scores):
        with tempfile.TemporaryDirectory() as tmp:
            run = terminal_bench.run_job("openai/local-model", "http://localhost:8000/v1", tmp,
                                         ["fix-git"], lambda msg: None, context_length=8192, max_tokens=256)
            agent, = json.loads((run / "job.json").read_text())["agents"]
            self.assertEqual(agent["model_name"], "openai/openai/local-model")
            self.assertEqual(agent["kwargs"]["llm_kwargs"]["max_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
