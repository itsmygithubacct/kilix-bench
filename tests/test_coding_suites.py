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

    @unittest.skipUnless(shutil.which("bwrap"), "requires Linux bubblewrap")
    def test_crux_checks_alternate_inputs_and_literal_outputs(self):
        doc = {"code": "def f(n):\n    return str(abs(n))", "input": "3", "output": "'3'"}
        self.assertEqual(tasks.score_input(doc, ["-3"]), {"pass@1": 1})
        self.assertEqual(tasks.score_input(doc, ["4"]), {"pass@1": 0})
        self.assertEqual(tasks.score_output(doc, ['"3"']), {"pass@1": 1})
        self.assertEqual(tasks.score_output(doc, ["3"]), {"pass@1": 0})

    def test_terminal_rewards_must_cover_every_selected_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "result.json").write_text(json.dumps({"finished_at": "done"}))
            trial = job / "trial-1"
            trial.mkdir()
            row = {"task_name": "fix-git", "reward_valid": True, "verifier_result": {"rewards": {"reward": 0}}}
            (trial / "result.json").write_text(json.dumps(row))
            self.assertEqual(terminal_bench.collect(job, ["fix-git"]), {"fix-git": 0})
            with self.assertRaises(ValueError):
                terminal_bench.collect(job, ["fix-git", "log-summary-date-ranges"])
            row["reward_valid"] = False
            (trial / "result.json").write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                terminal_bench.collect(job, ["fix-git"])

    @patch("kilix_bench.terminal_bench.run_job")
    def test_terminal_cli_selects_real_tasks(self, run):
        self.assertEqual(main(["--suite", "terminal-bench", "--endpoint", "http://localhost:8080",
                               "--limit", "2"]), 0)
        self.assertEqual(run.call_args.args[3], list(terminal_bench.TASKS[:2]))

    def test_terminal_rejects_non_profile_tasks(self):
        with self.assertRaises(ValueError):
            terminal_bench.run_job("model", None, "jobs", ["unknown-task"])


if __name__ == "__main__":
    unittest.main()
