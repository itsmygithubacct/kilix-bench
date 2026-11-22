import unittest
from unittest.mock import patch

from kilix_bench.__main__ import main


class LocalModelSuites(unittest.TestCase):
    @patch("kilix_bench.local_models.subprocess.run")
    def test_quality_tasks_reach_upstream_unchanged(self, run):
        run.return_value.returncode = 3
        for task in ("boolq", "ifeval", "gsm8k"):
            with self.subTest(task=task):
                status = main([f"--suite={task}", "--backend", "hf", "--model-args",
                               "pretrained=model dir", "--limit", "2", "--output", "scores",
                               "--apply-chat-template"])
                self.assertEqual(status, 3)
                run.assert_called_with(
                    ["lm-eval", "run", "--tasks", task, "--model", "hf",
                     "--model_args", "pretrained=model dir", "--apply_chat_template",
                     "--limit", "2", "--output_path", "scores"], check=False, env=None)

    @patch("kilix_bench.local_models.subprocess.run")
    def test_speed_defaults(self, run):
        run.return_value.returncode = 0
        self.assertEqual(main(["--suite", "llama-bench", "--model-file", "model.gguf"]), 0)
        run.assert_called_once_with(
            ["llama-bench", "-m", "model.gguf", "-p", "512", "-n", "128",
             "-r", "5", "-o", "json"], check=False, env=None)
