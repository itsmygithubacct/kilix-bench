import hashlib
import io
import json
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kilix_bench.yesno import records, suites, worker


class SuiteTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [
            dict(idx=5, passage="The signal is green.", question="is the signal green", label=True),
            dict(idx=8, passage="The signal is green.", question="is the signal red?", label=False),
            dict(idx=2, passage="No signal is shown.", question="is a signal shown", label=False),
        ]
        self.data = b"".join(json.dumps(row).encode() + b"\n" for row in self.rows)
        for setting in (patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root/"cache")}),
                        patch.object(suites, "BOOLQ_SHA256", hashlib.sha256(self.data).hexdigest()),
                        patch.object(suites, "BOOLQ_BYTES", len(self.data))):
            setting.start()
            self.addCleanup(setting.stop)

    def archive(self, data):
        stream = io.BytesIO()
        with ZipFile(stream, "w") as z:
            z.writestr(suites.BOOLQ_MEMBER, data)
        stream.seek(0)
        return stream

    def test_download_is_validated_and_the_cache_works_offline(self):
        with patch.object(suites, "urlopen", return_value=self.archive(self.data)) as fetch:
            first = suites.boolq_items(Mock())
            self.assertEqual(suites.boolq_items(Mock()), first)
            fetch.assert_called_once_with(suites.BOOLQ_URL, timeout=60)
        self.assertEqual([(item["id"], item["answer"]) for item in first],
                         [("boolq-validation-5", 0), ("boolq-validation-8", 1), ("boolq-validation-2", 1)])
        cache, = (self.root/"cache").rglob("*.jsonl")
        cache.write_bytes(b"incomplete")
        with patch.object(suites, "urlopen", return_value=self.archive(self.data.replace(b"green", b"black"))):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                suites.boolq_items(Mock())
        self.assertEqual(cache.read_bytes(), b"incomplete")

    def test_boolq_grades_strict_replies_and_keeps_legacy_rankings(self):
        k12 = suites.K12.folder(self.root)
        k12.mkdir(parents=True)
        old = k12/"legacy.jsonl"
        prior = [dict(event="begin", provider="kilix", model="same", total=1, split="dev"),
                 dict(event="answer", n=1, verdict="correct"), dict(event="end")]
        old.write_text("".join(json.dumps(row)+"\n" for row in prior))
        legacy_board = records.save_rankings(k12)
        # A restored version 3 snapshot may be the only copy of a historical score.
        for key in ("suite", "protocol"):
            del legacy_board[key]
        old.unlink()
        (k12/"rankings.json").write_text(json.dumps(legacy_board))
        legacy_bytes = (k12/"rankings.json").read_bytes()

        folder = suites.BOOLQ.folder(self.root)
        folder.mkdir(parents=True)
        path = folder/"run.jsonl"
        replies = iter(("YES", "NO", "NO, definitely"))
        prompts = []
        def answer(question, *, prompt, context):
            prompts.append((question, prompt, context))
            return next(replies)
        with patch.object(suites, "urlopen", return_value=self.archive(self.data)), \
                patch.object(worker, "answerer", return_value=nullcontext(answer)), \
                path.open("w", encoding="utf-8") as journal:
            worker.perform("kilix", "same", journal, path, suite=suites.BOOLQ)
        events = list(map(json.loads, path.read_text().splitlines()))
        self.assertEqual(events[-1]["event"], "end")
        self.assertEqual(prompts, [(f"<information>\n{row['passage']}\n</information>\n\n{question}",
                                   question, row["passage"])
                                  for row in self.rows for question in (row["question"].rstrip("?")+"?",)])
        answers = [e for e in events if e["event"] == "answer"]
        self.assertEqual([(e["expected"], e["verdict"]) for e in answers],
                         [("YES", "correct"), ("NO", "correct"), ("NO", "invalid")])
        self.assertEqual(answers[-1]["raw_reply"], "NO, definitely")
        self.assertEqual(answers[1]["context"], self.rows[1]["passage"])
        board = json.loads((folder/"rankings.json").read_text())
        self.assertEqual({k: board[k] for k in suites.BOOLQ.metadata()}, suites.BOOLQ.metadata())
        self.assertEqual((board["total_runs"], board["models"][0]["avg_correct"], board["models"][0]["invalid_answers"]),
                         (1, 2, 1))
        self.assertEqual((k12/"rankings.json").read_bytes(), legacy_bytes)
        self.assertEqual(records.compare_files([folder/"rankings.json"])["suite"], "boolq")
        with self.assertRaisesRegex(ValueError, "same suite"):
            records.compare_files([k12/"rankings.json", folder/"rankings.json"])
        # A misplaced journal cannot enter the other suite's standings.
        (k12/path.name).write_bytes(path.read_bytes())
        historical = records.save_rankings(k12)
        self.assertEqual((historical["total_runs"], historical["models"][0]["avg_correct"]), (1, 1))
        self.assertFalse(historical["models"][0]["history"][0]["journal_available"])

    def test_future_protocol_or_split_cannot_replace_current_standings(self):
        folder = suites.BOOLQ.folder(self.root)
        empty = records.save_rankings(folder, suites.BOOLQ)
        target = folder/"rankings.json"
        for field in ("protocol", "split", "suite"):
            other = json.dumps(dict(empty, **{field:"future"})).encode()
            target.write_bytes(other)
            with self.assertRaises(ValueError):
                records.save_rankings(folder, suites.BOOLQ)
            self.assertEqual(target.read_bytes(), other)


if __name__ == "__main__":
    unittest.main()
