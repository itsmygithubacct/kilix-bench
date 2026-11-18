"""The shipped OneBit item files: schema, balance, split, hygiene, canary.

The last class plants a defect in a copy of the files (a flipped key, an
unbalanced domain, a lost canary) and checks that the same checks catch it.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from kilix_bench.onebit import items


def load_all():
    return items.load(split="all")


class ShippedItemsTest(unittest.TestCase):
    def test_every_file_passes_the_full_check(self):
        self.assertEqual(items.check_files(items.item_files(), full_set=True),
                         [])

    def test_one_file_per_band(self):
        names = sorted(path.name for path in items.item_files())
        expected = sorted(f"{stem}.jsonl" for _d, _m, stem in
                          items.BANDS.values())
        self.assertEqual(names, expected)

    def test_every_row_matches_the_schema(self):
        rows = load_all()
        self.assertGreaterEqual(len(rows), 1470)
        for row in rows:
            self.assertEqual(items.validate_item(row), [], row.get("id"))

    def test_minimum_counts(self):
        counts = {}
        for row in load_all():
            counts[row["band"]] = counts.get(row["band"], 0) + 1
        for band, (_domains, minimum, _stem) in items.BANDS.items():
            self.assertGreaterEqual(counts.get(band, 0), minimum, band)
        self.assertEqual(sum(1 for band in counts if band.startswith(
            "grade-")), 12)

    def test_answers_balanced_in_every_band_and_domain(self):
        self.assertEqual(items.balance_problems(load_all()), [])

    def test_split_is_a_stratified_quarter(self):
        rows = load_all()
        self.assertEqual(items.split_problems(rows), [])
        dev = sum(row["split"] == "dev" for row in rows)
        self.assertAlmostEqual(dev / len(rows), 0.25, delta=0.02)

    def test_pairs_share_a_split(self):
        by_id = {row["id"]: row for row in load_all()}
        pairs = [row for row in by_id.values() if "pair" in row]
        self.assertGreaterEqual(len(pairs), 30)
        for row in pairs:
            self.assertEqual(row["split"], by_id[row["pair"]]["split"])

    def test_context_length_does_not_predict_the_key(self):
        # Per computer-use domain, mean context length of key-0 and key-1
        # items differs by under 15%.
        self.assertEqual(items.length_cue_problems(load_all()), [])

    def test_length_cue_check_fires(self):
        rows = [{"band": "computer-use", "domain": "logs", "answer": 0,
                 "context": "x" * 100},
                {"band": "computer-use", "domain": "logs", "answer": 1,
                 "context": "x" * 114},
                {"band": "computer-use", "domain": "safety", "answer": 0,
                 "context": "x" * 100},
                {"band": "computer-use", "domain": "safety", "answer": 1,
                 "context": "x" * 115}]
        (problem,) = items.length_cue_problems(rows)
        self.assertIn("computer-use/safety", problem)
        self.assertIn("15% apart", problem)

    def test_literal_question_items_are_tagged(self):
        rows = [row for row in load_all()
                if row["band"] == "traps" and row["domain"] ==
                "literal-question"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["polarity"], "literal-question", row["id"])

    def test_hygiene(self):
        for path in items.item_files():
            text = path.read_text(encoding="utf-8")
            self.assertEqual(items.hygiene_problems(text, path.name), [])

    def test_canary_in_every_file(self):
        marker = items.canary()
        self.assertIn("do not train", marker)
        for path in items.item_files():
            head, _rows = items.read_file(path)
            self.assertEqual(head["canary"], marker, path.name)
            self.assertIn(marker, path.read_text(encoding="utf-8"))

    def test_manifest_pins_the_keys(self):
        self.assertEqual(items.manifest_problems(), [])

    def test_hygiene_patterns_fire(self):
        samples = ("/" + "home/someone/x", "ask " + "ne" + "on", "gh" + "p_x",
                   "a" + "@" + "b.org", "Clau" + "de Monet")
        for sample in samples:
            self.assertTrue(items.hygiene_problems(sample), sample)
        self.assertEqual(items.hygiene_problems("/srv/app/x, claudex"), [])


class PlantedDefectTest(unittest.TestCase):
    """A copy of the files with one defect must fail the checks."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for path in items.ITEMS_DIR.iterdir():
            shutil.copy(path, self.root / path.name)
        self.file = self.root / "computer-use.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def rewrite(self, change):
        lines = self.file.read_text(encoding="utf-8").splitlines()
        self.file.write_text("\n".join(change(lines)) + "\n", encoding="utf-8")

    def problems(self):
        return items.check_files(items.item_files(self.root), full_set=True)

    def test_untouched_copy_is_clean(self):
        self.assertEqual(self.problems(), [])
        self.assertEqual(items.manifest_problems(self.root), [])

    def test_flipped_key_is_caught(self):
        def flip(lines):
            row = json.loads(lines[1])
            row["answer"] = 1 - row["answer"]
            lines[1] = json.dumps(row)
            return lines
        self.rewrite(flip)
        self.assertTrue(any("answers 0" in problem
                            for problem in self.problems()))
        self.assertTrue(any("keys_sha256" in problem
                            for problem in items.manifest_problems(self.root)))

    def test_unbalanced_domain_is_caught(self):
        self.rewrite(lambda lines: lines[:1] + lines[2:])
        problems = self.problems()
        self.assertTrue(any("answers 0" in problem for problem in problems))

    def test_dropped_canary_is_caught(self):
        def strip(lines):
            head = json.loads(lines[0])
            del head["canary"]
            lines[0] = json.dumps(head)
            return lines
        self.rewrite(strip)
        self.assertTrue(any("canary" in problem for problem in self.problems()))

    def test_leaked_path_is_caught(self):
        def leak(lines):
            row = json.loads(lines[1])
            row["prompt"] += " See /" + "home/someone/notes."
            lines[1] = json.dumps(row)
            return lines
        self.rewrite(leak)
        self.assertTrue(any("home-path" in problem
                            for problem in self.problems()))


if __name__ == "__main__":
    unittest.main()
