"""OneBit scoring on hand-computed fixtures. Expected values are literal."""

import unittest

from kilix_bench.onebit import scoring


def record(answer, parsed, band="grade-01", domain="maths", form="question",
           polarity="plain", ident=None, sha=None):
    return {
        "id": ident or f"x-{answer}-{parsed}",
        "band": band, "domain": domain, "form": form, "polarity": polarity,
        "split": "test", "answer": answer, "parsed": parsed,
        "item_sha256": sha or "s",
    }


def fixture():
    # key 0: three right, one wrong.  key 1: two right, one wrong, one
    # format failure.
    return ([record(0, 0)] * 3 + [record(0, 1)] + [record(1, 1)] * 2
            + [record(1, 0)] + [record(1, None)])


class MetricsTest(unittest.TestCase):
    def test_hand_computed_fixture(self):
        out = scoring.metrics(fixture())
        self.assertEqual(out["n"], 8)
        self.assertEqual(out["correct"], 5)
        self.assertAlmostEqual(out["accuracy"], 0.625)
        # recall on key 0 = 3/4, on key 1 = 2/4 (the failure is a miss)
        self.assertAlmostEqual(out["balanced_accuracy"], 0.625)
        # tp=3 fn=1 tn=2 fp=2: (6 - 2) / sqrt(5 * 4 * 4 * 3)
        self.assertAlmostEqual(out["mcc"], 0.2581988897, places=9)
        self.assertEqual(out["confusion"], {"tp": 3, "fn": 1, "tn": 2, "fp": 2})
        self.assertEqual(out["format_failures"], 1)
        self.assertAlmostEqual(out["format_failure_rate"], 0.125)
        # 4 zeros among 7 parsed replies
        self.assertAlmostEqual(out["zero_share"], 0.5714285714, places=9)
        self.assertAlmostEqual(out["key_zero_share"], 0.5)
        self.assertEqual(out["chance"], 0.5)

    def test_always_zero_scores_chance(self):
        rows = [record(0, 0)] * 5 + [record(1, 0)] * 5
        out = scoring.metrics(rows)
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertAlmostEqual(out["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(out["mcc"], 0.0)
        self.assertAlmostEqual(out["zero_share"], 1.0)

    def test_all_format_failures(self):
        rows = [record(0, None), record(1, None)]
        out = scoring.metrics(rows)
        self.assertEqual(out["accuracy"], 0.0)
        self.assertEqual(out["balanced_accuracy"], 0.0)
        self.assertAlmostEqual(out["mcc"], -1.0)
        self.assertEqual(out["format_failure_rate"], 1.0)
        self.assertIsNone(out["zero_share"])

    def test_perfect(self):
        out = scoring.metrics([record(0, 0), record(1, 1)] * 3)
        self.assertEqual(out["accuracy"], 1.0)
        self.assertAlmostEqual(out["mcc"], 1.0)

    def test_skipped_rows_are_counted_not_scored(self):
        rows = [record(0, 0), dict(record(1, None), skipped="too long")]
        out = scoring.metrics(rows)
        self.assertEqual(out["n"], 1)
        self.assertEqual(out["skipped"], 1)
        self.assertEqual(out["accuracy"], 1.0)

    def test_empty(self):
        out = scoring.metrics([])
        self.assertEqual(out["n"], 0)
        self.assertIsNone(out["accuracy"])

    def test_grouping(self):
        rows = [record(0, 0, form="verdict"), record(1, 0, form="question")]
        groups = scoring.summarize(rows)["by_form"]
        self.assertEqual(groups["verdict"]["accuracy"], 1.0)
        self.assertEqual(groups["question"]["accuracy"], 0.0)


def band(grade, right, of=5):
    name = f"grade-{grade:02d}"
    rows = []
    for index in range(of):
        answer = index % 2
        parsed = answer if index < right else 1 - answer
        rows.append(record(answer, parsed, band=name))
    return rows


class GradeLevelTest(unittest.TestCase):
    def grades(self, pattern):
        rows = []
        for grade, passed in enumerate(pattern, 1):
            rows += band(grade, 5 if passed else 3)
        return scoring.grade_level(rows)

    def test_one_miss_is_passed_over(self):
        # P P P F P F F P P P P P: the climb ends at grade 7 with grade 5.
        out = self.grades([1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(out["grade_level"], 5)
        self.assertEqual(out["stopped_at_grade"], 7)
        self.assertTrue(out["passes"]["grade-08"])

    def test_two_misses_at_the_start(self):
        out = self.grades([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(out["grade_level"], 0)
        self.assertEqual(out["stopped_at_grade"], 2)

    def test_all_pass(self):
        out = self.grades([1] * 12)
        self.assertEqual(out["grade_level"], 12)
        self.assertIsNone(out["stopped_at_grade"])

    def test_eighty_percent_is_a_pass(self):
        rows = band(1, 4) + band(2, 4)
        self.assertEqual(scoring.grade_level(rows)["grade_level"], 2)

    def test_missing_band_counts_as_a_miss(self):
        rows = band(1, 5) + band(4, 5)
        out = scoring.grade_level(rows)
        self.assertEqual(out["grade_level"], 1)
        self.assertEqual(out["stopped_at_grade"], 3)


class CompareTest(unittest.TestCase):
    def test_sign_test_values(self):
        self.assertAlmostEqual(scoring.sign_test(0, 6), 0.03125)
        self.assertAlmostEqual(scoring.sign_test(1, 5), 0.21875)
        self.assertAlmostEqual(scoring.sign_test(8, 2), 0.109375)
        self.assertEqual(scoring.sign_test(3, 3), 1.0)
        self.assertEqual(scoring.sign_test(0, 0), 1.0)

    def test_pairs_on_id_and_text(self):
        a = [record(0, 0, ident="i1"), record(1, 1, ident="i2"),
             record(0, 1, ident="i3"), record(1, 1, ident="i4", sha="old"),
             record(0, 0, ident="i5")]
        b = [record(0, 1, ident="i1"), record(1, 1, ident="i2"),
             record(0, 0, ident="i3"), record(1, 0, ident="i4", sha="new")]
        out = scoring.compare(a, b)
        overall = out["overall"]
        self.assertEqual(overall["n"], 3)
        self.assertEqual(overall["a_only"], 1)
        self.assertEqual(overall["b_only"], 1)
        self.assertEqual(overall["both_correct"], 1)
        self.assertEqual(overall["neither"], 0)
        self.assertEqual(out["text_mismatches"], 1)
        self.assertEqual(out["unpaired_a"], 1)
        self.assertEqual(overall["p_sign"], 1.0)

    def test_format_failure_loses_the_pair(self):
        a = [record(0, None, ident=f"i{n}") for n in range(6)]
        b = [record(0, 0, ident=f"i{n}") for n in range(6)]
        out = scoring.compare(a, b)["overall"]
        self.assertEqual(out["b_only"], 6)
        self.assertAlmostEqual(out["p_sign"], 0.03125)


if __name__ == "__main__":
    unittest.main()
