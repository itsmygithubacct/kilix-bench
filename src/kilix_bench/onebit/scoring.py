"""OneBit scores: accuracy, balanced accuracy, MCC, format failures, bias.

A record is one answered item: its key (``answer``), the parsed reply
(``parsed``, 0, 1 or None for a format failure) and the item's band, domain,
form and polarity. A format failure always counts as a wrong answer: in
accuracy, in balanced accuracy, and in MCC (where it is scored as the
opposite of the key). Skipped records (``skipped`` set) are left out and
counted on their own.
"""

import math

CHANCE = 0.5
GRADE_PASS = 0.80
GRADE_BANDS = tuple(f"grade-{grade:02d}" for grade in range(1, 13))


def _scored(records):
    return [row for row in records if not row.get("skipped")]


def metrics(records):
    rows = _scored(records)
    total = len(rows)
    out = {
        "n": total,
        "skipped": len(records) - total,
        "chance": CHANCE,
    }
    if not total:
        out.update({
            "correct": 0, "accuracy": None, "balanced_accuracy": None,
            "accuracy_all_items": 0.0 if records else None,
            "mcc": None, "parsed": 0, "format_failures": 0,
            "format_failure_rate": None, "zero_share": None,
            "key_zero_share": None})
        return out
    parsed = [row for row in rows if row["parsed"] is not None]
    correct = sum(row["parsed"] == row["answer"] for row in rows)
    zeros = sum(row["parsed"] == 0 for row in parsed)
    # 0 is the positive class (yes / pass). A format failure predicts the
    # opposite of the key, so it is a miss in whichever class it belongs to.
    tp = fn = tn = fp = 0
    for row in rows:
        predicted = row["parsed"]
        if predicted is None:
            predicted = 1 - row["answer"]
        if row["answer"] == 0:
            if predicted == 0:
                tp += 1
            else:
                fn += 1
        elif predicted == 1:
            tn += 1
        else:
            fp += 1
    recalls = []
    if tp + fn:
        recalls.append(tp / (tp + fn))
    if tn + fp:
        recalls.append(tn / (tn + fp))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out.update({
        "correct": correct,
        "accuracy": correct / total,
        # Skipped items count as wrong here, so a model with a short input
        # limit is not flattered by the items it never saw.
        "accuracy_all_items": correct / len(records),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "parsed": len(parsed),
        "format_failures": total - len(parsed),
        "format_failure_rate": (total - len(parsed)) / total,
        "zero_share": zeros / len(parsed) if parsed else None,
        "key_zero_share": sum(row["answer"] == 0 for row in rows) / total,
    })
    return out


def grouped(records, key):
    groups = {}
    for row in records:
        groups.setdefault(key(row), []).append(row)
    return {name: metrics(rows) for name, rows in sorted(groups.items())}


def grade_level(records):
    """The highest grade band with accuracy >= 80%, climbing from grade 1.

    One failed band is passed over; two failed bands in a row end the climb,
    so a lucky pass above two consecutive misses does not count. A band with
    no scored items counts as a miss.
    """
    per_band = grouped(
        [row for row in records if row["band"] in GRADE_BANDS],
        lambda row: row["band"])
    level = 0
    misses = 0
    passes = {}
    stopped_at = None
    for grade, band in enumerate(GRADE_BANDS, 1):
        accuracy = (per_band.get(band) or {}).get("accuracy")
        passed = accuracy is not None and accuracy >= GRADE_PASS
        passes[band] = passed
        if stopped_at is not None:
            continue
        if passed:
            level = grade
            misses = 0
        else:
            misses += 1
            if misses == 2:
                stopped_at = grade
    return {
        "grade_level": level,
        "threshold": GRADE_PASS,
        "passes": passes,
        "stopped_at_grade": stopped_at,
    }


def summarize(records):
    return {
        "overall": metrics(records),
        "by_band": grouped(records, lambda row: row["band"]),
        "by_domain": grouped(records, lambda row: row["domain"]),
        "by_band_domain": grouped(
            records, lambda row: f"{row['band']}/{row['domain']}"),
        "by_form": grouped(records, lambda row: row["form"]),
        "by_polarity": grouped(records, lambda row: row["polarity"]),
        "grade": grade_level(records),
    }


def sign_test(wins_a, wins_b):
    """Exact two-sided binomial sign test on the discordant pairs."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _correct(row):
    return not row.get("skipped") and row["parsed"] == row["answer"]


def compare(records_a, records_b):
    """Pair two runs on item id and item text; count who wins each item."""
    by_id_b = {row["id"]: row for row in records_b}
    paired = []
    mismatched = 0
    for row in records_a:
        other = by_id_b.get(row["id"])
        if other is None:
            continue
        if row.get("item_sha256") != other.get("item_sha256"):
            mismatched += 1
            continue
        if row.get("skipped") or other.get("skipped"):
            continue
        paired.append((row, other))

    def tally(pairs):
        a_only = sum(_correct(a) and not _correct(b) for a, b in pairs)
        b_only = sum(_correct(b) and not _correct(a) for a, b in pairs)
        both = sum(_correct(a) and _correct(b) for a, b in pairs)
        n = len(pairs)
        return {
            "n": n,
            "both_correct": both,
            "a_only": a_only,
            "b_only": b_only,
            "neither": n - both - a_only - b_only,
            "accuracy_a": sum(_correct(a) for a, _b in pairs) / n if n else None,
            "accuracy_b": sum(_correct(b) for _a, b in pairs) / n if n else None,
            "p_sign": sign_test(a_only, b_only),
        }

    def by(key):
        groups = {}
        for pair in paired:
            groups.setdefault(key(pair[0]), []).append(pair)
        return {name: tally(rows) for name, rows in sorted(groups.items())}

    return {
        "overall": tally(paired),
        "by_band": by(lambda row: row["band"]),
        "by_domain": by(lambda row: row["domain"]),
        "by_form": by(lambda row: row["form"]),
        "unpaired_a": len(records_a) - len(paired) - mismatched,
        "text_mismatches": mismatched,
        "chance": CHANCE,
    }


def _pct(value):
    return "   -  " if value is None else f"{100 * value:5.1f}%"


def _mcc(value):
    return "   -  " if value is None else f"{value:+.3f}"


def table(groups, title):
    lines = [
        f"{title:<28} {'n':>5} {'skip':>5} {'acc':>7} {'acc-all':>7} "
        f"{'bal':>7} {'mcc':>7} {'fmt-fail':>8} {'zeros':>7}  "
        "(chance 50.0%)"]
    for name, row in groups.items():
        lines.append(
            f"{name:<28} {row['n']:>5} {row['skipped']:>5} "
            f"{_pct(row['accuracy']):>7} "
            f"{_pct(row.get('accuracy_all_items')):>7} "
            f"{_pct(row['balanced_accuracy']):>7} {_mcc(row['mcc']):>7} "
            f"{_pct(row['format_failure_rate']):>8} "
            f"{_pct(row['zero_share']):>7}")
    return lines


def render_summary(summary):
    lines = table({"overall": summary["overall"]}, "")
    lines.append("")
    lines += table(summary["by_band"], "band")
    lines.append("")
    lines += table(summary["by_form"], "form")
    lines.append("")
    lines += table(summary["by_polarity"], "polarity")
    lines.append("")
    lines += table(summary["by_domain"], "domain")
    grade = summary["grade"]
    lines.append("")
    lines.append(
        "grade level {level} (highest grade band at >= {pct:.0f}% accuracy; "
        "two misses in a row end the climb)".format(
            level=grade["grade_level"], pct=100 * grade["threshold"]))
    return lines


def render_compare(result, name_a="A", name_b="B"):
    lines = [
        f"paired comparison: A = {name_a}, B = {name_b}",
        f"{'group':<28} {'n':>5} {'acc A':>7} {'acc B':>7} {'A only':>7} "
        f"{'B only':>7} {'p (sign)':>9}  (chance 50.0%)",
    ]

    def row(name, value):
        return (f"{name:<28} {value['n']:>5} {_pct(value['accuracy_a']):>7} "
                f"{_pct(value['accuracy_b']):>7} {value['a_only']:>7} "
                f"{value['b_only']:>7} {value['p_sign']:>9.4f}")

    lines.append(row("overall", result["overall"]))
    for section in ("by_band", "by_form", "by_domain"):
        lines.append("")
        for name, value in result[section].items():
            lines.append(row(name, value))
    lines.append("")
    lines.append(
        f"unpaired items in A: {result['unpaired_a']}  "
        f"items whose text differs: {result['text_mismatches']}")
    return lines
