"""OneBit items: the schema, the loader and the checks every item file must pass.

An item file is JSON Lines. Its first line is a header that carries the
canary string, and every other line is one item. ``python3 -m
kilix_bench.onebit.items check FILE...`` runs the schema, balance and hygiene
checks over files; without arguments it checks the shipped set.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ITEMS_DIR = Path(__file__).resolve().parent / "items"
SCHEMA = "kilix-bench.onebit.item/v1"

GRADE_DOMAINS = (
    "maths", "reading", "science", "social-studies", "language", "logic")

# band -> (domains, minimum items in the band, file stem)
BANDS = {}
for _grade in range(1, 13):
    BANDS[f"grade-{_grade:02d}"] = (GRADE_DOMAINS, 60, f"grade-{_grade:02d}")
BANDS["college"] = ((
    "maths", "physics", "chemistry", "biology", "cs-theory", "economics",
    "philosophy-logic", "statistics"), 150, "college")
BANDS["computer-use"] = ((
    "shell-output", "exit-codes", "logs", "test-reports", "diffs",
    "file-listings", "permissions", "git-status", "config-validity",
    "ui-state", "destructive", "safety"), 300, "computer-use")
BANDS["reading-shown"] = ((
    "science", "geography-history", "numbers-tables", "technical-docs",
    "everyday"), 150, "reading-shown")
BANDS["traps"] = ((
    "negation", "double-negative", "literal-question", "paired",
    "injection", "false-premise"), 150, "traps")

GRADE_BANDS = tuple(f"grade-{grade:02d}" for grade in range(1, 13))
FORMS = ("question", "verdict")
POLARITIES = ("plain", "literal-question", "negated")
SPLITS = ("dev", "test")
CONTEXT_KINDS = (
    "text", "log", "output", "diff", "table", "listing", "json", "config",
    "ui", "code")
REQUIRED = (
    "id", "band", "domain", "form", "prompt", "answer", "polarity",
    "rationale", "split", "source")
OPTIONAL = ("context", "context_kind", "pair")
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# The fleet hygiene arms (kept in step with the publishing checklist), plus
# rules of this item set: no /home paths at all, no e-mail addresses. The
# words are assembled from pieces so this file does not itself match the
# repository scan that uses the same patterns.
def _words(*parts):
    return "".join(parts)


HYGIENE = (
    ("attribution", re.compile(
        _words("cla", "ude($|[^x])|anthr", "opic|co-auth", "ored"), re.I)),
    ("host-or-name", re.compile(_words(
        r"\bp5", r"0\b|\bne", r"on\b|\bop", r"ib\b|mich", "ael|nev",
        "ille|@gm", "ail"), re.I)),
    ("secret", re.compile(_words(
        "gh", "p_|gh", "o_|oauth", "_token|BEGIN.*PRIV", "ATE KEY"), re.I)),
    ("home-path", re.compile(r"/home/")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")),
)


def canary():
    return (ITEMS_DIR / "CANARY").read_text(encoding="utf-8").strip()


def header(band):
    return {"schema": SCHEMA, "band": band, "canary": canary()}


def item_files(directory=None):
    directory = Path(directory) if directory else ITEMS_DIR
    return sorted(directory.glob("*.jsonl"))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_file(path):
    """(header, items) of one file. Raises ValueError on malformed JSON."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not rows or "canary" not in rows[0]:
        return None, rows
    return rows[0], rows[1:]


def load(directory=None, split=None, bands=None):
    """Every item from the shipped files, optionally one split or some bands."""
    items = []
    for path in item_files(directory):
        _header, rows = read_file(path)
        for row in rows:
            if split and split != "all" and row.get("split") != split:
                continue
            if bands and row.get("band") not in bands:
                continue
            items.append(row)
    return items


def load_rows(paths, draft=False):
    rows = []
    for path in paths:
        head, body = read_file(path)
        if draft and head is not None:
            body = [head] + body
        rows += body
    return rows


def set_digest(directory=None):
    """sha256 of every item file, and one digest over all of them."""
    files = {path.name: file_sha256(path) for path in item_files(directory)}
    combined = hashlib.sha256(
        "".join(f"{name}\0{digest}\n" for name, digest in sorted(
            files.items())).encode()).hexdigest()
    return files, combined


def validate_item(row):
    """Schema problems of one row, as a list of strings."""
    problems = []
    ident = row.get("id", "?")
    for key in REQUIRED:
        if key not in row:
            problems.append(f"{ident}: missing {key}")
    for key in row:
        if key not in REQUIRED and key not in OPTIONAL:
            problems.append(f"{ident}: unknown field {key}")
    if problems:
        return problems
    if not isinstance(row["id"], str) or not ID_PATTERN.match(row["id"]):
        problems.append(f"{ident}: id must be lower-case words joined by -")
    band = row["band"]
    if band not in BANDS:
        problems.append(f"{ident}: unknown band {band!r}")
    elif row["domain"] not in BANDS[band][0]:
        problems.append(f"{ident}: domain {row['domain']!r} not in {band}")
    if row["form"] not in FORMS:
        problems.append(f"{ident}: form must be question or verdict")
    if row["answer"] not in (0, 1) or isinstance(row["answer"], bool):
        problems.append(f"{ident}: answer must be the integer 0 or 1")
    if row["polarity"] not in POLARITIES:
        problems.append(f"{ident}: unknown polarity {row['polarity']!r}")
    if row["split"] not in SPLITS:
        problems.append(f"{ident}: split must be dev or test")
    for key in ("prompt", "rationale", "source"):
        value = row[key]
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{ident}: {key} must be a non-empty string")
    if isinstance(row["rationale"], str) and "\n" in row["rationale"]:
        problems.append(f"{ident}: rationale must be one line")
    if "context" in row:
        if not isinstance(row["context"], str) or not row["context"].strip():
            problems.append(f"{ident}: context, when present, must be text")
    if "context_kind" in row:
        if "context" not in row:
            problems.append(f"{ident}: context_kind without context")
        if row["context_kind"] not in CONTEXT_KINDS:
            problems.append(f"{ident}: unknown context_kind")
    needs_context = (
        (band in GRADE_BANDS and row["domain"] == "reading")
        or band == "reading-shown"
        or (band == "traps" and row["domain"] == "injection"))
    if needs_context and "context" not in row:
        problems.append(f"{ident}: this domain needs a context")
    if "pair" in row and not isinstance(row["pair"], str):
        problems.append(f"{ident}: pair must be an item id")
    if band == "traps" and row["domain"] == "paired" and "pair" not in row:
        problems.append(f"{ident}: paired items name their pair")
    return problems


FIELD_ORDER = (
    "id", "band", "domain", "form", "context_kind", "context", "prompt",
    "answer", "polarity", "pair", "rationale", "split", "source")


def canonical(row):
    """The row with its fields in one fixed order."""
    return {key: row[key] for key in FIELD_ORDER if key in row}


def _rank(ident, seed):
    return hashlib.sha256(f"{seed}:{ident}".encode()).hexdigest()


def assign_splits(items, share=0.25, seed="onebit-split-v1"):
    """Mark about a quarter dev, stratified by band x domain x answer.

    The two members of a pair stay in one split, so a pair is placed as one
    unit, stratified by its pair of answers. Within a stratum, units are
    ordered by a hash of their id and the first ones go to dev: about
    share * n of them, with the rounding carried across strata.
    """
    by_id = {row["id"]: row for row in items}
    units = []
    placed = set()
    for row in items:
        if row["id"] in placed:
            continue
        members = [row]
        if row.get("pair") in by_id:
            members.append(by_id[row["pair"]])
        placed.update(member["id"] for member in members)
        units.append(members)
    strata = {}
    for members in units:
        key = (members[0]["band"], members[0]["domain"],
               tuple(sorted(member["answer"] for member in members)))
        strata.setdefault(key, []).append(members)
    # Rounding is carried from stratum to stratum (in a fixed order), so a
    # stratum of 6 gets 1 or 2 dev units and the whole set stays at share.
    total = assigned = 0
    for _key, group in sorted(strata.items()):
        group.sort(key=lambda members: _rank(
            min(member["id"] for member in members), seed))
        total += len(group)
        dev = int(share * total + 0.5) - assigned
        assigned += dev
        for index, members in enumerate(group):
            for member in members:
                member["split"] = "dev" if index < dev else "test"
    return items


def keys_sha256(rows):
    """Digest of every (id, answer) in a file: changes when any key flips."""
    blob = "".join(f"{row['id']}={row['answer']}\n" for row in sorted(
        rows, key=lambda row: row["id"]))
    return hashlib.sha256(blob.encode()).hexdigest()


def manifest_problems(directory=None):
    """Item files agree with MANIFEST.json: counts, keys and splits."""
    directory = Path(directory) if directory else ITEMS_DIR
    path = directory / "MANIFEST.json"
    if not path.is_file():
        return ["MANIFEST.json is missing"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    files = {item.name: item for item in item_files(directory)}
    if sorted(files) != sorted(manifest["files"]):
        problems.append("item files differ from the manifest's list")
    for name, expected in manifest["files"].items():
        if name not in files:
            continue
        _head, rows = read_file(files[name])
        actual = {
            "items": len(rows),
            "dev": sum(row.get("split") == "dev" for row in rows),
            "keys_sha256": keys_sha256(rows),
        }
        for key, value in actual.items():
            if expected.get(key) != value:
                problems.append(f"{name}: {key} is {value}, manifest says "
                                f"{expected.get(key)}")
    return problems


def write_manifest(directory=None):
    directory = Path(directory) if directory else ITEMS_DIR
    files = {}
    for path in item_files(directory):
        _head, rows = read_file(path)
        files[path.name] = {
            "items": len(rows),
            "dev": sum(row["split"] == "dev" for row in rows),
            "keys_sha256": keys_sha256(rows),
        }
    body = {"schema": SCHEMA, "canary": canary(), "files": files}
    (directory / "MANIFEST.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_band(path, band, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header(band), ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(canonical(row), ensure_ascii=False) + "\n")


def hygiene_problems(text, where=""):
    problems = []
    for name, pattern in HYGIENE:
        for match in pattern.finditer(text):
            problems.append(f"{where}: {name} matched {match.group(0)!r}")
    return problems


def balance_problems(items):
    """Every band x domain has as many 0 answers as 1 answers."""
    counts = {}
    for row in items:
        key = (row["band"], row["domain"])
        counts.setdefault(key, [0, 0])[row["answer"]] += 1
    return [
        f"{band}/{domain}: {zeros} answers 0, {ones} answers 1"
        for (band, domain), (zeros, ones) in sorted(counts.items())
        if zeros != ones]


LENGTH_CUE_BANDS = ("computer-use",)
LENGTH_CUE_TOLERANCE = 0.15


def length_cue_problems(items, bands=LENGTH_CUE_BANDS,
                        tolerance=LENGTH_CUE_TOLERANCE):
    """Shown-information length must not predict the key.

    In each domain of the given bands, the mean context length (characters)
    of key-0 items and of key-1 items may differ by less than ``tolerance``
    of the smaller mean.
    """
    lengths = {}
    for row in items:
        if row["band"] in bands:
            lengths.setdefault((row["band"], row["domain"]), ([], []))[
                row["answer"]].append(len(row.get("context") or ""))
    problems = []
    for (band, domain), (zeros, ones) in sorted(lengths.items()):
        if not zeros or not ones:
            continue
        mean_0, mean_1 = sum(zeros) / len(zeros), sum(ones) / len(ones)
        smaller = min(mean_0, mean_1)
        gap = abs(mean_0 - mean_1) / smaller if smaller else (
            0.0 if mean_0 == mean_1 else float("inf"))
        if gap >= tolerance:
            problems.append(
                f"{band}/{domain}: mean context {mean_0:.0f} chars for key 0, "
                f"{mean_1:.0f} for key 1 ({100 * gap:.0f}% apart, limit "
                f"{100 * tolerance:.0f}%)")
    return problems


def coverage_problems(items):
    """Every band reaches its minimum and has every one of its domains."""
    problems = []
    for band, (domains, minimum, _stem) in BANDS.items():
        rows = [row for row in items if row["band"] == band]
        if len(rows) < minimum:
            problems.append(f"{band}: {len(rows)} items, needs {minimum}")
        present = {row["domain"] for row in rows}
        for domain in domains:
            if domain not in present:
                problems.append(f"{band}: no {domain} items")
    return problems


def split_problems(items, tolerance=0.08):
    """dev is about a quarter of every band x domain x answer stratum."""
    problems = []
    strata = {}
    for row in items:
        key = (row["band"], row["domain"], row["answer"])
        strata.setdefault(key, []).append(row["split"])
    for key, splits in sorted(strata.items()):
        dev = splits.count("dev")
        total = len(splits)
        if not round(total * 0.25) - 1 <= dev <= round(total * 0.25) + 1:
            problems.append(f"{key}: {dev} of {total} in dev")
    total = len(items)
    if total:
        share = sum(row["split"] == "dev" for row in items) / total
        if abs(share - 0.25) > tolerance:
            problems.append(f"dev share {share:.3f} is not about 0.25")
    return problems


def pair_problems(items):
    problems = []
    by_id = {row["id"]: row for row in items}
    for row in items:
        if "pair" not in row:
            continue
        other = by_id.get(row["pair"])
        if other is None:
            problems.append(f"{row['id']}: pair {row['pair']} is missing")
        elif other.get("pair") != row["id"]:
            problems.append(f"{row['id']}: pair is not mutual")
        elif other["form"] == row["form"]:
            problems.append(f"{row['id']}: a pair asks once as a question "
                            "and once as a verdict")
    return problems


def check_files(paths, full_set=False, draft=False):
    """All problems of the given files. full_set adds coverage, balance, split.

    draft files are a writer's working copy: they have no header line.
    """
    problems = []
    marker = canary()
    items = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        problems += hygiene_problems(text, Path(path).name)
        try:
            head, rows = read_file(path)
        except ValueError as error:
            problems.append(str(error))
            continue
        if draft:
            if head is not None:
                rows = [head] + rows
        elif head is None or head.get("canary") != marker:
            problems.append(f"{Path(path).name}: header lacks the canary")
        elif head.get("schema") != SCHEMA:
            problems.append(f"{Path(path).name}: header schema is not {SCHEMA}")
        for row in rows:
            problems += validate_item(row)
        items += rows
    seen = set()
    for row in items:
        ident = row.get("id")
        if ident in seen:
            problems.append(f"{ident}: duplicate id")
        seen.add(ident)
    prompts = {}
    for row in items:
        key = (row.get("context", ""), row.get("prompt", ""))
        if key in prompts:
            problems.append(f"{row.get('id')}: same text as {prompts[key]}")
        prompts[key] = row.get("id")
    if not problems:
        problems += pair_problems(items)
        problems += balance_problems(items)
        if full_set:
            problems += coverage_problems(items)
            problems += split_problems(items)
            problems += length_cue_problems(items)
    return problems


def main(argv):
    if len(argv) < 1 or argv[0] != "check":
        sys.stderr.write("usage: python3 -m kilix_bench.onebit.items check "
                         "[--draft] [FILE...]\n")
        return 2
    draft = "--draft" in argv[1:]
    paths = [arg for arg in argv[1:] if arg != "--draft"] or item_files()
    problems = check_files(paths, full_set=not argv[1:], draft=draft)
    for problem in problems:
        print(problem)
    count = len(load_rows(paths, draft))
    print(f"{count} items in {len(paths)} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
