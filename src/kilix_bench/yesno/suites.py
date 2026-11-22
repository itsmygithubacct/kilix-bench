"""Dataset identity and validation items for the strict YES/NO benchmarks."""
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile


PROTOCOL = "strict-yesno-v1"


@dataclass(frozen=True)
class Suite:
    key: str
    label: str
    dataset: str
    split: str

    def folder(self, state):
        # Keep the original OneBit journals, locks and summary-only rankings in place.
        if self == K12:
            return state / "bench" / "yesno"
        return state / "bench" / self.key / self.split / PROTOCOL

    def metadata(self):
        return dict(suite=self.key, benchmark=self.dataset, split=self.split, protocol=PROTOCOL)

    def matches(self, fields):
        if "suite" in fields:
            return all(fields.get(k) == v for k, v in self.metadata().items())
        # Before dataset selection, the only worker was the OneBit dev worker.
        return self == K12 and all(fields.get(k, v) == v for k, v in self.metadata().items())


K12 = Suite("kilix-k12-computer-use", "kilix-k12-computer use (yes/no)", "OneBit", "dev")
BOOLQ = Suite("boolq", "BoolQ (YES/NO)", "BoolQ", "validation")
SUITES = {suite.key: suite for suite in (K12, BOOLQ)}


def recorded_suite(fields):
    """Identify a saved run or board, including the original untagged OneBit format."""
    suite = SUITES.get(fields.get("suite", K12.key))
    if suite is None or not suite.matches(fields):
        raise ValueError("Unrecognized benchmark suite, split or scoring protocol.")
    return suite


BOOLQ_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
BOOLQ_MEMBER = "BoolQ/val.jsonl"
BOOLQ_SHA256 = "0c86a5045886e5795fe9052003873f7d94b88ed3028a33007c51d99e44fd66d9"
BOOLQ_BYTES = 2254565


def boolq_items(status):
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    path = cache / "kilix-bench" / "boolq" / "validation-v2.jsonl"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    if hashlib.sha256(data).hexdigest() != BOOLQ_SHA256:
        status("Downloading BoolQ validation questions from SuperGLUE.")
        with urlopen(BOOLQ_URL, timeout=60) as source:
            archive = source.read(8 * 1024 * 1024 + 1)
        if len(archive) > 8 * 1024 * 1024:
            raise ValueError("BoolQ download exceeded the expected archive size.")
        with ZipFile(io.BytesIO(archive)) as bundle:
            if bundle.getinfo(BOOLQ_MEMBER).file_size != BOOLQ_BYTES:
                raise ValueError("Unexpected BoolQ validation file size.")
            data = bundle.read(BOOLQ_MEMBER)
        if hashlib.sha256(data).hexdigest() != BOOLQ_SHA256:
            raise ValueError("BoolQ validation data failed its SHA-256 check.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            try:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    items = []
    for line in data.splitlines():
        row = json.loads(line)
        question = row["question"]
        items.append(dict(id=f"boolq-validation-{row['idx']}", band="BoolQ", context=row["passage"],
                          prompt=question if question.endswith("?") else question + "?",
                          answer=0 if row["label"] else 1, rationale=""))
    return items
