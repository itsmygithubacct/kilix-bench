"""Private bench results under the Kilix data directory."""

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


def bench_home():
    override = os.environ.get("KILIX_BENCH_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "gpu_terminal" / "kilix-bench"


def _private(path):
    """Create a user-owned directory that other users cannot read."""
    if path.is_symlink():
        raise SystemExit("refusing a symlink for bench storage")
    path.mkdir(parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode):
        raise SystemExit("bench storage is not a directory owned by this user")
    os.chmod(path, 0o700)
    return path


def results_dir():
    root = _private(bench_home())
    return _private(root / "results")


def summarize(samples):
    """samples are seconds. Returned times are milliseconds."""
    if not samples:
        raise ValueError("a measurement needs at least one sample")
    ordered = sorted(samples)
    count = len(ordered)

    def at(fraction):
        if count == 1:
            return ordered[0]
        cursor = (count - 1) * fraction
        low = int(cursor)
        high = min(low + 1, count - 1)
        weight = cursor - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    def ms(seconds):
        return round(seconds * 1000, 4)

    return {
        "samples": count,
        "min_ms": ms(ordered[0]),
        "median_ms": ms(at(0.5)),
        "p95_ms": ms(at(0.95)),
        "max_ms": ms(ordered[-1]),
    }


def scenario(name, what, samples, extra=None):
    row = {"name": name, "what": what, "unit": "ms"}
    row.update(summarize(samples))
    if extra:
        row["extra"] = extra
    return row


def assert_no_local_paths(payload, forbidden):
    blob = json.dumps(payload)
    for item in forbidden:
        if item and item in blob:
            raise SystemExit("bench result included a local path")


def render_text(payload):
    lines = [
        "kilix-bench {tool_version}  {subject} {subject_version} {revision}".format(
            tool_version=payload["tool_version"],
            subject=payload["subject"],
            subject_version=payload["subject_version"],
            revision=payload.get("subject_revision") or "uncommitted",
        ),
        "size {w}x{h}  python {python}  {machine}".format(
            w=payload["size"][0], h=payload["size"][1],
            python=payload["python"], machine=payload["machine"]),
        payload.get("note") or (
            "compose times are the PIL redraw. They do not include Kitty."),
        "",
    ]
    for row in payload["scenarios"]:
        lines.append(
            "{name:<22} median {median:8.3f} ms   "
            "min {min_:7.3f}  p95 {p95:8.3f}  max {max_:8.3f}".format(
                name=row["name"],
                median=row["median_ms"],
                min_=row["min_ms"],
                p95=row["p95_ms"],
                max_=row["max_ms"],
            ))
        extra = row.get("extra") or {}
        if "cpu" in extra:
            lines.append("                       CPU median {median_ms:.3f} ms/frame".format(**extra['cpu']))
        if "first_frame" in extra:
            lines.append("                       first frame median {median_ms:.3f} ms".format(**extra['first_frame']))
        if "compose_fps" in extra:
            lines.append(
                "                       compose only: {fps} fps".format(
                    fps=extra["compose_fps"]))
        if extra.get("kind") == "cells":
            lines.append(
                "                       {cells} cells  wire {wire} bytes  "
                "frame {full} bytes".format(
                    cells=extra.get("cells", 0),
                    wire=extra.get("wire_bytes", 0),
                    full=extra.get("full_bytes", 0)))
        elif "kind" in extra:
            detail = (
                "                       {kind}  {pixel} of {full} pixel bytes"
                .format(
                    kind=extra["kind"],
                    pixel=extra.get("pixel_bytes", 0),
                    full=extra.get("full_bytes", 0)))
            if "rect_count" in extra:
                detail += "  presenter rects {rects}".format(
                    rects=extra["rect_count"])
                if "rect_bytes" in extra:
                    detail += " ({rect_bytes} bytes)".format(
                        rect_bytes=extra["rect_bytes"])
            if "wire_bytes" in extra:
                detail += "  wire {wire} bytes".format(wire=extra["wire_bytes"])
            lines.append(detail)
        elif "diff_bytes" in extra:
            lines.append(
                "                       {count} rects  {nbytes} bytes".format(
                    count=extra.get("diff_rects", 0),
                    nbytes=extra["diff_bytes"]))
        if "us_per_cell" in extra:
            lines.append(
                "                       {cells} cells  {us:.4f} us/cell".format(
                    cells=extra.get("cells", 0),
                    us=extra["us_per_cell"]))
    return "\n".join(lines) + "\n"


def write_run(payload, forbidden=(), slug=None):
    assert_no_local_paths(payload, forbidden)
    directory = results_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = slug or payload.get("subject") or "kilix-95"
    stamped = directory / f"{stamp}-{name}.json"
    latest = directory / f"{name}-latest.json"
    text = directory / f"{name}-latest.txt"
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    rendered = render_text(payload)
    _replace(stamped, body)
    _replace(latest, body)
    _replace(text, rendered)
    return stamped, latest, text


def _replace(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
