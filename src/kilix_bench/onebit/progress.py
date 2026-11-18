"""OneBit scores across training: one progress line per checkpoint, and a chart.

    kilix-bench onebit progress --arm qwen --run r1 --step 400 RESULT.json
    kilix-bench onebit chart progress.jsonl [--group computer-use]

A progress line names the training *arm* (the model being tuned: needle,
qwen, lfm2, ...), the run and the step, and carries the scores of each
*decode* in the result file (raw and constrained). The chart draws accuracy
against step, one line per arm and run, for the whole split and for one
chosen band or band/domain, with the 50% chance line.
"""

import json
from datetime import datetime, timezone
from html import escape

PROGRESS_SCHEMA = "kilix-bench.onebit.progress/v1"
KEPT = ("n", "skipped", "accuracy", "accuracy_all_items",
        "balanced_accuracy", "mcc", "format_failure_rate", "zero_share")

# Categorical slots, light and dark (validated: blue, orange, aqua, yellow).
LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
DARK = ("#3987e5", "#d95926", "#199e70", "#c98500")
ARM_ORDER = ("needle", "qwen", "lfm2")


def progress_line(payload, arm, run, step, when=None):
    """The progress record of one result file."""
    decodes = {}
    for decode, body in (payload.get("arms") or {}).items():
        if not body.get("supported"):
            decodes[decode] = {"supported": False}
            continue
        summary = body["summary"]
        entry = {key: summary["overall"].get(key) for key in KEPT}
        entry["supported"] = True
        entry["grade_level"] = summary["grade"]["grade_level"]
        entry["by_band"] = {name: row["accuracy"] for name, row in
                            summary["by_band"].items()}
        entry["by_band_domain"] = {name: row["accuracy"] for name, row in
                                   summary["by_band_domain"].items()}
        decodes[decode] = entry
    model = payload.get("model") or {}
    return {
        "schema": PROGRESS_SCHEMA,
        "arm": arm,
        "run": run,
        "step": step,
        "recorded": when or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "model": model.get("label") or model.get("model"),
        "split": payload["items"]["split"],
        "set_sha256": payload["items"]["set_sha256"],
        "contract": payload["contract"]["version"],
        "decodes": decodes,
    }


def append(path, line):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def read(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def value(line, decode, group, metric="accuracy"):
    """One number of a progress line: overall, a band, or a band/domain."""
    entry = (line.get("decodes") or {}).get(decode) or {}
    if not entry.get("supported"):
        return None
    if group in (None, "", "overall"):
        return entry.get(metric)
    if "/" in group:
        return (entry.get("by_band_domain") or {}).get(group)
    return (entry.get("by_band") or {}).get(group)


def series(lines, decode, group):
    """(arm, run) -> [(step, accuracy)] sorted by step, in a fixed arm order."""
    found = {}
    for line in lines:
        number = value(line, decode, group)
        if number is None:
            continue
        found.setdefault((line["arm"], line["run"]), {})[line["step"]] = number
    def order(key):
        arm, run = key
        rank = ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)
        return (rank, arm, run)
    return {key: sorted(points.items()) for key, points in sorted(
        found.items(), key=lambda item: order(item[0]))}


# Panel geometry: the plot area of panel k spans x 50..290 and y 30..230,
# shifted right by 380 * k, leaving room for the end labels. Accuracy 1.0 is
# y = 30, 0.0 is y = 230.
PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM, PANEL_STEP = 50, 290, 30, 230, 380


def scale(step, accuracy, low, high, panel=0):
    span = (high - low) or 1
    x = PLOT_LEFT + (step - low) / span * (PLOT_RIGHT - PLOT_LEFT)
    y = PLOT_BOTTOM - accuracy * (PLOT_BOTTOM - PLOT_TOP)
    return round(x + panel * PANEL_STEP, 1), round(y, 1)


def _arm_slot(arm, arms_seen):
    if arm in ARM_ORDER:
        return ARM_ORDER.index(arm) % len(LIGHT)
    return (len(ARM_ORDER) + arms_seen.index(arm)) % len(LIGHT)


def _marker(slot, filled):
    """A point's look: the first run of an arm is filled (with a surface
    ring); later runs of the same arm are hollow, so they stay apart even as
    single points, where the dashed line does not show."""
    if filled:
        return (f'class="f{slot} s{slot}" fill="{LIGHT[slot]}" '
                'stroke="#fcfcfb" stroke-width="2"')
    return (f'class="bg s{slot}" fill="#fcfcfb" stroke="{LIGHT[slot]}" '
            'stroke-width="2"')


def chart_svg(lines, decode="constrained", group="computer-use"):
    groups = ["overall", group]
    steps = [line["step"] for line in lines]
    low, high = (min(steps), max(steps)) if steps else (0, 1)
    arms_seen = sorted({line["arm"] for line in lines
                        if line["arm"] not in ARM_ORDER})
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 290" '
        'width="760" height="290" font-family="system-ui, sans-serif" '
        'font-size="11">',
        "<style>",
        ".bg{fill:#fcfcfb}.ink{fill:#1f1f1e}.muted{fill:#6b6a64}"
        ".grid{stroke:#e4e3de;stroke-width:1}"
        ".chance{stroke:#8f8e86;stroke-width:1;stroke-dasharray:4 3}",
    ]
    # Light colours are presentation attributes on the marks, so a renderer
    # without CSS still draws them; the dark rules below override them.
    out.append("@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}"
               ".ink{fill:#f0efec}.muted{fill:#c3c2b7}"
               ".grid{stroke:#33332f}.chance{stroke:#8f8e86}")
    for slot in range(len(DARK)):
        out.append(f".s{slot}{{stroke:{DARK[slot]}}}.f{slot}{{fill:{DARK[slot]}}}")
    out.append("}</style>")
    out.append('<rect class="bg" fill="#fcfcfb" width="760" height="290"/>')
    for panel, name in enumerate(groups):
        shift = panel * PANEL_STEP
        out.append(
            f'<text class="ink" fill="#1f1f1e" x="{PLOT_LEFT + shift}" y="18" '
            f'font-weight="600">{escape(name)} accuracy ({escape(decode)}, '
            f'{escape(lines[0]["split"]) if lines else "dev"} split)</text>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            _x, y = scale(low, tick, low, high)
            if tick == 0.5:
                look = ('class="chance" stroke="#8f8e86" stroke-width="1" '
                        'stroke-dasharray="4 3"')
            else:
                look = 'class="grid" stroke="#e4e3de" stroke-width="1"'
            out.append(f'<line {look} x1="{PLOT_LEFT + shift}" '
                       f'x2="{PLOT_RIGHT + shift}" y1="{y}" y2="{y}"/>')
            out.append(f'<text class="muted" fill="#6b6a64" '
                       f'x="{PLOT_LEFT + shift - 6}" y="{y + 4}" '
                       f'text-anchor="end">{int(tick * 100)}%</text>')
        _x, chance_y = scale(low, 0.5, low, high)
        out.append(f'<text class="muted" fill="#6b6a64" '
                   f'x="{PLOT_RIGHT + shift}" y="{chance_y + 13}" '
                   'text-anchor="end">chance 50%</text>')
        for step in sorted({low, high}):
            x, _y = scale(step, 0, low, high, panel)
            out.append(f'<text class="muted" fill="#6b6a64" x="{x}" '
                       f'y="{PLOT_BOTTOM + 16}" text-anchor="middle">'
                       f"step {step}</text>")
        runs_of_arm = {}
        ends = []
        for (arm, run), points in series(lines, decode, name).items():
            slot = _arm_slot(arm, arms_seen)
            runs_of_arm.setdefault(arm, []).append(run)
            dash = "" if len(runs_of_arm[arm]) == 1 else \
                ' stroke-dasharray="6 3"'
            coords = [scale(step, acc, low, high, panel)
                      for step, acc in points]
            text = " ".join(f"{x},{y}" for x, y in coords)
            out.append(f'<polyline class="s{slot}" fill="none" '
                       f'stroke="{LIGHT[slot]}" stroke-width="2"{dash} '
                       f'points="{text}">'
                       f"<title>{escape(arm)} {escape(run)}</title>"
                       "</polyline>")
            for (x, y), (step, acc) in zip(coords, points):
                out.append(
                    f'<circle {_marker(slot, not dash)} cx="{x}" cy="{y}" '
                    f'r="5"><title>{escape(arm)} {escape(run)} step {step}: '
                    f"{100 * acc:.1f}%</title></circle>")
            ends.append([coords[-1][1], coords[-1][0],
                         f"{escape(arm)} {escape(run)}"])
        # Direct labels at the line ends, pushed apart to 13 px.
        ends.sort()
        for index in range(1, len(ends)):
            ends[index][0] = max(ends[index][0], ends[index - 1][0] + 13)
        for y, x, label in ends:
            out.append(f'<text class="ink" fill="#1f1f1e" x="{x + 9}" '
                       f'y="{y + 4}">{label}</text>')
    legend_x = PLOT_LEFT
    legend_runs = {}
    for (arm, run) in series(lines, decode, "overall"):
        slot = _arm_slot(arm, arms_seen)
        legend_runs.setdefault(arm, []).append(run)
        first = len(legend_runs[arm]) == 1
        dash = "" if first else ' stroke-dasharray="4 2"'
        out.append(f'<line class="s{slot}" stroke="{LIGHT[slot]}" '
                   f'stroke-width="2"{dash} x1="{legend_x - 2}" '
                   f'x2="{legend_x + 16}" y1="268" y2="268"/>')
        out.append(f'<circle {_marker(slot, first)} cx="{legend_x + 7}" '
                   'cy="268" r="4"/>')
        out.append(f'<text class="ink" fill="#1f1f1e" x="{legend_x + 19}" '
                   f'y="272">{escape(arm)} {escape(run)}</text>')
        legend_x += 120
    out.append("</svg>")
    return "\n".join(out) + "\n"


def chart_table(lines, decode="constrained", group="computer-use"):
    """The same numbers as text, one row per arm, run and step."""
    rows = [f"{'arm':<10} {'run':<12} {'step':>7} {'overall':>8} "
            f"{group:>16}  (chance 50.0%)"]
    overall = series(lines, decode, "overall")
    chosen = series(lines, decode, group)
    for key, points in overall.items():
        other = dict(chosen.get(key, []))
        for step, acc in points:
            pick = other.get(step)
            rows.append(
                f"{key[0]:<10} {key[1]:<12} {step:>7} {100 * acc:>7.1f}% "
                + (f"{100 * pick:>15.1f}%" if pick is not None
                   else f"{'-':>16}"))
    return "\n".join(rows) + "\n"
