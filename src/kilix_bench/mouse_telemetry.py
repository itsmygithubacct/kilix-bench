"""Optional read-to-callback analysis; never changes mouse run/report verdicts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

SCHEMA = 'neuronte.pointer.v1'
BOUNDARY = 'userspace evdev read completion to userspace DRM pageflip callback; not photon latency'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and minimum <= value <= (1 << 64)-1,
            f'{name} must be an integer in [{minimum}, 2**64-1]')
    return value


def boolean(value, name):
    require(type(value) is bool, f'{name} must be boolean')
    return value


def pair(row, prefix):
    x, y = row[prefix + '_x'], row[prefix + '_y']
    if x is None and y is None:
        return None
    integer(x, prefix + '_x')
    integer(y, prefix + '_y')
    return x, y


def distribution(values):
    values = sorted(v / 1e6 for v in values)
    def percentile(q):
        if not values:
            return None
        at = (len(values) - 1) * q
        low = int(at)
        return values[low] + (values[min(low + 1, len(values)-1)]-values[low]) * (at-low)
    return dict(samples=len(values), p50_ms=percentile(.5), p95_ms=percentile(.95),
                p99_ms=percentile(.99), max_ms=max(values) if values else None)


def analyze(rows):
    """Validate a complete stream and score the first successful observation per input.

    Repeated content redraws referencing an old pointer do not create latency samples.
    Queued input may retain a read timestamp older than an earlier frame callback.
    """
    metadata = None
    seq = 0
    latest = None
    last_input_ns = -1
    last_observation_ns = -1
    pending_count = pending_coalesced = 0
    dispatched = set()
    observed = set()
    totals = dict(input_events=0, dispatched=0, coalesced_motion=0, presented_observations=0,
                  cancelled_observations=0, vt_events=0, input_discontinuities=0,
                  inputs_reset_by_vt=0)
    frames = {kind: dict(presented=0, cancelled=0, repeated_pointer_observations=0,
                        uncorrelated_presentations=0, latencies=[])
              for kind in ('full', 'pointer')}
    discontinuities = []
    for line, row in enumerate(rows, 1):
        try:
            require(isinstance(row, dict), 'record must be an object')
            require(row.get('schema') == SCHEMA, 'unsupported schema')
            event = row['event']
            if metadata is None:
                require(event == 'metadata', 'first record must be metadata')
                require(row['clock'] == 'CLOCK_MONOTONIC', 'unsupported clock')
                require(row['input_boundary'] == 'userspace evdev read completion', 'unsupported input boundary')
                require(row['present_boundary'] == 'userspace DRM pageflip callback', 'unsupported presentation boundary')
                for key in ('diagnostic_snapshot_live', 'diagnostic_snapshot_history'):
                    boolean(row[key], key)
                for key in ('max_read_events', 'max_dispatch_events'):
                    integer(row[key], key, 1)
                # Copy only supported declarations, not arbitrary private metadata.
                metadata = {key: row[key] for key in ('clock', 'input_boundary', 'present_boundary',
                            'diagnostic_snapshot_live', 'diagnostic_snapshot_history',
                            'max_read_events', 'max_dispatch_events')}
            elif event == 'pointer_input':
                require(integer(row['seq'], 'seq', 1) == seq + 1, 'input sequence gap, duplicate or reordering')
                seq = row['seq']
                timestamp = integer(row['input_ns'], 'input_ns')
                require(timestamp >= last_input_ns, 'input read timestamps moved backward')
                last_input_ns = timestamp
                integer(row['x'], 'x'); integer(row['y'], 'y')
                action, button = row['action'], row['button']
                integer(action, 'action'); integer(button, 'button')
                physical = (1, 2, 3, 8, 9, 10, 11)
                require((action == 0 and 1 <= button <= 11) or
                        (action == 1 and button in physical) or
                        (action == 2 and button in (0, *physical)), 'invalid action/button')
                is_dispatched = boolean(row['dispatched'], 'dispatched')
                require(is_dispatched or action == 2, 'only motion may be coalesced')
                totals['input_events'] += 1
                pending_count += 1
                if is_dispatched:
                    dispatched.add(seq)
                    totals['dispatched'] += 1
                    latest = row
                else:
                    totals['coalesced_motion'] += 1
                    pending_coalesced += 1
            elif event == 'present':
                kind = row['frame_kind']
                require(kind in frames, 'unsupported frame_kind')
                status = row['status']
                require(status in ('presented', 'cancelled'), 'unsupported present status')
                ref = row['pointer_seq']
                if ref is not None:
                    integer(ref, 'pointer_seq', 1)
                require(ref == (latest['seq'] if latest else None), 'presentation references stale, unknown or undispatched input')
                requested, presented = pair(row, 'requested'), pair(row, 'presented')
                if latest:
                    require(requested == (latest['x'], latest['y']), 'requested coordinates differ from referenced input')
                render = integer(row['render_start_ns'], 'render_start_ns')
                require(render >= last_observation_ns, 'render observations moved backward')
                if latest:
                    require(render >= latest['input_ns'], 'render precedes referenced read')
                require(integer(row['input_events'], 'input_events') == pending_count, 'input counter differs from records')
                require(integer(row['coalesced_motion'], 'coalesced_motion') == pending_coalesced, 'coalesced counter differs from records')
                pending_count = pending_coalesced = 0
                group = frames[kind]
                if status == 'cancelled':
                    require(presented is None and row['submit_ns'] is None and row['pageflip_ns'] is None,
                            'cancelled observation must not claim presented coordinates or timestamps')
                    group['cancelled'] += 1
                    totals['cancelled_observations'] += 1
                    last_observation_ns = render
                else:
                    submit = integer(row['submit_ns'], 'submit_ns')
                    callback = integer(row['pageflip_ns'], 'pageflip_ns')
                    require(render <= submit <= callback, 'render/submit/callback order is invalid')
                    require(presented == requested, 'presented coordinates differ from requested coordinates')
                    totals['presented_observations'] += 1
                    group['presented'] += 1
                    last_observation_ns = callback
                    if latest is None:
                        group['uncorrelated_presentations'] += 1
                    elif ref in observed:
                        group['repeated_pointer_observations'] += 1
                    else:
                        group['latencies'].append(callback - latest['input_ns'])
                        observed.add(ref)
            elif event == 'vt':
                boolean(row['acquire'], 'acquire')
                timestamp = integer(row['monotonic_ns'], 'monotonic_ns')
                require(timestamp >= max(last_observation_ns, last_input_ns), 'VT timestamp moved backward')
                last_observation_ns = timestamp
                totals['vt_events'] += 1
                totals['inputs_reset_by_vt'] += pending_count
                pending_count = pending_coalesced = 0
                latest = None
            elif event == 'input_discontinuity':
                timestamp = integer(row['input_ns'], 'input_ns')
                require(timestamp >= last_input_ns, 'discontinuity read timestamp moved backward')
                last_input_ns = timestamp
                require(row['reason'] == 'SYN_DROPPED', 'unsupported discontinuity reason')
                totals['input_discontinuities'] += 1
                discontinuities.append(dict(line=line, after_seq=seq, reason='SYN_DROPPED'))
                latest = None
                # Runtime does NOT reset present counters on SYN_DROPPED.
            else:
                raise ValueError('unknown or repeated metadata event')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'line {line}: {exc}') from None
    require(metadata is not None, 'empty telemetry stream')
    for group in frames.values():
        group['read_to_callback'] = distribution(group.pop('latencies'))
    totals.update(first_observed_dispatched=len(observed),
                  dispatched_without_individual_callback=len(dispatched-observed),
                  inputs_since_last_observation=pending_count,
                  coalesced_since_last_observation=pending_coalesced)
    return dict(schema='kilix.mouse.telemetry.v1', status='valid', metric=BOUNDARY,
                metadata=metadata, counts=totals, frames=frames, discontinuities=discontinuities,
                loss_assessment='Not measurable from this stream: coalesced motion and dispatched inputs without individual callbacks are not transport-loss counts.',
                cancellation_semantics='Cancelled means observation interrupted; an already queued DRM flip may still complete. No pixels or latency are attributed to it.',
                completion='No end marker exists in v1; validity does not prove a clean close or exclude a record-aligned truncated tail.',
                join=dict(status='omitted', reason='Current telemetry v1 has no boot identity or declared coordinate mapping; no injected-to-callback metric or cross-file event attribution.'))


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse_json(text):
    def invalid(_):
        raise ValueError('nonfinite JSON number')
    def finite(value):
        number = float(value)
        require(math.isfinite(number), 'nonfinite JSON number')
        return number
    return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=invalid, parse_float=finite)


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line, text in enumerate(stream, 1):
            try:
                require(text.endswith('\n'), 'unterminated JSONL record (possibly truncated)')
                yield parse_json(text)
            except ValueError as exc:
                raise ValueError(f'line {line}: invalid JSONL: {exc}') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='closed neuronte.pointer.v1 JSONL')
    parser.add_argument('--output', type=Path, help='new JSON report; default stdout, existing files refused')
    parser.add_argument('--run', type=Path, help='optional run.json provenance only; current v1 cannot prove a timing join')
    args = parser.parse_args(argv)
    try:
        result = analyze(read_rows(args.input))
        result['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['telemetry_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        if args.run:
            raw = args.run.read_bytes()
            run = parse_json(raw)
            require(isinstance(run, dict) and run.get('schema') == 'kilix.mouse.v1', 'unsupported run schema')
            require(run.get('clock') == 'CLOCK_MONOTONIC' and isinstance(run.get('boot_id'), str)
                    and bool(run['boot_id']) and isinstance(run.get('geometry'), dict), 'missing run clock/geometry provenance')
            result['join'].update(run_sha256=hashlib.sha256(raw).hexdigest(),
                                  benchmark_verdict='Not assessed or changed; use kilix_bench.mouse report for the existing strict receipt criteria.')
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            # Private output; exclusive creation also prevents overwriting raw inputs.
            import os
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as stream:
                stream.write(text)
        else:
            sys.stdout.write(text)
        return 0
    except (OSError, UnicodeError, ValueError) as exc:
        print(f'mouse telemetry: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
