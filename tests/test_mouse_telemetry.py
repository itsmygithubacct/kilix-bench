import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from kilix_bench import mouse_telemetry as t


def record(event, **fields):
    return dict(schema=t.SCHEMA, event=event, **fields)


def metadata():
    return record('metadata', clock='CLOCK_MONOTONIC',
                  input_boundary='userspace evdev read completion',
                  present_boundary='userspace DRM pageflip callback',
                  diagnostic_snapshot_live=False, diagnostic_snapshot_history=False,
                  max_read_events=256, max_dispatch_events=256)


def input_row(seq, timestamp=1000000, dispatched=True, action=2, button=0):
    return record('pointer_input', seq=seq, input_ns=timestamp, x=seq, y=64,
                  action=action, button=button, dispatched=dispatched)


def present(seq, *, count=1, coalesced=0, kind='pointer', cancelled=False,
            render=2000000, submit=3000000, callback=4000000):
    return record('present', pointer_seq=seq, status='cancelled' if cancelled else 'presented',
                  requested_x=seq, requested_y=64 if seq is not None else None,
                  presented_x=None if cancelled else seq,
                  presented_y=None if cancelled or seq is None else 64,
                  render_start_ns=render, submit_ns=None if cancelled else submit,
                  pageflip_ns=None if cancelled else callback, frame_kind=kind,
                  input_events=count, coalesced_motion=coalesced)


class AnalysisTests(unittest.TestCase):
    def test_full_pointer_coalescing_and_unobserved_are_distinct(self):
        rows = [metadata(), present(None, count=0, kind='full'),
                input_row(1, 5000000, False), input_row(2, 5000000), input_row(3, 5000000),
                present(3, count=3, coalesced=1, render=6000000, submit=7000000, callback=8000000),
                input_row(4, 9000000),
                present(4, kind='full', render=10000000, submit=11000000, callback=14000000)]
        result = t.analyze(rows)
        self.assertEqual(result['counts']['coalesced_motion'], 1)
        self.assertEqual(result['counts']['dispatched'], 3)
        self.assertEqual(result['counts']['dispatched_without_individual_callback'], 1)
        self.assertEqual(result['frames']['pointer']['read_to_callback']['p50_ms'], 3)
        self.assertEqual(result['frames']['full']['read_to_callback']['p50_ms'], 5)
        self.assertEqual(result['frames']['full']['uncorrelated_presentations'], 1)
        self.assertNotIn('lost_events', result['counts'])

    def test_cancelled_observation_never_scores_then_retry_scores_once(self):
        rows = [metadata(), input_row(1), present(1, cancelled=True),
                present(1, count=0, kind='full', render=3000000, submit=4000000, callback=5000000),
                present(1, count=0, kind='full', render=6000000, submit=7000000, callback=8000000)]
        result = t.analyze(rows)
        self.assertEqual(result['counts']['cancelled_observations'], 1)
        self.assertEqual(result['frames']['pointer']['read_to_callback']['samples'], 0)
        self.assertEqual(result['frames']['full']['read_to_callback']['samples'], 1)
        self.assertEqual(result['frames']['full']['read_to_callback']['p50_ms'], 4)
        self.assertEqual(result['frames']['full']['repeated_pointer_observations'], 1)
        cancelled = t.analyze(rows[:3])
        self.assertEqual(cancelled['counts']['first_observed_dispatched'], 0)
        self.assertEqual(cancelled['counts']['dispatched_without_individual_callback'], 1)

    def test_backlogged_read_timestamps_can_precede_previous_callback(self):
        result = t.analyze([metadata(), input_row(1), present(1), input_row(2),
                            present(2, render=5000000, submit=6000000, callback=7000000)])
        self.assertEqual(result['frames']['pointer']['read_to_callback']['samples'], 2)

    def test_discontinuity_resets_reference_not_counters_vt_resets_both(self):
        rows = [metadata(), input_row(1), record('input_discontinuity', input_ns=1000000, reason='SYN_DROPPED'),
                present(None), input_row(2, 5000000),
                record('vt', acquire=False, monotonic_ns=6000000),
                record('vt', acquire=True, monotonic_ns=7000000),
                present(None, count=0, render=8000000, submit=9000000, callback=10000000)]
        result = t.analyze(rows)
        self.assertEqual(result['counts']['input_discontinuities'], 1)
        self.assertEqual(result['counts']['vt_events'], 2)
        self.assertEqual(result['counts']['inputs_reset_by_vt'], 1)
        self.assertEqual(result['counts']['dispatched_without_individual_callback'], 2)
        self.assertEqual(result['counts']['first_observed_dispatched'], 0)
        bad = copy.deepcopy(rows[:3]) + [present(1)]
        with self.assertRaisesRegex(ValueError, 'stale'):
            t.analyze(bad)

    def test_incomplete_tail_and_empty_distribution_are_explicit(self):
        result = t.analyze([metadata(), input_row(1, dispatched=False)])
        self.assertEqual(result['counts']['inputs_since_last_observation'], 1)
        self.assertEqual(result['counts']['coalesced_since_last_observation'], 1)
        self.assertIsNone(result['frames']['full']['read_to_callback']['p99_ms'])
        self.assertIn('No end marker', result['completion'])

    def test_malformed_order_reference_and_counter_failures(self):
        base = [metadata(), input_row(1), present(1)]
        mutations = [
            (0, 'clock', 'CLOCK_REALTIME'), (0, 'diagnostic_snapshot_live', 0),
            (1, 'seq', 2), (1, 'seq', True), (1, 'input_ns', -1),
            (1, 'input_ns', 1.5), (1, 'input_ns', 1 << 64), (1, 'input_ns', True), (1, 'x', None),
            (1, 'action', 3), (1, 'button', 4), (1, 'dispatched', 1),
            (2, 'pointer_seq', 2), (2, 'pointer_seq', True), (2, 'requested_x', 8),
            (2, 'presented_x', 8), (2, 'frame_kind', 'other'),
            (2, 'render_start_ns', 0), (2, 'submit_ns', 100),
            (2, 'pageflip_ns', 100), (2, 'input_events', 0),
            (2, 'coalesced_motion', 1), (2, 'status', 'cancelled')]
        for index, field, value in mutations:
            with self.subTest(index=index, field=field, value=value):
                rows = copy.deepcopy(base); rows[index][field] = value
                with self.assertRaises(ValueError): t.analyze(rows)
        for rows in ([], [input_row(1)], [metadata(), metadata()],
                     [metadata(), input_row(1), input_row(1)],
                     [metadata(), input_row(1), input_row(2, timestamp=0)],
                     [metadata(), input_row(1, dispatched=False, action=0, button=1)],
                     [metadata(), input_row(1, dispatched=False), present(1, coalesced=1)],
                     [metadata(), record('new_unknown_event')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError): t.analyze(rows)

    def test_cancelled_pixels_and_timestamps_are_rejected(self):
        for field in ('presented_x', 'presented_y', 'submit_ns', 'pageflip_ns'):
            row = present(1, cancelled=True); row[field] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                t.analyze([metadata(), input_row(1), row])

    def test_nonfinite_exponent_is_rejected_even_in_an_extension(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            t.parse_json('{"extension":1e999}')

    def test_percentiles_interpolate_and_empty_is_null(self):
        result = t.distribution([1000000, 3000000])
        self.assertEqual(result['p50_ms'], 2)
        self.assertAlmostEqual(result['p95_ms'], 2.9)
        self.assertAlmostEqual(result['p99_ms'], 2.98)
        self.assertIsNone(t.distribution([])['max_ms'])


class CliTests(unittest.TestCase):
    def test_cli_provenance_only_join_no_injected_latency_and_exclusive_private_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.jsonl'; output = root/'analysis.json'
            source.write_text(''.join(json.dumps(r)+'\n' for r in [metadata(), input_row(1), present(1)]))
            run = root/'run.json'
            # A plausible matching coordinate or local clock name is not proof of a shared boot.
            run.write_text(json.dumps(dict(schema='kilix.mouse.v1', clock='CLOCK_MONOTONIC',
                boot_id='another-boot', geometry={'origin_x':0, 'origin_y':64},
                expected=[dict(x=1, y=0, action=3, injected_ns=0)], failures=[{'type':'receipt_timeout'}])))
            before = run.read_bytes()
            args = ['--input', str(source), '--run', str(run), '--output', str(output)]
            self.assertEqual(t.main(args), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result['join']['status'], 'omitted')
            self.assertNotIn('injected_to_callback', result)
            self.assertEqual(run.read_bytes(), before)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with contextlib.redirect_stderr(io.StringIO()): self.assertEqual(t.main(args), 2)

    def test_bad_json_truncation_duplicate_keys_nan_and_wrong_run_schema_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.jsonl'
            for data in ('', '\n', '{', json.dumps(metadata()),
                         '{"schema":"x","schema":"y"}\n', '{"x":NaN}\n', '[]\n'):
                source.write_text(data)
                with self.subTest(data=data), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(t.main(['--input', str(source)]), 2)
            source.write_text(json.dumps(metadata())+'\n')
            run = root/'run.json'; run.write_text('{}')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(t.main(['--input', str(source), '--run', str(run)]), 2)


if __name__ == '__main__':
    unittest.main()
