"""Resume a frozen decomposition protocol from an immutable trial prefix.

Run with the ORIGINAL harness on PYTHONPATH. This adapter leaves its files and
native worker unchanged. Validation is repeated outside native timing; completed
timing rows are checked and replayed into a new journal without native calls.
The original journal is never opened for writing. Every missing job runs through
the original run_worker, including its normal checks and fresh process/handles.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

from kilix_bench import decomposition, fonts
from kilix_bench.neuronte import draw_reference


class CheckpointExit(Exception):
    """A requested boundary was reached before starting another native job."""


def schedule(args, manifest):
    rng = random.Random(args.seed)
    codes = list(manifest['corpus'])
    rng.shuffle(codes)
    jobs = []
    for case in manifest['cases']:
        jobs.append((case, args.cells[0], 'open'))
        if case['kind'] == 'bundle':
            jobs.append((case, (2, 2), 'route-bench'))
        jobs.extend((case, cell, mode) for cell in args.cells
                    for mode in args.expert_cases.split(','))
    plan = []
    for repeat in range(-args.warmup, args.repeats):
        order = list(jobs)
        rng.shuffle(order)
        for index, (case, cell, mode) in enumerate(order):
            calls = len(codes) * (512 if mode == 'route-bench' else
                                 1 if mode in ('open', 'draw-cold') else args.expert_passes)
            plan.append({'model': case['name'], 'cell': list(cell), 'case': mode,
                         'repeat': repeat, 'order': index, 'calls': calls})
    return codes, plan


def validate_prefix(data, plan):
    if data and not data.endswith(b'\n'):
        raise ValueError('partial journal line; preserve and investigate')
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) > len(plan):
        raise ValueError('prefix exceeds complete protocol')
    for row, job in zip(rows, plan):
        for key in ('model', 'cell', 'case', 'repeat', 'order'):
            if row[key] != job[key]:
                raise ValueError('prefix is not the exact seeded job order: ' + key)
        mode = job['case']
        if mode != 'open' and row['calls'] != job['calls']:
            raise ValueError('prefix call count differs')
        duration = row['initialize_ns'] if mode == 'open' else row['elapsed_ns']
        cpu = row['initialize_cpu_ns'] if mode == 'open' else row['cpu_ns']
        denominator = 1 if mode == 'open' else job['calls']
        if any(not math.isfinite(x) or x < 0 for x in (duration, cpu)):
            raise ValueError('invalid duration')
        if (row['ns_per_operation'] != duration / denominator or
                row['cpu_ns_per_operation'] != cpu / denominator):
            raise ValueError('prefix normalization differs')
    return rows


def verify_files(receipt):
    for item in receipt['files']:
        if fonts.digest(item['path']) != item['sha256']:
            raise ValueError('checkpoint provenance changed: ' + item['path'])
    if fonts.digest(receipt['binary']) != receipt['build']['binary_sha256']:
        raise ValueError('preserved native worker differs from original build')


def required_files(paths, binary):
    """The minimum provenance closure for the original native experiment."""
    package = Path(fonts.__file__).parent
    required = {Path(p).resolve() for p in paths.values()}
    required.update(p.resolve() for p in package.glob('*.py'))
    required.add((package / 'font_worker.c').resolve())
    required.add(Path(binary).resolve())
    for executable in (binary, paths['library']):
        linked = subprocess.check_output(['ldd', str(executable)], text=True)
        for line in linked.splitlines():
            if 'not found' in line:
                raise ValueError('native dependency is missing')
            for field in line.split():
                if field.startswith('/'):
                    required.add(Path(field).resolve())
    return required


def resume(receipt, output, *, stop_after=None):
    """Return the original result schema, plus explicit segment provenance."""
    verify_files(receipt)
    args = argparse.Namespace(**receipt['args'])
    args.cells = [tuple(cell) for cell in args.cells]
    family = receipt['family']
    paths = decomposition.inputs(args, family=family)
    recorded = {Path(item['path']).resolve() for item in receipt['files']}
    missing = required_files(paths, receipt['binary']) - recorded
    if missing:
        raise ValueError('missing required provenance: ' + ', '.join(map(str, sorted(missing))))
    manifest, _ = decomposition.read_experiment(paths['experiment'], family=family)
    codes, plan = schedule(args, manifest)
    prefix = Path(receipt['journal']).read_bytes()
    if hashlib.sha256(prefix).hexdigest() != receipt['journal_sha256']:
        raise ValueError('checkpoint journal changed')
    rows = validate_prefix(prefix, plan)
    if stop_after is not None and not len(rows) <= stop_after <= len(plan):
        raise ValueError('stop boundary precedes checkpoint or exceeds schedule')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoint-input.json').write_text(json.dumps(receipt, indent=2) + '\n')
    native = fonts.run_worker
    index = 0
    expected = {}
    cases = {case['name']: case for case in manifest['cases']}
    model_names = {str(paths['model:' + name]): name for name in manifest['models']}
    began = datetime.now(timezone.utc).isoformat()

    def checked_worker(binary, local_paths, engine, mode, cell, scalars, local_args,
                       directory, **kwargs):
        nonlocal index
        if list(scalars) != codes:
            raise ValueError('corpus order differs')
        if mode in ('nrt-routes', 'nrt-validate'):
            row, raw = native(binary, local_paths, engine, mode, cell, scalars,
                              local_args, directory, **kwargs)
            if mode == 'nrt-validate':
                member = model_names[str(local_paths['weights'])]
                expected[member, tuple(cell)] = raw[-cell[0] * cell[1] * 4:]
            return row, raw
        job = plan[index]
        case = cases[job['model']]
        actual_models = [str(local_paths['weights']), *map(str, kwargs.get('part_weights') or [])]
        wanted_models = [str(paths['model:' + name]) for name in case['members']]
        if (engine != 'neuronte' or mode != 'nrt-' + job['case'] or
                list(cell) != job['cell'] or local_args.raster_calls != job['calls'] or
                actual_models != wanted_models or case['kind'] != 'bundle'):
            raise ValueError('native invocation differs from frozen bundle schedule')
        # The previous row has now been flushed by the original measure loop.
        if index == len(rows):
            if (output / 'trials.jsonl').read_bytes() != prefix:
                raise ValueError('replayed prefix is not byte-for-byte identical')
            if fonts.digest(output / 'quality.json') != receipt['quality_sha256']:
                raise ValueError('quality revalidation differs from frozen run')
            for artifact in receipt.get('previews', []):
                if fonts.digest(output / artifact['name']) != artifact['sha256']:
                    raise ValueError('preview revalidation differs')
        if index < len(rows):
            row = dict(rows[index])
            raw = b''
            if mode not in ('nrt-open', 'nrt-route-bench'):
                raw = expected[case['members'][0], tuple(cell)]
                if mode != 'nrt-coverage':
                    raw = draw_reference(raw)
                if hashlib.sha256(raw).hexdigest() != row['output_sha256']:
                    raise ValueError('saved pixel digest differs from revalidation')
        else:
            if index == stop_after or (output / 'stop-requested').exists():
                raise CheckpointExit()
            row, raw = native(binary, local_paths, engine, mode, cell, scalars,
                              local_args, directory, **kwargs)
        index += 1
        return row, raw

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, KILIX_BENCH_ARTIFACT_DIR=str(output)))
        stack.enter_context(patch.object(fonts, 'build_worker',
                                        return_value=(Path(receipt['binary']), receipt['build'])))
        stack.enter_context(patch.object(fonts, 'run_worker', side_effect=checked_worker))
        try:
            result = decomposition.measure(args, family=family)
        except CheckpointExit:
            result = None
    data = (output / 'trials.jsonl').read_bytes()
    validate_prefix(data, plan)
    if not data.startswith(prefix):
        raise ValueError('completed measurements changed')
    if fonts.digest(output / 'quality.json') != receipt['quality_sha256']:
        raise ValueError('quality revalidation differs from frozen run')
    with (output / 'trials.jsonl').open('rb') as stream:
        os.fsync(stream.fileno())
    segment = {'started_utc': began, 'finished_utc': datetime.now(timezone.utc).isoformat(),
               'prior_trials': len(rows), 'total_trials': index, 'planned_trials': len(plan),
               'prior_sha256': receipt['journal_sha256'],
               'journal_sha256': hashlib.sha256(data).hexdigest(),
               'adapter_sha256': fonts.digest(__file__), 'complete': result is not None}
    (output / 'segment.json').write_text(json.dumps(segment, indent=2) + '\n')
    next_receipt = {**receipt, 'journal': str((output / 'trials.jsonl').resolve()),
                    'journal_sha256': segment['journal_sha256'],
                    'segments': [*receipt.get('segments', []), segment]}
    (output / 'checkpoint.json').write_text(json.dumps(next_receipt, indent=2) + '\n')
    if result is not None:
        if index != len(plan):
            raise ValueError('incomplete protocol')
        result['scheduling'] = {'segments': [{k: v for k, v in s.items() if k != 'gate'}
                                            for s in next_receipt['segments']],
                                'original_started_utc': receipt.get('original_started_utc'),
                                'note': 'Original prefix retained verbatim. Validation repeated outside timing. '
                                        'Hardware start describes this resume segment. Gaps affect host state.'}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    result = resume(json.loads(args.checkpoint.read_text()), args.output, stop_after=args.stop_after)
    return 0 if result is not None else 75


if __name__ == '__main__':
    sys.exit(main())
