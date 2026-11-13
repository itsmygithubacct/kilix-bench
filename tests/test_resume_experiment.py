"""Check native interrupted/uninterrupted protocol equality and refusal paths."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from kilix_bench import decomposition, fonts, neuronte

spec = importlib.util.spec_from_file_location('resume_experiment',
        Path(__file__).resolve().parents[1] / 'tools/resume_experiment.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class NativeResumeTests(unittest.TestCase):
    def test_native_prefix_resume_preserves_all_ten_repetitions(self):
        home = os.environ.get('NEURONTE_HOME')
        if not home:
            self.skipTest('NEURONTE_HOME required')
        with tempfile.TemporaryDirectory(prefix='resume-test-') as directory:
            root = Path(directory)
            shutil.copyfile(Path(home) / 'weights/neuronte-split-v5.nrt', root / 'control.nrt')
            codes = [32, 65, 0x391, 0x410, 0x2588]
            targets = {}
            for w, h in [(2, 4), (3, 5)]:
                p = root / f'{w}x{h}.npy'
                np.save(p, np.ones((len(codes), h, w), dtype=np.float32))
                targets[f'{w}x{h}'] = {'file': p.name, 'sha256': fonts.digest(p)}
            manifest = {'schema': 'kilix-bench.expert-experiment/v1', 'corpus': codes,
                        'models': {'control': {'file': 'control.nrt', 'sha256': fonts.digest(root / 'control.nrt')}},
                        'targets': targets, 'teacher': {'fixture': 'ones'}, 'runtime_revision': 'fixture',
                        'cases': [{'name': 'control', 'kind': 'bundle', 'members': ['control'],
                                   'routes': [0, 0, 3, 3, 4]}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            parser = argparse.ArgumentParser()
            fonts.add_arguments(parser)
            neuronte.add_arguments(parser)
            parser.add_argument('--repeats', type=int, default=10)
            parser.add_argument('--warmup', type=int, default=1)
            args = parser.parse_args(['--expert-experiment', str(root / 'manifest.json'),
                                      '--cells', '2x4,3x5', '--expert-passes', '4'])
            binary, build = fonts.build_worker(root)
            full = root / 'full'
            full.mkdir()
            with patch.dict(os.environ, KILIX_BENCH_ARTIFACT_DIR=str(full)), \
                    patch.object(fonts, 'build_worker', return_value=(binary, build)):
                original = decomposition.measure(args, family='expert')
            prefix = root / 'prefix.jsonl'
            lines = (full / 'trials.jsonl').read_bytes().splitlines(keepends=True)
            prefix.write_bytes(b''.join(lines[:17]))
            receipt = {'family': 'expert', 'args': vars(args), 'binary': str(binary), 'build': build,
                       'journal': str(prefix), 'journal_sha256': fonts.digest(prefix),
                       'quality_sha256': fonts.digest(full / 'quality.json'),
                       'files': [{'path': str(p), 'sha256': fonts.digest(p)}
                                 for p in sorted(adapter.required_files(
                                     decomposition.inputs(args, family='expert'), binary))]}
            native = fonts.run_worker
            invoked = []

            def tracking(*a, **kw):
                if a[3] not in ('nrt-routes', 'nrt-validate'):
                    invoked.append((a[3][4:], list(a[4]), a[6].raster_calls))
                return native(*a, **kw)

            with patch.object(fonts, 'run_worker', side_effect=tracking):
                self.assertIsNone(adapter.resume(receipt, root / 'part', stop_after=38))
                continuation = json.loads((root / 'part/checkpoint.json').read_text())
                result = adapter.resume(continuation, root / 'final')
            _, plan = adapter.schedule(args, manifest)
            self.assertEqual(len(original['trials']), 88)
            self.assertEqual(invoked, [(j['case'], j['cell'], j['calls']) for j in plan[17:]])
            self.assertEqual(len(result['trials']), len(original['trials']))
            self.assertEqual((root / 'final/trials.jsonl').read_bytes()[:len(prefix.read_bytes())], prefix.read_bytes())
            self.assertTrue((root / 'final/trials.jsonl').read_bytes().startswith((root / 'part/trials.jsonl').read_bytes()))
            for before, after in zip(original['trials'], result['trials']):
                for key in ('model', 'cell', 'case', 'repeat', 'order', 'calls', 'coverage_calls',
                            'draw_calls', 'priming_calls', 'output_sha256', 'route_checksum'):
                    self.assertEqual(before.get(key), after.get(key), key)
            self.assertTrue(all(s['samples'] == 10 for s in result['scenarios']))
            # A complete checkpoint produces the same rows and starts no timed job.
            invoked.clear()
            with patch.object(fonts, 'run_worker', side_effect=tracking):
                again = adapter.resume(json.loads((root / 'final/checkpoint.json').read_text()), root / 'again')
            self.assertEqual(invoked, [])
            self.assertEqual(again['trials'], result['trials'])
            self.assertEqual((root / 'again/trials.jsonl').read_bytes(), (root / 'final/trials.jsonl').read_bytes())
            with self.assertRaisesRegex(ValueError, 'journal changed'):
                adapter.resume({**receipt, 'journal_sha256': '0' * 64}, root / 'bad')
            with self.assertRaisesRegex(ValueError, 'provenance changed'):
                adapter.resume({**receipt, 'files': [{'path': str(binary), 'sha256': '0' * 64}]}, root / 'bad2')
            for omitted in (Path(decomposition.__file__).resolve(),
                            decomposition.inputs(args, family='expert')['library'],
                            root / 'manifest.json'):
                incomplete = {**receipt, 'files': [item for item in receipt['files']
                                                  if Path(item['path']) != omitted]}
                with self.assertRaisesRegex(ValueError, 'missing required provenance'):
                    adapter.resume(incomplete, root / 'incomplete')
            bad = json.loads(lines[0]); bad['order'] += 1
            with self.assertRaisesRegex(ValueError, 'exact seeded job order'):
                adapter.validate_prefix((json.dumps(bad) + '\n').encode(), plan)
            with self.assertRaisesRegex(ValueError, 'partial journal line'):
                adapter.validate_prefix(prefix.read_bytes()[:-1], plan)


if __name__ == '__main__':
    unittest.main()
