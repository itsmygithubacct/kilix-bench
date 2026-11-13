"""A completed batched checkpoint can regenerate exports without training again."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BatchCheckpointTests(unittest.TestCase):
    def test_completed_resume_restores_exports_and_rejects_changed_identity(self):
        home = os.environ.get('NEURONTE_HOME')
        if not home or importlib.util.find_spec('torch') is None:
            self.skipTest('NEURONTE_HOME and optional PyTorch required')
        with tempfile.TemporaryDirectory(prefix='braille-batch-test-') as directory:
            root = Path(directory)
            command = [sys.executable, '-m', 'kilix_bench.braille_batch_train', directory,
                       '--neuronte-home', home, '--layout', 'dots', '--patterns', '2',
                       '--steps', '2', '--device', 'cpu', '--seeds', '0']
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            before = {p.name: p.read_bytes() for p in root.glob('*.json')}
            self.assertEqual(len(before), 8)
            (root / next(iter(before))).unlink()
            resumed = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=120)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual({p.name: p.read_bytes() for p in root.glob('*.json')}, before)
            changed = list(command)
            changed[changed.index('--steps')+1] = '3'
            rejected = subprocess.run(changed + ['--resume'], capture_output=True, text=True, timeout=120)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('checkpoint identity differs', rejected.stderr)
            self.assertEqual({p.name: p.read_bytes() for p in root.glob('*.json')}, before)


if __name__ == '__main__':
    unittest.main()
