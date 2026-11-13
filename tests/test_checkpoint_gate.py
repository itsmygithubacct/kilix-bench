"""Atomic gate installation finishes an active batch before releasing a lock."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


def until(predicate):
    limit = time.monotonic() + 10
    while not predicate():
        if time.monotonic() > limit:
            raise AssertionError('fixture timed out')
        time.sleep(.01)


class GateTests(unittest.TestCase):
    def test_executing_inode_finishes_next_job_waits_and_lock_releases(self):
        if not shutil.which('cc'):
            self.skipTest('C compiler required')
        with tempfile.TemporaryDirectory(prefix='gate-test-') as tmp:
            root = Path(tmp)
            source = root / 'fixture.c'
            source.write_text('#include <stdio.h>\n#include <unistd.h>\nint main(int n,char **v) {'
                              'FILE *s=fopen(v[1],"w");fputs("started",s);fclose(s);'
                              'usleep(300000);puts("completed batch");return 0;}\n')
            binary = root / 'worker'
            subprocess.run(['cc', str(source), '-o', str(binary)], check=True)
            gate = Path(__file__).resolve().parents[1] / 'tools/checkpoint_gate.py'
            config = root / 'gate.json'
            config.write_text(json.dumps({'ready': str(root / 'ready'), 'authorization': str(root / 'authorize')}))
            stub = root / 'gate-stub'
            stub.write_text('#!' + sys.executable + '\nimport runpy,sys\n'
                            'sys.exit(runpy.run_path(' + repr(str(gate)) + ')["gate"](' + repr(str(config)) + '))\n')
            stub.chmod(0o700)
            driver = root / 'driver.py'
            driver.write_text('import fcntl,subprocess,sys\nfrom pathlib import Path\n'
                              'r=Path(sys.argv[1])\nwith (r/"lock").open("w") as lock:\n'
                              ' fcntl.flock(lock,fcntl.LOCK_EX)\n'
                              ' with (r/"journal").open("w") as journal:\n'
                              '  for i in range(5):\n'
                              '   p=subprocess.run([str(r/"worker"),str(r/"started"),str(i)],capture_output=True,text=True)\n'
                              '   if p.returncode:sys.exit(p.returncode)\n'
                              '   journal.write(p.stdout);journal.flush()\n')
            p = subprocess.Popen([sys.executable, str(driver), str(root)])
            try:
                until(lambda: (root / 'started').exists())
                os.replace(stub, binary)
                until(lambda: (root / 'ready').exists())
                ready = json.loads((root / 'ready').read_text())
                self.assertEqual(ready['ppid'], p.pid)
                self.assertEqual(ready['argv'][-1], '1')
                self.assertFalse(ready['native_started'])
                prefix = (root / 'journal').read_bytes()
                self.assertEqual(prefix, b'completed batch\n')
                self.assertIsNone(p.poll())
                self.assertNotEqual(subprocess.run(['flock', '-n', str(root / 'lock'), 'true']).returncode, 0)
                (root / 'authorize').write_text(json.dumps({'gate_pid': ready['pid'], 'action': 'checkpoint-saved-exit'}))
                self.assertEqual(p.wait(timeout=10), 75)
                self.assertEqual((root / 'journal').read_bytes(), prefix)
                self.assertEqual(subprocess.run(['flock', '-n', str(root / 'lock'), 'true']).returncode, 0)
            finally:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
