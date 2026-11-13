"""Verify the measurement boundary, failures, and sandbox for video runs."""
import base64
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from kilix_bench.isolate import sandbox
from kilix_bench.terminal import ConsumerTerm
from kilix_bench.video import measure_decoder


class VideoTest(unittest.TestCase):
    def test_exact_fragmented_child_frames(self):
        command = [sys.executable, '-c',
                   "import os; os.write(1,b'ab'); os.write(1,b'cdefghijkl')"]
        result = measure_decoder(command, 2, 1, 2)
        self.assertEqual(result['hashes'], [hashlib.sha256(x).hexdigest()
                                          for x in (b'abcdef', b'ghijkl')])
        self.assertGreater(result['cpu_seconds'], 0)
        self.assertGreater(result['max_rss_kib'], 0)
        self.assertGreaterEqual(result['wall_seconds'], result['first_frame_seconds'])

    def test_bad_frame_count_and_partial_frame(self):
        for raw, count in ((b'abcdef', 2), (b'abcdefg', 1), (b'abcdefghijkl', 1)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                measure_decoder([sys.executable, '-c', f'import os; os.write(1,{raw!r})'], 2, 1, count)

    def test_timeout_reaps_child(self):
        with tempfile.TemporaryDirectory() as root:
            pidfile = Path(root) / 'pid'
            script = 'import os,sys,time; open(sys.argv[1],"w").write(str(os.getpid())); time.sleep(30)'
            with self.assertRaises(TimeoutError):
                measure_decoder([sys.executable, '-c', script, str(pidfile)], 2, 1, 1, timeout=.3)
            self.assertTrue(pidfile.exists())
            with self.assertRaises(ChildProcessError):
                os.waitpid(int(pidfile.read_text()), os.WNOHANG)

    def test_timeout_also_bounds_child_that_closes_stdout(self):
        with self.assertRaises(TimeoutError):
            measure_decoder([sys.executable, '-c', 'import os,time; os.close(1); time.sleep(30)'],
                            2, 1, 1, timeout=.1)

    def test_consumer_checks_pixels_before_acknowledging(self):
        for raw, geometry, expected_error in ((b'abcdef', (2, 1), False),
                                               (b'abcde', (2, 1), True),
                                               (b'abcdef', (1, 2), True)):
            fd, name = tempfile.mkstemp(prefix='kilix-bench-test-', dir='/dev/shm')
            try:
                os.write(fd, raw)
                os.close(fd)
                encoded = base64.b64encode(Path(name).name.encode()).decode()
                terminal = ConsumerTerm(verify_frames=True, expected_size=(2, 1))
                payload = f'\x1b_Ga=T,t=s,f=24,s={geometry[0]},v={geometry[1]};{encoded}\x1b\\'.encode()
                if expected_error:
                    with self.assertRaises(ValueError):
                        terminal.write(payload)
                    self.assertEqual(terminal.frames, 0)
                else:
                    terminal.write(payload)
                    self.assertEqual(terminal.frames, 1)
                    self.assertEqual(terminal.last_hash, hashlib.sha256(raw).hexdigest())
                self.assertFalse(Path(name).exists())
            finally:
                Path(name).unlink(missing_ok=True)

    def test_video_sandbox_needs_no_desktop_state(self):
        with patch('kilix_bench.isolate.state_library', side_effect=AssertionError('desktop dependency')):
            env, root = sandbox(require_state=False)
        try:
            self.assertTrue(env['HOME'].startswith(root + '/'))
            self.assertEqual(env['KILIX_STATE_LIBRARY'], '')
            self.assertNotIn('KITTY_LISTEN_ON', env)
            self.assertNotIn('KILIX_RC_PASSWORD_FILE', env)
        finally:
            shutil.rmtree(root)

    def test_missing_state_fails_before_allocating_sandbox(self):
        with patch('kilix_bench.isolate.state_library', return_value=''), \
                patch('kilix_bench.isolate.tempfile.mkdtemp') as create:
            with self.assertRaises(SystemExit):
                sandbox('/unused')
            create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
