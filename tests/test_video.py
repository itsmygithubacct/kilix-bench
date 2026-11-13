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
from kilix_bench.video import arm_command, decoder_arms, measure_decoder, packed_hash, seek_scenarios


class VideoTest(unittest.TestCase):
    def test_seek_baseline_is_interleaved_and_must_match_the_command(self):
        import numpy as np
        from types import SimpleNamespace

        class Stream:
            failure = ''
            def __init__(self, value): self.value = value
            def read(self): return object()
            def copy(self, frame, out): out.fill(self.value)
            def close(self): pass

        candidate = SimpleNamespace(open=lambda *args, **kwargs: Stream(7))
        baseline = SimpleNamespace(open=lambda *args, **kwargs: Stream(7))
        module = SimpleNamespace(MediaDecoder=lambda path: candidate,
                                 decoder_command=lambda *args, **kwargs: ['ffmpeg', '-'])
        base_module = SimpleNamespace(MediaDecoder=lambda path: baseline)
        digest = packed_hash(np.full((3, 1, 2), 7, np.uint8), 2, 1)
        command = dict(first_frame_seconds=.01, cpu_seconds=.02, hashes=[digest])
        with tempfile.TemporaryDirectory() as temporary:
            home, base = Path(temporary)/'candidate', Path(temporary)/'baseline'
            for tree in (home, base):
                (tree/'build').mkdir(parents=True)
                (tree/'build/libneuronte-media.so').touch()
            with patch('kilix_bench.video.load_module', return_value=base_module), \
                    patch('kilix_bench.video.measure_decoder', return_value=command):
                clip = dict(fps=30, chroma=None, duration=40)
                scenarios, rows = seek_scenarios(module, home, 'input', 2, 1, clip, 20, 2, 1, base)
                self.assertEqual({row['name'] for row in scenarios},
                                 {'seek_command', 'seek_library', 'seek_baseline'})
                self.assertEqual(len(rows), 27)
                self.assertEqual(sum(row['warmup'] for row in rows), 9)
                self.assertTrue(all(row['hash'] == digest for row in rows))
                self.assertTrue(all(row['samples'] == 6 for row in scenarios))
                first = [(r['start'], r['engine']) for r in rows if r['repetition'] == 0]
                second = [(r['start'], r['engine']) for r in rows if r['repetition'] == 1]
                self.assertEqual(second, first[1:] + first[:1])
                baseline.open = lambda *args, **kwargs: Stream(8)
                with self.assertRaisesRegex(ValueError, 'different first frames'):
                    seek_scenarios(module, home, 'input', 2, 1, clip, 20, 1, 0, base)

    def test_exact_fragmented_child_frames(self):
        command = [sys.executable, '-c',
                   "import os; os.write(1,b'ab'); os.write(1,b'cdefghijkl')"]
        result = measure_decoder(command, 2, 1, 2)
        self.assertEqual(result['hashes'], [hashlib.sha256(x).hexdigest()
                                          for x in (b'abcdef', b'ghijkl')])
        self.assertGreater(result['cpu_seconds'], 0)
        self.assertGreater(result['max_rss_kib'], 0)
        self.assertGreaterEqual(result['wall_seconds'], result['first_frame_seconds'])

    def test_planar_frames_hash_as_packed_rgb(self):
        # G plane, B plane, R plane of a 2x1 frame; packed it is R G B R G B.
        command = [sys.executable, '-c', "import os; os.write(1,b'gGbBrR')"]
        result = measure_decoder(command, 2, 1, 1, planar=True)
        self.assertEqual(result['hashes'], [hashlib.sha256(b'rgbRGB').hexdigest()])

    def test_library_arm_passes_the_commands_own_arguments(self):
        class Module:
            @staticmethod
            def decoder_command(video, width, height, fps, *, start=0., chroma=None, decoder_threads=0,
                                output_threads=1, pix_fmt='rgb24'):
                return ['ffmpeg', '-filter_threads', '1', '-threads:v', str(decoder_threads),
                        '-ss', f'{start:.3f}', '-i', str(video), '-vf', f'fps={fps},scale={width}:{height}',
                        '-threads:v', str(output_threads), '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-']
        home = Path('/tree')
        library = dict(name='library_0_gbrp', engine='library', decoder_threads=0, output_threads=None,
                       pix_fmt='gbrp')
        self.assertEqual(arm_command(Module, home, library, 'v.mp4', 945, 531, 30., 20., None, 240),
                         ['/tree/build/neuronte-media-decode', 'v.mp4', '20.000', 'fps=30.0,scale=945:531',
                          'gbrp', '0', '1', '240'])
        command = dict(library, engine='command', output_threads=1, name='decode_0_raw_1_gbrp')
        self.assertEqual(arm_command(Module, home, command, 'v.mp4', 945, 531, 30., 20., None, 240),
                         ['ffmpeg', '-filter_threads', '1', '-threads:v', '0', '-ss', '20.000', '-i', 'v.mp4',
                          '-vf', 'fps=30.0,scale=945:531', '-threads:v', '1', '-f', 'rawvideo',
                          '-pix_fmt', 'gbrp', '-frames:v', '240', '-'])

    def test_decoder_arms_follow_what_the_tree_supports(self):
        class Old:
            @staticmethod
            def decoder_command(video, width, height, fps, *, start=0., chroma=None, decoder_threads=0,
                                output_threads=1):
                return []

        class New:
            @staticmethod
            def decoder_command(video, width, height, fps, *, start=0., chroma=None, decoder_threads=0,
                                output_threads=1, pix_fmt='rgb24'):
                return []
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            names = lambda module: [arm['name'] for arm in decoder_arms(module, home)]
            pairs = ['decode_0_raw_0', 'decode_0_raw_1', 'decode_1_raw_1', 'decode_2_raw_1', 'decode_4_raw_1']
            self.assertEqual(names(Old), pairs)
            self.assertEqual(names(New), pairs + ['decode_0_raw_1_gbrp'])
            (home / 'build').mkdir()
            (home / 'build/neuronte-media-decode').touch()
            self.assertEqual(names(New), pairs + ['decode_0_raw_1_gbrp', 'library_0_rgb24', 'library_0_gbrp'])

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
