import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import pty
import signal
import socket
import subprocess
import sys
import tempfile
import termios
import time
import unittest
from unittest.mock import patch

from kilix_bench import mouse

try:
    from kitty_input import Input
except ImportError:
    Input = None


class MappingAndReportTests(unittest.TestCase):
    def test_exact_inverse_mapping_all_child_rows(self):
        for y in range(2016):
            axis = mouse.inverse_axis(y+64, 2144, 2160)
            self.assertEqual(axis*2143//2159, y+64)
        self.assertEqual(mouse.inverse_axis(24, 2160, 2160), 24)
        with self.assertRaises(ValueError):
            mouse.inverse_axis(2144, 2144, 2160)
        with self.assertRaises(ValueError):
            mouse.inverse_axis(1, 3000, 20)

    def test_plan_deterministic_unique_keys_and_transitions(self):
        rows = mouse.plan(120, [60,125,250,500,1000], 3000,1500)
        self.assertEqual(rows, mouse.plan(120,[60,125,250,500,1000],3000,1500))
        self.assertEqual(len({mouse.key(r) for r in rows}), len(rows))
        held = 0
        for row in rows:
            if row['action'] == 1:
                self.assertEqual(held, 0)
                held = row['button']
            elif row['action'] == 2:
                self.assertEqual(held, row['button'])
                held = 0
            elif row['action'] == 3:
                self.assertEqual(held, row['button'])
        self.assertEqual(held, 0)

    def test_correlation_keeps_missing_duplicate_order_and_contamination(self):
        rows = mouse.plan(2, [60], 3000,1500)[:3]
        for row in rows:
            row['injected_ns'] = 100
        def receipt(row, timestamp=200):
            return dict(row, kind='mouse', receipt_ns=timestamp)
        receipts = [receipt(rows[1]), receipt(rows[0]), receipt(rows[1]),
                    dict(kind='key', receipt_ns=200), receipt(rows[2], 90)]
        matched, unexpected, duplicates, order, negative = mouse.correlate(rows, receipts)
        self.assertEqual(set(matched), {0,1})
        self.assertEqual((unexpected, duplicates, order, negative), ([3],[2],[1],[4]))

    def test_mock_run_failure_artifacts_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            output = str(Path(temp)/'run')
            app = {'boot_id': mouse.clock_id(), 'mode':'quiet', 'dependency':{'revision': mouse.PIN}}
            device = {'boot_id':mouse.clock_id(), 'axis_width':3840,'axis_height':2160}
            calls = []
            def fake(path, op, **kw):
                calls.append(op)
                if op == 'info':
                    return app if path == 'app' else device
                if op == 'drain':
                    return {'events':[], 'overflow':0}
                if op == 'emit':
                    return {'injected_ns':time.monotonic_ns(),'write_done_ns':time.monotonic_ns()}
                return {'ok':True}
            args = argparse.Namespace(output=output,app='app',injector='inject',width=3000,height=1500,
                origin_x=0,origin_y=64,mapped_width=3840,mapped_height=2144,count=1,rates=[60],
                backend='test',variant='baseline',revision='abc',timeout=.01)
            with patch.object(mouse,'rpc',side_effect=fake), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(mouse.run(args), 1)
            value = json.loads((Path(output)/'report.json').read_text())
            self.assertEqual(value['status'],'failed')
            self.assertEqual(value['scenarios'][0]['received'],0)
            self.assertEqual(value['scenarios'][0]['missing_seq'],[0,1,2])
            self.assertEqual(calls[-1],'release')
            self.assertEqual((Path(output)/'run.json').stat().st_mode & 0o777,0o600)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mouse.report(argparse.Namespace(input=output)),1)
            (Path(output)/'run.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                mouse.report(argparse.Namespace(input=output))

    def test_successful_runner_and_late_timeout_accounting(self):
        for late in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                output = str(Path(temp)/'run')
                pending = []
                sent = []
                def fake(path, op, **kw):
                    if op == 'info':
                        return ({'boot_id':mouse.clock_id(),'mode':'quiet','dependency':{'revision':mouse.PIN}}
                                if path == 'app' else {'boot_id':mouse.clock_id(),'axis_width':3840,'axis_height':2160})
                    if op == 'emit':
                        now = time.monotonic_ns()
                        pending.append(dict(kw,kind='mouse',receipt_ns=now+100))
                        sent.append(now)
                        return {'injected_ns':now,'write_done_ns':now+50}
                    if op == 'drain':
                        if late and pending and time.monotonic_ns()-sent[-1] < 15_000_000:
                            return {'events':[],'overflow':0}
                        value = list(pending)
                        pending.clear()
                        return {'events':value,'overflow':0}
                    return {'ok':True}
                args = argparse.Namespace(output=output,app='app',injector='inject',width=3000,height=1500,
                    origin_x=0,origin_y=64,mapped_width=3840,mapped_height=2144,count=1,rates=[10000],
                    backend='test',variant='baseline',revision='abc',timeout=.01)
                with patch.object(mouse,'rpc',side_effect=fake), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(mouse.run(args),int(late))
                report = json.loads((Path(output)/'report.json').read_text())
                if late:
                    self.assertEqual(report['scenarios'][0]['received'],1)
                    self.assertEqual(report['failures'][0]['type'],'receipt_timeout')
                else:
                    self.assertTrue(all(not row['missing_seq'] for row in report['scenarios']))
                    self.assertTrue(all(row['p99_ms'] == .0001 for row in report['scenarios']))

    def test_uinput_rejects_keyboard_or_outside_coordinates_without_writes(self):
        device = mouse.UInput(3840,2160)
        with patch.object(os,'write') as write:
            for request in ({'axis_x':0,'axis_y':0,'action':1,'button':30},
                            {'axis_x':3840,'axis_y':0,'action':3}):
                with self.assertRaises(ValueError):
                    device.emit(request)
            write.assert_not_called()

    def test_device_destroy_even_when_ioctl_fails(self):
        device = mouse.UInput(3840,2160)
        device.fd, device.created = 999, True
        with patch('fcntl.ioctl',side_effect=OSError), patch.object(os,'close') as close:
            with self.assertRaises(OSError):
                device.__exit__()
            close.assert_called_once_with(999)
            self.assertIsNone(device.fd)

    def test_existing_socket_not_unlinked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sock'
            path.write_text('occupied')
            with self.assertRaises(OSError), mouse.server(path):
                pass
            self.assertEqual(path.read_text(),'occupied')


@unittest.skipIf(Input is None,'install pinned mouse extra or put kitty-input/python/src on PYTHONPATH')
class ParserAndTerminalTests(unittest.TestCase):
    def test_pin(self):
        self.assertEqual(mouse.dependency()['revision'], mouse.PIN)

    def test_every_fragment_boundary_and_combined_events(self):
        wire = b'\x1b[<35;2801;1001M\x1b[<0;2802;1002M\x1b[<0;2802;1002m\x1b[<64;2803;1003M'
        for split in range(len(wire)+1):
            decoder = Input(pixel_coordinates=True)
            rows = mouse.decode_chunk(decoder,wire[:split],10)
            count = len(rows)
            rows += mouse.decode_chunk(decoder,wire[split:],20)
            self.assertEqual([r['action'] for r in rows],[3,1,2,4])
            self.assertEqual([r['x'] for r in rows],[2800,2801,2801,2802])
            self.assertTrue(all(r['pixel_coordinates'] for r in rows))
            self.assertTrue(all(r['receipt_ns']==20 for r in rows[count:]))

    def test_malformed_resync_and_nonmouse_redaction(self):
        rows = mouse.decode_chunk(Input(pixel_coordinates=False),b'\x1b[<x;3;4M\x1b[<35;3;4Msecret',20)
        mice = [r for r in rows if r['kind']=='mouse']
        self.assertEqual(len(mice),1)
        self.assertFalse(mice[0]['pixel_coordinates'])
        self.assertNotIn('secret',json.dumps(rows))

    def test_raw_mode_restored_on_error(self):
        master, slave = pty.openpty()
        try:
            before = termios.tcgetattr(slave)
            with self.assertRaises(RuntimeError):
                with mouse.terminal(slave):
                    self.assertFalse(termios.tcgetattr(slave)[3] & termios.ICANON)
                    raise RuntimeError()
            self.assertEqual(before,termios.tcgetattr(slave))
            wire = os.read(master,4096)
            self.assertIn(b'\x1b[?1016h',wire)
            self.assertIn(b'\x1b[?1016l',wire)
        finally:
            os.close(master)
            os.close(slave)

    def test_app_pty_quiet_visible_and_signal_cleanup(self):
        for mode in ('quiet','visible'):
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp)/'app.sock')
                pid, master = pty.fork()
                if pid == 0:
                    os.execv(sys.executable,[sys.executable,'-m','kilix_bench.mouse','app','--socket',path,'--mode',mode,'--lifetime','10'])
                try:
                    deadline = time.monotonic()+5
                    while not Path(path).exists() and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertEqual(mouse.rpc(path,'info')['mode'],mode)
                    os.read(master,4096)  # Entry controls.
                    os.write(master,b'\x1b[<35;20;30M\x1b[<35;21;31M')
                    time.sleep(.05)
                    rows = mouse.rpc(path,'drain')['events']
                    self.assertEqual([r['x'] for r in rows],[19,20])
                    ready = __import__('select').select([master],[],[],.05)[0]
                    self.assertEqual(bool(ready),mode=='visible')
                    if ready:
                        self.assertIn(b'mouse',os.read(master,4096))
                    os.kill(pid,signal.SIGTERM)
                    os.waitpid(pid,0)
                    pid = None
                    self.assertFalse(Path(path).exists())
                    self.assertIn(b'\x1b[?1016l',os.read(master,4096))
                finally:
                    if pid:
                        os.kill(pid,signal.SIGKILL)
                        os.waitpid(pid,0)
                    os.close(master)
