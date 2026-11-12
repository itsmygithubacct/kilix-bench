"""Machine-local terminal mouse sessions. No hardware is touched on import."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import struct
import sys
import termios
import time
import tty

from .report import _private, _replace, assert_no_local_paths

PIN = '0f009064f3f32c7ea972a3d90de1522eda409e89'
SCHEMA = 'kilix.mouse.v1'
MAX_EVENTS = 100000


def dependency():
    import kitty_input
    root = Path(kitty_input.__file__).parent
    expected = json.loads(Path(__file__).with_name('mouse-parser-pin.json').read_text())
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('kitty-input source differs from required revision ' + PIN)
    return {'name': 'kitty-input', 'version': kitty_input.__version__, 'revision': PIN,
            'source_sha256': expected}


def benchmark_source():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def clock_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def send_json(conn, value):
    conn.sendall(json.dumps(value, separators=(',', ':')).encode() + b'\n')


def recv_json(conn):
    data = bytearray()
    while not data.endswith(b'\n'):
        chunk = conn.recv(65536)
        if not chunk:
            raise ConnectionError('control peer closed before response')
        data.extend(chunk)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError('control message exceeds limit')
    return json.loads(data)


def rpc(path, op, **fields):
    with socket.socket(socket.AF_UNIX) as conn:
        conn.settimeout(5)
        conn.connect(str(path))
        send_json(conn, {'op': op, **fields})
        result = recv_json(conn)
    if 'error' in result:
        raise ValueError(result['error'])
    return result


@contextlib.contextmanager
def server(path, uid=None):
    path = Path(path)
    parent = path.parent.stat()
    owner = os.getuid() if uid is None else uid
    if parent.st_uid != owner or parent.st_mode & 0o077:
        raise ValueError('control socket requires a private directory owned by its client')
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(path))  # Never unlink an existing server.
        try:
            os.chmod(path, 0o600)
            if uid is not None:
                os.chown(path, uid, -1)
            sock.listen(4)
            yield sock
        finally:
            path.unlink(missing_ok=True)


@contextlib.contextmanager
def termination():
    def stop(signum, frame):
        raise InterruptedError('termination signal')
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


@contextlib.contextmanager
def terminal(fd):
    from kitty_input.terminal import MOUSE_ON, MOUSE_OFF
    old = termios.tcgetattr(fd)
    modes = (1000, 1002, 1003, 1006, 1016)
    try:
        tty.setraw(fd)
        os.write(fd, (''.join(f'\x1b[?{m}s' for m in modes) + MOUSE_ON + '\x1b[?1016h').encode())
        yield
    finally:
        try:
            os.write(fd, ('\x1b[?1016l' + MOUSE_OFF + ''.join(f'\x1b[?{m}r' for m in reversed(modes))).encode())
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old)


def decode_chunk(decoder, data, receipt_ns):
    from kitty_input import MouseEvent
    decoder.feed(data)
    result = []
    for event in decoder:
        row = {'receipt_ns': receipt_ns, 'parsed_ns': time.monotonic_ns(), 'kind': event.kind}
        if isinstance(event, MouseEvent):
            row.update(dataclasses.asdict(event))
        # Do not record keyboard/paste text in artifacts.
        result.append(row)
    return result


def app(args):
    from kitty_input import Input
    pin = dependency()
    events = []
    overflow = 0
    decoder = Input(pixel_coordinates=True)
    with termination(), open('/dev/tty', 'r+b', buffering=0) as stream:
        fd = stream.fileno()
        with terminal(fd), server(args.socket) as listener:
            deadline = time.monotonic() + args.lifetime
            while time.monotonic() < deadline:
                ready, _, _ = select.select([fd, listener], [], [], min(.1, max(0, deadline-time.monotonic())))
                if fd in ready:
                    data = os.read(fd, 65536)
                    received = time.monotonic_ns()
                    if not data:
                        break
                    rows = decode_chunk(decoder, data, received)
                    for row in rows:
                        if len(events) < MAX_EVENTS:
                            events.append(row)
                        else:
                            overflow += 1
                        if args.mode == 'visible' and row['kind'] == 'mouse':
                            os.write(fd, f'\x1b7\x1b[1;1Hmouse {received % 1000000000:09d} {row["x"]:5d},{row["y"]:5d}\x1b8'.encode())
                if listener in ready:
                    with listener.accept()[0] as conn:
                        conn.settimeout(1)
                        try:
                            req = recv_json(conn)
                            if req['op'] == 'info':
                                size = os.get_terminal_size(fd)
                                reply = {'schema': SCHEMA, 'boot_id': clock_id(), 'mode': args.mode,
                                         'dependency': pin, 'benchmark_source_sha256': benchmark_source(),
                                         'columns': size.columns, 'rows': size.lines}
                            elif req['op'] == 'drain':
                                reply = {'events': events, 'overflow': overflow}
                                send_json(conn, reply)
                                events = []
                                overflow = 0
                                continue
                            elif req['op'] == 'stop':
                                send_json(conn, {'ok': True})
                                break
                            else:
                                raise ValueError('unknown app operation')
                            send_json(conn, reply)
                        except (ValueError, KeyError, OSError) as exc:
                            try:
                                send_json(conn, {'error': type(exc).__name__})
                            except OSError:
                                pass
    return 0


def inverse_axis(pixel, mapped_size, axis_size):
    """Inverse floor(axis*(mapped_size-1)/(axis_size-1)), exact or fail."""
    if not 0 <= pixel < mapped_size or min(mapped_size, axis_size) < 2:
        raise ValueError('coordinate outside mapped bounds')
    axis = (pixel * (axis_size - 1) + mapped_size - 2) // (mapped_size - 1)
    if axis * (mapped_size - 1) // (axis_size - 1) != pixel:
        raise ValueError('target pixel is unreachable with these absolute axes')
    return axis


class UInput:
    """Linux uinput ABI; mouse capabilities only. Legacy setup is widely supported."""
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.fd = None
        self.created = False
        self.name = f'kilix-bench-absolute-{os.getpid()}'

    def __enter__(self):
        import fcntl
        self.fd = os.open('/dev/uinput', os.O_WRONLY | os.O_NONBLOCK)
        try:
            for ev in (1, 2, 3):  # EV_KEY, EV_REL, EV_ABS
                fcntl.ioctl(self.fd, 0x40045564, ev)
            for key in (272, 273, 274):  # BTN_LEFT, RIGHT, MIDDLE only
                fcntl.ioctl(self.fd, 0x40045565, key)
            for axis in (0, 1):
                fcntl.ioctl(self.fd, 0x40045567, axis)
            for axis in (6, 8):  # REL_HWHEEL, REL_WHEEL
                fcntl.ioctl(self.fd, 0x40045566, axis)
            fcntl.ioctl(self.fd, 0x4004556e, 0)  # INPUT_PROP_POINTER
            maxima = [self.width - 1, self.height - 1] + [0] * 62
            setup = struct.pack('80sHHHHI', self.name.encode(), 3, 0x1209, 1, 1, 0)
            setup += struct.pack('256i', *(maxima + [0] * 192))
            if os.write(self.fd, setup) != len(setup):
                raise OSError('short device setup write')
            fcntl.ioctl(self.fd, 0x5501)
            self.created = True
            buf = bytearray(128)
            fcntl.ioctl(self.fd, 0x8080552c, buf, True)
            self.sysname = buf.split(b'\0')[0].decode()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def emit(self, req):
        x, y = req['axis_x'], req['axis_y']
        if not isinstance(x, int) or not isinstance(y, int) or not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError('invalid absolute target')
        action = req['action']
        button = req.get('button', 0)
        codes = {1: 272, 2: 274, 3: 273}
        events = [(3, 0, x), (3, 1, y)]
        if action in (1, 2) and button in codes:
            events.append((1, codes[button], int(action == 1)))
        elif action == 4 and button in (4, 5, 6, 7):
            events.append((2, 8 if button < 6 else 6, 1 if button in (4, 7) else -1))
        elif action != 3 or button not in (0, 1, 2, 3):
            raise ValueError('invalid mouse-only action')
        events.append((0, 0, 0))
        blob = b''.join(struct.pack('llHHi', 0, 0, *e) for e in events)
        before = time.monotonic_ns()
        count = os.write(self.fd, blob)
        after = time.monotonic_ns()
        if count != len(blob):
            raise OSError('short uinput event write')
        return {'injected_ns': before, 'write_done_ns': after}

    def __exit__(self, *exc):
        import fcntl
        if self.fd is not None:
            try:
                if self.created:
                    fcntl.ioctl(self.fd, 0x5502)
            finally:
                os.close(self.fd)
                self.fd = None


def inject(args):
    if not args.allow_uinput or os.geteuid() != 0:
        raise ValueError('inject requires root and explicit --allow-uinput')
    with termination(), UInput(args.width, args.height) as device, server(args.socket, args.uid) as listener:
        info = {'schema': SCHEMA, 'boot_id': clock_id(), 'name': device.name,
                'sysname': device.sysname, 'axis_width': args.width, 'axis_height': args.height}
        print(json.dumps(info), flush=True)
        deadline = time.monotonic() + args.lifetime
        count = 0
        while time.monotonic() < deadline and count < MAX_EVENTS:
            if not select.select([listener], [], [], min(.1, max(0, deadline-time.monotonic())))[0]:
                continue
            with listener.accept()[0] as conn:
                conn.settimeout(1)
                try:
                    req = recv_json(conn)
                    if req['op'] == 'info':
                        reply = info
                    elif req['op'] == 'emit':
                        reply = device.emit(req)
                        count += 1
                    elif req['op'] == 'release':
                        for button in (1, 2, 3):
                            device.emit({'axis_x': 0, 'axis_y': 0, 'action': 2, 'button': button})
                        reply = {'ok': True}
                    elif req['op'] == 'stop':
                        send_json(conn, {'ok': True})
                        break
                    else:
                        raise ValueError('unknown injector operation')
                    send_json(conn, reply)
                except (ValueError, KeyError, OSError) as exc:
                    try:
                        send_json(conn, {'error': type(exc).__name__})
                    except OSError:
                        pass
    return 0


def key(row):
    return tuple(row.get(k, 0) for k in ('x', 'y', 'action', 'button', 'wheel_x', 'wheel_y'))


def plan(count, rates, width, height):
    # Disjoint targets throughout a session; no coordinate reuse across scenarios.
    specs = [('calibration', 0, 3, 0)] * 3
    specs += [('serialized', 0, 3, 0)] * count
    for rate in rates:
        specs += [(f'rate-{rate}', rate, 3, 0)] * count
    for button in (1, 2, 3):
        specs += [('buttons', 0, 3, 0), ('buttons', 0, 1, button), ('buttons', 0, 2, button)]
    for button in (4, 5, 6, 7):
        specs += [('wheel', 0, 3, 0), ('wheel', 0, 4, button)]
    specs += [('drag', 0, 3, 0), ('drag', 0, 1, 1)]
    specs += [('drag', 0, 3, 1)] * 12 + [('drag', 0, 2, 1)]
    available = (width - 32) * (height - 32)
    if width < 64 or height < 64 or len(specs) > available:
        raise ValueError('insufficient interior for unique targets')
    # A coprime stride covers the interior once, producing a deterministic path.
    stride = 7919
    while math.gcd(stride, available) != 1:
        stride += 2
    rows = []
    used_positions = set()
    for seq, (scenario, rate, action, button) in enumerate(specs):
        position = (0, available - 1, available // 2)[seq] if seq < 3 else seq * stride % available
        while position in used_positions:
            position = (position + stride) % available
        used_positions.add(position)
        x, y = 16 + position % (width-32), 16 + position // (width-32)
        # Button and wheel transitions occur at the previous motion target,
        # avoiding an extra unscored motion in the same SYN_REPORT.
        if action in (1, 2, 4):
            x, y = rows[-1]['x'], rows[-1]['y']
        rows.append({'seq': seq, 'scenario': scenario, 'rate_hz': rate, 'x': x, 'y': y,
                     'action': action, 'button': button, 'wheel_x': (-1 if button == 6 else 1 if button == 7 else 0),
                     'wheel_y': (-1 if button == 4 else 1 if button == 5 else 0)})
    return rows


def correlate(expected, receipts):
    lookup = {key(row): row for row in expected if 'injected_ns' in row}
    matched, contamination, duplicates, ordering, negative = {}, [], [], [], []
    last = -1
    for index, row in enumerate(receipts):
        exp = lookup.get(key(row)) if row.get('kind') == 'mouse' else None
        if exp is None:
            contamination.append(index)
            continue
        seq = exp['seq']
        if seq in matched:
            duplicates.append(index)
            continue
        latency = row['receipt_ns'] - exp['injected_ns']
        if latency < 0:
            negative.append(index)
            continue
        if seq < last:
            ordering.append(index)
        last = max(last, seq)
        matched[seq] = (row, latency)
    return matched, contamination, duplicates, ordering, negative


def summarize_run(raw):
    expected, receipts = raw['expected'], raw['receipts']
    matched, contamination, duplicates, ordering, negative = correlate(expected, receipts)
    scenarios = []
    for name in dict.fromkeys(r['scenario'] for r in expected):
        group = [r for r in expected if r['scenario'] == name]
        latencies = sorted(matched[r['seq']][1] / 1e6 for r in group if r['seq'] in matched)
        def percentile(q):
            if not latencies:
                return None
            at = (len(latencies)-1)*q
            low = int(at)
            return latencies[low] + (latencies[min(low+1, len(latencies)-1)]-latencies[low])*(at-low)
        missing = [r['seq'] for r in group if r['seq'] not in matched]
        observed = [matched[r['seq']][0]['receipt_ns'] for r in group if r['seq'] in matched]
        injected = [r['injected_ns'] for r in group if 'injected_ns' in r]
        span = max(observed)-min(injected) if observed and injected else 0
        injection_span = max(injected)-min(injected) if len(injected) > 1 else 0
        scenarios.append({'name': name, 'planned': len(group), 'injected': len(injected),
                          'received': len(latencies), 'missing_seq': missing,
                          'unacknowledged_attempts': sum('attempted_ns' in r and 'injected_ns' not in r for r in group),
                          'not_attempted': sum('attempted_ns' not in r for r in group),
                          'max_schedule_lateness_ms': max((max(0, r['injected_ns']-r['scheduled_ns'])/1e6 for r in group if 'scheduled_ns' in r and 'injected_ns' in r), default=None),
                          'loss_or_possible_coalescing': sum(r['action'] == 3 and r['seq'] in missing and 'injected_ns' in r for r in group),
                          'p50_ms': percentile(.5), 'p95_ms': percentile(.95), 'p99_ms': percentile(.99),
                          'max_ms': max(latencies) if latencies else None,
                          'receipt_throughput_hz': len(observed)*1e9/span if span > 0 else None,
                          'actual_injection_hz': (len(injected)-1)*1e9/injection_span if injection_span else None})
    failed = bool(raw['failures'] or contamination or duplicates or ordering or negative or any(s['missing_seq'] for s in scenarios))
    return {'schema': SCHEMA, 'status': 'failed' if failed else 'passed',
            'metric': 'uinput batch write-start to application completing-chunk read-return; not photon latency',
            'backend': raw['backend'], 'variant': raw['variant'], 'revision': raw['revision'],
            'mode': raw['app']['mode'], 'geometry': raw['geometry'], 'dependency': raw['app']['dependency'],
            'scenarios': scenarios, 'contamination_indices': contamination, 'duplicate_indices': duplicates,
            'out_of_order_indices': ordering, 'negative_latency_indices': negative, 'failures': raw['failures']}


def save_artifacts(directory, raw):
    report = summarize_run(raw)
    forbidden = (str(Path.home()), str(Path.cwd()), str(Path(directory).resolve()))
    assert_no_local_paths(report, forbidden)
    for name, value in (('run.json', raw), ('report.json', report)):
        _replace(directory / name, json.dumps(value, indent=2) + '\n')
    manifest = {'schema': SCHEMA, 'files': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                          for name in ('run.json', 'report.json')}}
    _replace(directory / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
    return report


def run(args):
    directory = Path(args.output)
    if directory.exists():
        raise ValueError('output directory already exists; use a fresh run directory')
    directory = _private(directory)
    app_info = rpc(args.app, 'info')
    device = rpc(args.injector, 'info')
    if app_info['boot_id'] != clock_id() or device['boot_id'] != clock_id():
        raise ValueError('app, injector and runner must share the local boot clock')
    geometry = {k: getattr(args, k) for k in ('width', 'height', 'origin_x', 'origin_y', 'mapped_width', 'mapped_height')}
    geometry.update(axis_width=device['axis_width'], axis_height=device['axis_height'])
    rows = plan(args.count, args.rates, args.width, args.height)
    for row in rows:
        row['axis_x'] = inverse_axis(row['x'] + args.origin_x, args.mapped_width, device['axis_width'])
        row['axis_y'] = inverse_axis(row['y'] + args.origin_y, args.mapped_height, device['axis_height'])
    raw = {'schema': SCHEMA, 'boot_id': clock_id(), 'backend': args.backend,
           'variant': args.variant, 'revision': args.revision, 'app': app_info, 'device': device,
           'geometry': geometry, 'expected': rows, 'receipts': [], 'failures': [],
           'benchmark_source_sha256': benchmark_source(),
           'timeout_s': args.timeout, 'clock': 'CLOCK_MONOTONIC',
           'baseline_runtime': '51d8b61', 'baseline_integration': 'c586369'}
    def drain():
        got = rpc(args.app, 'drain')
        raw['receipts'].extend(got['events'])
        if got['overflow']:
            raw['failures'].append({'type': 'app_overflow', 'count': got['overflow']})
    aborted = False
    try:
        drain()  # Preserve pre-existing contamination rather than silently resetting.
        scenario = None
        deadline = 0
        for row in rows:
            if scenario != row['scenario']:
                if scenario is not None:
                    time.sleep(args.timeout)
                    drain()
                scenario = row['scenario']
                deadline = time.monotonic_ns()
            if row['rate_hz']:
                remaining = (deadline-time.monotonic_ns())/1e9
                if remaining > 0:
                    time.sleep(remaining)
                row['scheduled_ns'] = deadline
                deadline += round(1e9/row['rate_hz'])
            row['attempted_ns'] = time.monotonic_ns()
            row.update(rpc(args.injector, 'emit', **row))
            if not row['rate_hz']:
                limit = time.monotonic() + args.timeout
                while True:
                    drain()
                    matched = correlate(rows, raw['receipts'])[0]
                    if row['seq'] in matched:
                        break
                    if time.monotonic() >= limit:
                        raw['failures'].append({'type': 'receipt_timeout', 'seq': row['seq']})
                        if scenario == 'calibration':
                            raise ValueError('calibration failed; inspect receipt coordinates and mapping')
                        break
                    time.sleep(.001)
        time.sleep(args.timeout)
        drain()
    except (Exception, KeyboardInterrupt) as exc:
        aborted = True
        raw['failures'].append({'type': 'run_aborted', 'error_class': type(exc).__name__})
        print('run aborted: ' + type(exc).__name__, file=sys.stderr)
    finally:
        if aborted:
            try:
                time.sleep(args.timeout)
                drain()
            except Exception:
                raw['failures'].append({'type': 'final_drain_failed'})
        # Release held buttons even after timeout/control failure. Cleanup receipts
        # are outside the scored window and are intentionally not drained here.
        try:
            rpc(args.injector, 'release')
        except Exception:
            raw['failures'].append({'type': 'release_failed'})
        report = save_artifacts(directory, raw)
    print(json.dumps({'status': report['status'], 'scenarios': report['scenarios']}))
    return int(aborted or report['status'] != 'passed')


def report(args):
    directory = Path(args.input)
    manifest = json.loads((directory/'manifest.json').read_text())
    for name in ('run.json', 'report.json'):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError('artifact checksum mismatch')
    value = summarize_run(json.loads((directory/'run.json').read_text()))
    assert_no_local_paths(value, (str(Path.home()), str(Path.cwd()), str(directory.resolve())))
    print(json.dumps(value, indent=2))
    return int(value['status'] != 'passed')


def bounded_int(value):
    result = int(value)
    if not 1 <= result <= 10000:
        raise argparse.ArgumentTypeError('must be between 1 and 10000')
    return result


def label(value):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value):
        raise argparse.ArgumentTypeError('use a short public label without paths')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('app', help='run directly inside the terminal being measured')
    a.add_argument('--socket', required=True)
    a.add_argument('--mode', choices=('quiet', 'visible'), required=True)
    a.add_argument('--lifetime', type=bounded_int, default=1800)
    i = sub.add_parser('inject', help='bounded root-only persistent absolute mouse device')
    i.add_argument('--socket', required=True)
    i.add_argument('--allow-uinput', action='store_true')
    i.add_argument('--uid', type=int, required=True)
    i.add_argument('--width', type=bounded_int, required=True)
    i.add_argument('--height', type=bounded_int, required=True)
    i.add_argument('--lifetime', type=bounded_int, default=1800)
    r = sub.add_parser('run', help='local deterministic calibration, path, sweep and transition cases')
    for name in ('app', 'injector', 'output'):
        r.add_argument('--'+name, required=True)
    for name in ('backend', 'variant', 'revision'):
        r.add_argument('--'+name, type=label, required=True)
    for name in ('width', 'height', 'mapped-width', 'mapped-height'):
        r.add_argument('--'+name, type=bounded_int, required=True)
    for name in ('origin-x', 'origin-y'):
        r.add_argument('--'+name, type=int, required=True)
    r.add_argument('--count', type=bounded_int, default=120)
    r.add_argument('--rates', type=bounded_int, nargs='+', default=[60, 125, 250, 500, 1000])
    r.add_argument('--timeout', type=float, default=1.0)
    p = sub.add_parser('report', help='verify manifest and print path-free report')
    p.add_argument('--input', required=True)
    c = sub.add_parser('control', help='inspect or stop an app/injector')
    c.add_argument('--socket', required=True)
    c.add_argument('op', choices=('info', 'stop', 'drain', 'release'))
    args = parser.parse_args(argv)
    if args.command == 'run' and (not .01 <= args.timeout <= 10 or len(args.rates) > 16 or args.count*(len(args.rates)+1)+40 > MAX_EVENTS):
        parser.error('timeout must be .01..10 seconds; maximum 16 rates and 100000 events')
    try:
        if args.command == 'control':
            print(json.dumps(rpc(args.socket, args.op)))
            return 0
        if args.command == 'run':
            with termination():
                return run(args)
        return globals()[args.command](args)
    except (ValueError, OSError, ImportError) as exc:
        print(f'mouse benchmark: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
