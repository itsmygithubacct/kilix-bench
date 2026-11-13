"""Measure Neuronte's RGB compositor, transport and FFmpeg decoder startup.

Pixel equality is checked outside kernel timings. Decoder timings include
startup, filtering and pipe output; first-frame latency includes accurate
input seeking. These measurements do not include a physical display.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from kilix_bench import __version__
from kilix_bench.report import scenario, summarize
from kilix_bench.terminal import ConsumerTerm


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(home):
    git = subprocess.run(['git', '-C', str(home), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(home), 'status', '--porcelain'], capture_output=True, text=True)
    return dict(revision=git.stdout.strip() if git.returncode == 0 else None,
                dirty=bool(dirty.stdout) if dirty.returncode == 0 else None,
                library_sha256=sha(home / 'build/libneuronte.so'),
                weights_sha256=sha(home / 'weights/neuronte-video-masks-v0.nrt'),
                module_sha256=sha(home / 'tools/neuronte_video.py'))


def measure_decoder(command, width, height, frames, timeout=60):
    """Read into one bounded frame buffer and reap this child's own rusage."""
    pixels = bytearray(width * height * 3)
    view = memoryview(pixels)
    hashes, first, used = [], None, 0
    start = time.perf_counter()
    driver = time.process_time()
    usage = None
    with tempfile.TemporaryFile() as errors:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
        try:
            fd = child.stdout.fileno()
            while True:
                if time.perf_counter() - start > timeout:
                    raise TimeoutError('decoder exceeded its time limit')
                if not select.select([fd], [], [], .1)[0]:
                    continue
                count = os.readv(fd, [view[used:]])
                if not count:
                    break
                used += count
                if used == len(view):
                    if first is None:
                        first = time.perf_counter() - start
                    hashes.append(hashlib.sha256(view).hexdigest())
                    used = 0
            while True:
                reaped, status, usage = os.wait4(child.pid, os.WNOHANG)
                if reaped:
                    break
                if time.perf_counter() - start > timeout:
                    raise TimeoutError('decoder stayed alive after closing its output')
                time.sleep(.01)
            child.returncode = os.waitstatus_to_exitcode(status)
            if child.returncode or used or len(hashes) != frames:
                raise ValueError(f'decoder exited {child.returncode}; {len(hashes)} frames and {used} trailing bytes')
        finally:
            if usage is None:
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
            child.stdout.close()
    return dict(wall_seconds=time.perf_counter() - start, first_frame_seconds=first,
                cpu_seconds=usage.ru_utime + usage.ru_stime, max_rss_kib=usage.ru_maxrss,
                driver_cpu_seconds=time.process_time() - driver, frames=len(hashes), hashes=hashes)


def kernel_scenarios(home, baseline, width, height, repeats, warmup, frames):
    import numpy as np
    module = load_module(home / 'tools/neuronte_video.py', 'video_candidate')
    prior = load_module(baseline / 'tools/neuronte_video.py', 'video_baseline') if baseline else module
    model, old = module.MultiInk(), prior.MultiInk()
    if not hasattr(model, 'render_rgb24') or not hasattr(module, 'RGBFrameReader'):
        raise ValueError('candidate needs render_rgb24 and RGBFrameReader support')
    pool = np.random.default_rng(20261113).integers(0, 256, (3, height, width, 3), dtype=np.uint8)
    output = np.empty((height, width, 3), np.uint8)

    def legacy(frame):
        padded = np.pad(frame, ((0, -height % 4), (0, -width % 2), (0, 0)), mode='edge')
        return old.decode(old.encode(padded))[:height, :width]

    references = [legacy(frame).copy() for frame in pool]
    for frame, reference in zip(pool, references):
        if not np.array_equal(model.render_rgb24(frame, output), reference):
            raise ValueError('candidate RGB differs from the palette/blend reference')
    # The input bytes are prepared before timing; both readers consume the
    # same number of complete frames from memory, without disk/cache effects.
    raw = pool[0].tobytes()
    reader = module.RGBFrameReader(width, height)
    # An extra byte prevents BytesIO.read from returning its complete backing
    # bytes object without copying, which would understate the old allocation.
    stream = io.BytesIO(raw + b'\0')

    def read_allocated(_):
        stream.seek(0)
        return stream.read(len(raw))

    def read_reused(_):
        stream.seek(0)
        return reader.read(stream)

    if read_allocated(None) != raw:
        raise ValueError('allocated reader changed input bytes')
    if read_reused(None).tobytes() != raw:
        raise ValueError('reused reader changed input bytes')

    # Import the real player's presenter; no replacement transport is timed.
    sys.path.insert(0, str(home / 'tools'))
    player = load_module(home / 'tools/neuronte-play.py', 'video_player')
    terminal = ConsumerTerm(verify_frames=True, expected_size=(width, height))
    presenter = player.Presenter(terminal, 'shm')

    def present(frame):
        payload = presenter.frame(frame, (width + 1)//2, (height + 3)//4, 1, 1, natural=True)
        if payload is None:
            raise ValueError('consumer did not acknowledge a frame')
        terminal.write(payload)

    operations = [('palette_reference', legacy), ('native_rgb24', lambda frame: model.render_rgb24(frame, output)),
                  ('read_allocated', read_allocated), ('read_reused', read_reused), ('shared_memory', present)]
    samples = {name: [] for name, _ in operations}
    cpu = {name: [] for name, _ in operations}
    try:
        for frame in pool:
            present(frame)
            if terminal.last_hash != hashlib.sha256(memoryview(frame)).hexdigest():
                raise ValueError('shared-memory consumer received different pixels')
        for repetition in range(warmup + repeats):
            offset = repetition % len(operations)
            for name, operation in operations[offset:] + operations[:offset]:
                start, process = time.perf_counter(), time.process_time()
                for index in range(frames):
                    operation(pool[index % len(pool)])
                elapsed, used = time.perf_counter() - start, time.process_time() - process
                if repetition >= warmup:
                    samples[name].append(elapsed / frames)
                    cpu[name].append(used / frames)
    finally:
        terminal.write(presenter.close())
    return [scenario(name, 'one complete RGB frame', samples[name], dict(
                cpu=summarize(cpu[name]), raw_seconds=samples[name], raw_cpu_seconds=cpu[name],
                inner_frames=frames, exact_rgb=True,
                native_available=bool(model.native_rgb24),
                boundary='create SHM + read/hash + acknowledge' if name == 'shared_memory' else 'application memory'))
            for name, _ in operations]


def decoder_scenarios(module, video, width, height, start, frames, repeats, warmup, thread_pairs):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
                            '-show_format', '-of', 'json', str(video)], capture_output=True, text=True, check=True)
    metadata = json.loads(probe.stdout)
    stream = metadata['streams'][0]
    num, den = map(int, stream['r_frame_rate'].split('/'))
    fps = num / den
    if start + frames/fps > float(metadata['format']['duration']):
        raise ValueError('requested frame interval exceeds the input')
    chroma = stream.get('chroma_location')
    if chroma not in ('left', 'center', 'topleft', 'top', 'bottomleft', 'bottom'):
        chroma = None
    rows, reference = [], None
    for repetition in range(warmup + repeats):
        offset = repetition % len(thread_pairs)
        for decoder, encoder in thread_pairs[offset:] + thread_pairs[:offset]:
            load_before = list(os.getloadavg())
            command = module.decoder_command(video, width, height, fps, start=start, chroma=chroma,
                                             decoder_threads=decoder, output_threads=encoder)
            command[-1:-1] = ['-frames:v', str(frames)]
            row = measure_decoder(command, width, height, frames)
            if reference is None:
                reference = row['hashes']
            row['exact_rgb'] = row['hashes'] == reference
            if not row['exact_rgb']:
                raise ValueError('thread configuration changed decoded pixels')
            row.update(decoder_threads=decoder, output_threads=encoder, repetition=repetition,
                       warmup=repetition < warmup, load_before=load_before, loadavg=list(os.getloadavg()),
                       command=['<video>' if str(item) == str(video) else item for item in command])
            rows.append(row)
    scenarios = []
    for decoder, encoder in thread_pairs:
        selected = [r for r in rows if not r['warmup'] and
                    (r['decoder_threads'], r['output_threads']) == (decoder, encoder)]
        scenarios.append(scenario(f'decode_{decoder}_raw_{encoder}', 'decode, scale and pipe one frame',
            [r['wall_seconds']/frames for r in selected], dict(
                cpu=summarize([r['cpu_seconds']/frames for r in selected]),
                first_frame=summarize([r['first_frame_seconds'] for r in selected]),
                max_rss_kib=[r['max_rss_kib'] for r in selected], exact_rgb=True,
                decoder_threads=decoder, output_threads=encoder)))
    source = {key: stream[key] for key in ('codec_name', 'width', 'height', 'pix_fmt', 'r_frame_rate',
                                          'color_space', 'color_range', 'chroma_location') if key in stream}
    source.update(sha256=sha(video), start=start, frames=frames)
    return scenarios, rows, source


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neuronte', type=Path, required=True)
    p.add_argument('--baseline-neuronte', type=Path)
    p.add_argument('--video', type=Path)
    p.add_argument('--start', type=float, default=0.)
    p.add_argument('--width', type=int, default=945)
    p.add_argument('--height', type=int, default=531)
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--frames', type=int, default=60)
    args = p.parse_args(argv)
    if (min(args.width, args.height, args.repeats, args.frames) < 1 or
            min(args.warmup, args.start) < 0 or not math.isfinite(args.start)):
        p.error('invalid size, repetition count or start')
    home = args.neuronte.resolve()
    if args.baseline_neuronte:
        args.baseline_neuronte = args.baseline_neuronte.resolve()
    model = identity(home)
    scenarios = kernel_scenarios(home, args.baseline_neuronte, args.width, args.height,
                                args.repeats, args.warmup, min(args.frames, 24))
    rows, source = [], None
    if args.video:
        module = load_module(home / 'tools/neuronte_video.py', 'video_decode')
        decoded, rows, source = decoder_scenarios(module, args.video.resolve(), args.width, args.height,
            args.start, args.frames, args.repeats, args.warmup, [(0,0), (0,1), (1,1), (2,1), (4,1)])
        scenarios.extend(decoded)
    import numpy as np
    ffmpeg = shutil.which('ffmpeg') if args.video else None
    decoder = dict(version=subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0],
                   sha256=sha(ffmpeg)) if ffmpeg else None
    payload = dict(tool='kilix-bench', tool_version=__version__, subject='neuronte-video',
                   subject_version='runtime', subject_revision=model['revision'],
                   python=platform.python_version(), machine=platform.machine(),
                   size=[args.width, args.height], model=model, scenarios=scenarios,
                   note='Offscreen RGB work; no physical display FPS. Decoder first frame includes seek/startup.',
                   decoder_runs=rows, source=source,
                   decoder=decoder, numpy=np.__version__,
                   baseline=identity(args.baseline_neuronte) if args.baseline_neuronte else model,
                   loadavg=list(os.getloadavg()))
    print(json.dumps(payload))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
