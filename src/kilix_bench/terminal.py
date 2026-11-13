"""A stand-in terminal that accepts Kitty graphics writes.

Shared-memory payloads are unlinked as Kitty would, so a presenter can reuse
its bounded slots. The bytes themselves are counted and discarded.
"""

import base64
import os
import re
import hashlib


class ConsumerTerm:
    """Count writes and release ``t=s`` names."""

    _shm = re.compile(r"\x1b_G([^;]*\bt=s\b[^;]*);([A-Za-z0-9+/=]+)\x1b\\")

    def __init__(self, *, verify_frames=False, expected_size=None):
        self.written = 0
        self.verify_frames = verify_frames
        self.expected_size = expected_size
        self.frames = 0
        self.last_hash = None

    def write(self, data):
        self.written += len(data)
        if isinstance(data, bytes):
            data = data.decode('ascii')
        for match in self._shm.finditer(data):
            name = base64.b64decode(match.group(2), validate=True).decode("ascii").lstrip('/')
            if not name or '/' in name or name in ('.', '..'):
                raise ValueError('invalid shared-memory name')
            path = "/dev/shm/" + name
            try:
                if self.verify_frames:
                    fields = dict(item.split('=', 1) for item in match.group(1).split(',') if '=' in item)
                    width, height, bits = (int(fields[key]) for key in ('s', 'v', 'f'))
                    if bits not in (24, 32) or min(width, height) < 1:
                        raise ValueError('invalid graphics frame')
                    if self.expected_size and (width, height) != tuple(self.expected_size):
                        raise ValueError('unexpected graphics geometry')
                    with open(path, 'rb') as stream:
                        raw = stream.read()
                    if len(raw) != width * height * (bits // 8):
                        raise ValueError('incomplete shared-memory frame')
                    self.last_hash = hashlib.sha256(raw).hexdigest()
                    self.frames += 1
            except FileNotFoundError:
                if self.verify_frames:
                    raise
            finally:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
