"""A stand-in terminal that accepts Kitty graphics writes.

Shared-memory payloads are unlinked as Kitty would, so a presenter can reuse
its bounded slots. The bytes themselves are counted and discarded.
"""

import base64
import os
import re


class ConsumerTerm:
    """Count writes and release ``t=s`` names."""

    _shm = re.compile(r"\x1b_G([^;]*\bt=s\b[^;]*);([A-Za-z0-9+/=]+)\x1b\\")

    def __init__(self):
        self.written = 0

    def write(self, data):
        self.written += len(data)
        for match in self._shm.finditer(data):
            name = base64.b64decode(match.group(2)).decode("ascii")
            path = "/dev/shm/" + name.lstrip("/")
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
