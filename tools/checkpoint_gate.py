"""One-shot process gate installed atomically at a worker executable pathname.

An already executing native inode keeps running. The next invocation reaches
this gate, proving the parent has consumed and flushed the preceding trial.
No native work starts here. A separately verified checkpoint authorizes exit75;
without that receipt the gate waits, leaving the original lock held.
"""
import json
import os
from pathlib import Path
import sys
import time


def gate(config_path):
    config = json.loads(Path(config_path).read_text())
    ready = Path(config['ready'])
    evidence = {'pid': os.getpid(), 'ppid': os.getppid(), 'argv': sys.argv,
                'ready_time': time.time(), 'native_started': False}
    temporary = ready.with_name(ready.name + '.tmp')
    if ready.exists():
        raise RuntimeError('gate ready receipt already exists')
    with temporary.open('x') as stream:
        json.dump(evidence, stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, ready)
    authorization = Path(config['authorization'])
    while not authorization.exists():
        time.sleep(.1)
    receipt = json.loads(authorization.read_text())
    if receipt.get('gate_pid') != os.getpid() or receipt.get('action') != 'checkpoint-saved-exit':
        raise RuntimeError('gate exit lacks matching checkpoint authorization')
    return 75


if __name__ == '__main__':
    sys.exit(gate(sys.argv[1]))
