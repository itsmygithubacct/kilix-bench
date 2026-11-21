"""Runner inventory and system-wide samples for a single unattended benchmark."""
import csv
import hashlib
import ipaddress
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import threading
from urllib.parse import urlsplit


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


def command(argv):
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                errors="replace", timeout=1.5, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def integer(value, base=10):
    try:
        n = int(value, base)
        return n if n >= 0 else None
    except (TypeError, ValueError):
        return None


def cpu_info(proc, sysroot):
    records = [dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
               for block in read(proc / "cpuinfo").split("\n\n")]
    records = [{k.strip(): v.strip() for k, v in row.items()} for row in records]
    names = dict.fromkeys(row.get("model name") or row.get("Hardware") or row.get("Processor") for row in records)
    cores = {(r["physical id"], r["core id"]) for r in records if "physical id" in r and "core id" in r}
    core_count = len(cores)
    if not core_count:
        siblings = {read(p) for p in (sysroot / "devices/system/cpu").glob("cpu[0-9]*/topology/thread_siblings_list")}
        core_count = len(siblings - {""})
    return dict(model=" / ".join(n for n in names if n) or None, physical_cores=core_count or None,
                logical_cpus=os.cpu_count(), architecture=platform.machine() or None)


def memory(proc):
    result = {}
    for line in read(proc / "meminfo").splitlines():
        name, _, value = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            fields = value.split()
            if len(fields) == 2 and fields[1] == "kB" and fields[0].isdecimal():
                result[name] = int(fields[0]) * 1024
    return result


def cpu_ticks(proc):
    line = read(proc / "stat").splitlines()
    fields = line[0].split() if line else []
    if len(fields) < 5 or fields[0] != "cpu" or not all(x.isdecimal() for x in fields[1:9]):
        return None
    ticks = list(map(int, fields[1:9]))
    return sum(ticks), sum(ticks[3:5])


def pci_slot(value):
    match = re.fullmatch(r"([0-9a-f]+):([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", value.lower())
    return f"{int(match[1], 16):04x}:{match[2]}" if match else value


def nvidia():
    fields = "pci.bus_id,name,memory.total,utilization.gpu,memory.used"
    rows = {}
    for values in csv.reader(command(["nvidia-smi", "--query-gpu=" + fields,
                                      "--format=csv,noheader,nounits"]).splitlines(), skipinitialspace=True):
        if len(values) == 5:
            slot, name, total, busy, used = (v.strip() for v in values)
            rows[pci_slot(slot)] = (name, integer(total), integer(busy), integer(used))
    return rows


def gpu_devices(sysroot, nv):
    # A compute-only NVIDIA adapter may be visible through NVML without a DRM card.
    devices = {k: None for k in nv}
    for path in sorted((sysroot / "bus/pci/devices").glob("*")):
        if read(path / "class").startswith("0x03"):
            devices[path.name] = path
    for card in sorted((sysroot / "class/drm").glob("card[0-9]*")):
        if card.name[4:].isdecimal() and (card / "device").is_dir():
            device = (card / "device").resolve()
            devices.setdefault(device.name, device)
            if devices[device.name] is None:
                devices[device.name] = device
    names = {}
    if devices:
        for line in command(["lspci", "-Dmm"]).splitlines():
            try:
                row = shlex.split(line)
            except ValueError:
                continue
            if len(row) >= 4:
                names[pci_slot(row[0])] = row[3]
    inventory = []
    for slot, path in sorted(devices.items()):
        name, total, _, _ = nv.get(slot, (None, None, None, None))
        vendor = read(path / "vendor") if path else ""
        device = read(path / "device") if path else ""
        driver = (path / "driver").resolve().name if path and (path / "driver").is_symlink() else None
        if not name:
            pci_id = f"PCI {vendor}:{device}" if vendor and device else None
            name = names.get(slot) or (read(path / "product_name") if path else "") or pci_id or driver or slot
        vram = total * 1024**2 if total is not None else (integer(read(path / "mem_info_vram_total")) if path else None)
        inventory.append(dict(slot=slot, model=name, vendor_id=vendor or None, device_id=device or None,
                              driver=driver, vram_bytes=vram))
    return inventory, devices


def machine_id():
    # Only the application-specific digest leaves the runner.
    value = read(Path("/etc/machine-id")) or read(Path("/var/lib/dbus/machine-id"))
    if not re.fullmatch(r"[a-fA-F0-9]{32}", value):
        value = platform.node()
    return hashlib.sha256(("puzzllm-yesno\0" + value).encode()).hexdigest()[:24] if value else None


def backend_scope(provider, url=None):
    if provider == "kilix":
        return "local Kilix"
    if provider == "llama.cpp":
        return "local llama.cpp"
    if provider == "ollama":
        url = url or "127.0.0.1"
        try:
            parts = urlsplit(url if "://" in url else "http://" + url)
            host = parts.hostname
            port = parts.port or (11434 if not parts.path.strip("/") else 443 if parts.scheme == "https" else 80)
        except ValueError:
            return "Ollama endpoint unknown"
        if not host:
            return "Ollama endpoint unknown"
        try:
            address = ipaddress.ip_address(host)
            local = address.is_loopback or address.is_unspecified
        except ValueError:
            local = host == "localhost"
        if local:
            return "loopback Ollama"
        endpoint = f"{parts.scheme}/{host}/{port}{parts.path.rstrip('/')}"
        tag = hashlib.sha256(("puzzllm-ollama\0" + endpoint).encode()).hexdigest()[:12]
        return f"remote Ollama [{tag}]"
    return "provider CLI"


class RunnerMonitor:
    interval = 2

    def __init__(self, proc=Path("/proc"), sysroot=Path("/sys")):
        self.proc = proc
        nv = nvidia()
        gpus, self.devices = gpu_devices(sysroot, nv)
        self.has_nvidia = bool(nv)
        self.profile = dict(id=machine_id(), label=os.environ.get("PUZZLLM_BENCH_MACHINE", "").strip()[:80] or None,
                            cpu=cpu_info(proc, sysroot), ram_bytes=memory(proc).get("MemTotal"), gpus=gpus)
        self.first_ticks = self.last_ticks = cpu_ticks(proc)
        self.peak_ram = None
        self.gpu_samples = {g["slot"]: [0, 0, None] for g in gpus}
        self.sample(nv)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def sample(self, nv):
        ticks = cpu_ticks(self.proc)
        if ticks:
            self.last_ticks = ticks
        mem = memory(self.proc)
        if "MemTotal" in mem and "MemAvailable" in mem:
            self.peak_ram = max(self.peak_ram or 0, mem["MemTotal"] - mem["MemAvailable"])
        for slot, path in self.devices.items():
            if slot in nv:
                _, _, busy, mib = nv[slot]
                used = mib * 1024**2 if mib is not None else None
            elif path:
                busy = integer(read(path / "gpu_busy_percent"))
                used = integer(read(path / "mem_info_vram_used"))
            else:
                continue
            sample = self.gpu_samples[slot]
            if busy is not None and 0 <= busy <= 100:
                sample[0] += busy
                sample[1] += 1
            if used is not None:
                sample[2] = max(sample[2] or 0, used)

    def _poll(self):
        while not self.done.wait(self.interval):
            self.sample(nvidia() if self.has_nvidia else {})

    def finish(self):
        self.done.set()
        self.thread.join()
        self.sample(nvidia() if self.has_nvidia else {})
        busy = None
        if self.first_ticks and self.last_ticks:
            total = self.last_ticks[0] - self.first_ticks[0]
            idle = self.last_ticks[1] - self.first_ticks[1]
            if total > 0 and 0 <= idle <= total:
                busy = round(100 * (total - idle) / total, 2)
        return dict(cpu_busy_percent=busy, peak_ram_used_bytes=self.peak_ram,
                    gpus=[dict(slot=slot, mean_busy_percent=round(summed / count, 2) if count else None,
                               peak_vram_used_bytes=peak, busy_samples=count)
                          for slot, (summed, count, peak) in self.gpu_samples.items()])
