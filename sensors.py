"""Machine and process sensors for the benchmark harness (macOS, Python standard library only).

    python3 harness/sensors.py          # probe: what this machine exposes, with a self-check of the libproc offsets

Sources (no sudo unless noted):
  * whole-system power: `ioreg` AppleSmartBattery PowerTelemetryData (SystemPowerIn on mains, BatteryPower on battery,
    milliwatts);
  * CPU / GPU / ANE power, GPU active residency and frequency, thermal pressure: `powermetrics` (sudo), one streaming
    process per session;
  * GPU utilisation and GPU memory in use: `ioreg` IOAccelerator PerformanceStatistics;
  * memory: `vm_stat` counters, `sysctl vm.swapusage`, kern.memorystatus_vm_pressure_level / _level;
  * per process: libproc proc_pid_rusage (RUSAGE_INFO_V6): physical footprint (current and lifetime peak), user /
    system time, instructions, cycles, CPU energy (nJ), disk bytes read / written, page-ins.
"""
from __future__ import annotations

import ctypes
import json
import os
import plistlib
import re
import subprocess
import sys
import threading
import time

GB = 1024 ** 3


def sh(cmd: list[str], timeout: float = 10.0, cwd: str | None = None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def sysctl(name: str) -> str:
    return sh(["sysctl", "-n", name]).strip()


def sysctl_int(name: str) -> int | None:
    try:
        return int(sysctl(name))
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------------ machine ---

def machine_info() -> dict:
    sw = {k: sh(["sw_vers", f"-{k}"]).strip() for k in ("productName", "productVersion", "buildVersion")}
    batt = sh(["pmset", "-g", "batt"])
    lpm = re.search(r"lowpowermode\s+(\d)", sh(["pmset", "-g"]))
    return dict(model=sysctl("hw.model"), chip=sysctl("machdep.cpu.brand_string"),
                memory_gb=round((sysctl_int("hw.memsize") or 0) / GB, 1), ncpu=sysctl_int("hw.ncpu"),
                perf_cores=sysctl_int("hw.perflevel0.physicalcpu"), eff_cores=sysctl_int("hw.perflevel1.physicalcpu"),
                os=sw, power_source=power_source(batt), battery=battery_percent(batt),
                low_power_mode=bool(int(lpm[1])) if lpm else None, python=sys.version.split()[0])


def power_source(batt: str | None = None) -> str:
    batt = sh(["pmset", "-g", "batt"]) if batt is None else batt
    m = re.search(r"drawing from '([^']+)'", batt)
    return m[1] if m else "unknown"


def battery_percent(batt: str | None = None) -> int | None:
    batt = sh(["pmset", "-g", "batt"]) if batt is None else batt
    m = re.search(r"(\d+)%", batt)
    return int(m[1]) if m else None


def top_processes(n: int = 8) -> list[dict]:
    """The busiest processes right now (CPU %), to document what else was running."""
    out = sh(["ps", "-A", "-o", "pid=,pcpu=,rss=,comm=", "-r"])
    rows = []
    for line in out.splitlines()[:n]:
        p = line.split(None, 3)
        if len(p) == 4:
            rows.append(dict(pid=int(p[0]), cpu=float(p[1]), rss_mb=int(p[2]) // 1024, comm=p[3].strip()))
    return rows


# ------------------------------------------------------------------------------------------------- ioreg ---

def _ioreg_dict(text: str, key: str) -> dict:
    """Parse `"key" = {"a"=1,"b"=2,...}` (flat, integer / string values) out of ioreg -w0 output."""
    m = re.search(rf'"{re.escape(key)}"\s*=\s*\{{([^{{}}]*)\}}', text)
    out = {}
    if m:
        for k, v in re.findall(r'"([^"]+)"=([^,]+)', m[1]):
            v = v.strip().strip('"')
            try:
                iv = int(v)
                out[k] = iv - (1 << 64) if iv >= (1 << 63) else iv   # ioreg prints negative values as uint64
            except ValueError:
                out[k] = v
    return out


def _ioreg_scalar(text: str, key: str):
    m = re.search(rf'"{re.escape(key)}"\s*=\s*(\w+)', text)
    if not m:
        return None
    v = m[1]
    if v in ("Yes", "No"):
        return v == "Yes"
    try:
        iv = int(v)
        return iv - (1 << 64) if iv >= (1 << 63) else iv
    except ValueError:
        return v


def system_power() -> dict:
    """Whole-system power from the SMC's power telemetry (ioreg AppleSmartBattery PowerTelemetryData).

    The instantaneous values (SystemLoad, BatteryPower, SystemPowerIn, mW) update only every few seconds (M5 Air on
    battery: none in 12 s of 4-core load, scratch/experiments/power_cadence.py). The SystemLoad *accumulator* sums
    the load every ~0.5 s but is published in batches (60 ticks every ~30 s): consecutive samples of `load_acc` /
    `load_count` give the exact mean system power over each batch (scoring.system_blocks), i.e. whole-system energy
    at ~30 s resolution, no sudo."""
    t = sh(["ioreg", "-rw0", "-c", "AppleSmartBattery"])
    tel = _ioreg_dict(t, "PowerTelemetryData")
    load = tel.get("SystemLoad")
    return dict(load_acc=tel.get("AccumulatedSystemLoad"), load_count=tel.get("SystemLoadAccumulatorCount"),
                system_w=None if load is None else abs(load) / 1000.0, on_mains=_ioreg_scalar(t, "ExternalConnected"),
                charging=_ioreg_scalar(t, "IsCharging"), battery_pct=_ioreg_scalar(t, "CurrentCapacity"))


def gpu_stats() -> dict:
    t = sh(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"])
    ps = _ioreg_dict(t, "PerformanceStatistics")
    return dict(gpu_util=ps.get("Device Utilization %"), gpu_renderer=ps.get("Renderer Utilization %"),
                gpu_mem_gb=None if ps.get("In use system memory") is None else ps["In use system memory"] / GB)


# ------------------------------------------------------------------------------------------------ memory ---

_VM_KEYS = {"free": "Pages free", "active": "Pages active", "inactive": "Pages inactive", "wired": "Pages wired down",
            "compressor": "Pages occupied by compressor", "compressions": "Compressions",
            "decompressions": "Decompressions", "pageins": "Pageins", "pageouts": "Pageouts", "swapins": "Swapins",
            "swapouts": "Swapouts", "purgeable": "Pages purgeable", "file_backed": "File-backed pages"}


def memory_stats() -> dict:
    out = sh(["vm_stat"])
    page = re.search(r"page size of (\d+) bytes", out)
    page = int(page[1]) if page else 16384
    v = {}
    for key, label in _VM_KEYS.items():
        m = re.search(rf"^{re.escape(label)}:\s+(\d+)", out, re.M)
        if m:
            v[key] = int(m[1])
    sw = re.search(r"used = ([\d.]+)M", sysctl("vm.swapusage"))
    gb = page / GB
    return dict(free_gb=v.get("free", 0) * gb, compressor_gb=v.get("compressor", 0) * gb,
                wired_gb=v.get("wired", 0) * gb, active_gb=v.get("active", 0) * gb,
                compressions=v.get("compressions"), decompressions=v.get("decompressions"), pageins=v.get("pageins"),
                pageouts=v.get("pageouts"), swapins=v.get("swapins"), swapouts=v.get("swapouts"),
                swap_used_gb=float(sw[1]) / 1024 if sw else None, page_bytes=page,
                pressure_level=sysctl_int("kern.memorystatus_vm_pressure_level"),
                free_pct=sysctl_int("kern.memorystatus_level"))


# ------------------------------------------------------------------------------------- per-process (libproc) ---

class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_libproc = _libc = None
_TB = 1.0
try:
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    _libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    _tb = _Timebase()
    _libc.mach_timebase_info(ctypes.byref(_tb))
    _TB = _tb.numer / _tb.denom                              # mach absolute time units -> ns
except OSError:
    pass

# rusage_info_v6 offsets (<sys/resource.h>): u64 fields after a 16-byte uuid
_RU = dict(user=16, system=24, pageins=48, footprint=72, child_user=96, child_system=104, disk_read=144,
           disk_write=152, logical_writes=232, max_footprint=240, instructions=248, cycles=256, energy_nj=336)
_RU_BYTES = 512


def proc_rusage(pid: int) -> dict | None:
    """Counters of one live process (None if it is gone or libproc is unavailable). Times in s, energy in J."""
    if _libproc is None:
        return None
    buf = ctypes.create_string_buffer(_RU_BYTES)
    if _libproc.proc_pid_rusage(pid, 6, buf) != 0:
        return None
    u = {k: int.from_bytes(buf.raw[o:o + 8], "little") for k, o in _RU.items()}
    return dict(footprint_gb=u["footprint"] / GB, max_footprint_gb=u["max_footprint"] / GB,
                user_s=u["user"] * _TB * 1e-9, system_s=u["system"] * _TB * 1e-9, pageins=u["pageins"],
                disk_read_gb=u["disk_read"] / GB, disk_write_gb=u["disk_write"] / GB,
                instructions=u["instructions"], cycles=u["cycles"], cpu_energy_j=u["energy_nj"] * 1e-9)


def descendants(root: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for line in sh(["ps", "-A", "-o", "pid=,ppid="]).splitlines():
        p = line.split()
        if len(p) == 2:
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


class ProcessTree:
    """Accumulates the counters of a process tree: per pid the last values seen (cumulative counters survive the
    exit of a child as its last sample), summed over pids."""

    SUM = ("user_s", "system_s", "pageins", "disk_read_gb", "disk_write_gb", "instructions", "cycles", "cpu_energy_j")

    def __init__(self, root: int):
        self.root, self.last = root, {}

    def sample(self) -> dict:
        live = 0.0
        for pid in descendants(self.root):
            r = proc_rusage(pid)
            if r is not None:
                self.last[pid] = r
                live += r["footprint_gb"]
        tot = {k: sum(r[k] for r in self.last.values()) for k in self.SUM}
        tot["footprint_gb"] = live
        tot["max_footprint_gb"] = max((r["max_footprint_gb"] for r in self.last.values()), default=0.0)
        tot["nproc"] = len(self.last)
        return tot


# ----------------------------------------------------------------------------------- powermetrics (sudo) ---

def sudo_ok() -> bool:
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def sudo_prompt() -> bool:
    """Ask for the sudo password once (interactive terminal only); False when it cannot be obtained."""
    if sudo_ok():
        return True
    if not sys.stdin.isatty():
        return False
    print("[harness] sudo is needed for powermetrics (CPU / GPU power); enter your password once:", flush=True)
    return subprocess.run(["sudo", "-v"]).returncode == 0


class PowerMetrics:
    """One streaming `sudo powermetrics` process for the whole session; samples are (monotonic t, dict)."""

    def __init__(self, interval_ms: int = 250):
        self.samples: list[tuple[float, dict]] = []
        self.proc = None
        self.error = None
        self.interval_ms = interval_ms
        self._lock = threading.Lock()

    def start(self) -> bool:
        cmd = ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,thermal", "-i", str(self.interval_ms),
               "-f", "plist"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            self.error = str(e)
            return False
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._keepalive, daemon=True).start()
        time.sleep(1.5 * self.interval_ms / 1000 + 0.5)
        if self.proc.poll() is not None:
            self.error = self.proc.stderr.read().decode(errors="replace").strip()[:300]
            return False
        return True

    def _keepalive(self):
        while self.proc is not None and self.proc.poll() is None:
            subprocess.run(["sudo", "-n", "-v"], capture_output=True)
            time.sleep(60)

    def _read(self):
        buf = b""
        while True:
            chunk = self.proc.stdout.read1(65536) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(4096)
            if not chunk:
                return
            buf += chunk
            while b"\x00" in buf:
                doc, buf = buf.split(b"\x00", 1)
                doc = doc.strip()
                if doc:
                    try:
                        s = parse_powermetrics(plistlib.loads(doc))
                    except Exception:                       # noqa: BLE001 - a malformed sample is skipped
                        continue
                    with self._lock:
                        self.samples.append((time.monotonic(), s))

    def window(self, ta: float, tb: float) -> list[tuple[float, dict]]:
        with self._lock:
            return [(t, s) for t, s in self.samples if ta <= t <= tb]

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


def parse_powermetrics(d: dict) -> dict:
    """The fields we keep from one powermetrics plist sample (W, fractions, Hz)."""
    p, g = d.get("processor", {}) or {}, d.get("gpu", {}) or {}
    out = {}
    for k in ("cpu_power", "gpu_power", "ane_power", "combined_power"):
        if k in p:
            out[k.replace("_power", "_w")] = p[k] / 1000.0
    if "gpu_power" not in p and "gpu_energy" in g and d.get("elapsed_ns"):
        out["gpu_w"] = g["gpu_energy"] / 1000.0 / (d["elapsed_ns"] * 1e-9)
    if "idle_ratio" in g:
        out["gpu_active"] = 1.0 - g["idle_ratio"]
    if "freq_hz" in g:
        out["gpu_mhz"] = g["freq_hz"] / 1e6
    if "thermal_pressure" in d:
        out["thermal"] = d["thermal_pressure"]
    return out


# --------------------------------------------------------------------------------------------- samplers ---

class Sampler(threading.Thread):
    """Calls fn() every `interval` s; keeps (monotonic t, dict) samples until stop()."""

    def __init__(self, fn, interval: float = 1.0):
        super().__init__(daemon=True)
        self.fn, self.interval = fn, interval
        self.samples: list[tuple[float, dict]] = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            t = time.monotonic()
            try:
                self.samples.append((t, self.fn()))
            except Exception as e:                          # noqa: BLE001 - keep sampling
                self.samples.append((t, dict(error=str(e))))
            self._halt.wait(max(0.0, self.interval - (time.monotonic() - t)))

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


def probe() -> dict:
    """What this machine exposes; checks the libproc offsets against os.times() of this process."""
    x = 0.0
    t_end = time.process_time() + 0.3
    while time.process_time() < t_end:                     # burn a little CPU so the counters move
        x += 1.0
    me = proc_rusage(os.getpid())
    ot = os.times()
    check = None
    if me is not None:
        check = dict(user_s=me["user_s"], os_times_user=ot.user, ok=abs(me["user_s"] - ot.user) < 0.1 + 0.1 * ot.user,
                     max_footprint_ge_footprint=me["max_footprint_gb"] >= me["footprint_gb"],
                     instructions=me["instructions"], cpu_energy_j=me["cpu_energy_j"])
    pw = []
    for _ in range(4):
        pw.append(system_power().get("system_w"))
        time.sleep(1.0)
    return dict(machine=machine_info(), system_power_samples_w=pw, gpu=gpu_stats(), memory=memory_stats(),
                libproc_check=check, sudo_without_password=sudo_ok())


if __name__ == "__main__":
    print(json.dumps(probe(), indent=1))
