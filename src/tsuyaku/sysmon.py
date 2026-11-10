"""この Mac の使用率(CPU・メモリ・GPU)。画面の隅に小さく出す(設定でオフ)。

標準の仕組みだけ・root 不要: CPU = host_statistics の差分、メモリ = vm_stat と hw.memsize、
GPU = ioreg の IOAccelerator の "Device Utilization %"。数秒ごとに呼ばれても軽いよう結果を温存し、
取れないものは入れない(画面はその欄を出さない)。この Mac の値だけで、ほかの機械の情報は扱わない。"""
from __future__ import annotations

import ctypes
import re
import subprocess
import threading
import time

CACHE_S = 2.0


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout


def parse_mem(vm_stat: str, total: int) -> "float | None":
    """使用中 = active + wired + compressor(アクティビティモニタの「使用メモリ」に近い)。"""
    if not total or total <= 0:
        return None
    m = re.search(r"page size of (\d+) bytes", vm_stat)
    page = int(m.group(1)) if m else 16384

    def pages(label: str) -> "int | None":
        mm = re.search(label + r":\s+(\d+)", vm_stat)
        return int(mm.group(1)) if mm else None
    parts = [pages("Pages active"), pages("Pages wired down"), pages("Pages occupied by compressor")]
    if parts[0] is None or parts[1] is None:
        return None
    used = sum(p or 0 for p in parts) * page
    return max(0.0, min(100.0, used / total * 100))


def parse_gpu(ioreg: str) -> "int | None":
    vals = [int(x) for x in re.findall(r'"Device Utilization %"\s*=\s*(\d+)', ioreg)]
    return min(100, max(vals)) if vals else None


def cpu_pct(prev, cur) -> "int | None":
    """host_statistics の (user, system, idle, nice) の2点から、その間の使用率。"""
    if not prev or not cur:
        return None
    d = [c - p for p, c in zip(prev, cur)]
    total = sum(d)
    if total <= 0 or min(d) < 0:
        return None
    return round((total - d[2]) / total * 100)


def _cpu_ticks():
    """HOST_CPU_LOAD_INFO(=3)。取れなければ None。"""
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.mach_host_self.restype = ctypes.c_uint
        buf = (ctypes.c_uint * 4)()
        cnt = ctypes.c_uint(4)
        lib.host_statistics.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        if lib.host_statistics(lib.mach_host_self(), 3, ctypes.byref(buf), ctypes.byref(cnt)) != 0:
            return None
        return (buf[0], buf[1], buf[2], buf[3])
    except Exception:
        return None


class Sysmon:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t = 0.0
        self._last: dict = {}
        self._ticks = None
        self._total = None

    def sample(self) -> dict:
        """{"cpu":%, "mem":%, "gpu":%}。取れた欄だけ。CACHE_S 以内の再呼び出しは前の値。"""
        with self._lock:
            now = time.monotonic()
            if self._t and now - self._t < CACHE_S:
                return dict(self._last)
            out: dict = {}
            cur = _cpu_ticks()
            c = cpu_pct(self._ticks, cur)
            if cur:
                self._ticks = cur
            if c is not None:
                out["cpu"] = c
            try:
                if self._total is None:
                    self._total = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
                m = parse_mem(_run(["vm_stat"]), self._total)
                if m is not None:
                    out["mem"] = round(m)
            except Exception:
                pass
            try:
                g = parse_gpu(_run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"]))
                if g is not None:
                    out["gpu"] = g
            except Exception:
                pass
            self._t, self._last = now, out
            return dict(out)
