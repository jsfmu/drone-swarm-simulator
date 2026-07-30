"""Best-effort, stdlib-only process memory reading (Phase 5 monitoring).

Deliberately standalone rather than imported from ``benchmarks/benchmark_avoidance.py``
(which has the same logic for its own memory-footprint reporting) -- the
simulation/API package must not depend on anything under ``benchmarks/``, the
same import-direction rule ``api/app.py``'s "only place that imports FastAPI"
comment already establishes for the kernel/API boundary. Returns ``None`` on
any platform where no stdlib-only method exists, rather than a fabricated
number.
"""

from __future__ import annotations

import ctypes
import platform
from typing import Optional


def resident_set_size_bytes() -> Optional[int]:
    """Current process resident/working-set size in bytes, or ``None`` if
    unavailable on this platform."""
    system = platform.system()
    if system in ("Linux", "Darwin"):
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return usage * 1024 if system == "Linux" else usage  # Linux: KB, macOS: bytes
        except Exception:
            return None
    if system == "Windows":
        try:
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
            ]
            ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        except Exception:
            return None
    return None
