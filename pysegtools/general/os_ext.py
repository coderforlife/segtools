"""
Emulates os.wait4() for Windows. If wait4 is imported from this module on a UNIX machine then the
real wait4 is properly imported. If this is imported on a system that is not Windows and does not
have a wait4 implementation then there will be an ImportError.

There are some differences in the returned rusage data. Differences:
  * Number of page faults cannot be split between minor and major so all are listed as minor (and
    major is always 0).
  * Many fields cannot be calculaed so are just left as 0 - however this is also true in Linux.
    Compared to Linux voluntary and involuntary context switches are always zero.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = ['wait4']

from os import name as os_name
if os_name != 'nt':
    # Not Windows, try to import the real wait4
    from os import wait4 # pylint: disable=no-name-in-module
else:
    from collections import namedtuple
    from ctypes import windll, WinError, POINTER, byref, Structure, sizeof
    from ctypes import c_ulonglong as ULONGLONG, c_size_t as SIZE_T
    from ctypes.wintypes import BOOL, DWORD, HANDLE, FILETIME

    def cb_winerrcheck(success, func, args):
        if not success:
            if len(args) > 0 and isinstance(args[0], HANDLE): CloseHandle(args[0])
            raise WinError()
        return True
    def cb_wait_check(result, func, args):
        return result == 0
    def cb_ValidHandle(h):
        if h == 0: raise WinError()
        return HANDLE(h)
    def ft2sec(ft):
        return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 10000000.0

    k32 = windll.kernel32

    PROCESS_QUERY_INFORMATION = 0x00000400
    SYNCHRONIZE               = 0x00100000
    OpenProcess = k32.OpenProcess
    OpenProcess.argtypes = [DWORD, BOOL, DWORD]
    OpenProcess.restype  = cb_ValidHandle # HANDLE

    INFINITE = 0xFFFFFFFF
    WaitForSingleObject = k32.WaitForSingleObject
    WaitForSingleObject.argtypes = [HANDLE, DWORD]
    WaitForSingleObject.restype  = DWORD
    WaitForSingleObject.errcheck = cb_wait_check

    GetExitCodeProcess = k32.GetExitCodeProcess
    GetExitCodeProcess.argtypes = [HANDLE, POINTER(DWORD)]
    GetExitCodeProcess.restype  = BOOL
    GetExitCodeProcess.errcheck = cb_winerrcheck

    GetProcessTimes = k32.GetProcessTimes
    GetProcessTimes.argtypes = [HANDLE,
                                POINTER(FILETIME), POINTER(FILETIME),
                                POINTER(FILETIME), POINTER(FILETIME)]
    GetProcessTimes.restype  = BOOL
    GetProcessTimes.errcheck = cb_winerrcheck

    class PROCESS_MEMORY_COUNTERS(Structure):
        _fields_ = [("cb", DWORD),
                    ("PageFaultCount", DWORD),
                    ("PeakWorkingSetSize",         SIZE_T), ("WorkingSetSize",         SIZE_T),
                    ("QuotaPeakPagedPoolUsage",    SIZE_T), ("QuotaPagedPoolUsage",    SIZE_T),
                    ("QuotaPeakNonPagedPoolUsage", SIZE_T), ("QuotaNonPagedPoolUsage", SIZE_T),
                    ("PagefileUsage",              SIZE_T), ("PeakPagefileUsage",      SIZE_T)]
    GetProcessMemoryInfo = windll.psapi.GetProcessMemoryInfo # or windll.K32GetProcessMemoryInfo on Windows 7 and newer
    GetProcessMemoryInfo.argtypes = [HANDLE, POINTER(PROCESS_MEMORY_COUNTERS), DWORD]
    GetProcessMemoryInfo.restype  = BOOL
    GetProcessMemoryInfo.errcheck = cb_winerrcheck

    class IO_COUNTERS(Structure):
        _fields_ = [("ReadOperationCount", ULONGLONG), ("WriteOperationCount", ULONGLONG), ("OtherOperationCount", ULONGLONG),
                    ("ReadTransferCount",  ULONGLONG), ("WriteTransferCount",  ULONGLONG), ("OtherTransferCount",  ULONGLONG)]
    GetProcessIoCounters = k32.GetProcessIoCounters
    GetProcessIoCounters.argtypes = [HANDLE, POINTER(IO_COUNTERS)]
    GetProcessIoCounters.restype  = BOOL
    GetProcessIoCounters.errcheck = cb_winerrcheck

    CloseHandle = k32.CloseHandle
    CloseHandle.argtypes = [HANDLE]
    CloseHandle.restype  = BOOL
    #CloseHandle.errcheck = cb_winerrcheck

    struct_rusage = namedtuple('struct_rusage',
                               ['ru_utime', 'ru_stime',
                                'ru_maxrss', 'ru_ixrss', 'ru_idrss', 'ru_isrss',
                                'ru_minflt', 'ru_majflt', 'ru_nswap',
                                'ru_inblock', 'ru_oublock',
                                'ru_msgsnd', 'ru_msgrcv',
                                'ru_nsignals',
                                'ru_nvcsw', 'ru_nivcsw'])

    def wait4(pid, options_ = 0):
        h = OpenProcess(PROCESS_QUERY_INFORMATION | SYNCHRONIZE, True, pid)
        WaitForSingleObject(h, INFINITE)

        exitcode = DWORD(-1)
        GetExitCodeProcess(h, byref(exitcode))

        ctime, etime, stime, utime = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        GetProcessTimes(h, byref(ctime), byref(etime), byref(stime), byref(utime))

        mem = PROCESS_MEMORY_COUNTERS(cb=sizeof(PROCESS_MEMORY_COUNTERS))
        GetProcessMemoryInfo(h, byref(mem), sizeof(PROCESS_MEMORY_COUNTERS))

        io = IO_COUNTERS()
        GetProcessIoCounters(h, byref(io))

        CloseHandle(h)

        rusage = struct_rusage(
            ru_utime=ft2sec(utime), ru_stime=ft2sec(stime),
            ru_maxrss=mem.PeakWorkingSetSize//1024, ru_ixrss=0, ru_idrss=0, ru_isrss=0,
            ru_minflt=mem.PageFaultCount, ru_majflt=0, ru_nswap=0,
            ru_inblock=io.ReadOperationCount, ru_oublock=io.WriteOperationCount,
            ru_msgsnd=0, ru_msgrcv=0, ru_nsignals=0,
            ru_nvcsw=0, ru_nivcsw=0,
            )

        return pid, exitcode.value, rusage
