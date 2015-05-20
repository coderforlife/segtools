from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def make_dir(d):
    """Makes a directory tree. If the path exists as a regular file already False is returned."""
    import os, os.path
    if os.path.isdir(d): return True
    if os.path.exists(d): return False
    try:
        os.makedirs(d)
        return True
    except OSError:
        return False


def only_keep_num(d, allowed, match_slice=slice(None), pattern='*'):
    """
    Searches for all files matching a particular glob pattern, extracts the given slice as an
    integer, and makes sure it is in the list of allowed numbers. If not, the file is deleted.
    """
    from glob import iglob
    from os import unlink
    from os.path import basename, join, isfile

    files = ((f, basename(f)[match_slice]) for f in iglob(join(d, pattern)) if isfile(f))
    for f in (f for f, x in files if x.isdigit() and int(x) not in allowed):
        try: unlink(f)
        except OSError: pass
    files = ((f, basename(f)[match_slice]) for f in iglob(join(d, '.'+pattern)) if isfile(f))
    for f in (f for f, x in files if x.isdigit() and int(x) not in allowed):
        try: unlink(f)
        except OSError: pass


def get_terminal_width():
    """Gets the width of the terminal if there is a terminal, in which case 80 is returned."""
    import os
    w = __get_terminal_width_windows() if os.name == "nt" else __get_terminal_width_nix()
    # Last resort, mainly for *nix, but also set the default of 80
    if not w: w = os.environ.get('COLUMNS', 80)
    return int(w)
def __get_terminal_width_windows():
    from ctypes import windll, c_short, c_ushort, c_int, c_uint, c_void_p, byref, Structure
    class COORD(Structure): _fields_ = [("X", c_short), ("Y", c_short)]
    class SMALL_RECT(Structure): _fields_ = [("Left", c_short), ("Top", c_short), ("Right", c_short), ("Bottom", c_short)]
    class CONSOLE_SCREEN_BUFFER_INFO(Structure): _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD), ("wAttributes", c_ushort), ("srWindow", SMALL_RECT), ("dwMaximumWindowSize", COORD)]
    GetStdHandle = windll.kernel32.GetStdHandle
    GetStdHandle.argtypes, GetStdHandle.restype = [c_uint], c_void_p
    GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
    GetConsoleScreenBufferInfo.argtypes, GetConsoleScreenBufferInfo.restype = [c_void_p, c_void_p], c_int
    def con_width(handle):
        handle = GetStdHandle(handle)
        if handle and handle != -1:
            csbi = CONSOLE_SCREEN_BUFFER_INFO()
            if GetConsoleScreenBufferInfo(handle, byref(csbi)): return csbi.dwSize.X
        return None
    return con_width(-11) or con_width(-12) or con_width(-10) # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE, STD_INPUT_HANDLE
def __get_terminal_width_nix():
    try:
        import os, fcntl, termios, struct
        def ioctl_GWINSZ(fd):
            try:
                return struct.unpack(str('hh'), fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))[1]
            except (AttributeError, OSError, struct.error): return None
        w = ioctl_GWINSZ(1) or ioctl_GWINSZ(2) or ioctl_GWINSZ(0) # stdout, stderr, stdin
        if not w:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY) # pylint: disable=no-member
                w = ioctl_GWINSZ(fd)
                os.close(fd)
            except (AttributeError, OSError): pass
    except ImportError: return None
    return w


def all_subclasses(cls):
    subcls = cls.__subclasses__() # pylint: disable=no-member
    for sc in list(subcls): subcls.extend(all_subclasses(sc))
    return subcls
