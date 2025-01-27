"""Utilities for library use."""

import os, sys
from collections.abc import Iterable

sys_endian = '<' if sys.byteorder == 'little' else '>'
sys_64bit = sys.maxsize > 2**32

def pairwise(iterable):
    """
    Makes an iterator that gives the first and second items from iterable, then the second and
    third items, until the end. Overall one less pair of items are generated than are in iterable.
    """
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


##### numpy-like functions for iterators #####
def prod(itr):
    from functools import reduce
    from operator import mul
    return reduce(mul, itr, 1)
def ravel(itr):
    return (x for i in itr for x in (ravel(i) if isinstance(i, Iterable) and not isinstance(i, str) else (i,)))
def __reshape(itr, shape, otype):
    return otype((next(itr) for _ in range(shape[0])) if len(shape) == 1 else (reshape(itr, shape[1:], otype) for _ in range(shape[0])))
def reshape(itr, shape, otype=list):
    return __reshape(iter(itr), tuple(shape) if isinstance(shape, Iterable) else (shape,), otype) # otype can be list or tuple


##### string and casting utilities #####
def re_search(re, s):
    re_search.match = re.search(s)
    return re_search.match is not None
def itr2str(itr, sep=' '):
    return sep.join(type(sep)(x) for x in itr)
def splitstr(s, cast=lambda x: x, sep=None):
    return [cast(x) for x in s.split(sep)]
def get_list(data, shape, cast=int, sep=None, otype=list):
    """
    Convert a string of values to a list of a particular data type. The data can also come from an
    iterable in which case all elements are ensured to be the right type. You can specify either a
    single value or a tuple for the shape. The dtype defaults to int, but can be others. The
    seperator in the string defaults to all whitespace. The output type defaults to a list, but you
    can also set it to tuple to get an imutable output.
    """
    shape = tuple(shape) if isinstance(shape, Iterable) else (shape,)
    data = (cast(x) for x in data.split(sep)) if isinstance(data, str) else ravel(data)
    #if isinstance(data, Iterable) else repeat(cast(data), prod(shape))
    return __reshape(data, shape, otype)
def _bool(x, strict_str=False):
    """
    Casts a value to a bool taking into acount the string values "true", "false", "t", "f", "1",
    and "0" (not case-sensitive). If strict_str is True, only these string values are allowed
    (otherwise other strings are sent to bool() which means an empty string is False and all other
    strings are True).
    """
    if isinstance(x, str):
        if x.lower() in ('false', 'f', '0'): return False
        if x.lower() in ('true',  't', '1'): return True
        if strict_str: raise ValueError('Invalid string to bool conversion')
    return bool(x)
def dtype_cast(x, dtype):
    """Casts a value using a dtype specification."""
    from numpy import array
    a = array([x], dtype.base)
    if a.shape[1:] != dtype.shape: raise ValueError('Cannot convert "' + x + '" to ' + str(dtype))
    return a[0] if a.ndim == 1 else tuple(a[0])


##### filesystem utilities #####
def make_dir(path):
    """Makes a directory tree. If the path exists as a regular file already False is returned."""
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        return exc.errno == errno.EEXIST and os.path.isdir(path)
    return True
def only_keep_num(d, allowed, match_slice=slice(None), pattern='*'):
    """
    Searches for all files matching a particular glob pattern, extracts the given slice as an
    integer, and makes sure it is in the list of allowed numbers. If not, the file is deleted.
    """
    from glob import iglob
    from os.path import basename, join, isfile

    files = ((f, basename(f)[match_slice]) for f in iglob(join(d, pattern)) if isfile(f))
    for f in (f for f, x in files if x.isdigit() and int(x) not in allowed):
        try:
            os.unlink(f)
        except OSError:
            pass
    files = ((f, basename(f)[match_slice]) for f in iglob(join(d, '.'+pattern)) if isfile(f))
    for f in (f for f, x in files if x.isdigit() and int(x) not in allowed):
        try:
            os.unlink(f)
        except OSError:
            pass


##### Python utilities #####
def all_subclasses(cls):
    subcls = cls.__subclasses__() # pylint: disable=no-member
    for sc in list(subcls):
        subcls.extend(all_subclasses(sc))
    return subcls
def load_plugins(name):
    mod = sys.modules[name]
    directory = os.path.dirname(mod.__file__)
    plugins = [os.path.basename(f)[:-3] for f in os.listdir(directory)
               if f[-3:] == ".py" and f[0] != "_" and os.path.isfile(os.path.join(directory, f))]
    if hasattr(mod, 'load_plugins'): del mod.load_plugins
    glbls = {
        '__name__': name,
        '__package__': mod.__package__,
        '__file__': mod.__file__,
        '__builtins__': __builtins__,
        '__all__': plugins,
        }
    mod.__all__ = plugins
    for plugin in plugins:
        try:
            setattr(mod, plugin, getattr(__import__(name, glbls, glbls, [plugin], 0), plugin))
        except ImportError as ex:
            import warnings
            warnings.warn("Failed to load %s plugin '%s': %s"%(name, plugin, ex))


##### OS utilities #####
def get_terminal_width():
    """Gets the width of the terminal if there is a terminal, in which case 80 is returned."""
    w = __get_terminal_width_windows() if os.name == "nt" else __get_terminal_width_nix()
    # Last resort, mainly for *nix, but also set the default of 80
    if not w:
        w = os.environ.get('COLUMNS', 80)
    return int(w)
def __get_terminal_width_windows():
    # pylint: disable=invalid-name, missing-docstring
    from ctypes import windll, c_short, c_ushort, c_int, c_uint, c_void_p, byref, Structure
    class COORD(Structure):
        _fields_ = [("X", c_short), ("Y", c_short)]
    class SMALL_RECT(Structure):
        _fields_ = [("Left", c_short), ("Top", c_short), ("Right", c_short), ("Bottom", c_short)]
    class CONSOLE_SCREEN_BUFFER_INFO(Structure):
        _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD), ("wAttributes", c_ushort), ("srWindow", SMALL_RECT), ("dwMaximumWindowSize", COORD)]
    GetStdHandle = windll.kernel32.GetStdHandle
    GetStdHandle.argtypes, GetStdHandle.restype = [c_uint], c_void_p
    GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
    GetConsoleScreenBufferInfo.argtypes = [c_void_p, c_void_p]
    GetConsoleScreenBufferInfo.restype = c_int
    def con_width(handle):
        handle = GetStdHandle(handle)
        if handle and handle != -1:
            csbi = CONSOLE_SCREEN_BUFFER_INFO()
            if GetConsoleScreenBufferInfo(handle, byref(csbi)): return csbi.dwSize.X
        return None
    return con_width(-11) or con_width(-12) or con_width(-10) # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE, STD_INPUT_HANDLE
def __get_terminal_width_nix():
    try:
        import fcntl, termios
        def ioctl_GWINSZ(fd):
            try:
                import struct
                return struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))[1]
            except (AttributeError, OSError, IOError, struct.error):
                return None
        w = ioctl_GWINSZ(1) or ioctl_GWINSZ(2) or ioctl_GWINSZ(0) # stdout, stderr, stdin
        if not w:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY) # pylint: disable=no-member
                w = ioctl_GWINSZ(fd)
                os.close(fd)
            except (AttributeError, OSError):
                pass
    except ImportError:
        return None
    return w

def get_rss_limit(pid=None):
    """
    Gets the RSS limit for a process if there is one, or -1 if there is not.
    On non-Linux machines this always returns -1.
    """
    import psutil
    if not hasattr(psutil, 'RLIMIT_RSS'): return -1
    return psutil.Process(pid).rlimit(psutil.RLIMIT_RSS)[0]

def get_cpu_limit(pid=None):
    """
    Gets the CPU limit for a process based on the CPU affinity.
    Normally this will return the number of logical CPUs.
    """
    import psutil
    return len(psutil.Process(pid).cpu_affinity())

def get_mem_usable():
    """
    Get the amount of usable memory for this process. This is the minimum of the RSS limit and
    the available amount of memory on the machine.
    """
    import psutil
    rss, avail = get_rss_limit(), psutil.virtual_memory().available
    return avail if rss == -1 else min(avail, rss)

def get_ncpus_usable():
    """
    Get the number of usable CPUs for this process. This is the minimum of the CPU limit and
    the total number of logical CPUs.
    """
    import psutil
    return min(get_cpu_limit(), psutil.cpu_count(True))
