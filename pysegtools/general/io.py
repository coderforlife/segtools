"""IO Utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, io
from enum import IntFlag
from contextlib import contextmanager
from numbers import Integral
from numpy import fromfile as npy_fromfile, nditer, empty, ascontiguousarray, asfortranarray

from . import GzipFile, prod

__all__ = ['FileMode', 'umask', 'fromfile', 'tofile',
           'check_file_obj', 'isfileobj', 'openfile', 'get_file_size', 'get_file_name',
           'array_read', 'array_skip', 'array_save', 'array_read_ascii', 'array_skip_ascii', 'array_save_ascii',
           'copy_data', 'fill_data', 'file_remove_ranges', 'FileInsertIO']

__is_py3 = sys.version_info[0] == 3

class FileMode(IntFlag):
    """
    File mode as a flags object instead of the Python standard string method. Methods are provided
    for converting between the Python string definitions.
    """
    Read = 1
    Write = 2
    ReadWrite = 2|1
    Append = 4|2 # always write when appending
    Binary = 8
    Truncate = 16

    # TODO: invalid combination: Truncate with Append

    @staticmethod
    def from_string(mode):
        m = FileMode(0)
        if 'r' in mode: m |= FileMode.Read
        if 'w' in mode: m |= FileMode.Write | FileMode.Truncate
        if 'a' in mode: m |= FileMode.Append
        if '+' in mode: m |= FileMode.ReadWrite
        if 'b' in mode: m |= FileMode.Binary
        return m

    @staticmethod
    def from_file(f): return FileMode.from_string(f.mode)

    @staticmethod
    def to_string(mode):
        if FileMode.Append in mode:
            m = 'a+' if FileMode.Read in mode else 'a'
        elif FileMode.Truncate in mode:
            m = 'w+' if FileMode.Read in mode else 'w'
        else:
            m = 'r' if FileMode.Write in mode else 'r+'
        if FileMode.Binary in mode: m += 'b'
        return m

@contextmanager
def umask(mask):
    """Context manager for temporarily changing the umask value."""
    orig = os.umask(mask)
    try: yield
    finally: os.umask(orig)
        
@contextmanager
def __as_file(f):
    """Takes a io.FileIO object and makes it usable as a raw file as a contextmanager."""
    f.flush()
    fileno = f.fileno()
    fd2 = os.dup(fileno)
    f2 = os.fdopen(fd2, f.mode)
    orig_pos = os.lseek(fd2, 0, os.SEEK_CUR)
    f2.seek(f.tell())
    try: yield f2
    finally:
        position = f2.tell()
        f2.close()
        os.lseek(fileno, orig_pos, os.SEEK_SET)
        f.seek(position)
    
def fromfile(f, dt=float, count=-1, sep=''):
    """Wrapper for numpy.fromfile that handles io.FileIO objects in Python 2"""
    if __is_py3 or isinstance(f, (basestring,file)): return npy_fromfile(f, dt, count, sep)
    with __as_file(f) as f2: arr = npy_fromfile(f2, dt, count, sep)
    return arr

def tofile(arr, f, sep='', format='%s'): #pylint: disable=redefined-builtin
    """Wrapper for ndarray.tofile that handles io.FileIO objects in Python 2"""
    if __is_py3 or isinstance(f, (basestring,file)): arr.tofile(f, sep, format); return
    with __as_file(f) as f2: arr.tofile(f2, sep, format)

def any_in(a, b): return any(x in b for x in a)
def check_file_obj(f, read, write, seek, binary=True, append=False):
    """
    Checks a file object to make sure it is readable, writable, and/or seekable. Also checks that it
    is binary or text and that it is appending or acting normally. This accepts io objects and
    file-like that have a mode attribute with the contents "[rwa]+?[bt]?".
    """
    if hasattr(f, 'closed') and f.closed: return False
    if isinstance(f, io.IOBase):
        return ((not read or f.readable()) and (not write or f.writable()) and
                (not seek or f.seekable()) and (binary != isinstance(f,io.TextIOBase)) and
                (not hasattr(f, 'mode') or append == ('a' in f.mode)))
    return (hasattr(f, 'mode') and (not read or any_in('r+', f.mode)) and (not write or any_in('wa+', f.mode)) and
            (not seek or hasattr(f, 'seek')) and (binary == ('b' in f.mode)) and (append == ('a' in f.mode)))

def isfileobj(f):
    """
    Checks if an object can be used with fromfile and tofile defined above. In Python 2, file-like
    objects will return True but cannot directly be used with numpy.fromfile or ndarray.tofile. They
    must be used with the wrapper functions defined above.

    File-like objects must define 'fileno', 'flush', 'tell', and 'seek' functions and a 'mode'
    attribute.
    """
    if isinstance(f, str if __is_py3 else (basestring,file)): return True
    if any(not hasattr(f,a) for a in ('fileno','flush','tell','seek','mode')): return False
    try: return int(f.fileno()) >= 0
    except IOError: return False

def openfile(f, mode, compression=None, comp_level=9, off=None):
    """
    Tries to make sure that the file is an IOBase file object and has the right mode. Accepts
    strings, IOBase file objects, and regular file-objects with the fileno() function. Supports
    wrapping the file in a compression handler (supports 'deflate', 'zlib', and 'gzip' along with
    'auto' for read streams to figure out what type of compression to use) and can seek the file
    to the starting offset for you (if negative and no compression, will seek from end).

    When changing the mode, the mode doesn't really appear to be changed. For example, changing a
    file from 'r+b' to 'wb' mode does not truncate the file and changing from append to write mode
    still appends (although tell() becomes all messed up).
    """
    if compression not in (None, 'deflate', 'zlib', 'gzip', 'auto') or compression and off is not None and off < 0: raise ValueError
    compressing = compression is not None
    if compression == 'auto': compression = None
    if isinstance(f, str):
        if compressing: return GzipFile(f, mode, method=compression, level=comp_level, start_off=off)
        f = io.open(f, mode)
    elif isinstance(f, io.IOBase) or not __is_py3 and (isinstance(f, file) or hasattr(f, 'fileno')):
        f = io.open(f.fileno(), mode, closefd=False)
    try:
        if compressing: f = GzipFile(f, method=compression, level=comp_level, start_off=off)
        elif off:       f.seek(off, io.SEEK_END if off < 0 else io.SEEK_SET)
    except:
        f.close()
        raise
    return f

def array_read(f, shape, dtype, order='C'):
    """
    Read the array data from a file or file-like object. The shape, dtype, and order are that of the
    data. The shape does not include the dtype shape.
    """
    if isfileobj(f):
        shape += dtype.shape
        return fromfile(f, dtype.base, count=prod(shape)).reshape(shape, order=order)
    arr = empty(shape, dtype, order)
    if f.readinto(arr.data) != len(arr.data): raise ValueError
    return arr

def array_skip(f, shape, dtype):
    """
    Skip the array data from a file or file-like object. The shape, dtype, and order are that of the
    data. The shape does not include the dtype shape.
    """
    f.seek(prod(shape)*dtype.itemsize, io.SEEK_CUR)

def array_save(f, arr, order=None):
    """
    Save the array data to a file or file-like object. Default is to try to figure out the order
    from the given array (defaulting to C if it is not or dual contiguous at all). You can force the
    order by explicitly specifying it.
    """
    if   order == 'F': frtrn = True
    elif order == 'C': frtrn = False
    else: frtrn = arr.flags.f_contiguous and not arr.flags.c_contiguous
    arr = asfortranarray(arr) if frtrn else ascontiguousarray(arr)
    if isfileobj(f): tofile(arr.T if frtrn else arr, f)
    else:
        for c in nditer(arr,flags=('external_loop','buffered'),
                        op_flags=('readonly','no_broadcast','contig','aligned'),
                        buffersize=max(16777216//arr.itemsize,1),
                        order='F' if frtrn else 'C'):
            f.write(c.data)

def array_read_ascii(f, shape, dtype, order='C'):
    """
    Read the array data from a file or file-like object containing the textual respresention of the
    values. The shape, dtype, and order are that of the data. The shape does not include the dtype
    shape.
    """
    if isfileobj(f):
        shape += dtype.shape
        return fromfile(f, dtype.base, count=prod(shape), sep=' ').reshape(shape, order=order)

    # TODO: move this to cython for great speed ups
    arr = empty(shape, dtype, order)
    arr_r = arr.ravel()
    i = 0
    total = arr.size
    s = sx = f.read(max(total*2-1, 0)) # at least one digit and one space per element
    while len(sx) > 0:
        vals = s.split()
        if s[-1].isspace(): s = ''
        else: s = vals[-1]; del vals[-1]
        arr_r[i:i+len(vals)] = vals
        i += len(vals)
        sx = f.read(max((total-i)*2-1, 0))
        s += sx
    arr_r[i:] = s.split()
    return arr

def array_skip_ascii(f, shape, dtype):
    """
    Skip the array data from a file or file-like object containing the textual respresention of the
    values. The shape and dtype are that of the array. The shape does not include the dtype shape.
    """
    # Basically array_read_ascii with parts removed
    # TODO: is this really faster?
    if isfileobj(f):
        shape += dtype.shape
        fromfile(f, dtype.base, count=prod(shape), sep=' ')
    else:
        i = 0
        total = prod(shape+dtype.shape)
        s = sx = f.read(max(total*2-1, 0)) # at least one digit and one space per element
        while len(sx) > 0:
            vals = s.split() # TODO: don't actually split the string - this is wasteful
            if s[-1].isspace(): s = ''
            else: s = vals[-1]; del vals[-1]
            i += len(vals)
            sx = f.read(max((total-i)*2-1, 0))
            s += sx

def array_save_ascii(f, arr, order=None):
    """
    Save the array data to a file or file-like object using the the textual respresention of the
    values. Default is to try to figure out the order from the given array (defaulting to C if it is
    not or dual contiguous at all). You can force the order by explicitly specifying it.
    """
    if   order == 'F': frtrn = True
    elif order == 'C': frtrn = False
    else: frtrn = arr.flags.f_contiguous and not arr.flags.c_contiguous
    arr = asfortranarray(arr) if frtrn else ascontiguousarray(arr)
    if isfileobj(f):
        tofile(arr.T if frtrn else arr, f, sep=' ')
    else: raise RuntimeError() # TODO
        #for c in nditer(arr,flags=('external_loop','buffered'),
        #                op_flags=('readonly','no_broadcast','contig','aligned'),
        #                buffersize=max(16777216//arr.itemsize,1),
        #                order='F' if frtrn else 'C'):
        #    f.write(c.data)

def get_file_size(f):
    """Get the size of a file, either from the filename or the file-object."""
    if isinstance(f, str): return os.path.getsize(f)
    try:
        return os.fstat(f.fileno()).st_size
    except StandardError:
        f.seek(0, io.SEEK_END)
        return f.tell()

def __get_name_from_fd(fd):
    """
    This tries to get a file name from a file descriptor. On Windows this is likely to succeed by
    using GetFileInformationByHandleEx. On Linux and Mac OS X if the file has changed names or has
    been deleted then None will be returned or the name will be flat out wrong. Linux uses readlink
    on /proc/self/fd/# and Mac OS X uses fcntl(F_GETPATH).

    In any case, this will return None in many cases when the path cannot be determined.
    """
    # this function is full of OS-specific imports/members
    #pylint: disable=no-member, import-error
    if os.name == 'nt':
        from ctypes import Structure, windll, byref, sizeof
        from ctypes.wintypes import HANDLE, INT, LPVOID, DWORD, BOOL, WCHAR
        from msvcrt import get_osfhandle

        FileNameInfo = 2
        class FILE_NAME_INFO(Structure):
            _fields_ = [("FileNameLength", DWORD), ("FileName", WCHAR*32767)]

        GetFileInformationByHandleEx = windll.kernel32.GetFileInformationByHandleEx
        GetFileInformationByHandleEx.argtypes = [HANDLE, INT, LPVOID, DWORD]
        GetFileInformationByHandleEx.restype = BOOL

        fni = FILE_NAME_INFO()
        return fni.FileName[:fni.FileNameLength] if GetFileInformationByHandleEx(get_osfhandle(fd), FileNameInfo, byref(fni), sizeof(fni)) else None
    
    try:
        ln = os.readlink('/proc/self/fd/%d' % fd)
        return os.path.abspath(os.path.join('/proc/self/fd', ln))
    except StandardError: pass

    try:
        from fcntl import fcntl
        try: from fcntl import F_GETPATH
        except ImportError: F_GETPATH = 50 # hard code Mac OS X value
        fn = fcntl(fd, F_GETPATH, b'\0' * 1024).rstrip(b'\0')
        return None if len(fn) == 1024 else fn
    except StandardError: pass

    return None

def get_file_name(f):
    """Get the absolute path of a file, either from the filename or file-object."""
    #pylint: disable=unsubscriptable-object
    if isinstance(f, str): return os.path.abspath(f)
    if hasattr(file, 'name'):
        if isinstance(file.name, str) and len(file.name) > 0 and file.name[0] != '<' and file.name[-1] != '>':
            return os.path.abspath(file.name)
        elif isinstance(file.name, Integral):
            return __get_name_from_fd(file.name)
    if hasattr(file, 'fileno'):
        try: return __get_name_from_fd(file.fileno())
        except StandardError: pass
    return None

def _copy_data(f, src, dst, buf):
    f.seek(src)
    read = f.readinto(buf)
    if read == 0: return 0 # all done
    f.seek(dst)
    return f.write(buf[:read]) # Return actual amount of data copied

def _copy_data_complete(f, src, dst, buf):
    f.seek(src)
    n = 0
    while n < len(buf): n += f.readinto(buf[n:])
    f.seek(dst)
    n = 0
    while n < len(buf): n += f.write(buf[n:])

def copy_data(f, src, dst, size=None, truncate=None, buf=16777216):
    """
    Copy data within a single file-like object `f` from `src` offset to `dst` offset of `size`
    bytes possibly truncating the file at `dst`+`size`. If `size` is not provided or is None, all
    data from `src` to the end of the file is copied. If truncate is not provided it defaults to
    False if `size` is given, True otherwise.
    """
    # Parameter and no-copy check
    if src < 0 or dst < 0: raise ValueError
    truncate = (size is None) if truncate is None else bool(truncate)
    file_size = get_file_size(f)
    if size is None or src+size > file_size: size = file_size-src
    src_start, dst_start = src, dst
    src_end, dst_end = src + size, dst + size
    if src == dst or size == 0: # nothing to copy, but may truncate
        if truncate and src_end != file_size: f.truncate(src_end)
        return

    # Setup buffers
    if isinstance(buf, Integral):
        buf_size = buf
        buf_raw = bytearray(buf)
        buf = memoryview(buf_raw) # allows us to slice without copying
    else:
        buf_size = len(buf)

    if src < dst:
        # Copy data moving backwards
        if dst_end > file_size: f.truncate(dst_end)
        src, dst = src_end - buf_size, dst_end - buf_size
        while src > src_start:
            _copy_data_complete(f, src, dst, buf)
            src -= buf_size
            dst -= buf_size
        if src < src_start: _copy_data_complete(f, src_start, dst_start, buf[:buf_size-src_start])
    else:
        # Copy data moving forward
        while size >= buf_size:
            copied = _copy_data(f, src, dst, buf)
            src += copied
            dst += copied
            size -= copied
        if size > 0: _copy_data_complete(f, src, dst, buf[:size])

    # Truncate the file
    if truncate: f.truncate(dst_end)

def fill_data(f, off=0, size=None, val=0, buf_size=16777216):
    """
    Fill a part of a file in with values. The parameters are off (the starting offset, defaults to
    0), size (the number of bytes to fill in, defaults to all after off), val (the value to fill in
    with, defaulting to 0), and a buffer size (default to 16MB).
    """
    from itertools import repeat
    if off < 0: raise ValueError
    file_size = get_file_size(f)
    if size is None or off+size > file_size: size = file_size-off
    if size == 0: return
    sz = min(size, buf_size)
    buf_raw = bytearray(sz if val == 0 else repeat(val, sz))
    buf = memoryview(buf_raw)
    f.seek(off)
    while size >= buf_size: size -= f.write(buf)
    while size > 0:         size -= f.write(buf[:size])

def file_remove_ranges(f, ranges, buf_size=16777216): # 16 MB
    """
    Remove the given ranges from the file, shift all contents after them toward the start of the
    file. This is done with copying data at most once. The ranges must be tuples with start,stop
    file offsets (stop is actually +1 the last file offset removed). The file is truncated at the
    end. The file-like object must support seek, complete readinto, complete write, truncate, and
    get_file_size. By complete, the function must not stop short of the amount of data requested
    (unless end-of-file for readinto).
    """
    from intervals import Interval, closedopen
    intervals = Interval([closedopen(start, stop) for start, stop in ranges])
    if intervals.is_empty(): return # nothing to remove
    keep_ints = Interval([closedopen(0, get_file_size(f))]) - intervals
    if keep_ints.is_empty(): f.truncate(0); return # remove everything
    buf_raw = bytearray(buf_size)
    buf = memoryview(buf_raw) # allows us to slice without copying
    position = 0
    for i in keep_ints:
        n = i.upper_bound - i.lower_bound
        if position != i.lower_bound:
            copy_data(f, i.lower_bound, position, n, buf=buf)
        position += n
    f.truncate(position)

class FileInsertIO(io.BufferedIOBase):
    """
    A class that wraps a file-like-object and inserts the data, pushing data further down in the
    file as necessary. This does use plenty of buffering to make it useable in the middle of large
    files.

    The logic is as follows:
     * if there is room in the file (given with an initial non-zero size), data is written directly
       to the file
     * otherwise the data is written into a buffer (default is 64 MB)
     * if the data would overflow the buffer, the data in the file is moved to make room for the
       buffer and given data, and the entire buffer and data is written to the file and the buffer
       is cleared
     * flush, close, detach, and truncate all make the file consistent in that if there is buffered
       data, room is made for it and it is written, if there is empty room remaining from an initial
       non-zero size then futher data is pulled into the empty space.
    """
    def __init__(self, f, off=None, size=0, buf_size=67108864): #pylint: disable=super-init-not-called
        self.__f = f
        if off is None: off = f.tell()
        elif f.seek(off) != off: raise IOError()
        self.__off_start = off
        self.__off = 0
        self.__room = size
        self.__buf_raw = bytearray(buf_size)
        self.__buf = memoryview(self.__buf_raw)
        self.__buf_size = 0
        self.closed = False

    #pylint: disable=no-self-use
    def isatty(self): return False
    def readable(self): return False
    def writable(self): return True
    def seekable(self): return False
    def fileno(self): raise IOError() # use .raw.fileno() if you want the underlying file number
    
    def tell(self): return self.__off_start + self.__off + self.__buf_size
    @property
    def raw(self): return self.__f
    def detach(self):
        self.flush()
        f = self.__f
        self.__f = None
        return f
    def close(self):
        if self.closed: return
        self.flush()
        self.__f = None
        self.closed = True

    def __make_room(self, nbytes):
        if nbytes < self.__room: return
        off = self.__off_start + self.__off
        copy_data(self._f, off + self.__room, off + nbytes)
        self.__room = nbytes
        if self.seek(off) != off: raise IOError()
    def __remove_room(self):
        if self.__room == 0: return
        off = self.__off_start + self.__off
        copy_data(self._f, off + self.__room, off)
        self.__room = 0
        if self.seek(off) != off: raise IOError()
    def __write_mv(self, mv):
        nbytes = len(mv)
        self.__off += nbytes
        self.__room -= nbytes
        nbytes -= self.__f.write(mv)
        while nbytes > 0: nbytes -= self.__f.write(mv[-nbytes:])

    def flush(self):
        if self.__buf_size > 0:
            self.__make_room(self.__buf_size)
            self.__write_mv(self.__buf[:self.__buf_size])
            self.__buf_size = 0
        elif self.__room > 0:
            self.__remove_room()

    def write(self, b):
        nbytes = len(b)
        if nbytes <= self.__room:
            self.__write_mv(memoryview(b))
        elif nbytes > len(self.__buf_raw) - self.__buf_size:
            self.__make_room(self.__buf_size + nbytes)
            if self.__buf_size:
                self.__write_mv(self.__buf[:self.__buf_size])
                self.__buf_size = 0
            self.__write_mv(memoryview(b))
        else:
            self.__buf[self.__buf_size:self.__buf_size+nbytes] = b
            self.__buf_size += nbytes
        return nbytes

    def truncate(self, size=None):
        if size is None: self.flush()
        else: raise IOError()
