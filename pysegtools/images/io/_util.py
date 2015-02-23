"""Utilities for IO library use."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, io
from numpy import fromfile, nditer, empty, prod
from numbers import Integral

from .._util import String
from ...general.gzip import GzipFile
from ...general.enum import Flags

class FileMode(int, Flags):
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

__is_py3 = sys.version_info[0] == 3
def isfileobj(f):
    """
    Checks if f is an object that can be given to fromfile and tofile. This includes strings of
    filenames and file objects (from open()). Not included are file-like objects which cannot be
    directly used by fromfile and tofile.
    """
    if sys.version_info[0] == 3:
        return isinstance(f, (String, io.FileIO)) or (isinstance(f, io.BufferedIOBase) and hasattr(f, 'raw') and isinstance(f.raw, io.FileIO))
    else:
        return isinstance(f, (String, file))

def openfile(f, mode, compression=None, comp_level=9, off=None):
    """
    Tries to make sure that the file is an IOBase file object and has the right mode. Accepts
    strings, IOBase file objects, and regular file-objects with the fileno() function. Supports
    wrapping the file in a compression handler (supports 'deflate', 'zlib', and 'gzip' along with
    'auto' for write streams to figure out what type of compression to use) and can seek the file
    to the starting offset for you (if negative and no compression, will seek from end).
    """
    if compression not in (None, 'deflate', 'zlib', 'gzip', 'auto') or compression and off < 0: raise ValueError
    compressing = compression is not None
    if compression == 'auto': compression = None
    if isinstance(f, String):
        if compressing: return GzipFile(f, mode, type=compression, level=comp_level, start_off=off)
        f = io.open(f, mode)
    elif isinstance(f, io.IOBase): f = f if hasattr(f, 'mode') and mode == f.mode else io.open(f.fileno(), mode)
    elif sys.version_info[0] == 2 and (isinstance(f, file) or hasattr(f, 'fileno')): f = io.open(f.fileno(), mode)
    try:
        if compressing: f = GzipFile(f, type=compression, level=comp_level, start_off=off)
        elif off:       f.seek(off, io.SEEK_END if off < 0 else io.SEEK_SET)
    except:
        f.close()
        raise
    return f

def imread_raw(f, shape, dtype, order='C'):
    """
    Read the raw image data from a file or file-like object. The shape, dtype, and order are that of
    the image. The shape does not include the dtype shape.
    """
    if isfileobj(f):
        full_shape = (shape+dtype.shape) if hasattr(dtype, 'shape') and dtype.shape else shape
        return fromfile(f, dtype, count=prod(shape)).reshape(full_shape, order=order)
    else:
        im = empty(shape, dtype, order)
        if f.readinto(im.data) != len(im.data): raise ValueError
        return im

def imskip_raw(f, shape, dtype):
    """
    Skip the raw image data from a file or file-like object. The shape, dtype, and order are that of
    the image. The shape does not include the dtype shape.
    """
    f.seek(prod(shape)*dtype.itemsize, io.SEEK_CUR)

def imsave_raw(f, im):
    """Save the raw image data to a file or file-like object."""
    frtrn = im.flags.f_contiguous and not im.flags.c_contiguous
    if isfileobj(f):
        if frtrn: im.T.tofile(f)
        else: im.tofile(f)
    else:
        for c in nditer(im,flags=['external_loop','buffered'],buffersize=max(16777216//im.itemsize,1),order='F' if frtrn else 'C'):
            f.write(c.data)

def imread_ascii_raw(f, shape, dtype, order='C'):
    """
    Read the raw image data from a file or file-like object containing the textual respresention of
    the values. The shape, dtype, and order are that of the image. The shape does not include the
    dtype shape.
    """
    if isfileobj(f):
        if hasattr(dtype, 'shape') and dtype.shape:
            shape += dtype.shape
            dtype = dtype.base
        im = fromfile(f,dtype,count=prod(shape),sep=' ').reshape(shape,order=order)
    else:
        im = empty(shape, dtype, order)
        im_r = im.ravel()
        i = 0
        total = im.size
        s = sx = f.read(max((total-i)*2-1, 0)) # at least one digit and one space per element
        while len(sx) > 0:
            vals = s.split()
            if s[-1].isspace(): s = ''
            else: s = vals[-1]; del vals[-1]
            im_r[i:i+len(vals)] = vals
            i += len(vals)
            sx = f.read(max((total-i)*2-1, 0))
            s += sx
        im_r[i:] = s.split()
    return im

def imskip_ascii_raw(f, shape, dtype):
    """
    Skip the raw image data from a file or file-like object containing the textual respresention of
    the values. The shape and dtype are that of the image. The shape does not include the dtype
    shape.
    """
    # Basically imread_ascii_raw with parts removed
    # TODO: is this really faster?
    if isfileobj(f):
        if hasattr(dtype, 'shape') and dtype.shape:
            shape += dtype.shape
            dtype = dtype.base
        fromfile(f,dtype,count=prod(shape),sep=' ')
    else:
        i = 0
        total = prod(shape+dtype.shape)
        s = sx = f.read(max((total-i)*2-1, 0)) # at least one digit and one space per element
        while len(sx) > 0:
            vals = s.split()
            if s[-1].isspace(): s = ''
            else: s = vals[-1]; del vals[-1]
            i += len(vals)
            sx = f.read(max((total-i)*2-1, 0))
            s += sx

def imsave_ascii_raw(f, im):
    """Save the raw image data to a file or file-like object using the the textual respresention of the values."""
    frtrn = im.flags.f_contiguous and not im.flags.c_contiguous
    if isfileobj(f):
        if frtrn: im.T.tofile(f, sep=' ')
        else: im.tofile(f, sep=' ')
    else:
        # TODO
        for c in nditer(im,flags=['external_loop','buffered'],buffersize=max(16777216//im.itemsize,1),order='F' if frtrn else 'C'):
            f.write(c.data)

def get_file_size(f):
    """Get the size of a file, either from the filename, the file-number, or seeking and telling."""
    if isinstance(f, String): return os.path.getsize(f)
    else:
        try:
            return os.fstat(f.fileno()).st_size
        except OSError:
            f.seek(0, io.SEEK_END)
            return f.tell()

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
    file. This is done with copying data at most once. The ranges must be tuples with start stop
    file offsets (stop is actually +1 the last file offset removed). The file is truncated at the
    end. The file-like object must support seek, complete readinto, complete write, truncate, and
    get_file_size. By complete, the function must not stop short of the amount of data requested
    (unless end-of-file for readinto).
    """
    from ...general.interval import Interval, IntervalSet
    intervals = IntervalSet(Interval(start,stop,upper_closed=False) for start,stop in ranges)
    if not intervals: return # nothing to remove
    keep_ints = IntervalSet([Interval(0, get_file_size(f), upper_closed=False)]) - intervals
    if not keep_ints: f.truncate(0); return # remove everything
    buf_raw = bytearray(buf_size)
    buf = memoryview(buf_raw) # allows us to slice without copying
    position = 0
    for i in keep_ints:
        n = i.upper_bound - i.lower_bound
        if position != i.lower_bound:
            copy_data(f, i.lower_bound, position, n, buf=buf)
        position += n
    f.truncate(position)
