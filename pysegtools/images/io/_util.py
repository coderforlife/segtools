"""Utilities for IO library use."""

import os
from io import open, FileIO, IOBase, BufferedIOBase, SEEK_SET, SEEK_CUR, SEEK_END
from numpy import fromfile, fromstring, nditer, empty

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
        m = FileMode._None
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
    
def isfileobj(f):
    """
    Checks if f is an object that can be given to fromfile and tofile. This includes strings of
    filenames and file objects (from open()). Not included are file-like objects which cannot be
    directly used by fromfile and tofile.
    """
    # TODO: are FileIO and BufferedIOBase usable in NumPy? or only in Python 3?
    return isinstance(f, (basestring, file))

def openfile(f, mode, compression=None, comp_level=9, off=None):
    """
    Tries to make sure that the file is an IOBase file object and has the right mode. Accepts
    strings, IOBase file objects, and regular file-objects with the fileno() function. Supports
    wrapping the file in a compression handler (supports 'deflate', 'zlib', and 'gzip' along with
    'auto' for write streams to figure out what type of compression to use) and can seek the file
    to the starting offset for you (if negative and no compression, will seek from end).
    """
    if compression not in (None, 'deflate', 'zlib', 'gzip', 'auto') or compression and off < 0: raise ValueError
    compressing = compression != None
    if compression == 'auto': compression = None
    if isinstance(f, basestring):
        if compressing: return GzipFile(f, mode, type=compression, level=comp_level, start_off=off)
        f = open(f, mode)
    elif isinstance(f, IOBase): f = f if hasattr(f, 'mode') and mode == f.mode else open(f.fileno(), mode)
    elif isinstance(f, file) or hasattr(f, 'fileno'): f = open(f.fileno(), mode)
    try:
        if compressing: f = GzipFile(f, type=compression, level=comp_level, start_off=off)
        elif off:       f.seek(off, SEEK_END if off < 0 else SEEK_SET)
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

def imskip_raw(f, shape, dtype, order='C'):
    """
    Skip the raw image data from a file or file-like object. The shape, dtype, and order are that of
    the image. The shape does not include the dtype shape.
    """
    f.seek(prod(shape)*dtype.itemsize, SEEK_CUR)

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
    
    # textual array loading do not support 'fancy' dtypes
    view_dtype = None
    if hasattr(dtype, 'fields') and dtype.fields:
        dts = [dt for dt, i in dtype.fields.itervalues()]
        if not all(dt == dts[0] for dt in dts[1:]): raise TypeError
        view_dtype = dtype
        sq_axis = len(shape)
        shape += (len(dts),)
        dtype = dts[0]
    
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

    return im.view(dtype=view_dtype).squeeze(axis=sq_axis) if view_dtype else im

def imskip_ascii_raw(f, shape, dtype):
    """
    Skip the raw image data from a file or file-like object containing the textual respresention of
    the values. The shape and dtype are that of the image. The shape does not include the dtype
    shape.
    """
    # Basically imread_ascii_raw with parts removed
    # TODO: is this really faster?
    if hasattr(dtype, 'fields') and dtype.fields:
        dts = [dt for dt, i in dtype.fields.itervalues()]
        if not all(dt == dts[0] for dt in dts[1:]): raise TypeError
        shape += (len(dts),)
        dtype = dts[0]
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
    if isinstance(f, basestring): return os.path.getsize(f)
    else:
        try: return os.fstat(f.fileno()).st_size
        except: f.seek(0,SEEK_END); return f.tell()
        
def copy_data(f, src, dst, buf):
    """
    Copy data within a single file-like object (f) from one offset (src) to another offset (dst).
    The provided buffer must be a pre-allocated memoryview of the right number of bytes that will
    be copied. The file-like object must support seek, complete readinto, and complete write. By
    complete, the function must not stop short of the amount of data requested.
    """
    # Read from source
    f.seek(src)
    read = f.readinto(buf)
    if read == 0: return 0 # all done

    # Write to destination
    f.seek(dst)
    #written = 0
    #while written < read: written += f.write(buf[written:read])
    f.write(buf[:read])

    # Return actual amount of data copied
    return read

def file_shift_contents(f, old_offset, new_offset, buf_size=16777216): # 16 MB
    """
    Shift file contents within a single file-like object (f) from old_offset to new_offset using a
    buffer (defaulting to 16MB in size). This is a more advanced version of copy_data. It copies all
    data from the old offset to the end to the new offset in chunks making sure not to overwrite
    data if moving forward. If the shift moves data backward, the file is truncated. The file-like
    object must support seek, complete readinto, complete write, truncate, and get_file_size. By
    complete, the function must not stop short of the amount of data requested.
    """
    # f is a file handle opened with r/w and supports readinto
    # we want to shift the data at old_offset to new_offset
    # nothing before min(old_offset, new_offset) will be changed
    # if new_offset > old_offset then zeros will be added
    # if new_offset < old_offset then the file size will decrease
    if old_offset < 0 or new_offset < 0: raise ValueError
    if old_offset == new_offset: return # no change

    # Setup buffers
    buf_raw = bytearray(buf_size)
    buf = memoryview(buf_raw) # allows us to slice without copying
    
    if old_offset < new_offset:
        # Grow the file
        shift = new_offset - old_offset
        old_size = get_file_size(f)
        new_size = old_size + shift
        f.truncate(new_size)
        
        # Copy data moving backwards
        orig_old_off, orig_new_off = old_offset, new_offset
        old_offset, new_offset = old_size - buf_size, new_size - buf_size
        while old_offset > orig_old_off:
            if copy_data(f, old_offset, new_offset, buf) != buf_size: raise IOError
            old_offset -= buf_size
            new_offset -= buf_size
        if old_offset < orig_old_off:
            rem = old_offset+buf_size-orig_old_off
            if copy_data(f, orig_old_off, orig_new_off, buf[:rem]) != rem: raise IOError

        # Fill in with zeros from orig_old_off to orig_new_off
        f.seek(orig_old_off)
        buf_raw = bytearray(min(shift, buf_size)) # re-initializes to all 0s of the right size
        buf = memoryview(buf_raw)
        while shift >= buf_size:
            f.write(buf)
            shift -= buf_size
            #shift -= f.write(buf)
        f.write(buf[:shift])
        #written = 0
        #while written < shift: written += f.write(buf[written:shift])

    else:
        # Copy data moving forward
        while True:
            copied = copy_data(f, old_offset, new_offset, buf)
            if copied == 0: break # all done
            old_offset += copied
            new_offset += copied
        
        # Shrink the file
        f.truncate(new_offset)

