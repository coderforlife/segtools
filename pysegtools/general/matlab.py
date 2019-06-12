"""
A library for reading, writing, and modifying MATLAB files. Supports MATLAB v4, v5 (6/7), and v7.3
formats. Most features of those formats are supported and unified into a single interface.

To open or create a file, use the openmat method. It takes the file name, the mode (one of r, r+, w
or w+), and a file version number (if creating a new file). The MATFile object returned is a
sequence/dictionary-like object. In general, only entries that are usable are enumerated (so entries
that are not supported are not listed). Supports:
    File Operations:
      x.filename        the filename
      x.rename(...)     don't use this yet...
      x.flush()         flush the underlying file
      x.close()         close the underlying file, all further actions are not allowed
      x.header          dict with info about the file, versions may have different entries
    Sequence Operations:
      iter(x)           MATEntry objects (details later), v4 and v5 files are given in file-order
      len(x)            number of usable entries
      x.total_entries   number of total entries in the file
      'name' in x       check if an entry is in the file and is usable
      x['n']            get an entry by array name
    Modification Operations:
      * Note: v7.3 does not support ordering, so any of these operations are not guaranteed to
        maintain any sort of order, even of previous entries
      * Note: v4 and v5 files will be potentially very slow when doing insert_before, del, or
        setting as lots of data might have to be copied/moved
      x.get('n', [def]) get an entry by array name, or a default
      x.append('n', a)  add a new entry to the end* of the file with the given name and data
      x.insert_before('n', 'm', a)   add a new entry before* another entry in the file
      x.set('n', a)     modify the data for an entry or append a new entry
      x['n'] = a        same as x.set('n', a)
      del x['n']        remove an entry

The entries are given as MATEntry objects:
    x.name    the name of the entry
    x.header  dict with info about the file, versions may have different entries
    x.shape   the shape of the entry, same as would be from x.data.shape but without loading it
    x.dtype   the dtype of the entry, same as would be from x.data.dtype but without loading it
    x.data    the data of the entry, returned as a numpy array

Main takeaway point is that the data is not loaded until requested. It is also not cached.
    
Character arrays are returned as arrays containing single unicode characters (and ASCII characters
from v4 files). Cell arrays are returned as object arrays with every object being an ndarray.
Structs are returned as structed arrays with every element being an ndarray object. Simple objects
are like structs but their dtype has a metadata entry of class_name. This also works while saving.
No format supports functions and opaque/complex objects (e.g. Java objects). The dtype can also
include the metadata 'global' to mark an entry as a global variable.

The v4 and v5 formats are written in pure Python, and besides standard library modules only requires
NumPy (and SciPy for sparse matrices). It also requires the local gzip and io modules. The v4
reader/writer supports everything except VAX D-float, VAX G-float, and Cray numeric formats. The v5
reader/writer supports everything except functions and opaque objects.

The v7.3 reader/writer uses h5py to do the actual reading and writing while it manages all of the
MATLAB specific details. If this module is not available, v7.3 files cannot be read or written. Its
reader/writer also supports everything besides functions and opaque objects.
"""

# This information was acquired through looking at .MAT files along with the following documents:
#    "MAT-File Format" by Mathworks, R2015a edition, covering v4 and v5 formats, in PDF format
#    US Patent 20080016023 describing v7.3 format
# Additionally, the SciPy MAT v4/5 readers and writers were looked at for inspirations as well

# Pylint says all of these code needs lots of refactoring... (20 instances of the following)
#pylint: disable=too-many-branches, too-many-locals, too-many-statements, too-many-arguments, too-many-return-statements

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re, codecs, io
from collections import OrderedDict
from itertools import izip, islice
from struct import Struct, pack, unpack, error as StructError
from weakref import proxy
from abc import ABCMeta, abstractproperty, abstractmethod
from warnings import warn

from numpy import ndarray, array, empty, zeros, frombuffer, fromiter, asanyarray, ascontiguousarray, asfortranarray
from numpy import dtype, concatenate, tile, delete, char, vectorize, iinfo, nditer
from numpy import (complex64, complex128, float32, float64, bool_, unicode_, string_, object_, void,
                   int8, int16, int32, int64, intc, uint8, uint16, uint32, uint64)

from .io import array_read, array_save, get_file_size, copy_data, file_remove_ranges, FileInsertIO
from . import GzipFile, prod, sys_endian, sys_64bit

try:
    import h5py
    _h5py_avail = True
except ImportError:
    _h5py_avail = False

try:
    import scipy.sparse
    _is_sparse = scipy.sparse.isspmatrix
except ImportError:
    _is_sparse = lambda m:False

__all__ = ['get_mat_version', 'openmat', 'is_invalid_matlab_name', 'mat_nice']

def _dt_eq_str(dt, s): return isinstance(dt, str) and dt == s

def _complexify_dtype(dt):
    dtb = dt.base # removes channel information, if there
    if dtb.kind != 'f': return dt if dt.shape == (2,) else dtype((dt,2))
    return dtype('c'+str(dtb.itemsize* 2)).newbyteorder(dtb.byteorder)

def _decomplexify_dtype(dt):
    return dt if dt.kind != 'c' else dtype('f'+str(dt.itemsize//2)).newbyteorder(dt.byteorder)

def _decomplexify(data):
    return data.view(dtype=_decomplexify_dtype(data.dtype))

def _create_dtype(base, big_endian, channels=1):
    return dtype((base, channels)).newbyteorder(big_endian if isinstance(big_endian, str) else ('>' if big_endian else '<'))

def _get_dtype_endian(dt):
    """Get a '<' or '>' from a dtype's byteorder (which can be |, =, <, or >)."""
    endian = dtype(dt).byteorder
    if endian == '|': return '<' # | means N/A (single byte), report as little-endian
    elif endian == '=': return sys_endian # is native byte-order
    return endian


__matlab_keywords = frozenset((
    'break', 'case', 'catch', 'classdef', 'continue', 'else', 'elseif', 'end', 'for', 'function'
    'global', 'if', 'otherwise', 'parfor', 'persistent', 'return', 'spmd', 'switch', 'try', 'while'))
__re_matlab_name = re.compile('^[a-zA-Z][a-zA-Z0-9_]{0,62}$')
def is_invalid_matlab_name(name): return name in __matlab_keywords or __re_matlab_name.match(name) is None

__platforms = { 'nt':b'PCWIN', 'linux':b'GLNXA64' if sys_64bit else b'GLNX86' }
def _get_matlab_header(base):
    from time import asctime
    from os import name
    return base % (__platforms.get(name, name), asctime()) 

def _squeeze2(a, min_dim = 2):
    """
    Makes sure the array is at least 2D and squeezes dims above 2D. It also makes character arrays
    use single character values.
    """
    k = a.dtype.kind
    if a.shape == (): a = a.reshape(1)
    if   k == 'S' and a.itemsize != 1: a = a.view(dtype((string_, 1))).reshape(a.shape + (a.itemsize,))
    elif k == 'U' and a.itemsize != 4: a = a.view(dtype((unicode_, 1))).reshape(a.shape + (a.itemsize//4,))
    if a.ndim < min_dim: return a.reshape((1,)*(min_dim-1) + (a.size,), order='F')
    for i in reversed([i for i in xrange(min_dim, a.ndim) if a.shape[i] == 1]): a = a.squeeze(i)
    return a
def _as_savable_array(x):
    """
    Converts the input to a standardized numpy array for saving to a MATLAB file. The following are
    allowed:
        bool, bool_, Integral, Real, Complex   ->   scalar array
        bytes, string, unicode                 ->   character array
        ndarray of the any of the above        ->   unchanged
        sparse matrix                          ->   unchanged
        lists/tuples that can be converted directly into arrays -> the array equivilent
        lists/tuples of savables               ->   "cell" array (dtype=object, items are ndarrays)
            also takes ndarrays of objects
        dictionary of string->savables         ->   "struct" (structured array)
            keys must be valid MATLAB names
            also takes structured arrays

    Every array is also passed through _squeeze2.
    """
    from numbers import Integral, Real, Complex
    from collections import Mapping
    if isinstance(x, (bool, bool_, Integral, Real, Complex)): return array(x).reshape((1,1)) # scalar
    if isinstance(x, (bytes, str)): return _squeeze2(array(x)) # string -> char array
    if _is_sparse(x) and x.dtype.kind in 'buifc': return x # sparse array pass through
    if isinstance(x, Mapping): # dict -> struct
        names = x.keys()
        if any(not isinstance(fn, str) or is_invalid_matlab_name(fn) for fn in names): raise ValueError("Invalid struct field name")
        out = zeros((1,1), dtype=dtype([(str(fn),object_) for fn in x.names]))
        for fn,v in izip(names, x.values()): out[0,0][fn] = _as_savable_array(v)
        return out
    if isinstance(x, (list, tuple)):
        if len(x) == 0: return array(x).reshape((0,0))
        x = array(x)
    if isinstance(x, ndarray):
        if x.dtype.kind in 'buifcUS': return _squeeze2(x)
        if x.dtype.kind == 'O':
            x_flat = x.flat
            for i in xrange(x.size): x_flat[i] = _as_savable_array(x_flat[i])
            return _squeeze2(x)
        if x.dtype.kind == 'V' and len(x.dtype.names):
            fns = x.dtype.names
            if any(is_invalid_matlab_name(k) for k in fns): raise ValueError("Invalid struct field name")
            fs = [(fn,object_) for fn in fns]
            dt = dtype(fs) if x.dtype.metadata is None else dtype(fs, metadata=x.dtype.metadata)
            out = zeros(x.shape, dtype=dt)
            out_flat, x_flat = out.flat, x.flat
            for i in xrange(x.size):
                for fn in fns: out_flat[i][fn] = _as_savable_array(x_flat[i][fn])
            return _squeeze2(out)
        raise ValueError("Array format cannot be save to MATLAB file")
    raise ValueError("Value cannot be saved to MATLAB file")

class _MATFile(object):
    __metaclass__ = ABCMeta
    # subclasses must also have a _version property for the header property
    # subclasses must also define class methods for open(fn, mode) and create(fn, mode)
    # The collection/mapping functions (e.g. len, in, [], and iterating) ignore/skip dummy entries

    _version = None
    @classmethod
    def _basic_open(cls, fn, mode):
        return io.open(fn, mode+'b')

    def __init__(self, filename, f, entries):
        self._filename = filename
        self._f = f
        self._entries = entries
    @property
    def filename(self): return self._filename
    @abstractmethod
    def rename(self, renamer, filename): pass
    def flush(self): self._f.flush()
    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def __str__(self):  return "<MATLAB v%g file '%s'>" % (self._version, self._filename)
    def __repr__(self): return "<MATLAB v%g file '%s'>" % (self._version, self._filename)
    @property
    def header(self):
        return {'version':self._version,'total-entries':self.total_entries,'num-usable-entries':len(self)}
    def __len__(self):
        return sum(1 for e in self._entries.itervalues() if not isinstance(e, _MATDummyEntry))
    @property
    def total_entries(self):
        return len(self._entries)
    def __contains__(self, name):
        return name in self._entries and not isinstance(self._entries[name], _MATDummyEntry)
    def __getitem__(self, name):
        e = self._entries[name]
        if isinstance(e, _MATDummyEntry): raise KeyError()
        return e
    def get(self, name, default=None):
        try: return self[name]
        except KeyError: return default
    def __iter__(self):
        return (e for e in self._entries.itervalues() if not isinstance(e, _MATDummyEntry))
    @abstractmethod
    def append(self, name, data): pass # returns the new entry
    @abstractmethod
    def insert_before(self, name, new_name, data): pass # returns the new entry
    @abstractmethod
    def set(self, name, data): pass # returns the new entry
    def __setitem__(self, name, data): self.set(name, data)
    @abstractmethod
    def __delitem__(self, name): pass

class _MATEntry(object):
    __metaclass__ = ABCMeta
    def __init__(self, name, shape, dt):
        self._name = name
        self._shape = shape
        self._dtype = dt
    @property
    def name(self): return self._name
    def __str__(self):  return "<MATLAB entry '%s', %s %s>" % (self._name, 'x'.join(str(x) for x in self._shape), self.dtype.name)
    def __repr__(self): return "<MATLAB entry '%s', %s %s>" % (self._name, 'x'.join(str(x) for x in self._shape), self.dtype.name)
    @property
    def header(self): return {'name':self._name}
    @property
    def shape(self): return self._shape # always the full shape, except for real/imag channels [also note that I believe always this will be 2D or greater with all higher-dimensions squeezed]
    @property
    def dtype(self): return self._dtype # always a "base" dtype except for the rare non-native complex types which will give 2 channels
    @abstractproperty
    def data(self): pass

class _MATDummyEntry(_MATEntry): # a placeholder for an entry that we cannot process
    def __init__(self, name): self._name = name #pylint: disable=super-init-not-called
    def __str__(self):  return '<MATLAB entry %s (not available)>' % (self._name)
    def __repr__(self): return '<MATLAB entry %s (not available)>' % (self._name)
    @property
    def header(self): raise RuntimeError()
    @property
    def shape(self): raise RuntimeError()
    @property
    def dtype(self): raise RuntimeError()
    @property
    def data(self): raise RuntimeError()

########### MATLAB v4 and v5 Files ###########
class _MAT45File(_MATFile): # a v4 or v5 file
    # subclass must define a class-variable '_entry' which is the class type to make entries with
    _entry = None
    _endian = None
    #pylint: disable=protected-access
    @classmethod
    def _open(cls, filename, f, *args): # raises ValueError if probably not a MATLAB v4/5 file
        entries = OrderedDict()
        mat = cls(filename, f, entries, *args)
        try:
            while True:
                e = cls._entry.open(mat, f)
                name = ('#%03d'%len(entries)) if e.name is None else e.name
                entries[name] = e
        except EOFError: pass
        #except StructError as ex: raise ValueError(ex.message)
        return mat

    def rename(self, renamer, filename):
        mode = self._f.mode
        self._f.close()
        renamer(filename)
        self._f = io.open(filename, mode)
        self._filename = filename
    
    def append(self, name, data):
        if name in self._entries or is_invalid_matlab_name(name): raise KeyError()
        self._f.seek(0, io.SEEK_END)
        self._entries[name] = entry = self._entry.create(self, self._f, name, data)
        return entry
    
    def insert_before(self, name, new_name, data):
        #pylint: disable=undefined-loop-variable
        if new_name in self._entries or is_invalid_matlab_name(new_name): raise KeyError()
        
        for idx, (n,cur_entry) in enumerate(self._entries.iteritems()):
            if n == name: break
        else: raise KeyError()
        start = cur_entry._start
        new_size = self._entry._calc_size(self, new_name, data)
        if new_size == -1:
            with FileInsertIO(self._f, start, 0) as f:
                entry = self._entry.create(self, f, new_name, data)
        else:
            copy_data(self._f, start, start + new_size)
            if self._f.seek(start) != start: raise IOError()
            entry = self._entry.create(self, self._f, new_name, data)
        self._entries[name] = entry # currently added to the end of the order, but during the next loop we fix that

        # Update the entry starts
        for n in list(islice(self._entries, idx, None)):
            self._entries[n] = e = self._entries.pop(n) # remove entry and put it at end
            e._update_start(e._start + new_size)
        
        return entry
    
    def set(self, name, data):
        #pylint: disable=undefined-loop-variable
        if len(self._entries) > 0 and name == next(reversed(self._entries)):
            # We are replacing the last entry, this one is easy (delete last entry and append)
            e = self._entries.popitem()[1]
            e._deleting()
            self._f.truncate(e._start)
            return self.append(name, data)
        items = self._entries.iteritems()
        for idx, (n,cur_entry) in enumerate(items):
            if n == name: break
        else: return self.append(name, data) # doesn't exist yet, add it to the end
        next_entry = next(items)[1]

        # Write the new entry
        start, next_start = cur_entry._start, next_entry._start
        new_size = self._entry._calc_size(self, name, data)
        if new_size == -1:
            with FileInsertIO(self._f, start, next_start - start) as f:
                entry = self._entry.create(self, f, name, data)
            new_end = self._f.tell()
        else:
            new_end = start + new_size
            copy_data(self._f, next_start, new_end)
            if self._f.seek(start) != start: raise IOError()
            entry = self._entry.create(self, self._f, name, data)
        self._entries[name] = entry
        delta = new_end - next_start
        
        # Update the entry starts
        for e in islice(self._entries.itervalues(), idx+1, None):
            e._update_start(e._start + delta)
        
        return entry
    
    def __delitem__(self, name):
        from numpy import append, cumsum, diff
        starts = append(fromiter(e._start for e in self._entries.itervalues()), get_file_size(self._f))

        if isinstance(name, str):
            # Simple case: just one name
            #pylint: disable=undefined-loop-variable
            for idx, n in enumerate(self._entries):
                if n == name: break
            else: raise KeyError()
            ranges = [(starts[idx], starts[idx+1])]
            file_remove_ranges(self._f, ranges)
            self._entries[name]._deleting()
            del self._entries[name]
            idxs = [idx]

        else:
            # Find the indicies for the names
            names = name
            if len(names) == 0: return
            lookup = OrderedDict(izip(self._entries, xrange(len(self._entries))))
            idxs = array(sorted(lookup[n] for n in names))

            # Get the ranges for all the indices, then collapse them, then remove them from the file
            rngs = concatenate((starts[idxs,None], starts[idxs+1,None]), axis=1)
            ranges = [list(rngs[0])]
            for r in rngs[1:]:
                if r[0] == ranges[-1][1]: ranges[-1][1] = r[1]
                else:                     ranges.append(list(r))
            file_remove_ranges(self._f, ranges)

            # Delete the entries
            for n in names:
                self._entries[n]._deleting()
                del self._entries[n]
        
        # Update the entry starts
        starts = cumsum(delete(diff(starts), idxs))
        idx = idxs[0]
        for e,start in izip(islice(self._entries.itervalues(), idx, None), starts[idx:]):
            e._update_start(start)
            
class _MAT45Entry(_MATEntry): # a v4 or v5 entry
    # subclasses must define class methods for open(mat,f) and create(mat,f,name,data)
    @abstractproperty
    def data(self): pass
    @classmethod
    def _calc_size(cls, mat, name, data): #pylint: disable=unused-argument
        return -1 # means the size could not be calculated
    def __init__(self, mat, start, off, name, raw_shape, shape, raw_dt, dt):
        self._mat = proxy(mat)
        self._start = start
        self._off = off
        self._raw_shape = raw_shape
        self._raw_dt = raw_dt
        super(_MAT45Entry, self).__init__(name, shape, dt)
    def _update_start(self, new_start):
        delta = new_start - self._start
        self._start = new_start
        self._off += delta
    def _deleting(self): pass
class _MAT45DummyEntry(_MATDummyEntry): # a v4 or v5 dummy entry
    def __init__(self, mat, start, name):
        self._mat = proxy(mat)
        self._start = start
        super(_MAT45DummyEntry, self).__init__(name)
    def _update_start(self, new_start): self._start = new_start

###### MATLAB v4 Files ######
class _MAT4Entry(_MAT45Entry):
    #pylint: disable=protected-access
    __long_le = Struct('<l')
    __structs = (Struct('<lllll'), Struct('>lllll'))
    __HDR_SIZE = 20
    
    # Numeric format:
    __LITTLE_ENDIAN = 0
    __BIG_ENDIAN = 1
    #__VAX_D_FLOAT = 2, __VAX_G_FLOAT = 3, __CRAY = 4
    __NUMERIC_FORMATS = (__LITTLE_ENDIAN, __BIG_ENDIAN)
    
    # Matrix Type:
    __FULL_MATRIX = 0
    __TEXT_MATRIX = 1
    __SPARSE_MATRIX = 2
    __MATRIX_TYPES = (__FULL_MATRIX, __TEXT_MATRIX, __SPARSE_MATRIX)

    __type2dtype = {
        0: float64,
        1: float32,
        2: int32,
        3: int16,
        4: uint16,
        5: uint8,
    }
    __dtype2type = {
        float64: 0,
        float32: 1,
        int32:   2,
        int16:   3,
        uint16:  4,
        uint8:   5,
    }

    @classmethod
    def openable(cls, header):
        if len(header) == 0: return True # v4 supports 0-length files
        if len(header) < cls.__HDR_SIZE: return False
        mopt = cls.__long_le.unpack_from(header)[0]
        endian = cls.__LITTLE_ENDIAN if (0<=mopt<5000) else cls.__BIG_ENDIAN
        mopt, _, _, imagf, namlen = cls.__structs[endian].unpack_from(header)
        if mopt >= 0 and imagf in (0,1) and namlen > 0:
            M, mopt = divmod(mopt, 1000)
            O, mopt = divmod(mopt, 100)
            P, T    = divmod(mopt, 10)
            return M == endian and O == 0 and P <= 5 and T <= 2 
        return False
    
    @classmethod
    def open(cls, mat, f):
        ### Always advances to after the data (except when raising ValueError)
        ### Parse and check header
        start = f.tell()
        header = f.read(cls.__HDR_SIZE)
        if len(header) == 0: raise EOFError()
        elif len(header) < cls.__HDR_SIZE: raise ValueError()
        if mat._endian is None:
            mopt = cls.__long_le.unpack_from(header)[0]
            mat._endian = endian = cls.__LITTLE_ENDIAN if (0<=mopt<5000) else cls.__BIG_ENDIAN
            mat._struct = cls.__structs[endian]
        else:
            endian = mat._endian
        mopt, mrows, ncols, imagf, namlen = mat._struct.unpack(header)
        if mopt < 0 or imagf not in (0,1) or namlen == 0: raise ValueError()
        M, mopt = divmod(mopt, 1000)
        O, mopt = divmod(mopt, 100)
        P, T    = divmod(mopt, 10)
        if M > 4 or O != 0 or P > 5 or T > 2: raise ValueError()
        ### Get data information
        name = str(f.read(namlen).rstrip(b'\0'))
        cmplx = imagf==1
        raw_dt = dt = _create_dtype(cls.__type2dtype[P], endian==cls.__BIG_ENDIAN, 2 if cmplx else 1)
        col_size = mrows*dt.itemsize
        total_size = ncols*col_size
        off = start+cls.__HDR_SIZE+namlen
        end = off+total_size
        if M not in cls.__NUMERIC_FORMATS or T not in cls.__MATRIX_TYPES:
            if f.seek(total_size, io.SEEK_CUR) != end: raise ValueError()
            return _MAT45DummyEntry(mat, start, name)
        if M != endian: raise ValueError('Endian mismatch') # NOTE: we assume all entries have the same endian-ness
        ### Handle sparse matrix
        if T == cls.__SPARSE_MATRIX:
            if cmplx or (ncols not in (3,4)) or mrows < 1: raise ValueError()
            data_size = col_size*(ncols-2)
            dt_sz = dt.itemsize
            if ncols == 4: dt = _complexify_dtype(dt)
            col_size -= dt_sz
            f.seek(col_size, io.SEEK_CUR); mr = frombuffer(f.read(dt_sz), raw_dt, 1)
            f.seek(col_size, io.SEEK_CUR); nc = frombuffer(f.read(dt_sz), raw_dt, 1)
            if f.seek(data_size, io.SEEK_CUR) != end: raise ValueError()
            shape = (int(mr[0]),int(nc[0]))
        else:
            if f.seek(total_size, io.SEEK_CUR) != end: raise ValueError()
            shape = (mrows,ncols)
            if T == cls.__TEXT_MATRIX: dt = dtype((string_, 1))
            elif cmplx:                dt = _complexify_dtype(dt)
                
        ### Make the entry
        return cls(mat, start, off, name, (mrows,ncols), shape, raw_dt, dt)
    
    @classmethod
    def create(cls, mat, f, name, data):
        start = f.tell()
        if mat._endian is None:
            mat._endian = endian = cls.__LITTLE_ENDIAN if sys_endian == '<' else cls.__BIG_ENDIAN
            mat._struct = cls.__structs[endian]
        else:
            endian = mat._endian

        ### Get, correct, and check the properties of the data
        data = _squeeze2(asanyarray(data))
        if data.kind == 'U': char.encode(data, 'latin1') # only 8-bit is supported


        dt = raw_dt = data.dtype
        shape = raw_shape = data.shape
        is_text = dt.kind == 'S'
        is_sparse = _is_sparse(data)
        is_complex = dt.kind == 'c' or len(shape) > 2 and shape[-1] == 2
        if is_text: # text is saved as a float64 matrix
            raw_dt = dtype(float64)
        elif is_complex:
            if dt.kind == 'c': raw_dt = _decomplexify_dtype(dt)
            else:
                raw_shape = shape = shape[:-1]
                raw_dt, dt = dtype((dt, 2)), _complexify_dtype(dt)
        if raw_dt.base.type not in cls.__dtype2type: raise ValueError('Invalid data type for MATLAB v4 file')
        if len(raw_shape) != 2: raise ValueError('Invalid shape for MATLAB v4 file (must be 2D)')
        if prod(data.shape) > 100000000: raise ValueError('MATLAB v4 files have a max of 100,000,000 elements')

        # Deal with text data
        if is_text:
            data = data.view(uint8).reshape(raw_shape).astype(float64)

        # Deal with sparse data
        elif is_sparse:
            from numpy import add
            if not scipy.sparse.isspmatrix_coo(data): data = data.tocoo()
            sparse, raw = data, data.data
            if is_complex:
                is_complex = False # is_complex stays False since the raw data is not complex
                raw_dt = raw_dt.base
                data = empty((len(raw)+1,4), dtype=raw_dt, order='F')
                data[:-1,2:] = raw.view(dtype=dtype((raw_dt,2)))
            else:
                data = empty((len(raw)+1,3), dtype=raw_dt, order='F')
                data[:-1,2] = raw
            del raw
            raw_shape = data.shape
            data[-1,0:2] = shape
            data[-1,2:] = 0
            add(sparse.row, 1, out=data[:-1,0])
            add(sparse.col, 1, out=data[:-1,1])
            del sparse
            
        # Deal with complex data
        elif is_complex:
            data = data.view(dtype=raw_dt)
            
        # Make sure the data is the right byte and Fortran ordering
        if raw_dt.base.type not in cls.__dtype2type: raise ValueError('Invalid data type for MATLAB v4 file')
        if (_get_dtype_endian(data.dtype)=='<') != (endian==cls.__LITTLE_ENDIAN): data.byteswap(True)
        data = asfortranarray(data)
        if prod(raw_shape) * raw_dt.itemsize > 2147483648: raise ValueError('MATLAB v4 files have a max of 2,147,483,648 bytes per variables')

        ### Create the header
        M = endian
        P = cls.__dtype2type[raw_dt.base.type]
        T = cls.__TEXT_MATRIX if is_text else (cls.__SPARSE_MATRIX if is_sparse else cls.__FULL_MATRIX)
        mopt = 1000*M+10*P+T
        mrows, ncols = raw_shape
        imagf = 1 if is_complex else 0
        if not isinstance(name, bytes): name = str(name).encode('ascii','ignore')
        name += b'\x00'
        namlen = len(name)
        if (f.write(mat._struct.pack(mopt, mrows, ncols, imagf, namlen)) != cls.__HDR_SIZE or
            f.write(name) != namlen): raise IOError('Unable to write header')
        off = start + cls.__HDR_SIZE + namlen
        
        ### Save data
        array_save(f, data)

        ### Make the entry
        return cls(mat, start, off, name, raw_shape, shape, raw_dt, dt)

    @classmethod
    def _calc_size(cls, mat, name, data):
        data = _squeeze2(asanyarray(data))
        dt = data.dtype
        shape = data.shape
        is_text = dt.kind == 'S'
        is_complex = dt.kind == 'c' or len(shape) > 2 and shape[-1] == 2
        if is_text: dt = dtype(float64)
        elif is_complex:
            if dt.kind == 'c': dt = _decomplexify_dtype(dt).base
            else: shape = shape[:-1]
        if dt.type not in cls.__dtype2type: raise ValueError('Invalid data type for MATLAB v4 file')
        if len(shape) != 2: raise ValueError('Invalid shape for MATLAB v4 file (must be 2D)')
        if prod(data.shape) > 100000000: raise ValueError('MATLAB v4 files have a max of 100,000,000 elements')
        nvals = ((data.nnz+1)*(4 if is_complex else 3)) if _is_sparse(data) else prod(shape)
        if dt.itemsize * nvals > 2147483648: raise ValueError('MATLAB v4 files have a max of 2,147,483,648 bytes per variables')
        return cls.__HDR_SIZE + len(name) + 1 + dt.itemsize * nvals
    
    @property
    def data(self):
        f = self._mat._f
        f.seek(self._off)
        data = array_read(f, self._raw_shape, self._raw_dt, 'F')
        if self._dtype.kind == 'S': # text matrix
            data = data.astype(uint8).view(self._dtype)
        elif self._shape != self._raw_shape: # sparse matrix (handles complex as well)
            data = data[:-1,:] # last row has shape, we already have that
            I = ascontiguousarray(data[:,0], dtype=intc); I -= 1 # row indices
            J = ascontiguousarray(data[:,1], dtype=intc); J -= 1 # col indices
            V = ascontiguousarray(data[:,2:]).view(self._dtype).squeeze(1) # values
            data = scipy.sparse.coo_matrix((V,(I,J)), self._shape)
        elif self._dtype.kind == 'c': # complex matrices
            data = ascontiguousarray(data).view(self._dtype).squeeze(2)
        return data
    
class _MAT4File(_MAT45File):
    _entry = _MAT4Entry
    _version = 4
    @classmethod
    def open(cls, fn, mode):
        f = cls._basic_open(fn, mode)
        try: return cls._open(fn, f)
        except: f.close(); raise
    @classmethod
    def create(cls, fn, mode): return cls(fn, cls._basic_open(fn, mode), OrderedDict())

###### MATLAB v5 Files ######
class _MAT5DummyEntry(_MAT45DummyEntry):
    #pylint: disable=protected-access
    _is_subsys_data = False
    def _update_start(self, new_start):
        self._start = new_start
        if self._is_subsys_data: self._mat._set_ssdo(new_start)
    def _deleting(self):
        if self._is_subsys_data: self._mat._set_ssdo(None)

class _MAT5Entry(_MAT45Entry):
    #pylint: disable=protected-access
    __class2dtype = {
        1  : object_,  # cell array
        2  : void,     # struct (also accepts dictionaries)
        3  : void,     # struct + 'class_name'; confused if #3 actually means logical in some versions?
        4  : unicode_, # char array
        5  : 'sparse',
        6  : float64,
        7  : float32,
        8  : int8,
        9  : uint8,
        10 : int16,
        11 : uint16,
        12 : int32,
        13 : uint32,
        14 : int64,
        15 : uint64,
        #16 : 'function',
        #17 : 'opaque',
        #18 : 'object', # again?
    }
    __dtype2class = {
        object_   : 1,
        void      : 2, # struct vs object arrays are detected from metadata on the dtype
        string_   : 4,
        unicode_  : 4,
        # sparse is special
        float64   : 6,
        complex128: 6, # complex is done with a flag
        float32   : 7,
        complex64 : 7,
        int8      : 8,
        uint8     : 9,
        bool      : 9,
        bool_     : 9,
        int16     : 10,
        uint16    : 11,
        int32     : 12,
        uint32    : 13,
        int64     : 14,
        uint64    : 15,
    }

    @classmethod
    def open(cls, mat, f, is_compressed=False):
        name = None
        start = f.tell()

        try:
            # Read the basic header
            mdt, nbytes, skip = mat._read_tag_raw(f, first=not is_compressed)
            end = f.tell() + skip
            if mdt == 15:
                # Restart using a compressed stream
                if is_compressed: raise ValueError()
                with GzipFile(f, mode='rb', method='zlib') as gzf:
                    e = cls.open(mat, gzf, True)
                e.__start_comp, e._start = e._start, start #pylint: disable=attribute-defined-outside-init
                if f.seek(end) != end: raise ValueError()
                return e
            elif mdt != 14: raise ValueError() # at the top level we only accept matrix or compressed
                
            # Parse matrix headers
            flags_class, nzmax = mat._read_subelem(f, uint32, 2)
            clazz = flags_class & 0xFF
            dt = cls.__class2dtype.get(clazz, None)
            if dt is None:
                if clazz != 17: # opaque class does not have shape or name, other unknowns may or may not, tread lightly...
                    try:
                        mat._read_subelem(f, int32) # skip shape
                        name  = mat._read_subelem_string(f, True)
                    except (IOError, ValueError, StructError): pass
                raise NotImplementedError()
            is_logical = (flags_class & 0x0200) != 0
            is_global  = (flags_class & 0x0400) != 0
            is_complex = (flags_class & 0x0800) != 0
            is_sparse  = (flags_class & 0x1000) != 0 or _dt_eq_str(dt, 'sparse')
            shape = tuple(mat._read_subelem(f, int32))
            name  = mat._read_subelem_string(f, True)
            if name == '': raise NotImplementedError() # no-name matrix added if there are anonymous functions, its their workspace
            if (flags_class & 0xFFFFE100) != 0:
                warn("MATLAB file entry '%s' uses unknown flags (#%x) and the data may be wrong or the file may fail to read - send the file in"%(name,flags_class&0xFFFFE100))
            if not is_sparse and nzmax != 0:
                warn("MATLAB file entry '%s' has a nzmax value (%d) and is not sparse - send the file in"%(name,nzmax))
            
            # Handle multi-matrix datas
            if dt in (object_, void):
                # These are all very similar things, differences:
                #   cell is must basic, just a bunch of matrices
                #   struct adds a name to each of those matrices
                #   object adds a class name to the whole thing as well

                if is_logical or is_complex or is_sparse: raise ValueError("Invalid flags for cell/struct/object entry")

                raw_shape = prod(shape) # number of matrices to read
                raw_dt = dtype(dt)
                metadata = {'global':True} if is_global else {}
                if dt == void: # struct or object (have field names)
                    if clazz == 3: metadata['class_name'] = mat._read_subelem_string(f, True)
                    name_len = int(mat._read_subelem(f, int32, 1)) # MATLAB uses 32 (inc NULL terminator)
                    names = mat._read_subelem_string(f, True)
                    if len(names) % name_len != 0: raise ValueError("Malformed field names for struct/object")
                    nfields = len(names) // name_len
                    field_names = [names[i*name_len:(i+1)*name_len].split('\0', 1)[0] for i in xrange(nfields)]
                    raw_shape *= nfields
                    dt = dtype([(str(fn), object_) for fn in field_names])
                else:
                    dt = dtype(object_)
                off = f.tell()
                # We don't need to read all the matrix headers, however we could read them and grab
                # all the offsets of the matrix datas though... but let's do that during loading...
                if len(metadata) > 0: dt = dtype(dt, metadata=metadata)
                return cls(mat, start, off, name, raw_shape, shape, raw_dt, dt,
                           is_compressed, False, False, None)

            # Parse matrix data areas
            endian = mat._endian
            sub_dt = uint8 if (flags_class & 0x1200) == 0x1200 else None
            off = f.tell()
            if is_sparse:
                if len(shape) > 2: raise ValueError()
                if dt == 'sparse': dt = float64 # we have to assume that it is this way since we have no other information
                skip = mat._read_tag(f, int32, nzmax)[2]
                f.seek(skip, io.SEEK_CUR)
                _, nbytes, skip = mat._read_tag(f, int32, (0 if len(shape) < 2 else shape[1])+1)
                # Get the last value in that array which is the actual number of non-zero values
                f.seek(nbytes-4, io.SEEK_CUR)
                nnz, = unpack(endian+'i', f.read(4))
                f.seek(skip-nbytes, io.SEEK_CUR)
                raw_shape = (nnz,)
                size = nnz

            else:
                raw_shape = shape
                size = prod(shape)

            # Numeric, Character, and Sparse Arrays
            real_dt, _, skip = mat._read_tag(f, None, size, sub_dt=sub_dt)
            if real_dt in _MAT5File._utf2type: raw_dt = real_dt
            elif not isinstance(real_dt, type): raise NotImplementedError()
            else: raw_dt = _create_dtype(real_dt, endian)
            if not is_compressed:
                # For compressed data we do no further checks, they are just too expensive
                if is_complex:
                    # Check the imaginary data (but not if we are compressed - its just too expensive)
                    f.seek(skip, io.SEEK_CUR)
                    imag_dt, _, skip = mat._read_tag(f, None, size, sub_dt=sub_dt)
                    if not isinstance(imag_dt, type): raise NotImplementedError()
                    raw_dt = dtype((raw_dt, 2) if real_dt == imag_dt else \
                                   [('real', raw_dt), ('imag', _create_dtype(imag_dt, endian))])
                if f.seek(skip, io.SEEK_CUR) != end: raise ValueError()
            
        except NotImplementedError:
            if not is_compressed and f.seek(end) != end: raise ValueError()
            return _MAT5DummyEntry(mat, start, name)

        # Finish up
        dt = _create_dtype(bool if is_logical else dt, endian, 2 if is_complex else 1)
        if is_complex: dt = _complexify_dtype(dt)
        if is_global:  dt = dtype(dt, metadata={'global':True})

        # Create entry
        return cls(mat, start, off, name, raw_shape, shape, raw_dt, dt,
                   is_compressed, is_complex, is_sparse, sub_dt)
                
    @classmethod
    def create(cls, mat, f, name, data, is_compressed=False):
        start = f.tell()

        if not is_compressed and mat._version == 7:
            # Restart with a compressed file stream
            if (f.write(mat._long.pack(15)) != 4 or 
                f.write(mat._long.pack(0))  != 4): raise ValueError()
            with GzipFile(f, mode='wb', method='zlib') as gzf:
                e = cls.create(mat, gzf, name, data, True)
            e.__start_comp, e._start = e._start, start
            # Go back and write the size of the compressed data
            end = f.tell()
            if (f.seek(start+4) != start+4 or
                f.write(mat._long.pack(end-start-8)) != 4 or
                f.seek(end) != end): raise ValueError()
            return e

        if not isinstance(name, bytes): name = str(name).encode('ascii','ignore')
        data = _as_savable_array(data)

        matinfo, off, raw_shape, shape, raw_dt, dt, is_complex, is_sparse, sub_dt = \
                 cls.__create(mat, name, data)
        cls.__write_matrix(mat, f, matinfo)

        return cls(mat, start, start+off, name, raw_shape, shape, raw_dt, dt,
                   is_compressed, is_complex, is_sparse, sub_dt)

    @classmethod
    def __create(cls, mat, name, data):
        """
        Analyzes and converts the data and returns a matinfo (along with some other info possibly)
        with the data ready to be written to disk. The matinfo can be given to __write_matrix. The
        analysis and data conversion process is possibly recursive for cell arrays and structs, and
        since we don't want to write anything until we are all ready, we do this in two steps.
        """
        endian = mat._endian

        # Get the properties of the data
        raw_shape, raw_dt = data.shape, data.dtype
        shape, dt = raw_shape, _create_dtype(raw_dt, endian)
        dims, nzmax, sub_dt, force_dt = raw_shape, 0, None, None
        
        def get_data_size(nbytes):
            if nbytes <= 4: return 8
            pad = (8-(nbytes%8))
            if pad == 8: pad = 0
            return 8 + nbytes + pad
        nbytes = 16 + get_data_size(4*len(dims)) + get_data_size(len(name))
        off = nbytes + 8

        clazz = cls.__dtype2class[raw_dt.type]
        is_logical = raw_dt.kind == 'b'
        is_global  = raw_dt.metadata is not None and bool(raw_dt.metadata.get('global', False))
        is_complex = raw_dt.kind == 'c'
        is_sparse  = _is_sparse(data)
        flags = ((0x0200 if is_logical else 0) |
                 (0x0400 if is_global  else 0) |
                 (0x0800 if is_complex else 0) |
                 (0x1000 if is_sparse  else 0))

        def correct_ordering(a):
            # Make sure the data is the right byte and Fortran ordering
            if (_get_dtype_endian(a.dtype)=='<') != (endian=='<'): a.byteswap(True)
            return asfortranarray(a)

        def reduce_size(a):
            t = cls.__get_reduced_type(a)
            return a if t is None else a.astype(_create_dtype(t, endian))

        if raw_dt.kind in 'OV':
            if raw_dt.kind == 'O': # cell array
                out = [None] * data.size
                for i, d in enumerate(nditer(data, ('refs_ok',), ('readonly',), order='F')):
                    out[i] = cls.__create(mat, b'', d[()])
                nbytes += sum(mi[0] for mi in out) + 8*len(out)
            else: # struct/object
                fns = list(dt.names)
                fn_len = 64 if any(len(fn) >= 32 for fn in fns) else 32
                nbytes += 8 + get_data_size(len(fns) * fn_len)
                pre_data = [fns]
                if dt.metadata is not None and not is_invalid_matlab_name(dt.metadata.get('class_name','')):
                    class_name, clazz = dt.metadata.get['class_name'], 3
                    nbytes += get_data_size(len(class_name))
                    pre_data.insert(0, bytes(class_name))
                off = nbytes + 8
                out = [None] * (len(pre_data) + len(fns) * data.size)
                out[:len(pre_data)] = pre_data
                i = len(pre_data)
                for d in nditer(data, ('refs_ok',), ('readonly',), order='F'):
                    for fn in fns:
                        out[i] = cls.__create(mat, b'', d[fn][()])
                        i += 1
                nbytes += sum(mi[0] for mi in islice(out, len(pre_data), None)) + 8*len(out)

            if nbytes > 2147483648: raise ValueError('MATLAB v6/7 files have a max of 2,147,483,648 bytes per variables')
            matinfo = (nbytes, (flags|clazz,nzmax), dims, name, out, force_dt)
            return matinfo if name=='' else (matinfo, off, raw_shape, shape, raw_dt, dt, is_complex, is_sparse, sub_dt)
        
        if is_sparse:
            if raw_dt in (bool, float64, complex128): clazz = 5
            elif raw_dt not in 'biufc': raise ValueError("Non-numeric sparse arrays are not supported")
            if not scipy.sparse.isspmatrix_csc(data): data = data.tocsc()
            IR = data.indices.astype(int32, copy=False)
            JC = data.indptr.astype(int32, copy=False)
            assert(JC[-1] == data.nnz)
            data = data.data
            nzmax = len(IR)
            raw_shape = data.shape
            if is_sparse and is_logical:
                raw_dt = sub_dt = uint8
                force_dt = float64 # MATLAB actually marks it as 'float64' even though it is uint8
                data = data.astype(_create_dtype(uint8, _get_dtype_endian(data.dtype)), copy=False)
        
        if dt.kind == 'S':
            # convert ASCII to UTF-16
            data = codecs.encode(asfortranarray(data).data, 'utf_16')
            offset = len(codecs.BOM_UTF16) if data.startswith(codecs.BOM_UTF16) else 0
            real = correct_ordering(frombuffer(data, uint16, offset=offset))
            raw_dt = real.dtype
        elif dt.kind == 'U':
            # convert unicode to UTF-8 or UTF-16 (kinda)
            data = codecs.decode(asfortranarray(data).data, 'utf_32' + ('le' if _get_dtype_endian(data.dtype) == '<' else 'be'))
            if mat._version == 7: # save as utf8
                real = frombuffer(data.encode('utf_8'), uint8)
                raw_dt = force_dt = 'utf8'
            else: # save as utf16 (kinda)
                data = data.encode('utf_16')
                offset = len(codecs.BOM_UTF16) if data.startswith(codecs.BOM_UTF16) else 0
                real = correct_ordering(frombuffer(data, uint16, offset=offset))
                raw_dt = real.dtype
            
        elif is_complex:
            # Deal with complex data
            data = correct_ordering(_decomplexify(data))
            real = reduce_size(data[...,0])
            imag = reduce_size(data[...,1])
            raw_dt = dtype((real.dtype, 2)) if real.dtype == imag.dtype else \
                     dtype([('real', real.dtype), ('imag', imag.dtype)])

        else:
            # Unlike in v4 files we do not assume every 2 channel array is complex because v5 supports nd matrices
            real = reduce_size(correct_ordering(data))
            raw_dt = real.dtype

        # Create the "matinfo" tuple for use with __write_matrix
        data = (IR, JC) if is_sparse else ()
        data += (real, imag) if is_complex else (real,)
        nbytes += sum(get_data_size(a.nbytes) for a in data)
        if nbytes > 2147483648: raise ValueError('MATLAB v6/7 files have a max of 2,147,483,648 bytes per variables')
        matinfo = (nbytes, (flags|clazz,nzmax), dims, name, data, force_dt)

        # Return info
        return matinfo if name=='' else (matinfo, off, raw_shape, shape, raw_dt, dt, is_complex, is_sparse, sub_dt)

    @classmethod
    def __write_matrix(cls, mat, f, matinfo):
        """Writes the matrix info to disk."""
        nbytes, arr_flags, dims, name, data, force_dt = matinfo

        # Write basic header
        if (f.write(mat._long.pack(14))     != 4 or # Matrix tag
            f.write(mat._long.pack(nbytes)) != 4): raise ValueError()
        mat._write_subelem(f, array(arr_flags, uint32))
        mat._write_subelem(f, array(dims, int32))
        mat._write_subelem(f, frombuffer(name, int8))
        
        if isinstance(data, tuple): # numeric or char array, may be sparse or complex as well
            # len == 1: real, 2: complex, 3: sparse real, 4: sparse complex
            is_sparse = len(data) >= 3
            for i, a in enumerate(data):
                mat._write_subelem_big(f, a, (force_dt if i == (3 if is_sparse else 0) else None))
        else: # cell, struct, or object array
            # If first item is:
            #    list  - field names - struct array
            #    bytes - class name, second element is list of field names - object array
            # Rest are matinfo tuples for recursive calls
            start = 0
            if isinstance(data[0], list):
                start, fns = 1, data[0]
                if isinstance(fns, bytes):
                    mat._write_subelem(f, frombuffer(fns, int8))
                    start, fns = 2, data[1]
                fn_len = 64 if any(len(fn) >= 32 for fn in fns) else 32
                mat._write_subelem(f, array(fn_len, int32))
                mat._write_subelem(f, array(fns, dtype((string_, fn_len))).view(int8))
            for mi in islice(data, start, None):
                cls.__write_matrix(mat, f, mi)
            
    def __init__(self, mat, start, off, name, raw_shape, shape, raw_dt, dt,
                 is_compressed, is_complex, is_sparse, sub_dt):
        super(_MAT5Entry, self).__init__(mat, start, off, name, raw_shape, shape, raw_dt, dt)
        self.__start_comp = None
        self.__is_compressed = is_compressed
        self.__is_complex = is_complex
        self.__is_sparse  = is_sparse
        self.__sub_dt     = sub_dt

    @staticmethod
    def __get_reduced_type(a):
        """Checks to see if a type can be stored in a smaller size according to MATLAB rules"""
        if a.dtype.type != float64 or (a%1.0).any(): return None
        # We now have all integers (stored as float64s)
        # Check the min and max to see if we can fit it in one of the int data types
        mn, mx = long(a.min()), long(a.max())
        return next((t for t in (uint8, uint16, uint32) if mx <= iinfo(t).max), None) if mn >= 0 else \
               next((t for t in (int16, int32) if mn >= iinfo(t).min and mx <= iinfo(t).max), None)
    
    @classmethod
    def _calc_size(cls, mat, name, data):
        if mat._version == 7: return -1 # we cannot calculate the size of compressed entries
        return cls.__calc_size_internal(name, _as_savable_array(data))
        
    @classmethod
    def __calc_size_internal(cls, name, data):
        if not isinstance(name, bytes): name = str(name).encode('ascii','ignore')

        def get_reduced_size(a):
            t = cls.__get_reduced_type(a)
            itemsize = (a.dtype if t is None else dtype(t)).itemsize
            return a.size * itemsize
    
        def get_data_size(nbytes):
            if nbytes <= 4: return 8
            pad = (8-(nbytes%8))
            if pad == 8: pad = 0
            return 8 + nbytes + pad
        
        dims = data.shape
        is_sparse  = _is_sparse(data)
        is_complex = data.dtype.kind == 'c'
        nbytes = 24 + get_data_size(4*len(dims)) + get_data_size(len(name))
        
        if is_sparse:
            if not scipy.sparse.isspmatrix_csc(data): data = data.tocsc()
            nbytes += len(data.indices)*4
            nbytes += (len(data.indptr)+1)*4
        if data.dtype.kind == 'O': # cell array
            nbytes += sum(cls.__calc_size_internal(b'', data) for data in data.flat)
        elif data.dtype.kind == 'V': # struct and object arrays
            dt = data.dtype
            if dt.metadata is not None and not is_invalid_matlab_name(dt.metadata.get('class_name','')):
                nbytes += len(dt.metadata.get['class_name'])
            fns = dt.names
            fn_len = 64 if any(len(fn) >= 32 for fn in fns) else 32
            nbytes += 8 + get_data_size(len(fns)*fn_len) # field names
            nbytes += sum(sum(cls.__calc_size_internal(b'', data[fn]) for fn in fns) for data in data.flat)
        elif data.dtype.kind in 'SU':
            nbytes += 2*data.size # in v6 always saved as uint16s
        elif is_complex:
            data = _decomplexify(data)
            nbytes += get_reduced_size(data[...,0])
            nbytes += get_reduced_size(data[...,1])
        else:
            nbytes += get_reduced_size(data)
        if nbytes-8 > 2147483648: raise ValueError('MATLAB v6/7 files have a max of 2,147,483,648 bytes per variables')
        return nbytes

    _is_subsys_data = False
    def _update_start(self, new_start):
        delta = new_start - self._start
        self._start = new_start
        self._off += delta
        if self.__is_compressed: self.__start_comp += delta
        if self._is_subsys_data: self._mat._set_ssdo(new_start)
    def _deleting(self):
        if self._is_subsys_data: self._mat._set_ssdo(None)

    @property
    def header(self):
        h = super(_MAT5Entry, self).header
        metadata = self._dtype.metadata or {}
        if metadata.get('global', False) or self.__is_compressed:
            flags = []
            if metadata.get('global', False): flags.append('global')
            if self.__is_compressed:          flags.append('compressed')
            h['flags'] = ','.join(flags)
        if 'class_name' in metadata:
            h['class_name'] = metadata['class_name']
        dt = self._dtype
        if dt.kind == 'c':   dt = _decomplexify_dtype(dt)
        elif dt.kind == 'b': dt = dtype(uint8)
        if self._raw_dt in _MAT5File._utf2type:
            h['storage-type'] = self._mat._endian + self._raw_dt
        elif dt.base != self._raw_dt.base:
            h['storage-type'] = self._raw_dt.base.str if self._raw_dt.names is None else \
                                ','.join(self._raw_dt[i].str for i in xrange(len(self._raw_dt)))
        return h

    @property
    def data(self):
        mat = self._mat.__repr__.__self__ # unproxy the object
        f = mat._f
        # Get compressed file stream
        if self.__is_compressed:
            f.seek(self.__start_comp)
            f = GzipFile(f, mode='rb', method='zlib')
        # Seek to the data
        f.seek(self._off)

        # Handle multi-matrices
        if isinstance(self._raw_dt, dtype) and self._raw_dt.kind in 'OV':
            return _MAT5Entry.__read_multimatrix(mat, f, self._shape, self._dtype)

        # Get single matrix
        return _MAT5Entry.__read_matrix(mat, f, self.__is_complex, self.__is_sparse,
                                        self._raw_shape, self._shape, self._dtype, self.__sub_dt)

    @classmethod
    def __read_matrix(cls, mat, f, is_complex, is_sparse, raw_shape, shape, dt, sub_dt):
        """Reads a character of numeric array"""
        metadata = dt.metadata
        # Handle sparse data
        if is_sparse:
            nnz, N = raw_shape[0], 0 if len(shape) < 2 else shape[1]
            IR = mat._read_subelem_big(f, int32, (nnz,), True) # row indices
            JC = mat._read_subelem_big(f, int32, (N+1,))       # col indices
            if nnz != JC[-1]:
                # raw_shape was nzmax, not nnz, correct things here
                nnz = JC[-1]
                raw_shape, IR = IR[:nnz], (nnz,)
        # Load data (either complex or regular)
        data = mat._read_subelem_big(f, None, raw_shape, chararr=dt.kind in 'SU', sub_dt=sub_dt)
        if is_complex:
            real = data
            imag = mat._read_subelem_big(f, None, raw_shape, sub_dt=sub_dt)
            data = empty(raw_shape, _decomplexify_dtype(dt))
            data[...,0] = real
            data[...,1] = imag
            del real, imag
            if dt.kind == 'c': data = data.view(dt).squeeze(-1)
            elif metadata is not None: data = data.view(dtype(data.dtype, metadata=metadata))
        else:
            data = data.astype(dt, copy=False)
        # Return the array
        if is_sparse:
            data = scipy.sparse.csc_matrix((data[:nnz],IR,JC), shape=shape, dtype=dt)
        return data

    @classmethod
    def __read_multimatrix(cls, mat, f, shape, dt):
        """Reads a cell array or struct/object"""
        val = ((lambda:cls.__read_matrix_embedded(mat, f)) if dt.names is None else               # cell array
               (lambda:tuple(cls.__read_matrix_embedded(mat, f) for _ in xrange(len(dt.names))))) # struct array
        out = empty(shape, dt, order='F')
        out_flat = out.ravel('F')
        for i in xrange(prod(shape)): out_flat[i] = val()
        return out

    @classmethod
    def __read_matrix_embedded(cls, mat, f):
        """Reads a matrix embedded in a multi-matrix"""
        try:
           # Read the basic header
            dt, _, skip = mat._read_tag(f)
            end = f.tell() + skip
            if dt != 'matrix': raise ValueError()
                
            # Parse matrix headers
            flags_class, nzmax = mat._read_subelem(f, uint32, 2)
            clazz = flags_class & 0xFF
            dt = cls.__class2dtype.get(clazz, None)
            if dt is None: raise NotImplementedError()
            is_logical = (flags_class & 0x0200) != 0
            is_global  = (flags_class & 0x0400) != 0
            is_complex = (flags_class & 0x0800) != 0
            is_sparse  = (flags_class & 0x1000) != 0 or _dt_eq_str(dt, 'sparse')
            shape = tuple(mat._read_subelem(f, int32))
            mat._read_subelem_string(f, True) # Name is not needed, and is usually 'random'
            if (flags_class & 0xFFFFE100) != 0:
                warn("MATLAB embedded matrix uses unknown flags (#%x) and the data may be wrong or the file may fail to read - send the file in"%(flags_class&0xFFFFE100))
            if not is_sparse and nzmax != 0:
                warn("MATLAB embedded matrix has a nzmax value (%d) and is not sparse - send the file in"%nzmax)
            
            # Handle multi-matrix datas
            if dt in (object_, void):
                if is_logical or is_complex or is_sparse: raise ValueError("Invalid flags for cell/struct/object entry")
                raw_shape = prod(shape) # number of matrices to read
                metadata = {'global':True} if is_global else {}
                if dt == void: # struct or object
                    if clazz == 3: metadata['class_name'] = mat._read_subelem_string(f, True)
                    name_len = int(mat._read_subelem(f, int32, 1)) # MATLAB uses 32 (inc NULL terminator)
                    names = mat._read_subelem_string(f, True)
                    if len(names) % name_len != 0: raise ValueError("Malformed field names for struct/object")
                    nfields = len(names) // name_len
                    field_names = [names[i*name_len:(i+1)*name_len].split('\0', 1)[0] for i in xrange(nfields)]
                    raw_shape *= nfields
                    dt = dtype([(str(fn), object_) for fn in field_names])
                else:
                    dt = dtype(object_)
                if len(metadata) > 0: dt = dtype(dt, metadata=metadata)
                matrix = cls.__read_multimatrix(mat, f, shape, dt)

            else:
                # Parse matrix data areas
                sub_dt = uint8 if (flags_class & 0x1200) == 0x1200 else None
                raw_shape = shape
                if is_sparse:
                    if len(shape) > 2: raise ValueError()
                    if dt == 'sparse': dt = float64 # we have to assume that it is this way since we have no other information
                    raw_shape = (nzmax,)
                    
                # Return matrix
                dt = _create_dtype(bool if is_logical else dt, mat._endian, 2 if is_complex else 1)
                if is_complex: dt = _complexify_dtype(dt)
                if is_global: dt = dtype(dt, metadata={'global':True})
                matrix = cls.__read_matrix(mat, f, is_complex, is_sparse, raw_shape, shape, dt, sub_dt)
            
        except NotImplementedError:
            matrix = None
            
        f.seek(end)
        return matrix

class _MAT5File(_MAT45File):
    #pylint: disable=protected-access
    _entry = _MAT5Entry
    _version = 0 # this is updated to 6 or 7 once an entry has been read (compressed or uses utf8/16/32 then it is 7)
    __mat2dtype = {
        1:  int8,
        2:  uint8,
        3:  int16,
        4:  uint16,
        5:  int32,
        6:  uint32,
        7:  float32,
        9:  float64,
        12: int64,
        13: uint64,
        14: 'matrix',
        15: 'compressed',
        # 8,10,11: reserved
        16: 'utf8',
        17: 'utf16',
        18: 'utf32',
    }
    __dtype2mat = {
        int8    : 1,
        uint8   : 2,
        int16   : 3,
        uint16  : 4,
        int32   : 5,
        uint32  : 6,
        float32 : 7,
        float64 : 9,
        int64   : 12,
        uint64  : 13,
        'utf8'  : 16,
        'utf16' : 17,
        'utf32' : 18,
    }
    _utf2type = { # data types to use to read the UTF encoded data, as a first pass
        'utf8'  : uint8,
        'utf16' : uint16,
        'utf32' : uint32,
    }
    
    @classmethod
    def open(cls, fn, mode):
        f = cls._basic_open(fn, mode)
        try:
            header = f.read(128)
            if len(header) != 128 or any(x == 0 for x in header[:4]): raise ValueError() # Always supposed to have non-null bytes in first 4 bytes
            version, endian = (header[125],header[124]), header[126:128]
            if endian not in (b'IM',b'MI'): raise ValueError()
            endian = '<' if endian == b'IM' else '>'
            if endian == '>': version = version[::-1]
            if version[0] != 1: raise ValueError()
            text = header[:116].rstrip(b'\0').rstrip() # or [:124]?
            ssdo = header[116:124]
            ssdo = None if ssdo in (b'        ',b'\0\0\0\0\0\0\0\0') else unpack(endian+'Q', ssdo)[0]
            mat = cls._open(fn, f, endian, text, ssdo, version)
        except: f.close(); raise
        if ssdo is not None:
            entries = mat._entries.iteritems()
            for _,e in entries:
                if e._start == ssdo:
                    e._is_subsys_data = True
                    break
        return mat

    @classmethod
    def create(cls, fn, mode, compressed=False):
        f = cls._basic_open(fn, mode)
        try:
            text = _get_matlab_header(b'MATLAB 5.0 MAT-file, Platform: %s, Created on: %s')
            mat = cls(fn, f, OrderedDict(), sys_endian, text, None, (1,0))
        except: f.close(); raise
        mat._version = 7 if compressed else 6
        header = bytearray(128)
        header[:len(text)] = text
        header[len(text):116] = b' '*(116-len(text))
        header[124:128] = b'\x00\x01IM' if sys_endian == '<' else b'\x01\x00MI'
        if f.write(header) != 128: raise IOError()
        return mat
    
    def __init__(self, filename, f, entries, endian, header_text, ssdo, version):
        self._endian = endian
        self._long = Struct(endian+'L')
        self.__header_text = header_text
        self.__subsys_data_off = ssdo
        self.__version = version
        super(_MAT5File, self).__init__(filename, f, entries)
        
    @property
    def header(self):
        h = super(_MAT5File, self).header
        h.update({'mat-version':'.'.join(str(x) for x in self.__version),'text':self.__header_text})
        if self.__subsys_data_off is not None: h['subsystem-data-off'] = self.__subsys_data_off
        return h

    def _set_ssdo(self, ssdo):
        if ssdo == 0 or ssdo is None:
            self.__subsys_data_off = None
            ssdo = 0
        else:
            self.__subsys_data_off = ssdo
        if self._f.seek(116) != 116 or self._f.write(pack(self._endian+'Q', ssdo)) != 8: raise IOError()

    def _read_tag_raw(self, f, first=False):
        """
        Reads a tag header and returns the raw data.

        Return the MATLAB data type identifier (a number), the number of bytes of data, and the
        number bytes including padding (enough to skip to the next tag). The file itself is
        positioned immediately after the tag so data is ready to be read or skipped. No checking
        of the data type is done. In nearly all cases, _read_tag should be used instead of this
        function as it converts the data type into something useful and adds checks for it.

        f is usually self._f, however it may also be a GzipFile-wrapped version of self._f as well.

        If first is True, EOFError instead of ValueError is raised if the first read reads no data.
        """
        mdt = f.read(4)
        if first and len(mdt) == 0: raise EOFError()
        mdt = self._long.unpack(mdt)[0]
        nbytes = mdt>>16 # for SDE
        if nbytes: # Small Data Element Format
            if nbytes > 4: raise ValueError()
            mdt = (mdt&0xFFFF)
            skip = 4
        else:
            nbytes = self._long.unpack(f.read(4))[0]
            skip = 8-(nbytes%8)
            if mdt == 15 or skip == 8: skip = 0 # non-compressed data must align to an 8-byte boundary
            skip += nbytes
        should_be_v7 = 15 <= mdt <= 18
        if self._version == 0:
            if should_be_v7: self._version = 7
            elif mdt == 14:  self._version = 6
        elif should_be_v7 and self._version != 7: raise ValueError('File version mismatch')
        return mdt, nbytes, skip
    
    def _read_tag(self, f, expected_type=None, expected_nvals=None, sub_dt=None):
        """
        Reads a tag header.

        Return the type of the data (None, a special string, or a numeric type), the number of bytes
        of data, and the number bytes including padding (enough to skip to the next tag). The
        file itself is positioned immediately after the tag so data is ready to be read or skipped.

        f is usually self._f, however it may also be a GzipFile-wrapped version of self._f as well.

        If expected_type or expected_nvals is not None this also checks the type and/or number of
        values. For special types (like compressed and matrix), do not provide expected_nvals.

        If sub_dt is not None, it is used instead of whatever the actual dtype is found to be.
        """
        mdt, nbytes, skip = self._read_tag_raw(f)
        if sub_dt is not None:
            dt = sub_dt
        else:
            dt = _MAT5File.__mat2dtype.get(mdt, None)
            if dt is None: raise NotImplementedError()
        if ((expected_type is not None and expected_type != dt) or
            (expected_nvals is not None and expected_nvals != nbytes//dtype(_MAT5File._utf2type.get(dt,dt)).itemsize)):
            raise ValueError()
        return dt, nbytes, skip
    
    def _read_subelem(self, f, expected_type=None, expected_nvals=None):
        """
        Read an entire subelement. Uses _read_tag then reads the data and uses frombuffer. It's type
        cannot be a special type (like compressed or matrix) but string types are supported and
        always return as unicode_ arrays (with 4-bytes per character). While most return arrays will
        have a data type that is the same as the file data type, unicode arrays will not.

        Returns the data as a 1D ndarray. The file is positioned at the end of the subelement.
        """
        dt, nbytes, skip = self._read_tag(f, expected_type, expected_nvals)
        data = f.read(skip)
        if len(data) != skip: raise ValueError()
        if dt in _MAT5File._utf2type:
            if skip != nbytes: data = memoryview(data)[:nbytes]
            if   dt == 'utf8':  data = codecs.decode(data, 'utf_8')
            elif dt == 'utf16': data = codecs.decode(data, 'utf_16' + ('le' if self._endian == '<' else 'be'))
            elif dt == 'utf32': data = codecs.decode(data, 'utf_32' + ('le' if self._endian == '<' else 'be'))
            data = unicode_(data).reshape(1).view(dtype((unicode_, 1)))
            return data.byteswap(True) if self._endian != sys_endian else data
        elif not isinstance(dt, type): raise ValueError()
        dt = dtype(dt).newbyteorder(self._endian)
        return frombuffer(data, dt, nbytes // dt.itemsize)

    def _read_subelem_string(self, f, var_name=False):
        """
        Read an entire subelement that is interpreted as a string. The underlying data type must be
        int8, uint8, uint16, utf8, utf16, or utf32. A unicode string is returned unless var_name is
        True, which causes an ASCII string to be returned.
        """
        # When reading array, field, and class names, they are supposed to be int8 but some
        # poorly formed files use "utf8", but we need to check that they are 7 bit safe...
        dt, nbytes, skip = self._read_tag(f)
        data = f.read(skip)
        if len(data) != skip: raise ValueError()
        if var_name:
            if dt == int8:
                # quick and easy case
                return data[:nbytes] if skip != nbytes else data
            if dt == 'utf8':
                # only other common-ish case  for variable names
                data = data[:nbytes] if skip != nbytes else data
                if any(x > 127 for x in data): raise ValueError("Invalid variable name stored in header")
                return data
        if skip != nbytes: data = memoryview(data)[:nbytes]
        e = ('le' if self._endian == '<' else 'be')
        if dt in (int8, uint8): s = codecs.decode(data, 'latin1')
        elif dt == uint16: s = codecs.decode(data, 'utf_16' + e)
        elif dt in _MAT5File._utf2type:
            dt = 'utf_' + dt[3:]
            s = codecs.decode(data, dt + (e if _MAT5File._utf2type[dt].itemsize > 1 else ''))
        else: raise ValueError("Expected string in header, found %s" % dt)
        return s.encode('ascii') if var_name else s

    def _read_subelem_big(self, f, dt, shape, relax=False, chararr=False, sub_dt=None):
        """
        Like _read_subelem except designed for large arrays. Instead of read/frombuffer this uses
        array_read (which uses either fromfile or readinto). The shape must always be known. If
        relax is True than the number of elements in the subelem must only be at least as many as
        the shape requests. Returns an ndarray of the given shape. If you except to get a string
        type back, pass chararr=True. This does different data checks since the data for a string
        may be saved encoded (and thus a different size than expected) and it also decodes the data
        as necessary.
        """
        dt, nbytes, skip = self._read_tag(f, dt, None if relax or chararr else prod(shape), sub_dt=sub_dt)
        if chararr:
            dtx = _create_dtype(unicode_, self._endian, 1)
            if nbytes == 0:
                # Sometimes strings will have 0 bytes even if they have a shape... create them space filled
                f.seek(skip, io.SEEK_CUR)
                data = array(u' ', dtype=dtx, order='F')
                if self._endian != sys_endian: data.byteswap(True)
                return tile(data, shape)
            # String must be read in completely, decoded, then finally made into an array
            data, e = f.read(nbytes), 'le' if self._endian == '<' else 'be'
            skip -= nbytes
            if len(data) != nbytes: raise ValueError()
            if dt in (uint16, 'utf16'): data = codecs.decode(data, 'utf_16' + e)
            elif dt == 'utf32':         data = codecs.decode(data, 'utf_32' + e)
            elif dt == 'utf8':          data = codecs.decode(data, 'utf_8')
            elif dt in (int8, uint8):   data = codecs.decode(data, 'latin1')
            else: raise ValueError("Cannot decode character data from dtype %s" % dt)
            data = unicode_(data).reshape(1).view(dtype((unicode_, 1))).reshape(shape, order='F')
            if self._endian != sys_endian: data.byteswap(True)
        elif not isinstance(dt, type) or relax and prod(shape) >= nbytes: raise ValueError()
        else:
            data = array_read(f, shape, dtype(dt).newbyteorder(self._endian), 'F')
            skip -= data.nbytes
        f.seek(skip, io.SEEK_CUR)
        return data
    
    def _write_tag(self, f, dt, nbytes):
        """
        Writes a tag of the given type and with the given number of bytes to the file f.

        f is usually self._f, however it may also be a GzipFile-wrapped version of self._f as well.
        """
        mdt = _MAT5File.__dtype2mat[dt]
        if nbytes <= 4: # Small Data Element Format
            if f.write(self._long.pack(mdt|(nbytes<<16))) != 4: raise ValueError()
        elif (f.write(self._long.pack(mdt))    != 4 or 
              f.write(self._long.pack(nbytes)) != 4): raise ValueError()

    def _write_subelem(self, f, data, force_dt=None):
        """
        Writes an entire subelement to the given file using data from the ndarray. The ndarray
        specifies the type and number of bytes to list in the tag. This will also write the
        necessary padding after the subelement. The subelement must be an ndarray. Strings are not
        directly supported but does support being passed a UTF dt in force_dt so that the tag
        header has the right value.
        """
        dt, nbytes = data.dtype.type, data.nbytes
        self._write_tag(f, force_dt or dt, nbytes)
        if f.write(data.data) != nbytes: raise ValueError()
        pad = (4-nbytes) if nbytes <= 4 else (8-(nbytes%8))
        if pad not in (0,8) and f.write(b'\0'*pad) != pad: raise ValueError()

    def _write_subelem_big(self, f, data, force_dt=None):
        """
        Like _write_subelem but uses array_save instead of f.write. This is meant for large amounts
        of data and can be faster.
        """
        dt, nbytes = data.dtype.type, data.nbytes
        self._write_tag(f, force_dt or dt, nbytes)
        array_save(f, data)
        pad = (4-nbytes) if nbytes <= 4 else (8-(nbytes%8))
        if pad not in (0,8) and f.write(b'\0'*pad) != pad: raise ValueError()


########### MATLAB v7.3 Files ###########
class _MAT73File(_MATFile):
    """MAT v7.3 files do not support "ordering", so some methods won't work as expected."""
    #pylint: disable=protected-access
    _version = 7.3

    @classmethod
    def open(cls, fn, mode):
        f = h5py.File(fn, mode)
        try:
            if f.userblock_size < 128: raise ValueError()
            with io.open(fn, 'rb') as fh: header = fh.read(128)
            if len(header) != 128 or any(x == 0 for x in header[:4]): raise ValueError() # Always supposed to have non-null bytes in first 4 bytes
            version, endian = (header[125],header[124]), header[126:128]
            if endian not in (b'IM',b'MI'): raise ValueError()
            endian = '<' if endian == b'IM' else '>'
            if endian == '>': version = version[::-1]
            if version[0] != 2: raise ValueError()
            entries = OrderedDict((name,_MAT73Entry.open(entry)) for name,entry in f.iteritems())
        except: f.close(); raise
        return cls(fn, f, entries, header[:116].rstrip(b'\0').rstrip(), version)

    @classmethod
    def create(cls, fn, mode):
        header = bytearray(128)
        text = _get_matlab_header(b'MATLAB 7.3 MAT-file, Platform: %s, Created on: %s HDF5 schema 1.00 .')
        header[:len(text)] = text
        header[len(text):116] = b' '*(116-len(text))
        header[124:128] = b'\x00\x02IM' if sys_endian == '<' else b'\x02\x00MI'
        f = h5py.File(fn, 'w' if mode == 'w+' else mode, libver='earliest', userblock_size=512L)
        try:
            with io.open(fn, 'wb') as fh:
                if fh.write(header) != 128: raise IOError()
        except: f.close(); raise
        return cls(fn, f, OrderedDict(), text, (2, 0))

    def __init__(self, fn, f, entries, header_text, version):
        super(_MAT73File, self).__init__(fn, f, entries)
        self.__header_text = header_text
        self.__version = version

    def rename(self, renamer, filename):
        mode = self._f.mode
        self._f.close()
        renamer(filename)
        self._f = h5py.File(filename, mode)
        self._filename = filename
        for entry,h5ent in izip(self._entries.itervalues(), self._f.itervalues()):
            assert entry._name == h5ent.name.lstrip('/')
            entry._entry = h5ent

    @property
    def header(self):
        h = super(_MAT73File, self).header
        h.update({'mat-version':'.'.join(str(x) for x in self.__version),'text':self.__header_text})
        return h

    def append(self, name, data):
        if name in self._entries or is_invalid_matlab_name(name): raise KeyError()
        self._entries[name] = entry = _MAT73Entry.create(self._f, name, data)
        return entry

    def insert_before(self, name, new_name, data):
        # This just results in an append since order is not maintained
        return self.append(new_name, data)

    def set(self, name, data):
        # The data is appended unless it can be placed into the old slot (means dtype and type need to be the same and shape must be compatible)
        if name not in self._entries: return self.append(name, data)
        return self.__simple_set(name, data)
        # TODO: Try to reuse old entry storage
        # This is quite complicated, and may not be worth it...
##        entry = self._entries[name]
##        if not entry.is_image: return self.__simple_set(name, data) # TODO
##        old_shape, old_dt = entry.shape + entry.dtype.shape, entry.dtype.base
##        new_shape, new_dt = data.shape + data.dtype.shape, data.dtype.base
##        if new_dt.type != old_dt.type or len(old_shape) != len(new_shape):
##            # Technically for things that were/are empty some things may be possible with this, but we aren't saving much and that is rare
##            return self.__simple_set(name, data)
##        was_empty, is_empty = 0 in old_shape, 0 in new_shape
##        was_sparse, is_sparse = entry._is_sparse, _is_sparse(data)
##        h5ent = entry._entry
##        if was_sparse != is_sparse: return self.__simple_set(name, data)
##        elif was_sparse:
##            if data.nnz == 0 and 'data' in h5ent: del h5ent['ir'], h5ent['data']
##            if is_empty:
##                h5ent.attrs['MATLAB_sparse'] = uint64(0)
##                _MAT73File.__force_data(h5ent, 'jc', array(0, dtype=uint64))
##            else:
##                if not scipy.sparse.isspmatrix_csc(data): data = data.tocsc()
##                h5ent.attrs['MATLAB_sparse'] = uint64(new_shape[0])
##                _MAT73File.__force_data(h5ent, 'jc', data.indptr.astype(uint64, copy=False))
##                if data.nnz != 0:
##                    _MAT73File.__force_data(h5ent, 'ir', data.indices.astype(uint64, copy=False))
##                    data = data.data
##                    if _get_dtype_endian(new_dt) != _get_dtype_endian(old_dt): data = data.byteswap()
##                    _MAT73File.__force_data(h5ent, 'data', data)
##        elif was_empty != is_empty: return self.__simple_set(name, data)
##        elif was_empty: h5ent.write_direct(array(new_shape, dtype=uint64))
##        else:
##            if old_shape != new_shape:
##                new_sh = new_shape[::-1]
##                if h5ent.chunks is None or any(n>m for n,m in izip(new_sh,h5ent.maxshape) if m is not None):
##                    return self.__simple_set(name, data)
##                h5ent.resize(new_sh)
##            if _get_dtype_endian(new_dt) != _get_dtype_endian(old_dt): data = data.byteswap()
##            h5ent.write_direct(ascontiguousarray(data.T))
##        entry._shape = new_shape
##        entry._dtype = new_dt
##        return entry
##
##    @staticmethod
##    def __force_data(entry, name, data):
##        # If the dataset does not exist, it is created
##        # Otherwise if it is the right size, the data is replaced
##        # Otherwise if the dataset can be resized, it is resized and the data is replaced
##        # Otherwise the dataset is deleted and re-created
##        dataset = entry.get(name)
##        comp = 'gzip' if len(data) > 1250 else None
##        if dataset is None:
##            entry.create_dataset(name, data=data, compression=comp)
##        elif dataset.shape == data.shape:
##            dataset.write_direct(data)
##        elif dataset.chunks is not None:
##            dataset.resize(data.shape)
##            dataset.write_direct(data)
##        else:
##            del entry[name]
##            entry.create_dataset(name, data=data, compression=comp)

    def __simple_set(self, name, data):
        self.__del_entry(name)
        return self.append(name, data)

    def __delitem__(self, name):
        if isinstance(name, str):
            if name not in self._entries: raise KeyError()
            self.__del_entry(name)
        else:
            if any(n not in self._entries for n in name): raise KeyError()
            for n in name: self.__del_entry(n)

    def __del_entry(self, name):
        del self._entries[name]
        self.__del_refs(self._f[name])

    _can_have_multi_ref = False # Undocummented feature, end-user can set this to True to be more careful when deleting refs
    def __del_refs(self, entry):
        if isinstance(entry, h5py.Group):
            for e in entry.itervalues(): self.__del_refs(e)
            del self._f[entry.name]
        elif isinstance(entry, h5py.Dataset) and h5py.check_dtype(ref=entry.dtype) == h5py.Reference:
            refs = entry[:].flat
            del self._f[entry.name]
            for ref in refs:
                e = self._f[ref]
                if not self._can_have_multi_ref or not _MAT73File.__is_referenced(self._f, e):
                    del self._f[e.name]
        else:
            del self._f[entry.name]

    @staticmethod
    def __is_referenced(grp, entry):
        for e in grp.itervalues():
            if isinstance(e, h5py.Group):
                if _MAT73File.__is_referenced(e, entry): return True
            elif isinstance(e, h5py.Dataset) and h5py.check_dtype(ref=e.dtype) == h5py.Reference:
                if any(grp.file[r] == entry for r in e[:].flat): return True
        return False

class _MAT73Entry(_MATEntry):
    __class2dtype = {
        b'int8' :int8, b'int16' :int16, b'int32' :int32, b'int64' :int64,
        b'uint8':uint8,b'uint16':uint16,b'uint32':uint32,b'uint64':uint64,
        b'single':float32,b'double':float64,
        b'logical':bool,b'char':unicode_,b'cell':object_,b'struct':void,
    }
    __dtype2class = {
        int8 :b'int8', int16 :b'int16', int32 :b'int32' ,int64 :b'int64',
        uint8:b'uint8',uint16:b'uint16',uint32:b'uint32',uint64:b'uint64',
        float32:b'single',float64:b'double',complex64:b'single',complex128:b'double',
        bool:b'logical',string_:b'char',unicode_:b'char',object_:b'cell',void:b'struct',
    }

    # MATLAB_int_decode:    0 -> NO_INT_HINT, 1 -> LOGICAL_HINT, 2 -> UTF16_HINT
    # MATLAB_object_decode: 0 -> NO_OBJ_HINT, 1 -> FUNCTION_HINT, 2 -> OBJECT_HINT, 3 -> OPAQUE_HINT

    @classmethod
    def open(cls, entry):
        name = entry.name.lstrip('/')
        if len(name) > 1 and name[0] == b'#' and name[-1] == b'#':
            return _MAT73DummyEntry(name, entry)
        try:
            shape, dt, is_sparse = cls.__get_entry_info(entry)
        except ValueError:
            return _MAT73DummyEntry(name, entry)
        return _MAT73Entry(name, shape, dt, entry, is_sparse)

    @classmethod
    def create(cls, f, name, data):
        data = _as_savable_array(data)
        entry, dt = cls.__create_entry(f, name, data)
        return _MAT73Entry(name, data.shape, dt, entry, _is_sparse(data))

    @classmethod
    def __create_entry(cls, grp, name, data):
        # Note: pylint thinks a call to a vectorized function returns a tuple...
        #pylint: disable=no-member
        shape, dt = data.shape, data.dtype.base
        is_sparse = _is_sparse(data)
        is_empty = data.size == 0

        if is_sparse and dt.kind in 'OVSU': raise ValueError("Unable to save non-numeric data as sparse arrays")
        if dt.kind == 'O': # cell array
            save_dt = h5py.special_dtype(ref=h5py.Reference)
            refs = grp.file.require_group('#refs#')
            data = vectorize(lambda elem: cls.__create_entry(refs, None, elem)[0].ref, otypes=(save_dt,))(data)
        elif dt.kind == 'V': # struct array
            pass # this is all handled later
        elif dt.kind in 'SU': # char array
            data = asfortranarray(data).data
            data = codecs.encode(data, 'utf_16') if dt.kind == 'S' else \
                   codecs.decode(data, 'utf_32' + ('le' if _get_dtype_endian(dt) == '<' else 'be')).encode('utf_16')
            offset = len(codecs.BOM_UTF16) if data.startswith(codecs.BOM_UTF16) else 0
            data = frombuffer(data, uint16, offset=offset).reshape(shape, order='F')
            if dt.kind == 'S':
                dt = (dtype((unicode_, 1)) if dt.metadata is None else
                      dtype((unicode_, 1), metadata=dt.metadata)).newbyteorder(dt.byteorder)
            save_dt = dtype(uint16)
        elif dt.kind == 'c': # complex
            save_dt = _decomplexify_dtype(dt).base
            save_dt = dtype([('real', save_dt), ('imag', save_dt)])
        else: # numeric/logical
            save_dt = dtype(uint8) if dt.kind == 'b' else dt

        if name is None:
            # Get the next available name
            cur_names = frozenset(grp.keys())
            name = next(n for n in _MAT73Entry.__name_gen() if n not in cur_names)
        if is_sparse:
            entry = grp.create_group(name)
            if is_empty:
                entry.attrs['MATLAB_sparse'] = uint64(0)
                entry.create_dataset('jc', data=uint64(0))
            else:
                if not scipy.sparse.isspmatrix_csc(data): data = data.tocsc()
                entry.attrs['MATLAB_sparse'] = uint64(shape[0])
                comp = 'gzip' if shape[1] >= 1250 else None
                entry.create_dataset('jc', None, uint64, data.indptr, compression=comp)
                if data.nnz != 0:
                    comp = 'gzip' if data.nnz > 1250 else None
                    entry.create_dataset('ir', None, uint64, data.indices, compression=comp)
                    entry.create_dataset('data', None, save_dt, data.data, compression=comp)
        elif is_empty:
            entry = grp.create_dataset(name, None, uint64, shape)
            entry.attrs['MATLAB_empty'] = int32(1)
        elif dt.kind == 'V':
            entry = grp.create_group(name)
            fns = [bytes(fn) for fn in dt.names]
            if len(fns) > 1:
                fn_dt = dtype('S1')
                field_names = empty(len(fns), h5py.special_dtype(vlen=fn_dt))
                for i, fn in enumerate(fns):
                    field_names[i] = string_(fn).reshape(1).view(fn_dt)
                entry.attrs['MATLAB_fields'] = field_names
            if prod(shape) == 1: # scalar struct, save the fields directly here
                for fn in fns: cls.__create_entry(entry, fn, data[0,0][fn])
            else: # saves a cell-array like entry for each field - but without any MATLAB attrs
                save_dt = h5py.special_dtype(ref=h5py.Reference)
                refs = grp.file.require_group('#refs#')
                comp = 'gzip' if data.size > 1250 else None
                shape_T = shape[::-1]
                vec = vectorize(lambda elem: cls.__create_entry(refs, None, elem)[0].ref, otypes=(save_dt,))
                for fn in fns:
                    entry.create_dataset(fn, shape_T, save_dt, vec(data[fn]).T, compression=comp)
        else:
            comp = 'gzip' if data.size > 1250 else None
            entry = grp.create_dataset(name, shape[::-1], save_dt, data.T, compression=comp)
        entry.attrs['MATLAB_class'] = _MAT73Entry.__dtype2class[dt.type]

        if   dt.kind == 'b': entry.attrs['MATLAB_int_decode'] = int32(1) # LOGICAL_HINT
        elif dt.kind == 'U': entry.attrs['MATLAB_int_decode'] = int32(2) # UTF16_HINT
        if dt.metadata is not None:
            if dt.metadata.get('global', False): entry.attrs['MATLAB_global'] = int32(1)
            if dt.kind == 'V' and not is_invalid_matlab_name(dt.metadata.get('class_name', '')):
                entry.attrs['MATLAB_class'] = dt.metadata['class_name']
                entry.attrs['MATLAB_object_decode'] = int32(2)
                
        return entry, dt

    @staticmethod
    def __name_gen():
        """Genereates names, going through the alphabet first, then every combination of letters."""
        from itertools import count, product
        for x in 'abcdefghijklmnopqrstuvwxtz':
            yield x
        for i in count(2):
            for x in product('abcdefghijklmnopqrstuvwxtz', repeat=i):
                yield ''.join(x)

    def __init__(self, name, shape, dt, entry, is_sparse):
        self._entry = entry
        self._is_sparse = is_sparse
        super(_MAT73Entry, self).__init__(name, shape, dt)
    
    @property
    def header(self):
        h = super(_MAT73Entry, self).header
        attrs = [k+'='+str(v) for k,v in self._entry.attrs.iteritems() if k not in ('MATLAB_class', 'MATLAB_empty', 'MATLAB_sparse')]
        if len(attrs) > 0: h['attrs'] = ','.join(attrs)
        data = self._entry.get('data') if self._is_sparse else self._entry
        if data is None: return h

        if data.compression is not None:
            comp = data.compression
            if data.compression_opts: comp += ':' + str(data.compression_opts)
            h['compression'] = comp
        if data.fletcher32 or data.shuffle or data.scaleoffset is not None:
            filters = []
            if data.fletcher32: filters.append('Fletcher32')
            if data.shuffle: filters.append('shuffle')
            if data.scaleoffset is not None: filters.append('scaleoffset:%d'%data.scaleoffset)
            h['filters'] = ','.join(filters)
        if 0 not in data.shape and data.shape != data.maxshape:
            h['resizeable'] = 'unlimited' if all(x is None for x in data.maxshape) else \
                              'to ('+','.join(('unlimited' if x is None else x) for x in data.maxshape)+')'
        if data.chunks is not None:
            h['chunks'] = data.chunks
        return h

    @property
    def data(self):
        return _MAT73Entry.__get_entry_data(self._entry, self._shape, self._dtype, self._is_sparse)

    @classmethod
    def __get_entry_info(cls, entry):
        raw_dt, clazz, sparse = None, entry.attrs.get('MATLAB_class'), entry.attrs.get('MATLAB_sparse')
        dt = _MAT73Entry.__class2dtype.get(clazz, None)
        metadata = {'global':True} if entry.attrs.get('MATLAB_global', 0) == 1 else {}
        int_dec, obj_dec = entry.attrs.get('MATLAB_int_decode', 0), entry.attrs.get('MATLAB_object_decode', 0)

        if dt is None and obj_dec == 2:
            dt = void # object stored as struct
            metadata['class_name'] = clazz
        elif obj_dec != 0: raise ValueError()
        if (dt is None or int_dec not in (0,1,2) or #pylint: disable=too-many-boolean-expressions
            dt == bool and int_dec != 1 or int_dec != 0 and dt not in (bool, unicode_) or
            dt == unicode_ and int_dec not in (0,2) or int_dec == 2 and dt != unicode_):
            raise ValueError()

        is_sparse = sparse is not None
        if is_sparse:
            shape = (sparse, entry['jc'].shape[0]-1)
            if 'data' in entry: raw_dt = entry['data'].dtype
        elif dt == void:
            field_names = entry.attrs.get('MATLAB_fields', None)
            if field_names is None: field_names = [str(fn) for fn in entry.keys()] # should only by one field here anyways
            else: field_names = [str(fn.tostring()) for fn in field_names] # an object array of ASCII strings
            dt = dtype([(fn, object_) for fn in field_names])
            f1 = next(entry.itervalues(), None)
            if entry.attrs.get('MATLAB_empty', 0) == 1:
                shape = tuple(entry)
            elif f1 is not None and 'MATLAB_class' not in f1.attrs:
                # NOTE: additional check could be:
                # all('MATLAB_class' not in f.attrs f.dtype.kind == 'O' and f.shape == f1.shape for f in entry.itervalues())
                shape = f1.shape[::-1]
            else:
                shape = (1,1) # scalar struct
        elif entry.attrs.get('MATLAB_empty', 0) == 1:
            shape = tuple(entry)
        else:
            shape, raw_dt = entry.shape[::-1], entry.dtype
        if raw_dt is not None and raw_dt.names == ('real','imag'):
            dt = _complexify_dtype(raw_dt[0])
        else:
            dt = dtype((unicode_, 1)) if dt == unicode_ else dtype(dt)
        if raw_dt is not None: dt = dt.newbyteorder(raw_dt.byteorder)

        if len(metadata) > 0:
            dt = dtype(dt, metadata=metadata)
        return shape, dt, is_sparse

    @classmethod
    def __get_entry_data(cls, entry, shape, dt, is_sparse):
        # Note: pylint thinks a call to a vectorized function returns a tuple...
        #pylint: disable=no-member
        if is_sparse:
            if 'data' not in entry: return scipy.sparse.csc_matrix(shape, dtype=dt)
            return scipy.sparse.csc_matrix(
                (cls.__read_array(entry['data'], dt),
                 cls.__read_array(entry['ir']),
                 cls.__read_array(entry['jc'])), shape=shape, dtype=dt)
        if 0 in shape: return empty(shape, dt)
        if dt.kind == 'O': # cell array
            data = cls.__read_array(entry, dt, shape[::-1])
            # Dereference every value
            f = entry.file
            def deref(elem):
                subentry = f[elem]
                return cls.__get_entry_data(subentry, *cls.__get_entry_info(subentry))
            return vectorize(deref, otypes=(object_,))(data).T
        if dt.kind == 'V':
            f1 = next(entry.itervalues(), None)
            data = empty(shape, dt)
            if f1 is not None and 'MATLAB_class' not in f1.attrs:
                for fn in dt.names:
                    subentry = entry[fn]
                    data[fn] = cls.__get_entry_data(subentry, shape, dtype(object_), False)
            else:
                for fn in dt.names:
                    subentry = entry[fn]
                    data[0,0][fn] = cls.__get_entry_data(subentry, *cls.__get_entry_info(subentry))
            return data
        if dt.kind in 'SU': # char array, needs decoding
            int_decode = entry.attrs['MATLAB_int_decode']
            data = codecs.decode(cls.__read_array(entry).data,
                                 ('utf_16' + ('le' if _get_dtype_endian(dt) == '<' else 'be')) if int_decode == 2 else 'latin1')
            dt = (dtype((unicode_, 1)) if dt.metadata is None else
                  dtype((unicode_, 1), metadata=dt.metadata)).newbyteorder(dt.byteorder)
            data = unicode_(data).reshape(1).view(dt).reshape(shape, order='F')
            return data.byteswap(True) if _get_dtype_endian(dt) != sys_endian else data
        # numeric array
        return cls.__read_array(entry, dt, shape[::-1]).T

    @classmethod
    def __read_array(cls, entry, dt=None, sh=None):
        arr = empty(sh or entry.shape, dt or entry.dtype)
        entry.read_direct(arr)
        return arr

class _MAT73DummyEntry(_MATDummyEntry):
    def __init__(self, name, entry):
        self._entry = entry
        super(_MAT73DummyEntry, self).__init__(name)


########### General ###########
def get_mat_version(f):
    """
    Return 0 for v4, 1 for v6/7, 2 for v7.3, or False if the version number cannot be determined.
    """
    f.seek(0)
    header = f.read(128)

    # Check for v4
    if _MAT4Entry.openable(header): return 0

    # Check v6/7 and v7.3
    if len(header) != 128 or any(x == 0 for x in header[:4]): return False
    endian = header[126:128]
    if endian not in (b'IM',b'MI'): return False
    major = header[125] if endian == b'IM' else header[124]
    return major if major in (1,2) else False

_MAT_open = [_MAT4File.open, _MAT5File.open, _MAT73File.open if _h5py_avail else None]
_MAT_create = {
    4: _MAT4File.create,
    6: _MAT5File.create,
    7: lambda fn,mode:_MAT5File.create(fn,mode,True),
    7.3: _MAT73File.create if _h5py_avail else None
    }

def openmat(filename, mode='r+', vers=None):
    """
    Opens a MATLAB data file. The filename must include the .mat extension. The mode must be r, r+,
    w or w+. The mode r+ is the default, opening an existing file and allowing for reading and
    writing. When creating new files (mode w or w+) you can specify one of 4, 6, 7, or 7.3 for the
    version number of the file to create. The default is 7.3 if the h5py module is available,
    otherwise version 7 is used.
    """
    if mode in ('r', 'r+'):
        with open(filename, 'rb') as f: vers = get_mat_version(f)
        if vers is False: raise ValueError('Unknown MATLAB file version')
        if vers == 2 and not _h5py_avail: raise ValueError('MATLAB v7.3 file requires h5py library')
        return _MAT_open[vers](filename, mode)
    elif mode in ('w', 'w+'):
        if vers is None: vers = 7.3 if _h5py_avail else 7
        elif isinstance(vers, str) and len(vers) > 0 and vers[0] == 'v': vers = vers[1:]
        try: vers = float(vers)
        except ValueError: raise ValueError('Invalid version, must be one of 4, 6, 7, 7.3')
        if vers not in (4, 6, 7, 7.3): raise ValueError('Invalid version, must be one of 4, 6, 7, 7.3')
        if vers == 7.3 and not _h5py_avail: raise ValueError('MATLAB v7.3 file requires h5py library')
        return _MAT_create[vers](filename, mode)
    else:
        raise ValueError('Mode for opening MATLAB file must be r, r+, w, or w+')

def mat_nice(mat, squeeze_me=True, chars_as_strings=True, inplace=True):
    """
    Makes using an array obtained from a MATLAB file nicer for using with Numpy. The entries in
    MATLAB data files are always at least 2D and store character data as individual characters
    instead of strings. This function can remove those 'features.' First, it always makes sure the
    byte order is the system byte order. If squeeze_me is True, all unit dimensions are squeezed
    out (if there is <=2 dimensions). If chars_as_strings is True, char arrays convert the last
    dimension into strings.
    """
    # Note: pylint thinks a call to a vectorized function returns a tuple...
    #pylint: disable=no-member

    if isinstance(mat, _MATEntry):
        mat = mat.data
        inplace = True
    
    if squeeze_me and mat.ndim <= 2 and 1 in mat.shape and 0 not in mat.shape:
        mat = mat[0,0] if mat.shape == (1,1) else mat.squeeze()

    dt = mat.dtype
    if dt.kind == 'O':
        mn = lambda m:mat_nice(m, squeeze_me, chars_as_strings, inplace)
        if inplace:
            mat_flat = mat.flat
            for i in xrange(mat.size): mat_flat[i] = mn(mat_flat[i])
        else: mat = vectorize(mn, otypes=(object_,))(mat)
        
    elif dt.kind == 'V':
        mn = lambda m:mat_nice(m, squeeze_me, chars_as_strings, inplace)
        out = mat if inplace else empty(mat.shape, mat.dtype)
        for fn in dt.names: out[fn] = mn(mat[fn])
        mat = out
        
    elif chars_as_strings and mat.ndim > 0 and mat.shape[-1] > 1 and (dt.kind, dt.itemsize) in (('S',1), ('U',4)):
        dt = dtype((dt.type, mat.shape[-1])).newbyteorder(_get_dtype_endian(dt))
        mat = ascontiguousarray(mat).view(dt).squeeze(-1)
        if mat.ndim == 0: mat = mat[()]
    
    if _get_dtype_endian(dt) != sys_endian:
        mat = mat.newbyteorder(sys_endian).byteswap(inplace)
    
    return mat
