from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, re
from time import asctime
from io import open, SEEK_CUR, SEEK_END #pylint: disable=redefined-builtin
from collections import OrderedDict
from itertools import izip, islice, chain
from struct import Struct, pack, unpack, error as StructError
from weakref import proxy
from abc import ABCMeta, abstractproperty, abstractmethod
from warnings import warn

from numpy import dtype, array, empty, frombuffer, ascontiguousarray, asfortranarray, concatenate, delete
from numpy import (complex64, complex128, float32, float64,
                    int8, int16, int32, int64, intc,
                    uint8, uint16, uint32, uint64)
import scipy.sparse

from .._util import imread_raw, imsave_raw, get_file_size, copy_data, file_remove_ranges, FileInsertIO
from ..._util import String, Byte, prod, sys_endian, sys_64bit, _bool
from ...types import create_im_dtype, get_dtype_endian, is_image_desc, im_dtype_desc
from ...types import im_complexify_dtype, im_decomplexify, im_decomplexify_dtype
from ...source import ImageSource
from ..._stack import ImageStack
from .._single import FileImageSource
from .._stack import FileImageStack, HomogeneousFileImageStack, FileImageSlice, FileImageStackHeader, FixedField
from ....general.gzip import GzipFile

try:
    from h5py import File as HDF5File
    h5py_avail = True
except ImportError:
    h5py_avail = False

__all__ = ['get_mat_version', 'MAT', 'MATStack']

__matlab_keywords = frozenset((
    'break', 'case', 'catch', 'classdef', 'continue', 'else', 'elseif', 'end', 'for', 'function'
    'global', 'if', 'otherwise', 'parfor', 'persistent', 'return', 'spmd', 'switch', 'try', 'while'))
__re_matlab_name = re.compile('^[a-zA-Z][a-zA-Z0-9_]{0,62}$')
def _is_invalid_matlab_name(name): return name in __matlab_keywords or __re_matlab_name.match() is None

__platforms = {
    'nt':'PCWIN',
    'linux':'GLNXA64' if sys_64bit else 'GLNX86',
    }
def _matlab_platform(): return __platforms.get(os.name, os.name)

def _squeeze2(a):
    if a.ndim < 2: return a.reshape((a.size, 1), order='F')
    for i in reversed([i for i in xrange(2, a.ndim) if a.shape[i] == 1]): a = a.squeeze(i)
    return a

class _MATFile(object):
    __metaclass__ = ABCMeta
    # subclasses must also have a _version property for the header property
    # subclasses must also define class methods for open(fn, mode) and create(fn, mode)
    # The collection/mapping functions (e.g. len, in, [], and iterating) ignore/skip dummy entries

    _version = None
    @classmethod
    def _basic_open(cls, fn, mode):
        return open(fn, mode+'b')

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
        self._f.close()
        self._f = None
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
    def append(self, name, im): pass # returns the new entry
    @abstractmethod
    def insert_before(self, name, new_name, im): pass # returns the new entry
    @abstractmethod
    def set(self, name, im): pass # returns the new entry
    def __setitem__(self, name, im): self.set(name, im)
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
    @property
    def header(self): return {'name':self._name}
    @property
    def shape(self): return self._shape # always the full shape, except for real/imag channels [also note that I believe always this will be 2D or greater with all higher-dimensions squeezed]
    @property
    def dtype(self): return self._dtype # always a "base" dtype except for the rare non-native complex types which will give 2 channels
    @property
    def is_image_slice(self): return is_image_desc(self._dtype, self._shape)
    @property
    def is_image_stack(self): return is_image_desc(self._dtype, self._shape[1:])
    @abstractproperty
    def data(self): pass

class _MATDummyEntry(_MATEntry): # a placeholder for an entry that we cannot process
    def __init__(self, name): self._name = name #pylint: disable=super-init-not-called
    @property
    def header(self): raise RuntimeError()
    @property
    def shape(self): raise RuntimeError()
    @property
    def dtype(self): raise RuntimeError()
    @property
    def is_image_slice(self): return False
    @property
    def is_image_stack(self): return False
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
                name = ('__no_name_%03d'%len(entries)) if e.name is None else e.name
                entries[name] = e
        except EOFError: pass
        except StructError as ex: raise ValueError(ex.message)
        return mat

    def rename(self, renamer, filename):
        mode = self._f.mode
        self._f.close()
        renamer(filename)
        self._f = open(filename, mode)
        self._filename = filename
    
    def append(self, name, im):
        if name in self._entries or _is_invalid_matlab_name(name): raise KeyError()
        self._f.seek(0, SEEK_END)
        self._entries[name] = entry = self._entry.create(self, self._f, name, im)
        return entry
    
    def insert_before(self, name, new_name, im):
        #pylint: disable=undefined-loop-variable
        if new_name in self._entries or _is_invalid_matlab_name(new_name): raise KeyError()
        
        for idx, (n,cur_entry) in enumerate(self._entries.iteritems()):
            if n == name: break
        else: raise KeyError()
        start = cur_entry._start
        new_size = self._entry._calc_size(self, new_name, im)
        if new_size == -1:
            with FileInsertIO(self._f, start, 0) as f:
                entry = self._entry.create(self, f, new_name, im)
        else:
            copy_data(self._f, start, start + new_size)
            if self._f.seek(start) != start: raise IOError()
            entry = self._entry.create(self, self._f, new_name, im)
        self._entries[name] = entry # currently added to the end of the order, but during the next loop we fix that

        # Update the entry starts
        for name in list(islice(self._entries, idx, None)):
            self._entries[name] = e = self._entries.pop(name) # remove entry and put it at end
            e._update_start(e._start + new_size)
        
        return entry
    
    def set(self, name, im):
        #pylint: disable=undefined-loop-variable
        if len(self._entries) > 0 and name == next(reversed(self._entries)):
            # We are replacing the last entry, this one is easy (delete last entry and append)
            e = self._entries.popitem()[1]
            e._deleting()
            self._f.truncate(e._start)
            return self.append(name, im)
        items = self._entries.iteritems()
        for idx, (n,cur_entry) in enumerate(items):
            if n == name: break
        else: return self.append(name, im) # doesn't exist yet, add it to the end
        next_entry = next(items)[1]

        # Write the new entry
        start, next_start = cur_entry._start, next_entry._start
        new_size = self._entry._calc_size(self, name, im)
        if new_size == -1:
            with FileInsertIO(self._f, start, next_start - start) as f:
                entry = self._entry.create(self, f, name, im)
            new_end = self._f.tell()
        else:
            new_end = start + new_size
            copy_data(self._f, next_start, new_end)
            if self._f.seek(start) != start: raise IOError()
            entry = self._entry.create(self, self._f, name, im)
        self._entries[name] = entry
        delta = new_end - next_start
        
        # Update the entry starts
        for e in islice(self._entries.itervalues(), idx+1, None):
            e._update_start(e._start + delta)
        
        return entry
    
    def __delitem__(self, name):
        from numpy import append, fromiter, cumsum, diff
        starts = append(fromiter(e._start for e in self._entries.itervalues()), get_file_size(self._f))

        if isinstance(name, String):
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
    # subclasses must define class methods for open(mat,f) and create(mat,f,name,im)
    @abstractproperty
    def data(self): pass
    @classmethod
    def _calc_size(cls, mat, name, im): #pylint: disable=unused-argument
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
    __long_le = Struct(str('<l'))
    __structs = (Struct(str('<lllll')), Struct(str('>lllll')))
    __HDR_SIZE = 20
    
    # Numeric format:
    __LITTLE_ENDIAN = 0
    __BIG_ENDIAN = 1
    #__VAX_D_FLOAT = 2, __VAX_G_FLOAT = 3, __CRAY = 4
    __NUMERIC_FORMATS = (__LITTLE_ENDIAN, __BIG_ENDIAN)
    
    # Matrix Type:
    __FULL_MATRIX = 0
    #__TEXT_MATRIX = 1
    __SPARSE_MATRIX = 2
    __MATRIX_TYPES = (__FULL_MATRIX, __SPARSE_MATRIX)

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
        raw_dt = dt = create_im_dtype(cls.__type2dtype[P], endian==cls.__BIG_ENDIAN, 2 if cmplx else 1)
        col_size = mrows*dt.itemsize
        total_size = ncols*col_size
        off = start+cls.__HDR_SIZE+namlen
        end = off+total_size
        if M not in cls.__NUMERIC_FORMATS or T not in cls.__MATRIX_TYPES:
            if f.seek(total_size, SEEK_CUR) != end: raise ValueError()
            return _MAT45DummyEntry(mat, start, name)
        if M != endian: raise ValueError('Endian mismatch') # NOTE: we assume all entries have the same endian-ness
        ### Handle sparse matrix
        if T == cls.__SPARSE_MATRIX:
            if cmplx or (ncols not in (3,4)) or mrows < 1: raise ValueError()
            data_size = col_size*(ncols-2)
            dt_sz = dt.itemsize
            if ncols == 4:
                try: dt = im_complexify_dtype(dt)
                except ValueError: pass
            col_size -= dt_sz
            f.seek(col_size, SEEK_CUR); mr = frombuffer(f.read(dt_sz), raw_dt, 1)
            f.seek(col_size, SEEK_CUR); nc = frombuffer(f.read(dt_sz), raw_dt, 1)
            if f.seek(data_size, SEEK_CUR) != end: raise ValueError()
            shape = (int(mr[0]),int(nc[0]))
        else:
            if f.seek(total_size, SEEK_CUR) != end: raise ValueError()
            shape = (mrows,ncols)
            if cmplx:
                try: dt = im_complexify_dtype(dt)
                except ValueError: pass
                
        ### Make the entry
        return cls(mat, start, off, name, (mrows,ncols), shape, raw_dt, dt)
    
    @classmethod
    def create(cls, mat, f, name, im):
        start = f.tell()
        if mat._endian is None:
            mat._endian = endian = cls.__LITTLE_ENDIAN if sys_endian == '<' else cls.__BIG_ENDIAN
            mat._struct = cls.__structs[endian]
        else:
            endian = mat._endian

        ### Get the data into a decent format
        dt = raw_dt = im.dtype
        im = _squeeze2(im)
        if im.ndim != 2 and (im.ndim != 3 or dt.kind == 'c' or im.shape[2] != 2):
            raise ValueError('Invalid shape for MATLAB v4 file (must be 2D)')
        shape = raw_shape = im.shape[:2]
        
        # Deal with sparse data
        is_sparse = scipy.sparse.isspmatrix(im)
        is_complex = False
        if is_sparse:
            from numpy import add
            if not scipy.sparse.isspmatrix_coo(im): im = im.tocoo()
            sparse, data = im, im.data
            # since sparse matrices cannot have 3 dimensions, we don't need to check for that kind of complex
            if dt.kind == 'c': # is_complex stays False since the raw data is not complex
                raw_dt = im_decomplexify_dtype(dt).base
                im = empty((len(data)+1,4), dtype=raw_dt, order='F')
                im[:-1,2:] = data.view(dtype=dtype((raw_dt,2)))
            else:
                im = empty((len(data)+1,3), dtype=raw_dt, order='F')
                im[:-1,2] = data
            del data
            raw_shape = im.shape
            im[-1,0:2] = shape
            im[-1,2:] = 0
            add(sparse.row, 1, out=im[:-1,0])
            add(sparse.col, 1, out=im[:-1,1])
            del sparse
        elif dt.kind == 'c':
            # Deal with complex data
            is_complex = True
            raw_dt = im_decomplexify_dtype(im.dtype)
            im = im.view(dtype=raw_dt)
        elif im.ndim == 3:
            # Deal with de-complex data
            is_complex = True
            raw_dt = dtype((dt, 2))
            try:               dt = im_complexify_dtype(dt)
            except ValueError: dt = raw_dt
        # Make sure the data is the right byte and Fortran ordering
        if (get_dtype_endian(im.dtype)=='<') != (endian==cls.__LITTLE_ENDIAN): im.byteswap(True)
        im = asfortranarray(im)

        ### Create the header
        M = endian
        P = cls.__dtype2type[raw_dt.base.type]
        T = cls.__SPARSE_MATRIX if is_sparse else cls.__FULL_MATRIX
        mopt = 1000*M+10*P+T
        mrows, ncols = raw_shape
        imagf = 1 if is_complex else 0
        if not isinstance(name, bytes): name = unicode(name).encode('ascii','ignore')
        name += b'\x00'
        namlen = len(name)
        if (f.write(mat._struct.pack(mopt, mrows, ncols, imagf, namlen)) != cls.__HDR_SIZE or
            f.write(name) != namlen): raise IOError('Unable to write header')
        off = start + cls.__HDR_SIZE + namlen
        
        ### Save data
        imsave_raw(f, im)

        ### Make the entry
        return cls(mat, start, off, name, raw_shape, shape, raw_dt, dt)

    @classmethod
    def _calc_size(cls, mat, name, im):
        im, dt = _squeeze2(im), im.dtype
        if im.ndim != 2 and (im.ndim != 3 or dt.kind == 'c' or im.shape[2] != 2):
            raise ValueError('Invalid shape for MATLAB v4 file (must be 2D)')
        if scipy.sparse.isspmatrix(im):
            im_nbytes = (im.nnz+1)*dt.itemsize*(2 if dt.kind == 'c' else 3)
        else: im_nbytes = im.nbytes
        return cls.__HDR_SIZE + len(name) + 1 + im_nbytes
    
    @property
    def data(self):
        f = self._mat._f
        f.seek(self._off)
        im = imread_raw(f, self._raw_shape, self._raw_dt, 'F')
        if self._shape != self._raw_shape: # sparse matrix
            im = im[:-1,:] # last row has shape, we already have that
            I = ascontiguousarray(im[:,0], dtype=intc); I -= 1 # row indices
            J = ascontiguousarray(im[:,1], dtype=intc); J -= 1 # col indices
            V = ascontiguousarray(im[:,2:]).view(self._dtype).squeeze(1) # values
            im = scipy.sparse.coo_matrix((V,(I,J)), self._shape)
        elif self._dtype.kind == 'c':
            im = ascontiguousarray(im).view(self._dtype).squeeze(2)
        return im

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
        # 1,2,3,4: cell array, struct, object (and maybe logical in some old versions?), char
        # 16,17,18: function, opaque, object (again)
    }
    __dtype2class = {
        float64   : 6,
        complex128: 6,
        float32   : 7,
        complex64 : 7,
        int8      : 8,
        uint8     : 9,
        bool      : 9,
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
            dt, nbytes, skip = mat._read_tag(f, first=not is_compressed)
            end = f.tell() + skip
            if dt == 'compressed':
                # Restart using a compressed stream
                if is_compressed: raise ValueError()
                with GzipFile(f, mode='rb', method='zlib') as gzf:
                    e = cls.open(mat, gzf, True)
                e.__start_comp, e._start = e._start, start #pylint: disable=attribute-defined-outside-init
                if f.seek(end) != end: raise ValueError()
                return e
            elif dt != 'matrix': raise ValueError() # at the top level we only accept matrix or compressed
                
            # Parse matrix headers
            flags_class, nzmax = mat._read_subelem(f, uint32, 2)
            clazz = flags_class & 0xFF
            dt = cls.__class2dtype.get(clazz, None)
            if dt is None:
                if clazz != 17: # opaque class does not have shape or name, other unknowns may or may not, tread lightly...
                    try:
                        mat._read_subelem(f, int32) # skip shape
                        name  = mat._read_subelem(f, int8).tostring()
                    except (IOError, ValueError, StructError): pass
                raise NotImplementedError()
            is_logical = (flags_class & 0x0200) != 0
            is_global  = (flags_class & 0x0400) != 0
            is_complex = (flags_class & 0x0800) != 0
            is_sparse  = (flags_class & 0x1000) != 0 or dt=='sparse'
            shape = tuple(mat._read_subelem(f, int32).tolist())
            name  = mat._read_subelem(f, int8).tostring()
            if name == '': raise NotImplementedError() # no-name matrix added if there are anonymous functions, its their workspace
            if (flags_class & 0xFFFFE100) != 0:
                warn("MATLAB file entry '%s' uses unknown flags (#%x) and the data may be wrong or the file may fail to read - send the file in"%(name,flags_class&0xFFFFE100))

            # Parse matrix data areas
            endian = mat._endian
            off = f.tell()
            sub_dt = uint8 if (flags_class & 0x1200) == 0x1200 else None
            if is_sparse:
                if len(shape) > 2: raise ValueError()
                if dt == 'sparse': dt = float64 # we have to assume that it is this way since we have no other information
                skip = mat._read_tag(f, int32, nzmax)[2]
                f.seek(skip, SEEK_CUR)
                _, nbytes, skip = mat._read_tag(f, int32, (0 if len(shape) < 2 else shape[1])+1)
                # Get the last value in that array which is the actual number of non-zero values
                f.seek(nbytes-4, SEEK_CUR)
                nnz, = unpack(str(endian+'i'), f.read(4))
                f.seek(skip-nbytes, SEEK_CUR)
                raw_shape = (nnz,)
                size = nnz
            else:
                if nzmax != 0:
                    warn("MATLAB file entry '%s' has a nzmax value (%d) and is not sparse - send the file in"%(name,nzmax))
                raw_shape = shape
                size = prod(shape)

            real_dt, _, skip = mat._read_tag(f, None, size, sub_dt=sub_dt)
            if not isinstance(real_dt, type): raise NotImplementedError()
            raw_dt = create_im_dtype(real_dt, endian)
            if not is_compressed:
                # For compressed data we do no further checks, they are just too expensive
                if is_complex:
                    # Check the imaginary data (but not if we are compressed - its just too expensive)
                    f.seek(skip, SEEK_CUR)
                    imag_dt, _, skip = mat._read_tag(f, None, size, sub_dt=sub_dt)
                    if not isinstance(imag_dt, type): raise NotImplementedError()
                    raw_dt = dtype((raw_dt, 2) if real_dt == imag_dt else \
                                   [('real', raw_dt), ('imag', create_im_dtype(imag_dt, endian))])
                if f.seek(skip, SEEK_CUR) != end: raise ValueError()
            
        except NotImplementedError:
            if not is_compressed and f.seek(end) != end: raise ValueError()
            return _MAT5DummyEntry(mat, start, name)

        # Finish up
        dt = create_im_dtype(bool if is_logical else dt, endian)
        if is_complex:
            dt = dtype((dt, 2))
            try: dt = im_complexify_dtype(dt)
            except ValueError: pass

        # Create entry
        return cls(mat, start, off, name, raw_shape, shape, raw_dt, dt,
                   is_compressed, is_global, is_complex, is_sparse, sub_dt)

    @classmethod
    def create(cls, mat, f, name, im, is_compressed=False):
        endian = mat._endian
        start = f.tell()

        if not is_compressed and mat._version == 7:
            # Restart with a compressed file stream
            if (f.write(mat._long.pack(15)) != 4 or 
                f.write(mat._long.pack(0))  != 4): raise ValueError()
            with GzipFile(f, mode='wb', method='zlib') as gzf:
                e = cls.create(mat, gzf, name, im, True)
            e.__start_comp, e._start = e._start, start
            # Go back and write the size of the compressed data
            end = f.tell()
            if (f.seek(start+4) != start+4 or
                f.write(mat._long.pack(end-start-8)) != 4 or
                f.seek(end) != end): raise ValueError()
            return e

        # Convert the image into a usable format while getting all the properties
        im = _squeeze2(im)
        raw_shape, raw_dt = im.shape, im.dtype
        shape, dt = raw_shape, create_im_dtype(raw_dt, endian)
        dims, nzmax, sub_dt = raw_shape, 0, None
        
        clazz = cls.__dtype2class[raw_dt.type]
        if not isinstance(name, bytes): name = unicode(name).encode('ascii','ignore')
        is_logical = raw_dt.kind == 'b'
        is_global  = False
        is_complex = raw_dt.kind == 'c'
        is_sparse  = scipy.sparse.isspmatrix(im)
        flags = ((0x0200 if is_logical else 0) |
                 (0x0400 if is_global  else 0) |
                 (0x0800 if is_complex else 0) |
                 (0x1000 if is_sparse  else 0))

        def correct_ordering(a):
            # Make sure the data is the right byte and Fortran ordering
            if (get_dtype_endian(im.dtype)=='<') != (endian=='<'): a.byteswap(True)
            return asfortranarray(a)

        def reduce_size(a):
            # Checks to see if a type can be stored in a smaller size according to MATLAB rules
            if a.dtype == float64 and not (a%1.0).any():
                # Have all integers (stored as floats)
                # Check the min and max to see if we can fit it in one of the int data types
                mn, mx = long(a.min()), long(a.max())
                if mn < 0:
                    if   mn >= -32768      and mx <= 32767:      a = a.astype(int16)
                    elif mn >= -2147483648 and mx <= 2147483647: a = a.astype(int32)
                else:
                    if   mx <= 255:        a = a.astype(uint8)
                    elif mx <= 65535:      a = a.astype(uint16)
                    elif mx <= 2147483647: a = a.astype(int32)
            return a
        
        if is_sparse:
            if raw_dt in (bool, float64, complex128): clazz = 5
            if not scipy.sparse.isspmatrix_csc: im = im.tocsc()
            IR = im.indices.astype(int32, copy=False)
            JC = concatenate((im.indptr.astype(int32, copy=False), array((im.nnz,), int32)))
            im = im.data
            nzmax = len(IR)
            raw_shape = im.shape
            ## TODO: sub_dt = ...
        
        if is_complex:
            # Deal with complex data
            im = correct_ordering(im_decomplexify(im))
            real = reduce_size(im[...,0])
            imag = reduce_size(im[...,1])
            raw_dt = dtype((real.dtype, 2)) if real.dtype == imag.dtype else \
                     dtype([('real', real.dtype), ('imag', imag.dtype)])
        else:
            # Unlike in v4 files we do not assume every 2 channel image is complex because v5 supports nd matrices
            real = reduce_size(correct_ordering(im))
            raw_dt = real.dtype
            

        # Calculate the size of the matrix in the file
        def get_data_size(nbytes):
            if nbytes <= 4: return 8
            pad = (8-(nbytes%8))
            if pad == 8: pad = 0
            return 8 + nbytes + pad
        nbytes = 16 + get_data_size(4*len(dims)) + get_data_size(len(name))
        off = start + nbytes
        if is_sparse: nbytes += get_data_size(IR.nbytes) + get_data_size(JC.nbytes)
        nbytes += get_data_size(real.nbytes)
        if is_complex: nbytes += get_data_size(imag.nbytes)

        # Write the data
        if (f.write(mat._long.pack(14))     != 4 or # Matrix tag
            f.write(mat._long.pack(nbytes)) != 4): raise ValueError()
        mat._write_subelem(f, array((flags|clazz,nzmax), uint32))
        mat._write_subelem(f, array(dims, int32))
        mat._write_subelem(f, frombuffer(name, uint8))
        if is_sparse:
            mat._write_subelem_big(f, IR)
            mat._write_subelem_big(f, JC)
        mat._write_subelem_big(f, real)
        if is_complex: mat._write_subelem_big(f, imag)

        # Create entry
        return cls(mat, start, off, name, raw_shape, shape, raw_dt, dt,
                   is_compressed, is_global, is_complex, is_sparse, sub_dt)
    
    def __init__(self, mat, start, off, name, raw_shape, shape, raw_dt, dt,
                 is_compressed, is_global, is_complex, is_sparse, sub_dt):
        super(_MAT5Entry, self).__init__(mat, start, off, name, raw_shape, shape, raw_dt, dt)
        self.__start_comp = None
        self.__is_compressed = is_compressed
        self.__is_global  = is_global
        self.__is_complex = is_complex
        self.__is_sparse  = is_sparse
        self.__sub_dt = sub_dt

    @classmethod
    def _calc_size(cls, mat, name, im):
        if mat._version == 7: return -1 # we cannot calculate the size of compressed entries
        
        if not isinstance(name, bytes): name = unicode(name).encode('ascii','ignore')

        def get_reduced_size(a):
            itemsize = a.dtype.itemsize
            if a.dtype == float64 and not (a%1.0).any():
                mn, mx = long(a.min()), long(a.max())
                if mn < 0:
                    if   mn >= -32768      and mx <= 32767:      itemsize = 2
                    elif mn >= -2147483648 and mx <= 2147483647: itemsize = 4
                else:
                    if   mx <= 255:        itemsize = 1
                    elif mx <= 65535:      itemsize = 2
                    elif mx <= 2147483647: itemsize = 4
            return a.size * itemsize
        
        im = _squeeze2(im)
        dims = im.shape
        is_sparse  = scipy.sparse.isspmatrix(im)
        if is_sparse:
            if not scipy.sparse.isspmatrix_csc: im = im.tocsc()
            IR_nbytes = len(im.indices)*4
            JC_nbytes = (len(im.indptr)+1)*4
        is_complex = im.dtype.kind == 'c'
        if is_complex:
            im = im_decomplexify(im)
            real_nbytes = get_reduced_size(im[...,0])
            imag_nbytes = get_reduced_size(im[...,1])
        else:
            real_nbytes = get_reduced_size(im)

        def get_data_size(nbytes):
            if nbytes <= 4: return 8
            pad = (8-(nbytes%8))
            if pad == 8: pad = 0
            return 8 + nbytes + pad
        nbytes = 8 + get_data_size(4*len(dims)) + get_data_size(len(name))
        if is_sparse: nbytes += get_data_size(IR_nbytes) + get_data_size(JC_nbytes)
        if is_complex: nbytes += get_data_size(imag_nbytes)
        return nbytes + get_data_size(real_nbytes)

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
        if self.__is_global or self.__is_compressed:
            flags = []
            if self.__is_global:     flags.append('global')
            if self.__is_compressed: flags.append('compressed')
            h['flags'] = ','.join(flags)
        dt = self._dtype
        if dt.kind == 'c': dt = im_decomplexify_dtype(dt)
        elif dt.kind == 'b': dt = dtype(uint8)
        if dt.base != self._raw_dt.base:
            h['storage-type'] = im_dtype_desc(self._raw_dt.base) if self._raw_dt.names is None else \
                                ','.join(im_dtype_desc(self._raw_dt[i]) for i in xrange(len(self._raw_dt)))
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
        # Handle sparse data
        if self.__is_sparse:
            shape = self._shape
            nnz, N = self._raw_shape[0], 0 if len(shape) < 2 else shape[1]
            IR = mat._read_subelem_big(f, int32, (nnz,), True) # row indices
            JC = mat._read_subelem_big(f, int32, (N+1,))       # col indices
        # Load data (either complex or regular)
        dt = self._dtype
        im = mat._read_subelem_big(f, None, self._raw_shape, sub_dt=self.__sub_dt)
        if self.__is_complex:
            real = im
            imag = mat._read_subelem_big(f, None, self._raw_shape, sub_dt=self.__sub_dt)
            im = empty(self._raw_shape, im_decomplexify_dtype(dt), 'C')
            im[...,0] = real
            im[...,1] = imag
            del real, imag
            if dt.kind == 'c': im = im.view(dt).squeeze(-1)
        else:
            im = im.astype(dt, copy=False)
        # Return the image
        if self.__is_sparse:
            im = scipy.sparse.csc_matrix((im[:nnz],IR,JC), shape=shape)
        return im

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
        # 16,17,18: utf8, utf16, utf32
    }
    __dtype2mat = {
        int8        : 1,
        uint8       : 2,
        int16       : 3,
        uint16      : 4,
        int32       : 5,
        uint32      : 6,
        float32     : 7,
        float64     : 9,
        int64       : 12,
        uint64      : 13,
    }
    
    @classmethod
    def open(cls, fn, mode):
        f = cls._basic_open(fn, mode)
        try:
            header = f.read(128)
            if len(header) != 128 or any(x == 0 for x in header[:4]): raise ValueError() # Always supposed to have non-null bytes in first 4 bytes
            version, endian = (Byte(header[125]),Byte(header[124])), header[126:128]
            if endian not in (b'IM',b'MI'): raise ValueError()
            endian = '<' if endian == b'IM' else '>'
            if endian == '>': version = version[::-1]
            if version[0] != 1: raise ValueError()
            text = header[:116].rstrip(b'\0').rstrip() # or [:124]?
            ssdo = header[116:124]
            ssdo = None if ssdo in (b'        ',b'\0\0\0\0\0\0\0\0') else unpack(str(endian+'Q'), ssdo)[0]
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
            text = b'MATLAB 5.0 MAT-file, Platform: %s, Created on: %s' % (_matlab_platform(), asctime())
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
        self._long = Struct(str(endian+'L'))
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
        if self._f.seek(116) != 116 or self._f.write(pack(str(self._endian+'Q'), ssdo)) != 8: raise IOError()

    def _read_tag(self, f, expected_type=None, expected_nvals=None, first=False, sub_dt=None):
        """
        Reads a tag header.

        Return the type of the data (None, a special string, or a numeric type), the number of bytes
        of data, and the number bytes including padding (enough to skip right over the data). The
        file itself is positiioned immediately after the tag so data is ready to be read or skipped.

        f is usually self._f, however it may also be a GzipFile-wrapped version of self._f as well.

        If expected_type or expected_nvals is not None this also checks the type and/or number of
        values. For special types (like compressed and matrix), do not provide expected_nvals.

        If first is True, EOFError instead of ValueError is raised if the first read reads no data.

        If sub_dt is not None, it is used instead of whatever the actual dtype is found to be.
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
        if sub_dt is not None:
            dt = sub_dt
        else:
            dt = _MAT5File.__mat2dtype.get(mdt, None)
            if dt is None: raise NotImplementedError()
        if ((expected_type is not None and expected_type != dt) or
            (expected_nvals is not None and expected_nvals != nbytes//dtype(dt).itemsize)):
            raise ValueError()
        return dt, nbytes, skip
    
    def _read_subelem(self, f, expected_type=None, expected_nvals=None):
        """
        Read an entire subelement. Uses _read_tag then reads the data and uses frombuffer. It's type
        cannot be a special type (like compressed or matrix).

        Returns the data as a 1D ndarray. The file is positioned at the end of the subelement.
        """
        dt, nbytes, skip = self._read_tag(f, expected_type, expected_nvals)
        data = f.read(skip)
        if not isinstance(dt, type) or len(data) != skip: raise ValueError()
        dt = dtype(dt).newbyteorder(self._endian)
        return frombuffer(data, dt, nbytes//dt.itemsize)

    def _read_subelem_big(self, f, dt, shape, relax=False, sub_dt=None):
        """
        Like _read_subelem except designed for large arrays. Instead of read/frombuffer this uses
        imread_raw (which uses either fromfile or readinto). The dtype and shape must always be
        known. Make sure the dtype has the right byteorder. If relax is True than the number of
        elements in the subelem must only be at least as many as the shape requests. Returns an
        ndarray of the given dtype and shape.
        """
        dt, nbytes, skip = self._read_tag(f, dt, None if relax else prod(shape), sub_dt=sub_dt)
        if not isinstance(dt, type) or relax and prod(shape) >= nbytes: raise ValueError()
        data = imread_raw(f, shape, dtype(dt).newbyteorder(self._endian), 'F')
        f.seek(skip-data.nbytes, SEEK_CUR)
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

    def _write_subelem(self, f, data):
        """
        Writes an entire subelement to the given file using data from the ndarray. The ndarray
        specifies the type and number of bytes to list in the tag. This will also write the
        necessary padding after the subelement.
        """
        dt, nbytes = data.dtype.type, data.nbytes
        self._write_tag(f, dt, nbytes)
        if f.write(data.data) != nbytes: raise ValueError()
        pad = (4-nbytes) if nbytes <= 4 else (8-(nbytes%8))
        if pad not in (0,8) and f.write(b'\0'*pad) != pad: raise ValueError()

    def _write_subelem_big(self, f, data):
        """
        Like _write_subelem but uses imsave_raw instead of f.write. This is meant for large amounts
        of data and can be faster.
        """
        dt, nbytes = data.dtype.type, data.nbytes
        self._write_tag(f, dt, nbytes)
        imsave_raw(f, data)
        pad = (4-nbytes) if nbytes <= 4 else (8-(nbytes%8))
        if pad not in (0,8) and f.write(b'\0'*pad) != pad: raise ValueError()


########### MATLAB v7.3 Files ###########
class _MAT73File(_MATFile):
    """MAT v7.3 files do not support "ordering", so some methods won't work as expected."""
    #pylint: disable=protected-access
    _version = 7.3

    @classmethod
    def open(cls, fn, mode):
        f = HDF5File(fn, mode)
        try:
            if f.userblock_size < 128: raise ValueError()
            with open(fn, 'rb') as fh: header = fh.read(128)
            if len(header) != 128 or any(x == 0 for x in header[:4]): raise ValueError() # Always supposed to have non-null bytes in first 4 bytes
            version, endian = (Byte(header[125]),Byte(header[124])), header[126:128]
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
        text = b'MATLAB 7.3 MAT-file, Platform: %s, Created on: %s HDF5 schema 1.00 .' % (_matlab_platform(), asctime())
        header[:len(text)] = text
        header[len(text):116] = b' '*(116-len(text))
        header[124:128] = b'\x00\x02IM' if sys_endian == '<' else b'\x02\x00MI'
        f = HDF5File(fn, 'w' if mode == 'w+' else mode, libver='earliest', userblock_size=512L)
        try:
            with open(fn, 'wb') as fh:
                if fh.write(header) != 128: raise IOError()
        except: f.close(); raise
        return  cls(fn, f, OrderedDict(), text, (2, 0))

    def __init__(self, fn, f, entries, header_text, version):
        super(_MAT73File, self).__init__(fn, f, entries)
        self.__header_text = header_text
        self.__version = version

    def rename(self, renamer, filename):
        mode = self._f.mode
        self._f.close()
        renamer(filename)
        self._f = HDF5File(filename, mode)
        self._filename = filename
        for entry,h5ent in izip(self._entries.itervalues(), self._f.itervalues()):
            assert entry._name == h5ent.name.lstrip('/')
            entry._entry = h5ent

    @property
    def header(self):
        h = super(_MAT73File, self).header
        h.update({'mat-version':'.'.join(str(x) for x in self.__version),'text':self.__header_text})
        return h

    def append(self, name, im):
        if name in self._entries or _is_invalid_matlab_name(name): raise KeyError()
        self._entries[name] = entry = _MAT73Entry.create(self._f, name, im)
        return entry

    def insert_before(self, name, new_name, im):
        # This just results in an append
        return self.append(new_name, im)

    def set(self, name, im):
        # The data is appended unless it can be placed into the old slot (means dtype and type need to be the same and shape must be compatible)
        if name not in self._entries: return self.append(name, im)
        entry = self._entries[name]
        if not entry.is_image: return self.__simple_set(name, im)
        old_shape, old_dt = entry.shape + entry.dtype.shape, entry.dtype.base
        new_shape, new_dt = im.shape + im.dtype.shape, im.dtype.base
        if new_dt.type != old_dt.type or len(old_shape) != len(new_shape):
            # Technically for things that were/are empty some things may be possible with this, but we aren't saving much and that is rare
            return self.__simple_set(name, im)
        was_empty, is_empty = 0 in old_shape, 0 in new_shape
        was_sparse, is_sparse = entry._is_sparse, scipy.sparse.isspmatrix(im)
        h5ent = entry._entry
        if was_sparse != is_sparse: return self.__simple_set(name, im)
        elif was_sparse:
            if im.nzz == 0 and 'data' in h5ent: del h5ent['ir'], h5ent['data']
            if is_empty:
                h5ent.attrs['MATLAB_sparse'] = uint64(0)
                _MAT73File.__force_data(h5ent, 'jc', array(0, dtype=uint64))
            else:
                if not scipy.sparse.isspmatrix_csc: im = im.tocsc()
                h5ent.attrs['MATLAB_sparse'] = uint64(new_shape[0])
                _MAT73File.__force_data(h5ent, 'jc', im.indptr.astype(uint64, copy=False))
                if im.nzz != 0:
                    _MAT73File.__force_data(h5ent, 'ir', im.indices.astype(uint64, copy=False))
                    im = im.data
                    if get_dtype_endian(new_dt) != get_dtype_endian(old_dt): im = im.byteswap()
                    _MAT73File.__force_data(h5ent, 'data', im)
        elif was_empty != is_empty: return self.__simple_set(name, im)
        elif was_empty: h5ent.write_direct(array(new_shape, dtype=uint64))
        else:
            if old_shape != new_shape:
                new_sh = new_shape[::-1]
                if h5ent.chunks is None or any(n>m for n,m in izip(new_sh,h5ent.maxshape) if m is not None):
                    return self.__simple_set(name, im)
                h5ent.resize(new_sh)
            if get_dtype_endian(new_dt) != get_dtype_endian(old_dt): im = im.byteswap()
            h5ent.write_direct(ascontiguousarray(im.T))
        entry._shape = new_shape
        entry._dtype = new_dt
        return entry

    def __simple_set(self, name, im):
        del self._f[name]
        del self._entries[name]
        return self.append(name, im)

    @staticmethod
    def __force_data(entry, name, data):
        # If the dataset does not exist, it is created
        # Otherwise if it is the right size, the data is replaced
        # Otherwise if the dataset can be resized, it is resized and the data is replaced
        # Otherwise the dataset is deleted and re-created
        dataset = entry.get(name)
        comp = 'gzip' if len(data) > 1250 else None
        if dataset is None:
            entry.create_dataset(name, data=data, compression=comp)
        elif dataset.shape == data.shape:
            dataset.write_direct(data)
        elif dataset.chunks is not None:
            dataset.resize(data.shape)
            dataset.write_direct(data)
        else:
            del entry[name]
            entry.create_dataset(name, data=data, compression=comp)
    
    def __delitem__(self, name):
        if isinstance(name, String): name = [name]
        for n in name:
            del self._f[n]
            del self._entries[n]

class _MAT73Entry(_MATEntry):
    # doesn't support: cell, struct, char, function_handle, <object>
    class2dtype = {
        b'int8' :int8, b'int16' :int16, b'int32' :int32, b'int64' :int64,
        b'uint8':uint8,b'uint16':uint16,b'uint32':uint32,b'uint64':uint64,
        b'single':float32,b'double':float64,
        b'logical':bool,
    }
    dtype2class = {
        int8 :b'int8', int16 :b'int16', int32 :b'int32' ,int64 :b'int64',
        uint8:b'uint8',uint16:b'uint16',uint32:b'uint32',uint64:b'uint64',
        float32:b'single',float64:b'double',complex64:b'single',complex128:b'double',
        bool:b'logical',
    }

    @classmethod
    def open(cls, entry):
        name = entry.name.lstrip('/')
        if len(name) > 1 and name[0] == b'#' and name[-1] == b'#':
            return _MAT73DummyEntry(name, entry)
        raw_dt, cls, sparse = None, entry.attrs.get('MATLAB_class'), entry.attrs.get('MATLAB_sparse')
        dt = _MAT73Entry.class2dtype.get(cls, None)
        if dt is None: return _MAT73DummyEntry(name, entry)
        is_sparse = sparse is not None
        is_empty = entry.attrs.get('MATLAB_empty', 0) == 1
        if is_sparse:
            shape = (sparse, entry['jc'].shape[0]-1)
            if 'data' in entry: raw_dt = entry['data'].dtype
        elif is_empty: shape = tuple(entry)
        else:          shape, raw_dt = entry.shape[::-1], entry.dtype
        is_complex = raw_dt is not None and raw_dt.names == ('real','imag')
        if is_complex:
            try:               dt = im_complexify_dtype(raw_dt[0])
            except ValueError: dt = dtype((raw_dt[0], 2))
        else: dt = dtype(dt)
        if raw_dt is not None: dt = dt.newbyteorder(raw_dt.byteorder)
        return _MAT73Entry(name, shape, dt, entry, is_sparse, is_empty)

    @classmethod
    def create(cls, f, name, im):
        shape, dt = im.shape, im.dtype.base.type
        is_sparse = scipy.sparse.isspmatrix(im)
        is_empty = im.size == 0

        if dt.kind == 'c':
            save_dt = im_decomplexify_dtype(dt).base
            save_dt = dtype([('real', save_dt), ('imag', save_dt)])
        else: save_dt = uint8 if dt == bool else dt

        if is_sparse:
            entry = f.create_group(name)
            if is_empty:
                entry.attrs['MATLAB_sparse'] = uint64(0)
                entry.create_dataset('jc', data=uint64(0))
            else:
                if not scipy.sparse.isspmatrix_csc: im = im.tocsc()
                entry.attrs['MATLAB_sparse'] = uint64(shape[0])
                comp = 'gzip' if shape[1] >= 1250 else None
                entry.create_dataset('jc', None, uint64, im.indptr, compression=comp)
                if im.nzz != 0:
                    comp = 'gzip' if im.nzz > 1250 else None
                    entry.create_dataset('ir', None, uint64, im.indices, compression=comp)
                    entry.create_dataset('data', None, save_dt, im.data, compression=comp)
        elif is_empty:
            entry = f.create_dataset(name, None, uint64, shape)
            entry.attrs['MATLAB_empty'] = int32(1)
        else:
            comp = 'gzip' if im.size > 1250 else None
            entry = f.create_dataset(name, shape, save_dt, im.T, compression=comp)
        entry.attrs['MATLAB_class'] = _MAT73Entry.dtype2class[dt]
        if dt == bool: entry.attrs['MATLAB_int_decode'] = int32(1)
        return _MAT73Entry(name, shape, dt, entry, is_sparse, is_empty)

    def __init__(self, name, shape, dt, entry, is_sparse, is_empty):
        self._entry = entry
        self._is_sparse = is_sparse
        self.__is_empty = is_empty
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
        if not self.__is_empty and data.shape != data.maxshape:
            h['resizeable'] = 'unlimited' if all(x is None for x in data.maxshape) else \
                              'to ('+','.join(('unlimited' if x is None else x) for x in data.maxshape)+')'
        if data.chunks is not None:
            h['chunks'] = data.chunks
        return h

    def data(self):
        entry, shape, dt = self._entry, self._shape, self._dtype
        if self._is_sparse:
            if 'data' not in entry: return scipy.sparse.csc_matrix(shape, dtype=dt)
            return scipy.sparse.csc_matrix(
                (_MAT73Entry.__read_array(entry['data'], dt),
                 _MAT73Entry.__read_array(entry['ir']),
                 _MAT73Entry.__read_array(entry['jc'])), shape=shape)
        return empty(shape, dt) if self.__is_empty else _MAT73Entry.__read_array(entry, dt, shape[::-1]).T
    @staticmethod
    def __read_array(dataset, dt=None, shape=None):
        if shape is None: shape = dataset.shape
        if dt is None:    dt    = dataset.dtype
        arr = empty(shape, dt)
        dataset.read_direct(arr)
        return arr

class _MAT73DummyEntry(_MATDummyEntry):
    def __init__(self, name, entry): self._entry = entry; super(_MAT73DummyEntry, self).__init__(name)



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
    if len(header) != 128 or any(Byte(x) == 0 for x in header[:4]): return False
    endian = header[126:128]
    if endian not in (b'IM',b'MI'): return False
    major = Byte(header[125] if endian == b'IM' else header[124])
    return major if major in (1,2) else False

_MAT_open = [_MAT4File.open, _MAT5File.open, _MAT73File.open if h5py_avail else None]
_MAT_create = {
    4: _MAT4File.create,
    6: _MAT5File.create,
    7: lambda fn,mode:_MAT5File.create(fn,mode,True),
    7.3: _MAT73File.create if h5py_avail else None
    }

class MAT(FileImageSource):
    @classmethod
    def _open(cls, filename, mode, no_vers_0=False):
        with open(filename, 'rb') as f: vers = get_mat_version(f)
        if vers is False: raise ValueError('Unknown MATLAB file version')
        if vers == 2 and not h5py_avail: raise ValueError('MATLAB v7.3 file requires h5py library')
        if vers == 0 and no_vers_0: raise ValueError("MAT v4 files do not support image stacks")
        return _MAT_open[vers](filename, mode)

    @classmethod
    def _uniq_name(cls, mat, prefix='image_'):
        #pylint: disable=protected-access
        names = set(n for n in mat._entries if n.startswith(prefix))
        i = 0
        pattern = prefix + '%03d'
        name = pattern % i
        while name in names: i += 1; name = pattern % i
        return name
    
    @classmethod
    def open(cls, filename, readonly=False, name=None, **options):
        if len(options) > 0: raise ValueError('Invalid option given')
        mat = MAT._open(filename, 'r' if readonly else 'r+')
        try:
            if name is None:
                e = next((e for e in mat if e.is_image_slice), None)
                if e is None: raise ValueError('MAT file has no usable entries')
            else:
                if _is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
                e = mat.get(name)
                if e is None: raise ValueError('MAT file has no matrix \''+name+'\'')
                if not e.is_image_slice: raise ValueError('Matrix \''+name+'\' is not an image slice')
        except: mat.close(); raise
        return MAT(mat, e, readonly, name is None)

    @classmethod
    def _openable(cls, filename, f, readonly=False, name=None, **options):
        return len(options) == 0 and (name is None or not _is_invalid_matlab_name(name)) and get_mat_version(f) is not False

    @classmethod
    def create(cls, filename, im, writeonly=False, name=None, version=None, append=False, **options):
        if len(options) > 0: raise ValueError('Invalid option given')
        if name is not None and _is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
        if _bool(append):
            if version is not None: raise ValueError('Cannot use options version and append together')
            if os.path.isfile(filename):
                mat = MAT._open(filename, 'r+')
                try: e = mat.append(MAT._uniq_name(mat), im.data) if name is None else mat.set(name, im.data)
                except: mat.close(); raise
                return MAT(mat, e, False, False)
            # else we let it open normally, file doesn't exist so no appending
        version = MAT._parse_vers(version)
        if version is False: raise ValueError('Invalid version option, must be one of 4, 6, 7, 7.3')
        if version == 7.3 and not h5py_avail: raise ValueError('MATLAB v7.3 file requires h5py library')
        mat = _MAT_create[version](filename, 'w' if writeonly else 'w+')
        try: e = mat.append(name or 'image', im.data)
        except: mat.close(); raise
        return MAT(mat, e, False, False)

    @classmethod
    def _creatable(cls, filename, ext, writeonly=False, name=None, version=None, append=False, **options):
        if len(options) > 0 or (name is not None and _is_invalid_matlab_name(name)) or ext != '.mat': return False
        if not _bool(append): return MAT._parse_vers(version) is not False
        if version is not None: return False
        if not os.path.isfile(filename): return True
        with open(filename, 'r+b') as f:
            return get_mat_version(f) is not False

    @classmethod
    def _parse_vers(cls, version):
        if version is None: return 7.3 if h5py_avail else 7
        if isinstance(version, String) and len(version) > 0 and version[0] == 'v': version = version[1:]
        try: version = float(version)
        except ValueError: return False
        return version if version in (4, 6, 7, 7.3) else False

    @classmethod
    def name(cls): return "MAT"

    @classmethod
    def exts(cls): return (".mat",)

    @classmethod
    def print_help(cls, width):
        from ....imstack import Help
        p = Help(width)
        p.title("MATLAB File Image Handler")
        p.text("""
Loads a matrix from a MATLAB file (.mat) as images. This supports loading 2D matrices and 3D
matrices with a limited length of the 3rd dimension (<5). This supports the numbers types (int#,
uint#, single, double, logical) including sparse and complex matrices. This does not support other
types such as cell arrays, structs, chars, and objects.

The following MATLAB file formats are supported given the caveats:""")
        p.list("v4 only supports uint8, uint16, int16, int32, float32, float64, and complex 2D images",
               "v6 and v7 are fully supported (with v7 being compressed)",
               "v7.3 is only supported if the h5py module is available")
        p.newline()
        p.text("""
By default this loads the first usable matrix found in the file. To specify a particular matrix, use
the 'name' option. When saving the default name is 'image'.

When saving, the default is to create a v7.3 file if possible otherwise a v7 file is created. You
can force it by setting the option 'version' to one of 4, 6, 7, or 7.3. If you specify v7.3 and h5py
is not available, saving will fail. The only difference between v6 and v7 is that v7 will create a
compressed file. Note that v7.3 is the only format that supports >2GB images, so it must be used if
you plan to save large images.

Also when saving the default is to completely overwrite any existing file. However, you can specify
the option 'append' as 'true' to force the image to be appended onto the file instead. If no name is
given, a unique name will be found and the image will be placed at the end. Otherwise this may incur
a high I/O cost because large chunks (if not the entire file) will need to be read and written to
update the matrix with the given name.
""")
        p.newline()
        p.text("See also:")
        p.list('MAT-Stack')

    def __init__(self, mat, entry, readonly, set_data_at_end):
        self._mat = mat
        self._entry = entry
        self._set_data_at_end = set_data_at_end
        self._set_props(entry.dtype, entry.shape)
        super(MAT, self).__init__(mat.filename, readonly)
    def close(self): self._mat.close()
    @property
    def header(self):
        h = self._mat.header
        h.update(self._entry.header)
        return h
    def _get_props(self): pass
    def _get_data(self): return self._entry.data
    def _set_data(self, im):
        name = MAT._uniq_name(self._mat) if self._set_data_at_end else self._entry.name
        self._entry = e = self._mat.append(name, im.data)
        self._set_props(e.dtype, e.shape)
        self._set_data_at_end = False
    def _set_filename(self, filename):
        self._mat.rename(self._rename, filename)

class MATStack(HomogeneousFileImageStack):
    #pylint: disable=protected-access
    @staticmethod
    def __parse_pattern(names):
        start = names.index('#')
        end = start + 1
        while names[end] == '#': end += 1
        return names[:start]+'%0'+str(end-start)+'d'+names[end:]
    @staticmethod
    def __get_entries(mat, pattern):
        a = pattern.index('%')
        b = pattern.index('d', a+3) - len(pattern)
        x, y = pattern[:a], pattern[len(pattern)-b]
        return sorted((e for e,n in ((e,e.name) for e in mat if e.is_image_slice)
                       if n.startswith(x) and n.endswith(y) and n[a:len(n)-b].isdigit()),
                      key=lambda e:int(e.name[a:len(e.name)-b]))

    @classmethod
    def open(cls, filename, readonly=False, name=None, names=None, mode=None, **options):
        if len(options) > 0: raise ValueError('Invalid option given')
        if mode not in (None, 'slices', 'stack'): raise ValueError('Invalid mode given, must be slices or stack')
        if None not in (name, names): raise ValueError('name and names options cannot both be given')
        if name is not None:
            if mode == 'slices': raise ValueError("If mode is 'slices', an image stack name cannot be given")
            if _is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
        elif names is not None:
            if mode == 'stack': raise ValueError("If mode is 'stack', image slices names cannot be given")
            if isinstance(names, String) and ',' in names or '#' not in names: names = names.split(',')
            pattern = None
            if isinstance(names, String):
                pattern = MATStack.__parse_pattern(names)
                if _is_invalid_matlab_name(pattern%0): raise ValueError('Invalid names for MAT file entries')
            elif any(_is_invalid_matlab_name(n) for n in names): raise ValueError('Invalid names for MAT file entries')
        mat = MAT._open(filename, 'r' if readonly else 'r+', name is not None or mode == 'stack')
        try:
            if name is not None:
                try: entry = mat[name]
                except KeyError: raise ValueError("Named matrix cannot be found")
                if not entry.is_image_stack: raise ValueError("Named matrix cannot be used as a stack")
                return MATStack(mat, entry, readonly)
            elif names is not None:
                if pattern is not None:
                    entries = MATStack.__get_entries(mat, pattern)
                else:
                    pattern = 'slice_%03d'
                    try: entries = [mat[n] for n in names]
                    except KeyError: raise ValueError("Not all named matrices could be found")
                    if not all(e.is_image_slice for e in entries): raise ValueError("Not all named matrices can be used as slices")
                return MATSlices(mat, entries, pattern, readonly)
            elif mode == 'slices' or mode != 'stack' and all(e.is_image_slice for e in mat):
                return MATSlices(mat, [e for e in mat if e.is_image_slice], 'slice_%03d', readonly)
            else:
                return MATStack(mat, next(e for e in mat if e.is_image_stack), readonly)
        except StopIteration: mat.close(); raise ValueError("MAT file has no usable stacks")
        except: mat.close(); raise
        
    @classmethod
    def _openable(cls, filename, f, readonly=False, name=None, names=None, mode=None, **options):
        if len(options) > 0 or mode not in (None, 'slices', 'stack') or None not in (name, names): return False
        if name is not None:
            if mode == 'slices' or _is_invalid_matlab_name(name): return False
        elif names is not None:
            if mode == 'stack': return False
            if isinstance(names, String) and ',' in names or '#' not in names: names = names.split(',')
            if isinstance(names, String):
                if _is_invalid_matlab_name(MATStack.__parse_pattern(names)%0): return False
            elif any(_is_invalid_matlab_name(n) for n in names): return False            
        vers = get_mat_version(f)
        return vers is not False and (vers != 0 or name is None and mode != 'stack')

    @classmethod
    def create(cls, filename, ims, writeonly=False,
               name=None, names=None, mode=None, version=None, append=False, **options):
        if len(options) > 0: raise ValueError('Invalid option given')
        if mode not in (None, 'slices', 'stack'): raise ValueError('Invalid mode given, must be slices or stack')
        if None not in (name, names): raise ValueError('name and names options cannot both be given')
        if name is not None:
            if mode == 'slices': raise ValueError("If mode is 'slices', an image stack name cannot be given")
            if _is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
            mode = 'stack'
        elif names is not None:
            if mode == 'stack': raise ValueError("If mode is 'stack', image slices names cannot be given")
            if '#' not in names: raise ValueError("Option 'names' must contain number signs for slice numebr")
            names = MATStack.__parse_pattern(names)
            if _is_invalid_matlab_name(names%0): raise ValueError('Invalid names for MAT file entries')
        if mode == 'stack':
            try:
                if isinstance(ims, ImageStack): stack = ims.stack
                else:
                    ims = list(ims)
                    stack = empty((len(ims),) + ims[0].shape, dtype=ims[0].dtype)
                    for i, im in enumerate(ims): stack[i,:,:,...] = im.data
            except: raise ValueError("If writing a single stack of images (mode=stack or using 'name') all slices must be homogeneous")
        else: names = 'slice_%03d'
        if _bool(append):
            if version is not None: raise ValueError('Cannot use options version and append together')
            if os.path.isfile(filename):
                mat = MAT._open(filename, 'r+', name is not None or mode == 'stack')
                try:
                    if mode == 'stack':
                        entry = mat.append(MAT._uniq_name(mat, 'stack_'), stack) if name is None else mat.set(name, stack)
                        return MATStack(mat, entry, False)
                    else:
                        entries = [mat.append(names%z, im.data) for z,im, in enumerate(ims, MATSlices._get_next(mat, names))]
                        return MATSlices(mat, entries, names, False)
                except: mat.close(); raise
            # else we let it open normally, file doesn't exist so no appending
        version = MAT._parse_vers(version)
        if version is False: raise ValueError('Invalid version option, must be one of 4, 6, 7, 7.3')
        if version == 7.3 and not h5py_avail: raise ValueError('MATLAB v7.3 file requires h5py library')
        mat = _MAT_create[version](filename, 'w' if writeonly else 'w+')
        try:
            if mode == 'stack':
                entry = mat.append(name or 'stack', stack)
                return MATStack(mat, entry, False)
            else:
                entries = [mat.append(names%z, im.data) for z,im in enumerate(ims)]
                return MATSlices(mat, entries, names, False)
        except: mat.close(); raise

    @classmethod
    def _creatable(cls, filename, ext, writeonly=False,
                   name=None, names=None, mode=None, version=None, append=False, **options):
        if len(options) > 0 or ext != '.mat' or mode not in (None, 'slices', 'stack') or None not in (name, names): return False
        if name is not None and (mode == 'slices' or _is_invalid_matlab_name(name)): return False
        if names is not None and (mode == 'stack' or '#' not in names or _is_invalid_matlab_name(MATStack.__parse_pattern(names)%0)): return False
        if not _bool(append): return MAT._parse_vers(version) is not False
        if version is not None: return False
        if not os.path.isfile(filename): return True
        with open(filename, 'r+b') as f:
            vers = get_mat_version(f)
            return vers is not False and (vers != 0 or name is None and mode != 'stack')

    @classmethod
    def name(cls): return "MAT-Stack"
    @classmethod
    def print_help(cls, width):
        from ....imstack import Help
        p = Help(width)
        p.title("MATLAB File Image Stack Handler")
        p.text("""
Loads a single 3D matrix or a series of 2D matrices from a MATLAB file (.mat) as an image stack.
This supports loading 3D (or 2D) matrices or 4D (or 3D) matrices with a limited length of the final
dimension (<5). This supports the numbers types (int#, uint#, single, double, logical) including
sparse and complex matrices. This does not support other types such as cell arrays, structs, chars,
and objects.

The following MATLAB file formats are supported given the caveats:""")
        p.list("v4 only supports uint8, uint16, int16, int32, float32, float64, and complex 2D images",
               "v6 and v7 are fully supported (with v7 being compressed)",
               "v7.3 is only supported if the h5py module is available")
        p.newline()
        p.text("""
When loading, by default this looks at the usuable matrices. If all are suitable 2D slices then they
are loaded each as a seperate slice. If not, the first suitable matrix as an image stack is loaded.
The behavior can be forces with the option 'mode' being set to 'slices' or 'stack'. Additionally,
if the option 'name' is given, it is treated as a 3D image stack and loaded. If the option 'names'
is given, it is a comma-seperated list of 2D slices to load or a single name with number signs in
place of indices (the count of number signs indicates leading zeros).

When saving, the default is to create a v7.3 file if possible otherwise a v7 file is created. You
can force it by setting the option 'version' to one of 4, 6, 7, or 7.3. If you specify v7.3 and h5py
is not available, saving will fail. The only difference between v6 and v7 is that v7 will create a
compressed file. Note that v7.3 is the only format that supports >2GB images, so it must be used if
you plan to save large images.

The image stack will be saved as a series of matrices, one for each image slice and named slice_000,
slice_001, and so forth. The names of the matrices can be changed with the 'names' option, where
number signs will be replaced with the slice number (and more number signs will zero-pad numbers).
If you wish to save the entire stack as a single 3D matrix, then set the option 'mode' to 'stack' or
use the option 'name' to specify its name (the default name is 'stack').

Also when saving the default is to completely overwrite any existing file. However, you can specify
the option 'append' as 'true' to force the image to be appended onto the file instead. If saving a
stack and no name is given, a unique name will be found and the image will be placed at the end.
Otherwise this may incur a high I/O cost because large chunks (if not the entire file) will need to
be read and written to update the matrix with the given name. If saving slices, unique names are
always found.
""")
        p.newline()
        p.text("See also:")
        p.list('MAT')

    def __init__(self, mat, entry, readonly=False):
        self._mat = mat
        self._entry = entry
        self._data = None
        self._data_ro = None
        self._data_slices = None
        self._data_slices_ro = None
        self._modified = False
        shape = entry.shape
        dt = dtype((entry.dtype.base, shape[3])) if len(shape) == 4 else entry.dtype
        sh = shape[1:3]
        h = mat.header
        h.update(entry.header)
        slices = [_MATSlice(self, dt, sh, z) for z in xrange(shape[0])]
        super(MATStack, self).__init__(_MATHeader(h), slices, shape[2], shape[1], dt, readonly)
    def _get_data(self):
        if self._data is None:
            self._data = self._entry.data
            self._data_ro = ImageSource.get_unwriteable_view(self._data)
            self._data_slices = list(self._data)
            self._data_slices_ro = list(self._data_ro)
    def _update_data(self):
        self._data_ro = ImageSource.get_unwriteable_view(self._data)
        self._data_slices = list(self._data)
        self._data_slices_ro = list(self._data_ro)
    def _get_slice(self, z):
        self._get_data()
        return self._data_slices_ro[z]
    def _set_slice(self, z, im):
        self._get_data()
        self._data_slices[z][...] = im
        self._modified = True
    def _delete(self, idx):
        self._get_data()
        for start,stop in idx: self._delete_slices(start, stop)
        if self._data is False:
            for start,stop in idx:
                del self._data_slices[start:stop]
                del self._data_slices_ro[start:stop]
        else:
            idx = sorted(chain.from_iterable(xrange(start,stop) for start,stop in idx))
            self._data = delete(self._data, idx, axis=0)
            self._update_data()
        self._modified = True
    def _insert(self, idx, ims):
        if any(self._shape != im.shape or self._dtype != im.dtype for im in ims): raise ValueError('The same data type and size are not homogeneous')
        self._get_data()
        self._data = self._data_ro = False
        datas = [im.data for im in ims]
        self._data_slices.insert(idx, datas)
        self._data_slices_ro.insert(idx, (ImageSource.get_unwriteable_view(d) for d in datas))
        self._modified = True
        self._insert_slices(idx, [_MATSlice(self, self._dtype, self._shape, z) for z in xrange(idx, idx+len(ims))])
    def save(self):
        super(MATStack, self).save()
        if self._modified:
            if self._data is False:
                self._data = self.stack
                self._update_data()
            self._entry = self._mat.set(self._entry.name, self._data)
            self._modified = False
    def close(self):
        if self._mat:
            self.save()
            self._mat.close()
            self._mat = None
    @property
    def stack(self):
        self._get_data()
        if self._data is False: return self._data_ro
        ims = empty(self._shape, self._dtype)
        for i,slc in enumerate(self._slices): ims[i,...] = slc
        return ims
class _MATSlice(FileImageSlice):
    #pylint: disable=protected-access
    def __init__(self, stack, dt, sh, z):
        super(_MATSlice, self).__init__(stack, z)
        self._set_props(dt, sh)
    def _get_props(self): pass
    def _get_data(self): return self._stack._get_slice(self._z)
    def _set_data(self, im):
        if self._shape != im.shape or self._dtype != im.dtype: raise ValueError('The same data type and size are not homogeneous')
        im = im.data
        self._stack._set_slice(self._z, im)
        return im

class MATSlices(FileImageStack):
    #pylint: disable=protected-access
    def __init__(self, mat, entries, pattern='slice_%03d', readonly=False):
        self._mat = mat
        self._pattern = pattern
        self._next = MATSlices._get_next(mat, pattern)
        slices = [_MATEntrySlice(self, entry, z) for z,entry in enumerate(entries)]
        super(MATSlices, self).__init__(_MATHeader(mat.header), slices, readonly)
    @staticmethod
    def _get_next(mat, pattern):
        a = pattern.index('%')
        b = pattern.index('d', a+3) - len(pattern) + 1
        x, y = pattern[:a], pattern[len(pattern)-b:]
        return max([int(n[a:len(n)-b]) for n in (e.name for e in mat._entries.itervalues())
                    if n is not None and n.startswith(x) and n.endswith(y) and n[a:len(n)-b].isdigit()] + [0])
    def close(self):
        if self._mat: self._mat.close(); self._mat = None
    def _delete(self, idx_ranges):
        idx = chain.from_iterable(xrange(start,stop) for start,stop in idx_ranges)
        del self._mat[[self._slices[i]._entry.name for i in idx]]
        for start,stop in idx: self._delete_slices(start, stop)
    def _insert(self, idx, ims):
        if idx != self._d: raise RuntimeError('Inserting slices not supported for a set of slices from a MAT file (must append)')
        n = len(ims)
        names = [(self._pattern%z) for z in xrange(self._next, self._next+n)]
        self._next += n
        entries = [self._mat.append(name, im.data) for name,im in izip(names,ims)]
        self._insert_slices(idx, [_MATEntrySlice(self, e, z) for z,e in enumerate(entries, idx)])
class _MATEntrySlice(FileImageSlice):
    def __init__(self, stack, entry, z):
        super(_MATEntrySlice, self).__init__(stack, z)
        self._set_entry(entry)
    @property
    def header(self): return self._entry.header
    def _set_entry(self, entry):
        shape = entry.shape
        dt = dtype((entry.dtype.base, shape[2])) if len(shape) == 3 else entry.dtype
        self._set_props(dt, shape[:2])
        self._entry = entry
    def _get_props(self): pass
    def _get_data(self): return self._entry.data
    def _set_data(self, im):
        im = im.data
        self._set_entry(self._stack._mat.set(self._entry.name, im)) #pylint: disable=protected-access
        return im

class _MATHeader(FileImageStackHeader):
    _fields = None
    def __init__(self, data, **options):
        if len(options): data['options'] = options
        self._fields = {k:FixedField(lambda x:x,v,False) for k,v in data.iteritems()}
        super(_MATHeader, self).__init__(data)
    def save(self):
        if self._imstack._readonly: raise AttributeError('header is readonly') #pylint: disable=protected-access
    def _update_depth(self, d): pass
    def _get_field_name(self, f):
        return f if f in self._fields else None
