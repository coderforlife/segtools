from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, io
from itertools import izip, chain

from numpy import dtype, empty, delete

from .._single import FileImageSource
from .._stack import FileImageStack, HomogeneousFileImageStack, FileImageSlice, FileImageStackHeader, FixedField
from ...source import ImageSource
from ..._stack import ImageStack
from ...types import is_image_desc
from ....general import String, _bool
from ....general.matlab import openmat, get_mat_version, is_invalid_matlab_name

__all__ = ['MAT', 'MATStack']

def _is_image_slice(entry):
    try: return is_image_desc(entry.dtype, entry.shape)
    except StandardError: return False

def _is_image_stack(entry):
    try: return is_image_desc(entry.dtype, entry.shape[1:])
    except StandardError: return False

class MAT(FileImageSource):
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
    def open(cls, filename, readonly=False, name=None, **options): #pylint: disable=arguments-differ
        if len(options) > 0: raise ValueError('Invalid option given')
        mat = openmat(filename, 'r' if readonly else 'r+')
        try:
            if name is None:
                e = next((e for e in mat if _is_image_slice(e)), None)
                if e is None: raise ValueError('MAT file has no usable entries')
            else:
                if is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
                e = mat.get(name)
                if e is None: raise ValueError('MAT file has no matrix \''+name+'\'')
                if not _is_image_slice(e): raise ValueError('Matrix \''+name+'\' is not an image slice')
        except: mat.close(); raise
        return MAT(mat, e, readonly, name is None)

    @classmethod
    def _openable(cls, filename, f, readonly=False, name=None, **options):#pylint: disable=arguments-differ
        return len(options) == 0 and (name is None or not is_invalid_matlab_name(name)) and get_mat_version(f) is not False

    @classmethod
    def create(cls, filename, im, writeonly=False, name=None, version=None, append=False, **options):#pylint: disable=arguments-differ
        if len(options) > 0: raise ValueError('Invalid option given')
        if name is not None and is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
        if _bool(append):
            if version is not None: raise ValueError('Cannot use options version and append together')
            if os.path.isfile(filename):
                mat = openmat(filename, 'r+')
                try: e = mat.append(MAT._uniq_name(mat), im.data) if name is None else mat.set(name, im.data)
                except: mat.close(); raise
                return MAT(mat, e, False, False)
            # else we let it open normally, file doesn't exist so no appending
        mat = openmat(filename, 'w' if writeonly else 'w+', version)
        try: e = mat.append(name or 'image', im.data)
        except: mat.close(); raise
        return MAT(mat, e, False, False)

    @classmethod
    def _creatable(cls, filename, ext, writeonly=False, name=None, version=None, append=False, **options):#pylint: disable=arguments-differ
        if len(options) > 0 or (name is not None and is_invalid_matlab_name(name)) or ext != '.mat': return False
        if not _bool(append): return MAT._parse_vers(version) is not False
        if version is not None: return False
        if not os.path.isfile(filename): return True
        with io.open(filename, 'r+b') as f:
            return get_mat_version(f) is not False

    @classmethod
    def _parse_vers(cls, version):
        if version is None: return 7 # 7.3 if h5py_avail else 7 - not needed here
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
    #pylint: disable=protected-access, too-many-arguments
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
        return sorted((e for e,n in ((e,e.name) for e in mat if _is_image_slice(e))
                       if n.startswith(x) and n.endswith(y) and n[a:len(n)-b].isdigit()),
                      key=lambda e:int(e.name[a:len(e.name)-b]))

    @classmethod
    def open(cls, filename, readonly=False, name=None, names=None, mode=None, **options): #pylint: disable=arguments-differ
        if len(options) > 0: raise ValueError('Invalid option given')
        if mode not in (None, 'slices', 'stack'): raise ValueError('Invalid mode given, must be slices or stack')
        if None not in (name, names): raise ValueError('name and names options cannot both be given')
        if name is not None:
            if mode == 'slices': raise ValueError("If mode is 'slices', an image stack name cannot be given")
            if is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
        elif names is not None:
            if mode == 'stack': raise ValueError("If mode is 'stack', image slices names cannot be given")
            if isinstance(names, String) and ',' in names or '#' not in names: names = names.split(',')
            pattern = None
            if isinstance(names, String):
                pattern = MATStack.__parse_pattern(names)
                if is_invalid_matlab_name(pattern%0): raise ValueError('Invalid names for MAT file entries')
            elif any(is_invalid_matlab_name(n) for n in names): raise ValueError('Invalid names for MAT file entries')
        mat = openmat(filename, 'r' if readonly else 'r+')
        if (name is not None or mode == 'stack') and mat.header['version'] == 4:
            raise ValueError("MAT v4 files do not support image stacks")
        try:
            if name is not None:
                try: entry = mat[name]
                except KeyError: raise ValueError("Named matrix cannot be found")
                if not _is_image_stack(entry): raise ValueError("Named matrix cannot be used as a stack")
                return MATStack(mat, entry, readonly)
            elif names is not None:
                if pattern is not None:
                    entries = MATStack.__get_entries(mat, pattern)
                else:
                    pattern = 'slice_%03d'
                    try: entries = [mat[n] for n in names]
                    except KeyError: raise ValueError("Not all named matrices could be found")
                    if not all(_is_image_slice(e) for e in entries): raise ValueError("Not all named matrices can be used as slices")
                return MATSlices(mat, entries, pattern, readonly)
            elif mode == 'slices' or mode != 'stack' and all(_is_image_slice(e) for e in mat):
                return MATSlices(mat, [e for e in mat if _is_image_slice(e)], 'slice_%03d', readonly)
            else:
                return MATStack(mat, next(e for e in mat if _is_image_stack(e)), readonly)
        except StopIteration: mat.close(); raise ValueError("MAT file has no usable stacks")
        except: mat.close(); raise
        
    @classmethod
    def _openable(cls, filename, f, readonly=False, name=None, names=None, mode=None, **options): #pylint: disable=arguments-differ
        if len(options) > 0 or mode not in (None, 'slices', 'stack') or None not in (name, names): return False
        if name is not None:
            if mode == 'slices' or is_invalid_matlab_name(name): return False
        elif names is not None:
            if mode == 'stack': return False
            if isinstance(names, String) and ',' in names or '#' not in names: names = names.split(',')
            if isinstance(names, String):
                if is_invalid_matlab_name(MATStack.__parse_pattern(names)%0): return False
            elif any(is_invalid_matlab_name(n) for n in names): return False            
        vers = get_mat_version(f)
        return vers is not False and (vers != 0 or name is None and mode != 'stack')

    @classmethod
    def create(cls, filename, ims, writeonly=False, #pylint: disable=arguments-differ
               name=None, names=None, mode=None, version=None, append=False, **options):
        if len(options) > 0: raise ValueError('Invalid option given')
        if mode not in (None, 'slices', 'stack'): raise ValueError('Invalid mode given, must be slices or stack')
        if None not in (name, names): raise ValueError('name and names options cannot both be given')
        if name is not None:
            if mode == 'slices': raise ValueError("If mode is 'slices', an image stack name cannot be given")
            if is_invalid_matlab_name(name): raise ValueError('Invalid name for MAT file entries')
            mode = 'stack'
        elif names is not None:
            if mode == 'stack': raise ValueError("If mode is 'stack', image slices names cannot be given")
            if '#' not in names: raise ValueError("Option 'names' must contain number signs for slice numebr")
            names = MATStack.__parse_pattern(names)
            if is_invalid_matlab_name(names%0): raise ValueError('Invalid names for MAT file entries')
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
                mat = openmat(filename, 'r+')
                if (name is not None or mode == 'stack') and mat.header['version'] == 4:
                    raise ValueError("MAT v4 files do not support image stacks")
                try:
                    if mode == 'stack':
                        entry = mat.append(MAT._uniq_name(mat, 'stack_'), stack) if name is None else mat.set(name, stack)
                        return MATStack(mat, entry, False)
                    entries = [mat.append(names%z, im.data) for z,im, in enumerate(ims, MATSlices._get_next(mat, names))]
                    return MATSlices(mat, entries, names, False)
                except: mat.close(); raise
            # else we let it open normally, file doesn't exist so no appending
        mat = openmat(filename, 'w' if writeonly else 'w+', version)
        try:
            if mode == 'stack':
                entry = mat.append(name or 'stack', stack)
                return MATStack(mat, entry, False)
            entries = [mat.append(names%z, im.data) for z,im in enumerate(ims)]
            return MATSlices(mat, entries, names, False)
        except: mat.close(); raise

    @classmethod
    def _creatable(cls, filename, ext, writeonly=False, #pylint: disable=arguments-differ
                   name=None, names=None, mode=None, version=None, append=False, **options):
        if len(options) > 0 or ext != '.mat' or mode not in (None, 'slices', 'stack') or None not in (name, names): return False
        if name is not None and (mode == 'slices' or is_invalid_matlab_name(name)): return False
        if names is not None and (mode == 'stack' or '#' not in names or is_invalid_matlab_name(MATStack.__parse_pattern(names)%0)): return False
        if not _bool(append): return MAT._parse_vers(version) is not False
        if version is not None: return False
        if not os.path.isfile(filename): return True
        with io.open(filename, 'r+b') as f:
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
    @property
    def filenames(self): return (self._mat.filename,)
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
    def _delete(self, idxs):
        self._get_data()
        for start,stop in idxs: self._delete_slices(start, stop)
        if self._data is False:
            for start,stop in idxs:
                del self._data_slices[start:stop]
                del self._data_slices_ro[start:stop]
        else:
            idxs = sorted(chain.from_iterable(xrange(start,stop) for start,stop in idxs))
            self._data = delete(self._data, idxs, axis=0)
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
    @property
    def filenames(self): return (self._mat.filename,)
    def _delete(self, idxs):
        idxs = chain.from_iterable(xrange(start,stop) for start,stop in idxs)
        del self._mat[[self._slices[i]._entry.name for i in idxs]]
        for start,stop in idxs: self._delete_slices(start, stop)
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
