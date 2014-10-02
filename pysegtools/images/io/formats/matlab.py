from numpy import complex64, complex128

from .._single import iminfo, imread
from ...types import *

# TODO: doesn't handle endian-ness, unsure if anything needs to be done
# TODO: support imsave and stacks

# NOTE: to use .MAT v7 or older files (from R2006a and earlier) you must have SciPy v0.12 or newer 
# NOTE: to use .MAT v7.3 of newer files (from R2006b and later) you must have h5py

__all__ = ['iminfo_mat','imread_mat']

# don't support: cell, struct, object, char, function, opaque, unknown
mat_classes = {
    'int8' :IM_INT8, 'int16' :IM_INT16, 'int32' :IM_INT32, 'int64' :IM_INT64,
    'uint8':IM_UINT8,'uint16':IM_UINT16,'uint32':IM_UINT32,'uint64':IM_UINT64,
    'single':IM_FLOAT32,'double':IM_FLOAT64,'sparse':IM_FLOAT64,'logical':IM_BIT,
    }

def find_non_special(d, special = '__'):
    name = special
    for name,value in d.iteritems():
        if not name.startswith(special): return value
    else: raise KeyError() # no non-special variables

def get_target_dtype(dtype, cls):
    return (dtype(complex128) if dtype.itemsize == 16 else dtype(complex64)) if dtype != None and dtype.names == ('real','imag') else mat_classes.get(cls, None)

def get_mat_entry_sp(filename, name = None, return_data = False):
    from scipy.io import whosmat
    keys = whosmat(file_name)
    if len(keys) == 0: raise KeyError()
    if name == None:
        name, shape, cls = find_non_special({x[0]:x for x in keys}, '__')
    else:
        for n, shape, cls in keys:
            if n == name: break
        else: raise KeyError()
    if return_data:
        from scipy.io import loadmat
        data = loadmat(filename, variable_names = name, mat_dtype = True)[name]
        if cls == 'logical' and data.dtype == IM_FLOAT64: data = data.astype(IM_BIT)
        return (name, cls, data)
    else:
        return (name, cls, shape, get_target_dtype(None, cls))

def get_mat_entry_h5py(f, name = None, return_data = False):
    from h5py import File as HDF5File
    with HDF5File(filename, 'r') as f: # IOError if it doesn't exist or is the wrong format
        x = find_non_special(f, special='#') if name == None else f[name]
        name, cls, sparse = x.name.lstrip('/'), x.attrs['MATLAB_class'], x.attrs.get('MATLAB_sparse', None)
        if sparse != None:
            data = x.get('data', None)
            shape = (sparse, x['jc'].shape[0] - 1)
            dtype = get_target_dtype(data.dtype if data != None else None, cls)
            if return_data:
                from scipy.sparse import csc_matrix
                if data != None:
                    data = csc_matrix((x.value.view(dtype=dtype) if dtype != None else x.value, x['ir'].value, x['jc'].value), shape=shape)
                else: # all values are zeros
                    data = csc_matrix(shape=shape, dtype=dtype)
        elif x.attrs.get('MATLAB_empty', 0) == 1:
            from numpy import empty
            shape = tuple(x.value)
            dtype = get_target_dtype(None, cls)
            if return_data: data = empty(shape, dtype=dtype)
        else:
            shape = x.shape[::-1]
            dtype = get_target_dtype(data.dtype, cls)
            if return_data: data = x.value.T.view(dtype=dtype) if dtype != None else x.value.T
    return (name, cls, data) if return_data else (name, cls, shape, dtype)

def iminfo_mat(filename, name = None):
    """
    Read the info for an 'image' from a MATLAB .MAT file. The file can be any
    version. Files that are v7.3 require the h5py module. If no name is given,
    the first variable is taken.
    """
    try:
        # Try SciPy built-in first (doesn't work for v7.3+ files)
        name, cls, shape, dtype = get_mat_entry_sp(filename, name, False) # TODO: does not detect complex matricies (limitation of whosmat)
    except NotImplementedError:
        # Try v7.3 file which is an HDF5 file, we have to use h5py for this
        name, cls, shape, dtype = get_mat_entry_h5py(filename, name, False)
    if len(shape) == 3 and dtype == IM_UINT8:
        if   shape[2] == 4: return shape[:2], IM_RGBA32
        elif shape[2] == 3: return shape[:2], IM_RGB24
        else: raise ValueError('MAT file matrix has unsupported shape or data type for images')
    elif len(shape) == 2 and cls in mat_classes: return shape, dtype
    else: raise ValueError('MAT file matrix has unsupported shape or data type for images')
iminfo.register('.mat', iminfo_mat)

def imread_mat(filename, name = None):
    """
    Read an 'image' from a MATLAB .MAT file. The file can be any version. Files
    that are v7.3 require the h5py module. If no name is given, the first
    variable is taken.
    """
    try:
        # Try SciPy built-in first (doesn't work for v7.3+ files)
        # Supports loading just the given variable name
        # Otherwise have to load all variables and skip special keys starting with "__" to find the variable to load
        # Loaded matrices are already arrays
        name, cls, im = get_mat_entry_sp(filename, name, True)

    except NotImplementedError:
        # Try v7.3 file which is an HDF5 file
        # We have to use h5py for this (or PyTables...)
        # Always loads entire metadata (not just specific variable) but none of the data
        # Data needs to be actually loaded (.value) and transposed (.T)
        name, cls, im = get_mat_entry_h5py(filename, name, True)

    if len(im.shape) == 3 and im.dtype == IM_UINT8:
        if   im.shape[2] == 4: return im.view(dtype=IM_RGBA32)
        elif im.shape[2] == 3: return im.view(dtype=IM_RGB24)
        else: raise ValueError('MAT file matrix has unsupported shape or data type for images')
    elif len(im.shape) == 2 and cls in mat_classes: return im
    else: raise ValueError('MAT file matrix has unsupported shape or data type for images')
imread.register('.mat', imread_mat)
