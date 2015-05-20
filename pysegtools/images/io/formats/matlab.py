from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numpy import bool_, int8,int16,int32,int64, uint8,uint16,uint32,uint64, float32,float64, complex64,complex128

from .._single import iminfo, imread
from ...types import is_image, create_im_dtype

# TODO: doesn't handle endian-ness, unsure if anything needs to be done
# TODO: support imsave and stacks

# NOTE: to use .MAT v7 or older files (from R2006a and earlier) you must have SciPy v0.12 or newer
# NOTE: to use .MAT v7.3 of newer files (from R2006b and later) you must have h5py

__all__ = ['iminfo_mat','imread_mat']

# don't support: cell, struct, object, char, function, opaque, unknown
mat_classes = {
    'int8' :int8, 'int16' :int16, 'int32' :int32, 'int64' :int64,
    'uint8':uint8,'uint16':uint16,'uint32':uint32,'uint64':uint64,
    'single':float32,'double':float64,'sparse':float64,'logical':bool_,
}

def find_non_special(d, special='__'):
    for name,value in d.iteritems():
        if not name.startswith(special): return value
    raise KeyError() # no non-special variables

def get_target_dtype(dtype, cls):
    return dtype(complex128 if dtype.itemsize == 16 else complex64) if dtype is not None and dtype.names == ('real','imag') else mat_classes.get(cls, None)

def get_mat_entry_sp(filename, name=None, return_data=False):
    from scipy.io import whosmat, loadmat
    keys = whosmat(filename)
    if len(keys) == 0: raise KeyError()
    if name is None:
        name, shape, cls = find_non_special({x[0]:x for x in keys}, '__')
    else:
        for n, shape, cls in keys:
            if n == name: break
        else: raise KeyError()
    if return_data:
        data = loadmat(filename, variable_names=name, mat_dtype=True)[name]
        if cls == 'logical' and data.dtype.type == float64: data = data.astype(bool_)
        return (name, data)
    else:
        return (name, (shape, get_target_dtype(None, cls)))

def get_mat_entry_h5py(filename, name=None, return_data=False):
    # TODO: x.value is deprecated as of h5py v2.1
    from h5py import File as HDF5File
    with HDF5File(filename, 'r') as f: # IOError if it doesn't exist or is the wrong format
        x = find_non_special(f, special='#') if name is None else f[name]
        name, cls, sparse = x.name.lstrip('/'), x.attrs['MATLAB_class'], x.attrs.get('MATLAB_sparse')
        data = x.get('data')
        if sparse is not None:
            shape = (sparse, x['jc'].shape[0] - 1)
            dtype = get_target_dtype(data.dtype if data is not None else None, cls)
            if return_data:
                from scipy.sparse import csc_matrix
                if data is not None:
                    data = csc_matrix((x.value.view(dtype=dtype) if dtype is not None else x.value, x['ir'].value, x['jc'].value), shape=shape)
                else: # all values are zeros
                    data = csc_matrix(shape, dtype=dtype)
        elif x.attrs.get('MATLAB_empty', 0) == 1:
            from numpy import empty
            shape = tuple(x.value)
            dtype = get_target_dtype(None, cls)
            if return_data: data = empty(shape, dtype=dtype)
        else:
            shape = x.shape[::-1]
            dtype = get_target_dtype(data.dtype, cls)
            if return_data: data = x.value.T.view(dtype=dtype) if dtype is not None else x.value.T
    return (name, data) if return_data else (name, (shape, dtype))

def iminfo_mat(filename, name=None):
    """
    Read the info for an 'image' from a MATLAB .MAT file. The file can be any
    version. Files that are v7.3 require the h5py module. If no name is given,
    the first variable is taken.
    """
    try:
        # Try SciPy built-in first (doesn't work for v7.3+ files)
        name, (shape, dtype) = get_mat_entry_sp(filename, name, False) # TODO: does not detect complex matricies (limitation of whosmat)
    except NotImplementedError:
        # Try v7.3 file which is an HDF5 file, we have to use h5py for this
        try:
            name, (shape, dtype) = get_mat_entry_h5py(filename, name, False)
        except ImportError:
            import sys
            print("h5py library not found which is required to read MATLAB files v7.3+", file=sys.stderr)
            raise
    shape = shape[:2] + [x for x in shape[2:] if x != 1] # squeeze high dimensions
    if dtype is None or len(shape) not in (2,3) or len(shape) == 3 and dtype.kind == 'c': raise ValueError('MAT file matrix has unsupported shape or data type for images')
    return tuple(shape[:2]), create_im_dtype(dtype, '=', shape[2] if len(shape) == 3 else 1)
iminfo.register('.mat', iminfo_mat)

def imread_mat(filename, name=None):
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
        name, im = get_mat_entry_sp(filename, name, True)

    except NotImplementedError:
        # Try v7.3 file which is an HDF5 file
        # We have to use h5py for this (or PyTables...)
        # Always loads entire metadata (not just specific variable) but none of the data
        # Data needs to be actually loaded (.value) and transposed (.T)
        name, im = get_mat_entry_h5py(filename, name, True)

    if not is_image(im): raise ValueError('MAT file matrix has unsupported shape or data type for images')
    return im
imread.register('.mat', imread_mat)
