# Helpful Python, Cython, and Numpy functions and general initialization code
# Provides:
#     npy_array_resize1D, npy_array_resize2D
#     npy_view_slice
#     npy_view_single1D, npy_view_single2D
#     npy_view_trim1D, npy_view_trim2D
#     npy_view_shortened1D, npy_view_shortened2D
#     npy_view_col
#     npy_squeeze_last
#     npy_ravel_rows
#     npy_check, npy_check1D, npy_check2D
#     npy_lexsort2D
# Along with cimporting all of npy_helper and running import_array()

from npy_helper cimport *

### Required to use the Numpy C-API ###
import_array()


########## Array Utility Functions ##########
cdef inline void npy_array_resize1D(ndarray a, intp size):
    """
    Shortens/resizes an array by modifying the internal data. The array must own its data and have
    a minimal number of references.
    """
    cdef PyArray_Dims dims
    dims.ptr = &size; dims.len = 1
    PyArray_Resize(a, &dims, True, NPY_CORDER)

cdef inline void npy_array_resize2D(ndarray a, intp nrows):
    """
    Shortens/resizes an array by modifying the internal data. The array must own its data and have
    a minimal number of references.
    """
    cdef intp d[2]
    d[0] = nrows; d[1] = PyArray_DIM(a, PyArray_NDIM(a)-1)
    cdef PyArray_Dims dims
    dims.ptr = d; dims.len = 2
    PyArray_Resize(a, &dims, True, NPY_CORDER)

cdef inline ndarray npy_view_slice(ndarray a, int ndim, intp* dims, intp* strides, void* data, int flags):
    """
    Creates a new view of an array (a) with the given dimensions (ndim/dims), strides, underlying
    data, and flags.
    """
    cdef PyArray_Descr* dt = PyArray_DESCR(a)
    Py_INCREF(<PyObject*>dt)
    cdef ndarray b = PyArray_NewFromDescr(&PyArray_Type, dt, ndim, dims, strides, data, flags, NULL)
    Py_INCREF(a)
    PyArray_SetBaseObject(b, a)
    return b

cdef inline ndarray npy_view_single1D(ndarray a, intp i):
    """View the element at the index from the array, equivalent to a[i]"""
    cdef intp size = 1
    return npy_view_slice(a, 1, &size, PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,0)*i), NPY_ARRAY_CARRAY)

cdef inline ndarray npy_view_single2D(ndarray a, intp i):
    """View the row at the index from the array, equivalent to a[i,:]"""
    cdef intp dims[2]
    dims[0] = 1; dims[1] = PyArray_DIM(a,1)
    return npy_view_slice(a, 2, dims, PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,0)*i), NPY_ARRAY_CARRAY)

cdef inline ndarray npy_view_trim1D(ndarray a, intp start, intp end):
    """
    View the elements from the array after removing start elements from the beginning and end elements
    from the end, equivalent to a[start:-end] (except end can be 0).
    """
    cdef intp size = PyArray_DIM(a,0)-start-end
    return npy_view_slice(a, 1, &size, PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,0)*start), NPY_ARRAY_CARRAY)

cdef inline ndarray npy_view_trim2D(ndarray a, intp start_row, intp end_row, intp start_col=0, intp end_col=0):
    """
    View the rows from the array after removing start rows from the beginning and end rows from the
    end, equivalent to a[start_row:-end_row,start_col:-end_col] (except end_row and end_col can be 0).
    """
    cdef intp dims[2]
    dims[0] = PyArray_DIM(a,0)-start_row-end_row; dims[1] = PyArray_DIM(a,1)-start_col-end_col
    cdef intp off = PyArray_STRIDE(a,0)*start_row+PyArray_STRIDE(a,1)*start_col
    return npy_view_slice(a, 2, dims, PyArray_STRIDES(a), PyArray_BYTES(a)+off, NPY_ARRAY_CARRAY)

#cdef inline ndarray npy_view_flipud2D(ndarray a):
#    cdef intp strides[2]
#    strides[0] = -PyArray_STRIDE(a,0)
#    strides[1] = PyArray_STRIDE(a,1)
#    return npy_view_slice(a, PyArray_DATA(a), PyArray_SHAPE(a), strides, NPY_ARRAY_CARRAY)

cdef inline ndarray npy_view_shortened1D(ndarray a, intp size):
    """View the first size elements from the array, equivalent to a[:size]."""
    return npy_view_slice(a, 1, &size, PyArray_STRIDES(a), PyArray_DATA(a), NPY_ARRAY_CARRAY)

cdef inline ndarray npy_view_shortened2D(ndarray a, intp nrows):
    """View the first size rows from the array, equivalent to a[:size,:]."""
    cdef intp dims[2]
    dims[0] = nrows; dims[1] = PyArray_DIM(a,1)
    return npy_view_slice(a, 2, dims, PyArray_STRIDES(a), PyArray_DATA(a), NPY_ARRAY_CARRAY)
    
cdef inline ndarray npy_view_col(ndarray a, intp i):
    """View the a column from an array, equivalent to a[:,i]."""
    return npy_view_slice(a, 1, PyArray_SHAPE(a), PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,1)*i), NPY_ARRAY_BEHAVED)

cdef inline ndarray npy_squeeze_last(ndarray a):
    """Squeeze the last dimension if it is singleton. Equivalent to a.squeeze(-1) if a.shape[-1] == 1 else a."""
    if PyArray_DIM(a, PyArray_NDIM(a)-1) != 1: return a
    return npy_view_slice(a, PyArray_NDIM(a)-1, PyArray_SHAPE(a), PyArray_STRIDES(a), PyArray_BYTES(a), NPY_ARRAY_CARRAY)

cdef inline ndarray npy_ravel_rows(ndarray a):
    """
    Like ndarray.ravel but keeps the rows un-raveled. The result is always 2D.
    """
    assert PyArray_NDIM(a) >= 2
    if PyArray_NDIM(a) == 2: return a
    cdef intp d[2]
    d[0] = PyArray_MultiplyList(PyArray_SHAPE(a), PyArray_NDIM(a)-1)
    d[1] = PyArray_DIM(a, PyArray_NDIM(a)-1)
    cdef PyArray_Dims dims
    dims.ptr = d; dims.len = 2
    return PyArray_Newshape(a, &dims, NPY_CORDER)

cdef int npy_chk_flags = NPY_ARRAY_CARRAY|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES
cdef inline ndarray npy_check(object arr, int min_d):
    """
    Checks an object, converting it to an array if necessary and making sure that it has at least the
    given number of dimensions, is C-contiguous, not byte-swapped, has element-sized strides.
    """
    return PyArray_CheckFromAny(arr, NULL, min_d, 0, npy_chk_flags, NULL)

cdef inline ndarray npy_check1D(object arr):
    """Like npy_check with min_d=1 and the output array is raveled (so always 1D)."""
    cdef ndarray a = PyArray_CheckFromAny(arr, NULL, 1, 0, npy_chk_flags, NULL)
    return a if PyArray_NDIM(a)==1 else PyArray_Ravel(a, NPY_CORDER) # could use PyArray_Newshape as well (that is what ravel uses internally)

cdef inline ndarray npy_check2D(object arr):
    """Like npy_check with min_d=2 and the output array is row-raveled (so always 2D)."""
    return npy_ravel_rows(PyArray_CheckFromAny(arr, NULL, 2, 0, npy_chk_flags, NULL))

cdef inline ndarray npy_lexsort2D(ndarray a):
    """
    Calls lexsort on a 2D array however the columns are the keys and they are kept in increasing
    priority and the actual transformed array is returned instead of just the indices.
    """
    cdef intp n = PyArray_DIM(a,1), i
    cdef list l = PyList_New(n)
    cdef ndarray c
    for i in range(n): c = npy_view_col(a, i); Py_INCREF(to_c(c)); PyList_SET_ITEM(l, n-i-1, c)
    return PyArray_TakeFrom(a, PyArray_LexSort(l, 0), 0, NULL, NPY_RAISE)
