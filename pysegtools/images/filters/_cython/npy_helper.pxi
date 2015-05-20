# Helpful Python, Cython, and Numpy functions and general initialization code

DEF ADD_FUSED_TYPE_TO_FUNCTION=False

cimport cython
from cython cimport sizeof
import sys
import numpy

from npy_helper cimport *


############### Fused Function Helpers ###############
# This replaces the default getitem function (one used for []) for Cython fused functions with one
# with a bit more flexibility. To make it useful you need to use "register_fused_type" after
# creating a new fused type and the "fused" decorator on fused functions.
cdef binaryfunc __get_fused_orig = __pyx_FusedFunctionType.tp_as_mapping.mp_subscript
cdef PyObject* __get_fused(__pyx_FusedFunctionObject* self, PyObject* idx):
    if self.__signatures__ == NULL:
        PyErr_SetString(PyExc_TypeError, "Function is not fused")
        return NULL
    cdef PyObject* unbound = PyDict_GetItem(self.__signatures__, idx)
    if unbound == NULL:
        unbound = PyObject_GetAttrString(<PyObject*>self, "__fallback")
        if unbound == NULL or unbound == Py_None:
            PyErr_Clear()
            return __get_fused_orig(<PyObject*>self, idx)
    else: Py_INCREF(unbound)
    if self.self == NULL and self.type == NULL:
        return unbound
    cdef __pyx_FusedFunctionObject* unbound_ff = <__pyx_FusedFunctionObject*>unbound
    Py_CLEAR(unbound_ff.func.func_classobj)
    Py_XINCREF(self.func.func_classobj)
    unbound_ff.func.func_classobj = self.func.func_classobj
    cdef PyObject* f = __pyx_FusedFunctionType.tp_descr_get(unbound, self.self, self.type)
    Py_DECREF(unbound)
    return f
__pyx_FusedFunctionType.tp_as_mapping.mp_subscript = <binaryfunc>__get_fused
__py2 = sys.version_info[0] == 2
cdef dict __fused_types = {}
cdef inline void register_fused_type(str fused_type_name, dict types):
    """
    Registers the fused type named "fused_type_name" and has the types that are the values in the
    dictionary "types". The keys of "types" are values you want to use to lookup the fused function
    with. For example, an integer (for Numpy which gives a number to each type), a type object or a
    string.
    """
    IF ADD_FUSED_TYPE_TO_FUNCTION:
        __fused_types[frozenset(types.itervalues() if __py2 else types.values())] = (fused_type_name,types)
    ELSE:
        __fused_types[frozenset(types.itervalues() if __py2 else types.values())] = types
def fused(f=None, fallback=None): 
    """
    Operates on a fused function to make it more accessible. If the fused type was not registered
    with register_fused_type this will fail. It adds the lookups registered. If the function has
    multiple fused types then a tuple of those lookups is used (for example fused_func[2,3]). It
    also adds a "__fused_type" attribute to the function which is the name of the fused type.
    Finally, you may specify a fallback function to return in case the lookup fails. If provided
    it prevents you for using the default string lookups to get the function.
    """
    if f is None:
        from functools import partial
        return partial(fused, fallback=fallback)
    cdef dict sigs = f.__signatures__
    cdef str s
    IF ADD_FUSED_TYPE_TO_FUNCTION:
        for k,v in sigs.iteritems() if __py2 else sigs.items(): v.__fused_type = k
        if '|' in next(iter(sigs)):
            from itertools import izip, product
            fts = [__fused_types[fs] for fs in (frozenset(s) for s in izip(*[s.split('|') for s in sigs]))]
            f.__fused_type = '|'.join(ft[0] for ft in fts)
            fts = list(product(*[(ft[1].iteritems() if __py2 else ft[1].items()) for ft in fts]))
            itr = izip((tuple(k for k,_ in ft) for ft in fts), ('|'.join(v for _,v in ft) for ft in fts))
        else:
            f.__fused_type, ids = __fused_types[frozenset(sigs)]
            itr = ids.iteritems() if __py2 else ids.items()
    ELSE:
        if '|' in next(iter(sigs)):
            from itertools import izip, product
            fts = list(product(*[
                    __fused_types[fs].iteritems() if __py2 else __fused_types[fs].items() for fs in
                    (frozenset(s) for s in izip(*[s.split('|') for s in sigs]))
                ]))
            itr = izip((tuple(k for k,_ in ft) for ft in fts), ('|'.join(v for _,v in ft) for ft in fts))
        else:
            itr = __fused_types[frozenset(sigs)].iteritems() if __py2 else __fused_types[frozenset(sigs)].items()
    sigs.update({k:sigs[v] for k,v in itr})
    f.__fallback = fallback
    return f

    
########## Initialization ##########

### Required to use the Numpy C-API ###
import_array()

### Register all of the Numpy fused types defined above ###
cdef void init_fused_types():
    def __merge_dict(*args):
        cdef dict out = {}
        for d in args: out.update(d)
        return out
    cdef dict __sint = {1:'npy_byte',3:'npy_short',5:'npy_int',7:'npy_long',9:'npy_longlong'}
    cdef dict __uint = {2:'npy_ubyte',4:'npy_ushort',6:'npy_uint',8:'npy_ulong',10:'npy_ulonglong'}
    cdef dict __flt  = {23:'npy_half',11:'npy_float',12:'npy_double',13:'npy_longdouble'}
    cdef dict __cflt = {14:'npy_cfloat',15:'npy_cdouble',16:'npy_clongdouble'}
    cdef dict __tmpr = {21:'npy_cdatetime',22:'npy_timedelta'}
    cdef dict __char = {18:'npy_str',19:'npy_unicode'}
    cdef dict __int  = __merge_dict(__sint,__uint)
    cdef dict __inxt = __merge_dict(__flt, __cflt)
    cdef dict __num  = __merge_dict(__int,__inxt)
    cdef dict __flx  = __merge_dict({20:'npy_void'},__char)
    register_fused_type('npy_signedinteger',__uint)
    register_fused_type('npy_unsignedinteger',__sint)
    register_fused_type('npy_floating',__flt)
    register_fused_type('npy_complexfloating',__cflt)
    register_fused_type('npy_temporal',__tmpr)
    register_fused_type('npy_character',__char)
    register_fused_type('npy_integer',__int)
    register_fused_type('npy_inexact',__inxt)
    register_fused_type('npy_number',__num)
    register_fused_type('npy_flexible',__flx)
    register_fused_type('npy_generic',__merge_dict({0:'npy_bool',17:'npy_object'},__num,__flx,__tmpr))
init_fused_types()


########## Debug Functions ##########
cdef _print_info(object o):
    print("object at %08x [%d refs]" % (<uintp>(to_c(o)), PyArray_REFCOUNT(o)))
    sys.stdout.flush()
cdef _print_info_arr(ndarray a):
    cdef str shape = 'x'.join(str(x) for x in a.shape)
    cdef str strides = ', '.join(str(x) for x in a.strides)
    print("array at %08x [%d refs]: %08x [%s] (%s) %04x; base: %08x" % (
            <uintp>(to_c(a)), PyArray_REFCOUNT(a), <uintp>PyArray_DATA(a), shape, strides, PyArray_FLAGS(a), <uintp>PyArray_BASE(a)))
    sys.stdout.flush()
cdef _print_info_dtype(dtype dt):
    print("dtype at %08x [%d refs]" % (<uintp>(to_c(dt)), PyArray_REFCOUNT(dt)))
    sys.stdout.flush()
def get_ref_count(object o): return PyArray_REFCOUNT(o)


########## Utility Functions ##########
cdef inline void __array_resize1D(ndarray a, intp size):
    """
    Shortens/resizes an array by modifying the internal data. The array must own its data and have
    a minimal number of references.
    """
    cdef PyArray_Dims dims
    dims.ptr = &size; dims.len = 1
    PyArray_Resize(a, &dims, True, NPY_CORDER)

cdef inline void __array_resize2D(ndarray a, intp nrows):
    """
    Shortens/resizes an array by modifying the internal data. The array must own its data and have
    a minimal number of references.
    """
    cdef intp d[2]
    d[0] = nrows; d[1] = PyArray_DIM(a, PyArray_NDIM(a)-1)
    cdef PyArray_Dims dims
    dims.ptr = d; dims.len = 2
    PyArray_Resize(a, &dims, True, NPY_CORDER)

cdef inline ndarray __view_slice(ndarray a, intp ndim, intp* dims, intp* strides, void* data, int flags):
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

cdef inline ndarray __view_single1D(ndarray a, intp i):
    """View the element at the index from the array, equivalent to a[i]"""
    cdef intp size = 1
    return __view_slice(a, 1, &size, PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,0)*i), NPY_ARRAY_CARRAY)

cdef inline ndarray __view_single2D(ndarray a, intp i):
    """View the row at the index from the array, equivalent to a[i,:]"""
    cdef intp dims[2]
    dims[0] = 1; dims[1] = PyArray_DIM(a,1)
    return __view_slice(a, 2, dims, PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,0)*i), NPY_ARRAY_CARRAY)

cdef inline ndarray __view_trim1D(ndarray a, intp start, intp end):
    """
    View the elements from the array after removing start elements from the beginning and end elements
    from the end, equivalent to a[start:-end] (except end can be 0).
    """
    cdef intp size = PyArray_DIM(a,0)-start-end
    return __view_slice(a, 1, &size, PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,0)*start), NPY_ARRAY_CARRAY)

cdef inline ndarray __view_trim2D(ndarray a, intp start, intp end):
    """
    View the rows from the array after removing start rows from the beginning and end rows from the
    end, equivalent to a[start:-end,:] (except end can be 0).
    """
    cdef intp dims[2]
    dims[0] = PyArray_DIM(a,0)-start-end; dims[1] = PyArray_DIM(a,1)
    return __view_slice(a, 2, dims, PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,0)*start), NPY_ARRAY_CARRAY)

#cdef inline ndarray __view_flipud2D(ndarray a):
#    cdef intp strides[2]
#    strides[0] = -PyArray_STRIDE(a,0)
#    strides[1] = PyArray_STRIDE(a,1)
#    return __view_slice(a, PyArray_DATA(a), PyArray_SHAPE(a), strides, NPY_ARRAY_CARRAY)

cdef inline ndarray __view_shortened1D(ndarray a, intp size):
    """View the first size elements from the array, equivalent to a[:size]."""
    return __view_slice(a, 1, &size, PyArray_STRIDES(a), PyArray_DATA(a), NPY_ARRAY_CARRAY)

cdef inline ndarray __view_shortened2D(ndarray a, intp nrows):
    """View the first size rows from the array, equivalent to a[:size,:]."""
    cdef intp dims[2]
    dims[0] = nrows; dims[1] = PyArray_DIM(a,1)
    return __view_slice(a, 2, dims, PyArray_STRIDES(a), PyArray_DATA(a), NPY_ARRAY_CARRAY)
    
cdef inline ndarray __view_col(ndarray a, intp i):
    """View the a column from an array, equivalent to a[:,i]."""
    return __view_slice(a, 1, PyArray_SHAPE(a), PyArray_STRIDES(a), PyArray_BYTES(a)+(PyArray_STRIDE(a,1)*i), NPY_ARRAY_BEHAVED)

cdef inline ndarray __squeeze_last(ndarray a):
    """Squeeze the last dimension if it is singleton. Equivalent to a.squeeze(-1) if a.shape[-1] == 1 else a."""
    if PyArray_DIM(a, PyArray_NDIM(a)-1) != 1: return a
    return __view_slice(a, PyArray_NDIM(a)-1, PyArray_SHAPE(a), PyArray_STRIDES(a), PyArray_BYTES(a), NPY_ARRAY_CARRAY)

cdef inline ndarray __ravel_rows(ndarray a):
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

cdef int __chk_flags = NPY_ARRAY_CARRAY|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES
cdef inline ndarray __check(object arr, int min_d):
    """
    Checks an object, converting it to an array if necessary and making sure that it has at least the
    given number of dimensions, is C-contiguous, not byte-swapped, has element-sized strides.
    """
    return PyArray_CheckFromAny(arr, NULL, min_d, 0, __chk_flags, NULL)

cdef inline ndarray __check1D(object arr):
    """Like __check with min_d=1 and the output array is raveled (so always 1D)."""
    cdef ndarray a = PyArray_CheckFromAny(arr, NULL, 1, 0, __chk_flags, NULL)
    return a if PyArray_NDIM(a)==1 else PyArray_Ravel(a, NPY_CORDER) # could use PyArray_Newshape as well (that is what ravel uses internally)

cdef inline ndarray __check2D(object arr):
    """Like __check with min_d=2 and the output array is row-raveled (so always 2D)."""
    return __ravel_rows(PyArray_CheckFromAny(arr, NULL, 2, 0, __chk_flags, NULL))

cdef inline ndarray __lexsort2D(ndarray a):
    """
    Calls lexsort on a 2D array however the columns are the keys and they are kept in increasing
    priority and the actual transformed array is returned instead of just the indices.
    """
    cdef intp n = PyArray_DIM(a,1), i
    cdef list l = PyList_New(n)
    cdef ndarray c
    for i in range(n): c = __view_col(a, i); Py_INCREF(c); PyList_SET_ITEM(l, n-i-1, c)
    return PyArray_TakeFrom(a, PyArray_LexSort(l, 0), 0, NULL, NPY_RAISE)
