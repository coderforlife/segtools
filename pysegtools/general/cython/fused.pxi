# Adds an enhanced fused type to Cython. If there are no fused functions, including this file will
# cause compile errors.

DEF ADD_FUSED_TYPE_TO_FUNCTION=False

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
cdef dict __fused_types = {}
cdef inline void register_fused_type(bytes fused_type_name, dict types):
    """
    Registers the fused type named "fused_type_name" and has the types that are the values in the
    dictionary "types". The keys of "types" are values you want to use to lookup the fused function
    with. For example, an integer (for Numpy which gives a number to each type), a type object or a
    string.
    """
    import sys
    cdef bint py2 = sys.version_info[0] == 2
    IF ADD_FUSED_TYPE_TO_FUNCTION:
        __fused_types[frozenset(types.itervalues() if py2 else types.values())] = (fused_type_name,types)
    ELSE:
        __fused_types[frozenset(types.itervalues() if py2 else types.values())] = types
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
    import sys
    cdef bint py2 = sys.version_info[0] == 2
    cdef dict sigs = f.__signatures__
    cdef bytes s
    IF ADD_FUSED_TYPE_TO_FUNCTION:
        for k,v in sigs.iteritems() if py2 else sigs.items(): v.__fused_type = k
        if b'|' in next(iter(sigs)):
            from itertools import izip, product
            fts = [__fused_types[fs] for fs in (frozenset(s) for s in izip(*[s.split(b'|') for s in sigs]))]
            f.__fused_type = b'|'.join(ft[0] for ft in fts)
            fts = list(product(*[(ft[1].iteritems() if py2 else ft[1].items()) for ft in fts]))
            itr = izip((tuple(k for k,_ in ft) for ft in fts), (b'|'.join(v for _,v in ft) for ft in fts))
        else:
            f.__fused_type, ids = __fused_types[frozenset(sigs)]
            itr = ids.iteritems() if py2 else ids.items()
    ELSE:
        if b'|' in next(iter(sigs)):
            from itertools import izip, product
            fts = list(product(*[
                    __fused_types[fs].iteritems() if py2 else __fused_types[fs].items() for fs in
                    (frozenset(s) for s in izip(*[s.split(b'|') for s in sigs]))
                ]))
            itr = izip((tuple(k for k,_ in ft) for ft in fts), (b'|'.join(v for _,v in ft) for ft in fts))
        else:
            itr = __fused_types[frozenset(sigs)].iteritems() if py2 else __fused_types[frozenset(sigs)].items()
    sigs.update({k:sigs[v] for k,v in itr})
    f.__fallback = fallback
    return f

### Register all of the Numpy fused types defined in npy_helper.pxd ###
cdef void __init_fused_types():
    def __merge_dict(*args):
        cdef dict out = {}
        for d in args: out.update(d)
        return out
    cdef dict __sint = {1:b'npy_byte',3:b'npy_short',5:b'npy_int',7:b'npy_long',9:b'npy_longlong'}
    cdef dict __uint = {2:b'npy_ubyte',4:b'npy_ushort',6:b'npy_uint',8:b'npy_ulong',10:b'npy_ulonglong'}
    cdef dict __flt_basic = {11:b'npy_float',12:b'npy_double',13:b'npy_longdouble'}
    cdef dict __flt  = {23:b'npy_half',11:b'npy_float',12:b'npy_double',13:b'npy_longdouble'}
    cdef dict __cflt = {14:b'npy_cfloat',15:b'npy_cdouble',16:b'npy_clongdouble'}
    cdef dict __tmpr = {21:b'npy_cdatetime',22:b'npy_timedelta'}
    cdef dict __char = {18:b'npy_str',19:b'npy_unicode'}
    cdef dict __int  = __merge_dict(__sint,__uint)
    cdef dict __inxt = __merge_dict(__flt, __cflt)
    cdef dict __num_basic  = __merge_dict(__int,__flt_basic)
    cdef dict __num  = __merge_dict(__int,__inxt)
    cdef dict __flx  = __merge_dict({20:b'npy_void'},__char)
    register_fused_type(b'npy_signedinteger',__uint)
    register_fused_type(b'npy_floating_basic',__flt_basic)
    register_fused_type(b'npy_floating',__flt)
    register_fused_type(b'npy_complexfloating',__cflt)
    register_fused_type(b'npy_temporal',__tmpr)
    register_fused_type(b'npy_character',__char)
    register_fused_type(b'npy_integer',__int)
    register_fused_type(b'npy_inexact',__inxt)
    register_fused_type(b'npy_number_basic',__num_basic)
    register_fused_type(b'npy_number',__num)
    register_fused_type(b'npy_flexible',__flx)
    register_fused_type(b'npy_generic',__merge_dict({0:b'npy_bool',17:b'npy_object'},__num,__flx,__tmpr))
__init_fused_types()
