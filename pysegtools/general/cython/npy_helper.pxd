# Helpful Python, Cython, and Numpy definitions

# TODO: PY_VERSION_HEX?

from libc.string cimport  memcmp, memcpy, memmove

############### Python ###############
cdef extern from "Python.h":
    ctypedef int Py_intptr_t
    ctypedef int Py_uintptr_t

    ##### Objects #####
    ctypedef struct PyObject

    cdef PyObject* Py_None
    cdef PyObject* to_c "(PyObject*)" (object) # These don't really cast, they just trick Cython
    #cdef object to_py "(PyObject*)" (PyObject*)

    cdef void Py_INCREF(object)
    cdef void Py_INCREF(PyObject*)
    cdef void Py_XINCREF(PyObject*)
    cdef void Py_DECREF(object)
    cdef void Py_DECREF(PyObject*)
    cdef void Py_XDECREF(PyObject*)
    cdef void Py_CLEAR(PyObject*)
    
    enum PyCompOp: Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE

    cdef PyObject* PyObject_GetAttrString(PyObject*, const char *attr_name) # except NULL - we check ourselves # new reference
    object PyObject_RichCompare(object, object, PyCompOp)

    ##### CObjects / PyCapsules #####
    object PyCObject_FromVoidPtr(void* cobj, void (*destr)(void *))
    object PyCObject_FromVoidPtrAndDesc(void* cobj, void* desc, void (*destr)(void *, void *))

    ctypedef void (*PyCapsule_Destructor)(object)
    object PyCapsule_New(void *pointer, const char *name, PyCapsule_Destructor destructor)
    int PyCapsule_SetContext(object capsule, void *context) except -1
    void* PyCapsule_GetContext(object capsule) except? NULL

    ##### Types #####
    #ctypedef PyObject*(*unaryfunc)(PyObject*)
    ctypedef PyObject*(*binaryfunc)(PyObject*,PyObject*)
    ctypedef PyObject*(*ternaryfunc)(PyObject*,PyObject*,PyObject*)
    ctypedef int(*inquiry)(PyObject*)
    #ctypedef int(*coercion)(PyObject**,PyObject**)

    ctypedef void(*destructor)(PyObject*)
    ctypedef int(*printfunc)(PyObject*,void*,int) # the void* is a FILE
    ctypedef PyObject*(*getattrfunc)(PyObject*,char*)
    ctypedef int(*setattrfunc)(PyObject*,char*,PyObject*)
    ctypedef int(*cmpfunc)(PyObject*,PyObject*)
    ctypedef PyObject*(*reprfunc)(PyObject*)
    ctypedef struct PyNumberMethods
    ctypedef struct PySequenceMethods
    ctypedef Py_ssize_t(*lenfunc)(PyObject*)
    ctypedef int(*objobjargproc)(PyObject*,PyObject*,PyObject*)
    ctypedef struct PyMappingMethods:
        lenfunc mp_length
        binaryfunc mp_subscript
        objobjargproc mp_ass_subscript
    ctypedef long(*hashfunc)(PyObject*)
    ctypedef PyObject*(*getattrofunc)(PyObject*,PyObject*)
    ctypedef int(*setattrofunc)(PyObject*,PyObject*,PyObject*)
    ctypedef struct PyBufferProcs
    ctypedef int(*visitproc)(PyObject*,void*)
    ctypedef int(*traverseproc)(PyObject*,visitproc,void*)
    ctypedef PyObject*(*richcmpfunc)(PyObject*,PyObject*,int)
    ctypedef PyObject*(*getiterfunc)(PyObject*)
    ctypedef PyObject*(*iternextfunc)(PyObject*)
    ctypedef struct PyMethodDef
    ctypedef struct PyMemberDef
    ctypedef struct PyGetSetDef
    ctypedef PyObject*(*descrgetfunc)(PyObject*,PyObject*,PyObject*)
    ctypedef int(*descrsetfunc)(PyObject*,PyObject*,PyObject*)
    ctypedef int(*initproc)(PyObject*,PyObject*,PyObject*)
    ctypedef PyObject*(*allocfunc)(PyTypeObject*,Py_ssize_t)
    ctypedef PyObject*(*newfunc)(PyTypeObject*,PyObject*,PyObject*)
    ctypedef void(*freefunc)(void*)

    ctypedef struct PyTypeObject:
        Py_ssize_t ob_refcnt
        PyTypeObject* ob_type
        Py_ssize_t ob_size
        const char* tp_name
        Py_ssize_t tp_basicsize
        Py_ssize_t tp_itemsize
        destructor tp_dealloc
        printfunc tp_print
        getattrfunc tp_getattr
        setattrfunc tp_setattr
        cmpfunc tp_compare
        reprfunc tp_repr
        PyNumberMethods* tp_as_number
        PySequenceMethods* tp_as_sequence
        PyMappingMethods* tp_as_mapping
        hashfunc tp_hash
        ternaryfunc tp_call
        reprfunc tp_str
        getattrofunc tp_getattro
        setattrofunc tp_setattro
        PyBufferProcs* tp_as_buffer
        long tp_flags
        const char* tp_doc
        traverseproc tp_traverse
        inquiry tp_clear
        richcmpfunc tp_richcompare
        Py_ssize_t tp_weaklistoffset
        getiterfunc tp_iter
        iternextfunc tp_iternext
        PyMethodDef* tp_methods
        PyMemberDef* tp_members
        PyGetSetDef* tp_getset
        PyTypeObject* tp_base
        PyObject* tp_dict
        descrgetfunc tp_descr_get
        descrsetfunc tp_descr_set
        Py_ssize_t tp_dictoffset
        initproc tp_init
        allocfunc tp_alloc
        newfunc tp_new
        freefunc tp_free
        inquiry tp_is_gc
        PyObject* tp_bases
        PyObject* tp_mro
        PyObject* tp_cache
        PyObject* tp_subclasses
        PyObject* tp_weaklist
        destructor tp_del
        unsigned int tp_version_tag
    #ctypedef class __builtin__.type [object PyTypeObject]:
    #    pass
    cdef bint PyType_Check(PyObject* o)
    
    ##### Functions #####
    ctypedef struct PyMethodDef
    ctypedef struct PyCFunctionObject:
        Py_ssize_t    ob_refcnt
        PyTypeObject* ob_type
        PyMethodDef*  m_ml
        PyObject*     m_self
        PyObject*     m_module
    #ctypedef class types.FunctionType [object PyCFunctionObject]:
    #    pass
    #ctypedef FunctionType function

    ##### Exceptions #####
    cdef PyObject* PyExc_TypeError
    cdef PyObject* PyExc_ValueError
    cdef PyObject* PyExc_MemoryError
    cdef void PyErr_SetString(PyObject* type, const char* message)
    cdef void PyErr_Clear()

    ##### List #####
    list PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(list, Py_ssize_t, object) # only used on new lists, steals reference

    ##### Dictionary #####
    PyObject* PyDict_GetItem(PyObject*, PyObject*) # borrowed reference, no error on NULL return

############### Cython ###############
cdef extern from *:
    ctypedef struct __pyx_CyFunctionObject:
        PyCFunctionObject func
        # TODO: IF PY_VERSION_HEX < 0x030500A0:
        PyObject* func_weakreflist # field doesn't exist in Python v3.5+, without the above conditional compilation it will likely result in a warning
        PyObject* func_dict
        PyObject* func_name
        PyObject* func_qualname
        PyObject* func_doc
        PyObject* func_globals
        PyObject* func_code
        PyObject* func_closure
        PyObject* func_classobj
        void* defaults
        int defaults_pyobjects
        int flags
        PyObject* defaults_tuple
        PyObject* defaults_kwdict
        PyObject* (*defaults_getter)(PyObject*)
        PyObject* func_annotations
    ctypedef struct __pyx_FusedFunctionObject:
        __pyx_CyFunctionObject func
        PyObject* __signatures__ "__signatures__"
        PyObject* type
        PyObject* self
    cdef PyTypeObject* __pyx_FusedFunctionType


############### Helpers ###############
ctypedef Py_intptr_t intp
ctypedef Py_uintptr_t uintp
cdef extern from "npy_helper.h":
    # Makes Numpy a bit nicer to use from Cython
    # The half and complex classes below come from this header
	
	# These types add aligned (A) or restricted (R) attributes to the types for speed
    ctypedef       double* DOUBLE_PTR_R
    ctypedef const double* DOUBLE_PTR_CR
    ctypedef       double* DOUBLE_PTR_AR
    ctypedef const double* DOUBLE_PTR_CAR
    ctypedef       intp* INTP_PTR_R
    ctypedef const intp* INTP_PTR_CR
    ctypedef       intp* INTP_PTR_AR
    ctypedef const intp* INTP_PTR_CAR
    ctypedef char* CHAR_PTR_A8R # 8-byte aligned
    ctypedef const char* CHAR_PTR_CA8R


############### Numpy ###############
cdef extern from "numpy/arrayobject.h":
    ##### Scalar Types #####
    ctypedef unsigned char      npy_bool
    # Signed Integers
    ctypedef signed char        npy_byte
    ctypedef signed short       npy_short
    ctypedef signed int         npy_int
    ctypedef signed long        npy_long
    ctypedef signed long long   npy_longlong
    ctypedef fused npy_signedinteger:
        npy_byte
        npy_short
        npy_int
        npy_long
        npy_longlong
    # Unsigned Integers
    ctypedef unsigned char      npy_ubyte
    ctypedef unsigned short     npy_ushort
    ctypedef unsigned int       npy_uint
    ctypedef unsigned long      npy_ulong
    ctypedef unsigned long long npy_ulonglong
    ctypedef fused npy_unsignedinteger:
        npy_ubyte
        npy_ushort
        npy_uint
        npy_ulong
        npy_ulonglong
    # Floating-Point Numbers
    ctypedef struct npy_half "half": # Numpy uses uint16_t, we wrap it in a nice class in npy_helper
        unsigned short raw
    cdef npy_half half_zero
    ctypedef float        npy_float
    ctypedef double       npy_double
    ctypedef long double  npy_longdouble
    ctypedef fused npy_floating:
        npy_half
        npy_float
        npy_double
        npy_longdouble
    ctypedef fused npy_floating_basic: # does not include npy_half which is a bit annoying
        npy_float
        npy_double
        npy_longdouble
    # Complex Floating-Point Numbers
    # For complex numbers Numpy uses a struct or C99 complex numbers and Cython sometimes uses
    # std::complex (calling it "float/double complex" in Cython code). The npy_helper.h ones are
    # defined as a subclass of std::complex (and typedef-ed as _cfloat, ...). The definitions below
    # are 2-element array. All of these are binary compatible.
    #ctypedef float npy_cfloat "_cfloat"[2]
    #ctypedef double npy_cdouble "_cdouble"[2]
    #ctypedef long double npy_clongdouble "_clongdouble"[2]
    ctypedef struct npy_cfloat "_cfloat":
        float R
        float I
    ctypedef struct npy_cdouble "_cdouble":
        double R
        double I
    ctypedef struct npy_clongdouble "_clongdouble":
        long double R
        long double I
    ctypedef fused npy_complexfloating:
        npy_cfloat
        npy_cdouble
        npy_clongdouble
    # Temporal Types (Numpy doesn't group these, but they don't talk about them much...)
    ctypedef signed long long npy_timedelta
    ctypedef signed long long npy_datetime
    ctypedef fused npy_temporal:
        npy_timedelta
        npy_datetime
    # Combined Types (note: Cython does not support fusing fused types, so alot of copy-paste from above...)
    ctypedef fused npy_integer:
        #npy_signedinteger
        npy_byte
        npy_short
        npy_int
        npy_long
        npy_longlong
        #npy_unsignedinteger
        npy_ubyte
        npy_ushort
        npy_uint
        npy_ulong
        npy_ulonglong
    ctypedef fused npy_inexact:
        #npy_floating
        npy_half
        npy_float
        npy_double
        npy_longdouble
        #npy_complexfloating
        npy_cfloat
        npy_cdouble
        npy_clongdouble
    ctypedef fused npy_number:
        ##npy_integer
        #npy_signedinteger
        npy_byte
        npy_short
        npy_int
        npy_long
        npy_longlong
        #npy_unsignedinteger
        npy_ubyte
        npy_ushort
        npy_uint
        npy_ulong
        npy_ulonglong
        ##npy_inexact
        #npy_floating
        npy_half
        npy_float
        npy_double
        npy_longdouble
        #npy_complexfloating
        npy_cfloat
        npy_cdouble
        npy_clongdouble
    ctypedef fused npy_number_basic: # does not include npy_half or complex types which are a bit annoying
        ##npy_integer
        #npy_signedinteger
        npy_byte
        npy_short
        npy_int
        npy_long
        npy_longlong
        #npy_unsignedinteger
        npy_ubyte
        npy_ushort
        npy_uint
        npy_ulong
        npy_ulonglong
        ##npy_inexact
        #npy_floating_basic
        npy_float
        npy_double
        npy_longdouble
    # Other types and their combinations
    ctypedef void* npy_void
    ctypedef object npy_object
    ctypedef bytes npy_str
    ctypedef unicode npy_unicode
    ctypedef fused npy_character:
        npy_str
        npy_unicode
    ctypedef fused npy_flexible:
        npy_void
        #npy_character
        npy_str
        npy_unicode
    ## The overarching scalar type: generic
    ## Currently we don't need this and it would require a lot of copying and pasting
    #ctypedef fused npy_generic:
    #    npy_bool
    #    npy_object
    #    #npy_number
    #    #npy_flexible
    #    #npy_temporal

    ##### Enums #####
    ctypedef enum NPY_ORDER: NPY_ANYORDER=-1, NPY_CORDER=0, NPY_FORTRANORDER=1
    ctypedef enum NPY_CLIPMODE: NPY_CLIP=0, NPY_WRAP, NPY_RAISE
    ctypedef enum NPY_SORTKIND: NPY_QUICKSORT=0, NPY_HEAPSORT, NPY_MERGESORT, NPY_NSORTS
    ctypedef enum NPY_SELECTKIND: NPY_INTROSELECT=0
    ctypedef enum NPY_SEARCHSIDE: NPY_SEARCHLEFT=0, NPY_SEARCHRIGHT
    cdef int PyArray_SearchsideConverter(object, NPY_SEARCHSIDE*) except 0
    ctypedef enum NPY_CASTING: NPY_NO_CASTING=0, NPY_EQUIV_CASTING, NPY_SAFE_CASTING, NPY_SAME_KIND_CASTING, NPY_UNSAFE_CASTING,
    ctypedef enum NPY_TYPES:
        NPY_BOOL=0
        NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT, NPY_INT, NPY_UINT, NPY_LONG, NPY_ULONG, NPY_LONGLONG, NPY_ULONGLONG,
        NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
        NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
        NPY_OBJECT=17,
        NPY_STRING, NPY_UNICODE,
        NPY_VOID,
        NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,
        NPY_NTYPES,
        NPY_NOTYPE,
        NPY_CHAR, # special flag
        NPY_USERDEF=256,  # leave room for characters
        NPY_NTYPES_ABI_COMPATIBLE=21, # The number of types not including the new 1.6 types (datetime, timedelta, half)
    cdef NPY_TYPES NPY_INTP
    cdef NPY_TYPES NPY_UINTP
    cdef enum:
        NPY_ARRAY_C_CONTIGUOUS   = 0x0001
        NPY_ARRAY_F_CONTIGUOUS   = 0x0002
        NPY_ARRAY_OWNDATA        = 0x0004
        NPY_ARRAY_FORCECAST      = 0x0010
        NPY_ARRAY_ENSURECOPY     = 0x0020
        NPY_ARRAY_ENSUREARRAY    = 0x0040
        NPY_ARRAY_ELEMENTSTRIDES = 0x0080 # only for PyArray_CheckFromAny
        NPY_ARRAY_ALIGNED        = 0x0100
        NPY_ARRAY_NOTSWAPPED     = 0x0200 # only for PyArray_CheckFromAny
        NPY_ARRAY_WRITEABLE      = 0x0400
        #NPY_ARR_HAS_DESCR        = 0x0800
        NPY_ARRAY_UPDATEIFCOPY   = 0x1000

        NPY_ARRAY_BEHAVED    = NPY_ARRAY_ALIGNED|NPY_ARRAY_WRITEABLE
        NPY_ARRAY_BEHAVED_NS = NPY_ARRAY_BEHAVED|NPY_ARRAY_NOTSWAPPED # only for PyArray_CheckFromAny
        NPY_ARRAY_CARRAY     = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_BEHAVED
        NPY_ARRAY_CARRAY_RO  = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED
        NPY_ARRAY_FARRAY     = NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_BEHAVED
        NPY_ARRAY_FARRAY_RO  = NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_ALIGNED
        NPY_ARRAY_UPDATE_ALL = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_ALIGNED
        NPY_ARRAY_DEFAULT    = NPY_ARRAY_CARRAY

        NPY_ARRAY_IN_ARRAY     = NPY_ARRAY_CARRAY_RO
        NPY_ARRAY_OUT_ARRAY    = NPY_ARRAY_CARRAY
        NPY_ARRAY_INOUT_ARRAY  = NPY_ARRAY_CARRAY|NPY_ARRAY_UPDATEIFCOPY
        NPY_ARRAY_IN_FARRAY    = NPY_ARRAY_FARRAY_RO
        NPY_ARRAY_OUT_FARRAY   = NPY_ARRAY_FARRAY
        NPY_ARRAY_INOUT_FARRAY = NPY_ARRAY_FARRAY|NPY_ARRAY_UPDATEIFCOPY

    ##### Utility Types #####
    PyTypeObject PyArray_Type
    ctypedef class numpy.ndarray [object PyArrayObject]:
        pass
    ctypedef struct PyArrayObject
    ctypedef class numpy.dtype [object PyArray_Descr]:
        pass
    ctypedef struct PyArray_Descr:
        Py_ssize_t ob_refcnt
        PyTypeObject* ob_type
        PyTypeObject* typeobj
        char kind
        char type
        char byteorder
        char unused
        int flags
        int num "type_num"
        int itemsize "elsize"
        int alignment
        PyArray_ArrayDescr* subarray
        PyObject* fields
        PyArray_ArrFuncs* f
    ctypedef struct PyArray_ArrayDescr:
        PyArray_Descr* base
        PyObject*      shape
    ctypedef void (*PyArray_VectorUnaryFunc)(void*,void*,intp,void*,void*)
    ctypedef object (*PyArray_GetItemFunc)(void*,void*)
    ctypedef int (*PyArray_SetItemFunc)(object,void*,void*)
    ctypedef void (*PyArray_CopySwapNFunc)(void*,intp,void*,intp,intp,bint,void*)
    ctypedef void (*PyArray_CopySwapFunc)(void*,void*,bint,void*)
    ctypedef int (*PyArray_CompareFunc)(const void*,const void*,void*)
    ctypedef int (*PyArray_ArgFunc)(void*,intp,intp*,void*)
    ctypedef void (*PyArray_DotFunc)(void*,intp,void*,intp,void*,intp,void*)
    ctypedef int (*PyArray_ScanFunc)(void*,void*,char* ignore,dtype)
    ctypedef int (*PyArray_FromStrFunc)(char*s,void*,char**,dtype)
    ctypedef npy_bool (*PyArray_NonzeroFunc)(void*,void*)
    ctypedef int (*PyArray_FillFunc)(void*,intp,void*)
    ctypedef int (*PyArray_FillWithScalarFunc)(void*,intp,void*,void*)
    ctypedef int (*PyArray_SortFunc)(void*,intp,void*)
    ctypedef int (*PyArray_ArgSortFunc)(void*,intp*,intp,void*)
    ctypedef int (*PyArray_ScalarKindFunc)(void*)
    ctypedef void (*PyArray_FastClipFunc)(void*,intp,void*,void*,void*)
    ctypedef void (*PyArray_FastPutmaskFunc)(void*,void*,intp,void*,intp)
    ctypedef int (*PyArray_FastTakeFunc)(void*,void*,intp*,intp,intp,intp,intp,NPY_CLIPMODE)
    ctypedef struct PyArray_ArrFuncs:
        PyArray_VectorUnaryFunc cast[<int>NPY_NTYPES] 
        PyArray_GetItemFunc getitem
        PyArray_SetItemFunc setitem
        PyArray_CopySwapNFunc copyswapn
        PyArray_CopySwapFunc copyswap
        PyArray_CompareFunc compare
        PyArray_ArgFunc argmax
        PyArray_DotFunc dotfunc
        PyArray_ScanFunc scanfunc
        PyArray_FromStrFunc fromstr
        PyArray_NonzeroFunc nonzero
        PyArray_FillFunc fill
        PyArray_FillWithScalarFunc fillwithscalar
        PyArray_SortFunc sort[<int>NPY_NSORTS]
        PyArray_ArgSortFunc argsort[<int>NPY_NSORTS]
        PyObject* castdict
        PyArray_ScalarKindFunc scalarkind
        int** cancastscalarkindto
        int* cancastto
        PyArray_FastClipFunc fastclip
        PyArray_FastPutmaskFunc fastputmask
        PyArray_FastTakeFunc fasttake
        PyArray_ArgFunc argmin
    ctypedef struct PyArray_Dims:
        intp* ptr
        int len
    ctypedef extern class numpy.flatiter [object PyArrayIterObject]:
        pass
    ctypedef struct PyArrayIterObject:
        intp* coordinates
    
    ##### Functions #####
    # Technically everything that returns ndarray actually returns object but this works faster
    void import_array()
    intp PyArray_MultiplyList(intp* seq, int n)
    PyArray_Descr* PyArray_DescrFromType(NPY_TYPES)
    
    ### Basic array properties ###
    # Assume a valid array, no errors
    bint PyArray_Check(object)
    int PyArray_FLAGS(ndarray) nogil
    void PyArray_ENABLEFLAGS(ndarray, int flags) nogil
    void PyArray_CLEARFLAGS(ndarray, int flags) nogil
    bint PyArray_ISCARRAY(ndarray) nogil
    bint PyArray_ISFARRAY(ndarray) nogil
    bint PyArray_ISCARRAY_RO(ndarray) nogil
    bint PyArray_ISFARRAY_RO(ndarray) nogil
    bint PyArray_ISBEHAVED(ndarray) nogil
    bint PyArray_ISBEHAVED_RO(ndarray) nogil
    bint PyArray_ISWRITEABLE(ndarray) nogil
    int PyArray_NDIM(ndarray) nogil
    intp* PyArray_SHAPE(ndarray) nogil # == PyArray_DIMS
    intp PyArray_DIM(ndarray,int) nogil
    intp PyArray_SIZE(ndarray) nogil # calls product(shape)
    int PyArray_ITEMSIZE(ndarray) nogil
    intp PyArray_NBYTES(ndarray) nogil # calls product(shape)*itemsize
    intp* PyArray_STRIDES(ndarray) nogil
    intp PyArray_STRIDE(ndarray,int) nogil
    dtype PyArray_DTYPE(ndarray) nogil # borrowed ref # == PyArray_DESCR
    PyArray_Descr* PyArray_DESCR(ndarray) nogil # borrowed ref # same as above, but C object
    bint PyArray_EquivArrTypes(ndarray,ndarray)
    cdef NPY_TYPES PyArray_TYPE "(NPY_TYPES)PyArray_TYPE" (ndarray) nogil # dtype.num with a cast
    int PyArray_ITEMSIZE(ndarray) nogil # dtype.itemsize
    PyObject* PyArray_BASE(ndarray) nogil # borrowed ref
    intp PyArray_REFCOUNT(PyObject*) nogil
    intp PyArray_REFCOUNT(object) nogil
    bint PyArray_CanCastSafely(NPY_TYPES, NPY_TYPES)

    ### Array data ###
    # Assume a valid array, no errors
    void* PyArray_DATA(ndarray) nogil
    char* PyArray_BYTES(ndarray) nogil
    void* PyArray_GETPTR1(ndarray, intp i) nogil
    void* PyArray_GETPTR2(ndarray, intp i, intp j) nogil
    void* PyArray_GETPTR3(ndarray, intp i, intp j, intp k) nogil
    void* PyArray_GETPTR4(ndarray, intp i, intp j, intp k, intp l) nogil

    ### Reshaping an array ###
    ndarray PyArray_Newshape(ndarray self, PyArray_Dims*, NPY_ORDER) # in-place if possible
    ndarray PyArray_Ravel(ndarray, NPY_ORDER) # in-place if possible
    #ndarray PyArray_Flatten(ndarray, NPY_ORDER) # not in-place
    ndarray PyArray_Transpose(ndarray, PyArray_Dims* permute) # in-place
    #ndarray PyArray_CopyAndTranspose(ndarray)
    
    ### Array creation ###
    ndarray PyArray_NewFromDescr(PyTypeObject*, PyArray_Descr*, int ndim, intp* dims, intp* strides, void* data, int flags, PyObject*) # steals dtype reference
    ndarray PyArray_SimpleNewFromData(int ndims, intp* dims, NPY_TYPES, void* data)
    int PyArray_SetBaseObject(ndarray, object) except -1 # steals reference
    ndarray PyArray_CheckFromAny(object, PyArray_Descr*, int min_depth, int max_depth, int flags, PyObject*) # steals dtype reference
    ndarray PyArray_ContiguousFromAny(object, NPY_TYPES, int min_depth, int max_depth)
    ndarray PyArray_FROMANY(object, NPY_TYPES, int min_depth, int max_depth, int requirements)
    ndarray PyArray_EMPTY(int ndims, intp* dims, NPY_TYPES, bint fortran)
    ndarray PyArray_ZEROS(int ndims, intp* dims, NPY_TYPES, bint fortran)
    ndarray PyArray_Arange(double start, double stop, double step, NPY_TYPES)
    ndarray PyArray_Concatenate(object, int axis)

    ### Extracting from an array ###
    ndarray PyArray_Compress(ndarray, object condition, int axis, ndarray out)
    ndarray PyArray_Compress(ndarray, object condition, int axis, PyArrayObject* out)
    ndarray PyArray_TakeFrom(ndarray, object indices, int axis, ndarray ret, NPY_CLIPMODE)
    ndarray PyArray_TakeFrom(ndarray, object indices, int axis, PyArrayObject* ret, NPY_CLIPMODE)
    object PyArray_Resize(ndarray, PyArray_Dims*, bint refcheck, NPY_ORDER) # returns None on success
    tuple PyArray_Nonzero(ndarray)
    int PyArray_CopyInto(ndarray, ndarray) except -1

    ### Sorting an array ###
    int PyArray_Sort(ndarray, int axis, NPY_SORTKIND) except -1 # in-place
    ndarray PyArray_LexSort(object, int axis)
    ndarray PyArray_SearchSorted(ndarray, object values, NPY_SEARCHSIDE, PyObject*)
    
    ### Calculations ###
    ndarray PyArray_Min(ndarray, int axis, PyArrayObject* out)
    ndarray PyArray_Max(ndarray, int axis, PyArrayObject* out)
    ndarray PyArray_Any(ndarray, int axis, ndarray out)
    ndarray PyArray_MatrixProduct(object, object)
    dict PyArray_GetNumericOps()
    intp PyArray_CountNonzero(ndarray) except -1

    ### Scalars ###
    object PyArray_ToScalar(void*, ndarray)

    ### Iterators ###
    object PyArray_IterNew(object)
    object PyArray_IterAllButAxis(object, int*)
    void PyArray_ITER_NEXT(PyArrayIterObject* it) nogil
    int PyArray_ITER_NOTDONE(PyArrayIterObject* it) nogil
    void PyArray_ITER_RESET(PyArrayIterObject* it) nogil
    void* PyArray_ITER_DATA(PyArrayIterObject* it) nogil
    
    ### Memory ###
    void *PyDataMem_NEW(size_t) nogil
    void PyDataMem_FREE(void *) nogil
    void *PyDataMem_RENEW(void *, size_t) nogil
