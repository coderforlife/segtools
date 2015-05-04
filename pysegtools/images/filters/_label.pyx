#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool as py_bool
from libc.string cimport memcmp, memcpy

np.import_array()

#TODO: add support for the following (and future types) if cython/numpy.pxd ever creates IFDEFs for them
#  float128 (#13)
#  complex256 (#16)
#  float80/96 (#20/21?)
#  complex160/192 (#20/21?)
#  float16 (#23)
# Future types: int96, int128, uint96, uint128

ctypedef fused FT: # all but complex
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t

ctypedef fused FTC: # complex
    np.complex64_t
    np.complex128_t
    
ctypedef fused FTA: # all
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t


########## Single Channel ##########

cdef np.ndarray[FT, ndim=1, mode="c"] __uniq_merge(np.ndarray[FT, ndim=1, mode="c"] a, np.ndarray[FT, ndim=1, mode="c"] b):
    cdef Py_ssize_t i = 0, j = 0, k = 0
    cdef Py_ssize_t LA = a.shape[0], LB = b.shape[0], NN = a.itemsize
    while j < LB:
        while a[i] < b[j]:
            i += 1
            if i == LA: memcpy(&b[k], &b[j], NN*(LB-j)); k += LB-j; j = LB; break
        else:
            if a[i] != b[j]: b[k] = b[j]; k += 1
            j += 1
    LB = k
    if LB == 0: return a
    cdef np.intp_t *dims = [LA+LB]
    cdef np.ndarray[FT, ndim=1, mode="c"] out = np.PyArray_EMPTY(1, dims, a.dtype.num, 0)
    if out is None: raise MemoryError()
    i = 0; j = 0; k = 0
    while i < LA and j < LB:
        if a[i] < b[j]: out[k] = a[i]; i += 1
        else:           out[k] = b[j]; j += 1
        k += 1
    memcpy(&out[k], &a[i] if i < LA else &b[j], NN*(LA+LB-k))
    return out
cdef np.ndarray[FTC, ndim=1, mode="c"] __uniq_merge_cmplx(np.ndarray[FTC, ndim=1, mode="c"] a, np.ndarray[FTC, ndim=1, mode="c"] b):
    cdef Py_ssize_t i = 0, j = 0, k = 0
    cdef Py_ssize_t LA = a.shape[0], LB = b.shape[0], NN = a.itemsize
    while j < LB:
        while a[i].real < b[j].real or (a[i].real == b[j].real and a[i].imag < b[j].imag):
            i += 1
            if i == LA: memcpy(&b[k], &b[j], NN*(LB-j)); k += LB-j; j = LB; break
        else:
            if a[i] != b[j]: b[k] = b[j]; k += 1
            j += 1
    LB = k
    if LB == 0: return a
    cdef np.intp_t *dims = [LA+LB]
    cdef np.ndarray[FTC, ndim=1, mode="c"] out = np.PyArray_EMPTY(1, dims, a.dtype.num, 0)
    if out is None: raise MemoryError()
    i = 0; j = 0; k = 0
    while i < LA and j < LB:
        if a[i].real < b[j].real or (a[i].real == b[j].real and a[i].imag < b[j].imag):
              out[k] = a[i]; i += 1
        else: out[k] = b[j]; j += 1
        k += 1
    memcpy(&out[k], &a[i] if i < LA else &b[j], NN*(LA+LB-k))
    return out


########## Multi-Channel ##########

cdef inline int lt(FT* a, FT* b, Py_ssize_t N) nogil:
    cdef Py_ssize_t i = 0
    while i < N and a[i] == b[i]: i += 1
    return i != N and a[i] < b[i]
cdef inline int lt_uniq(FT* a, FT* b) nogil: # assumes a and b are not equal
    cdef Py_ssize_t i = 0
    while a[i] == b[i]: i += 1
    return a[i] < b[i]
cdef inline Py_ssize_t __unique_rows2(FT[:,:] a, FT[:,:] b, Py_ssize_t NN) nogil:
    cdef Py_ssize_t i = 0, j = 0, k = 0
    cdef Py_ssize_t LA = a.shape[0], N = a.shape[1]
    cdef Py_ssize_t LB = b.shape[0]
    while j < LB:
        while lt(&a[i,0], &b[j,0], N):
            i += 1
            if i == LA: memcpy(&b[k,0], &b[j,0], NN*(LB-j)); k += LB-j; j = LB; break
        else:
            if memcmp(&a[i,0], &b[j,0], NN): memcpy(&b[k,0], &b[j,0], NN); k += 1
            j += 1
    return k
cdef inline void __merge_rows(FT[:,:] a, FT[:,:] b, Py_ssize_t LB, FT[:,:] out, Py_ssize_t NN) nogil:
    cdef Py_ssize_t i = 0, j = 0, k = 0
    cdef Py_ssize_t LA = a.shape[0]
    cdef FT* A
    cdef FT* B
    cdef FT* P
    while i < LA and j < LB:
        A = &a[i,0]; B = &b[j,0]
        if lt_uniq(A, B): P = A; i += 1
        else:             P = B; j += 1
        memcpy(&out[k,0], P, NN)
        k += 1
    memcpy(&out[k,0], &a[i,0] if i < LA else &b[j,0], NN*(LA+LB-k))
cdef np.ndarray[FT, ndim=2, mode="c"]  __uniq_merge_rows(np.ndarray[FT, ndim=2, mode="c"] A, np.ndarray[FT, ndim=2, mode="c"] B):
    cdef FT[:,:] a = A, b = B
    cdef Py_ssize_t N = a.shape[1], NN = N * a.itemsize, LB
    with nogil: LB = __unique_rows2(a, b, N)
    if LB == 0: return a
    cdef np.intp_t *dims = [a.shape[0]+LB, N]
    cdef np.ndarray[FT, ndim=2, mode="c"] OUT = np.PyArray_EMPTY(2, dims, A.dtype.num, 0)
    if OUT is None: raise MemoryError()
    cdef FT[:,:] out = OUT
    with nogil: __merge_rows(a, b, LB, out, N)
    return OUT


cdef inline Py_ssize_t __unique_rows_internal(FTA[:,::1] a, Py_ssize_t N) nogil:
    cdef Py_ssize_t i, k = 1
    cdef FTA* last = &a[0,0]
    for i in range(1, a.shape[0]):
        if memcmp(last, &a[i,0], N): last = &a[k,0]; memcpy(last, &a[i,0], N); k += 1
    return k
cdef np.ndarray[FTA, ndim=2, mode="c"] __unique_rows(np.ndarray[FTA, ndim=2, mode="c"] A):
    cdef FTA[:,::1] a = A
    cdef Py_ssize_t N = a.shape[1]*a.itemsize, k
    with nogil: k = __unique_rows_internal(a, N)
    return A[:k]


########## Wrapper Functions ##########

@cython.wraparound(True)
def uniq_merge(np.ndarray a, np.ndarray b):
    """
    Merges two sorted, unique, lists into a new sorted, unique, list. The second list is modified.
    This one is for uni-dimensional basic data types.
    """
    cdef np.ndarray flag, c
    cdef int n = a.dtype.num if a.dtype.isnative else -1
    if n == 5 or n == 6: n = np.dtype(a.dtype.name).num
    if   n == 1:  return __uniq_merge(<np.ndarray[np.int8_t,    ndim=1, mode="c"]>a, <np.ndarray[np.int8_t,    ndim=1, mode="c"]>b)
    elif n == 2:  return __uniq_merge(<np.ndarray[np.uint8_t,   ndim=1, mode="c"]>a, <np.ndarray[np.uint8_t,   ndim=1, mode="c"]>b)
    elif n == 3:  return __uniq_merge(<np.ndarray[np.int16_t,   ndim=1, mode="c"]>a, <np.ndarray[np.int16_t,   ndim=1, mode="c"]>b)
    elif n == 4:  return __uniq_merge(<np.ndarray[np.uint16_t,  ndim=1, mode="c"]>a, <np.ndarray[np.uint16_t,  ndim=1, mode="c"]>b)
    elif n == 7:  return __uniq_merge(<np.ndarray[np.int32_t,   ndim=1, mode="c"]>a, <np.ndarray[np.int32_t,   ndim=1, mode="c"]>b)
    elif n == 8:  return __uniq_merge(<np.ndarray[np.uint32_t,  ndim=1, mode="c"]>a, <np.ndarray[np.uint32_t,  ndim=1, mode="c"]>b)
    elif n == 9:  return __uniq_merge(<np.ndarray[np.int64_t,   ndim=1, mode="c"]>a, <np.ndarray[np.int64_t,   ndim=1, mode="c"]>b)
    elif n == 10: return __uniq_merge(<np.ndarray[np.uint64_t,  ndim=1, mode="c"]>a, <np.ndarray[np.uint64_t,  ndim=1, mode="c"]>b)
    elif n == 11: return __uniq_merge(<np.ndarray[np.float32_t, ndim=1, mode="c"]>a, <np.ndarray[np.float32_t, ndim=1, mode="c"]>b)
    elif n == 12: return __uniq_merge(<np.ndarray[np.float64_t, ndim=1, mode="c"]>a, <np.ndarray[np.float64_t, ndim=1, mode="c"]>b)
    elif n == 14: return __uniq_merge_cmplx(<np.ndarray[np.complex64_t,  ndim=1, mode="c"]>a, <np.ndarray[np.complex64_t,  ndim=1, mode="c"]>b)
    elif n == 15: return __uniq_merge_cmplx(<np.ndarray[np.complex128_t, ndim=1, mode="c"]>a, <np.ndarray[np.complex128_t, ndim=1, mode="c"]>b)
    else:
        c = np.concatenate((a, b))
        c.sort(kind='mergesort')
        flag = np.empty(len(c), dtype=bool)
        flag[0] = True
        np.not_equal(c[1:], c[:-1], out=flag[1:])
        return c.compress(flag)

@cython.wraparound(True)
def uniq_merge_rows(np.ndarray a, np.ndarray b):
    """
    Merges two sorted, unique, lists into a new sorted, unique, list. The second list is modified.
    This one is for bi-dimensional basic data types and rows are treated as whole units.
    """
    cdef np.ndarray flag, c
    cdef int n = a.dtype.num if a.dtype.isnative else -1
    if n == 5 or n == 6: n = np.dtype(a.dtype.name).num
    if   n == 1:  return __uniq_merge_rows(<np.ndarray[np.int8_t,    ndim=2, mode="c"]>a, <np.ndarray[np.int8_t,    ndim=2, mode="c"]>b)
    elif n == 2:  return __uniq_merge_rows(<np.ndarray[np.uint8_t,   ndim=2, mode="c"]>a, <np.ndarray[np.uint8_t,   ndim=2, mode="c"]>b)
    elif n == 3:  return __uniq_merge_rows(<np.ndarray[np.int16_t,   ndim=2, mode="c"]>a, <np.ndarray[np.int16_t,   ndim=2, mode="c"]>b)
    elif n == 4:  return __uniq_merge_rows(<np.ndarray[np.uint16_t,  ndim=2, mode="c"]>a, <np.ndarray[np.uint16_t,  ndim=2, mode="c"]>b)
    elif n == 7:  return __uniq_merge_rows(<np.ndarray[np.int32_t,   ndim=2, mode="c"]>a, <np.ndarray[np.int32_t,   ndim=2, mode="c"]>b)
    elif n == 8:  return __uniq_merge_rows(<np.ndarray[np.uint32_t,  ndim=2, mode="c"]>a, <np.ndarray[np.uint32_t,  ndim=2, mode="c"]>b)
    elif n == 9:  return __uniq_merge_rows(<np.ndarray[np.int64_t,   ndim=2, mode="c"]>a, <np.ndarray[np.int64_t,   ndim=2, mode="c"]>b)
    elif n == 10: return __uniq_merge_rows(<np.ndarray[np.uint64_t,  ndim=2, mode="c"]>a, <np.ndarray[np.uint64_t,  ndim=2, mode="c"]>b)
    elif n == 11: return __uniq_merge_rows(<np.ndarray[np.float32_t, ndim=2, mode="c"]>a, <np.ndarray[np.float32_t, ndim=2, mode="c"]>b)
    elif n == 12: return __uniq_merge_rows(<np.ndarray[np.float64_t, ndim=2, mode="c"]>a, <np.ndarray[np.float64_t, ndim=2, mode="c"]>b)
    else:
        c = np.concatenate((a, b))
        c = c.take(np.lexsort(c.T[::-1]), axis=0)
        flag = np.empty(a.shape[0], dtype=bool)
        flag[0] = True
        (c[1:]!=c[:-1]).any(axis=1, out=flag[1:])
        return c.compress(flag, axis=0)

@cython.wraparound(True)
def unique_rows(np.ndarray a):
    """
    Finds unique rows in the given 2D array.
    """
    cdef np.ndarray flag
    a = a.take(np.lexsort(a.T[::-1]), axis=0)
    cdef int n = a.dtype.num if a.dtype.isnative else -1
    if n == 5 or n == 6: n = np.dtype(a.dtype.name).num
    if   n == 1:  return __unique_rows(<np.ndarray[np.int8_t,    ndim=2, mode="c"]>a)
    elif n == 2:  return __unique_rows(<np.ndarray[np.uint8_t,   ndim=2, mode="c"]>a)
    elif n == 3:  return __unique_rows(<np.ndarray[np.int16_t,   ndim=2, mode="c"]>a)
    elif n == 4:  return __unique_rows(<np.ndarray[np.uint16_t,  ndim=2, mode="c"]>a)
    elif n == 7:  return __unique_rows(<np.ndarray[np.int32_t,   ndim=2, mode="c"]>a)
    elif n == 8:  return __unique_rows(<np.ndarray[np.uint32_t,  ndim=2, mode="c"]>a)
    elif n == 9:  return __unique_rows(<np.ndarray[np.int64_t,   ndim=2, mode="c"]>a)
    elif n == 10: return __unique_rows(<np.ndarray[np.uint64_t,  ndim=2, mode="c"]>a)
    elif n == 11: return __unique_rows(<np.ndarray[np.float32_t, ndim=2, mode="c"]>a)
    elif n == 12: return __unique_rows(<np.ndarray[np.float64_t, ndim=2, mode="c"]>a)
    elif n == 14: return __unique_rows(<np.ndarray[np.complex64_t,  ndim=2, mode="c"]>a)
    elif n == 15: return __unique_rows(<np.ndarray[np.complex128_t, ndim=2, mode="c"]>a)
    else:
        flag = np.empty(a.shape[0], dtype=bool)
        flag[0] = True
        (a[1:]!=a[:-1]).any(axis=1, out=flag[1:])
        return a.compress(flag, axis=0)
