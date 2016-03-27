#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
#distutils: language=c++
#
# Cython helper module for label filters.
# These are not required but are much faster then the pure-Python ones.

from __future__ import division

include "npy_helper.pxi"
include "fused.pxi"

from libc.string cimport memcmp, memcpy, memmove

__all__ = ['unique_fast', 'unique_rows_fast',
           'unique_merge', 'unique_rows_merge',
           'replace', 'replace_rows',
           'searchsorted_rows',
           'number', 'number_rows',
           'renumber', 'renumber_rows',
           'relabel2', 'relabel3']

with_cython = True
           
######################################################################
#################### Unique Fast, Rows, and Merge ####################
######################################################################
# The algorithms are unique fast, unique merge (set union), replace, renumber, and search sorted.
# All of these have specific versions for row-based data. All are implemented in C++ in the label.h
# file. For the most part only integral, floating-point, and complex are optimized. All others will
# fall back to some other code.

cdef extern from "_label.h" nogil:
    pass

cdef ndarray __unique_sorted(ndarray a):
    cdef ndarray flag = PyArray_EMPTY(1, PyArray_SHAPE(a), NPY_BOOL, False)
    (<npy_bool*>PyArray_DATA(flag))[0] = True
    PyArray_CopyInto(__view_trim1D(flag, 1, 0), PyObject_RichCompare(__view_trim1D(a, 1, 0), __view_trim1D(a, 0, 1), Py_NE))
    return PyArray_Compress(a, flag, 0, NULL)
    ## Lower-memory version but just slightly slower
    #cdef ndarray a1 = __view_trim1D(a, 1, 0)
    #cdef ndarray flag = PyArray_Nonzero(PyObject_RichCompare(a1, __view_trim1D(a, 0, 1), Py_NE))[0]
    #cdef intp size = PyArray_DIM(flag,0)+1
    #cdef ndarray out = PyArray_EMPTY(1, &size, PyArray_TYPE(a), False)
    #PyArray_DESCR(a).f.copyswap(PyArray_DATA(out), PyArray_DATA(a), False, to_c(a))
    #PyArray_TakeFrom(a1, flag, 0, __view_trim1D(out, 1, 0), NPY_RAISE)
    #return out
    
cdef ndarray __unique_sorted_rows(ndarray a):
    cdef ndarray flag = PyArray_EMPTY(1, PyArray_SHAPE(a), NPY_BOOL, False)
    (<npy_bool*>PyArray_DATA(flag))[0] = True
    PyArray_Any(PyObject_RichCompare(__view_trim2D(a, 1, 0), __view_trim2D(a, 0, 1), Py_NE), 1, __view_trim1D(flag, 1, 0))
    return PyArray_Compress(a, flag, 0, NULL)


############### Unique Fast ###############
cdef extern from * nogil:
    cdef intp merge_sort_unique[T](T*,T*) except +
    cdef intp merge_sort_unique_rows[T](T*,T*,intp) except +

def __unique_fast_fallback(ndarray a not None):
    PyArray_Sort(a, 0, NPY_QUICKSORT)
    return __unique_sorted(a)

@fused(fallback=__unique_fast_fallback)
def __unique_fast(ndarray[npy_number] a not None):
    cdef intp size = PyArray_DIM(a,0)
    with nogil: size = merge_sort_unique(<npy_number*>PyArray_DATA(a), <npy_number*>PyArray_DATA(a)+size)
    return size

def __unique_fast_rows_fallback(ndarray a not None):
    return __unique_sorted_rows(__lexsort2D(a))

@fused(fallback=__unique_fast_rows_fallback)
def __unique_fast_rows(ndarray[npy_number, ndim=2] a not None):
    cdef intp* shape = PyArray_SHAPE(a)
    cdef intp nrows = shape[0], ncols = shape[1]
    with nogil: nrows = merge_sort_unique_rows(<npy_number*>PyArray_DATA(a), <npy_number*>PyArray_DATA(a)+(nrows*ncols), ncols)
    return nrows

def unique_fast(arr not None):
    """
    Sorts and finds unique values in the given array. Significantly faster than using unique.

    For non-basic data types this falls back to using sort and compress.

    Compared to the built-in unique this does not support the optional arguments return_index,
    return_inverse, or return_counts.
    """
    cdef ndarray a = PyArray_CheckFromAny(arr, NULL, 1, 0, __chk_flags|NPY_ARRAY_ENSURECOPY, NULL)
    if PyArray_NDIM(a) != 1: __array_resize1D(a, PyArray_SIZE(a))
    if PyArray_DIM(a,0) != 0:
        x = __unique_fast[PyArray_TYPE(a)](a)
        if isinstance(x, ndarray): return x
        __array_resize1D(a, x)
    return a

def unique_rows_fast(arr not None):
    """
    Sorts and finds unique rows in the given 2D array. Significantly faster than using unique.

    For non-basic data types this falls back to using lexsort and compress.
    """
    cdef ndarray a = PyArray_CheckFromAny(arr, NULL, 2, 0, __chk_flags|NPY_ARRAY_ENSURECOPY, NULL)
    if PyArray_NDIM(a) != 2: __array_resize2D(a, PyArray_MultiplyList(PyArray_SHAPE(a), PyArray_NDIM(a)-1))
    cdef intp* shape = PyArray_SHAPE(a)
    if shape[0] != 0 and shape[1] != 0:
        x = __unique_fast_rows[PyArray_TYPE(a)](a)
        if isinstance(x, ndarray): return x
        __array_resize2D(a, x)
    return a


############### Unique Merge ###############
cdef extern from * nogil:
    cdef intp set_union[T](const T*, const T*, const T*, const T*, T*)
    cdef intp row_set_union[T](const T*, const T*, const T*, const T*, T*, intp)

def __unique_merge_fallback(ndarray a not None, ndarray b not None):
    a = PyArray_Concatenate((a,b), 0)
    PyArray_Sort(a, 0, NPY_MERGESORT)
    return __unique_sorted(a)

@fused(fallback=__unique_merge_fallback)
def __unique_merge(ndarray[npy_number] a, ndarray[npy_number] b):
    cdef intp size = PyArray_DIM(a,0)+PyArray_DIM(b,0)
    cdef ndarray out = PyArray_EMPTY(1, &size, PyArray_TYPE(a), False)
    with nogil:
        size = set_union(<npy_number*>PyArray_DATA(a), <npy_number*>PyArray_DATA(a)+PyArray_DIM(a,0),
                         <npy_number*>PyArray_DATA(b), <npy_number*>PyArray_DATA(b)+PyArray_DIM(b,0),
                         <npy_number*>PyArray_DATA(out))
    __array_resize1D(out, size)
    return out

def __unique_rows_merge_fallback(ndarray a not None, ndarray b not None):
    a = PyArray_Concatenate((a,b), 0)
    return __unique_sorted_rows(__lexsort2D(a))

@fused(fallback=__unique_rows_merge_fallback)
def __unique_rows_merge(ndarray[npy_number, ndim=2] a not None, ndarray[npy_number, ndim=2] b not None):
    cdef intp nrows = PyArray_DIM(a,0)+PyArray_DIM(b,0), ncols = PyArray_DIM(a,1)
    cdef intp d[2]
    d[0] = nrows; d[1] = ncols
    cdef ndarray out = PyArray_EMPTY(2, d, PyArray_TYPE(a), False)
    with nogil:
        nrows = row_set_union(<npy_number*>PyArray_DATA(a), <npy_number*>PyArray_DATA(a)+PyArray_DIM(a,0)*ncols,
                              <npy_number*>PyArray_DATA(b), <npy_number*>PyArray_DATA(b)+PyArray_DIM(b,0)*ncols,
                              <npy_number*>PyArray_DATA(out), ncols)
    __array_resize2D(out, nrows)
    return out

def unique_merge(arr not None, brr not None):
    """
    Merges two sorted, unique, 1D arrays into a new sorted, unique, 1D array.
    
    This method does not check if the inputs are sorted and unique. If they are not, the results
    are undefined.

    For non-basic data types this falls back to using concatenate, mergesort, and compress.
    """
    cdef ndarray a = __check1D(arr)
    cdef ndarray b = __check1D(brr)
    if not PyArray_EquivArrTypes(a,b): raise ValueError('Input arrays must have the same dtype')
    if   PyArray_DIM(a,0) == 0: return b
    elif PyArray_DIM(b,0) == 0: return a
    return __unique_merge[PyArray_TYPE(a)](a, b)

def unique_rows_merge(arr not None, brr not None):
    """
    Merges two sorted, unique, 2D arrays into a new sorted, unique, 2D array. Rows are treated as
    whole units.
    
    This method does not check if the inputs are sorted and unique. If they are not, the results
    are undefined.

    For non-basic data types this falls back to using concatenate, lexsort, and compress.
    """
    cdef ndarray a = __check2D(arr)
    cdef ndarray b = __check2D(brr)
    if not PyArray_EquivArrTypes(a,b): raise ValueError('Input arrays must have the same dtype')
    if PyArray_DIM(a,1) != PyArray_DIM(b,1): raise ValueError('Input arrays must have the same number of columns')
    if   PyArray_DIM(a,0) == 0 or PyArray_DIM(a,1) == 0: return b
    elif PyArray_DIM(b,0) == 0: return a
    return __unique_rows_merge[PyArray_TYPE(a)](a, b)


############### Replace ###############
cdef extern from * nogil:
    cdef void map_replace[K,V](const K*, const V*, intp, const K*, V*, intp) except +
    cdef void map_replace_rows[K,V](const K*, const V*, intp, const K*, V*, intp, intp) except +

def __replace_fallback(ndarray k not None, ndarray v not None, ndarray a not None, ndarray out not None):
    with nogil: map_replace_rows(
        <npy_ubyte*>PyArray_DATA(k), <npy_ubyte*>PyArray_DATA(v), PyArray_DIM(k,0),
        <npy_ubyte*>PyArray_DATA(a), <npy_ubyte*>PyArray_DATA(out), PyArray_DIM(a,0), PyArray_STRIDE(a,0))

@fused(fallback=__replace_fallback)
def __replace(ndarray[npy_number] k not None, ndarray[npy_unsignedinteger] v not None,
              ndarray[npy_number] a not None, ndarray[npy_unsignedinteger] out not None):
    with nogil: map_replace(
        <npy_number*>PyArray_DATA(k), <npy_unsignedinteger*>PyArray_DATA(v), PyArray_DIM(k,0),
        <npy_number*>PyArray_DATA(a), <npy_unsignedinteger*>PyArray_DATA(out), PyArray_DIM(a,0))

@fused(fallback=__replace_fallback)
def __replace_rows(ndarray[npy_number, ndim=2] k not None, ndarray[npy_unsignedinteger, ndim=1] v not None,
                   ndarray[npy_number, ndim=2] a not None, ndarray[npy_unsignedinteger, ndim=1] out not None):
    with nogil: map_replace_rows(
        <npy_number*>PyArray_DATA(k), <npy_unsignedinteger*>PyArray_DATA(v), PyArray_DIM(k,0),
        <npy_number*>PyArray_DATA(a), <npy_unsignedinteger*>PyArray_DATA(out), PyArray_DIM(a,0), PyArray_DIM(a,1))

def replace(keys not None, vals not None, arr not None):
    """
    Replaces all elements of arr with vals by using keys as a lookup. All elements in arr must exist
    in keys and vals must be integers. If they are not, the results are undefined.
    
    For non-basic data types this falls back to using the same method on the raw binary data of each
    element.
    """
    # TODO: support any values (should be relatively easy actually)
    cdef ndarray a = __check(arr, 1)
    cdef ndarray k = __check1D(keys)
    cdef ndarray v = __check1D(vals)
    if not PyArray_EquivArrTypes(k,a): raise ValueError('Input and key arrays must have the same dtype')
    if PyArray_DIM(k,0) != PyArray_DIM(v,0): raise ValueError('Key and val arrays must be the same size')
    cdef ndarray out = PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(v), False)
    if PyArray_SIZE(a) != 0:
        __replace[PyArray_TYPE(k),PyArray_TYPE(v)](k, v, PyArray_Ravel(a, NPY_CORDER), PyArray_Ravel(out, NPY_CORDER))
    return out

def replace_rows(keys not None, vals not None, arr not None):
    """
    Replaces all rows of arr with vals by using keys as a lookup. All rows in arr must exist in keys
    and vals must be integers. If they are not, the results are undefined.

    For non-basic data types this falls back to using the same method on the raw binary data of each
    row.
    """
    # TODO: support any values (should be relatively easy actually)
    cdef ndarray a = __check(arr, 2)
    cdef ndarray k = __check2D(keys)
    cdef ndarray v = __check1D(vals)
    if not PyArray_EquivArrTypes(k,a): raise ValueError('Input and key arrays must have the same dtype')
    if PyArray_DIM(k,1) != PyArray_DIM(a,PyArray_NDIM(a)-1): raise ValueError('Input and key arrays must have the same number of columns')
    if PyArray_DIM(k,0) != PyArray_DIM(v,0): raise ValueError('Key and val arrays must be the same size')
    cdef ndarray out = PyArray_EMPTY(PyArray_NDIM(a)-1, PyArray_SHAPE(a), PyArray_TYPE(v), False)
    if PyArray_SIZE(a) != 0:
        __replace_rows[PyArray_TYPE(k),PyArray_TYPE(v)](k, v, __ravel_rows(a), PyArray_Ravel(out, NPY_CORDER))
    return out


########## Search Sorted ##########
cdef extern from * nogil:
    cdef void row_lower_bounds[T](const T*, const T*, const T*, const T*, uintp*, intp)
    cdef void row_upper_bounds[T](const T*, const T*, const T*, const T*, uintp*, intp)

    
def __searchsorted_rows_left_fallback(ndarray s not None, ndarray a not None, ndarray out not None):
    with nogil:
        row_lower_bounds(<npy_ubyte*>PyArray_DATA(s), <npy_ubyte*>PyArray_DATA(s)+PyArray_STRIDE(s,0),
                         <npy_ubyte*>PyArray_DATA(a), <npy_ubyte*>PyArray_DATA(a)+PyArray_STRIDE(a,0),
                         <uintp*>PyArray_DATA(out), PyArray_STRIDE(a,0))

@fused(fallback=__searchsorted_rows_left_fallback)
def __searchsorted_rows_left(ndarray[npy_number, ndim=2] s not None, ndarray[npy_number, ndim=2] a not None, ndarray out not None):
    cdef intp ncols = PyArray_DIM(a,1)
    with nogil:
        row_lower_bounds(<npy_number*>PyArray_DATA(s), <npy_number*>PyArray_DATA(s)+PyArray_DIM(s,0)*ncols,
                         <npy_number*>PyArray_DATA(a), <npy_number*>PyArray_DATA(a)+PyArray_DIM(a,0)*ncols,
                         <uintp*>PyArray_DATA(out), ncols)
            
def __searchsorted_rows_right_fallback(ndarray s not None, ndarray a not None, ndarray out not None):
    with nogil:
        row_upper_bounds(<npy_ubyte*>PyArray_DATA(s), <npy_ubyte*>PyArray_DATA(s)+PyArray_STRIDE(s,0),
                         <npy_ubyte*>PyArray_DATA(a), <npy_ubyte*>PyArray_DATA(a)+PyArray_STRIDE(a,0),
                         <uintp*>PyArray_DATA(out), PyArray_STRIDE(a,0))

@fused(fallback=__searchsorted_rows_right_fallback)
def __searchsorted_rows_right(ndarray[npy_number, ndim=2] s not None, ndarray[npy_number, ndim=2] a not None, ndarray out not None):
    cdef intp ncols = PyArray_DIM(a,1)
    with nogil:
        row_upper_bounds(<npy_number*>PyArray_DATA(s), <npy_number*>PyArray_DATA(s)+PyArray_DIM(s,0)*ncols,
                         <npy_number*>PyArray_DATA(a), <npy_number*>PyArray_DATA(a)+PyArray_DIM(a,0)*ncols,
                         <uintp*>PyArray_DATA(out), ncols)

def searchsorted_rows(sorted not None, arr not None, side='left'):
    """
    Takes each row in arr and searches for it in sorted using a binary search. This is basically the
    Numpy method searchsorted but it works on rows and doesn't support the "sorter" option argument.

    For non-basic data types this falls back to using the same method on the raw binary data of each
    row.
    """
    cdef ndarray a = __check(arr, 2)
    cdef ndarray s = __check2D(sorted)
    cdef NPY_SEARCHSIDE sside
    PyArray_SearchsideConverter(side, &sside)
    if not PyArray_EquivArrTypes(s,a): raise ValueError('Input and sorted arrays must have the same dtype')
    if PyArray_DIM(s,1) != PyArray_DIM(a,PyArray_NDIM(a)-1): raise ValueError('Input and sorted arrays must have the same number of columns')
    cdef ndarray out = PyArray_EMPTY(PyArray_NDIM(a)-1, PyArray_SHAPE(a), NPY_UINTP, False)
    if PyArray_SIZE(a) == 0: pass
    elif sside == NPY_SEARCHLEFT:  __searchsorted_rows_left[PyArray_TYPE(a)](__ravel_rows(a), s, out)
    elif sside == NPY_SEARCHRIGHT: __searchsorted_rows_right[PyArray_TYPE(a)](__ravel_rows(a), s, out)
    return out


########## Number ##########
cdef ndarray __fix_zero(ndarray vals, ndarray zero, ndarray pos0_arr):
    """
    This fixes the 0 entry in the replacement data by making sure there always is one and it is
    always first.
    """
    cdef intp stride = PyArray_STRIDE(vals, 0)
    cdef char* vals_p = PyArray_BYTES(vals)
    cdef void* zero_p = PyArray_DATA(zero)
    cdef uintp pos0 = (<uintp*>PyArray_DATA(pos0_arr))[0]
    if pos0 == PyArray_DIM(vals,0) or memcmp(vals_p+pos0*stride, zero_p, stride)!=0:
        return PyArray_Concatenate((zero, vals), 0) # add 0 to the beginning
    elif pos0 != 0:
        memmove(vals_p+stride, vals_p, pos0*stride) # all negatives move up
        memcpy(vals_p, zero_p, stride) # add 0 to the beginning
    return vals

def number(arr not None):
    """
    Numbers an image while keeping order. Just like renumber except that order is maintained.
    This uses unique_fast and replace.
    """
    # See scipy-lectures.github.io/advanced/image_processing/#measuring-objects-properties-ndimage-measurements for the unqiue/searchsorted method
    # First get the sorted, unique values
    cdef ndarray a = PyArray_CheckFromAny(arr, NULL, 1, 0, __chk_flags, NULL)
    cdef ndarray vals = unique_fast(a)
    # Correct the 0 entry
    cdef ndarray zero = PyArray_ZEROS(0, NULL, PyArray_TYPE(vals), False)
    vals = __fix_zero(vals, zero, PyArray_SearchSorted(vals, zero, NPY_SEARCHLEFT, NULL))
    # Use replace to create the output
    cdef ndarray nums = PyArray_Arange(0, PyArray_DIM(vals,0), 1, NPY_UINTP)
    return replace(vals, nums, a), PyArray_DIM(vals,0)-1
    
def number_rows(arr not None):
    """
    Numbers an image while keeping order. Just like renumber except that order is maintained.
    This uses unique_rows_fast and replace_rows.
    """
    # See scipy-lectures.github.io/advanced/image_processing/#measuring-objects-properties-ndimage-measurements for the unqiue/searchsorted method
    # First get the sorted, unique values
    cdef ndarray a = PyArray_CheckFromAny(arr, NULL, 1, 0, __chk_flags, NULL)
    cdef ndarray vals = unique_rows_fast(a)
    # Correct the 0 entry
    cdef intp d[2]
    d[0] = 1; d[1] = PyArray_DIM(vals,1)
    cdef ndarray zero = PyArray_ZEROS(2, d, PyArray_TYPE(a), False)
    vals = __fix_zero(vals, zero, searchsorted_rows(vals, zero))
    # Use replace to create the output
    cdef ndarray nums = PyArray_Arange(0, PyArray_DIM(vals,0), 1, NPY_UINTP)
    return replace_rows(vals, nums, a), PyArray_DIM(vals,0)-1


########## Renumber ##########
cdef extern from * nogil:
    cdef V map_renumber[K,V](const K*, V*, intp) except +
    cdef V map_renumber_rows[K,V](const K*, V*, intp, intp) except +

def __renumber_fallback(ndarray a not None, ndarray out not None):
    cdef intp N
    with nogil: N = map_renumber_rows(<npy_ubyte*>PyArray_DATA(a), <uintp*>PyArray_DATA(out), PyArray_DIM(a,0), PyArray_STRIDE(a,0))
    return N

@fused(fallback=__renumber_fallback)
def __renumber(ndarray[npy_number] a not None, ndarray out not None):
    cdef intp N
    with nogil: N = map_renumber(<npy_number*>PyArray_DATA(a), <uintp*>PyArray_DATA(out), PyArray_DIM(a,0))
    return N

@fused(fallback=__renumber_fallback)
def __renumber_rows(ndarray[npy_number, ndim=2] a not None, ndarray out not None):
    cdef intp N
    with nogil: N = map_renumber_rows(<npy_number*>PyArray_DATA(a), <uintp*>PyArray_DATA(out), PyArray_DIM(a,0), PyArray_DIM(a,1))
    return N

def renumber(arr not None):
    """
    Renumbers an array by giving every distinct element a unqiue value. The numbers are consecutive
    in that if there are N distinct element values in arr (not including 0) then the output will
    have a max of N. The value 0 is always kept as number 0.
    
    Along with the renumbered array it returns the max value given.
    
    For non-basic data types this falls back to using the same method on the raw binary data of each
    element and considers an all-0 element as 0.
    """
    cdef ndarray a = __check(arr, 1)
    cdef ndarray out = PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_UINTP, False)
    cdef intp N = 0
    if PyArray_SIZE(a) != 0:
        __renumber[PyArray_TYPE(a)](PyArray_Ravel(a, NPY_CORDER), PyArray_Ravel(out, NPY_CORDER))
    return out, N

def renumber_rows(arr not None):
    """
    Renumbers an array by giving every distinct row a unqiue value. The numbers are consecutive in
    that if there are N distinct rows in a (not including 0) then the output will have a max of N.
    The row of all 0s is always kept as number 0.
    
    Along with the renumbered array it returns the max value given.

    For non-basic data types this falls back to using the same method on the raw binary data of each
    row and considers an all-0 row as 0.
    """
    cdef ndarray a = __check(arr, 2)
    cdef ndarray out = PyArray_EMPTY(PyArray_NDIM(a)-1, PyArray_SHAPE(a), NPY_UINTP, False)
    cdef intp N = 0
    if PyArray_SIZE(a) != 0:
        N = __renumber_rows[PyArray_TYPE(a)](__ravel_rows(a), PyArray_Ravel(out, NPY_CORDER))
    return out, N


#################################################
#################### Relabel ####################
#################################################
### Relabel is inspired by SciPy's _ni_label, licensed under a BSD license. ###
# Copyright (c) 2001, 2002 Enthought, Inc. All rights reserved.
# Copyright (c) 2003-2013 SciPy Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of Enthought nor the names of the SciPy Developers may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cdef inline void zero_line(uintp* line, intp L) nogil:
    cdef intp i
    for i in range(L): line[i] = 0
cdef inline void read_line(void* p, intp stride, uintp* line, intp L) nogil:
    cdef intp i
    for i in range(L): line[i] = (<uintp*>((<char*>p)+i*stride))[0]
cdef inline void write_line(void* p, intp stride, uintp* line, intp L) nogil:
    cdef intp i
    for i in range(L): (<uintp*>((<char*>p)+i*stride))[0] = line[i]

########## Mark two labels to be merged ##########
cdef inline uintp mark_for_merge(uintp a, uintp b, uintp* mergetable) nogil:
    # find smallest root for each of a and b
    cdef uintp A = a, B = b
    while A != mergetable[A]: A = mergetable[A]
    while B != mergetable[B]: B = mergetable[B]
    cdef uintp minlbl = A if (A < B) else B

    # merge all the way down to minlabel
    mergetable[a] = mergetable[b] = minlbl
    while a != minlbl: a, mergetable[a] = mergetable[a], minlbl
    while b != minlbl: b, mergetable[b] = mergetable[b], minlbl

    return minlbl

########## Take the label of a neighbor, or mark them for merging ##########
cdef inline uintp take_label_or_merge(uintp cur_lbl, uintp other_lbl, uintp* mergetable) nogil:
    if cur_lbl == 0: return other_lbl # no current label, use the other label
    return mark_for_merge(other_lbl, cur_lbl, mergetable) if cur_lbl != other_lbl else cur_lbl

########## Label one line of input, using a prior line that has already been labeled ##########
cdef uintp process_line(uintp* current_lbls, uintp* current_vals, uintp* prior_lbls, uintp* prior_vals, intp L,
                       bint use_prior_prev, bint use_prior_adj, bint use_prior_next, bint force, bint use_prev,
                       uintp next_lbl, uintp* mergetable) nogil:
    cdef intp i
    cdef uintp v, lbl
    for i in range(L):
        v = current_vals[i]
        if v != 0:
            lbl = current_lbls[i]
            if use_prior_prev and v == prior_vals[i-1]: lbl = take_label_or_merge(lbl, prior_lbls[i-1], mergetable)
            if use_prior_adj  and v == prior_vals[i]:   lbl = take_label_or_merge(lbl, prior_lbls[i],   mergetable)
            if use_prior_next and v == prior_vals[i+1]: lbl = take_label_or_merge(lbl, prior_lbls[i+1], mergetable)
            if force:
                if use_prev and v == current_vals[i-1]: lbl = take_label_or_merge(lbl, current_lbls[i-1], mergetable)
                if lbl == 0:  # still needs a label
                    lbl = next_lbl
                    mergetable[next_lbl] = next_lbl
                    next_lbl += 1
            current_lbls[i] = lbl
    return next_lbl
cdef uintp process_line_wo_prior(uintp* current_lbls, uintp* current_vals, intp L, bint use_prev,
                                 uintp next_lbl, uintp* mergetable) nogil:
    cdef intp i
    cdef uintp v, lbl
    for i in range(L):
        v = current_vals[i]
        if v != 0:
            lbl = current_lbls[i]
            if use_prev and v == current_vals[i-1]: lbl = take_label_or_merge(lbl, current_lbls[i-1], mergetable)
            if lbl == 0:  # still needs a label
                lbl = next_lbl
                mergetable[next_lbl] = next_lbl
                next_lbl += 1
            current_lbls[i] = lbl
    return next_lbl

########## Compact the mergetable, mapping each value to its destination ##########
cdef uintp __compact_mergetable(uintp* mergetable, uintp next_lbl,
                               PyArrayIterObject *ito, intp stride,
                               uintp* line, intp L) nogil:
    if next_lbl < 2: return 0 # we found no regions
    cdef uintp src_lbl, dst_lbl = 1
    cdef intp i
    mergetable[0] = 0
    mergetable[1] = 1
    for src_lbl in range(2, next_lbl):
        # labels that map to themselves are new regions
        if mergetable[src_lbl] == src_lbl:
            mergetable[src_lbl] = dst_lbl
            dst_lbl += 1
        else:
            # we've compacted every label below this, and the
            # mergetable has an invariant (from mark_for_merge()) that
            # it always points downward.  Therefore, we can fetch the
            # final lable by two steps of indirection.
            mergetable[src_lbl] = mergetable[mergetable[src_lbl]]

    PyArray_ITER_RESET(ito)
    while PyArray_ITER_NOTDONE(ito):
        read_line(PyArray_ITER_DATA(ito), stride, line, L)
        for i in range(L): line[i] = mergetable[line[i]]
        write_line(PyArray_ITER_DATA(ito), stride, line, L)
        PyArray_ITER_NEXT(ito)

    return dst_lbl

########## Relabel regions ##########
cdef uintp __relabel2(ndarray input, ndarray structure, ndarray output):
    cdef intp L = PyArray_DIM(input,1), L_2 = L+2, stride = PyArray_STRIDE(input,1)
    cdef int axis = 1
    cdef flatiter _ito = PyArray_IterAllButAxis(output, &axis)
    cdef flatiter _iti = PyArray_IterAllButAxis(input, &axis)
    cdef PyArrayIterObject* ito = <PyArrayIterObject*>_ito
    cdef PyArrayIterObject* iti = <PyArrayIterObject*>_iti

    # Create buffer arrays for reading/writing labels and values.
    # Add an entry at the end and beginning to simplify some bounds checks.
    cdef ndarray _current_lbls = PyArray_ZEROS(1, &L_2, NPY_UINTP, False)
    cdef ndarray _current_vals = PyArray_EMPTY(1, &L_2, NPY_UINTP, False)
    cdef ndarray _prior_lbls   = PyArray_EMPTY(1, &L_2, NPY_UINTP, False)
    cdef ndarray _prior_vals   = PyArray_EMPTY(1, &L_2, NPY_UINTP, False)
    cdef uintp* current_lbls  = <uintp*>PyArray_DATA(_current_lbls)
    cdef uintp* current_vals  = <uintp*>PyArray_DATA(_current_vals)
    cdef uintp* prior_lbls    = <uintp*>PyArray_DATA(_prior_lbls)
    cdef uintp* prior_vals    = <uintp*>PyArray_DATA(_prior_vals)

    # Add fenceposts with 0 values
    current_lbls[0]   = current_vals[0]   = prior_lbls[0]   = prior_vals[0]   = 0
    current_lbls[L+1] = current_vals[L+1] = prior_lbls[L+1] = prior_vals[L+1] = 0
    current_lbls = current_lbls + 1
    current_vals = current_vals + 1
    prior_lbls = prior_lbls + 1
    prior_vals = prior_vals + 1

    # Take prior labels?
    cdef bint use_prior_prev = (<npy_bool*>PyArray_GETPTR2(structure, 0,0))[0]
    cdef bint use_prior_adj  = (<npy_bool*>PyArray_GETPTR2(structure, 0,1))[0]
    cdef bint use_prior_next = (<npy_bool*>PyArray_GETPTR2(structure, 0,2))[0]
    cdef bint use_prev       = (<npy_bool*>PyArray_GETPTR2(structure, 1,0))[0]

    cdef uintp mergetable_size = 2 * L
    cdef uintp* mergetable = <uintp*>PyDataMem_NEW(mergetable_size*sizeof(uintp))
    if mergetable == NULL: raise MemoryError()

    cdef uintp* tmp
    cdef uintp next_lbl = 1
    
    try:
        with nogil:
            # first row
            read_line(PyArray_ITER_DATA(iti), stride, current_vals, L)
            next_lbl = process_line_wo_prior(current_lbls, current_vals, L, use_prev, next_lbl, mergetable)
            write_line(PyArray_ITER_DATA(ito), stride, current_lbls, L)
            PyArray_ITER_NEXT(iti)
            PyArray_ITER_NEXT(ito)
            
            while PyArray_ITER_NOTDONE(iti):
                # swap prev and current
                tmp = current_lbls; current_lbls = prior_lbls; prior_lbls = tmp
                tmp = current_vals; current_vals = prior_vals; prior_vals = tmp

                # fill in data
                read_line(PyArray_ITER_DATA(iti), stride, current_vals, L)
                zero_line(current_lbls, L)

                # be conservative about how much space we may need
                if mergetable_size < next_lbl + L:
                    mergetable_size *= 2
                    mergetable = <uintp*>PyDataMem_RENEW(mergetable, mergetable_size*sizeof(uintp))
                
                next_lbl = process_line(current_lbls, current_vals, prior_lbls, prior_vals, L,
                                        use_prior_prev, use_prior_adj, use_prior_next, True, use_prev,
                                        next_lbl, mergetable)
                
                write_line(PyArray_ITER_DATA(ito), stride, current_lbls, L)
                PyArray_ITER_NEXT(iti)
                PyArray_ITER_NEXT(ito)
                
            next_lbl = __compact_mergetable(mergetable, next_lbl, ito, stride, current_lbls, L)     
    except:
        # clean up and re-raise
        PyDataMem_FREE(mergetable)
        raise

    PyDataMem_FREE(mergetable)
    return next_lbl
    
cdef uintp __relabel3(ndarray input, ndarray structure, ndarray output):
    cdef intp L0 = PyArray_DIM(input,0), L1 = PyArray_DIM(input,1), L = PyArray_DIM(input,2), L_2 = L+2
    cdef intp S0 = PyArray_STRIDE(input,0), S1 = PyArray_STRIDE(input,1), stride = PyArray_STRIDE(input,2), D0, D1
    cdef int axis = 2, ni
    cdef flatiter _ito = PyArray_IterAllButAxis(output, &axis)
    cdef flatiter _iti = PyArray_IterAllButAxis(input, &axis)
    cdef flatiter _itstruct = PyArray_IterAllButAxis(structure, &axis)
    cdef PyArrayIterObject* ito = <PyArrayIterObject*>_ito
    cdef PyArrayIterObject* iti = <PyArrayIterObject*>_iti
    cdef PyArrayIterObject* itstruct = <PyArrayIterObject*>_itstruct

    # Create two buffer arrays for reading/writing labels.
    # Add an entry at the end and beginning to simplify some bounds checks.
    cdef ndarray _current_lbls = PyArray_EMPTY(1, &L_2, NPY_UINTP, False)
    cdef ndarray _current_vals = PyArray_EMPTY(1, &L_2, NPY_UINTP, False)
    cdef ndarray _prior_lbls   = PyArray_EMPTY(1, &L_2, NPY_UINTP, False)
    cdef ndarray _prior_vals   = PyArray_EMPTY(1, &L_2, NPY_UINTP, False)
    cdef uintp* current_lbls  = <uintp*>PyArray_DATA(_current_lbls)
    cdef uintp* current_vals  = <uintp*>PyArray_DATA(_current_vals)
    cdef uintp* prior_lbls    = <uintp*>PyArray_DATA(_prior_lbls)
    cdef uintp* prior_vals    = <uintp*>PyArray_DATA(_prior_vals)

    # Add fenceposts with background values
    current_lbls[0]   = current_vals[0]   = prior_lbls[0]   = prior_vals[0]   = 0
    current_lbls[L+1] = current_vals[L+1] = prior_lbls[L+1] = prior_vals[L+1] = 0
    current_lbls = current_lbls + 1
    current_vals = current_vals + 1
    prior_lbls = prior_lbls + 1
    prior_vals = prior_vals + 1

    # Take previous labels?
    cdef bint use_prev = (<npy_bool*>PyArray_GETPTR3(structure, 1,1,0))[0]

    cdef uintp mergetable_size = 2 * L
    cdef uintp* mergetable = <uintp*>PyDataMem_NEW(mergetable_size*sizeof(uintp))
    if mergetable == NULL: raise MemoryError()

    cdef uintp next_lbl = 1
    cdef npy_bool* use_priors
    cdef bint use_prior_prev, use_prior_adj, use_prior_next
        
    try:
        with nogil:
            while PyArray_ITER_NOTDONE(iti):
                # fill in data
                read_line(PyArray_ITER_DATA(iti), stride, current_vals, L)
                zero_line(current_lbls, L)

                # Take neighbor labels
                PyArray_ITER_RESET(itstruct)
                for ni in range(4):
                    use_priors = <npy_bool*>PyArray_ITER_DATA(itstruct)
                    use_prior_prev = use_priors[0]
                    use_prior_adj  = use_priors[1]
                    use_prior_next = use_priors[2]
                    if not (use_prior_prev or use_prior_adj or use_prior_next):
                        PyArray_ITER_NEXT(itstruct)
                        continue

                    # be conservative about how much space we may need
                    if mergetable_size < next_lbl + L:
                        mergetable_size *= 2
                        mergetable = <uintp*>PyDataMem_RENEW(mergetable, mergetable_size*sizeof(uintp))

                    D0 = itstruct.coordinates[0] - 1
                    D1 = itstruct.coordinates[1] - 1
                    if (0 <= (ito.coordinates[0] + D0) < L0) and (0 <= (ito.coordinates[1] + D1) < L1):
                        D0 = D0*S0 + D1*S1
                        read_line(<char*>PyArray_ITER_DATA(iti) + D0, stride, prior_vals, L)
                        read_line(<char*>PyArray_ITER_DATA(ito) + D0, stride, prior_lbls, L)
                        next_lbl = process_line(current_lbls, current_vals, prior_lbls, prior_vals, L,
                                                use_prior_prev, use_prior_adj, use_prior_next, ni == 3, use_prev,
                                                next_lbl, mergetable)
                    elif ni == 3:
                        next_lbl = process_line_wo_prior(current_lbls, current_vals, L, use_prev, next_lbl, mergetable)
                    PyArray_ITER_NEXT(itstruct)

                write_line(PyArray_ITER_DATA(ito), stride, current_lbls, L)
                PyArray_ITER_NEXT(iti)
                PyArray_ITER_NEXT(ito)

            next_lbl = __compact_mergetable(mergetable, next_lbl, ito, stride, current_lbls, L)     
    except:
        # clean up and re-raise
        PyDataMem_FREE(mergetable)
        raise

    PyDataMem_FREE(mergetable)
    return next_lbl

cdef inline bint __sym2D(ndarray a, intp i, intp j):
    return (<npy_bool*>PyArray_GETPTR2(a,i,j))[0] == (<npy_bool*>PyArray_GETPTR2(a,2-i,2-j))[0]

cdef inline bint __sym3D(ndarray a, intp i, intp j, intp k):
    return (<npy_bool*>PyArray_GETPTR3(a,i,j,k))[0] == (<npy_bool*>PyArray_GETPTR3(a,2-i,2-j,2-k))[0]

cdef inline bint __sym3Dx(ndarray a, intp i, intp j):
    return ((<npy_bool*>PyArray_GETPTR3(a,i,j,0))[0] == (<npy_bool*>PyArray_GETPTR3(a,2-i,2-j,2))[0] and
            (<npy_bool*>PyArray_GETPTR3(a,i,j,1))[0] == (<npy_bool*>PyArray_GETPTR3(a,2-i,2-j,1))[0] and
            (<npy_bool*>PyArray_GETPTR3(a,i,j,2))[0] == (<npy_bool*>PyArray_GETPTR3(a,2-i,2-j,0))[0])

def relabel2(im, structure=None):
    """
    Re-labels a 2D image. Like scipy.ndimage.measurements.label function except for the following:
     * The connected components algorithm has been adjusted to put edges only between neighboring
       elements with identical values instead of all neighboring non-zero elements. This means that
       a "1" element and a "2" element next to each will not map to the same label. If this
       function is given an output from label or an array of only 0s and 1s, the output will be
       identical to label.
     * The input must be 2D and safely convertible to uintp (see relabel3 for 3D inputs). If the
       input is 3D or not integral then renumber is called on the image.
    """
    if PyArray_NDIM(im) == 3: im = __squeeze_last(im)
    if PyArray_NDIM(im) == 3: im,_ = renumber_rows(im)
    elif not PyArray_CanCastSafely(PyArray_TYPE(im), NPY_UINTP): im,_ = renumber(im)
    cdef ndarray IM = PyArray_ContiguousFromAny(im, NPY_UINTP, 2, 2), S
    del im
    if structure is None:
        from scipy.ndimage.morphology import generate_binary_structure
        S = generate_binary_structure(2,1)
    else:
        S = PyArray_ContiguousFromAny(structure, NPY_BOOL, 2, 2)
        del structure
        if PyArray_DIM(S,0) != 3 or PyArray_DIM(S,1) != 3 or not (
            __sym2D(S,0,0) and __sym2D(S,0,1) and __sym2D(S,0,2) and __sym2D(S,1,0)):
                raise ValueError('Invalid structure')
    cdef ndarray out = PyArray_EMPTY(2, PyArray_SHAPE(IM), NPY_UINTP, False)
    return out, __relabel2(IM, S, out)

def relabel3(im, structure=None):
    """
    Re-labels a 3D image. Like scipy.ndimage.measurements.label function except for the following:
     * The connected components algorithm has been adjusted to put edges only between neighboring
       elements with identical values instead of all neighboring non-zero elements. This means that
       a "1" element and a "2" element next to each will not map to the same label. If this
       function is given an output from label or an array of only 0s and 1s, the output will be
       identical to label.
     * The input must be 3D and safely convertible to uintp (see relabel2 for 2D inputs). If the
       input is 4D or not integral then renumber is called on the image.
    """
    if PyArray_NDIM(im) == 4: im = __squeeze_last(im)
    if PyArray_NDIM(im) == 4: im,_ = renumber_rows(im)
    elif not PyArray_CanCastSafely(PyArray_TYPE(im), NPY_UINTP): im,_ = renumber(im)
    cdef ndarray IM = PyArray_ContiguousFromAny(im, NPY_UINTP, 3, 3), S
    del im
    if structure is None:
        from scipy.ndimage.morphology import generate_binary_structure
        S = generate_binary_structure(3,1)
    else:
        S = PyArray_ContiguousFromAny(structure, NPY_BOOL, 3, 3)
        del structure
        if PyArray_DIM(S,0) != 3 or PyArray_DIM(S,1) != 3 or PyArray_DIM(S,2) != 3 or not (
            __sym3Dx(S, 0,0) and __sym3Dx(S, 0,1) and __sym3Dx(S, 0,2) and __sym3Dx(S, 1,0) and __sym3D(S, 1,1,0)):
                raise ValueError('Invalid structure')
    cdef ndarray out = PyArray_EMPTY(3, PyArray_SHAPE(IM), NPY_UINTP, False)
    return out, __relabel3(IM, S, out)
