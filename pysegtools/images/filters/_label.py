# Python fallback helper module for label filters.
# These are used when Cython cannot compile the high-speed module.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import repeat

from numpy import zeros, empty, asarray, ascontiguousarray, concatenate, dtype, uintp
from numpy import lexsort, equal, place
from scipy.ndimage.measurements import label

__all__ = ['unique_fast','unique_rows_fast','unique_merge','unique_rows_merge',
           'number','number_rows','renumber','renumber_rows','relabel2','relabel3']

with_cython = False

def __ravel_rows(a): return a if a.ndim == 2 else a.reshape((-1, a.shape[-1]))
def __check(a, min_d): a = asarray(a); assert a.ndim >= min_d; return a
def __check1D(a): return __check(a, 1).ravel()
def __check2D(a): return __ravel_rows(__check(a, 2))
def __unique_sorted(a):
    flag = empty(len(a), dtype=bool)
    flag[0] = True
    flag[1:] = a[1:] != a[:-1]
    return a.compress(flag, axis=0)
    ## Less memory but slightly slower (<1% slower)
    #flag = nonzero(a[1:] != a[:-1])[0]
    #out = empty(len(flag)+1, dtype=a.dtype)
    #out[0] = a[0]
    #a[1:].take(flag, axis=0, out=out[1:])
    #return out
def __unique_sorted_rows(a):
    flag = empty(a.shape[0], dtype=bool)
    flag[0] = True
    (a[1:]!=a[:-1]).any(axis=1, out=flag[1:])
    return a.compress(flag, axis=0)


def unique_fast(a):
    """
    Sorts and finds unique values in the given array.
    """
    a = a.flatten()
    if a.size == 0: return a
    a.sort()
    return __unique_sorted(a)
def unique_rows_fast(a):
    """
    Sorts and finds unique rows in the given 2D array.
    """
    a = asarray(a)
    assert a.ndim >= 2
    a = __ravel_rows(a)
    if a.size == 0: return a
    return __unique_sorted_rows(a.take(lexsort(a.T[::-1]), axis=0))
def unique_merge(a, b):
    """
    Merges two sorted, unique, 1D arrays into a new sorted, unique, 1D array.
    
    This method does not check if the inputs are sorted and unique. If they are not, the results
    are undefined.
    """
    a, b = __check1D(a), __check1D(b)
    if a.dtype != b.dtype: raise ValueError('Input arrays must have the same dtype')
    if   a.size == 0: return b
    elif b.size == 0: return a
    a = concatenate((a, b))
    a.sort(kind='mergesort')
    return __unique_sorted(a)
def unique_rows_merge(a, b):
    """
    Merges two sorted, unique, 2D arrays into a new sorted, unique, 2D array. Rows are treated as
    whole units.
    
    This method does not check if the inputs are sorted and unique. If they are not, the results
    are undefined.
    """
    a, b = __check2D(a), __check2D(b)
    if a.dtype != b.dtype: raise ValueError('Input arrays must have the same dtype')
    if a.shape[1] != b.shape[1]: raise ValueError('Input arrays must have the same number of columns')
    if   a.shape[0] == 0 or a.shape[1] == 0: return b
    elif b.shape[0] == 0: return a
    a = concatenate((a, b))
    return __unique_sorted_rows(a.take(lexsort(a.T[::-1]), axis=0))


def number(a):
    """
    Numbers an image while keeping order. Just like _renumber except that order is maintained.
    The non-optimized version uses unique_fast and searchsorted.
    """
    # See scipy-lectures.github.io/advanced/image_processing/#measuring-objects-properties-ndimage-measurements for the unqiue/searchsorted method
    # First get the sorted, unique values
    vals = unique_fast(a)
    zero = a.dtype.type(0)
    pos0 = vals.searchsorted(zero) # we also may need to correct the 0 position
    # Use searchsorted to create the output
    out, N = vals.searchsorted(a).view(uintp), len(vals)-1
    if pos0 == len(vals) or vals[pos0] != zero:
        out += 1; N += 1   # account for the 0 which did not exist
    elif pos0 != 0:        # there were negative values
        out[out<pos0] += 1 # all negatives go up
        place(out, a==zero, 0) # set 0s to 0
    return out, N
def number_rows(a):
    """
    Numbers an image while keeping order. Just like _renumber except that order is maintained.
    The non-optimized version uses unique_rows_fast and searchsorted.
    """
    # See scipy-lectures.github.io/advanced/image_processing/#measuring-objects-properties-ndimage-measurements for the unqiue/searchsorted method
    # First get the sorted, unique values
    vals = unique_rows_fast(a)
    # View image and values as structured arrays)
    dt = dtype(zip(repeat(str('')), repeat(a.dtype, a.shape[-1])))
    a = ascontiguousarray(a)
    vals = ascontiguousarray(vals).view(dt).squeeze(-1)
    zero = zeros(1, dtype=dt)
    # Use searchsorted to create the output
    pos0 = vals.searchsorted(zero) # we may need to correct the 0 position
    out, N = vals.searchsorted(a.view(dt).squeeze(-1)).view(uintp), len(vals)-1
    if pos0 == len(vals) or vals[pos0] != zero:
        out += 1; N += 1   # account for the 0 which did not exist
    elif pos0 != 0:        # there were negative values
        out[out<pos0] += 1 # all negatives go up
        place(out, (a==0).all(axis=1), 0) # set 0s to 0
    return out, N
def renumber(a):
    """
    Renumbers an array by giving every distinct element a unqiue value. The numbers are consecutive
    in that if there are N distinct element values in arr (not including 0) then the output will
    have a max of N. The value 0 is always kept as number 0.
    
    Along with the renumbered array it returns the max value given.

    The non-optimized version simply uses "number".
    """
    return number(a)
def renumber_rows(a):
    """
    Renumbers an array by giving every distinct row a unqiue value. The numbers are consecutive in
    that if there are N distinct rows in a (not including 0) then the output will have a max of N.
    The row of all 0s is always kept as number 0.
    
    Along with the renumbered array it returns the max value given.

    The non-optimized version simply uses "number_rows".
    """
    return number_rows(a)


def __relabel_core(im, N, structure):
    mask = empty(im.shape, dtype=bool)
    lbl = empty(im.shape, dtype=uintp)
    for i in xrange(1, N+1):
        n = label(equal(im, i, out=mask), structure, lbl)
        for j in xrange(2, n+1):
            N += 1
            place(im, equal(lbl, j, out=mask), N)
    return im, N
def relabel2(im, structure):
    """
    Re-labels a 2D image. Like scipy.ndimage.measurements.label function except for the following:
     * The connected components algorithm has been adjusted to put edges only between neighboring
       elements with identical values instead of all neighboring non-zero elements. This means that
       a "1" element and a "2" element next to each will not map to the same label. If this
       function is given an output from label or an array of only 0s and 1s, the output will be
       identical to label.
     * number is called on the image (in the non-optimized version at least).
    """
    if im.ndim == 3 and im.shape[2] == 1: im = im.squeeze(2)
    im, N = (number_rows if im.ndim == 3 else number)(im)
    return __relabel_core(im, N, structure)
def relabel3(ims, structure):
    """
    Re-labels a 3D image. Like scipy.ndimage.measurements.label function except for the following:
     * The connected components algorithm has been adjusted to put edges only between neighboring
       elements with identical values instead of all neighboring non-zero elements. This means that
       a "1" element and a "2" element next to each will not map to the same label. If this
       function is given an output from label or an array of only 0s and 1s, the output will be
       identical to label.
     * number is called on the image (in the non-optimized version at least).
    """
    if ims.ndim == 4 and ims.shape[3] == 1: ims = ims.squeeze(3)
    ims, N = (number_rows if ims.ndim == 4 else number)(ims)
    return __relabel_core(ims, N, structure)

def searchsorted_rows(sorted_, arr_, side_='left'):
    raise NotImplementedError('No non-optimized version of searchsorted_rows is available')
def replace(keys_, vals_, arr_):
    raise NotImplementedError('No non-optimized version of replace is available')
def replace_rows(keys_, vals_, arr_):
    raise NotImplementedError('No non-optimized version of replace_rows is available')
