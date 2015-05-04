"""Labeling Images"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from itertools import repeat
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import zeros, empty, asarray, ascontiguousarray, concatenate, dtype, int8, uint8, intp
from numpy import unique, lexsort, equal, not_equal, place

from ..types import check_image, get_dtype_max, get_im_dtype_and_nchan, get_im_dtype, create_im_dtype
from ._stack import FilteredImageStack, FilteredImageSlice
from ...imstack import CommandEasy, Opt, Help

__all__ = ['label','relabel','consecutively_number','shrink_integer',
           'LabelImageStack','RelabelImageStack','ConsecutivelyNumberImageStack']
try:
    import numpy as np
    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()})
    from _label import uniq_merge, uniq_merge_rows, unique_rows
except ImportError:
    __not_warned = True
    def uniq_merge(a, b):
        """
        Merges two sorted, unique, 1D arrays into a new sorted, unique, 1D array. The optimized
        version does part of this in-place and the second list is modified. Does no checks, only
        give it the expected arguments.
        """
        if __not_warned:
            print('Warning: cannot load optimized label functions. Install Cython. In the mean time, some things might be slow.', file=sys.stderr)
            __not_warned = False
        c = concatenate((a, b))
        c.sort(kind=kind)
        flag = empty(len(c), dtype=bool)
        flag[0] = True
        not_equal(c[1:], c[:-1], out=flag[1:])
        return c.compress(flag)
    def uniq_merge_rows(a, b):
        """
        Merges two sorted, unique, 2D arrays into a new sorted, unique, 2D array (where the rows are
        unique). The optimized version does part of this in-place and the second list is modified.
        Does no checks, only give it the expected arguments.
        """
        return unique_rows_py(concatenate((a, b)))
    def unique_rows(a):
        """
        Finds unique rows in the given 2D array. The optimized version does returns a view of a
        larger array. If you keep this around it is recommended you make a copy of the view to
        reduce the memory used. Does no checks, only give it the expected arguments.
        """
        if __not_warned:
            print('Warning: cannot load optimized label functions. Install Cython. In the mean time, some things might be slow.', file=sys.stderr)
            __not_warned = False
        a = a.take(lexsort(a.T[::-1]), axis=0)
        flag = empty(a.shape[0], dtype=bool)
        flag[0] = True
        (a[1:]!=a[:-1]).any(axis=1, out=flag[1:])
        return a.compress(flag, axis=0)

def label(im, structure=None):
    """
    Performs a connected-components analysis on the provided image. 0s are considered background.
    Any other values are connected into consecutively numbered contigous regions (where contigous is
    having a neighbor above, below, left, or right). Returns the labeled image and the max label
    value.
    """
    from scipy.ndimage.measurements import label
    check_image(im)
    if im.ndim == 2:
        im = im != 0
    elif im.ndim == 3 and im.shape[2] == 1:
        im = im.squeeze(2) != 0
    else:
        im = (im != 0).any(2)
    return label(im, structure, intp)

def relabel(im, structure=None):
    """
    Relabels a labeled image. Basically, makes sure that all labels are consecutive and checks that
    every label is one connected region. For labels that specified disjoint regions, one connected
    region is given the previous label and the other is given a new label that is greater than all
    other labels (except when there were gaps in the numbering which were filled in). Returns the
    labeled image and the max label value.
    """
    from scipy.ndimage.measurements import label
    #check_image(im) # not needed, consecutively_number does it for us
    im, N = consecutively_number(im)
    mask = empty(im.shape, dtype=bool)
    lbl = empty(im.shape, dtype=intp)
    for i in xrange(1, N+1):
        n = label(equal(im, i, out=mask), structure, lbl)
        for j in xrange(2, n+1):
            N += 1
            place(im, equal(lbl, j, out=mask), N)
    return im, N


def consecutively_number(im):
    """
    Creates a consecutively numbered image from an image. Every pixel is is ordered then assigned a
    number based on its order. Same valued pixels are given the same value. The value 0 is only ever
    assigned to the 0 pixel. For multi-channel images, the pixels are lex-sorted and all channels
    must be 0 to considered 0. Returns the re-numbered image and the max number assigned.
    """
    check_image(im)
    # See scipy-lectures.github.io/advanced/image_processing/#measuring-objects-properties-ndimage-measurements for the unqiue/searchsorted method

    # First get the sorted, unique values
    if im.ndim == 2 or im.shape[2] == 1:
        # Single-channel image
        if im.ndim == 3: im = im.squeeze(2)
        values = unique(im)
        zero = im.dtype.type(0)
    else:
        # Multi-channel image
        nchans = im.shape[2]
        values = unique_rows(im.reshape(-1, nchans))

        # View image and values as structured arrays
        dt = dtype(zip(repeat(''), repeat(im.dtype, nchans)))
        im = ascontiguousarray(im).view(dt).squeeze(1)
        values = ascontiguousarray(values).view(dt).squeeze(1)
        zero = zeros(1, dtype=dt)

    # Second use searchsorted to create the output
    out, N = values.searchsorted(im), len(values)-1
    # We also need to correct the 0 position
    pos0 = values.searchsorted(zero)
    if pos0 == len(values) or values[pos0] != zero:
        out += 1; N += 1   # account for the 0 which did not exist
    elif pos0 != 0:        # there were negative values
        out[out<pos0] += 1 # all negatives go up
        out[im==zero] = 0  # set 0s to 0
    return out, N


def shrink_integer(im, min_dt=None):
    """
    Take an integer image (either signed or unsigned) and shrink the size of the integer so that it
    still fits the min and max values. By default this will shrink the image down to single bytes,
    keeping it signed or unsigned. You can also specify a minimum data type size and the integer
    size won't be reduced below that. If the minimum data type is not the same signed/unsigned as
    the image, the image will be converted if possible (an example of where it will raise an
    exception is if a the image is signed and has negative values and it is requested to be
    unsigned).

    [Technically, if the min_dt is larger than the current image's data-type the image will "grow",
    not shrink]
    """
    check_image(im)
    return im.astype(_shrink_int_dtype(im, min_dt), casting='safe', copy=False)
def _shrink_int_dtype(im, min_dt):
    if im.dtype.kind not in 'iu': raise ValueError('Can only take integral data types')
    unsigned = im.dtype.kind == 'u'
    min_dt = (uint8 if unsigned else int8) if min_dt is None else dtype(min_dt)
    mn, mx = (0 if unsigned else im.min()), im.max()
    return _shrink_int_dtype_raw(mn, mx, min_dt)
def _shrink_int_dtype_raw(mn, mx, min_dt):
    # At this point min_dt must be a dtype and the min and max values are passed directly
    if min_dt.kind == 'u':
        if mn < 0: raise ValueError('Cannot change to unsigned if there are negative values')
        types = 'uint'
        f = lambda dt:(dt.itemsize if get_dtype_max(dt)>=mx else 1000)
    elif min_dt.kind == 'i':
        types = 'int'
        f = lambda dt:(dt.itemsize if get_dtype_max(dt)>=mx and get_dtype_min(dt)<=mn else 1000)
    else:
        raise ValueError('Can only take integral data types')
    dt = min((dtype(t) for t in sctypes[types]), key=f)
    if f(dt) == 1000: raise ValueError('Cannot find an integeral data type to convert to that doesn\'t clip values')
    return dt


##### Image Stacks #####
class __LabeledImageStack(FilteredImageStack):
    def __init__(self, ims, slcs):
        super(__LabeledImageStack, self).__init__(ims, slcs)
class __LabeledImageSlice(FilteredImageSlice):
    def _get_props(self):
        _, nchans = get_im_dtype_and_nchan(self._input._dtype)
        self._set_props(create_im_dtype(intp, channels=nchans), self._input.shape)

class __LabeledImageStackWithStruct(__LabeledImageStack):
    __metaclass__ = ABCMeta
    def __init__(self, ims, per_slice=True, structure=None):
        ndim = (2 if per_slice else 3)
        if structure is not None:
            structure = asarray(structure, dtype=bool)
            if structure.ndim != ndim or any(x!=3 for x in structure.shape): raise ValueError('Invalid structure')
        else:
            from scipy.ndimage.morphology import generate_binary_structure
            structure = generate_binary_structure(ndim, 1)
        self._structure = structure
        if per_slice: super(__LabeledImageStackWithStruct, self).__init__(ims, imperslc)
        elif not ims.is_homogeneous: raise ValueError('Cannot label the entire stack if it is not homogeneous')
        else:
            super(__LabeledImageStackWithStruct, self).__init__(ims, imslc)
            self._shape = ims.shape
            self._homogeneous = Homogeneous.Shape
    @abstractmethod
    def _calc_labels(self, ims): pass
    def _calc_labels_full(self):
        ims = self._ims._stack
        ims.squeeze(3) if ims.ndim == 4 and ims.shape[3] == 1 else ims
        ims, N = self._calc_labels(ims)
        ims.flags.writeable = False
        self._labelled, self._n_labels = ims, N
        for slc,lbl in zip(self._slices, ims): slc._labelled = lbl
    @property
    def n_labels(self):
        if not hasattr(self, '_n_labels'): self._stack._calc_labels_full()
        return self._n_labels
    @property
    def stack(self):
        if not hasattr(self, '_labelled'): self._stack._calc_labels_full()
        return self._labelled
class __LabelImagePerSliceWithStruct(__LabeledImageSlice):
    def _get_data(self): return self._stack._calc_label(self._input.data, self._stack._structure)
class __LabelImageSlice(__LabeledImageSlice):
    def _get_data(self):
        if not hasattr(self, '_labelled'): self._stack._calc_labels_full()
        return self._labelled


class LabelImageStack(__LabeledImageStackWithStruct):
    def __init__(self, ims, per_slice=True, structure=None):
        super(LabelImageStack, self).__init__(ims, per_slice, structure)
    _calc_label = lambda im,s: label(im, s)
    def _calc_labels(self, ims):
        from scipy.ndimage.measurements import label
        ims = ims != 0
        return label(ims.any(3) if ims.ndim == 4 else ims, self._structure, intp)
class RelabelImageStack(__LabeledImageStackWithStruct):
    def __init__(self, ims, per_slice=True, structure=None):
        super(RelabelImageStack, self).__init__(ims, per_slice, structure)
    _calc_label = relabel
    def _calc_labels(self, ims):
        from scipy.ndimage.measurements import label
        # Pretend the entire stack is just a single slice to consecutively number it
        shape = ims.shape
        ims = ims.reshape((1,-1) if len(shape) == 3 else (1,-1,shape[3]))
        ims, N = consecutively_number(ims)
        ims = ims.reshape(shape)
        # Re-label
        mask = empty(ims.shape, dtype=bool)
        lbl = empty(ims.shape, dtype=intp)
        for i in xrange(1, N+1):
            n = label(equal(ims, i, out=mask), self._structure, lbl)
            for j in xrange(2, n+1):
                N += 1
                place(ims, equal(lbl, j, out=mask), N)
        return ims, N


class ConsecutivelyNumberImageStack(__LabeledImageStack):
    def __init__(self, ims, per_slice=True):
        if per_slice: super(ConsecutivelyNumberImageStack, self).__init__(ims, ConsecutivelyNumberImagePerSlice)
        elif not ims.is_dtype_homogeneous: raise ValueError('Cannot consecutively number the entire stack if it\'s data-type is not homogeneous')
        else: super(ConsecutivelyNumberImageStack, self).__init__(ims, ConsecutivelyNumberImageSlice)
    def _calc_values(self):
        if not hasattr(self, '_values'):
            # First get the sorted, unique values
            if self._d == 0:
                self._values, self._zero, self._pos0 = asarray([], dtype=self._dtype), 0, -1
            else:
                dt, nchans = get_im_dtype_and_nchan(self._dtype)
                slices = iter(self._slices)
                im = next(slices)._input.data
                
                if nchans == 1:
                    # Single-channel image
                    values = unique(im.squeeze(2) if im.ndim == 3 else im)
                    for slc in slices:
                        im = slc._input._data
                        values = uniq_merge(values, unique(im.squeeze(2) if im.ndim == 3 else im))
                    zero = im.dtype.type(0)

                else:
                    # Multi-channel image
                    values = unique_rows(im.reshape(-1, nchans))
                    for slc in slices:
                        im = slc._input._data
                        values = uniq_merge_rows(values, unique_rows(im.reshape(-1, nchans)))
                    self._dt = dt = dtype(zip(repeat(''), repeat(dt, nchans)))
                    zero = zeros(1, dtype=dt)
                    values = ascontiguousarray(values).view(dt).squeeze(1)

                pos0 = values.searchsorted(zero)
                self._values, self._zero = values, zero
                self._pos0 = -1 if pos0 == len(values) or values[pos0] != zero else pos0
        return self._values, self._zero, self._pos0
    @property
    def n_labels(self):
        values, _, pos0 = self._stack._calc_values()
        return len(values) - (0 if self._pos0 == -1 else 1)
class ConsecutivelyNumberImagePerSlice(__LabeledImageSlice):
    def _get_data(self): return consecutively_number(self._input.data)
class ConsecutivelyNumberImageSlice(__LabeledImageSlice):
    def _get_data(self):
        im = self._input.data
        # Get the values and re-view the image
        values, zero, pos0 = self._stack._calc_values()
        if im.ndim == 3:
            if im.shape[2] != 1: im = ascontiguousarray(im).view(self._stack._dt)
            im = im.squeeze(2)
        # Second use searchsorted to create the output
        out = values.searchsorted(im)
        # We also need to correct the 0 position
        if pos0 == -1: out += 1 # account for the 0 which did not exist
        elif pos0 != 0:         # there were negative values
            out[out<pos0] += 1  # all negatives go up
            out[im==zero] = 0   # set 0s to 0
        return out


class ShrinkIntegerImageStack(FilteredImageStack):
    def __init__(self, ims, min_dt=None, per_slice=True):
        if min_dt is not None:
            min_dt = dtype(min_dt)
            if min_dt.kind not in 'iu': raise ValueError('Can only take integral data types')
        self._min_dt = min_dt
        super(ShrinkIntegerImageStack, self).__init__(ims,
            ShrinkIntegerImagePerSlice if per_slice else ShrinkIntegerImageSlice)
    def _calc_dtype(self):
        if not hasattr(self, '__dtype'):
            if self._d == 0:
                self.__dtype = uint8 if self._min_dt is None else self._min_dt
            else:
                kinds = [slc.dtype.base.kind for slc in self._slices]
                if any(k not in 'iu' for k in kinds): raise ValueError('Can only take integral data types')
                kinds = [k == 'u' for k in kinds]
                slices = iter(self._slices)
                slc = next(slices)
                im = slc.data
                mn, mx = (0 if kinds[0] else im.min()), im.max()
                for slc,unsigned in zip(slices,kinds):
                    im = slc.data
                    mn, mx = min((0 if unsigned else im.min()), mn), max(im.max(), mx)
                min_dt = self._min_dt
                if min_dt is None:
                    min_dt = uint8 if mn >= 0 and any(u for u in kinds) else int8
                self.__dtype = _shrink_int_dtype_raw(mn, mx, min_dt)
        return self.__dtype
class ShrinkIntegerImagePerSlice(FilteredImageSlice):
    def _get_props(self):
        dt = _shrink_int_dtype(self._input.data, self._stack._min_dt)
        _, nchans = get_im_dtype_and_nchan(self._input._dtype)
        self._set_props(create_im_dtype(dt, channels=nchans), self._input.shape)
    def _get_data(self):
        im = shrink_integer(self._input.data, self._stack._min_dt)
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im
class ShrinkIntegerImageSlice(FilteredImageSlice):
    def _get_props(self):
        _, nchans = get_im_dtype_and_nchan(self._input._dtype)
        self._set_props(create_im_dtype(self._stack._calc_dtype(), channels=nchans), self._input.shape)
    def _get_data(self):
        dt = self._stack._calc_dtype()
        im = self._input.data.astype(dt, casting='safe', copy=False)
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im


##### Commands #####
class LabelImageCommand(CommandEasy):
    _per_slice = None
    @classmethod
    def name(cls): return 'label'
    @classmethod
    def _desc(cls): return """
Labels an image by performing connected-components analysis. 0s are considered background and any
other values are connected into consecutively numbered contiguous regions (where contiguous is
having a neighbor above, below, left, or right). This can also operate on the entire input stack if
it has a homogeneous shape. In this case contiguous also includes above and below.
"""
    @classmethod
    def flags(cls): return ('l', 'label')
    @classmethod
    def _opts(cls): return (
        Opt('per-slice', 'If false, operate on the entire stack at once', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be labelled',)
    @classmethod
    def _produces(cls): return ('Labelled image stack, either int32 or int64 depending on OS',)
    @classmethod
    def _see_also(cls): return ('relabel','consecutively-number','shrink-int')
    def __str__(self): return 'label'+('' if self._per_slice else ' entire stack at-once')
    def execute(self, stack): stack.push(LabelImageStack(stack.pop(), self._per_slice))

class RelabelImageCommand(CommandEasy):
    _per_slice = None
    @classmethod
    def name(cls): return 'relabel'
    @classmethod
    def _desc(cls): return """
Relabels a labeled image. The basic idea is it does a consecutive renumbering then runs "label" for
each value. This makes sure that all labels are consecutive and checks that every label is one
connected region. For labels that specified disjoint regions, one connected region is given the
previous label and the other is given a new label that is greater than all other labels (except when
there were gaps in the numbering which were filled in).
"""
    @classmethod
    def flags(cls): return ('relabel',)
    @classmethod
    def _opts(cls): return (
        Opt('per-slice', 'If false, operate on the entire stack at once', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be re-labelled',)
    @classmethod
    def _produces(cls): return ('Relabelled image stack, either int32 or int64 depending on OS',)
    @classmethod
    def _see_also(cls): return ('label','consecutively-number','shrink-int')
    def __str__(self): return 'relabel'+('' if self._per_slice else ' entire stack at-once')
    def execute(self, stack): stack.push(RelabelImageStack(stack.pop(), self._per_slice))

class ConsecutivelyNumberImageCommand(CommandEasy):
    _per_slice = None
    @classmethod
    def name(cls): return 'consecutively number'
    @classmethod
    def _desc(cls): return """
Consecutively numbers an image. Every pixel is is ordered then assigned a number based on its order.
Same valued pixels are given the same value. The value 0 is only ever assigned to the 0 pixel. For
multi-channel images, the pixels are lex-sorted and all channels must be 0 to considered 0.
"""
    @classmethod
    def flags(cls): return ('consecutively-number',)
    @classmethod
    def _opts(cls): return (
        Opt('per-slice', 'If false, operate on the entire stack at once', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be numbered',)
    @classmethod
    def _produces(cls): return ('Numbers image stack, either int32 or int64 depending on OS',)
    @classmethod
    def _see_also(cls): return ('label','relabel','shrink-int')
    def __str__(self): return 'consecutively-number'+('' if self._per_slice else ' entire stack at-once')
    def execute(self, stack): stack.push(ConsecutivelyNumberImageStack(stack.pop(), self._per_slice))

class ShrinkIntegerImageCommand(CommandEasy):
    @staticmethod
    def _cast_dt(x):
        x = x.lower()
        if len(x) < 2 or x[0] not in 'iu' or not x[1:].isdigit(): raise ValueError()
        nbits = int(x[1:])
        if nbits % 8 != 0: raise ValueError()
        nbytes = nbits // 8
        dt = dtype(x[0] + str(nbytes))
        if dt.itemsize != nbytes: raise ValueError()
        return dt
    _min_dt = None
    _per_slice = None
    @classmethod
    def name(cls): return 'shrink integer'
    @classmethod
    def _desc(cls): return """
Take an integer image (either signed or unsigned) and shrink the size of the integer so that it
still fits the min and max values. By default this will shrink the image down to single bytes,
keeping it signed or unsigned. You can also specify a minimum data type size and the integer size
won't be reduced below that. If the minimum data type is not the same signed/unsigned as the image,
the image will be converted if possible (an example of where it will raise an exception is if a the
image is signed and has negative values and it is requested to be unsigned).

[Technically, if the min_dt is larger than the current image's data-type the image will "grow", not
shrink]
"""
    @classmethod
    def flags(cls): return ('shrink-int',)
    @classmethod
    def _opts(cls): return (
        Opt('min-dt', 'The minimum data type to shrink too, specified as an u or i followed by the number of bits as a multiple of 8', ShrinkIntegerImageCommand._cast_dt, None, 'either u8 or i8 depending on image type'),
        Opt('per-slice', 'If false, operate on the entire stack at once', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be shrunk',)
    @classmethod
    def _produces(cls): return ('Shrunk image stack',)
    @classmethod
    def _see_also(cls): return ('label','relabel','consecutively-number')
    def __str__(self): return 'shrink-int'+('' if self._per_slice else ' entire stack at-once')
    def execute(self, stack): stack.push(ShrinkIntegerImageStack(stack.pop(), self._min_dt, self._per_slice))
