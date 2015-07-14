"""Labeling Images"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from warnings import warn
from itertools import repeat
from abc import ABCMeta, abstractmethod

from numpy import zeros, asarray, ascontiguousarray, concatenate, arange, place
from numpy import dtype, int8, uint8, uintp, sctypes
from scipy.ndimage.measurements import label as sp_label

from ..types import check_image, get_dtype_max, get_dtype_min
from ..types import create_im_dtype, im_dtype_desc, get_im_dtype_and_nchan, get_im_dtype
from ._stack import FilteredImageStack, FilteredImageSlice
from .._stack import Homogeneous
from ...imstack import CommandEasy, Opt

__all__ = ['number','label','relabel','shrink_integer',
           'LabelImageStack','RelabelImageStack','ConsecutivelyNumberImageStack','ShrinkIntegerImageStack']


########## Core Functions ##########
# These are implemented in Cython (with fallbacks in Python). See _label.pyx. The _label.pyx can
# take quite a bit of time to compile the first time (on the order of minutes). The _label.pyd/.so
# created can be moved between different "identical" systems. If the directory containing _label.pyx
# is writable, it will be placed into that directory. If the directory is not writable, it will be
# placed somewhere in ~/.pyxbld. It is aways checked for in those places before re-compiling.
from . import _cython; _cython.install()
from . import _label
def __squeeze_last(a): return a.squeeze(-1) if a.shape[-1] == 1 else a
def _number2(a):
    if a.ndim == 3: a = __squeeze_last(a)
    return (_label.number_rows if a.ndim == 3 else _label.number)(a)
def _number3(a):
    if a.ndim == 4: a = __squeeze_last(a)
    return (_label.number_rows if a.ndim == 4 else _label.number)(a)
def _renumber2(a):
    if a.ndim == 3: a = __squeeze_last(a)
    return (_label.renumber_rows if a.ndim == 3 else _label.renumber)(a)
def _renumber3(a):
    if a.ndim == 4: a = __squeeze_last(a)
    return (_label.renumber_rows if a.ndim == 4 else _label.renumber)(a)
def _label2(a, structure):
    if a.ndim == 3: a = __squeeze_last(a)
    a = a != 0
    return sp_label(a if a.ndim == 2 else a.any(2), structure, uintp)
def _label3(a, structure):
    if a.ndim == 4: a = __squeeze_last(a)
    a = a != 0
    return sp_label(a if a.ndim == 3 else a.any(3), structure, uintp)


########## Single Slice Functions ##########
def number(im, ordered=False):
    """
    Creates a consecutively numbered image from an image. Every distinct pixel value is assigned a
    positive integer and all pixels with that value are replaced by that number. The value 0 is only
    ever assigned to the 0-valued pixel (all channels equal to 0). There are no gaps in the numbering
    except if there is no 0-valued pixel (then there is a gap in that no pixel is assigned 0).
    
    If ordered is True, the pixel values are kept in the same order. For multi-channel images the
    pixels are lex-sorted.
    
    Returns the re-numbered image and the max number assigned.
    """
    check_image(im)
    return _number2(im) if ordered else _renumber2(im)

def label(im, structure=None):
    """
    Performs a connected-components analysis on the provided image. 0s are considered background.
    Any other values are connected into consecutively numbered contigous regions (where contigous is
    having a neighbor above, below, left, or right). Returns the labeled image and the max label
    assigned.
    """
    check_image(im)
    return _label2(im, structure)

def relabel(im, structure=None):
    """
    Relabels a labeled image. This makes sure that all the image is consecutively numbered (no gaps
    in the numbering) and that every labeled region is contiguous. Returns the relabeled image and
    the max label assigned.
    """
    check_image(im)
    return _label.relabel2(im, structure)

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
    return im.astype(_shrink_int_dtype(im, min_dt), copy=False)
def _shrink_int_dtype(im, min_dt):
    if im.dtype.kind not in 'iu': raise ValueError('Can only take integral data types')
    unsigned = im.dtype.kind == 'u'
    min_dt = dtype(uint8 if unsigned else int8) if min_dt is None else dtype(min_dt)
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


########## Image Stacks ##########
class _LabeledImageStack(FilteredImageStack):
    def __init__(self, ims, slcs):
        super(_LabeledImageStack, self).__init__(ims, slcs)
class _LabeledImageSlice(FilteredImageSlice):
    def _get_props(self): self._set_props(dtype(uintp), self._input.shape)
    @abstractmethod
    def _get_data(self): pass
        
class _LabeledImageStackWithStruct(_LabeledImageStack):
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
        self._labelled = None
        self._n_labels = None
        if per_slice: super(_LabeledImageStackWithStruct, self).__init__(ims, _LabelImagePerSliceWithStruct)
        elif not ims.is_homogeneous: raise ValueError('Cannot label the entire stack if it is not homogeneous')
        else:
            super(_LabeledImageStackWithStruct, self).__init__(ims, _LabelImageSlice)
            self._shape = ims.shape
            self._homogeneous = Homogeneous.Shape
    @abstractmethod
    def _calc_label(self, im): pass
    @abstractmethod
    def _calc_labels(self, ims): pass
    def _calc_labels_full(self):
        self._labelled, self._n_labels = self._calc_labels(self._ims.stack)
        self._labelled.flags.writeable = False
    @property
    def n_labels(self):
        if self._n_labels is None: self._calc_labels_full()
        return self._n_labels
    @property
    def stack(self):
        if self._labelled is None: self._calc_labels_full()
        return self._labelled
class _LabelImagePerSliceWithStruct(_LabeledImageSlice):
    #pylint: disable=protected-access
    def _get_data(self): return self._stack._calc_label(self._input.data)
class _LabelImageSlice(_LabeledImageSlice):
    def _get_data(self): return self._stack.stack[self._z]

class LabelImageStack(_LabeledImageStackWithStruct):
    def __init__(self, ims, per_slice=True, structure=None):
        super(LabelImageStack, self).__init__(ims, per_slice, structure)
    def _calc_label(self, im): return _label2(im, self._structure)[0]
    def _calc_labels(self, ims): return _label3(ims, self._structure)
class RelabelImageStack(_LabeledImageStackWithStruct):
    def __init__(self, ims, per_slice=True, structure=None):
        super(RelabelImageStack, self).__init__(ims, per_slice, structure)
    def _calc_label(self, im): return _label.relabel2(im, self._structure)[0]
    def _calc_labels(self, ims): return _label.relabel3(ims, self._structure)

class ConsecutivelyNumberImageStack(_LabeledImageStack):
    #pylint: disable=protected-access
    def __init__(self, ims, ordered=False, per_slice=True):
        if per_slice:
            self._number = _number2 if ordered else _renumber2
            super(ConsecutivelyNumberImageStack, self).__init__(ims, ConsecutivelyNumberImagePerSlice)
        elif not ims.is_dtype_homogeneous: raise ValueError('Cannot consecutively number the entire stack if it\'s data-type is not homogeneous')
        elif ordered and ims.is_shape_homogeneous:
            self._renumbered = None
            self._n_labels = None
            self._calc_n_labels = self._calc_renumbered
            super(ConsecutivelyNumberImageStack, self).__init__(ims, ConsecutivelyRenumberImageSlice)
        else:
            if ordered: warn('Cannot optimize for unordered numbering of the entire stack if it\'s shape is not homogeneous')
            self._n_labels = None
            self.__calc_im = None
            self._calc_n_labels = self._calc_values
            super(ConsecutivelyNumberImageStack, self).__init__(ims, ConsecutivelyNumberImageSlice)
    def _calc_values(self):
        # This calculates the sorted, unique values
        if self._d == 0:
            self.__calc_im = lambda im: None
            self._n_labels = 0
            return
        dt, nchans = get_im_dtype_and_nchan(self.dtype)
        single_chan = nchans == 1
        slices = iter(self._slices)
        if single_chan:
            vals = _label.unique_fast(next(slices)._input.data)
            for slc in slices: vals = _label.unique_merge(vals, _label.unique_fast(slc._input._data))
            zero = dt.type(0)
        else:
            vals = _label.unique_rows_fast(next(slices)._input.data)
            for slc in slices: vals = _label.unique_rows_merge(vals, _label.unique_rows_fast(slc._input._data))

        if _label.with_cython:
            # Prepare to use replace (vals, idxs)
            if single_chan:
                pos0 = vals.searchsorted(zero)
            else:
                zero = zeros((1,nchans), dtype=dt)
                pos0 = _label.searchsorted_rows(vals, zero)[0]
            if pos0 == len(vals) or (vals[pos0] != 0).any():
                vals = concatenate((zero, vals)) # add 0 to the beginning
            elif pos0 != 0:
                vals[1:pos0+1] = vals[:pos0]     # all negatives go up
                vals[0] = 0                      # add 0 to the beginning
            idxs = arange(len(vals), dtype=uintp)
            self._n_labels = len(vals) - 1
            self.__calc_im = (lambda im:_label.replace(vals, idxs, im)) if single_chan else \
                             (lambda im:_label.replace_rows(vals, idxs, im))
        else:
            # Prepare to use search sorted (vals, pos0 and sometimes dt)
            if not single_chan:
                dt = dtype(zip(repeat(str('')), repeat(dt, nchans)))
                vals = ascontiguousarray(vals).view(dt).squeeze(-1)
                zero = zeros(1,dtype=dt)
            pos0 = vals.searchsorted(zero)
            pos0 = (-1 if pos0 == len(vals) or vals[pos0] != zero else pos0)
            self._n_labels = len(vals) - (0 if pos0 == -1 else 1)
            if single_chan:
                def __calc_im(im):
                    out = vals.searchsorted(im).view(uintp)
                    if pos0 == -1: out += 1     # account for the 0 which did not exist
                    elif pos0 != 0:             # there were negative values
                        out[out<pos0] += 1      # all negatives go up
                        place(out, im==0, 0)    # set 0s to 0
                    return out
            else:
                def __calc_im(im):
                    im = ascontiguousarray(im)
                    out = vals.searchsorted(im.view(dt).squeeze(-1)).view(uintp)
                    if pos0 == -1: out += 1     # account for the 0 which did not exist
                    elif pos0 != 0:             # there were negative values
                        out[out<pos0] += 1      # all negatives go up
                        place(out, (im==0).all(axis=1), 0) # set 0s to 0
                    return out
            self.__calc_im = __calc_im
    def _calc_renumbered(self):
        self._renumbered, self._n_labels = _renumber3(self._ims.stack)
        self._renumbered.flags.writeable = False
    def _calc_im(self, im):
        if self.__calc_im is None: self._calc_values()
        return self.__calc_im(im)
    @property
    def n_labels(self):
        if self._n_labels is None: self._calc_n_labels()
        return self._n_labels
    @property
    def stack(self):
        if self._renumbered is None: self._calc_renumbered()
        return self._renumbered
class ConsecutivelyNumberImagePerSlice(_LabeledImageSlice):
    #pylint: disable=protected-access
    def _get_data(self): return self._stack._number(self._input.data)[0]
class ConsecutivelyNumberImageSlice(_LabeledImageSlice):
    #pylint: disable=protected-access
    def _get_data(self): return self._stack._calc_im(self._input.data)
class ConsecutivelyRenumberImageSlice(_LabeledImageSlice):
    def _get_data(self): return self._stack.stack[self._z]

class ShrinkIntegerImageStack(FilteredImageStack):
    def __init__(self, ims, min_dt=None, per_slice=True):
        if min_dt is not None:
            min_dt = dtype(min_dt)
            if min_dt.kind not in 'iu': raise ValueError('Can only take integral data types')
        self._min_dt = min_dt
        self.__dtype = None
        super(ShrinkIntegerImageStack, self).__init__(ims,
            ShrinkIntegerImagePerSlice if per_slice else ShrinkIntegerImageSlice)
    def _calc_dtype(self):
        if self.__dtype is None:
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
                self.__dtype = _shrink_int_dtype_raw(mn, mx, dtype(min_dt))
        return self.__dtype
class ShrinkIntegerImagePerSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        dt = _shrink_int_dtype(self._input.data, self._stack._min_dt)
        _, nchans = get_im_dtype_and_nchan(self._input.dtype)
        self._set_props(create_im_dtype(dt, channels=nchans), self._input.shape)
    def _get_data(self):
        im = shrink_integer(self._input.data, self._stack._min_dt)
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im
class ShrinkIntegerImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        _, nchans = get_im_dtype_and_nchan(self._input.dtype)
        self._set_props(create_im_dtype(self._stack._calc_dtype(), channels=nchans), self._input.shape)
    def _get_data(self):
        dt = self._stack._calc_dtype()
        im = self._input.data.astype(dt, copy=False)
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im


########## Commands ##########
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
    def _see_also(cls): return ('relabel','number','shrink-int')
    def __str__(self): return 'label'+('' if self._per_slice else ' - entire stack at-once')
    def execute(self, stack): stack.push(LabelImageStack(stack.pop(), self._per_slice))

class RelabelImageCommand(CommandEasy):
    _per_slice = None
    @classmethod
    def name(cls): return 'relabel'
    @classmethod
    def _desc(cls): return """
Relabels a labeled image. This makes sure that all the image is consecutively numbered (no gaps in
the numbering) and that every labeled region is contiguous. The basic idea is it does a consecutive
renumbering then runs "label" for each value. This makes sure that all labels are consecutive and
checks that every label is one connected region, splitting disjoint regions.
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
    def _see_also(cls): return ('label','number','shrink-int')
    def __str__(self): return 'relabel'+('' if self._per_slice else ' - entire stack at-once')
    def execute(self, stack): stack.push(RelabelImageStack(stack.pop(), self._per_slice))

class NumberImageCommand(CommandEasy):
    _ordered = None
    _per_slice = None
    @classmethod
    def name(cls): return 'number'
    @classmethod
    def _desc(cls): return """
Consecutively numbers an image. Every distinct pixel value is assigned a positive integer and all
pixels with that value are replaced by that number. The value 0 is only ever assigned to the
0-valued pixel (all channels equal to 0). There are no gaps in the numbering except if there is no
0-valued pixel (then there is a gap in that no pixel is assigned 0).

It can also keep the pixels in the same order (based on their sorted order, or for multi-channel
images, their lex-sorted order).
"""
    @classmethod
    def flags(cls): return ('number','renumber')
    @classmethod
    def _opts(cls): return (
        Opt('ordered', 'If true, the values are kept in the same order', Opt.cast_bool(), False),
        Opt('per-slice', 'If false, operate on the entire stack at once', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be numbered',)
    @classmethod
    def _produces(cls): return ('Numbers image stack, either int32 or int64 depending on OS',)
    @classmethod
    def _see_also(cls): return ('label','relabel','shrink-int')
    def __str__(self): return 'number%s%s'%(
        (' maintaining order' if self._ordered else ''),
        ('' if self._per_slice else ' - entire stack at-once'))
    def execute(self, stack): stack.push(ConsecutivelyNumberImageStack(stack.pop(), self._ordered, self._per_slice))

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
    def _see_also(cls): return ('label','relabel','number')
    def __str__(self): return 'shrink-int%s%s'%(
        ('' if self._min_dt is None else ' to '+im_dtype_desc(self._min_dt)),
        ('' if self._per_slice else ' - entire stack at-once'))
    def execute(self, stack): stack.push(ShrinkIntegerImageStack(stack.pop(), self._min_dt, self._per_slice))
