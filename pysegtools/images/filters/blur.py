"""Blurring Image Filters"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod
from functools import partial

from ._stack import UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from .._stack import Homogeneous
from ..types import check_image
from ...imstack import CommandEasy, Opt

__all__ = ['gaussian_blur','mean_blur','median_blur',
           'GaussianBlur','MeanBlur','MedianBlur',]

def _filter(im, flt):
    check_image(im)
    if im.ndim == 3: # Multi-channel images
        from numpy import empty
        out = empty(im.shape, dtype=im.dtype)
        for i in xrange(im.shape[2]): flt(im[...,i], output=out[...,i])
        return out
    else: # Single-channel images
        return flt(im)

def _filter3D(ims, flt):
    if ims.ndim not in (3,4): raise ValueError('Unknown image stack type')
    check_image(ims[0])
    if ims.ndim == 4: # Multi-channel images
        from numpy import empty
        out = empty(ims.shape, dtype=ims.dtype)
        for i in xrange(ims.shape[3]): flt(ims[...,i], output=out[...,i])
        return out
    else: # Single-channel images
        return flt(ims)

##### 2D #####
def gaussian_blur(im, sigma=1.0):
    """Blur an image using a Gaussian filter. Works on color types by blurring each channel seperately."""
    from scipy.ndimage.filters import gaussian_filter
    return _filter(im, partial(gaussian_filter, sigma=sigma))

def mean_blur(im, size=3):
    """Blur an image using a mean filter. Works on color types by blurring each channel seperately."""
    from scipy.ndimage.filters import uniform_filter
    return _filter(im, partial(uniform_filter, size=size))

def median_blur(im, size=3):
    """Blur an image using a median filter. Works on color types by blurring each channel seperately."""
    from scipy.ndimage.filters import median_filter
    return _filter(im, partial(median_filter, size=size))

##### 3D #####
class BlurFilterImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims, flt, per_slice):
        self._filter = flt
        self._per_slice = per_slice
        self._stack = None
        if per_slice: super(BlurFilterImageStack, self).__init__(ims, BlurFilterImageSlice)
        elif not ims.is_homogeneous: raise ValueError('Cannot blur the entire stack if it is not homogeneous')
        else:
            super(BlurFilterImageStack, self).__init__(ims, BlurStackFilterImageSlice)
            self._shape = ims.shape
            self._dtype = ims.dtype
            self._homogeneous = Homogeneous.All
    @property
    def stack(self):
        if self._stack is None:
            self._stack = _filter3D(self._ims.stack, self._filter)
            self._stack.flags.writeable = False
        return self._stack

class BlurFilterImageSlice(UnchangingFilteredImageSlice):
    #pylint: disable=protected-access
    def _get_data(self): return _filter(self._input.data, self._stack._filter)

class BlurStackFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return self._stack.stack[self._z]

class GaussianBlur(BlurFilterImageStack):
    def __init__(self, ims, sigma=1.0, per_slice=True):
        from scipy.ndimage.filters import gaussian_filter
        self._sigma = sigma
        super(GaussianBlur, self).__init__(ims, partial(gaussian_filter, sigma=sigma), per_slice)

class MeanBlur(BlurFilterImageStack):
    def __init__(self, ims, size=3, per_slice=True):
        from scipy.ndimage.filters import uniform_filter
        self._size = size
        super(MeanBlur, self).__init__(ims, partial(uniform_filter, size=size), per_slice)

class MedianBlur(BlurFilterImageStack):
    def __init__(self, ims, size=3, per_slice=True):
        from scipy.ndimage.filters import median_filter
        self._size = size
        super(MedianBlur, self).__init__(ims, partial(median_filter, size=size), per_slice)

##### Commands #####
class BlurCommand(CommandEasy):
    @classmethod
    def _opts(cls): return (
        Opt('size', 'The amount of blurring as an integer >=2', Opt.cast_int(lambda x:x>1), 3),
        Opt('per_slice', 'If false, blurs data in 3D (input must be homogeneous)', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Image to be blurred',)
    @classmethod
    def _produces(cls): return ('Blurred image',)
    @abstractmethod
    def __str__(self): pass
    @abstractmethod
    def execute(self, stack): pass
    
class GaussianBlurCommand(BlurCommand):
    _sigma = None
    _per_slice = None
    @classmethod
    def name(cls): return 'gaussian blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by using a Gaussian filter of the specified sigma.'
    @classmethod
    def flags(cls): return ('G', 'gaussian-blur')
    @classmethod
    def _opts(cls): return (
        Opt('sigma', 'The amount of blurring as a positive floating-point number', Opt.cast_float(lambda x:x>0), 1.0),
        Opt('per_slice', 'If false, blurs data in 3D (input must be homogeneous)', Opt.cast_bool(), True),
        )
    def __str__(self): return 'Gaussian blur with sigma=%f%s'%(self._sigma, '' if self._per_slice else ' (3D)')
    def execute(self, stack): stack.push(GaussianBlur(stack.pop(), self._sigma, self._per_slice))
    
class MeanBlurCommand(BlurCommand):
    _size = None
    _per_slice = None
    @classmethod
    def name(cls): return 'mean blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the average/mean of the pixels square around the pixel of the given size.'
    @classmethod
    def flags(cls): return ('mean-blur',)
    def __str__(self): return 'mean blur with size=%d%s'%(self._size, '' if self._per_slice else ' (3D)')
    def execute(self, stack): stack.push(MeanBlur(stack.pop(), self._size, self._per_slice))

class MedianBlurCommand(BlurCommand):
    _size = None
    _per_slice = None
    @classmethod
    def name(cls): return 'median blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the median of the pixels in a square around the pixel of the given size.'
    @classmethod
    def flags(cls): return ('median-blur',)
    def __str__(self): return 'median blur with size=%d%s'%(self._size, '' if self._per_slice else ' (3D)')
    def execute(self, stack): stack.push(MedianBlur(stack.pop(), self._size, self._per_slice))
