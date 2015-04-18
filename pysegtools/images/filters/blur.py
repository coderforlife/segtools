"""Blurring Image Filters"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import partial

from ._stack import UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from ..types import check_image
from ...imstack import CommandEasy, Opt

__all__ = ['gaussian_blur','mean_blur','median_blur',
           'GaussianBlur','MeanBlur','MedianBlur',]

def _filter(im, flt):
    check_image(im)
    if im.ndim == 3: # Multi-channel images
        from numpy import empty
        out = empty(im.shape, dtype=im.dtype)
        for i in xrange(im.shape[2]): flt(im[:,:,i], output=out[:,:,i])
        return out
    else: # Single-channel images
        return flt(im)

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
    return _filter(im, partial(median_blur, size=size))

##### 3D #####
class BlurFilterImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims, flt):
        super(BasicFilterImageStack, self).__init__(ims, BlurFilterImageSlice)
        self._filter = flt

class BlurFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self):
        return _filter(self._input.data, self._stack._filter)

class GaussianBlur(BlurFilterImageStack):
    def __init__(self, ims, sigma=1.0):
        from scipy.ndimage.filters import gaussian_filter
        self._sigma = sigma
        super(GaussianBlur, self).__init__(ims, partial(gaussian_filter, sigma=sigma))

class MeanBlur(BlurFilterImageStack):
    def __init__(self, ims, size=3):
        from scipy.ndimage.filters import uniform_filter
        self._size = size
        super(MeanBlur, self).__init__(ims, partial(uniform_filter, size=size))

class MedianBlur(BlurFilterImageStack):
    def __init__(self, ims, size=3):
        from scipy.ndimage.filters import median_filter
        self._size = size
        super(MedianBlur, self).__init__(ims, partial(median_filter, size=size))

##### Commands #####
class BlurCommand(CommandEasy):
    @classmethod
    def _opts(cls): return (
        Opt('size', 'The amount of blurring as an integer >=2', Opt.cast_int(lambda x:x>1)),
        )
    @classmethod
    def _consumes(cls, dtype): return ('Image to be blurred',)
    @classmethod
    def _produces(cls, dtype): return ('Blurred image',)
    
class GaussianBlurCommand(BlurCommand):
    @classmethod
    def name(cls): return 'gaussian blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by using a Gaussian filter of the specified sigma.'
    @classmethod
    def flags(cls): return ('G', 'gaussian-blur')
    @classmethod
    def _opts(cls): return (
        Opt('sigma', 'The amount of blurring as a positive floating-point number', Opt.cast_float(lambda x:x>0)),
        )
    def __str__(self): return 'Gaussian blur with sigma=%f'%self._sigma
    def execute(self, stack): stack.push(GaussianBlurImageStack(stack.pop(), self._sigma))
    
class MeanBlurCommand(BlurCommand):
    @classmethod
    def name(cls): return 'mean blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the average/mean of the pixels square around the pixel of the given size.'
    @classmethod
    def flags(cls): return ('mean-blur',)
    def __str__(self): return 'mean blur with size=%d'%self._size
    def execute(self, stack): stack.push(MeanBlurImageStack(stack.pop(), self._size))

class MedianBlurCommand(BlurCommand):
    @classmethod
    def name(cls): return 'median blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the median of the pixels in a square around the pixel of the given size.'
    @classmethod
    def flags(cls): return ('median-blur',)
    def __str__(self): return 'median blur with size=%d'%self._size
    def execute(self, stack): stack.push(MedianBlurImageStack(stack.pop(), self._size))
