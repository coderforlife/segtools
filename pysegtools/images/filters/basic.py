"""Basic Image Filters"""
from functools import partial
from numbers import Integral

from _stack import UnchangingFilteredImageStack, UnchangingFilteredImageSlice, FilterOption as Opt
from ..types import im_standardize_dtype, im_raw_dtype, IM_COLOR_TYPES, IM_RANGED_TYPES

__all__ = ['gaussian_blur','mean_blur','median_blur']

def _filter(im, flt):
    im = im_standardize_dtype(im)
    if im.dtype in IM_COLOR_TYPES: # Multi-channel images
        from numpy import empty
        im = im_raw_dtype(im)
        out = empty(im.shape, dtype=im.dtype)
        for i in xrange(im.shape[2]): flt(im[:,:,i], output=out[:,:,i])
        return im_standardize_dtype(out)
    elif im.dtype in IM_RANGED_TYPES: # Single-channel images
        return flt(im)
    else: raise ValueError('Unsupported image type')

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

class BasicFilterImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims, flt):
        super(BasicFilterImageStack, self).__init__(ims, BasicFilterImageSlice)
        self._filter = flt

class BasicFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return _filter(self._input.data, self._stack._filter)

class GaussianBlurImageStack(BasicFilterImageStack):
    @classmethod
    def _name(cls): return 'Gaussian Blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by using a Gaussian filter of the specified sigma.'
    @classmethod
    def _flags(cls): return ('G', 'gaussian-blur')
    @classmethod
    def _opts(cls): return (Opt('sigma', 'The amount of blurring as a positive floating-point number', Opt.cast_float(lambda x:x>0)),)
    @classmethod
    def _supported(cls, dtype): return dtype in IM_COLOR_TYPES+IM_RANGED_TYPES
    def __str__(self): return 'Gaussian blur with sigma=%f'%self._sigma
    def __init__(self, ims, sigma=1.0):
        from scipy.ndimage.filters import gaussian_filter
        self._sigma = sigma
        super(GaussianBlurImageStack, self).__init__(ims, partial(gaussian_filter, sigma=sigma))

class MeanBlurImageStack(BasicFilterImageStack):
    @classmethod
    def _name(cls): return 'Mean Blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the average/mean of the pixels square around the pixel of the given size.'
    @classmethod
    def _flags(cls): return ('mean-blur',)
    @classmethod
    def _opts(cls): return (Opt('size', 'The amount of blurring as an integer >=2', Opt.cast_int(lambda x:x>1)),)
    @classmethod
    def _supported(cls, dtype): return dtype in IM_COLOR_TYPES+IM_RANGED_TYPES
    def __str__(self): return 'mean blur with size=%d'%self._size
    def __init__(self, ims, size=3):
        from scipy.ndimage.filters import uniform_filter
        self._size = size
        super(MeanBlurImageStack, self).__init__(ims, partial(uniform_filter, size=size))

class MedianBlurImageStack(BasicFilterImageStack):
    @classmethod
    def _name(cls): return 'Median Blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the median of the pixels in a square around the pixel of the given size.'
    @classmethod
    def _flags(cls): return ('median-blur',)
    @classmethod
    def _opts(cls): return (Opt('size', 'The amount of blurring as an integer >=2', Opt.cast_int(lambda x:x>1)),)
    @classmethod
    def _supported(cls, dtype): return dtype in IM_COLOR_TYPES+IM_RANGED_TYPES
    def __str__(self): return 'median blur with size=%d'%self._size
    def __init__(self, ims, size=3):
        from scipy.ndimage.filters import median_filter
        self._size = size
        super(MeanBlurImageStack, self).__init__(ims, partial(median_filter, size=size))
