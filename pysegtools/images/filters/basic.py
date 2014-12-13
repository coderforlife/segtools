"""Basic Image Filters"""
from functools import partial
from numbers import Integral

from _stack import UnchangingFilteredImageStack, UnchangingFilteredImageSlice, FilterOption as Opt
from ..types import check_image

__all__ = ['gaussian_blur','mean_blur','median_blur','flip']

def _filter(im, flt):
    check_image(im)
    if im.ndim == 3: # Multi-channel images
        from numpy import empty
        out = empty(im.shape, dtype=im.dtype)
        for i in xrange(im.shape[2]): flt(im[:,:,i], output=out[:,:,i])
        return out
    else: # Single-channel images
        return flt(im)

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

def flip(im, direction='v'):
    """
    Flips an image either vertically (default) or horizontally (by giving an 'h'). The returned
    value is a view - not a copy.
    """
    from numpy import flipud, fliplr
    if direction not in ('v', 'h'): raise ValueError('Unsupported direction')
    return (flipud if direction == 'v' else fliplr)(im)



class BlurFilterImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims, flt):
        super(BasicFilterImageStack, self).__init__(ims, BlurFilterImageSlice)
        self._filter = flt

class BlurFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return _filter(self._input.data, self._stack._filter)

class GaussianBlurImageStack(BlurFilterImageStack):
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
    def description(self): return 'Gaussian blur with sigma=%f'%self._sigma
    def __init__(self, ims, sigma=1.0):
        from scipy.ndimage.filters import gaussian_filter
        self._sigma = sigma
        super(GaussianBlurImageStack, self).__init__(ims, partial(gaussian_filter, sigma=sigma))

class MeanBlurImageStack(BlurFilterImageStack):
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
    def description(self): return 'mean blur with size=%d'%self._size
    def __init__(self, ims, size=3):
        from scipy.ndimage.filters import uniform_filter
        self._size = size
        super(MeanBlurImageStack, self).__init__(ims, partial(uniform_filter, size=size))

class MedianBlurImageStack(BlurFilterImageStack):
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
    def description(self): return 'median blur with size=%d'%self._size
    def __init__(self, ims, size=3):
        from scipy.ndimage.filters import median_filter
        self._size = size
        super(MeanBlurImageStack, self).__init__(ims, partial(median_filter, size=size))

class FlipImageStack(UnchangingFilteredImageStack):
    @classmethod
    def _name(cls): return 'Flip'
    @classmethod
    def _desc(cls): return 'Flips the images in the image stack, either in the x, y, and z directions.'
    @classmethod
    def _flags(cls): return ('f', 'flip')
    @classmethod
    def _opts(cls): return (Opt('dir', 'The direction of the flip: x (left-to-right), y (top-to-bottom), or z (first-to-last)', Opt.cast_in('x','y','z'), 'y'),)
    @classmethod
    def _supported(cls, dtype): return True
    def description(self): return 'flip with dir=%s'%self._dir
    def __init__(self, ims, dir='y'):
        self._dir = dir
        if dir == 'z':
            slcs = [DoNothingFilterImageSlice(im,self,z) for z,im in enumerate(reversed(ims))]
        elif dir in ('x','y'):
            from numpy import flipud, fliplr
            self._flip = flipud if dir == 'y' else fliplr
            slcs = FlipFilterImageSlice
        else: raise ValueError()
        super(FlipImageStack, self).__init__(ims, slcs)
        
class DoNothingFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return self._input.data
    
class FlipFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return self._stack._flip(self._input.data)
