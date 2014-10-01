"""Blurring image filters"""
from functools import partial

__all__ = ['gaussian_blur','mean_blur','median_blur']

def _filter(im, flt):
    im = im_standardize_dtype(im)
    if im.dtype in IM_COLOR_TYPES: # Multi-channel images
        from numpy import empty
        im = im_raw_dtype(im)
        out = empty(im.shape, dtype=im.dtype)
        for i in xrange(im.shape[2]): flt(im[:,:,i], output=out[:,:,i])
        return out
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
