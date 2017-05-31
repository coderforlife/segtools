"""Blurring Image Filters"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod
from functools import partial

from ._stack import UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from .._stack import Homogeneous
from ..types import check_image, im2double, double2im
from ...imstack import CommandEasy, Opt

__all__ = ['gaussian_blur','mean_blur','median_blur','max_blur','min_blur','anisotropic_diffusion',
           'GaussianBlur','MeanBlur','MedianBlur','MaxBlur','MinBlur','AnisotropicDiffusion']

def _filter(im, flt):
    check_image(im)
    if im.ndim == 3: # Multi-channel images
        from numpy import empty
        out = empty(im.shape, dtype=im.dtype)
        for i in xrange(im.shape[2]): flt(im[...,i], output=out[...,i])
        return out
    return flt(im) # Single-channel images

def _filter3D(ims, flt):
    if ims.ndim not in (3,4): raise ValueError('Unknown image stack type')
    check_image(ims[0])
    if ims.ndim == 4: # Multi-channel images
        from numpy import empty
        out = empty(ims.shape, dtype=ims.dtype)
        for i in xrange(ims.shape[3]): flt(ims[...,i], output=out[...,i])
        return out
    return flt(ims) # Single-channel images

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

def max_blur(im, size=3):
    """Blur an image using a max filter. Works on color types by blurring each channel seperately."""
    from scipy.ndimage.filters import maximum_filter
    return _filter(im, partial(maximum_filter, size=size))

def min_blur(im, size=3):
    """Blur an image using a min filter. Works on color types by blurring each channel seperately."""
    from scipy.ndimage.filters import minimum_filter
    return _filter(im, partial(minimum_filter, size=size))

def _exp(di):
    """Computes exp(-im_grad**2) for anisotropic diffusion"""
    from numpy import exp, negative, multiply
    return exp(negative(multiply(di, di, di), di), di)
def _inv(di):
    """Computes 1/(1+im_grad**2) for anisotropic diffusion"""
    from numpy import divide, add, multiply
    return divide(1, add(1, multiply(di, di, di), di), di)
def _anisotropic_diffusion_2(niters=15, dt=0.15, scale=10, diff_coeff=_inv):
    if   diff_coeff == 'inv': diff_coeff = _inv
    elif diff_coeff == 'exp': diff_coeff = _exp
    if dt > 0.25: raise ValueError('dt parameter must be <=0.25 for numerical stability')
    def anisotropic_diffusion_2(im, output=None):
        from numpy import empty, multiply, divide, subtract, pad
        im,t = im2double(pad(im, 1, str('edge')), return_dtype=True) # add boundary conditions and convert to double
        # 8-connected pixels
        ne,se,nw,sw = im[:-2,:-2],  im[2:,:-2],  im[:-2,2:],   im[2:,2:]
        n,s,e,w     = im[:-2,1:-1], im[2:,1:-1], im[1:-1,:-2], im[1:-1,2:]
        im = im[1:-1,1:-1] # central region
        di,di_scld,A = empty(im.shape), empty(im.shape), empty(im.shape)
        for _ in xrange(niters):
            subtract(ne, im, di); A  = multiply(diff_coeff(divide(di, scale, di_scld)), di, A)
            subtract(se, im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, di)
            subtract(nw, im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, di)
            subtract(sw, im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, di)
            A *= 0.5 # diagonals are weighted by half
            subtract(n,  im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, di)
            subtract(s,  im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, di)
            subtract(e,  im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, di)
            subtract(w,  im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, di)
            A *= dt
            im += A
        if output is None: return double2im(im, t)
        output[:] = double2im(im, output.dtype)
        return output
    return anisotropic_diffusion_2

def anisotropic_diffusion(im, niters=15, dt=0.15, scale=10, diff_coeff=_inv):
    """
    Blur an image using anisotropic diffusion, based on Perona and Malik (1990). Works on color
    types by blurring each channel seperately.

    Inputs:
        niters      the number of iterations to perform, defaults to 10
        dt          the integration constant, the parameter lambda in the paper, defaults to 0.15
        scale       the scaling to perform, the parameter K in the paper, defaults to 10
        diff_coeff  a function that calculates the diffusion coefficient, it is given the pre-scaled
                    image gradient (so already divided by K), similar to g in the original paper,
                    defaults to 1/(1+im_grad**2); if it returns a constant this becomes Gaussian
                    blurring
    """
    return _filter(im, _anisotropic_diffusion_2(niters, dt, scale, diff_coeff))


##### 3D #####
class BlurFilterImageStack(UnchangingFilteredImageStack):
    _stack = None
    def __init__(self, ims, flt, per_slice):
        self._filter = flt
        self._per_slice = per_slice
        if per_slice: super(BlurFilterImageStack, self).__init__(ims, BlurFilterImageSlice)
        elif not ims.is_homogeneous: raise ValueError('Cannot blur the entire stack if it is not homogeneous')
        else:
            super(BlurFilterImageStack, self).__init__(ims, BlurStackFilterImageSlice)
            self._shape = ims.shape
            self._dtype = ims.dtype
            self._homogeneous = Homogeneous.All
    @property
    def stack(self):
        if self._per_slice: return super(BlurFilterImageStack, self).stack
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

class MaxBlur(BlurFilterImageStack):
    def __init__(self, ims, size=3, per_slice=True):
        from scipy.ndimage.filters import maximum_filter
        self._size = size
        super(MaxBlur, self).__init__(ims, partial(maximum_filter, size=size), per_slice)

class MinBlur(BlurFilterImageStack):
    def __init__(self, ims, size=3, per_slice=True):
        from scipy.ndimage.filters import minimum_filter
        self._size = size
        super(MinBlur, self).__init__(ims, partial(minimum_filter, size=size), per_slice)

def _anisotropic_diffusion_3(niters=5, dt=0.075, scale=35, diff_coeff=_inv, zscale=1):
    # zscale = z-spacing / x-spacing
    if   diff_coeff == 'inv': diff_coeff = _inv
    elif diff_coeff == 'exp': diff_coeff = _exp
    if dt > 0.25: raise ValueError('dt parameter must be <=0.15 for numerical stability')
    def anisotropic_diffusion_3(im, output=None):
        from numpy import empty, multiply, divide, subtract, pad
        im,t = im2double(pad(im, 1, str('edge')), return_dtype=True) # add boundary conditions and convert to double
        # z = 0, n = -1, and p = +1
        z,p,n = slice(1,-1), slice(2,None), slice(None,-2)
        # 26-connected pixels
        groups = (
            # all three dimensions are non-zero
            (1/(2+zscale*zscale), im[n,n,n],(im[p,n,n],im[n,p,n],im[p,p,n],im[n,n,p],im[p,n,p],im[n,p,p],im[p,p,p])),
            # z dimension is non-zero and one dimension of x/y is non-zero
            (1/(1+zscale*zscale), im[z,n,n],(im[z,p,n],im[z,n,p],im[z,p,p],im[n,z,n],im[p,z,n],im[n,z,p],im[p,z,p])),
            # z dimension is zero and both x and y dimensions are non-zero
            (1/2,                 im[n,n,z],(im[p,n,z],im[n,p,z],im[p,p,z])),
            # z dimension is non-zero and both x and y dimensions are zero
            (1/(zscale*zscale),   im[z,z,n],(im[z,z,p],)))
        # z dimension is zero and one dimension of x/y is non-zero (factor of 1 - can be added on directly)
        fnl = (im[n,z,z],im[p,z,z],im[z,n,z],im[z,p,z])
        im = im[z,z,z] # central region
        di,di_scld,A,B = empty(im.shape), empty(im.shape), empty(im.shape), empty(im.shape)
        for _ in xrange(niters):
            for i,(factor,first,grp) in enumerate(groups):
                #print(first.shape, c.shape, di.shape, di_scld.shape, B.shape)
                subtract(first, im, di); B  = multiply(diff_coeff(divide(di, scale, di_scld)), di, B)
                for g in grp:
                    subtract(g, im, di); B += multiply(diff_coeff(divide(di, scale, di_scld)), di, B)
                B *= factor
                if i != 0: A += B
                else: A,B = B,A
            for g in fnl: subtract(g, im, di); A += multiply(diff_coeff(divide(di, scale, di_scld)), di, A)
            A *= dt
            im += A
        if output is None: return double2im(im, t)
        output[:] = double2im(im, output.dtype)
        return output
    return anisotropic_diffusion_3
class AnisotropicDiffusion(BlurFilterImageStack):
    def __init__(self, ims, niters=15, dt=0.15, scale=10, diff_coeff=_inv, zscale=1, per_slice=True):
        self._niters = niters
        self._scale = scale
        self._dt = dt
        self._diff_coeff = diff_coeff
        self._zscale = zscale
        f = _anisotropic_diffusion_2(niters, dt, scale, diff_coeff) if per_slice else \
            _anisotropic_diffusion_3(niters, dt, scale, diff_coeff, zscale)
        super(AnisotropicDiffusion, self).__init__(ims, f, per_slice)
        

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
    @classmethod
    def _see_also(cls): return ('anisotropic diffusion','mean','median','max','min','convolve')
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
    @classmethod
    def _see_also(cls): return ('Gaussian blur','anisotropic diffusion','median','max','min','convolve')
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
    @classmethod
    def _see_also(cls): return ('Gaussian blur','anisotropic diffusion','mean','max','min','convolve')
    def __str__(self): return 'median blur with size=%d%s'%(self._size, '' if self._per_slice else ' (3D)')
    def execute(self, stack): stack.push(MedianBlur(stack.pop(), self._size, self._per_slice))

class MaxBlurCommand(BlurCommand):
    _size = None
    _per_slice = None
    @classmethod
    def name(cls): return 'max blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the minimum of the pixels in a square around the pixel of the given size.'
    @classmethod
    def flags(cls): return ('max',)
    @classmethod
    def _see_also(cls): return ('Gaussian blur','anisotropic diffusion','mean','median','min','convolve')
    def __str__(self): return 'max blur with size=%d%s'%(self._size, '' if self._per_slice else ' (3D)')
    def execute(self, stack): stack.push(MaxBlur(stack.pop(), self._size, self._per_slice))

class MinBlurCommand(BlurCommand):
    _size = None
    _per_slice = None
    @classmethod
    def name(cls): return 'min blur'
    @classmethod
    def _desc(cls): return 'Blurs the image by setting the value of a pixel equal to the maximum of the pixels in a square around the pixel of the given size.'
    @classmethod
    def flags(cls): return ('min',)
    @classmethod
    def _see_also(cls): return ('Gaussian blur','anisotropic diffusion','mean','median','max','convolve')
    def __str__(self): return 'min blur with size=%d%s'%(self._size, '' if self._per_slice else ' (3D)')
    def execute(self, stack): stack.push(MinBlur(stack.pop(), self._size, self._per_slice))

class AnisotropicDiffusionCommand(BlurCommand):
    _niters = None
    _scale = None
    _dt = None
    _diff_coeff = None
    _zscale = None
    _per_slice = None
    @classmethod
    def name(cls): return 'anisotropic diffusion'
    @classmethod
    def _desc(cls): return """
Blur an image using anisotropic diffusion, based on Perona and Malik (1990). Similar to Gaussian
blurring except that it lessens the blurring around edges, thus reducing noise in the image but
preserving edges.

The diffusion coefficient can be calculated in two different ways: 
 * inv: 1/(1+(||grad im||/scale)^2) 
        favors wide over smaller regions 
 * exp: exp(-(||grad im||/scale)^2) 
        favors high-contrast over low-contrast edges
"""
    @classmethod
    def flags(cls): return ('anisotropic-diffusion','anisotropic-diff','anisotropic')
    @classmethod
    def _opts(cls): return (
        Opt('niters',     'The number of iterations to perform', Opt.cast_int(lambda x:x>0), 15),
        Opt('dt',         'The integration constant, the parameter lambda in the paper', Opt.cast_float(lambda x:0<x<=0.25), 0.15),
        Opt('scale',      'The scaling to perform, the parameter K in the paper', Opt.cast_float(lambda x:x>0), 10),
        Opt('diff_coeff', 'Diffusion coefficient function', Opt.cast_or('inv','exp'), 'inv'),
        Opt('zscale',     'Only used if per_slice is false, determines the relative spacing of the xy planes with the z dimension', Opt.cast_float(lambda x:x>0), 1),
        Opt('per_slice',  'If false, blurs data in 3D (input must be homogeneous), if this is false it is likely that dt, niters, and scale should not be their defaults', Opt.cast_bool(), True),
        )
    @classmethod
    def _see_also(cls): return ('Gaussian blur','mean','median','max','min')
    def __str__(self):
        return 'anisotropic diffusion with niters=%d, dt=%.2f, scale=%.2f, diff_coeff=%s%s'%(
            self._niters, self._dt, self._scale, self._diff_coeff,
            '' if self._per_slice else (', zscale=%.2f (3D)'%self._zscale))
    def execute(self, stack):
        stack.push(AnisotropicDiffusion(stack.pop(), self._niters, self._dt, self._scale, self._diff_coeff, self._zscale, self._per_slice))
