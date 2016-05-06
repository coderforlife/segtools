"""Filters that perform edge detection."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod

from numpy import sqrt, hypot, arctan2, pad, pi, dtype, float64, array, ones
from scipy.ndimage import convolve, binary_dilation, maximum_filter

from .threshold import otsus_multithresh
from ..types import check_image_single_channel, im2double, double2im
from ...general import delayed
from ._stack import FilteredImageStack, FilteredImageSlice
from .._stack import Homogeneous
from ...imstack import CommandEasy, Opt

__all__ = ['prewitt','sobel','scharr','canny']

def __simple_edge_detect(im, return_angle, scale, k):
    im,dt = im2double(check_image_single_channel(im), return_dtype=True)
    Gx = convolve(im, k)
    Gy = convolve(im, k.T)
    G_mag = hypot(Gx, Gy)
    G_mag *= scale
    G_mag = double2im(G_mag, dt)
    return G_mag, arctan2(Gy, Gx) if return_angle else G_mag

__prewitt_kernel = delayed(lambda:(1/sqrt(10), array([[-1,0,1],[-1,0,1],[-1,0,1]])), tuple)
def prewitt(im, return_angle=False):
    """
    Perform Prewitt edge detection on the image.

    Returns the magnitude of the gradient (sqrt(gx^2+gy^2)) scaled to the max range of the input
    image data type. If return_angle is True then the angle of the gradient is returned as well
    (from -pi to pi with 0 to the right and clockwise).
    """
    return __simple_edge_detect(im, return_angle, *__prewitt_kernel)

__scharr_kernel = delayed(lambda:(1/sqrt(356), array([[-3,0,3],[-10,0,10],[-3,0,3]])), tuple)
def scharr(im, return_angle=False):
    """
    Perform Scharr edge detection on the image.

    Returns the magnitude of the gradient (sqrt(gx^2+gy^2)) scaled to the max range of the input
    image data type. If return_angle is True then the angle of the gradient is returned as well
    (from -pi to pi with 0 to the right and clockwise).
    """
    return __simple_edge_detect(im, return_angle, *__scharr_kernel)

__sobel_kernels = delayed(lambda:{
        3: (1/sqrt(20),   array([[-1,0,1],[-2,0,2],[-1,0,1]])),
        5: (1/sqrt(2628), array([[-1,-2,0,2,1],[-4,-8,0,8,4],[-6,-12,0,12,6],[-4,-8,0,8,4],[-1,-2,0,2,1]])),
        }, dict)
def sobel(im, size=3, return_angle=False):
    """
    Perform Sobel edge detection on the image. The size of the Sobel kernel can be either 3 or 5.

    Returns the magnitude of the gradient (sqrt(gx^2+gy^2)) scaled to the max range of the input
    image data type. If return_angle is True then the angle of the gradient is returned as well
    (from -pi to pi with 0 to the right and clockwise).
    """
    if size not in __sobel_kernels: raise ValueError('unsupported filter size')
    return __simple_edge_detect(im, return_angle, *__sobel_kernels[size])


__canny_dir = delayed(lambda:(
        array([[0,0,0],[1,0,1],[0,0,0]], dtype=bool), # W/E
        array([[1,0,0],[0,0,0],[0,0,1]], dtype=bool), # NW/SE
        array([[0,1,0],[0,0,0],[0,1,0]], dtype=bool), # N/S
        array([[0,0,1],[0,0,0],[1,0,0]], dtype=bool), # NE/SW
    ), tuple)
def canny(im, thresh=None, size=3, quick=False):
    """
    Perform Canny edge detection on the image. The size of the Sobel kernel internally used can be
    either 3 or 5. This does not perform any blurring on the image. Returns a bool/logical image.

    Canny filtering uses two thresholds, the value for which should depend on the filter size,
    quick or not, and the image itself. If these are not given they are calculated using Otsu's
    method.

    If quick is set to True, some shortcuts are taken to make the filter faster but slightly less
    accurate. When quick, the gradient magnitude is calculated using |gx|+|gy| instead of
    sqrt(gx^2+gy^2) and picks nearest neighbor instead of interpolating the angle for non-maximal
    supression. These two steps are each about twice as fast when quick is True (the overall
    process is not twice as fast though as the other steps still take the same amount of time).
    """
    if size not in __sobel_kernels: raise ValueError('unsupported filter size')
    _, k = __sobel_kernels[size]

    if thresh is not None and (len(thresh) != 2 or thresh[0] >= thresh[1]):
        raise ValueError('unsupported thresholds')
    
    im = im2double(check_image_single_channel(im))

    # Sobel Filter
    Gx = convolve(im, k)
    Gy = convolve(im, k.T)
    G_mag = (abs(Gx) + abs(Gy)) if quick else hypot(Gx, Gy) # no scaling here is necessary
    G_dir = arctan2(Gy, Gx)

    # Non-Maximal Suppression
    # TODO: these could be done MUCH faster with less memory in Cython
    pi_4_inv = 4/pi
    if quick:
        # Just take the closest neighbor
        q = (pi_4_inv*G_dir+4).round().astype(int)%4
        local_max  = (G_mag > maximum_filter(G_mag, footprint=__canny_dir[0], mode='constant')) & (q == 0)
        local_max |= (G_mag > maximum_filter(G_mag, footprint=__canny_dir[1], mode='constant')) & (q == 1)
        local_max |= (G_mag > maximum_filter(G_mag, footprint=__canny_dir[2], mode='constant')) & (q == 2)
        local_max |= (G_mag > maximum_filter(G_mag, footprint=__canny_dir[3], mode='constant')) & (q == 3)
    else:
        # Interpolate between the two closest neighbors
        rem = pi_4_inv*G_dir+4
        q = rem.astype(int)
        rem -= q # how far along we are to the next quarter (clockwise) from the quarter start (0-1)
        q %= 4
        rem_1 = 1 - rem
        G_mag_pad = pad(G_mag, 1, 'constant')
        s = [G_mag_pad[1:-1,:-2], G_mag_pad[:-2,:-2], G_mag_pad[:-2,1:-1], G_mag_pad[:-2,2:],
             G_mag_pad[1:-1,2:],  G_mag_pad[2:,2:],   G_mag_pad[2:,1:-1],  G_mag_pad[2:,:-2]]
        local_max  = ((G_mag > (s[0]*rem_1+s[1]*rem)) & (G_mag > (s[4]*rem_1+s[5]*rem))) & (q == 0)
        local_max |= ((G_mag > (s[1]*rem_1+s[2]*rem)) & (G_mag > (s[5]*rem_1+s[6]*rem))) & (q == 1)
        local_max |= ((G_mag > (s[2]*rem_1+s[3]*rem)) & (G_mag > (s[6]*rem_1+s[7]*rem))) & (q == 2)
        local_max |= ((G_mag > (s[3]*rem_1+s[4]*rem)) & (G_mag > (s[7]*rem_1+s[0]*rem))) & (q == 3)
    G_mag *= local_max

    # Hysteresis Thresholding
    if thresh is None:
        thresh = otsus_multithresh(G_mag, 3, local_max)
    yes   = G_mag >= thresh[1]
    maybe = G_mag >= thresh[0]
    return binary_dilation(yes, ones((3,3), bool), -1, maybe) # TODO: should structure be generate_binary_structure(2, 1)


########## Image Stacks ##########
class EdgeDetectionImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def __init__(self, im, stack, z):
        super(EdgeDetectionImageSlice, self).__init__(im, stack, z)
        self._set_props(dtype(float64), None)
    def _get_props(self): self._set_props(None, self._input.shape)
    def _get_data(self):
        return __simple_edge_detect(self._input.data, False, self._stack._scale, self._stack._k)

class PrewittImageStack(FilteredImageStack):
    def __init__(self, ims):
        self._scale, self._k = __prewitt_kernel #pylint: disable=unpacking-non-sequence
        self._dtype, self._homogeneous = dtype(float64), Homogeneous.DType
        super(PrewittImageStack, self).__init__(ims, EdgeDetectionImageSlice)

class ScharrImageStack(FilteredImageStack):
    def __init__(self, ims):
        self._scale, self._k = __scharr_kernel #pylint: disable=unpacking-non-sequence
        self._dtype, self._homogeneous = dtype(float64), Homogeneous.DType
        super(ScharrImageStack, self).__init__(ims, EdgeDetectionImageSlice)

class SobelImageStack(FilteredImageStack):
    def __init__(self, ims, size=3):
        if size not in __sobel_kernels: raise ValueError('unsupported filter size')
        self._scale, self._k = __sobel_kernels[size]
        self._dtype, self._homogeneous = dtype(float64), Homogeneous.DType
        super(SobelImageStack, self).__init__(ims, EdgeDetectionImageSlice)

class CannyImageStack(FilteredImageStack):
    def __init__(self, ims, thresh=None, size=3, quick=False):
        if size not in __sobel_kernels: raise ValueError('unsupported filter size')
        if thresh is not None and (len(thresh) != 2 or thresh[0] >= thresh[1]):
            raise ValueError('unsupported thresholds')
        self._thresh = thresh
        self._size = size
        self._quick = bool(quick)
        self._dtype, self._homogeneous = dtype(bool), Homogeneous.DType
        super(CannyImageStack, self).__init__(ims, CannyImageSlice)

class CannyImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def __init__(self, im, stack, z):
        super(CannyImageSlice, self).__init__(im, stack, z)
        self._set_props(dtype(bool), None)
    def _get_props(self): self._set_props(None, self._input.shape)
    def _get_data(self):
        return canny(self._input.data, self._stack._thresh, self._stack._size, self._stack._quick)


########## Commands ##########
class __SimpleEdgeDetectionCommand(CommandEasy):
    @classmethod
    def _consumes(cls): return ('Grayscale image stack to be edge detected',)
    @classmethod
    def _produces(cls): return ('Gradient magnitudes, scaled to the input image data type',)
    @classmethod
    def _see_also(cls): return tuple(x for x in ('prewitt','scharr','sobel','canny') if x != cls.name())
    def __str__(self): return self.__class__.name()
    @abstractmethod
    def execute(self, stack): pass

class PrewittCommand(__SimpleEdgeDetectionCommand):
    @classmethod
    def name(cls): return 'prewitt'
    @classmethod
    def _desc(cls): return "Prewitt Edge Detection"
    def execute(self, stack): stack.push(PrewittImageStack(stack.pop()))

class ScharrCommand(__SimpleEdgeDetectionCommand):
    @classmethod
    def name(cls): return 'scharr'
    @classmethod
    def _desc(cls): return "Scharr Edge Detection"
    def execute(self, stack): stack.push(ScharrImageStack(stack.pop()))

class SobelCommand(__SimpleEdgeDetectionCommand):
    _size = None
    @classmethod
    def name(cls): return 'sobel'
    @classmethod
    def _desc(cls): return "Sobel Edge Detection"
    @classmethod
    def _opts(cls): return (
        Opt('size', 'The size of the Sobel filter, either 3 or 5', Opt.cast_int(lambda x: x in (3,5)), 3),
        )
    def __str__(self): return 'sobel-%d' % self._size
    def execute(self, stack): stack.push(SobelImageStack(stack.pop(), self._size))

class CannyCommand(__SimpleEdgeDetectionCommand):
    _thresh = None
    _size = None
    _quick = None
    @classmethod
    def name(cls): return 'canny'
    @classmethod
    def _desc(cls): return """
Canny Edge Detection

This does not perform any blurring on the image.

Canny filtering uses two thresholds, the value for which should depend on the filter size, quick or
not, and the image itself. If these are not given they are calculated using Otsu's method which is
more likely the be accurate.

If quick is set to true, some shortcuts are taken to make the filter faster but slightly less
accurate. When quick, the gradient magnitude is calculated using |gx|+|gy| instead of
sqrt(gx^2+gy^2) and picks nearest neighbor instead of interpolating the angle.
"""
    @classmethod
    def _opts(cls): return (
        Opt('thresh', 'The two threshold values as a pair of integers/floats or \'auto\'',
            Opt.cast_or('auto', Opt.cast_tuple_of(Opt.cast_number(), 2, 2)), 'auto'),
        Opt('size', 'The size of the internally used Sobel filter, either 3 or 5', Opt.cast_int(lambda x: x in (3,5)), 3),
        Opt('quick', 'If true processing is done faster with a simplified gradient magnitude calculation and angle interpolation', Opt.cast_bool(), False),
        )
    @classmethod
    def _produces(cls): return ('Edge mask - a logical/bool image',)
    def __str__(self):
        s = ('canny' if self._size == 3 else 'canny-5') + (' (quick)' if self._quick else '')
        if self._thresh != 'auto': s += ' with thresholds %s and %s' % self._thresh
        return s
    def execute(self, stack):
        stack.push(CannyImageStack(stack.pop(), None if self._thresh == 'auto' else self._thresh, self._size, self._quick))
