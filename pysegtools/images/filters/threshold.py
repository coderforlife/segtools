"""Filters that threshold the image."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import izip, islice
from collections import Sequence
from math import ceil, floor
from numbers import Real

from numpy import zeros, array, arange, dtype, uint8, intp, tile, linspace
from numpy import where, triu_indices, nan_to_num, seterr
from scipy.ndimage import histogram, generate_binary_structure, binary_dilation

from ..types import check_image, get_im_dtype_and_nchan, get_dtype_min_max
from ._stack import FilteredImageStack, FilteredImageSlice
from .._stack import Homogeneous
from ...imstack import CommandEasy, Opt
from ...general import delayed, pairwise

__all__ = ['otsus_multithresh', 'threshold', 'hysteresis_threshold'] # , 'multithreshold'

def otsus_multithresh(im, levels=2, mask=None, output_metric=False):
    """
    Calculate thresholds for an image to split the image into the specified number of levels using
    the multilevel version of Otsu's method. One less threshold than levels is returned. The
    default is 2 levels (a single threshold, making a binary image).

    If mask is provided, only those pixels within the mask are used to calculate the threshold.
    Instead of an image, a tuple of histogram counts/density and left-hand-side bin values can be
    given instead as the first argument. In this case mask cannot be given.

    If output metric is given, a value from 0 to 1 is returned indicating how well the thresholds
    represent the given image (with 1 being the best representation).

    References:
        Otsu N (1979). "A threshold selection method from gray-level histograms". IEEE Trans Sys,
        Man, Cyber 9 (1): 62-66. doi:10.1109/TSMC.1979.4310076

        Liao P-S, Chen T-S, Chung P-C (2001). "A Fast Algorithm for Multilevel Thresholding". J
        Inf Sci Eng 17 (5): 713-727.
    """
    levels = int(levels)
    if levels < 2 or levels > 32: raise ValueError('levels must be an integer 2-32')
    # levels == 2 means just a single threshold (not-multilevel), still returned as a tuple though

    if isinstance(im, tuple):
        # TODO: is this accurate?
        if mask is not None: raise ValueError('mask cannot be given with histogram data')
        p, bins = im
        if len(p) != len(bins): raise ValueError('histogram counts and values must be the same length')
        if p.dtype.kind in 'biu': p = p.astype(intp, copy=False)
        mn, mx = bins[0], bins[-1]
        scale = (mx-mn)/(len(bins)-1)
        dt = bins.dtype
        
    else:
        if im.dtype.kind == 'c': raise ValueError('complex types not accepted')
        if mask is not None: im = im[mask]

        # Calculate pdf
        mn = im.min()
        mx = im.max()
        if mn == mx:
            if levels == 2: return (mn,0.0) if output_metric else mn
            else: return getDegenerateThresholds([mn], levels-1) # TODO
        p, bins = histogram(im, mn, mx, 256).astype(intp), arange(256)
        scale = (mx-mn)/(len(bins)-1)
        dt = im.dtype

    # Calculate omega and mu
    # Note: for simplicity we actually calculate everything as times npix (so p is actually counts
    # instead of pdf)
    # mu_t is still scaled by npix since that is necessary to make everything calculate right
    omega = p.cumsum()
    mu = (p * bins).cumsum()
    mu_t = mu[-1] / omega[-1]

    thresh, sigma_max = __otsus_multithresh(omega, mu, mu_t, levels - 1)
    
    thresh = mn+array(thresh)*scale
    if dt.kind in 'biu': thresh = (thresh+0.000000001).round()
    thresh = tuple(thresh.astype(dt))
    if output_metric:
        x = bins-mu_t
        metric = sigma_max/(p*x*x).sum()
        if levels == 2: metric *= omega[-1]
        return thresh, metric
    return thresh

def __otsus_multithresh(omega, mu, mu_t, N):
    if N == 1:
        npix = omega[-1]
        omega = omega[:-1] # including these values causes divide-by-zero
        mu = mu[:-1]
        x = mu_t * omega - mu
        sigma_b_2 = x*x / (omega * (npix - omega))
        sigma_max = sigma_b_2.max()
        return [where(sigma_b_2 == sigma_max)[0].mean()], sigma_max
        
    elif N == 2:
        # This generates all possible omegas in the strict upper triangle 
        # We only want to use the strict upper triangle to force lower threshold is less than higher threshold
        # Also filter out all places where omega1 is 0 due to division by zero for mu_1_t
        npix = omega[-1]
        nbins_1 = len(omega)-1
        omega = omega[:,None] # turn these into column vectors for broadcasting
        mu = mu[:,None]

        tri = triu_indices(nbins_1, 1)
        omega_1 = omega.T-omega
        valid = omega_1[tri] != 0
        valid = (tri[0][valid], tri[1][valid])
        
        omega_1 = omega_1[valid]
        mu_1_t = mu_t-(mu.T-mu)[valid]/omega_1
        omega_0 = tile(omega, (1,nbins_1))[valid]
        mu_0_t = tile(mu_t-mu/omega, (1,nbins_1))[valid]

        # Calculate the 3 terms for the 3 levels
        H1 = omega_0*mu_0_t*mu_0_t
        H2 = omega_1*mu_1_t*mu_1_t
        x = omega_0*mu_0_t + omega_1*mu_1_t
        H3 = x*x/(npix - (omega_0+omega_1))
        
        # Calculate sigma_b**2, find the max, and get the original indices from those positions
        sigma_b_2 = H1 + H2 + H3
        sigma_max = sigma_b_2.max()
        sigma_maxes = sigma_b_2 == sigma_max
        return [valid[0][sigma_maxes].mean(), valid[1][sigma_maxes].mean()], sigma_max
        
    # Otherwise N > 2 - operate recursively to limit the range of values we look over
    # This does give accurate results but gets really slow above N = 5 or so and also starts
    # taking lots of memory.
    # TODO: use this accurate method until N = 4 or 5 or 6, then switch over to a
    # minimization-optimization method using the results from this to seed the search which by
    # N = 4 means that only about 64 numbers in each range need to be minimized over, increasing
    # its accuracy.
    nbins = len(omega)
    
    # Recurse with N-1
    prev_thresh, _ = __otsus_multithresh(omega, mu, mu_t, N-1)
    # TODO: make sure still strictly increasing even with the addition of the 1 and nbins-2
    prev_thresh.insert(0, 1)
    prev_thresh.append(nbins-2)

    slices = [slice(int(ceil(ta)), int(floor(tb))+1) for ta,tb in pairwise(prev_thresh)]
    slices.insert(0, slice(0, 1))
    slices.append(slice(nbins-1, nbins))
    
    old_settings = seterr(divide='ignore', invalid='ignore')
    H = zeros(tuple(slc.stop-slc.start for slc in slices))
    for slc,slc_next in pairwise((None,)*i + (slc,) + (None,)*(N+1-i) for i,slc in enumerate(slices)):
        d_omega = omega[slc_next] - omega[slc]
        d_mu = (mu[slc_next] - mu[slc])/d_omega - mu_t
        H += nan_to_num(d_mu*d_mu*d_omega)
    seterr(**old_settings)
    sigma_max = H.max()
    inds = where(H == sigma_max)
    return [ind.mean() + slc.start for ind,slc in islice(izip(inds,slices), 1, len(slices)-1)], \
            sigma_max + (mu[0]/omega[0]-mu_t)*(mu[0]-mu_t*omega[0]) # constant value needed to get the values to work out

def threshold(im, thresh=None):
    """
    Convert image to black and white. The threshold is used to determine what is made black and
    white. Every value at or above the threshold will be white and below it will be black. If the
    threshold is not given it is automatically calculated using Otsu's method.
    """
    check_image(im)
    dt, nchan = get_im_dtype_and_nchan(im)
    if dt.kind == 'c' or nchan != 1: raise ValueError('unsupported image type')
    if threshold is None: thresh = otsus_multithresh(im, 2)
    return im>=thresh

def hysteresis_threshold(im, thresh=None, footprint=None):
    """
    Convert image to black and white using hysteresis thresholding. Two threshold values are used
    to determine what is made black and white, with everything at or above the higher threshold
    will be white, everything below the lower threshold is black, and everything inbetween is
    calculated using hysteresis - everything that has a white neighbor is made white, otherwise it
    is made black. Neighbors are determined using the footprint given or a footprint with
    connectivity 1 (no diagonals) if not provided. If the thresholds are not provided they are
    calculated using Otsu's method.
    """
    check_image(im)
    dt, nchan = get_im_dtype_and_nchan(im)
    if dt.kind == 'c' or nchan != 1: raise ValueError('unsupported image type')
    if thresh is None:     t1,t2 = otsus_multithresh(im, 3)
    elif len(thresh) != 2: raise ValueError('thresh must be two values')
    else:                  t1,t2 = sorted(thresh)
    if footprint is None:  footprint = generate_binary_structure(im.ndim, 1)
    elif footprint.ndim != im.ndim: raise ValueError('footprint must have the same number of dimensions as the image')
    return binary_dilation(im>=t2, iterations=-1, mask=im>=t1, structure=footprint)

def multithreshold(im, thresh=4):
    """
    Simplify an image by reducing the number of shades of gray. The thresholds are used to
    determine how many shades or gray (+1 from the number of thresholds), with the thresholds
    defining the lower limit of each shade of gray. If a single value is given, it indicates the
    number of shades of gray and automatically calculates the thresholds using Otsu's method.
    Default value is 4 levels.
    """
    from numpy import digitize
    check_image(im)
    dt, nchan = get_im_dtype_and_nchan(im)
    if dt.kind == 'c' or nchan != 1: raise ValueError('unsupported image type')
    thresh = sorted(thresh) if isinstance(thresh, Sequence) else list(otsus_multithresh(im, thresh))
    mn = im.min()
    if mn not in thresh: thresh.insert(0, mn) #pylint: disable=no-member
    return digitize(im.ravel(), thresh).reshape(im.shape) - 1


########## Image Stacks ##########
class ThresholdImageStack(FilteredImageStack):
    _thresh = None
    def __init__(self, ims, thresh='auto'):
        self._dtype, self._homogeneous = dtype(bool), Homogeneous.DType
        if thresh == 'auto-stack':
            if not ims.is_dtype_homogeneous: raise ValueError('Cannot threshold the entire stack if it does not have a homogeneous data type')
            if len(ims.dtype.shape): raise ValueError('Multichannel images not supported')
            if ims.dtype.kind == 'c': raise ValueError('Complex type not supported')
            thresh = [delayed(self._stack_threshold, Real)]*len(ims)
        elif thresh == 'auto':
            thresh = [None]*len(ims)
        elif isinstance(thresh, Sequence):
            if len(thresh) < len(ims):
                thresh = list(thresh) + [thresh[-1]]*(len(ims)-len(thresh))
        else:
            thresh = [thresh] * len(ims)
        super(ThresholdImageStack, self).__init__(ims,
            [ThresholdImageSlice(im, self, z, t) for z,(im,t) in enumerate(izip(ims, thresh))])
    def _stack_threshold(self):
        if self._thresh is None:
            #pylint: disable=protected-access
            mn,mx = get_dtype_min_max(self._ims.dtype)
            h = zeros(256, intp)
            for slc in self._slices: h += histogram(slc._input.data, mn, mx, 256)
            self._thresh = otsus_multithresh((h, linspace(mn, mx, 256)), 2)[0]
        return self._thresh
class ThresholdImageSlice(FilteredImageSlice):
    def __init__(self, im, stack, z, thresh):
        super(ThresholdImageSlice, self).__init__(im, stack, z)
        self.__threshold = thresh
        self._set_props(dtype(bool), None)
    def _get_props(self): self._set_props(None, self._input.shape)
    def _get_data(self): return threshold(self._input.data, self.__threshold)

class HysteresisThresholdImageStack(FilteredImageStack):
    _thresh = None
    def __init__(self, ims, thresh='auto'):
        self._dtype, self._homogeneous = dtype(bool), Homogeneous.DType
        if thresh == 'auto-stack':
            if not ims.is_dtype_homogeneous: raise ValueError('Cannot threshold the entire stack if it does not have a homogeneous data type')
            if ims.dtype.kind == 'c': raise ValueError('Complex type not supported')
            thresh = [delayed(self._stack_threshold, tuple)]*len(ims)
        elif thresh == 'auto':
            thresh = [None]*len(ims)
        elif isinstance(thresh, Sequence):
            if len(thresh) < len(ims):
                thresh = list(thresh) + [thresh[-1]]*(len(ims)-len(thresh))
        else:
            thresh = [thresh] * len(ims)
        super(HysteresisThresholdImageStack, self).__init__(ims,
            [HysteresisThresholdImageSlice(im, self, z, t) for z,(im,t) in enumerate(izip(ims, thresh))])
    def _stack_threshold(self):
        if self._thresh is None:
            #pylint: disable=protected-access
            mn,mx = get_dtype_min_max(self._ims.dtype)
            h = zeros(256, intp)
            for slc in self._slices: h += histogram(slc._input.data, mn, mx, 256)
            self._thresh = otsus_multithresh((h, linspace(mn, mx, 256)), 3)
        return self._thresh
class HysteresisThresholdImageSlice(FilteredImageSlice):
    def __init__(self, im, stack, z, thresh):
        super(HysteresisThresholdImageSlice, self).__init__(im, stack, z)
        self.__threshold = thresh
        self._set_props(dtype(bool), None)
    def _get_props(self): self._set_props(None, self._input.shape)
    def _get_data(self): return hysteresis_threshold(self._input.data, self.__threshold)

class MultithresholdImageStack(FilteredImageStack):
    _thresh = None
    def __init__(self, ims, thresh=4):
        if isinstance(thresh, Sequence):
            thresh =  list(thresh)
            if len(thresh) > 255: raise ValueError('Only up to 255 levels supported')
        elif abs(thresh) > 255 or abs(thresh) < 2: raise ValueError('Only up to 255 levels supported')
        elif thresh < 0:
            self._thresh = -thresh
            thresh = delayed(self._stack_threshold, tuple)
        super(MultithresholdImageStack, self).__init__(ims, MultithresholdImageSlice, thresh)
    def _stack_threshold(self):
        if not isinstance(self._thresh, Sequence):
            #pylint: disable=protected-access
            mn,mx = get_dtype_min_max(self._ims.dtype)
            h = zeros(256, intp)
            for slc in self._slices: h += histogram(slc._input.data, mn, mx, 256)
            self._thresh = otsus_multithresh((h, linspace(mn, mx, 256)), self._thresh)
        return self._thresh
#def __init_uint_maxes():
#    from numpy import iinfo, sctypes
#    return [1,bool] + sorted((iinfo(dt).max,dt) for dt in sctypes['uint'])
#_uint_maxes = delayed(__init_uint_maxes, list)
class MultithresholdImageSlice(FilteredImageSlice):
    def __init__(self, im, stack, z, thresh):
        super(MultithresholdImageSlice, self).__init__(im, stack, z)
        self.__threshold = thresh
        #n = len(thresh) if isinstance(thresh, Sequence) else thresh
        #self._set_props(dtype(next(dt for um,dt in _uint_maxes if n <= um)), None)
        self._set_props(dtype(uint8), None)
    def _get_props(self): self._set_props(None, self._input.shape)
    def _get_data(self): return multithreshold(self._input.data, self.__threshold)


########## Commands ##########
class ThresholdCommand(CommandEasy):
    _thresh = None
    @classmethod
    def name(cls): return 'threshold'
    @classmethod
    def _desc(cls): return """
Threshold a gray-scale image converting it to just black and white. This uses a threshold with all
pixels less than the threshold will be made black/0 and all pixels greater than or equal to it will
be made white/1. By default this uses Otsu's method on each slice to determine the threshold
('auto'). Another option is to automatically determine the threshold using all slices
('auto-stack'). When giving specific values, you can give a single threshold or a comma-seperated
list of thresholds - one for each slice.
"""
    @classmethod
    def flags(cls): return ('bw', 'threshold', 'thresh', 't')
    @classmethod
    def _opts(cls): return (
        Opt('thresh', "The threshold value: auto, auto-stack, an integer/float, or a comma-seperated list of numbers",
            Opt.cast_or('auto', 'auto-stack', Opt.cast_tuple_of(Opt.cast_number())), 'auto'),
        )
    @classmethod
    def _consumes(cls): return ('Grayscale image stack to be thresholded',)
    @classmethod
    def _produces(cls): return ('Thresholded image - a logical/bool image',)
    @classmethod
    def _see_also(cls): return ('hysteresis-threshold','multithreshold','invert','scale')
    def __str__(self):
        if self._thresh == 'auto':
            return 'threshold'
        elif self._thresh == 'auto-stack':
            return 'threshold stack'
        elif len(self._thresh) == 1:
            return ('threshold at %s' % self._thresh)
        else:
            return 'threshold at [%s]' % (",".join(str(t) for t in self._thresh))
    def execute(self, stack): stack.push(ThresholdImageStack(stack.pop(), self._thresh))

class HysteresisThresholdCommand(CommandEasy):
    _thresh = None
    @classmethod
    def name(cls): return 'hysteresis-threshold'
    @classmethod
    def _desc(cls): return """
Threshold a gray-scale image converting it to just black and white using hysteresis thresholding.
It uses two thresholds. All pixels greater than or equal to the larger threshold are made white/1
and all pixels less than the smaller threshold are made black/0. All pixels between the two
thresholds are made white/1 if they are next to another white pixel. By default this uses Otsu's
method to calculate the two thresholds for each slice ('auto'). You can also use the entire stack
to calculate the thresholds ('auto-stack'). To use specific values, use two numbers separated by a
colon or a comma-seperated list of colon-separated thresholds - one for each slice.
"""
    @classmethod
    def flags(cls): return ('hysteresis-threshold', 'hyst-thresh')
    @classmethod
    def _opts(cls): return (
        Opt('thresh', "The threshold value: auto, auto-stack, two integers/floats separated with a colon, or a comma-seperated list of colon-separated numbers",
            Opt.cast_or('auto', 'auto-stack', Opt.cast_tuple_of(Opt.cast_tuple_of(Opt.cast_number(), 2, 2, ':'))), 'auto'),
        )
    @classmethod
    def _consumes(cls): return ('Grayscale image stack to be thresholded',)
    @classmethod
    def _produces(cls): return ('Thresholded image - a logical/bool image',)
    @classmethod
    def _see_also(cls): return ('threshold','multithreshold','invert','scale')
    def __str__(self):
        if self._thresh == 'auto':
            return 'hysteresis-threshold'
        elif self._thresh == 'auto-stack':
            return 'hysteresis-threshold stack'
        elif len(self._thresh) == 1:
            return 'hysteresis-threshold at %s:%s' % self._thresh
        else:
            return 'hysteresis-threshold at [%s]' % (",".join("%s:%s"%t for t in self._thresh))
    def execute(self, stack): stack.push(HysteresisThresholdImageStack(stack.pop(), self._thresh))

class MultithresholdCommand(CommandEasy):
    _thresh = None
    @classmethod
    def name(cls): return 'multithreshold'
    @classmethod
    def _desc(cls): return """
Simplify a grayscale image by reducing the number of shades of gray. The thresholds are used to
determine how many shades or gray (+1 from the number of thresholds), with the thresholds
defining the lower limit of each shade of gray. If a single value is given, it indicates the
number of shades of gray and automatically calculates the thresholds using Otsu's method for each
slice. If the value is negative it operates across all slices. Default value is 4 levels.
"""
    @classmethod
    def flags(cls): return ('multithreshold', 'multithresh')
    @classmethod
    def _opts(cls): return (
        Opt('thresh', "The number of thresholds or comma-seperated list of numbers for the thresholds", Opt.cast_tuple_of(Opt.cast_number(), 1), 4),
        )
    @classmethod
    def _consumes(cls): return ('Grayscale image stack to be thresholded',)
    @classmethod
    def _produces(cls): return ('Thresholded image - an integer image',)
    @classmethod
    def _see_also(cls): return ('threshold','hysteresis-threshold','invert','scale')
    def __str__(self):
        if len(self._thresh) == 1:
            if self._thresh[0] < 0:
                return 'multithreshold of %d levels across entire stack' % -self._thresh[0]
            else:
                return 'multithreshold of %d levels' % self._thresh[0]
        else:
            return 'multithreshold with ' + (",".join(str(t) for t in self._thresh))
    def execute(self, stack): stack.push(MultithresholdImageStack(stack.pop(), self._thresh[0] if len(self._thresh) == 1 else self._thresh))
