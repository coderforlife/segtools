from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Integral
from itertools import repeat as irepeat, izip
from sys import stdin, stdout
from StringIO import StringIO

from numpy import dtype, int64, intp, float64, finfo
from numpy import array, empty, zeros, linspace, tile, repeat, vstack, savetxt, loadtxt
from numpy import add, divide, left_shift, floor, sqrt
from numpy import lexsort, spacing, count_nonzero
from scipy.ndimage import histogram

from ._stack import UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from .._stack import ImageStack
from ..types import get_im_min_max, get_dtype_min_max, get_dtype_max, get_dtype_min, check_image_single_channel
from ...imstack import Command, Opt, Help
from ...general import String, delayed

__all__ = ['imhist', 'histeq_trans', 'histeq_apply', 'histeq', 'histeq_exact']

def __as_unsigned(im):
    o_dt = im.dtype
    if im.dtype.kind == 'i':
        u_dt = dtype(o_dt.byteorder+'u'+str(o_dt.itemsize))
        im = im.view(u_dt)
        im -= get_dtype_min(o_dt)
    return im, o_dt
def __restore_signed(im, dt):
    if dt.kind == 'i': im -= -long(get_dtype_min(dt))
    return im.view(dt)

def __imhist(im, nbins):
    mn,mx = get_im_min_max(im)
    return histogram(im, mn, mx, nbins)

def imhist(im, nbins=256, mask=None):
    """Calculate the histogram of an image. By default it uses 256 bins (nbins)."""
    im = check_image_single_channel(im)
    if mask is not None:
        mask = check_image_single_channel(mask)
        if mask.dtype != bool or mask.shape != im.shape: raise ValueError('The mask must be a binary image with equal dimensions to the image')
        im = im[mask]
    return __imhist(im, nbins)

__eps = sqrt(spacing(1))

def histeq_trans(h_src, h_dst, dt):
    """
    Calculates the histogram equalization transform. It takes a source histogram, destination
    histogram, and a data type. It returns the transform, which has len(h_src) elements of a
    data-type similar to the given data-type but unsigned. This transform can be used with
    histeq_apply. This allows you to calculate the transform just once for the same source and
    destination histograms and use it many times.

    This is really just one-half of histeq, see it for more details.
    """
    dt = dtype(dt)
    if dt.base != dt or dt.kind not in 'iuf': raise ValueError("Unsupported data-type")
    if dt.kind == 'i': dt = dtype(dt.byteorder+'u'+str(dt.itemsize))
    
    h_src = h_src.ravel()/sum(h_src)
    h_src_cdf = h_src.cumsum()
    nbins_src = len(h_src)

    h_dst = tile(1/h_dst, h_dst) if isinstance(h_dst, Integral) else h_dst.ravel()/sum(h_dst)
    h_dst_cdf = h_dst.cumsum()
    nbins_dst = len(h_dst)

    if nbins_dst < 2 or nbins_src < 2: raise ValueError('Invalid histograms')

    xx = vstack((h_src, h_src))
    xx[0,-1],xx[1,0] = 0.0,0.0
    tol = tile(xx.min(0)/2.0,(nbins_dst,1))
    err = tile(h_dst_cdf,(nbins_src,1)).T - tile(h_src_cdf,(nbins_dst,1)) + tol
    err[err < -__eps] = 1.0
    T = err.argmin(0)*(get_dtype_max(dt)/(nbins_dst-1.0))
    T = T.round(out=T).astype(dt, copy=False)
    return T

def __histeq_apply(im, T):
    im,o_dt = __as_unsigned(im)
    nlevels = get_dtype_max(im)
    if o_dt.kind != 'f' and nlevels == len(T)-1: idx = im # perfect fit, we don't need to scale the indices
    else: idx = (im*(float(len(T)-1)/nlevels)).round(out=empty(im.shape, dtype=intp)) # scale the indices
    return __restore_signed(T.take(idx),o_dt)

def histeq_apply(im, T, mask=None):
    """
    Apply a histogram-equalization transformation to an image. The transform can be created with
    histeq_trans. The image must have the same data-type as given to histeq_trans.

    This is really just one-half of histeq, see it for more details.
    """
    im = check_image_single_channel(im)
    if im.dtype.kind not in 'iuf': raise ValueError("Unsupported data-type")
    if mask is not None:
        mask = check_image_single_channel(mask)
        if mask.dtype != bool or mask.shape != im.shape: raise ValueError('The mask must be a binary image with equal dimensions to the image')
        im[mask] = __histeq_apply(im[mask], T)
        return im
    return __histeq_apply(im, T)
    
def __histeq(im, h_dst, h_src):
    im,o_dt = __as_unsigned(im)
    nlevels = get_dtype_max(im.dtype)
   
    h_dst = tile(1/h_dst, h_dst) if isinstance(h_dst, Integral) else h_dst.ravel()/sum(h_dst)
    h_dst_cdf = h_dst.cumsum()
    nbins_dst = len(h_dst)

    if h_src is None: h_src = histogram(im, 0, nlevels, 256)
    h_src = h_src.ravel()/sum(h_src)
    h_src_cdf = h_src.cumsum()
    nbins_src = len(h_src)

    if nbins_dst < 2 or nbins_src < 2: raise ValueError('Invalid histograms')

    xx = vstack((h_src, h_src))
    xx[0,-1],xx[1,0] = 0.0,0.0
    tol = tile(xx.min(0)/2.0,(nbins_dst,1))
    err = tile(h_dst_cdf,(nbins_src,1)).T - tile(h_src_cdf,(nbins_dst,1)) + tol
    err[err < -__eps] = 1.0
    T = err.argmin(0)*(nlevels/(nbins_dst-1.0))
    T = T.round(out=T).astype(im.dtype, copy=False)

    if o_dt.kind != 'f' and nlevels == len(T)-1: idx = im # perfect fit, we don't need to scale the indices
    else: idx = (im*(float(len(T)-1)/nlevels)).round(out=empty(im.shape, dtype=intp)) # scale the indices
    return __restore_signed(T.take(idx),o_dt)

def histeq(im, h_dst=64, h_src=None, mask=None):
    """
    Equalize the histogram of an image. The destination histogram is either given explicitly (as a
    sequence) or a uniform distribution of a fixed number of bins (defaults to 64 bins).

    Additionally, you can specify the source historgram instead of calculating it from the image
    itself (using 256 bins). This is useful if you want to use the source histogram to be from
    multiple images and so that each image will be mapped the same way (although using
    histeq_trans/histeq_apply will be more efficient for something like that).

    Supports using a mask in which only those elements will be transformed and considered in
    calcualting h_src.
    
    Supports integral and floating-point (from 0.0 to 1.0) image data types. To do something similar
    with bool/logical, use convert.bw. The h_dst and h_src must have at least 2 bins.

    The performs an approximate histogram equalization. This is the most "standard" technique used.
    An exact histogram equalization is available with histeq_exact, however it takes substantially
    more memory and time, and cannot be given the source histogram or split into two functions
    (thus it cannot be easily parallelized).
    """
    im = check_image_single_channel(im)
    if im.dtype.kind not in 'iuf': raise ValueError("Unsupported data-type")
    if mask is not None:
        mask = check_image_single_channel(mask)
        if mask.dtype != bool or mask.shape != im.shape: raise ValueError('The mask must be a binary image with equal dimensions to the image')
        im[mask] = __histeq(im[mask], h_dst, h_src)
        return im
    return __histeq(im, h_dst, h_src)

def histeq_exact(im, h_dst=256, mask=None, method='VA', **kwargs):
    """
    Like histeq except the histogram of the output image is exactly as given in h_dst. This is
    acomplished by strictly ordering all pixels based on other properties of the image. There is
    currently no way to specify the source histogram or to calculate the transform to apply to
    different images. See histeq for the details of the arguments h_dst and mask.
    
    This method takes significantly more time than the approximate ("standard") version. See below
    for more details.
    
    Internally this uses either the local mean ordering (LM) by Coltuc, Bolon and Chassery or the
    variational approach (VA) of Nikolova, Wen and Chan to create the strict ordering of pixels.
    The method defaults to 'VA' which is a bit slower than 'LM' for 8-bit images but faster for
    other types. Also it is more accurate in general.
    
    LM:
        This uses the gray levels of expanding mean filters to distinguish between same-valued
        pixels. Accepts a `order` argument which has a maximum and default of 6 that controls how
        far away pixels are used to help distinguish pixels from each other. An order of 1 would be
        using only the pixel itself and no neighbors (but this isn't allowed). It has been
        optimized for 8-bit images which take about twice the memory and 8x the time from standard
        histeq. Other image types which can take 7x the memory and 40x the time.
        
    VA:
        This attempts to reconstruct the original real-valued version of the image and thus is a
        continuous-valued version of the image which could be strictly ordered. Accepts a `niters`
        argument that specifies how many minimization iterations to perform to make sure the real-
        valued image is faithly reproduced. The default is 5. If takes about twice the memory and
        10x the time from standard histeq regardless of type.

    REFERENCES:
      1. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications"
      2. Coltuc D, Bolon P and Chassery J-M, 2006, "Exact histogram specification", IEEE
         Trans. on Image Processing 15(5):1143-1152
      3. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
         using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
      4. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
         IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    # Check arguments
    im = check_image_single_channel(im)
    n, sh, dt = im.size, im.shape, im.dtype
    if dt.kind not in 'iuf': raise ValueError('Unsupported data-type')
    mn, mx = get_dtype_min_max(dt)
    if mask is not None:
        mask = check_image_single_channel(mask)
        if mask.dtype != bool or mask.shape != sh: raise ValueError('The mask must be a binary image with equal dimensions to the image')
        mask = mask.ravel()
        n = count_nonzero(mask)
    h_dst = tile(n/h_dst, h_dst) if isinstance(h_dst, Integral) else h_dst.ravel()*(n/h_dst.sum()) #pylint: disable=no-member
    if len(h_dst) < 2: raise ValueError('Invalid histogram')
    
    ##### Assign strict order to pixels #####
    if method == 'LM':   idx = __pixel_order_lm(im, **kwargs)
    elif method == 'VA': idx = __pixel_order_va(im, **kwargs)
    else:                raise ValueError('method')
    del im
    
    ##### Handle the mask #####
    if mask is not None:
        idx[~mask] = 0 #pylint: disable=invalid-unary-operand-type
        idx[mask] = idx[mask].argsort().argsort()
        del mask

    ##### Create the transform that is the size of the image but with sorted histogram values #####
    # Since there could be fractional amounts, make sure they are added up and put somewhere
    H_whole = floor(h_dst).astype(intp, copy=False)
    nw = H_whole.sum()
    if n == nw: h_dst = H_whole
    else:
        R = __n_argmax(h_dst-H_whole, n-nw)
        h_dst = H_whole
        h_dst[R] += 1
        del R
    T = empty(idx.size, dtype=dt)
    T[-n:] = repeat(linspace(mn, mx, len(h_dst), dtype=dt), h_dst)
    del h_dst, H_whole
    
    ##### Create the equalized image #####
    return T.take(idx).reshape(sh)
    
def __n_argmax(a, n):
    # argpartition is way faster but was not introduced until numpy v1.8 and we want to work with v1.7
    return (a.argpartition(-n) if hasattr(a, 'argpartition') else a.argsort())[-n:]

############### Histeq-Exact LM ###############
def __pixel_order_lm(im, order=6):
    """
    Assign strict ordering to image pixels. Outputs an array that has the same size as the input.
    Its element entries correspond to the order of the gray level pixel in that position.

    This implements the method by Coltuc et al which takes increasing uniform filter sizes to find
    the local means. It takes an argument for the order (called k in the paper) which must be from
    2 to 6 and defaulting to 6.

    REFERENCES:
      1. Coltuc D. and Bolon P., 1999, "Strict ordering on discrete images and applications"
      2. Coltuc D., Bolon P. and Chassery J-M., 2006, "Exact histogram specification", IEEE
         Transcations on Image Processing 15(5):1143-1152
    """
    from scipy.ndimage.filters import correlate

    if order < 2 or order > 6: raise ValueError('Invalid order')
    im,_ = __as_unsigned(im)
    
    if im.dtype.kind == 'u' and im.dtype.itemsize <= 2:
        Fs = __create_uint_filter(order, im.dtype.itemsize)
        if im.dtype.itemsize == 1 or order <= 3:
            ##### Single convolution and no lexsort #####
            shift = 63-8*im.dtype.itemsize
            im = im.astype(int64)
            FR = correlate(im, Fs[0])
            left_shift(im, shift, im)
            return add(FR, im, im).ravel().argsort().argsort()
    else: # if im.dtype.kind == 'f' or im.dtype.itemsize > 2:
        Fs = __filters_float[-order+1:]

    ##### Convolve filters with the image and lexsort #####
    FR = empty(im.shape+(len(Fs)+1,), float64)
    FR[...,-1] = im = im.astype(float64)
    for i,F in enumerate(Fs): correlate(im, F, FR[...,i])
    del im
    return lexsort(FR.reshape((-1, FR.shape[-1])).T, 0).argsort()

__filter3 = array([[3,2,3],[2,1,2],[3,2,3]])
__filter5 = array([[6,5,4,5,6],[5,3,2,3,5],[4,2,1,2,4],[5,3,2,3,5],[6,5,4,5,6]])
__filters = [None, array([[1]]), __filter3, __filter3, __filter5, __filter5, __filter5]
__filters_uint = {}
__filters_float = delayed(lambda:(
    (__filter5==6).astype(float64),(__filter5==5).astype(float64),(__filter5==4).astype(float64),
    (__filter3==3).astype(float64),(__filter3==2).astype(float64)), tuple)

def __create_uint_filter(order, nbytes):
    # TODO: work backwards so that the unused bits are at the front instead of the back
    idx = (order, nbytes)
    if idx in __filters_uint: return __filters_uint[idx]
    
    extra_bits = [0, 0, 2, 2, 2, 3, 2]
    out = ()
    nbits = nbytes*8
    while order >= 2:
        avail_bits = finfo(float64).nmant
        base = __filters[order]
        fltr = zeros(base.shape, float64)
        out += (fltr,)
        fltr[base==order] = 1.0
        used_bits = nbits+extra_bits[order]
        order -= 1
        while order >= 2 and used_bits + nbits+extra_bits[order] <= avail_bits:
            fltr[base==order] = float64(1 << used_bits)
            used_bits += nbits+extra_bits[order]
            order -= 1

    __filters_uint[idx] = out
    return out

############### Histeq-Exact VA ###############
def __pixel_order_va(im, niters=5):
    """
    Assign strict ordering to image pixels. Outputs an array that has the same size as the input.
    Its element entries correspond to the order of the gray level pixel in that position.
    
    This implements the method by Nikolova et al which attempts to reconstruct the original real-
    valued version of the image using a very fast minimization method. It takes an argument for the
    number of iterations to perform, defaulting to 5.
    
    Always uses eta=4 (4-connected instead of 8-connected), alpha1=0.05, alpha2=0.05, beta=0.1, and
    theta-2.
    
    REFERENCE:
      1. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
         using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
      2. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
         IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    assert niters > 0
    from numpy import abs #pylint: disable=redefined-builtin
    ALPHA_1 = 0.05
    ALPHA_2 = 0.05
    BETA    = 0.1
    f = im.astype(float64)
    # Minimization method from 2014 paper using theta-2
    R = empty(im.shape)
    for _ in xrange(niters):
        # Up/down difference
        t = f[1:,:] - f[:-1,:]
        tmp = abs(t); tmp += ALPHA_2; t /= tmp # t/(alpha2+|t|), t = diff(f, axis=0)
        R[0].fill(0); R[1:,:] = t; R[:-1,:] -= t
        # Left/right difference
        t = f[:,1:] - f[:,:-1]
        tmp = abs(t); tmp += ALPHA_2; t /= tmp # t/(alpha2+|t|), t = diff(f, axis=1)
        R[:,1:] += t; R[:,:-1] -= t
        # Core
        R *= BETA
        abs(R, f); f -= 1; divide(ALPHA_1, f, f); f *= R # R*alpha1 / (|R|-1)
        f += im
        del t, tmp
        # This, in general, is slower than just doing more iterations
        #if k >= 1 and __is_unique(f.ravel()): break
    return f.ravel().argsort().argsort()

# Slower minimization method using theta-1 (doesn't take into account 2014 paper)
#def J(f, im):
#    f = f.reshape(im.shape)
#    return Psi(f, im, ALPHA_1) + BETA*Phi(f, ALPHA_2)
#def Psi(f, im, alpha): return theta(f-im, alpha).sum()
#def Phi(f, alpha):
#    return (theta(f[:,:-1]-f[:,1:], alpha).sum() + # left neighbor
#            theta(f[:,1:]-f[:,:-1], alpha).sum() + # right neighbor
#            theta(f[:-1,:]-f[1:,:], alpha).sum() + # top neighbor
#            theta(f[1:,:]-f[:-1,:], alpha).sum() + # bottom neighbor
#            theta(array(0.0), alpha)*2*(f.shape[0]+f.shape[1])) # edges
#def theta(t, alpha): t *= t; t += alpha; return sqrt(t, out=t) # sqrt(t*t+alpha)
#def jac(f, im):
#    f = f.reshape(im.shape)
#    out = dPhi(f, ALPHA_2); out *= BETA; out += dPsi(f, im, ALPHA_1)
#    return out.ravel()
#def dPsi(f, im, alpha): return dtheta(f-im, alpha)
#def dPhi(f, alpha):
#    fp = pad(f, 1, 'edge')
#    out  = dtheta(f-fp[1:-1,2:],  alpha) # left neighbor
#    out += dtheta(f-fp[1:-1,:-2], alpha) # right neighbor
#    out += dtheta(f-fp[2:,1:-1],  alpha) # top neighbor
#    out += dtheta(f-fp[:-2,1:-1], alpha) # bottom neighbor
#    out *= 2
#    return out
#def dtheta(t, alpha): tt = t*t; tt += alpha; sqrt(tt, out=tt); t /= tt; return t # t/sqrt(t*t+alpha)
#from scipy.optimize import minimize
#f = minimize(J, f, (im,), jac=jac, method='L-BFGS-B', options={'maxiter':niters, 'gtol':5e-04, 'ftol':1e-06}).x
#return f.argsort().argsort()


##### Image Stack #####
class HistEqImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims, h_dst=None, h_src=None, mask=None, exact=False, **kwargs):
        """
        If h_src is set, exact must be False and the image stack must have a homogeneous dtype.
        Besides h_src taking the values that histeq can take, it can also take "True" which will
        cause it to be calculated from the entire image stack using 256 bins.
        h_dst defaults to 256 if exact is True and 64 if it is not.
        If exact is True then kwargs can have extra options that will be sent to histeq_exact.
        """
        if exact:
            if h_dst is None: h_dst = 256
            if h_src is not None: raise ValueError()
            # TODO: check the kwargs
            self._histeq = lambda im,mk: histeq_exact(im, h_dst, mk, **kwargs)
        elif len(kwargs) != 0: raise ValueError('Invalid extra arguments')
        else:
            if h_dst is None: h_dst = 64
            if h_src is not None:
                if h_src is True:
                    self._h_dst = h_dst
                    self._T = None
                    self._histeq = lambda im,mk: histeq_apply(im, self._get_trans(), mask=mk)
                else:
                    T = histeq_trans(h_src, h_dst, ims.dtype)
                    self._histeq = lambda im,mk: histeq_apply(im, T, mask=mk)
            else:
                self._histeq = lambda im,mk: histeq(im, h_dst, None, mk)
        if (h_dst if isinstance(h_dst, Integral) else len(h_dst)) < 2: raise ValueError('Histogram too small')
        if mask is not None:
            mask = ImageStack.as_image_stack(mask)
            if mask.dtype != bool or len(mask) != len(ims): raise ValueError('mask must be of bool/logical type of the same shape as image')
        else:
            mask = irepeat(None)
        super(HistEqImageStack, self).__init__(ims,
            [HistEqImageSlice(im, self, z, mk) for z,(im,mk) in enumerate(izip(ims, mask))])
    def _get_trans(self):
        #pylint: disable=protected-access
        if self._T is None:
            h_src = zeros(256, intp)
            for slc in self._slices: h_src += imhist(slc._input.data, 256, slc.mask)
            self._T = histeq_trans(h_src, self._h_dst, self._ims.dtype)
        return self._T

class HistEqImageSlice(UnchangingFilteredImageSlice):
    #pylint: disable=protected-access
    def __init__(self, image, stack, z, mask):
        super(HistEqImageSlice, self).__init__(image, stack, z)
        self._mask = mask
        if image.dtype.kind is 'c' or (len(image.shape) > 2 and image.shape[2:] != (1,)): raise ValueError('only single-channel images can be histogramed')
        if mask is not None and mask.shape != image.shape: raise ValueError('mask must be same shape as image')
    @property
    def mask(self): return None if self._mask is None else self._mask.data
    def _get_data(self): return self._stack._histeq(self._input.data, self.mask)


##### Commands #####
    
class HistCommand(Command):
    @classmethod
    def name(cls): return 'histogram'
    @classmethod
    def flags(cls): return ('imhist','hist')
    @classmethod
    def _opts(cls): return (
        Opt('output_file', 'The file to output the histogram to or - for stdout', Opt.cast_or('-', Opt.cast_writable_file())),
        Opt('nbins',       'The number of bins to use in the histogram', Opt.cast_int(lambda x:x>=2), 256),
        Opt('per_slice',   'Calculate the histogram for each slice instead of the entire stack', Opt.cast_bool(), False),
        Opt('use_mask',    'Only take the histogram of the pixels given in the mask', Opt.cast_bool(), False),
        )
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Histogram Calculation")
        p.text("""Calculate the histogram and save to a file.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("""
Consumes:  image stack to take the histogram of 
           possibly a mask""")
        p.newline()
        p.text("Command format:")
        p.cmds("--hist output_file [nbins] [per_slice] [use_mask]")
        p.newline()
        p.text("Options:")
        p.opts(*cls._opts())
        p.newline()
        p.text("""
Calculates the histogram of an image stack or for each slice and saves it to a file (or stdout). The
file is written as a tab-seperated file (each bin's value is seperated by a tab, each slice is on
its own line). If the file ends with '.gz' then it will be compressed. If a mask is used, then only
pixels where the mask is True are counted.""")
        p.newline()
        p.text("See also:")
        p.list('histeq')
    def __str__(self):
        return 'saving histogram of %d bins%s to %s%s' % (
            self.__nbins, ' per slice' if self.__per_slice else '',
            self.__file, ' using a mask' if self.__use_mask else '')
    def __init__(self, args, stack):
        self.__file,self.__nbins,self.__per_slice,self.__use_mask = args.get_all(*HistCommand._opts())
        stack.pop()
        if self.__use_mask: stack.pop()
    def execute(self, stack):
        ims = stack.pop()
        nbins = self.__nbins
        mask = stack.pop() if self.__use_mask else irepeat(None)
        if self.__per_slice:
            H = empty((nbins, len(ims)), intp)
            for i,(im,mk) in enumerate(izip(ims,mask)):
                H[:,i] = imhist(im.data, nbins, None if mk is None else mk.data)
        else:
            H = zeros(nbins, intp)
            for im,mk in zip(ims,mask):
                H += imhist(im.data, nbins, None if mk is None else mk.data)
        savetxt(stdout if self.__file == '-' else self.__file, H, str('%u'), '\t')

class HistEqCommand(Command):
    @staticmethod
    def __cast_exact(x):
        if isinstance(x, dict): return x
        x = x.lower()
        if x == 'false': return {'exact':False}
        if x == 'true': return {'exact':True}
        if x.startswith('lm'):
            if x == 'lm': return {'exact':True, 'method':'LM'}
            if x[2] != ':': raise ValueError()
            n = int(x[3:])
            if n < 2 or n > 6: raise ValueError()
            return {'exact':True, 'method':'LM', 'order':n}
        if x.startswith('va'):
            if x == 'va': return {'exact':True, 'method':'VA'}
            if x[2] != ':': raise ValueError()
            n = int(x[3:])
            if n < 1: raise ValueError()
            return {'exact':True, 'method':'VA', 'niters':n}
        raise ValueError()
    @staticmethod
    def __exact_desc(x):
        if not x.get('exact', False) or 'method' not in x: return ''
        s = ' (' + x['method']
        if x['method'] == 'LM' and 'order'  in x: s += ':' + str(x['order'])
        if x['method'] == 'VA' and 'niters' in x: s += ':' + str(x['niters'])
        return s + ')'
        
    @classmethod
    def name(cls): return 'histogram equalization'
    @classmethod
    def flags(cls): return ('H','histeq','histogram-equalization')
    @classmethod
    def _opts(cls): return (
        Opt('hist',     'The histogram to match to, specified either as an integer (for a uniform histogram with that many bins) or a readable file (or - for stdin) where one line will be read that has white-space seperated value', Opt.cast_or(Opt.cast_int(lambda x:x>=2), '-', Opt.cast_readable_file()), None, '256 if exact is true, 64 otherwise'),
        Opt('use_mask', 'Only use and update the pixels where the mask is True', Opt.cast_bool(), False),
        Opt('exact',    'Force the image\'s histogram to exactly the given histogram with some control over the quality vs speed of the algorithm', HistEqCommand.__cast_exact, {}, 'False'),
        Opt('src_hist', 'The histogram to map from, either the current slice, the entire stack, or a custom one from a file or stdin (-); must be slice if exact is true or the input data has a heterogeneous data type', Opt.cast_or('slice', 'stack', '-', Opt.cast_readable_file()), 'slice'),
        )
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Histogram Equalization")
        p.text("""Change the histogram of an image to match a given histogram.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("""
Consumes:  image stack to change the histogram of 
           possibly a mask 
Produces:  image stack with the histogram changed""")
        p.newline()
        p.text("Command format:")
        p.cmds("--histeq [hist] [use_mask] [exact] [src_hist]")
        p.newline()
        p.text("Options:")
        p.opts(*cls._opts())
        p.newline()
        p.text("""
Changes the histogram of an image stack to match a given histogram, either approximately or exactly.
The histogram to match can be a uniform histogram (specified with an integer for the number of bins)
or from the first line of a file/stdin that has tab-seperated values for the bins. The file can be
compressed and have a .gz or .bz2 ending. Ever slice is changed to the same histogram.

If a mask is used, then only pixels where the mask is True are changed.

If exact is used, then the output image will have a histogram that perfectly matches the given
histogram, but takes longer and more memory. The exact option can be one of several options,
including the following:
""")
        p.list('false - use approximate method',
               'true  - use default exact options, equivilent to VA:5',
               'LM    - use the local means ordering method based on Coltuc et al',
               'LM:#  - where # is from 2 to 6 for the order, default is 6',
               'VA    - use the variational method based on Nikolova et al',
               'VA:#  - where # is >=0 for the iterations to use, default is 5')
        p.newline()
        p.text("""
The option src_hist overrides the source histogram, allowing one to define the direct mapping of
pixel values. This option can only be used when exact is false and the image stack has a homogeneous
data type. The most common non-default value will be 'stack' which allows the entire stack to mapped
at once.

Exact Histogram Equalization References:
""")
        p.list('Coltuc D, Bolon P, 1999, "Strict ordering on discrete images and applications"',
               'Coltuc D, Bolon P, Chassery J-M, 2006, "Exact histogram specification", IEEE Transcations on Image Processing 15(5):1143-1152',
               'Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325',
               'Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification" IEEE Trans. on Image Processing, 23(12):5274-5283')
        p.newline()
        p.text("See also:")
        p.list('imhist')
    def __str__(self):
        return '%shistogram equalization with %s%s%s%s' % (
            'exact ' if self.__exact.get('exact', False) else '',
            ('bins from %s'%('stdin' if self.__hist=='-' else self.__hist)) if isinstance(self.__hist, String) else '%s equal bins'%self.__hist,
            ' using a mask' if self.__use_mask else '',
            ' across entire stack' if self.__src_hist is True else ('' if self.__src_hist is None else (' using source histogram from %s'%('stdin' if self.__src_hist=='-' else self.__src_hist))),
            HistEqCommand.__exact_desc(self.__exact),
            )
    def __init__(self, args, stack):
        self.__hist,self.__use_mask,self.__exact,self.__src_hist = args.get_all(*HistEqCommand._opts())
        stack.pop()
        if self.__use_mask: stack.pop()
        stack.push()
        if self.__exact.get('exact', False) and self.__src_hist != 'slice':
            raise ValueError('When exact, src_hist must be \'slice\'')
        if self.__hist is None: self.__hist = 256 if self.__exact.get('exact', False) else 64
        if self.__src_hist == 'slice': self.__src_hist = None
        elif self.__src_hist == 'stack': self.__src_hist = True

    @staticmethod
    def __get_hist(h):
        if isinstance(h, String):
            if h == '-': l = stdin.readline()
            else:
                with open(h, 'r') as f: l = f.readline()
            h = loadtxt(StringIO(l.trim()))
            if h.ndim == 0 or len(h) < 2: raise ValueError('not enough data for histogram')
        return h

    def execute(self, stack):
        ims = stack.pop()
        mask = stack.pop() if self.__use_mask else None
        h_dst = HistEqCommand.__get_hist(self.__hist)
        h_src = HistEqCommand.__get_hist(self.__src_hist)
        stack.push(HistEqImageStack(ims, h_dst, h_src, mask, **self.__exact))
