"""
Frangi Vesselness Filter

Based on:
Frangi A, Niessen W, Vincken K, Viergever M - "Multiscale vessel enhancement filtering", 1998,
Medical Image Computing and Computer-Assisted Intervention - MICCAI.
http://www.dtic.upf.edu/~afrangi/articles/miccai1998.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numpy import zeros, dtype, float64

from ..types import check_image_single_channel, check_stack_single_channel, im2double, double2im
from ._stack import FilteredImageStack, FilteredImageSlice
from .._stack import Homogeneous
from ...imstack import CommandEasy, Opt

__all__ = ['frangi2', 'frangi3', 'FrangiImageStack']


########## Single Slice Functions ##########
# These are implemented in Cython (with fallbacks in Python). See _frangi.pyx. The _frangi.pyx can
# take some time to compile the first time. The _frangi.pyd/.so created can be moved between
# different "identical" systems. If the directory containing _frangi.pyx is writable, it will be
# placed into that directory. If the directory is not writable, it will be placed somewhere in
# ~/.pyxbld. It is aways checked for in those places before re-compiling.
from ...general import cython; cython.install()
from . import _frangi
def frangi2(im, out=None, sigmas=(1.0, 3.0, 5.0, 7.0, 9.0), beta=0.5, c=None, black=True, return_full=False):
    """
    Computes the 2D Frangi filter using the eigenvectors of the Hessian to compute the likeliness of
    an image region to contain vessels or other image ridges, according to the method described by
    Frangi (1998).
    
    `out = frangi2(im, **opts)`
    `out,sigs,dirs = frangi2(im, return_full=True, **opts)`
    
    Inputs:
        im          the input image, must be a 2D grayscale image
        out         the output results, default is to allocate it
        sigmas      the sigmas used, default is (1, 3, 5, 7, 9)
        beta        constant for the threshold for blob-like structure, default is 0.5
        c           constant for the threshold for second order structureness, default is dynamic
        black       if True then detect black ridges (default), otherwise detect white ridges
        return_full if True then `sigs` and `dirs` are returned, the default is False
    
    The default value of c is 'dynamic' which means it is calculated for each sigma as half of the
    Frobenius maximum norm of all Hessian matrices. This is the recommended approach for most cases
    by Frangi (1998). This adds an insignificant amount of additional computation.
    
    Outputs:
        out         the filtered image
        sigs        the sigmas for which the maximum intensity of every pixel is found
        dirs        the directions of the minor eigenvectors
    
    Memory usage:
        Cython version: 4 times the size of the image, plus 2 more if return_full is True
        Python version: 13 times the size of the image, plus 5 more if return_full is True
        Both versions: 1 time the image size if the input image is not a float64 image plus
                       1 time the image size if the output image is not provided

    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by M. Schrijver (2001) and D. Kroon (2009)
    
    Differences for the MATLAB version:
        * sigmas are given as a squence instead of with FrangiScaleRange/FrangiScaleRatio
        * FrangiBetaOne argument renamed to beta
        * FrangiBetaTwo argument renamed to c and now defaults to the recommended calculation
        * BlackWhite argument renamed to black
        * verbose argument dropped
        * sigmas and directions outputs are now optional and not always calculated (calculating
        them adds about 50% to time)
        * Not all intermediate filtered images are stored but instead the max is calculated as it
        goes along (saving on memory)
    """
    # Since pylint is not able to properly detect Cython-compiled code members, so we disable it here
    #pylint: disable=no-member
    im,dt = im2double(check_image_single_channel(im), return_dtype=True)
    if out is None: out = zeros(im.shape)
    sigmas = tuple(float(s) for s in sigmas)
    if beta <= 0 or any(s <= 0 for s in sigmas): raise ValueError('negative constant')
    if c is None: c = 0.0
    elif c <= 0: raise ValueError('negative constant')
    return double2im(_frangi.frangi2(im, out, sigmas, beta, c, black, return_full), dt)

def frangi3(im, out=None, sigmas=(1.0, 3.0, 5.0, 7.0, 9.0), alpha=0.5, beta=0.5, c=None, black=True, return_full=False):
    """
    Computes the 3D Frangi filter using the eigenvectors of the Hessian to compute the likeliness of
    an image region to contain vessels or other image ridges, according to the method described by
    Frangi (1998).
    
    `out = frangi3(im, **opts)`
    `out,sigs,vx,vy,vz = frangi3(im, return_full=True, **opts)`
    
    Inputs:
        im          the input image stack, must be a 3D grayscale image
        out         the output results, default is to allocate it
        sigmas      the sigmas used, default is (1, 3, 5, 7, 9)
        alpha       constant for the threshold for plate-like structure, default is 0.5
        beta        constant for the threshold for blob-like structure, default is 0.5
        c           constant for the threshold for second order structureness, default is dynamic
        black       if True then detect black ridges (default), otherwise detect white ridges
        return_full if True then `sigs`, `vx`, `vy`, and `vz` are returned, the default is False
    
    The default value of c is 'dynamic' which means it is calculated for each sigma as half of the
    Frobenius maximum norm of all Hessian matrices. This is the recommended approach for most cases
    by Frangi (1998). This adds an insignificant amount of additional computation.
    
    Outputs:
        out         the filtered image
        sigs        the sigmas for which the maximum intensity of every pixel is found
        vx,vy,vz    the directions of the minor eigenvectors

    Memory usage:
        Cython version: 7 times the size of the image, plus 4 more if return_full is True
        Python version: 17 times the size of the image, plus 18 more if return_full is True
        Both versions: 1 time the image size if the input image is not a float64 image plus
                       1 time the image size if the output image is not provided

    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by D. Kroon (2009)
    
    Differences for the MATLAB version:
        * sigmas are given as a squence instead of with FrangiScaleRange/FrangiScaleRatio
        * FrangiAlpha argument renamed to alpha
        * FrangiBeta argument renamed to beta
        * FrangiC argument renamed to c and now defaults to the recommended calculation
        * BlackWhite argument renamed to black
        * verbose argument dropped
        * sigmas and directions outputs are now optional and not always calculated (calculating
        them adds about ??% to time)
        * Not all intermediate filtered images are stored but instead the max is calculated as it
        goes along (saving on memory)
    """
    # Since pylint is not able to properly detect Cython-compiled code members, so we disable it here
    #pylint: disable=no-member
    im,dt = im2double(check_stack_single_channel(im), return_dtype=True)
    if out is None: out = zeros(im.shape)
    sigmas = tuple(float(s) for s in sigmas)
    if alpha <= 0 or beta <= 0 or any(s <= 0 for s in sigmas): raise ValueError('negative constant')
    if c is None: c = 0.0
    elif c <= 0: raise ValueError('negative constant')
    return double2im(_frangi.frangi3(im, out, sigmas, alpha, beta, c, black, return_full), dt)


########## Image Stacks ##########
class FrangiImageStack(FilteredImageStack):
    def __init__(self, ims, sigmas=(1, 3, 5, 7, 9), alpha=0.5, beta=0.5, c=None, black=True, per_slice=True):
        self.sigmas = tuple(float(s) for s in sigmas)
        self.alpha = float(alpha)
        self.beta = float(beta)
        if self.alpha <= 0 or self.beta <= 0 or any(s <= 0 for s in self.sigmas): raise ValueError('negative constant')
        if c is not None:
            c = float(c)
            if c <= 0: raise ValueError('negative constant')
        self.c = c
        self.black = bool(black)
        self.__per_slice = per_slice = bool(per_slice)
        self.__stack = None
        self._dtype = dtype(float64)
        if per_slice:
            self._homogeneous = Homogeneous.DType
        elif not ims.is_homogeneous:
            raise ValueError('Cannot perform Frangi filter on the entire stack if it is is not homogeneous')
        else:
            self._shape = ims.shape
            self._homogeneous = Homogeneous.All
        super(FrangiImageStack, self).__init__(ims, FrangiPerSlice if per_slice else FrangiImageSlice)
    @property
    def stack(self):
        if self.__per_slice: return super(FrangiImageStack, self).stack
        if self.__stack is None:
            self.__stack = frangi3(self._ims.stack, None, self.sigmas, self.alpha, self.beta, self.c, self.black)
        return self.__stack
class FrangiPerSlice(FilteredImageSlice):
    def _get_props(self): self._set_props(dtype(float64), self._input.shape)
    def _get_data(self): return frangi2(self._input.data, None, self._stack.sigmas, self._stack.beta, self._stack.c, self._stack.black)
class FrangiImageSlice(FilteredImageSlice):
    def _get_props(self): self._set_props(dtype(float64), self._input.shape)
    def _get_data(self): return self._stack.stack[self._z]


########## Commands ##########
_cast_num = Opt.cast_float(lambda x:x>0)
class FrangiCommand(CommandEasy):
    _sigmas = None
    _alpha = None
    _beta = None
    _c = None
    _black = None
    _per_slice = None
    @classmethod
    def name(cls): return 'frangi'
    @classmethod
    def _desc(cls): return """
Filters an image using the Frangi filter by using the eigenvectors of the Hessian to compute the
likeliness of an image region to contain vessels, tubes, or other image ridges. The vessels/tubes
can either be dark or light and this can operate on individual slices or the entire input stack (if
it has a homogeneous shape and type).

Frangi A, Niessen W, Vincken K, Viergever M. "Multiscale vessel enhancement filtering", 1998,
Medical Image Computing and Computer-Assisted Intervention - MICCAI.
"""
    @classmethod
    def flags(cls): return ('frangi',)
    @classmethod
    def _opts(cls): return (
        Opt('sigmas', 'the sigmas used as a comma-separated list', Opt.cast_tuple_of(_cast_num, 1), (1, 3, 5, 7, 9)),
        Opt('alpha', 'constant for the threshold for plate-like structure, only used if per-slice is false', _cast_num, 0.5),
        Opt('beta', 'constant for the threshold for blob-like structure', _cast_num, 0.5),
        Opt('c', 'constant for the threshold for second order structureness', Opt.cast_or('dynamic', _cast_num), 'dynamic'),
        Opt('black', 'if true, detect black ridges, otherwise detect white ridges', Opt.cast_bool(), True),
        Opt('per-slice', 'if false, operate on the entire stack at once', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Gray-scale image stack to be filtered',)
    @classmethod
    def _produces(cls): return ('Filtered image stack',)
    def __str__(self):
        ps, c = self._per_slice, self._c
        return 'Frangi filter %s ridges in %s with sigmas=%s %sbeta=%.2f c=%s'%(
            'black' if self._black else 'white', '2D' if ps else '3D', ','.join(str(s) for s in self._sigmas),
            'alpha=%.2f '%self._alpha if ps else '', self._beta, 'dynamic' if c is None else c,
            )
    def __init__(self, args, stack):
        super(FrangiCommand, self).__init__(args, stack)
        if self._c == 'dynamic': self._c = None
    def execute(self, stack):
        stack.push(FrangiImageStack(stack.pop(), self._sigmas, self._alpha, self._beta, self._c,
                                    self._black, self._per_slice))
