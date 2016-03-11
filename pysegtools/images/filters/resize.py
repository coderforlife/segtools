"""Image Filters to Resize Images"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from types import FunctionType
from itertools import product, izip

from numpy import mean, median, empty
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.interpolation import zoom

from ..types import check_image
from ._stack import FilteredImageStack, FilteredImageSlice, ImageSlice
from .._stack import Homogeneous
from ...imstack import CommandEasy, Opt
from ...general import splitstr, delayed


__all__ = ['im_bin', 'resize']

def __block_view(im, bs):
    # Gets the image as a set of blocks. Any blocks that don't fit in are simply dropped (on bottom
    # and right edges). The blocks are made into a single axis (axis=2). To change this and keep the
    # blocks as rectangles (in axis=2 and 3) remove the final reshape.
    shape   = tuple(i/b for i,b in zip(im.shape, bs))
    strides = tuple(i*b for i,b in zip(im.strides, bs)) + im.strides
    return as_strided(im, shape=shape+bs, strides=strides).reshape(shape+(-1,))
def __im_bin(f, size, im, out, n):
    edges = [i%size for i in im.shape[:n]]
    out_shp = [i - (1 if e else 0) for i,e in izip(out.shape, edges)] # the main part of out, not including the partial edges
    for use_edges in product((False, True), repeat=n):
        # In use_edges, a True is to use edge for that dimension instead of size
        bs = tuple((e if ue else size) for ue,e in izip(use_edges, edges))
        if any(b==0 for b in bs): continue # edge is nothing, skip it
        ii = tuple((slice(-e,None) if ue else slice(None)) for ue,e in izip(use_edges, edges))
        oi = tuple((slice(-1,None) if ue else slice(o))    for ue,o in izip(use_edges, out_shp))
        f(__block_view(im[ii], bs), axis=n, out=out[oi])
def _im_bin_size(sh, sz): return tuple((i+sz-1)//sz for i in sh)
def _im_bin(f, size, im, n):
    out_shp = _im_bin_size(im.shape[:n], size)
    if im.ndim == n+1 and im.shape[n] > 1: # Multi-channels images
        out = empty(out_shp+(im.shape[n],), dtype=im.dtype)
        for i in xrange(im.shape[n]): __im_bin(f, size, im[:,:,i], out[:,:,i], n)
    else: # Single-channel images
        out = empty(out_shp, dtype=im.dtype)
        __im_bin(f, size, im, out, n)
    return out
def im_bin(im, size=2, method='median'):
    """
    Shrink an image by binning the data. Every size-by-size area is reduced to a single pixel using
    the given method. The method can be mean or median (default). The bottom and right edges may be
    truncated and only evaulate over smaller areas.
    """
    check_image(im)
    size = int(size)
    if size < 2: raise ValueError('Unsupported binning size')
    if method not in ('mean', 'median'): raise ValueError('Unsupported binning method')
    return _im_bin(mean if method == 'mean' else median, size, im, 2)

def __init_scipy_zoom_round():
    from scipy.version import version
    return round if splitstr(version, int, '.') >= [0,13,0] else lambda x:x
__scipy_zoom_round = delayed(__init_scipy_zoom_round, FunctionType)
def _resize_size(sh, f): return tuple(int(__scipy_zoom_round(i*f)) for i in sh) #pylint: disable=not-callable

def resize(im, factor):
    """
    Shrink or grow an image using bicubic interpolation. If the factor is less than 1.0 the image is
    shrunk while if it is greater than 1.0 it is grown. For example, if it is 0.5 the image is
    halved in the x and y directions, while if it is 2.0 the image is doubled in the x and y
    directions.
    """
    check_image(im)
    if im.dtype.kind == 'c': raise ValueError('Complex type not supported')
    factor = float(factor)
    if factor <= 0.0: raise ValueError('Factor must be positive')
    if factor == 1.0: return im
    return zoom(im, (factor, factor) if im.ndim == 2 else (factor, factor, 1.0))


##### Image Stacks #####
class BinImageStack(FilteredImageStack):
    _stack = None
    def __init__(self, ims, size=2, method='median', per_slice=True):
        self._per_slice = bool(per_slice)
        self._size = int(size)
        if self._size < 2: raise ValueError('Unsupported binning size')
        if method not in ('mean', 'median'): raise ValueError('Unsupported binning method')
        self._f = mean if method == 'mean' else median
        if per_slice: super(BinImageStack, self).__init__(ims, BinImageSlice)
        elif not ims.is_homogeneous: raise ValueError('Cannot resize the entire stack if it is not homogeneous')
        elif ims.dtype.kind == 'c': raise ValueError('Complex type not supported')
        else:
            self._homogeneous = Homogeneous.All
            self._dtype = dt = ims.dtype
            sh = _im_bin_size((len(ims),) + ims.shape, self._size)
            zs = sh[0]
            self._shape = sh = sh[1:]
            super(BinImageStack, self).__init__(None, [StackImageSlice(dt, sh, self, z) for z in xrange(zs)])
    @property
    def stack(self):
        if self._per_slice: return super(BinImageStack, self).stack
        if self._stack is None:
            self._stack = _im_bin(self._f, self._size, self._ims.stack, 3)
            self._stack.flags.writeable = False
        return self._stack
class BinImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        self._set_props(self._input.dtype, _im_bin_size(self._input.shape, self._stack._size))
    def _get_data(self):
        out = _im_bin(self._stack._f, self._stack._size, self._input.data, 2)
        self._set_props(out.dtype, out.shape)
        return out

class ResizeImageStack(FilteredImageStack):
    _stack = None
    def __init__(self, ims, factor, per_slice=True):
        self._per_slice = bool(per_slice)
        self._factor = float(factor)
        if self._factor <= 0.0: raise ValueError('Factor must be positive')
        if per_slice: super(ResizeImageStack, self).__init__(ims, ResizeImageSlice)
        elif not ims.is_homogeneous: raise ValueError('Cannot resize the entire stack if it is not homogeneous')
        elif ims.dtype.kind == 'c': raise ValueError('Complex type not supported')
        else:
            self._homogeneous = Homogeneous.All
            self._dtype = dt = ims.dtype
            sh = _resize_size((len(ims),) + ims.shape, self._factor)
            zs = sh[0]
            self._shape = sh = sh[1:]
            super(ResizeImageStack, self).__init__(None, [StackImageSlice(dt, sh, self, z) for z in xrange(zs)])
    @property
    def stack(self):
        if self._per_slice: return super(ResizeImageStack, self).stack
        if self._stack is None:
            ims, f = self._ims.stack, self._factor
            self._stack = ims if f == 1.0 else zoom(ims, (f, f, f) if ims.ndim == 3 else (f, f, f, 1.0))
            self._stack.flags.writeable = False
        return self._stack
class ResizeImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        self._set_props(self._input.dtype, _resize_size(self._input.shape, self._stack._factor))
    def _get_data(self):
        im = resize(self._input.data, self._stack._factor)
        self._set_props(im.dtype, im.shape[:2])
        return im

class StackImageSlice(ImageSlice):
    def __init__(self, dt, sh, stack, z):
        super(StackImageSlice, self).__init__(stack, z)
        self._set_props(dt, sh)
    def _get_props(self): pass
    def _get_data(self): return self._stack.stack[self._z]


##### Commands #####
class BinCommand(CommandEasy):
    _size = None
    _method = None
    _per_slice = None
    @classmethod
    def name(cls): return 'bin'
    @classmethod
    def _desc(cls): return """
Shrink an image by binning the data. Every size-by-size area is reduced to a single pixel by taking
the mean or median (default) of the square. The bottom and right edges may be truncated and only
evaulate over smaller areas."""
    @classmethod
    def flags(cls): return ('bin',)
    @classmethod
    def _opts(cls): return (
        Opt('size', 'The size of the squares to reduce to single pixels', Opt.cast_int(lambda x:x>=2), 2),
        Opt('method', 'The method for calculating the new pixel, either mean or median', Opt.cast_in('mean','median'), 'median'),
        Opt('per-slice', 'If false, this also bins in the z-direction', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Full-size image',)
    @classmethod
    def _produces(cls): return ('Smaller, binned, image',)
    @classmethod
    def _see_also(cls): return ('resize',)
    def __str__(self): return 'bin %dx%d%s boxes using %s'%(self._size, self._size, ('' if self._per_slice else 'x%d' % self._size), self._method)
    def execute(self, stack): stack.push(BinImageStack(stack.pop(), self._size, self._method, self._per_slice))
    
class ResizeCommand(CommandEasy):
    _factor = None
    _per_slice = None
    @classmethod
    def name(cls): return 'resize'
    @classmethod
    def _desc(cls): return """
Shrink or grow an image using bicubic interpolation. If the factor is less than 1.0 the image is
shrunk while if it is greater than 1.0 it is grown. For example, if it is 0.5 the image is halved
in the x and y directions, while if it is 2.0 the image is doubled in the x and y directions."""
    @classmethod
    def flags(cls): return ('resize',)
    @classmethod
    def _opts(cls): return (
        Opt('factor', 'The amount of zooming', Opt.cast_float(lambda x:x>0.0 and x!=1.0)),
        Opt('per-slice', 'If false, this also effects the z-direction', Opt.cast_bool(), True),
        )
    @classmethod
    def _consumes(cls): return ('Original image',)
    @classmethod
    def _produces(cls): return ('Resized image',)
    @classmethod
    def _see_also(cls): return ('bin',)
    def __str__(self): return 'resize to %.2fx%s'%(self._factor, '' if self._per_slice else ' (including z)')
    def execute(self, stack): stack.push(ResizeImageStack(stack.pop(), self._factor, self._per_slice))
