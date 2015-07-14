"""Image Filters to Resize Images"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numpy import mean, median, empty
from numpy.lib.stride_tricks import as_strided

from ..types import check_image
from .._util import splitstr
from ._stack import FilteredImageStack, FilteredImageSlice
from ...imstack import CommandEasy, Opt

__all__ = ['im_bin']

def _block_view(im, bs):
    # Gets the image as a set of blocks. Any blocks that don't fit in are simply dropped (on bottom
    # and right edges). The blocks are made into a single axis (axis=2). To change this and keep the
    # blocks as rectangles (in axis=2 and 3) remove the final reshape.
    shape = (im.shape[0]//bs[0], im.shape[1]//bs[1])
    strides = (bs[0]*im.strides[0], bs[1]*im.strides[1]) + im.strides
    return as_strided(im, shape=shape+bs, strides=strides).reshape(shape+(bs[0]*bs[1],))

def _im_bin(f, size, im, out):
    # Setup shapes
    b_edge, r_edge = im.shape[0]%size, im.shape[1]%size
    out_shp = out.shape[0] - (1 if b_edge else 0), out.shape[1] - (1 if r_edge else 0) # the main part of out, not including the partial edges

    # Compute the bulk of the blocks here
    f(_block_view(im, size), axis=2, out=out[:out_shp[0],:out_shp[1]])

    # Now do the straglers: bottom edge, right edge, and bottom-right corner
    if b_edge: f(_block_view(im[-b_edge:,:], (b_edge, size)), axis=2, out=out[-1,:out_shp[1]])
    if r_edge: f(_block_view(im[:,-r_edge:], (size, r_edge)), axis=2, out=out[:out_shp[0],-1])
    if b_edge and r_edge: f(im[-b_edge:,-r_edge:], axis=2, out=out[-1,-1])

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
    f = mean if method == 'mean' else median
    out_shp = ((im.shape[0]+size-1)//size, (im.shape[1]+size-1)//size)
    if im.ndim == 3 and im.shape[2] > 1: # Multi-channels images
        out = empty(out_shp+(im.shape[2],), dtype=im.dtype)
        for i in xrange(im.shape[2]): _im_bin(f, size, im[:,:,i], out[:,:,i])
    else: # Single-channel images
        out = empty(out_shp, dtype=im.dtype)
        _im_bin(f, size, im, out)
    return out

def resize(im, factor):
    """
    Shrink or grow an image using bicubic interpolation. If the factor is less than 1.0 the image is
    shrunk while if it is greater than 1.0 it is grown. For example, if it is 0.5 the image is
    halved in the x and y directions, while if it is 2.0 the image is doubled in the x and y
    directions.
    """
    from scipy.ndimage.interpolation import zoom
    check_image(im)
    if im.dtype.kind == 'c': raise ValueError('Complex type not supported')
    factor = float(factor)
    if factor <= 0.0: raise ValueError('Factor must be positive')
    if factor == 1.0: return im
    return zoom(im, (factor, factor) if im.ndim == 2 else (factor, factor, 1.0))


##### Image Stacks #####
class BinImageStack(FilteredImageStack):
    def __init__(self, ims, size=2, method='median'):
        self._size = int(size)
        if self._size < 2: raise ValueError('Unsupported binning size')
        if method not in ('mean', 'median'): raise ValueError('Unsupported binning method')
        self._f = mean if method == 'mean' else median
        super(BinImageStack, self).__init__(ims, BinImageSlice)
class BinImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        sh, sz = self._input.shape, self._stack._size
        self._set_props(self._input.dtype, ((sh[0]+sz-1)//sz, (sh[1]+sz-1)//sz))
    def _get_data(self):
        im, f, size = self._input.data, self._stack._f, self._stack._size
        out_shp = ((im.shape[0]+size-1)//size, (im.shape[1]+size-1)//size)
        self._set_props(self._input.dtype, out_shp)
        if im.ndim == 3 and im.shape[2] > 1: # Multi-channels images
            out = empty(out_shp+(im.shape[2],), dtype=im.dtype)
            for i in xrange(im.shape[2]): _im_bin(f, size, im[:,:,i], out[:,:,i])
        else: # Single-channel images
            out = empty(out_shp, dtype=im.dtype)
            _im_bin(f, size, im, out)
        return out

class ResizeImageStack(FilteredImageStack):
    def __init__(self, ims, factor):
        self._factor = float(factor)
        if self._factor <= 0.0: raise ValueError('Factor must be positive')
        super(ResizeImageStack, self).__init__(ims, ResizeImageSlice)
class ResizeImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        from scipy.version import version
        sh, f = self._input.shape, self._stack._factor
        sh = tuple((int(round(i*f)) for i in sh)
                   if splitstr(version, int, '.') >= [0,13,0] else
                   (int(i*f) for i in sh))
        self._set_props(self._input.dtype, sh)
    def _get_data(self):
        im = resize(self._input.data, self._stack._factor)
        self._set_props(self._input.dtype, im.shape[:2])
        return im


##### Commands #####
class BinCommand(CommandEasy):
    _size = None
    _method = None
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
        )
    @classmethod
    def _consumes(cls): return ('Full-size image',)
    @classmethod
    def _produces(cls): return ('Smaller, binned, image',)
    @classmethod
    def _see_also(cls): return ('resize',)
    def __str__(self): return 'bin %dx%d boxes using %s'%(self._size, self._size, self._method)
    def execute(self, stack): stack.push(BinImageStack(stack.pop(), self._size, self._method))
    
class ResizeCommand(CommandEasy):
    _factor = None
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
        )
    @classmethod
    def _consumes(cls): return ('Original image',)
    @classmethod
    def _produces(cls): return ('Resized image',)
    @classmethod
    def _see_also(cls): return ('bin',)
    def __str__(self): return 'resize to %.2fx'%(self._factor)
    def execute(self, stack): stack.push(ResizeImageStack(stack.pop(), self._factor))
