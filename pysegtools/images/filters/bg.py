"""Filters that change, remove, or add a background to the image."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from enum import Enum
from numbers import Integral
from itertools import islice

from numpy import zeros, ones, empty, array, vstack, hstack, dstack, dtype
from numpy import roll, flipud, fliplr, transpose
from numpy import mean, nonzero, sign, logical_and, logical_or, equal, argsort

from ._stack import FilteredImageStack, FilteredImageSlice, UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from ..types import check_image
from .._stack import ImageStack, Homogeneous
from ..colors import get_color, is_color
from ...imstack import CommandEasy, Opt
from ...general import DelayLoadedList

__all__ = ['get_bg_color','get_bg_padding','get_bg_mask','fill','fill_padding','crop','pad']

def get_bg_color(im):
    """Get the color of the background by looking at the edges of the image."""
    # Calculate bg color using solid strips on top, bottom, left, or right
    check_image(im)
    x = im[0,0,...]
    if (im[0,1:,...] == x).all() or (im[1:,0,...] == x).all(): return x
    x = im[-1,-1,...]
    if (im[-1,:-1,...] == x).all() or (im[:-1,-1,...] == x).all(): return x
    return None

def get_bg_padding(im, bg_color=None):
    """
    Get the area of the background as the amount on the top, left, bottom, and right. If bg is not
    given, it is calculated from the edges of the image.
    """
    check_image(im)
    if bg_color is None:
        bg_color = get_bg_color(im)
        if bg_color is None: return (0,0,0,0) # no discoverable bg color, no paddding
    else:
        bg_color = get_color(bg_color, im)
    w,h = im.shape
    t,l,b,r = 0, 0, h-1, w-1
    mask = empty(max(w,h), dtype=bool)
    while t < h and equal(im[t,:,...], bg_color, out=mask[:w]).all(): t += 1
    while b > t and equal(im[b,:,...], bg_color, out=mask[:w]).all(): b -= 1
    while l < w and equal(im[:,l,...], bg_color, out=mask[:h]).all(): l += 1
    while r > l and equal(im[:,r,...], bg_color, out=mask[:h]).all(): r -= 1
    return (t,l,h-1-b,w-1-r)

def get_bg_mask(im, bg_color=None):
    """
    Get background pixel mask. These pixels are on the outside and are all the same color. However,
    unlike get_bg_color/get_bg_padding, this function does not assume a rectangular region as the
    foreground. If the background color is not given it is calculated (in which case there must be
    at least 4 pixels grouped along the edge [or 3 in a corner] to find the background color).

    Returns a mask of the background pixels. The background color can be obtained using im[bg][0]
    (if there are any background pixels, if bg.any()).
    """
    #pylint: disable=invalid-unary-operand-type
    from scipy.ndimage.morphology import binary_fill_holes
    from scipy.stats import mode

    check_image(im)
    if im.ndim == 2: im = im[:,:,None]
    if bg_color is None:
        h,w = im.shape[:2]
        flat_x = vstack(((im[1:,:] == im[:-1,:]).all(2), ones((1, w), dtype=bool)))
        flat_y = hstack(((im[:,1:] == im[:,:-1]).all(2), ones((h, 1), dtype=bool)))
        flat = dstack((flat_x, roll(flat_x, 1, axis=0), flat_y, roll(flat_y, 1, axis=1))).all(2)
        fg = binary_fill_holes(~flat)
        bg_color = mode(im[~fg])[0][0]
        #bg_color = unique(im[~fg], return_counts=True)...?
    return ~binary_fill_holes((im != bg_color).any(2))

def padding2mask(im, padding):
    """
    Converts a padding tuple (from get_bg_padding) to a mask (like from get_bg_mask). The image is
    required so the shape is known (the first argument can also be the image shape).

    The reverse conversion can be done with get_bg_padding(mask, True).
    """
    if len(padding) != 4 or any(not isinstance(x, (int, long)) for x in padding): raise ValueError
    t,l,b,r = padding
    h,w = im[:2] if isinstance(im, (tuple, list)) else im.shape[:2]
    B,R = h-b,w-r
    bg = zeros((h,w), dtype=bool)
    bg[ :t, : ,...] = True
    bg[B: , : ,...] = True
    bg[t:B, :l,...] = True
    bg[t:B,R: ,...] = True
    return bg

def __check_padding(padding, h, w):
    return len(padding) == 4 and all(isinstance(x, Integral) and x>=0 for x in padding) and \
           padding[0]+padding[2] < h and padding[1]+padding[3] < w

def __mean(im, axis=2):
    if im.ndim == axis or im.shape[axis] == 1: return mean(im)
    return array([mean(im[...,i]) for i in xrange(im.shape[axis])])\

def fill_padding(im, padding=None, fill='black'): #pylint: disable=redefined-outer-name
    """
    Fills the 'background' of the image. The background is given by the amount of padding (top,
    left, bottom, right). If the padding is not given, it is calculated with get_bg_padding().

    The fill value specifies how the background pixels are filled in, it can be one of:
     * 'reflect' - every edge is reflected into the background
     * 'mirror'  - like reflect except the reflected edge is not duplicated
     * 'wrap'    - wraps the foreground around, like tiling
     * 'nearest' - every background is set to the value of the nearest foreground value 
     * 'mean'    - solid, with the mean foreground color
     * Any value supported by get_color for the image type - solid, with that color
    Default is black/0.

    If the image is writeable, the values are changed directly. Otherwise a copy is made. In both
    cases the modified image is returned.
    """
    check_image(im)
    h,w = im.shape[:2]
    if padding is None: padding = get_bg_padding(im)
    elif not __check_padding(padding, h, w): raise ValueError
    if all(x==0 for x in padding): return im
    if not im.flags.writeable: im = im.copy()
    t,l,b,r = padding #pylint: disable=unpacking-non-sequence
    B,R = h-b,w-r
    fg = im[t:B,l:R]
    
    if fill in ('reflect', 'mirror'):
        off = 1 if fill == 'mirror' else 0 # mirror shifts everything by 1
        H,W = h-b-t-off, w-r-l-off # foreground width/height
        # Complete all the rows first
        while H < t: im[t-H:t+off, l:R, ...] = flipud(fg); t -= H;         H *= 2; fg = im[t:B,l:R]
        while H < b: im[b-off:b-H, l:R, ...] = flipud(fg); b -= H; B += H; H *= 2; fg = im[t:B,l:R]
        im[:t+off, l:R, ...] = flipud(fg[:t+off, :, ...])
        if b != 0: im[B-off:, l:R, ...] = flipud(fg[-b-off:, :, ...])
        fg = im[:,l:R]
        # Now the columns
        while W < l: im[:, l-W:l+off, ...] = fliplr(fg); l -= W;         W *= 2; fg = im[t:B,l:R]
        while W < r: im[:, r+off:r-W, ...] = fliplr(fg); r -= W; R += W; W *= 2; fg = im[t:B,l:R]
        im[:, :l+off, ...] = fliplr( fg[:, :l+off, ...] )
        if r != 0: im[:, R-off:, ...] = fliplr( fg[:, -r-off:, ...] )
        # Corners are done automatically
        
    elif fill == 'wrap':
        H,W = h-b-t, w-r-l # foreground width/height
        # Complete all the rows first
        while H < t: im[t-H:t,l:R,...] = fg; t -= H;         H *= 2; fg = im[t:B,l:R]
        while H < b: im[b:b-H,l:R,...] = fg; b -= H; B += H; H *= 2; fg = im[t:B,l:R]
        if t != 0: im[:t,l:R,...] = fg[-t:,:,...]
        im[B:,l:R,...] = fg[:b,:,...]
        fg = im[:,l:R]
        # Now the columns
        while W < l: im[:,l-W:l,...] = fg; l -= W;         W *= 2; fg = im[t:B,l:R]
        while W < r: im[:,r:r-W,...] = fg; r -= W; R += W; W *= 2; fg = im[t:B,l:R]
        if l != 0: im[:,:l,...] = fg[:,-l:,...]
        im[:,R:,...] = fg[:,:r,...]
        # Corners are done automatically
        
    elif fill == 'nearest':
        im[ :t,l:R,...] = fg[  :1,  : ,...].repeat(t,0)
        im[t:B, :l,...] = fg[  : ,  :1,...].repeat(l,1)
        im[B: ,l:R,...] = fg[-1: ,  : ,...].repeat(b,0)
        im[t:B,R: ,...] = fg[  : ,-1: ,...].repeat(r,1)
        im[ :t, :l,...] = fg[ 0, 0,...] # corners
        im[ :t,R: ,...] = fg[ 0,-1,...]
        im[B: , :l,...] = fg[-1, 0,...]
        im[B: ,R: ,...] = fg[-1,-1,...]
        
    else:
        fill = __mean(fg) if fill == 'mean' else get_color(fill, im)
        im[ :t, : ,...] = fill
        im[B: , : ,...] = fill
        im[t:B, :l,...] = fill
        im[t:B,R: ,...] = fill
        
    return im

#__tree_eps = 1e-6 # 1e-6 allows almost a distance of 500k to still be distinguishable
def __nearest_neighbors(mask):
    bg_idx, fg_idx = nonzero(mask), nonzero(~mask)
    try:
        from scipy.spatial import cKDTree as KDTree #pylint: disable=no-name-in-module
    except ImportError:
        from scipy.spatial import KDTree
    tree = KDTree(transpose(fg_idx))
    dist,i = tree.query(transpose(bg_idx))
    return bg_idx, fg_idx, dist, i

def fill(im, mask=None, fill='black'): #pylint: disable=redefined-outer-name
    """
    Fills the 'background' of the image. The background is given by the mask (True is background).
    If the background is not given, it is calculated with get_bg_mask().

    The fill value specifies how the background pixels are filled in, it can be one of:
     * 'reflect' - like mirror but duplicates the values along the interface of background and foreground
     * 'mirror'  - background is filled with a reflection of the closest foreground pixel
     * 'nearest' - every background is set to the value of the nearest foreground value 
     * 'mean'    - solid, with the mean foreground color
     * Any value supported by get_color for the image type - solid, with that color
    Default is black/0. Mirror and reflect will have undefined values in some cases where there is a
    lot of background and thin portions of foreground.

    If the image is writeable, the values are changed directly. Otherwise a copy is made. In both
    cases the modified image is returned.
    """
    check_image(im)
    h,w = im.shape[:2]
    if mask is None: mask = get_bg_mask(im)
    elif mask.shape != (h,w) or mask.dtype != bool or all(mask.flat): raise ValueError
    if not any(mask.flat): return im #pylint: disable=no-member
    if not im.flags.writeable: im = im.copy()
    
    if fill in ('reflect', 'mirror'):
        bg_idx, fg_idx, dist, i = __nearest_neighbors(mask)
        di = argsort(dist)
        dst_y, dst_x = bg_idx
        src_y, src_x = fg_idx[0][i], fg_idx[1][i]
        if fill == 'mirror':
            src_y, src_x = 2*src_y-dst_y, 2*src_x-dst_x
        else: # fill == 'reflect'
            src_y, src_x = 2*src_y-dst_y-sign(src_y-dst_y), 2*src_x-dst_x-sign(src_x-dst_x)
        valids = logical_and(logical_and(src_y>=0,src_y<h), logical_and(src_x>=0,src_x<w))
        di = di.compress(valids.take(di))
        # At this point we need to set every src pixel to its dst pixel's value
        # However, this needs to be done starting close to the foreground so that we get mirrors-of-mirrors
        ##im[dst_y[di], dst_x[di]] = im[src_y[di], src_x[di]] # doesn't work since it doesn't go in the right order
        # This is the straight forward way to go in order
        for DY,DX,SY,SX in zip(dst_y.take(di), dst_x.take(di), src_y.take(di), src_x.take(di)):
            im[DY, DX] = im[SY, SX]
        
    elif fill == 'nearest':
        bg_idx, fg_idx, _, i = __nearest_neighbors(mask)
        im[bg_idx[0], bg_idx[1]] = im[fg_idx[0][i], fg_idx[1][i]]

    elif fill == 'wrap': raise ValueError
    
    else: im[mask] = __mean(im[~mask], 1) if fill == 'mean' else get_color(fill, im) #pylint: disable=invalid-unary-operand-type
    
    return im


def crop(im, padding=None):
    """
    Crops an image, removing the padding from each side (top, left, bottom, right). If the padding
    is not given, it is calculated with get_bg_padding. Returns a view, not a copy.
    """
    check_image(im)
    h,w = im.shape[:2]
    if padding is None: padding = get_bg_padding(im)
    elif not __check_padding(padding, h, w): raise ValueError
    t,l,b,r = padding #pylint: disable=unpacking-non-sequence
    return im[t:h-b,l:w-r,...]

def pad(im, padding, alloc=zeros):
    """Pad an image with 0s. The amount of padding is given by top, left, bottom, and right."""
    check_image(im)
    h,w = im.shape[:2]
    if not __check_padding(padding, h, w): raise ValueError
    t,l,b,r = padding
    im_out = alloc((h+b+t,w+r+l,im.shape[2]),dtype=im.dtype)
    im_out[t:t+h,l:l+w,...] = im
    return im_out


##### Stack Classes #####

class PaddingForImageStack(DelayLoadedList):
    """A sequence of paddings from an image stack that calculates as needed."""
    def __init__(self, ims, color=None):
        ims = ImageStack.as_image_stack(ims)
        super(PaddingForImageStack, self).__init__(len(ims))
        self.__ims, self.__color = ims, color
    def _loaditem(self, i):
        return get_bg_padding(self.__ims[i].data, self.__color)
    @staticmethod
    def as_padding_list(mask):
        #pylint: disable=protected-access
        return mask._padding if isinstance(mask, BackgroundMask) and hasattr(mask, '_padding') else PaddingForImageStack(mask, True)
class MinPaddingForImageStack(DelayLoadedList): # "Conservative"
    """
    Like PaddingForImageStack except all values are the minimum padding. This is designed for
    homogeneous data sets so that they stay homoegeneous and all foreground pixels are included
    at the expense of still possibly having some backgrond pixels.
    """
    def __init__(self, ims, color=None):
        ims = ImageStack.as_image_stack(ims)
        super(MinPaddingForImageStack, self).__init__(len(ims))
        self.__ims, self.__color = ims, color
    def _loaditem(self, i):
        # loads all once the first one is requested
        min_t,min_l,min_b,min_r = get_bg_padding(self.__ims[0].data, self.__color)
        for im in islice(self.__ims, 1, None):
            t,l,b,r = get_bg_padding(im.data, self.__color)
            if t < min_t: min_t = t
            if l < min_l: min_l = l
            if b < min_b: min_b = b
            if r < min_r: min_r = r
        p = (min_t,min_l,min_b,min_r)
        self._data = [p] * len(self._data)
        return p
class MaxPaddingForImageStack(DelayLoadedList): # "Aggressive"
    """
    Like PaddingForImageStack except all values are the maximum padding. This is designed for
    homogeneous data sets so that they stay homoegeneous and all background pixels are excluded
    at the expense of possibly removing some foregrond pixels.
    """
    def __init__(self, ims, color=None):
        ims = ImageStack.as_image_stack(ims)
        super(MaxPaddingForImageStack, self).__init__(len(ims))
        self.__ims, self.__color = ims, color
    def _loaditem(self, i):
        # loads all once the first one is requested
        max_t,max_l,max_b,max_r = get_bg_padding(self.__ims[0].data, self.__color)
        for im in islice(self.__ims, 1, None):
            t,l,b,r = get_bg_padding(im.data, self.__color)
            if t > max_t: max_t = t
            if l > max_l: max_l = l
            if b > max_b: max_b = b
            if r > max_r: max_r = r
        p = (max_t,max_l,max_b,max_r)
        self._data = [p] * len(self._data)
        return p


class MaskProjection(Enum): #pylint: disable=no-init
    None_ = 0
    All = 1
    Any = 2
class BackgroundMask(FilteredImageStack):
    """Calculates the background mask for an image stack."""
    def __init__(self, ims, color=None, rect=False, mode=MaskProjection.None_):
        self._color = color
        if rect:
            self._padding = {
                MaskProjection.None_ : PaddingForImageStack,
                MaskProjection.All   : MinPaddingForImageStack,
                MaskProjection.Any   : MaxPaddingForImageStack,
            }[mode](ims, color)
            super(BackgroundMask, self).__init__(ims, BackgroundMaskSlice_Rect)
        elif mode is MaskProjection.None_:
            super(BackgroundMask, self).__init__(ims, BackgroundMaskSlice)
        else:
            self._op, self._mask = {
                MaskProjection.All : logical_and,
                MaskProjection.Any : logical_or}[mode], None
            super(BackgroundMask, self).__init__(ims, BackgroundMaskSlice_OP)
        self._dtype, self._homogeneous = dtype(bool), Homogeneous.DType
class BackgroundMaskSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self): self._set_props(dtype(bool), self._input.shape)
    def _get_data(self): return get_bg_mask(self._input.data, self._stack._color)
class BackgroundMaskSlice_OP(BackgroundMaskSlice):
    #pylint: disable=protected-access
    def _get_data(self):
        if self._stack._mask is None:
            op, c, ims = self._stack._op, self._stack._color, self._stack._ims
            mask = get_bg_mask(ims[0], c)
            for im in islice(ims, 1, None): op(mask, get_bg_mask(im, c), mask)
            self._stack._mask = mask
        return self._stack._mask
class BackgroundMaskSlice_Rect(BackgroundMaskSlice):
    #pylint: disable=protected-access
    def _get_data(self): return padding2mask(self.shape, self._stack._padding[self._z])


class FillImageStack(UnchangingFilteredImageStack):
    """Calculates the background mask for an image stack."""
    def __init__(self, ims, fill, mask_or_padding): #pylint: disable=redefined-outer-name
        """mask_or_padding must derive from ImageStack if it is a mask"""
        if len(mask_or_padding) != len(ims): raise ValueError('mask/padding must be same shape as image')
        self._fill = fill
        if isinstance(mask_or_padding, ImageStack):
            if mask_or_padding.dtype != bool: raise ValueError('mask must be of bool/logical type')
            super(FillImageStack, self).__init__(ims, FillImageSlice, mask_or_padding)
        else:
            self._padding = mask_or_padding
            super(FillImageStack, self).__init__(ims, FillPaddingImageSlice)
class FillImageSlice(UnchangingFilteredImageSlice):
    #pylint: disable=protected-access
    def __init__(self, image, stack, z, mask):
        super(FillImageSlice, self).__init__(image, stack, z)
        self._mask = mask = mask[z]
        if mask.shape != image.shape: raise ValueError('mask must be same shape as image')
    def _get_data(self): return fill(self._input.data, self._mask.data, self._stack._fill) # TODO: .copy()?
class FillPaddingImageSlice(UnchangingFilteredImageSlice):
    #pylint: disable=protected-access
    def _get_data(self): return fill_padding(self._input.data, self._stack._padding[self._z], self._stack._fill) # TODO: .copy()?


class CropImageStack(FilteredImageStack):
    def __init__(self, ims, padding):
        """
        padding can also be a background mask image stack in which case it is automatically wrapped
        with a PaddingForImageStack(mask, True) object. However in that case it must already derive
        from ImageStack (so use ImageStack.as_image_stack on it). Providing padding as a background
        mask actually allows for additional checks to be performed during construction.
        """
        if isinstance(padding, ImageStack):
            mask = padding
            if mask.dtype != bool: raise ValueError('mask must be of bool/logical type')
            if any(m.shape != im.shape for m,im in zip(mask,ims)): raise ValueError('mask must be same shape as image')
            padding = PaddingForImageStack.as_padding_list(mask)
        if len(padding) != len(ims): raise ValueError('image stack and padding sequence must be same length')
        super(CropImageStack, self).__init__(ims, CropImageSlice)
        self._padding = padding
class CropImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        p, s = self._stack._padding[self._z], self._input.shape
        self._set_props(self._input.dtype, (s[0]-p[0]-p[2], s[1]-p[1]-p[3]))
    def _get_data(self): return crop(self._input.data, self._stack._padding[self._z])


class PadImageStack(FilteredImageStack):
    def __init__(self, ims, fill, mask_or_padding): #pylint: disable=redefined-outer-name
        """
        mask_or_padding must derive from ImageStack if it is a mask. If it is a mask, the padding
        representation of it is used to create the actual padding but the mask is used to fill.
        """
        if len(mask_or_padding) != len(ims): raise ValueError('image stack and padding sequence must be same length')
        if isinstance(mask_or_padding, ImageStack):
            mask = mask_or_padding
            if mask.dtype != bool: raise ValueError('mask must be of bool/logical type')
            self._padding = PaddingForImageStack.as_padding_list(mask)
            super(PadImageStack, self).__init__(ims, [PadImageSlice(im, self, z, m) for z,(im,m) in enumerate(zip(ims, mask))])
        else:
            self._padding = mask_or_padding
            super(PadImageStack, self).__init__(ims, PadImageSlice, None)
        self._fill = fill
class PadImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def __init__(self, image, stack, z, mask):
        super(PadImageSlice, self).__init__(image, stack, z)
        self._mask = mask
    def _get_props(self):
        p, s = self._stack._padding[self._z], self._input.shape
        self._set_props(self._input.dtype, (s[0]+p[0]+p[2], s[1]+p[1]+p[3]))
    def _get_data(self):
        padding = self._stack._padding[self._z]
        im = pad(self._input.data, padding, empty)
        return fill_padding(im, padding, self._stack._fill) if self._mask is None else fill(im, self._mask, self._stack._fill)


##### Commands #####

_cast_fill = Opt.cast_or('mean', 'mirror', 'reflect', 'nearest', 'wrap', Opt.cast_check(is_color))

class BackgroundMaskCommand(CommandEasy):
    _color = None
    _rect = None
    _projection = None
    @classmethod
    def name(cls): return 'background mask'
    @classmethod
    def flags(cls): return ('bg-mask',)
    @classmethod
    def _desc(cls): return 'Calculate the background area as a mask, calculated from solid regions of color on the sides of the images. This color can be calculated from the image slices themselves or a predetermined color. This operates per-slice and not on the whole volume.'
    @classmethod
    def _opts(cls): return (
        Opt('color', 'The current background color (see --help colors) or \'auto\' to calculate it per-slice',
            Opt.cast_or('auto', Opt.cast_check(is_color)), 'auto'),
        Opt('rect',  'Force the background area to be around a rectangular foreground',
            Opt.cast_bool(), False),
        Opt('projection', 'Make every slice the same where a pixel is marked as background if: only all slices would have had it marked as background (all) or any slice would have had it marked as background (any)',
            Opt.cast_lookup({'none':MaskProjection.None_,'all':MaskProjection.All,'any':MaskProjection.Any}), MaskProjection.None_, 'none'),
        )
    @classmethod
    def _consumes(cls): return ('Image to calculate background mask of',)
    @classmethod
    def _produces(cls): return ('Background mask',)
    @classmethod
    def _see_also(cls): return ('fill', 'crop', 'pad', 'invert', 'threshold', 'colors')
    def __str__(self): return 'calculate background padding '+(('of color %s '%self._color) if self._color!='auto' else '')+('(rectangular) ' if self._rect else '')+('(projected as %s)'%('all' if self._projection == MaskProjection.All else 'any') if self._projection != MaskProjection.None_ else '')
    def execute(self, stack): stack.push(BackgroundMask(stack.pop(), None if self._color=='auto' else self._color, self._rect, self._projection))

class FillCommand(CommandEasy):
    _fill = None
    _rect = None
    @classmethod
    def name(cls): return 'fill'
    @classmethod
    def flags(cls): return ('fill',)
    @classmethod
    def _desc(cls): return 'Fill in a masked area. Besides solid colors, the fill color can be based on the non-masked area: either the mean value, mirrored, nearest, or wrapped.'
    @classmethod
    def _opts(cls): return (
        Opt('fill', 'The name of a color (see --help colors) to use to fill with, or one of the special values \'mean\', \'mirror\', \'reflect\', \'nearest\', or \'wrap\'',
            _cast_fill, 'black'),
        Opt('rect', 'Force the non-masked area to be a rectangular space; this can cause large speedups with the special fills and allows \'wrap\'',
            Opt.cast_bool(), False),
        )
    @classmethod
    def _consumes(cls): return ('Image to fill in','Mask to fill')
    @classmethod
    def _produces(cls): return ('Image with mask area filled in',)
    @classmethod
    def _see_also(cls): return ('bg-mask', 'crop', 'pad', 'colors')
    def __str__(self): return 'fill %swith %s'%(
        ('outside rectangular area ' if self._rect else ''),
        {'mean':'mean color','reflect':'reflection','wrap':'wrapping'}.get(self._fill, self._fill))
    def __init__(self, args, stack):
        super(FillCommand, self).__init__(args, stack)
        if self._fill is 'wrap' and not self._rect: raise ValueError('wrap cannot be used unless rect is true')
    def execute(self, stack):
        ims, mask = stack.pop(), stack.pop()
        stack.push(FillImageStack(ims, self._fill, PaddingForImageStack.as_padding_list(mask) if self._rect else mask))

class CropCommand(CommandEasy):
    @classmethod
    def name(cls): return 'crop'
    @classmethod
    def flags(cls): return ('c','crop')
    @classmethod
    def _desc(cls): return 'Crop out the masked area so that marked areas are removed. This forces the mask to be around a rectangular region.'
    @classmethod
    def _consumes(cls): return ('Image to crop','Mask to crop out')
    @classmethod
    def _produces(cls): return ('Cropped image',)
    @classmethod
    def _see_also(cls): return ('bg-mask', 'fill', 'pad')
    def __str__(self): return 'crop'
    def execute(self, stack):
        ims, mask = stack.pop(), stack.pop()
        stack.push(CropImageStack(ims, mask))

class PadCommand(CommandEasy):
    _fill = None
    _rect = None
    @classmethod
    def name(cls): return 'pad'
    @classmethod
    def flags(cls): return ('p','pad','uncrop')
    @classmethod
    def _desc(cls): return """
Pad an image by adding the masked region around an image. The rectangular region around the
non-masked area must be the same size as the image being padded and the padded image becomes the
size of the mask. This may be a bit confusing at first, but is designed to work in reverse from the
crop command so that you can crop an image then restore it using the same mask. The fill option is
the color to fill in the entire masked area (even if it was inside the rectangle) unless rect is
specified."""
    @classmethod
    def _opts(cls): return (
        Opt('fill', 'The name of a color (see --help colors) to use to fill with, or one of the special values \'mean\', \'mirror\', \'reflect\', \'nearest\', or \'wrap\'',
            _cast_fill, 'black'),
        Opt('rect', 'Only fill in the added padding region; this can cause large speedups with the special fills and allows \'wrap\'',
            Opt.cast_bool(), False),
        )
    @classmethod
    def _consumes(cls): return ('Image to pad','Mask to pad with')
    @classmethod
    def _produces(cls): return ('Padded image',)
    @classmethod
    def _see_also(cls): return ('bg-mask', 'fill', 'crop', 'colors')
    def __str__(self): return 'pad and fill %swith %s'%(
        ('padding ' if self._rect else ''),
        {'mean':'mean color','reflect':'reflection','wrap':'wrapping'}.get(self._fill, self._fill))
    def execute(self, stack):
        ims, mask = stack.pop(), stack.pop()
        stack.push(PadImageStack(ims, self._fill, PaddingForImageStack.as_padding_list(mask) if self._rect else mask))
