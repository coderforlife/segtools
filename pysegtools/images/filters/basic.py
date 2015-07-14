"""Basic Image Filters"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Integral
from collections import Sequence
from itertools import izip

from numpy import flipud, fliplr, rot90

from ._stack import FilteredImageStack, FilteredImageSlice, UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from .._stack import ImageStack, ImageSlice
from ..source import ImageSource
from ..types import check_image, get_im_dtype_and_nchan, create_im_dtype, get_im_min_max, get_dtype_max
from ...imstack import Command, CommandEasy, Opt, Help

__all__ = ['flip','rotate','inv',
           'FlipImageStack','RotateImageStack','InvertImageStack',
           'ExtractChannelsImageStack','CombineChannelsImageStack']

##### 2D #####
def flip(im, direction='v'):
    """
    Flips an image either vertically (default) or horizontally (by giving an 'h'). The returned
    value is a view - not a copy.
    """
    check_image(im)
    if direction not in ('v', 'h'): raise ValueError('Unsupported direction')
    return (flipud if direction == 'v' else fliplr)(im)

def rotate(im, direction='cw'):
    """
    Rotates an image either 90 degrees clockwise, 90 degrees counter-clockwise, or a full 180
    degrees. A view may be returned in some cases. The direction can be 'cw', 'ccw', or 'full'.
    """
    check_image(im)
    try: return rot90(im, ('ccw', 'full', 'cw').index(direction) + 1)
    except ValueError: raise ValueError('Unsupported direction')

def inv(im):
    """
    Inverts each image in an image stack, essentially making the lowest value the highest and vice
    versa. Works on all image types except complex.
    """
    from numpy import logical_not, dtype
    check_image(im)
    if im.dtype.kind == 'c': raise ValueError('Cannot invert complex numbers')
    if im.dtype.kind == 'b': return logical_not(im)
    if im.dtype.kind == 'u': return get_dtype_max(im.dtype) - im
    if im.dtype.kind == 'i':
        u_dt = dtype(im.dtype.byteorder+'u'+str(im.dtype.itemsize))
        return (get_dtype_max(u_dt)-im.view(u_dt)).view(im.dtype)
    # im.dtype.kind == 'f'
    mn, mx = get_im_min_max(im)
    if mn == 0.0: return mx - im
    return mx - im + mn


##### 3D #####
class FlipImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims, direction='y'):
        """dir can be x, y, or z"""
        self._dir = direction
        if direction == 'z':
            slcs = [DoNothingFilterImageSlice(im,self,z) for z,im in enumerate(reversed(ims))]
        elif direction in ('x','y'):
            self._flip = flipud if dir == 'y' else fliplr
            slcs = FlipImageSlice
        else:
            raise ValueError()
        super(FlipImageStack, self).__init__(ims, slcs)
class DoNothingFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return self._input.data
class FlipImageSlice(UnchangingFilteredImageSlice):
    #pylint: disable=protected-access
    def _get_data(self): return self._stack._flip(self._input.data)


class RotateImageStack(FilteredImageStack):
    def __init__(self, ims, direction='cw'):
        """dir can be cw, cww, or full"""
        self._dir = direction
        try: self._k = ('ccw', 'full', 'cw').index(direction) + 1
        except ValueError: raise ValueError('Unsupported direction')
        super(RotateImageStack, self).__init__(ims, RotateImageSlice)
class RotateImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def _get_props(self):
        self._set_props(self._input.dtype, self._input.shape[::(-1 if (self._stack._k&1) else 1)])
    def _get_data(self): return rot90(self._input.data, self._stack._k)

    
class InvertImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims): super(InvertImageStack, self).__init__(ims, InvertImageSlice)
class InvertImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return inv(self._input.data)
    

class ExtractChannelsImageStack(FilteredImageStack):
    def __init__(self, ims, channels):
        """channels must be an integer value (0-based) or a sequence of integers."""
        
        ims = [ImageSource.as_image_source(im) for im in ims]
        if isinstance(channels, Integral):
            channels = [channels]
        elif not (isinstance(channels, Sequence) and all(isinstance(c,Integral) for c in channels)):
            raise ValueError('channel number is invalid')
        dtypes,nchans = [list(i) for i in izip(get_im_dtype_and_nchan(dt) for dt in im.dtype for im in ims)]
        if len(channels) == 0 or min(channels) < 0 or max(channels) >= min(nchans):
            raise ValueError('channel number is invalid')
        super(ExtractChannelsImageStack, self).__init__(ims,
            [ExtractChannelsImageSlice(self,z,im,dt,channels) for z,(im,dt) in enumerate(zip(ims,dtypes))])
class ExtractChannelsImageSlice(FilteredImageSlice):
    def __init__(self, stack, z, im, dt, channels):
        super(ExtractChannelsImageSlice, self).__init__(stack, z, im)
        self.__channels = channels
        self._set_props(create_im_dtype(dt, dt.byteorder, len(channels)), None)
    def _get_props(self):
        self._set_props(None, self._input.shape)
    def _get_data(self):
        im = self._input.data
        if im.ndim == 2: im = im[:,:,None]
        return im.take(self.__channels, axis=2)


class CombineChannelsImageStack(FilteredImageStack):
    def __init__(self, imss):
        from itertools import islice
        from numpy import sum #pylint: disable=redefined-builtin
        self._imss = imss = [ImageStack.as_image_stack(ims) for ims in imss]
        if len(imss) == 0: raise ValueError('no channels to combine')
        d = len(imss[0])
        if any(d != len(ims) for ims in islice(imss, 1, None)):
            raise ValueError('all image stacks must have the same depth')
        dtypes_and_nchans = [[get_im_dtype_and_nchan(im.dtype) for im in ims] for ims in imss]
        dtypes = [[dtype for dtype,_ in ims] for ims in dtypes_and_nchans]
        nchans = [[nchan for _,nchan in ims] for ims in dtypes_and_nchans]
        nchans = sum(nchans, axis=0)
        if any(dtypes[0] != dtypes_col for dtypes_col in islice(dtypes, 1, None)):
            raise ValueError('all image stacks must have the same data type for each slice')
        shapes = [im.shape for im in imss[0]]
        if any(shapes != [im.shape for im in ims] for ims in islice(imss, 1, None)):
            raise ValueError('all image stacks must have the same shape for each slice')
        dtypes = [create_im_dtype(dt, dt.byteorder, nc) for dt,nc in zip(dtypes[0],nchans)]
        super(CombineChannelsImageStack, self).__init__(None,
            [CombineChannelsImageSlice(self,z,dt,sh) for z,(dt,sh) in enumerate(zip(dtypes,shapes))])
class CombineChannelsImageSlice(ImageSlice):
    #pylint: disable=protected-access
    def __init__(self, stack, z, dtype, shape):
        super(CombineChannelsImageSlice, self).__init__(stack, z)
        self._set_props(dtype, shape)
    def _get_props(self): pass
    def _get_data(self):
        from numpy import empty
        dst, chan = empty(self._shape, dtype=self._dtype), 0
        if dst.ndim == 2: dst = dst[:,:,None]
        z = self._z
        for ims in self._stack._imss:
            im = ims[z].data
            if im.ndim == 2: im = im[:,:,None]
            dst[:,:,chan:chan+im.shape[2]] = im
            chan += im.shape[2]
        return dst


##### Commands #####
class FlipImageCommand(CommandEasy):
    _dir = None
    @classmethod
    def name(cls): return 'flip'
    @classmethod
    def _desc(cls): return 'Flips the images in the image stack, either in the x, y, and z directions.'
    @classmethod
    def flags(cls): return ('f', 'flip')
    @classmethod
    def _opts(cls): return (
        Opt('dir', 'The direction of the flip: x (left-to-right), y (top-to-bottom), or z (first-to-last)', Opt.cast_in('x','y','z'), 'y'),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be flipped',)
    @classmethod
    def _produces(cls): return ('Flipped image stack',)
    @classmethod
    def _see_also(cls): return ('z','rotate')
    def __str__(self): return 'flip with dir=%s'%self._dir
    def execute(self, stack): stack.push(FlipImageStack(stack.pop(), self._dir))

class RotateImageCommand(CommandEasy):
    _dir = None
    @classmethod
    def name(cls): return 'rotate'
    @classmethod
    def _desc(cls): return 'Rotates the images in the image stack, either 90 degrees clockwise, 90 degrees count-clockwise, or a full 180 degrees.'
    @classmethod
    def flags(cls): return ('R', 'rotate')
    @classmethod
    def _opts(cls): return (
        Opt('dir', 'The direction of the flip: cw, ccw, or full', Opt.cast_in('cw','cww','full'), 'cw'),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be rotated',)
    @classmethod
    def _produces(cls): return ('Rotated image stack',)
    @classmethod
    def _see_also(cls): return ('flip',)
    def __str__(self): return 'rotate with dir=%s'%self._dir
    def execute(self, stack): stack.push(RotateImageStack(stack.pop(), self._dir))

class InvertImageCommand(CommandEasy):
    @classmethod
    def name(cls): return 'invert'
    @classmethod
    def _desc(cls): return 'Inverts the images in the image stack, making dark light and light dark.'
    @classmethod
    def flags(cls): return ('i', 'invert')
    @classmethod
    def _consumes(cls): return ('Image stack to be inverted',)
    @classmethod
    def _produces(cls): return ('Inverted image stack',)
    @classmethod
    def _see_also(cls): return ('bw','scale')
    def __str__(self): return 'invert'
    def execute(self, stack): stack.push(InvertImageStack(stack.pop()))

class ExtractChannelsCommand(CommandEasy):
    @staticmethod
    def cast(x):
        from .._util import splitstr
        x = splitstr(x, int, ',')
        if not all(0 <= x < 5): raise ValueError()
        return x

    _channels = None
    @classmethod
    def name(cls): return 'extract channels'
    @classmethod
    def flags(cls): return ('extract-channels',)
    @classmethod
    def _desc(cls): return 'Extracts and reorganizes channels of an image stack.'
    @classmethod
    def _opts(cls): return (
        Opt('channels', 'The channels to extract, which are 0-based and comma-seperated, to take in order', ExtractChannelsCommand.cast),
        )
    @classmethod
    def _consumes(cls): return ('The image to extract channels from',)
    @classmethod
    def _produces(cls): return ('The image with just the extracted channels in the order given',)
    @classmethod
    def _see_also(cls): return ('combine-channels')
    def __str__(self):
        from .._util import itr2str
        return ('extract channels '+itr2str(self._channels,',')) if len(self._channels)>1 else 'extract channel '+str(self._channels[0])
    def execute(self, stack): stack.push(ExtractChannelsImageStack(stack.pop(), self._channels))

class CombineChannelsCommand(Command):
    @classmethod
    def name(cls): return 'combine channels'
    @classmethod
    def flags(cls): return ('combine-channels',)
    @classmethod
    def _opts(cls): return (
            Opt('nstacks',    'The number of stacks to combine', Opt.cast_int(lambda x:x>=2), 2),
            )
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Combine Channels")
        p.text("""Combine several image stacks into a single stack with as channels.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("""
Consumes:  2+ image stacks 
Produces:  1 combined image stack""")
        p.newline()
        p.text("Command format:")
        p.cmds("--combine-channels [nstacks]")
        p.newline()
        p.text("Options:")
        p.opts(*cls._opts())
        p.newline()
        p.text("""
Combines two or more image stacks into a single stack by adding the additional stacks as extra channels
(3rd dimension of each image). The next image stack is used first, followed by the second-to-next, and
so forth.""")
        p.newline()
        p.text("See also:")
        p.list('extract-channels')
    def __str__(self): return "combining channels of %d image stacks"%self.__nstacks
    def __init__(self, args, stack):
        self.__nstacks, = args.get_all(*CombineChannelsCommand._opts())
        for _ in xrange(self.__nstacks): stack.pop()
        stack.push()
    def execute(self, stack): stack.push(CombineChannelsImageStack(stack.pop() for _ in xrange(self.__nstacks)))
