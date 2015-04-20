"""Basic Image Filters"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Integral
from collections import Sequence
from itertools import izip
        
from ._stack import FilteredImageStack, FilteredImageSlice, UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from .._stack import ImageStack, ImageSlice
from ..source import ImageSource
from ..types import check_image, get_im_dtype_and_nchan, create_im_dtype
from ...imstack import Command, CommandEasy, Opt

__all__ = ['flip',
           'FlipImageStack',]

##### 2D #####
def flip(im, direction='v'):
    """
    Flips an image either vertically (default) or horizontally (by giving an 'h'). The returned
    value is a view - not a copy.
    """
    from numpy import flipud, fliplr
    if direction not in ('v', 'h'): raise ValueError('Unsupported direction')
    return (flipud if direction == 'v' else fliplr)(im)

##### 3D #####
class FlipImageStack(UnchangingFilteredImageStack):
    def __init__(self, ims, dir='y'):
        """dir can be x, y, or z"""
        self._dir = dir
        if dir == 'z':
            slcs = [DoNothingFilterImageSlice(im,self,z) for z,im in enumerate(reversed(ims))]
        elif dir in ('x','y'):
            from numpy import flipud, fliplr
            self._flip = flipud if dir == 'y' else fliplr
            slcs = FlipFilterImageSlice
        else:
            raise ValueError()
        super(FlipImageStack, self).__init__(ims, slcs)
        
class DoNothingFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return self._input.data
    
class FlipFilterImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return self._stack._flip(self._input.data)


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
    def __init__(self, stack, z, im, dtype, channels):
        super(ExtractChannelsImageSlice, self).__init__(stack, z, im)
        self.__channels = channels
        self._set_props(create_im_dtype(dtype, dtype.byteorder, len(channels)), None)
    def _get_props(self):
        self._set_props(None, self._input.shape)
    def _get_data(self):
        im = self._input.data
        if im.ndim == 2: im = im[:,:,None]
        return im[:,:,self.__channels]


class CombineChannelsImageStack(FilteredImageStack):
    def __init__(self, imss):
        from itertools import islice
        from numpy import sum
        
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
    def __init__(self, stack, z, dtype, shape):
        super(CombineChannelsImageSlice, self).__init__(stack, z)
        self._set_props(dtype, shape)
    def _get_props(self): pass
    def _get_data(self):
        dst, chan = empty(self._shape, dtype=self._dtype), 0
        if dst.ndim == 2: dst = dst[:,:,None]
        for ims in self._stack._imss:
            im = ims[z].data
            if im.ndim == 2: im = im[:,:,None]
            dst[:,:,chan:chan+im.shape[2]] = im
            chan += im.shape[2]
        return dst


##### Commands #####
class FlipImageCommand(CommandEasy):
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
    def _consumes(cls, dtype): return ('Image to be flipped',)
    @classmethod
    def _produces(cls, dtype): return ('Flipped image',)
    @classmethod
    def _see_also(cls): return ('z',)
    def __str__(self): return 'flip with dir=%s'%self._dir
    def execute(self, stack): stack.push(FlipImageStack(stack.pop(), self._size))

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
        p.stack_changes(consumes=("Image stacks to combine",), produces=("Image stack with each consumsed image stack as a seperate channel",))
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
        self.__nstacks = args.get_all(*CombineChannelsCommand._opts())[0]
        for _ in xrange(self.__nstacks): stack.pop()
        stack.push()
    def execute(self, stack): stack.push(CombineChannelsImageStack(stack.pop() for _ in xrange(self.__nstacks)))
