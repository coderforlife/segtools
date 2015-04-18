"""Basic Image Filters"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ._stack import UnchangingFilteredImageStack, UnchangingFilteredImageSlice
from ..types import check_image
from ...imstack import CommandEasy, Opt

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
