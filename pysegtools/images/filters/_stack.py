"""Filtered Image Stack class bases."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod

from ..source import ImageSource
from .._stack import ImageStack, ImageSlice

__all__ = ['FilteredImageStack','FilteredImageSlice',
           'UnchangingFilteredImageStack','UnchangingFilteredImageSlice']

class FilteredImageStack(ImageStack):
    def __init__(self, ims, slices, *args, **kwargs):
        """
        Create a new filtered image stack that filters the given image stack (which is stored to
        self._ims) and the `slices`. Instead of providing an explicit list of slices, the `slices`
        can be a type which derives from FilteredImageSlice which takes an image, stack, and z (just
        like the FilteredImageSlice/UnchangingFilteredImageSlice constructor) in which case the list
        of slices is created by calling the constructor for each image/z in the set of images.
        """
        if ims is not None: self._ims = ImageStack.as_image_stack(ims)
        if isinstance(slices, type): slices = [slices(im,self,z,*args,**kwargs) for z,im in enumerate(self._ims)]
        super(FilteredImageStack, self).__init__(slices)
    #def print_detailed_info(self, width=None):
    #    fill = ImageStack._get_print_fill(width)
    #    print(fill("Filter:      " + str(self)))
    #    super(FilteredImageStack, self).print_detailed_info(width)

class FilteredImageSlice(ImageSlice):
    """A slice from a filtered image. This base class simply stores the image source as `_input`."""
    def __init__(self, image, stack, z):
        super(FilteredImageSlice, self).__init__(stack, z)
        self._input = ImageSource.as_image_source(image)
    @abstractmethod
    def _get_props(self): pass
    @abstractmethod
    def _get_data(self): pass

class UnchangingFilteredImageStack(FilteredImageStack):
    """A stack of images that does not change the shape or data type of the filtered image stack."""
    def _get_homogeneous_info(self):
        #pylint: disable=protected-access
        return self._ims._get_homogeneous_info()

class UnchangingFilteredImageSlice(FilteredImageSlice):
    """
    A slice that does not change the shape or data type of the original image slice. Can be used
    without UnchangingFilteredImageStack.
    """
    def _get_props(self): self._set_props(self._input.dtype, self._input.shape)
    @abstractmethod
    def _get_data(self): pass
