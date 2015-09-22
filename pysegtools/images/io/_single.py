"""Base classes for 2D image file formats, or 'single slices'."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod
import os

from ._handler_manager import HandlerManager
from ..source import ImageSource, DeferredPropertiesImageSource

class FileImageSource(DeferredPropertiesImageSource, HandlerManager):
    """
    A single 2D image slices on disk. Initially only the header is loaded. The image data is not
    read until accessed. This class is designed to mirror the FileImageStack class except it
    derives from ImageSource instead of ImageStack.
    """
    def __init__(self, filename, readonly=False):
        self._filename = os.path.abspath(filename)
        self._readonly = bool(readonly)

    @classmethod
    def _create_trans(cls, im): return ImageSource.as_image_source(im)
    
    def close(self): pass
    def __delete__(self): self.close()
    @property
    def readonly(self): return self._readonly
    @property
    def header(self): #pylint: disable=no-self-use
        """
        Return 'header' information for an image. This should be a copy of a dictionary or None.
        """
        return None

    @abstractmethod
    def _get_props(self):
        """From DeferredPropertiesImageSource, should call self._set_props(dtype, shape)"""
        pass

    @property
    def data(self):
        return ImageSource.get_unwriteable_view(self._get_data())

    @property
    def filename(self):
        """Get or set the primary file name."""
        return self._filename
    @filename.setter
    def filename(self, filename):
        filename = os.path.abspath(filename)
        if self._filename != filename:
            self._set_filename(filename)
            self._filename = filename

    @abstractmethod
    def _set_filename(self, filename):
        """
        Internal function for setting the filename - basically means the physical storage of this
        image needs to change. Anything that needs to happen in response to the file moving needs
        to be done now, including renaming the file itself, which means this function is minimally
        `self._rename(filename)`.
        """
        pass

    def _rename(self, filename):
        # TODO: renaming may run into problems on Windows because if the file exists the rename
        # will fail. A solution is to delete the file first. Also, what if the new file name
        # exists and is a directory or unremovable?
        #os.remove(filename)
        os.rename(self._filename, filename)

    @property
    def filenames(self):
        """
        Get all file names associated with this file (for example the primary file might only
        contain header information and the data is saved in another file). By default this just
        returns a tuple containing the primary filename.
        """
        return (self._filename,)

    @data.setter
    def data(self, im): #pylint: disable=arguments-differ
        if self._readonly: raise ValueError('cannot set data for readonly image source')
        self._set_data(ImageSource.as_image_source(im))

    @abstractmethod
    def _get_data(self):
        """
        Internal function for getting image data. Must return an ndarray with shape and dtype of
        this slice (which should be a standardized type). The data returned should either be a copy
        (so that modifications to it do not effect the underlying image data) or an unwritable view.
        """
        pass
    
    @abstractmethod
    def _set_data(self, im):
        """
        Internal function for setting image data. The image is an ImageSource object. If the image
        data is not acceptable (for example, the shape or dtype is not acceptable for this handler)
        then an exception should be thrown. In the case of an exception, it must be thrown before
        any changes are made to this FileImageSource properties or the data on disk.

        This method can optionally copy any metadata it is aware of from the image to this slice.
        """
        pass

# Import formats
#from . import handlers (doesn't work, next line is roughly equivalent) 
__import__(('.'.join(__name__.split('.')[:-1]+['handlers'])), globals(), locals()) #pylint: disable=unused-import
