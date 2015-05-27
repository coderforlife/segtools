"""Base classes for 2D image file formats, or 'single slices'."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta, abstractproperty, abstractmethod
from io import open

from ..source import ImageSource, DeferredPropertiesImageSource
from ...imstack import Help
from ...general.utils import all_subclasses

class _FileImageSourceMeta(ABCMeta):
    """The meta-class for file image stacks, which extends ABCMeta and calls Help.register if applicable"""
    def __new__(cls, clsname, bases, dct):
        c = super(_FileImageSourceMeta, cls).__new__(cls, clsname, bases, dct)
        n = c.name()
        if n is not None: Help.register((n,c.__name__), c.print_help)
        return c

class FileImageSource(DeferredPropertiesImageSource):
    """
    A single 2D image slices on disk. Initially only the header is loaded. The image data is not
    read until accessed. This class is designed to mirror the FileImageStack class except it
    derives from ImageSource instead of ImageStack.
    """
    
    __metaclass__ = _FileImageSourceMeta

    @classmethod
    def is_handler(cls, handler):
        """Checks that the given string is a valid handler for image source files."""
        return any(handler == cls.name() for cls in all_subclasses(cls))

    @classmethod
    def handlers(cls, read=True):
        """Get a list of all image source handlers."""
        handlers = []
        for cls in all_subclasses(cls):
            h = cls.name()
            if h is not None and (read and cls._can_read or not read and cls._can_write):
                handlers.append(h)
        return handlers

    @classmethod
    def __open_handlers(cls, readonly=False, handler=None):
        for cls in all_subclasses(cls):
            if cls._can_read() and (readonly or cls._can_write()) and \
               (handler is None or handler == cls.name()):
                yield cls

    @classmethod
    def open(cls, filename, readonly=False, handler=None, **options):
        """
        Opens an existing image file. Extra options are only supported by some file handlers.
        """
        for cls in cls.__open_handlers(readonly, handler):
            with open(filename, 'rb') as f:
                if cls._openable(filename, f, readonly, **options):
                    return cls.open(filename, readonly, **options)
        raise ValueError('Unable to find image source handler for file "'+filename+'"')

    @classmethod
    def openable(cls, filename, readonly=False, handler=None, **options):
        """
        Checks if an existing image file can be opened with the given arguments. Extra options are
        only supported by some file handlers.
        """
        try:
            for cls in cls.__open_handlers(readonly, handler):
                with open(filename, 'rb') as f:
                    if cls._openable(filename, f, readonly, **options): return True
        except StandardError: pass
        return False

    @classmethod
    def __create_handlers(cls, writeonly=False, handler=None):
        for cls in all_subclasses(cls):
            if cls._can_write() and (writeonly or cls._can_read()) and \
               (handler is None or handler == cls.name()):
                yield cls

    @classmethod
    def create(cls, filename, im, writeonly=False, handler=None, **options):
        """
        Creates an image file. Extra options are only supported by some file handlers.

        The new image file is created from the given ndarray or ImageSource. Selection of a handler
        and format is purely on file extension and options given.

        Note that the "writeonly" flag is only used for optimization and may not always been
        honored. It is your word that you will not use any functions that get data from the
        stack.
        """
        from os.path import splitext
        im = ImageSource.as_image_source(im)
        ext = splitext(filename)[1].lower()
        for cls in cls.__create_handlers(writeonly, handler):
            if cls._creatable(filename, ext, writeonly, **options):
                return cls.create(filename, im, writeonly, **options)
        raise ValueError('Unknown file extension or options')

    @classmethod
    def creatable(cls, filename, writeonly=False, handler=None, **options):
        """
        Checks if a filename can written to as a new image file. Extra options are only supported by
        some file handlers.
        """
        try:
            from os.path import splitext
            ext = splitext(filename)[1].lower()
            return any(cls._creatable(filename, ext, writeonly, **options)
                       for cls in cls.__create_handlers(writeonly, handler))
        except StandardError: pass
        return False

    @classmethod
    def _openable(cls, filename, f, readonly, **opts): #pylint: disable=unused-argument
        """
        [To be implemented by handler, default is nothing is openable]

        Return if a file is openable as a FileImageSource given the filename, file object, and
        dictionary of options. If this returns True then the class must provide a static/class
        method like:
            `open(filename_or_file, readonly, **options)`
        Option keys are always strings, values can be either strings or other values (but strings
        must be accepted for any value and you must convert, if possible). While _openable should
        return False if there any unknown option keys or option values cannot be used, open should
        throw exceptions.
        """
        return False

    @classmethod
    def _creatable(cls, filename, ext, writeonly, **opts): #pylint: disable=unused-argument
        """
        [To be implemented by handler, default is nothing is creatable]

        Return if a filename/ext (ext always lowercase and includes .) is creatable as a
        FileImageSource given the dictionary of options. If this returns True then the class must
        provide a static/class method like:
            `create(filename, ImageSource, writeonly, **options)`
        Option keys are always strings, values can be either strings or other values (but strings
        must be accepted for any value and you must convert, if possible). While _creatable should
        return False if there any unknown option keys or option values cannot be used, create should
        throw exceptions.

        Note that the "writeonly" flag is only used for optimization and may not always been
        honored. It is the word of the caller they will not use any functions that get data from
        the stack. The handler may ignore this and treat it as read/write.
        """
        return False

    @classmethod
    def _can_read(cls):
        """
        [To be implemented by handler, default is readable]

        Returns True if this handler can, under any circumstances, read images.
        """
        return True

    @classmethod
    def _can_write(cls):
        """
        [To be implemented by handler, default is writable]

        Returns True if this handler can, under any circumstances, write images.
        """
        return True

    @classmethod
    def name(cls):
        """
        [To be implemented by handler, default causes the handler to not have a help page, be
        unusable by name, and not be listed, but still can handle things]

        Return the name of this image source handler to be displayed in help outputs.
        """
        return None

    @classmethod
    def print_help(cls, width):
        """
        [To be implemented by handler, default prints nothing]

        Prints the help page of this image source handler.
        """
        pass

    # General
    def __init__(self, readonly=False):
        self._readonly = bool(readonly)
    def close(self): pass
    def __delete__(self): self.close()
    @property
    def readonly(self): return self._readonly

    @abstractmethod
    def _get_props(self):
        """From DeferredPropertiesImageSource, should call self._set_props(dtype, shape)"""
        pass

    @property
    def data(self):
        return ImageSource.get_unwriteable_view(self._get_data())

    @data.setter
    def set_data(self, im):
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

# Import additional formats
from . import handlers # pylint: disable=unused-import
