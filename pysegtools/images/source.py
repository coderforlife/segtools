from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta, abstractproperty, abstractmethod
from collections import Sequence
from numpy import ndarray
from .types import check_image, get_im_dtype

__all__ = ['ImageSource', 'ArrayImageSource', 'DeferredPropertiesImageSource']

class ImageSource(object):
    """
    A class that represents a source of image data. This is used when we want to not load or compute
    the actual image data until it is actually needed. The dtype should be that which is returned by
    create_im_dtype/get_im_dtype - a dtype that has the base type and possibly a number of channels.
    The shape should be (height,width) only. In general accessing data should not cache the results
    unless otherwise specified.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def w(self): pass
    @abstractproperty
    def h(self): pass
    @abstractproperty
    def dtype(self): pass
    @abstractproperty
    def shape(self): pass
    @abstractproperty
    def data(self): pass

    @staticmethod
    def as_image_source(im):
        """
        Takes an image and returns an ImageSource. Nothing is done to ImageSources ndarray are
        wrapped in an ArrayImageSource. Other things cause a TypeError to be raised.
        """
        if isinstance(im, ImageSource): return im
        if isinstance(im, ndarray): return ArrayImageSource(im)
        raise TypeError('image must be an ndarray or an ImageSource')

class ArrayImageSource(ImageSource):
    """A simple image source that is backed by an ndarray."""
    def __init__(self, im):
        check_image(im)
        self._im = im.view()
        self._im.flags.writeable = False
    @property
    def w(self): return self._im.shape[1]
    @property
    def h(self): return self._im.shape[0]
    @property
    def dtype(self): return get_im_dtype(self._im)
    @property
    def shape(self): return self._im.shape[:2]
    @property
    def data(self): return self._im

class DeferredPropertiesImageSource(ImageSource):
    """An image source where the shape and dtype properties are deferred but cached."""
    __metaclass__ = ABCMeta
    _w = None
    _h = None
    _shape = None
    _dtype = None

    @abstractmethod
    def _get_props(self):
        """Get the properies for this image source. Should call _set_props(...)."""
        pass

    def _set_props(self, dtype, shape):
        """Set the dtype and width/height/shape properties."""
        if shape is not None:
            self._h, self._w = shape
            self._shape = shape
        if dtype is not None:
            self._dtype = dtype

    @property
    def w(self):
        if self._w is None: self._get_props()
        return self._w
    @property
    def h(self):
        if self._h is None: self._get_props()
        return self._h
    @property
    def dtype(self):
        if self._dtype is None: self._get_props()
        return self._dtype
    @property
    def shape(self):
        if self._shape is None: self._get_props()
        return self._shape
