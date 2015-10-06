from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from collections import Iterable, OrderedDict
from itertools import islice
from numbers import Integral
from weakref import proxy

from numpy import ndarray

from ..general import Flags
from .types import is_image, check_image, get_im_dtype, im_dtype_desc
from .source import ImageSource, DeferredPropertiesImageSource

__all__ = ["ImageStack", "HomogeneousImageStack", "ImageSlice", "Homogeneous"]

class Homogeneous(int, Flags):
    None_ = 0
    Shape = 1
    DType = 2
    All   = 3

class ImageStack(object):
    """
    A stack of 2D image slices. This may represent an image on disk or image filter to be/already
    applied to a stack.

    Individual 2D slices are returned with [] or when iterating. Slice image data is only loaded or
    calculated as needed and by default are not cached. The number of slices is available with
    len(). The [] also accepts slice-notation and iterables of indicies and returns a list of
    ImageSlice objects.
    """
    #pylint: disable=protected-access
    __metaclass__ = ABCMeta

    @classmethod
    def as_image_stack(cls, ims):
        """
        Takes an image-stack like object and makes sure it is an ImageStack object. If it is already
        an ImageStack then it is returned. Other types supported are iterables/sequences of image
        sources or 2D ndarrays or a 3D ndarray.
        """
        if isinstance(ims, ImageStack): return ims
        if isinstance(ims, ndarray):
            if is_image(ims): return ImageStackCollection((ims,))      # single slice
            else:             return ImageStackArray(ims)              # multi-slice
        elif isinstance(ims, ImageSource): return ImageStackCollection((ims,)) # single ImageSource
        elif isinstance(ims, Iterable):    return ImageStackCollection(ims)    # iterable of (presumably) ImageSources/ndarrays
        else: raise ValueError()

    def __init__(self, slices):
        self._slices = slices
        self._d = len(slices)
        self._cache_size = 0
        self._cache = None
        self._homogeneous = Homogeneous.All if self._d <= 1 else None

    # General
    @property
    def d(self): return self._d
    def __len__(self): return self._d
    def __str__(self):
        """Gets a basic representation of this class as a string."""
        h,s,d = self._get_homogeneous_info()
        if d is None and self._d == 0: return "(no slices)"
        if h == Homogeneous.All: return "%s: %dx%dx%d %s" % (type(self).__name__, s[1], s[0], self._d, im_dtype_desc(d))
        line = "%0"+str(len(str(self._d-1)))+"d: %dx%d %s"
        return type(self).__name__+": "+", ".join(line%(z,im.w,im.h,im_dtype_desc(im.dtype)) for z,im in enumerate(self._slices))
    @staticmethod
    def _get_print_fill(width):
        from textwrap import TextWrapper
        return (lambda x:x) if width is None else TextWrapper(width=width, subsequent_indent=' '*12).fill
    def _print_general_header(self, width_=None): pass
    def _print_homo_slice_header_gen(self, width_=None): return xrange(self._d)
    def _print_hetero_slice_header_gen(self, width=None):
        fill = ImageStack._get_print_fill(width)
        line = "{z:0>%d}: {w}x{h} {dt} {nb}kb" % len(str(self._d-1))
        for z,im in enumerate(self._slices):
            nb = int(im.w*im.h*im.dtype.itemsize//1024)
            print(fill(line.format(z=z, w=im.w, h=im.h, dt=im_dtype_desc(im.dtype), nb=nb)))
    def print_detailed_info(self, width=None):
        # we use deque to consume a generator completely and quickly
        # (see https://docs.python.org/2/library/itertools.html#recipes)
        from collections import deque
        fill = ImageStack._get_print_fill(width)
        h,s,d = self._get_homogeneous_info()
        print(fill("Handler:    %s" % type(self).__name__))
        if d is None and self._d == 0:
            print(fill("Depth:      0"))
            print(fill("Total Size: 0 kb"))
            self._print_general_header(width)
        elif h == Homogeneous.All:
            print(fill("Dimensions: %d x %d x %d (WxHxD)" % (s[1], s[0], self._d)))
            print(fill("Data Type:  %s" % im_dtype_desc(d)))
            nb = s[1] * s[0] * d.itemsize
            print(fill("Slice Size: %d kb" % (nb//1024)))
            print(fill("Total Size: %d kb" % (nb*self._d//1024)))
            self._print_general_header(width)
            deque(self._print_homo_slice_header_gen(width), maxlen=0)
        else:
            print(fill("Depth:      %d" % self._d))
            print(fill("Total Size: %d kb" % (sum(im.w*im.h*im.dtype.itemsize for im in self._slices)//1024)))
            self._print_general_header(width)
            deque(self._print_hetero_slice_header_gen(width), maxlen=0)

    # Homogeneous interface
    def _get_homogeneous_info(self):
        if self._d == 0: return Homogeneous.All, (None, None), None
        im = self._slices[0]
        shape, dtype = im.shape, im.dtype
        if self._homogeneous is None: self._homogeneous = Homogeneous.None_
        if Homogeneous.Shape not in self._homogeneous:
            if all(shape == im.shape for im in islice(self._slices, 1, None)):
                self._homogeneous |= Homogeneous.Shape
            else: shape = None
        if Homogeneous.DType not in self._homogeneous:
            if all(dtype == im.dtype for im in islice(self._slices, 1, None)):
                self._homogeneous |= Homogeneous.DType
            else: dtype = None
        return self._homogeneous, shape, dtype
    def _update_homogeneous_set(self, z, shape, dtype):
        s = self._slices[-1 if z == 0 else 0]
        if Homogeneous.Shape in self._homogeneous and shape != s.shape:
            self._homogeneous &= ~Homogeneous.Shape
        if Homogeneous.DType in self._homogeneous and dtype != s.dtype:
            self._homogeneous &= ~Homogeneous.DType
    def _has_homogeneous_prop(self, H, attr):
        return self._homogeneous is not None and H in self._homogeneous and hasattr(self, attr)

    @property
    def is_homogeneous(self): return Homogeneous.All == self._get_homogeneous_info()[0]
    @property
    def is_shape_homogeneous(self): return Homogeneous.Shape in self._get_homogeneous_info()[0]
    @property
    def is_dtype_homogeneous(self): return Homogeneous.DType in self._get_homogeneous_info()[0]
    @property
    def w(self): return self.shape[1]
    @property
    def h(self): return self.shape[0]
    @property
    def shape(self):
        if self._has_homogeneous_prop(Homogeneous.Shape, '_shape'): return self._shape #pylint: disable=no-member
        h = self._get_homogeneous_info()
        if Homogeneous.Shape not in h[0]: raise AttributeError('property unavailable on heterogeneous image stacks')
        return h[1]
    @property
    def dtype(self):
        if self._has_homogeneous_prop(Homogeneous.DType, '_dtype'): return self._dtype #pylint: disable=no-member
        h = self._get_homogeneous_info()
        if Homogeneous.DType not in h[0]: raise AttributeError('property unavailable on heterogeneous image stacks')
        return h[2]
    @property
    def stack(self):
        """Get the entire stack as a single 3D image."""
        from numpy import empty
        stack = empty((self._d,) + self.shape, dtype=self.dtype)
        for i, slc in enumerate(self): stack[i,:,:,...] = slc.data
        return stack

    ## Caching of slices ##
    # Note that much of the caching is in ImageSlice or subclasses
    @property
    def cache_size(self): return self._cache_size
    @cache_size.setter
    def cache_size(self, value):
        """
        Set the size of the cache. This number of recently accessed or set slices will be available
        without disk reads or calculations. Default is 0 which means no slices are cached. If -1 then
        all slices will be cached as they are accessed.
        """
        # The cache uses the following member variables:
        #  ._cache_size       either 0 (cache off), -1 (unlimited cache), or a value >0 (max cache size)
        #  ._cache            the LRU cache, an OrderedDict of indices which are cached with popitem(False) as least recently used
        #  ._slices[]._cache  the cached data for a slice (if it exists)
        value = int(value)
        if value < -1: raise ValueError
        if value == 0: # removing cache
            if self._cache_size:
                self._cache = None
                for s in self._slices:
                    s._cache = None
        elif value != 0:
            if not self._cache_size: # creating cache
                self._cache = OrderedDict()
            elif value != -1:
                while len(self._cache) > value: # cache is shrinking
                    self._slices[self._cache.popitem(False)[0]]._cache = None
        self._cache_size = value
    # TODO: def set_cache_size_in_bytes(self, bytes): self.cache_size = bytes // self._sec_bytes;
    def _cache_it(self, i):
        # Places an index into the cache list (but doesn't do anything with the cached data itself)
        # Returns True if the index is already cached (in which case it is moved to the back of the LRU)
        # Otherwise if the queue is full then the oldest thing is removed from the cache
        already_in_cache = self._cache.pop(i, False)
        if not already_in_cache and len(self._cache) == self._cache_size: # cache full
            self._slices[self._cache.popitem(False)]._cache = None
        self._cache[i] = True
        return already_in_cache

    # Getting Slices
    def __getitem__(self, idx):
        """
        Get image slices. Accepts integers, index slices, or iterable indices. When using an integral
        index this returns an ImageSlice object. For index slice and iterable indices it returns a
        list of ImageSlice objects. Images slice data is not loaded until the data attribute of the
        ImageSlice object is used.
        """
        if isinstance(idx, (Integral, slice)): return self._slices[idx]
        elif isinstance(idx, Iterable):        return [self._slices[i] for i in idx]
        else: raise TypeError('index')
    def __iter__(self): return iter(self._slices)

class HomogeneousImageStack(ImageStack):
    """
    An image stack where every slice has the same shape and data type. Provides speed ups for many
    of the homogeneous properties and adds the stack property. It also adds some protected
    properties for convience in deriving classes.
    """
    def __init__(self, w, h, dtype, slices=None, super_init_args=None):
        """
        The constructor is designed to work as a base-class in multiple inheritance with other
        classes being initialized before ImageStack is. If super_init_args is given it must be a
        dictionary and it is expanded and passed to super().__init__(). Otherwise the slices
        argument is passes to it (without expansion). If both are not-None, then slices is added as
        a keyword argument "slices".
        """
        if super_init_args is None:
            if slices is None: raise ValueError()
            super(HomogeneousImageStack, self).__init__(slices)
        else:
            if slices is not None: super_init_args['slices'] = slices
            super(HomogeneousImageStack, self).__init__(**super_init_args)
        self._w = w
        self._h = h
        self._shape = (h, w)
        self._dtype = dtype
        self._slc_pxls  = w * h
        self._slc_bytes = w * h * dtype.itemsize
        self._homogeneous = Homogeneous.All

    def _get_homogeneous_info(self): return Homogeneous.All, self._shape, self._dtype
    def _update_homogeneous_set(self, z, shape, dtype): pass
    @property
    def is_homogeneous(self): return True

    @property
    def w(self): return self._w
    @property
    def h(self): return self._h
    @property
    def shape(self): return self._shape
    @property
    def dtype(self): return self._dtype

class ImageSlice(DeferredPropertiesImageSource):
    """
    A image slice from an image stack. These must be implemented for specific formats and filters.
    The implementor must either call _set_props during initialization or implement a non-trivial
    _get_props function (the trivial one would be def _get_props(self): pass).
    """
    #pylint: disable=protected-access
    def __init__(self, stack, z):
        self._stack = proxy(stack)
        self._z = z
        self._cache = None

    @property
    def stack(self): return self._stack
    @property
    def z(self): return self._z

    @property
    def data(self):
        if not self._stack._cache_size: return self._get_data()
        if not self._stack._cache_it(self._z): self._cache = self._get_data()
        return ImageSource.get_unwriteable_view(self._cache)

    @abstractmethod
    def _get_data(self):
        """
        Internal function for getting image data. Must return an ndarray with shape and dtype of
        this slice (which should be a standardized type). The data returned should either be a copy
        (so that modifications to it do not effect the underlying image data) or an unwritable view.
        """
        pass
    
    @abstractmethod
    def _get_props(self):
        pass
    
# Some generic image stacks that are wrappers are other image datas
class ImageStackArray(HomogeneousImageStack):
    """
    ImageStack that wraps a 3D array of data. Supports setting data slices.
    """
    def __init__(self, arr):
        from numpy import empty
        if arr.ndim not in (3,4) or arr.ndim == 4 and not (0 < arr.shape[-1] <= 5): raise ValueError()
        sh = arr.shape
        im = empty(sh[1:], dtype=arr.dtype) if sh[0]==0 else arr[0,...] # hack to allow 0-depth image stacks
        check_image(im)
        dt = get_im_dtype(im)
        self.__arr = arr
        self.__arr_readonly = ImageSource.get_unwriteable_view(arr)
        super(ImageStackArray, self).__init__(sh[2], sh[1], dt,
            [ImageSliceFromArray(self, z, im, dt) for z,im in enumerate(arr)])
    @ImageStack.cache_size.setter
    def cache_size(self, value): pass # prevent actual caching - all in memory #pylint: disable=arguments-differ
    @property
    def stack(self): return self.__arr_readonly
class ImageSliceFromArray(ImageSlice):
    def __init__(self, stack, z, im, dt):
        super(ImageSliceFromArray, self).__init__(stack, z)
        self._set_props(dt, im.shape[0:2])
        self._im = im
        self._im_readonly = ImageSource.get_unwriteable_view(im)
    def _get_props(self): pass
    def _get_data(self): return self._im_readonly
    @ImageSlice.data.setter
    def data(self, im): #pylint: disable=arguments-differ
        im = ImageSource.as_image_source(im)
        if self._shape != im.shape or self._dtype != im.dtype: raise ValueError('requires all slices to be the same data type and size')
        self._im[:,:,...] = im.data[:,:,...]
        
class ImageStackCollection(ImageStack):
    """ImageStack that wraps a collection of ImageSources."""
    def __init__(self, ims):
        ims = [ImageSource.as_image_source(im) for im in ims]
        super(ImageStackCollection, self).__init__(
            [ImageSliceFromCollection(self, z, im) for z,im in enumerate(ims)])
class ImageSliceFromCollection(ImageSlice):
    def __init__(self, stack, z, im):
        super(ImageSliceFromCollection, self).__init__(stack, z)
        self._im = im
    def _get_props(self): self._set_props(self._im.dtype, self._im.shape[:2])
    def _get_data(self): return self._im.data


# Import commands
from . import _commands #pylint: disable=unused-import
