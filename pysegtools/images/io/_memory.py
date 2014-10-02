from numpy import ndarray, empty, newaxis, delete
from collections import Iterable, Sequence

from ..types import im_standardize_dtype, imstack_standardize_dtype
from _stack import ImageStack, Header

__all__ = ['MemoryImageStack']

class MemoryImageStack(ImageStack):
    """
    An image stack that is completely in-memory and not file-based. It does not use the standard
    open/create interface of other ImageStacks but instead uses a standard constructor.
    """
    def __init__(self, im, readonly=False):
        """
        im should be a single image as an array, a complete image stack as a single array, or a
        sequence/iterable of single images.
        """
        self._im = None
        if isinstance(im, ndarray):
            try:                       self._im = imstack_standardize_dtype(im).copy()
            except ValueError:         self._im = im_standardize_dtype(im)[newaxis,...].copy()
        elif isinstance(im, Sequence):
            self._im = empty((len(im),) + im_standardize_dtype(im[0]).shape)
            for i,im in enumerate(im): self._im[i,...] = im_standardize_dtype(im)
        elif isinstance(im, Iterable): self._im = vstack(im_standardize_dtype(im)[newaxis,...] for im in itr)
        if self._im == None: raise ValueError

        super(MemoryImageStack, self).__init__(self._im.shape[2], self._im.shape[1], self._im.shape[0],
                                               self._im.dtype, MemoryImageStackHeader(self), readonly)

    @ImageStack.cache_size.setter
    def cache_size(self, value): pass # prevent actual caching as this is already all in-memory
    def _get_section(self, i, seq):       return self._im[i,...].copy() # TODO: make copy-on-write (to either)
    def _set_section(self, i, im, seq):   self._im[i,...] = im
    def _del_sections(self, start, stop): self._im = delete(self._im, slice(start,stop), axis=0)
    @property
    def stack(self): return self._im.copy() # TODO: make copy-on-write (to either)

class MemoryImageStackHeader(Header):
    """
    In-memory image stack header supports any field names.
    """
    _imstack = None
    _fields = None
    _data = None
    def __init__(self, ims): self._imstack = ims; self._fields = {}; self._data = {};
    def _get_field_name(self, f): return f if isinstance(f, basestring) else None
    def _update_depth(self, d): pass
    def save(self, update_pixel_values=True):
        if self._imstack._readonly: raise AttributeError('header is readonly')
