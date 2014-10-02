from collections import Iterable
import os

from _stack import ImageStack, Header, Field, NumericField
from _single import iminfo, imread, imsave

__all__ = ['ImageStackCollection']

class ImageStackCollection(ImageStack):
    """
    An image stack that is composed of many other files. It uses iminfo/imread/imsave from images so
    supports any of those file formats.
    """

    @classmethod
    def open(cls, files, readonly=False, pattern=None, start=0, step=1, **options):
        """
        Opens many files (from an iterable) as a single image-stack. You can specify if changes can
        be made or not. Extra options supported are:
         * pattern: a string with a %d (or similar) printf-pattern used when adding extra slices,
           if not provided new slices cannot be added. The pattern must include a file extension
           which is used to determine the file-format to save as.
         * start: a non-negative integer (or convertible) for the first index of the image stack to
           feed into pattern
         * step: a positive integer (or convertible) for the step index of the image stack to feed
           into pattern

        Note that "pattern" is not automatically used to find files to open, instead it is only used
        when appending new slices. If you wish to have all the files from a pattern, you can do:
          ims = ImageStackCollection.open((pattern%i for i in xrange(existing_count)), pattern=pattern)
        """
        if isinstance(files, basestring): files = [files]
        if not isinstance(files, Iterable): raise ValueError('files must be an iterable with at least one filename')
        files = [os.path.abspath(f) for f in files]
        if len(files) == 0 or any(not os.path.isfile(f) for f in files): raise ValueError('files must be an iterable with at least one entry of existing filenames')
        if len(options) > 0: raise ValueError('unsupported options provided')
        h = ImageStackCollectionHeader(pattern, start, step, files)
        return ImageStackCollection(h, files, files, readonly)

    @classmethod
    def create(cls, files, shape, dtype, pattern=None, start=0, step=1, **options):
        """
        Creates in image-stack saving to multiple files. The files are an iterable of filenames to
        save slices as, or if None then only the pattern is used. The files are overwritten and
        never read. Extra options supported are:
         * pattern: a string with a %d (or similar) printf-pattern used when adding slices beyond
           the end of the files list. If not provided new slices cannot be added. The pattern must
           include a file extension which is used to determine the file-format to save as.
         * start: a non-negative integer (or convertible) for the first index of the image stack to
           feed into pattern
         * step: a positive integer (or convertible) for the step index of the image stack to feed
           into pattern
        """
        if isinstance(files, basestring): files = [files]
        if files != None and not isinstance(files, Iterable): raise ValueError('files must be an iterable with at least one filename')
        files = [] if files == None else [os.path.abspath(f) for f in files]
        if len(options) > 0: raise ValueError('unsupported options provided')
        h = ImageStackCollectionHeader(pattern, start, step, shape=shape, dtype=dtype)
        return ImageStackCollection(h, files, [], False)
    
    def __init__(self, h, all_file_names, starting_files, readonly=False):
        super(ImageStackCollection, self).__init__(h._shape[1], h._shape[0], len(starting_files), h._dtype, h, readonly)
        h._imstack = self
        self._orig_files = all_file_names
        self._files = starting_files

    def _get_section(self, i, seq): return imread(self._files[i])
    def _set_section(self, i, im, seq):
        if i == self._d:
            if i < len(self._orig_files): f = self._orig_files[i]
            elif 'pattern' in self._h:    f = self._h.pattern % (self._h.start + i * self._h.step)
            else: raise ValueError('ran out of usable filenames without a pattern option')
            self._files.append(f)
        imsave(self._files[i], im)
    def _del_sections(self, start, stop):
        move = stop - start
        for f in self._files[start:stop]: os.remove(f)
        for i,f in enumerate(self._files[stop:]): os.rename(f, self._files[i-move])
        del self._files[start:stop]

def cast_pattern(s):
    s = os.path.abspath(str(s))
    try: _ = s % 0
    except: raise ValueError('pattern must have a single printf-style replacement option similar to %d')
    return s

class ImageStackCollectionHeader(Header):
    """
    Header for a collection of images as a stack. The only fields are a tuple of the files that back
    the collection. If there is a pattern for files to load, it along with the starting and step
    values are also available.
    """
    __fields_raw = {
        'files':  Field(tuple, True, False),
        'pattern':Field(cast_pattern, True, True),
        'start':  NumericField(int, 0, None, True, True, 0),
        'step':   NumericField(int, 1, None, True, True, 1),
        }
    
    # Setup all instance variables to make sure they are in __dict__

    # Required for headers
    _imstack = None
    _fields = None
    _data = None
    
    # Specific to collections
    _shape = None
    _dtype = None
    def __init__(self, pattern, start, step, files=(), shape=None, dtype=None):
        self._fields = ImageStackCollectionHeader.__fields_raw.copy()
        self._data = {} if pattern == None else {'pattern':pattern,'start':start,'step':step}
        if shape == None and dtype == None:
            shape, dtype = iminfo(files[0])
            if any(iminfo(f) != (shape, dtype) for f in files[1:]): raise ValueError('all files must have same dimensions and data-type')
        self._data['files'] = tuple(files)
        self._check()
        self._shape = shape
        self._dtype = dtype
    def save(self, update_pixel_values=True):
        if self._imstack._readonly: raise AttributeError('header is readonly')
    def _update_depth(self, d): self._data['files'] = tuple(self._imstack._files)
    def _get_field_name(self, f): return f if f in self._fields else None
