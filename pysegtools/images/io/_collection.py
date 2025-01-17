from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Iterable
from itertools import izip
import os

from ._stack import FileImageStack, FileImageSlice, FileImageStackHeader, Field, NumericField
from ._single import FileImageSource
from ..types import get_im_dtype
from ..source import ImageSource
from ...general import ravel

__all__ = ['FileCollectionStack']

class FileCollectionStack(FileImageStack):
    """
    An image stack that is composed of many 2D image files. It uses FileImageSource for each slice.
    """
    
    #pylint: disable=protected-access

    @classmethod
    def open(cls, files, readonly=False, handler=None, pattern=None, start=0, step=1, **options): #pylint: disable=arguments-differ
        """
        Opens many files (from an iterable) as a single image-stack. You can specify if changes can
        be made or not. The list of files can contain both existing and non-existing files, however
        only the leading existing files are used as "existing", all files after the first listed
        non-existing file are assumed to be non-existent (and may be overwritten).

        Extra options supported are:
         * handler: the name of the handler to use for individual slices
         * pattern: a string with a %d (or similar) printf-pattern used when adding extra slices,
           if not provided new slices cannot be added. The pattern must include a file extension
           which is used to determine the file-format to save as.
         * start: a non-negative integer (or convertible) for the first index of the image stack to
           feed into pattern
         * step: a positive integer (or convertible) for the step index of the image stack to feed
           into pattern

        Note that "pattern" is not automatically used to find files to open, instead it is only used
        when appending new slices. If you wish to have all the files from a pattern, you can do:
          ims = FileCollectionStack.open((pattern%(i*step+start) for i in xrange(existing_count)), pattern=pattern, start=start, step=step)
        """
        if isinstance(files, str): files = [files]
        elif not isinstance(files, Iterable): raise ValueError('files must be an iterable of filenames')
        files = [os.path.abspath(f) for f in files]
        num_files_found = next((i for i,f in enumerate(files) if not os.path.isfile(f)), len(files))
        if readonly and num_files_found != len(files): raise ValueError('opening file collection as readonly requires all files to already exist')
        h = FileCollectionStackHeader(handler, pattern, start, step, files, **options)
        return FileCollectionStack(h, True, not readonly, files, files[:num_files_found], handler, **options)

    @classmethod
    def openable(cls, files, readonly=False, handler=None, pattern=None, start=0, step=1, **options):#pylint: disable=arguments-differ
        """
        Checks if a set of files can be opened with the given options.
        """
        try:
            if pattern is not None: cast_pattern(pattern); int(start); int(step)
            if isinstance(files, str): files = [files]
            elif not isinstance(files, Iterable): return False
            files = [os.path.abspath(f) for f in files]
            num_files_found = next((i for i,f in enumerate(files) if not os.path.isfile(f)), len(files))
            return (not readonly or num_files_found == len(files)) and \
                   all(FileImageSource.openable(f, readonly, handler, **options) for f in files[:num_files_found]) and \
                   all(FileImageSource.creatable(f, False, handler, **options) for f in files[num_files_found:])
        except StandardError: pass
        return False

    @classmethod
    def create(cls, files, ims, writeonly=False, handler=None, pattern=None, start=0, step=1, **options):#pylint: disable=arguments-differ
        """
        Creates an image-stack saving to multiple files. The files are an iterable of filenames to
        save slices as, or if None then only the pattern is used. The files are overwritten and
        never read. Extra options supported are:
         * handler: the name of the handler to use for individual slices
         * pattern: a string with a %d (or similar) printf-pattern used when adding slices beyond
           the end of the files list. If not provided new slices cannot be added. The pattern must
           include a file extension which is used to determine the file-format to save as.
         * start: a non-negative integer (or convertible) for the first index of the image stack to
           feed into pattern
         * step: a positive integer (or convertible) for the step index of the image stack to feed
           into pattern
        """
        if isinstance(files, str): files = [files]
        elif files is None: files = []
        elif not isinstance(files, Iterable): raise ValueError('files must be an iterable of filenames')
        files = [os.path.abspath(f) for f in files]
        h = FileCollectionStackHeader(handler, pattern, start, step, [], **options)
        s = FileCollectionStack(h, not writeonly, True, files)
        s._insert(0, ims)
        return s

    @classmethod
    def creatable(cls, files, writeonly=False, handler=None, pattern=None, start=0, step=1, **options):#pylint: disable=arguments-differ
        """
        Checks if a set of files can be opened with the given options.
        """
        try:
            if pattern is not None: cast_pattern(pattern); int(start); int(step)
            if isinstance(files, str): files = [files]
            elif files is None: files = []
            elif not isinstance(files, Iterable): return False
            files = [os.path.abspath(f) for f in files]
            return all(FileImageSource.creatable(f, writeonly, handler, **options) for f in files)
        except StandardError: pass
        return False

    def __init__(self, h, read, write, filenames, starting_files=(), handler=None, **options):
        readonly = read and not write
        slices = [FileSlice(self, FileImageSource.open(f,readonly,handler,**options), z) 
                  for z,f in enumerate(starting_files)]
        super(FileCollectionStack, self).__init__(h, slices, readonly)
        self._writeonly = write and not read
        self._handler = handler
        self._orig_files = filenames

    def _print_homo_slice_header_gen(self, width=None):
        from .._stack import ImageStack
        fill = ImageStack._get_print_fill(width)
        z_width = len(str(self._d-1))
        line = "  {z:0>%d}: {handler}: {filename}" % z_width
        skip = self._shared_header().viewkeys()
        for z,im in enumerate(self._slices):
            print(fill(line.format(z=z, filename=im._source.filename, handler=type(im._source).__name__)))
            yield FileImageStack._print_header(im.header, width, None, z_width+4, skip=skip)
    def _print_hetero_slice_header_gen(self, width=None):
        from .._stack import ImageStack
        from ..types import im_dtype_desc
        fill = ImageStack._get_print_fill(width)
        z_width = len(str(self._d-1))
        line = "{z:0>%d}: {handler}: {w}x{h} {dt} {nb}kb {filename}" % z_width
        skip = self._shared_header().viewkeys()
        for z,im in enumerate(self._slices):
            nb = im.w*im.h*im.dtype.itemsize//1024
            print(fill(line.format(z=z, w=im.w, h=im.h, dt=im_dtype_desc(im.dtype), nb=nb,
                                   filename=im._source.filename, handler=type(im._source).__name__)))
            yield FileImageStack._print_header(im.header, width, None, z_width+2, skip=skip)

    @property
    def filenames(self): return tuple(ravel(s._source.filenames for s in self._slices))

    @staticmethod
    def __rename(slices, filenames):
        """
        Rename all slices given by "shifting" them into the filenames given. The first slice is
        given the first filename in filenames and so forth, when filenames is depleted the next
        slice take the filename from the first slice (which has already been renamed). You can
        think of the used filenames as filenames + [s._source.filename for s in slices].
        """
        for s in slices:
            fn = s._source.filename
            s._source.filename = filenames.pop(0)
            filenames.append(fn)

    def _insert(self, idx, ims):
        # Note: the renaming may run into problems on Windows because if the file exists the rename
        # will fail. This only affects insert and only the first len(ims) renames.
        if self._d == 0 and len(ims) > 1:
            # due to the way _update_homogeneous_set is implemented, we need to insert the first
            # image by itself first when we are inserting into an emtpy collection
            self._insert(0, ims[:1])
            idx = 1
            ims = ims[1:]
        end = self._d+len(ims)
        filenames = self._orig_files[self._d:end]
        if len(filenames) < len(ims):
            if 'pattern' not in self._header: raise ValueError('ran out of usable filenames without a pattern option')
            start, step = self._header.start, self._header.step
            start, stop = start+(idx+len(filenames))*step, start+end*step
            filenames.extend(self._header.pattern % i for i in xrange(start, stop, step))
        FileCollectionStack.__rename(reversed(self._slices[idx:self._d]), reversed(filenames))
        self._insert_slices(idx, [FileSlice(self, DummyFileImageSource(f), z)
                                  for z,f in izip(xrange(idx,idx+len(ims)),filenames)])
        opts = self._header.get('options', {})
        for im,f,slc in izip(ims, filenames, self._slices[idx:idx+len(ims)]):
            im = im.data # get the image data so we can use it multiple times without reloading
            src = ImageSource.as_image_source(im) # but we actually need it to be an image source
            slc._source = FileImageSource.create(f, src, self._writeonly, self._handler, **opts)
            slc._cache_data(im)

    def _delete(self, idxs):
        for start, stop in idxs:
            # This could be done in a slightly better way by going from the lowest start to the
            # highest like how file_remove_ranges does it (instead of highest to lowest like is
            # easier). This would only reduce the number of renames while complicating the process.
            filenames = [s._filename for s in self._slices[start:stop]]
            for f in filenames: os.remove(f)
            FileCollectionStack.__rename(self._slices[stop:], filenames)
            self._delete_slices(start, stop)

class FileSlice(FileImageSlice):
    def __init__(self, stack, source, z):
        super(FileSlice, self).__init__(stack, z)
        self._source = source
    @property
    def header(self): return self._source.header
    def _get_props(self): self._set_props(self._source.dtype, self._source.shape)
    def _get_data(self):
        im = self._source.data
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im
    def _set_data(self, im):
        im = im.data
        self._source.data = im
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im

class DummyFileImageSource(FileImageSource):
    # This allows us to have an object with the right filename but doesn't do anything
    # These abstract methods needs to be implemented but should never be called
    def _get_props(self): raise RuntimeError()
    def _set_filename(self, filename): raise RuntimeError()
    def _get_data(self): raise RuntimeError()
    def _set_data(self, im): raise RuntimeError()

def cast_pattern(s):
    s = os.path.abspath(str(s))
    try: _ = s % 0
    except: raise ValueError('pattern must have a single printf-style replacement option similar to %d')
    return s

class FileCollectionStackHeader(FileImageStackHeader):
    #pylint: disable=protected-access
    """
    Header for a image file collection stack. The only fields are a tuple of the files being used.
    If there is a pattern for files to load, it along with the starting and step values are also
    available.
    """
    __fields_raw = {
        'handler': Field(Field.cast_check(FileImageSource.is_handler), True, True), #pylint: disable=no-member
        'options': Field(dict, True, True),
        'files':   Field(tuple, True, False),
        'pattern': Field(cast_pattern, True, True),
        'start':   NumericField(int, 0, None, True, True, 0),
        'step':    NumericField(int, 1, None, True, True, 1),
        }

    # Setup all instance variables to make sure they are in __dict__
    _fields = None
    def __init__(self, handler, pattern, start, step, files, **options):
        self._fields = FileCollectionStackHeader.__fields_raw.copy()
        data = {} if pattern is None else {'pattern':pattern,'start':start,'step':step}
        if handler is not None: data['handler'] = handler
        if len(options): data['options'] = options
        data['files'] = tuple(files)
        super(FileCollectionStackHeader, self).__init__(data)
    def save(self):
        if self._imstack._readonly: raise AttributeError('header is readonly')
    def _update_depth(self, d):
        fs = self._data['files']
        if d <= len(fs): fs = fs[:d]
        else: fs = fs + tuple(s._source.filename for s in self._imstack._slices[len(fs):])
        self._data['files'] = fs
    def _get_field_name(self, f): return f if f in self._fields else None
