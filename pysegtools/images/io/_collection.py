from collections import Iterable
import os

from _stack import FileImageStack, FileImageSlice, FileImageStackHeader, Field, NumericField
from _single import iminfo, imread, imsave
from ..types import get_im_dtype

__all__ = ['FileCollectionStack']

class FileCollectionStack(FileImageStack):
    """
    An image stack that is composed of many 2D image files. It uses iminfo/imread/imsave from images
    so supports any of those file formats.
    """

    @classmethod
    def open(cls, files, readonly=False, pattern=None, start=0, step=1, **options):
        """
        Opens many files (from an iterable) as a single image-stack. You can specify if changes can
        be made or not. The list of files can contain both existing and non-existing files, however
        only the leading existing files are used as "existing", all files after the first listed
        non-existing file are assumed to be non-existent (and may be overwritten).

        Extra options supported are:
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
        if isinstance(files, basestring): files = [files]
        if not isinstance(files, Iterable): raise ValueError('files must be an iterable of filenames')
        files = [os.path.abspath(f) for f in files]
        num_files_found = next((i for i,f in enumerate(files) if not os.path.isfile(f)), len(files))
        if len(options) > 0: raise ValueError('unsupported options provided')
        h = FileCollectionStackHeader(pattern, start, step, files)
        return FileCollectionStack(h, files, files[:num_files_found], readonly)

    @classmethod
    def create(cls, files, ims, pattern=None, start=0, step=1, **options):
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
        if files != None and not isinstance(files, Iterable): raise ValueError('files must be an iterable of filenames')
        files = [] if files == None else [os.path.abspath(f) for f in files]
        if len(options) > 0: raise ValueError('unsupported options provided')
        h = FileCollectionStackHeader(pattern, start, step)
        s = FileCollectionStack(h, files, [], False)
        s._insert(0, ims)
        return s
    
    def __init__(self, h, all_file_names, starting_files, readonly=False):
        super(FileCollectionStack, self).__init__(h, [FileSlice(self, f, z) for z, f in enumerate(starting_files)], readonly)
        self._orig_files = all_file_names

    def __rename(self, slices, filenames):
        """
        Rename all slices given by "shifting" them into the filenames given. The first slice is
        given the first filename in filenames and so forth, when filenames is depleted the next
        slice take the filename from the first slice (which has already been renamed). You can
        think of the used filenames as filenames + [s._filename for s in slices].
        """
        for s in slices:
            src, dst = s._filename, filenames.pop(0)
            os.rename(src, dst)
            filenames.append(src)
            s._filename = dst

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
        filenames.reverse()
        self.__rename(reversed(self._slices[idx:self._d]), filenames)
        filenames.reverse()
        self._insert_slices(idx, [FileSlice(self, f, z+idx) for z, f in enumerate(filenames)])
        for s, im in zip(self._slices[idx:end], ims):
            im = im.data
            imsave(s._filename, im)
            s._cache_data(im)
        
    def _delete(self, idx):
        for start, stop in idx:
            # This could be done in a slightly better way by going from the lowest start to the
            # highest like how file_remove_ranges does it (instead of highest to lowest like is
            # easier). This would only reduce the number of renames while complicating the process.
            filenames = [s._filename for s in self._slices[start:stop]]
            for f in filenames: os.remove(f)
            self.__rename(self._slices[stop:], filenames)
            self._delete_slices(start, stop)

class FileSlice(FileImageSlice):
    def __init__(self, stack, filename, z):
        super(FileSlice, self).__init__(stack, z)
        self._filename = filename
    def _get_props(self):
        shape, dtype = iminfo(self._filename)
        self._set_props(dtype, shape)
    def _get_data(self):
        im = imread(self._filename)
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im
    def _set_data(self, im):
        im = im.data
        imsave(self._filename, im)
        self._set_props(get_im_dtype(im), im.shape[:2])
        return im

def cast_pattern(s):
    s = os.path.abspath(str(s))
    try: _ = s % 0
    except: raise ValueError('pattern must have a single printf-style replacement option similar to %d')
    return s

class FileCollectionStackHeader(FileImageStackHeader):
    """
    Header for a image file collectionstack. The only fields are a tuple of the files being used. If
    there is a pattern for files to load, it along with the starting and step values are also
    available.
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
    def __init__(self, pattern, start, step, files=()):
        self._fields = FileCollectionStackHeader.__fields_raw.copy()
        self._data = {} if pattern == None else {'pattern':pattern,'start':start,'step':step}
        self._data['files'] = tuple(files)
        self._check()
    def save(self, update_pixel_values=True):
        if self._imstack._readonly: raise AttributeError('header is readonly')
    def _update_depth(self, d):
        fs = self._data['files']; l = len(fs)
        self._data['files'] = fs[:d] if d<=l else fs + tuple(s._filename for s in self._imstack._slices[l:])
    def _get_field_name(self, f): return f if f in self._fields else None
