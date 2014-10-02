from abc import ABCMeta, abstractmethod
from collections import Iterable
from numbers import Integral
from numpy import ndarray

from ...general.datawrapper import DictionaryWrapperWithAttr
from ...general.enum import Enum
from ..types import im_standardize_dtype

__all__ = ['ImageStack','Header','Field','FixedField','NumericField','MatchQuality']

# TODO: improve sequential get/set ability to work even when not explicit

class MatchQuality(int, Enum):
    NotAtAll = 0
    Unlikely = 25
    Maybe = 50
    Likely = 75
    Definitely = 100

def read_slice(idx, d):
    # Fix up the start, stop, and step values (supporting negative and/or missing values)
    # Note that idx.stop can be either None or 2147483647 indicating "all the way to the end" and
    # if a step is not given, negative values are pre-converted to positive values using the __len__
    start, stop, step = idx.start, idx.stop, idx.step
    if step == None: step = 1
    elif step == 0: raise ValueError('slice step cannot be zero')
    if step > 0:
        start = 0 if start == None else max(start+d if start < 0 else start, 0)
        stop  = d if stop  == None else min(stop +d if stop  < 0 else stop , d)
        if start >= stop: return 0, 1, 1 # empty
    else:
        start = d-1 if start == None else min(start+d if start < 0 else start, d-1)
        stop  =  -1 if stop  == None else max(stop +d if stop  < 0 else stop ,  -1)
        if start <= stop: return 0, 1, 1 # empty
    return start, stop, step

def is_sequential(l):
    if len(l) > 0:
        y = l[0] - 1
        for x in l:
            if x != y + 1: return False
            y = x
    return True

class ImageStack(object):
    """
    A stack of images. This is either backed by a file format that already has multiple images (like
    MRC, MHA/MHD, and TIFF) or a collection of seperate images.

    When loading an image stack only the header(s) is/are loaded. Individual 2D slices are returned
    with the [] or when iterating. Slices are only loaded as needed and by default are not cached.
    The number of slices is available with len(). The [] also accepts slice-notation and iterables
    of indicies and always returns a view in these cases (so as to not force everything to load into
    memory usage). To force everything to load you can use the stack() function which returns the
    entire stack as a 3D array or one can use list() on the return value of [] to get a list of 2D
    slices.

    When writing, slices are typically saved immediately but the header typically is not. Call
    the save() function to force all data including the header to be saved.
    """
    
    __metaclass__ = ABCMeta
    
    @classmethod
    def open(cls, filename, readonly=False, **options):
        """
        Opens an existing image-stack file or series of images as a stack. If 'filename' is a
        string then it is treated as an existing file. Otherwise it needs to be an iterable of
        file names. Extra options are only supported by some file formats. 
        """
        if isinstance(filename, basestring):
            highest_cls, highest_mq = None, MatchQuality.NotAtAll
            for cls in ImageStack.__subclasses__():
                with open(filename, 'r') as f: mq = cls._openable(f, **options)
                if mq > highest_mq: highest_cls, highest_mq = cls, mq
                if mq == MatchQuality.Definitely: break
            if highest_mq == MatchQuality.NotAtAll: raise ValueError
            return highest_cls.open(filename, readonly, **options)
        elif isinstance(filename, Iterable):
            from _collection import ImageStackCollection
            return ImageStackCollection.open(filename, readonly, **options)
        else:
            raise ValueError
        
    @classmethod
    def create(cls, filename, shape, dtype, **options):
        """
        Creates an empty image-stack file or writes to a series of images as a stack. If 'filename'
        is a string then it is treated as a new file. Otherwise it needs to be an iterable of file
        names (even empty) or None in which case a collection of files are used to write to. Extra
        options are only supported by some file formats. When filenames is None or an empty iterable
        then you need to give a "pattern" option with an extension and %d.
        """
        if isinstance(filename, basestring):
            from os.path import splitext
            ext = splitext(filename)[1].lower().lstrip('.')
            highest_cls, highest_mq = None, MatchQuality.NotAtAll
            for cls in ImageStack.__subclasses__():
                mq = cls._creatable(filename, ext, **options)
                if mq > highest_mq: highest_cls, highest_mq = cls, mq
                if mq == MatchQuality.Definitely: break
            if highest_mq == MatchQuality.NotAtAll: raise ValueError
            return highest_cls.create(filename, shape, dtype, **options)
        elif filename == None or isinstance(filename, Iterable):
            from _collection import ImageStackCollection
            return ImageStackCollection.create(filename, shape, dtype, **options)
        else:
            raise ValueError

    @classmethod
    def supported_list(cls):
        descs = []
        for cls in ImageStack.__subclasses__():
            desc = cls._description()
            if desc is not None: descs.append(desc)
        return descs

    @classmethod
    def _openable(cls, f, **opts):
        """
        Return how likely a readable file-like object is openable as an ImageStack given the
        dictionary of options. Returns a MatchQuality rating.
        """
        return MatchQuality.NotAtAll

    @classmethod
    def _creatable(cls, filename, ext, **opts):
        """
        Return how likely a filename/ext (without .) is creatable as an ImageStack given the
        dictionary of options. Returns a MatchQuality rating.
        """
        return MatchQuality.NotAtAll
    
    @classmethod
    def _description(cls):
        """
        Return the description of this image stack handler to be displayed in help outputs.
        """
        return None
    
    def __init__(self, w, h, d, dtype, header, readonly=False):
        self._w = w
        self._h = h
        self._d = d
        self._dtype = dtype
        self._shape = (h, w)
        self._header = header
        self._readonly = bool(readonly)
        self._sec_pxls  = w * h
        self._sec_bytes = w * h * dtype.itemsize
        self.__cache_size = 0


    # General
    def save(self):
        if self._readonly: raise AttributeError('readonly')
        self._header.save()
    def close(self): pass
    def __del__(self): self.close()
    @property
    def w(self): return self._w
    @property
    def h(self): return self._h
    @property
    def d(self): return self._d
    @property
    def dtype(self): return self._dtype
    @property
    def shape(self): return self._shape
    @property
    def readonly(self): return self._readonly
    @property
    def header(self): return self._header
    def __len__(self): return self._d


    # Internal section reading and writing - primary functions to be implemented by base classes
    @abstractmethod
    def _get_section(self, i, seq): pass
    @abstractmethod
    def _set_section(self, i, im, seq): pass # if i == self._d we are appending
    @abstractmethod
    def _del_sections(self, start, stop): pass # step of 1, if stop == self._d then they are being removed from the end

    # Caching of slices (without caching these just forward to the above functions 
    @property
    def cache_size(self): return self.__cache_size
    @cache_size.setter
    def cache_size(self, value):
        """
        Set the size of the cache. This number of recently accessed or set slices will be available
        without disk reads. Default is 0 which means no slices are cached. If -1 then all slices
        will be cached as they are accessed.
        """
        value = int(value)
        if value < -1: raise ValueError
        if value == 0: # removing cache
            if self.__cache_size != 0:
                del self.__cache
                del self.__cache_order
        elif value != 0:
            if self.__cache_size == 0: # creating cache
                self.__cache_ = [None]*self._d
                self.__cache_order = deque()
            elif value != -1:
                while len(self.__cache_order) > value: # cache is shrinking
                    self.__cache[self.__cache_order.popleft()] = None
        self.__cache_size = value
    def cache_size_in_bytes(self, bytes): self.cache_size = bytes // self._sec_bytes;
    def __cache_it(self, i):
        # Places an index into the cache list (but doesn't do anything with the cached data itself)
        # Returns true if the index is already cached (in which case it is moved to the back of the queue)
        # Otherwise if the queue is full then the oldest thing is removed from the cache
        already_in_cache = self.__cache[i] != None
        if already_in_cache:
            self.__cache_order.remove(i)
        elif len(self.__cache_order) == self.__cache_size: # cache full
            self.__cache[self.__cache_order.popleft()] = None
        self.__cache_order.append(i)
        return already_in_cache
    def __get_section(self, i, seq):
        if self.__cache_size == 0: return self._get_section(i, seq)
        if not self.__cache_it(i):
            self.__cache[i] = im_standardize_dtype(self._get_section(i, seq))
        return self.__cache[i] # the cache is full on un-writeable copies already, so no .copy()
    def __set_section(self, i, im, seq):
        im = im_standardize_dtype(im)
        if self._shape != im.shape or self._dtype != im.dtype: raise ValueError('im')
        self._set_section(i, im)
        if self.__cache_size != 0:
            self.__cache_it(i)
            if im.flags.writeable:
                # TODO: not writable != truly read-only/immutable...
                # TODO: make copy-on-write
                im = im.copy()
                im.flags.writeable = False
            if i == self._d: self.__cache.append(im)
            else:            self.__cache[i] = im
    def __del_sections(self, start, stop):
        self._del_sections(start, stop)
        if self.__cache_size != 0:
            for i in xrange(start, stop):
                if self.__cache[i] != None: self.__cache_order.remove(i)
            if stop != self._d:
                # need to update all values greater than stop
                shift = stop - start
                vals = range(stop, self._d)
                for _ in xrange(len(self.__cache_order)):
                    val = self.__cache_order.popleft()
                    if val in vals: val -= shift
                    d.append(val)
            del self.__cache[start:stop]
            
        
    # Getting Slices
    def __getitem__(self, idx):
        """
        Get data of slices. Accepts integral, slice, or iterable indices. When using an integral
        index this returns the actual slice. For slice and iterable indices it returns a view into
        the ImageStack. The images are still not loaded until actually needed in these cases.
        """
        if isinstance(idx, Integral):
            if idx < 0: idx += self._d
            if not (0 <= idx < self._d): raise IndexError()
            return self.__get_section(idx, False)
        elif isinstance(idx, slice):
            start, stop, step = read_slice(idx, self._d)
            return ImageStackView(self, range(start, stop, step), step == 1)
        elif isinstance(idx, Iterable):
            idx = list(idx)
            return ImageStackView(self, idx, is_sequential(idx))
        else: raise TypeError('index')
    def __iter__(self):
        if self._d > 0:
            yield self._get_section(0, False)
            for i in xrange(1, self._d): yield self.__get_section(i, True)
    @property
    def stack(self):
        """Get the entire stack as a single 3D image."""
        from numpy import empty
        s = empty((self._d,) + self._shape, dtype=self._dtype)
        for i, sec in enumerate(self): stack[i,:,:] = sec
        return s


    # Setting and adding slices        
    def __setitem__(self, idx, im):
        """Sets a slice to a new image, writing it to disk. Accepts only integral indices."""
        if self._readonly: raise Exception('readonly')
        if not isinstance(idx, Integral): raise TypeError('index')
        if idx < 0: idx = self._d - idx
        if idx >= self._d: raise ValueError('index')
        self._set_section(idx, im, False)
    def append(self, im):
        """Appends a single slice, writing it to disk."""
        if self._readonly: raise Exception('readonly')
        self._set_section(self._d, im, False)
        self._d += 1
        self._header._update_depth(self._d)
    def extend(self, ims):
        """Appends many slices, writing them to disk."""
        if self._readonly: raise Exception('readonly')
        seq = False
        for im in ims:
            self._set_section(im, self._d, seq)
            self._d += 1
            seq = True
        self._header._update_depth(self._d)
    def __iadd__(self, im):
        """Either appends or extends depending on the data type."""
        if isinstance(im, ndarray):
            try:               self.append(im_standardize_dtype(im))
            except ValueError: self.extend(im)
        else:
            self.extend(im)
    
    # Removing slices
    def __delitem__(self, idx):
        """
        Removes slices. Accepts integral and slices indices. Updates the disk immediately.
        Typically only efficient when removing from the end (e.g. del ims[-1] or del ims[x:]).
        """
        if self._readonly: raise Exception('readonly')
        if isinstance(idx, Integral):
            if idx < 0: idx += self._d
            if not (0 <= idx < self._d): raise IndexError()
            self.__del_sections(idx, idx+1)
            self._d -= 1
        elif isinstance(idx, slice):
            start, stop, step = read_slice(idx, self._d)
            if start + 1 == stop: return
            if step == +1:
                self.__del_sections(start, stop)
                self._d -= stop - start
            elif step == -1:
                self.__del_sections(stop+1, start+1)
                self._d -= start - stop
            else:
                # make step negative so we always move from the end towards the start
                if step > 0: start, stop, step = stop-1, start-1, -step
                for i in xrange(start, stop, step): self.__del_sections(i, i+1)
                self._d -= (1 - start + stop + step) // step
        else: raise TypeError('index')
        self._header._update_depth(self._d)
    def shorten(self, count=1):
        """Removes 'count' slices from the end of the stack (default 1)."""
        #del self[-count:]
        if self._readonly: raise Exception('readonly')
        if count <= 0 or count > self._d: raise ValueError('count')
        self.__del_sections(self._d - count, self._d)
        self._d -= count
        self._header._update_depth(self._d)
    def clear(self):
        """Remove all slices from the stack."""
        #del self[:]
        if self._readonly: raise Exception('readonly')
        self.__del_sections(0, self._d)
        self._d = 0
        self._header._update_depth(0)


class ImageStackView(ImageStack):
    """
    A view of an image stack. This is given when requesting more than one slice from an image stack.
    It acts mostly like the underlying image stack (allowing to get, set, and delete slices for the
    base image stack). It however does not support the following:
     * changing caching options (it will use the caching of the base image stack)
     * appending slices
     * the header is completely none-functional (cannot get or set anything or save it)
    """
    # TODO: add the ability for headers to have some utility?
    def __init__(self, ims, ind, seq): # ImageStack, list of indices, if those indices are sequential
        if isinstance(ims, ImageStackView):
            # TODO: decide if we should "collapse" views or not (currently this collapses)
            seq = seq and ims._seq
            ind = [ims._ind[i] for i in ind]
            ims = ims._ims
        self._ims = ims
        self._ind = list(ind)
        self._seq = bool(seq)
        super(ImageStackView, self).__init__(self._ims._w, self._ims._h, len(ind), self._ims._dtype,
                                             ImageStackViewHeader(self), self._ims._readonly)

    @property
    def base(self):
        """Get the base image stack for this view."""
        return base._ims
    def save(self): raise AttributeError('views cannot be saved')
    @ImageStack.cache_size.setter
    def cache_size(self, value): raise AttributeError('views do not directly support caching, set the cache size on the base ImageStack instead')
    def _get_section(self, i, seq): return self._ims._get_section(self._ind[i], seq and self._seq)
    def _set_section(self, i, im, seq):
        if i >= len(self._ind): raise AttributeError('views cannot be appended to')
        self._ims._set_section(self._ind[i], im, seq and self._seq)
    def _del_sections(self, start, stop):
        if self._seq : self._ims._del_sections(self, self._ind[start], self._ind[stop])
        else:
            for i in self._ind[stop-1:start-1:-1]: self._ims._del_sections(i, i+1)
        del self._ind[start:stop]

class Header(DictionaryWrapperWithAttr):
    """
    The header of an image stack. This is primarily a dictionary with built-in checking of names and
    values based on the image stack type. In general this provides image-format specific information
    and cannot be reliably queried between image stack types. One day a generalized "converter" may
    be created for common header values between different types.

    **Implementation Notes**
    The implementor must set the class fields _imstack, _fields, and _data before calling
    super.__init__(). These contain the image stack for which this header is connected to, the
    known fields as a dictionary or ordered dictionary containing field-name to Field object, and
    the data for those fields as a dictionary of field-name to data. If you wish to delay-load, then
    make sure to call super._check() whenever the header is loaded (and do not call
    super.__init__()).

    The functions self.save(), self._update_depth(d), and self._get_field_name(f) must also be
    implemented. More information about those is provided in the abstract definitions.

    As a reminder, since this is an exension of DictionaryWrapperWithAttr, to set class fields that
    are not header fields you must either have them in the class definition or interact with
    self.__dict__.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        if any((x not in self.__dict__) for x in ('_imstack', '_fields', '_data')): raise TypeError
        self._check()

    def _check(self):
        for _key,value in self._data.items():
            key = self._get_field_name(_key)
            if key == None: raise KeyError('%s cannot be in header' % _key)
            if key != _key: # key was changed, update data
                self._data[key] = value
                del self._data[_key]
        for key,field in self._fields.iteritems():
            if key in self._data: self._data[key] = field.cast(self._data[key], self)
            elif not field.opt: raise KeyError('%s must be in header' % key)


    @abstractmethod
    def save(self): pass
    @abstractmethod
    def _update_depth(self, d):
        """
        This is called whenever the depth of the associated image stack is changed. This may be
        because the image stack has shrunk or grown. Any header properties that need to be updated
        because of the this change should be updated here. The new depth is given. The image stack
        depth has already been changed as well (so the old depth is not directly queriable).
        """
        pass
    @abstractmethod
    def _get_field_name(self, f):
        """
        Get the field name given a requested name. This should tranform the field name if applicable
        (for example, if the format treats field names in a case-insensitive way this should change
        the name to lower-case, or if there are multiple names for the same field this should pick
        one). If the field name is illegal, None should be returned. If the returned field-name is
        not in self._fields then a new field is created with that name that is optional and has no
        restrictions on value. Some examples:

        Formats that have no fields:
            return None
        Formats that have a fixed set of fields, case-sensitive:
            return f if f in self._fields else None
        Formats that accept any field name, case-insensitive:
            return f.lower()
        """
        pass


    # Retrieval
    def __getitem__(self, key):  return self._data[self._get_field_name(key)]
    def __contains__(self, key): return self._get_field_name(key) in self._data

    # Deleting functions
    def __delitem__(self, key):
        if self._imstack._readonly: raise AttributeError('header is readonly')
        _key, key = key, self._get_field_name(key)
        if key not in self._data: raise KeyError('%s not in header' % _key)
        if key in self._fields and not self._fields[key].opt: raise AttributeError('%s cannot be removed from header' % _key)
        del self._data[key]
    def clear(self):
        # Only clears optional and custom values
        if self._imstack._readonly: raise AttributeError('header is readonly')
        for k in self._data:
            if k not in self._fields or self._fields[k].opt:
                del self._data[k]
            else:
                pass # TODO: produce a warning
    def pop(self, key, default=None):
        if self._imstack._readonly: raise AttributeError('header is readonly')
        _key, key = key, self._get_field_name(key)
        if key not in self._data: raise KeyError('%s not in header' % _key)
        if key in self._fields and not self._fields[key].opt: raise AttributeError('%s cannot be removed from header' % _key)
        value = self._data[key]
        del self._data[key]
        return value
    def popitem(self): raise AttributeError('Cannot pop fields from header')

    # Setting/inserting functions
    def __setitem__(self, key, value):
        if self._imstack._readonly: raise AttributeError('header is readonly')
        _key, key = key, self._get_field_name(key)
        if key == None: raise KeyError('%s cannot be added to header' % _key)
        f = self._fields.get(key, None)
        if f == None: self._data[key] = value
        elif f.ro: raise AttributeError('%s cannot be edited in header' % _key)
        else: self._data[key] = f.cast(value, self)
    def setdefault(self, key, default = DictionaryWrapperWithAttr._marker):
        if self._imstack._readonly: raise AttributeError('header is readonly')
        _key, key = key, self._get_field_name(key)
        if key == None: raise KeyError('%s cannot be added to header' % _key)
        if key not in self._data:
            if default is DictionaryWrapper._marker: raise KeyError
            f = self._fields.get(key, None)
            if f == None: self._data[key] = default
            elif f.ro: raise AttributeError('%s cannot be edited in header' % _key)
            else: self._data[key] = f.cast(default, self)
        return self._data[key]
    def update(self, d=None, **kwargs):
        if self._imstack._readonly: raise AttributeError('header is readonly')
        if d == None: d = kwargs
        itr = d.iteritems() if isinstance(d, dict) else d
        for k,v in itr: self[k] = v

class Field():
    """
    A image stack header field. The base class takes a casting function, if the value is read-only
    (to the external world, default False), if the field is optional (default True), and a default
    value (which is currently not used).

    The cast function should take a wide array of values and convert them if possible to the true
    type. If the input value cannot be converted, a TypeError or ValueError should be raised.
    """
    def __init__(self, cast=None, ro=False, opt=True, default=None):
        self._cast = cast
        self.ro    = ro
        self.opt   = opt
        # TODO: use default value somewhere
    def cast(self, v, h): return self._cast(v)
class FixedField(Field):
    """
    This is an image stack header field that can have only one value ever. It is readonly. We still
    take a casting function to do type conversion, but we also do an automatic check on the return
    from cast to see if it is identical to the value we are fixed to.
    """
    def __init__(self, cast, value, opt=True):
        self._cast = cast
        self.value = cast(value)
        self.ro    = True
        self.opt   = opt
    def cast(self, v, h):
        if self._cast(v) != self.value: raise ValueError
        return self.value
class NumericField(Field):
    """
    A numeric-based field. By default the casting operator is "int" and no upper or lower bound is
    placed on the value. You can set cast, min, and max to change this behavior.
    """
    def __init__(self, cast=int, min=None, max=None, ro=False, opt=True, default=None):
        # Note: min and max are inclusive, if None no restriction on that end
        self._cast = cast
        self.min   = cast(min) if min != None else None
        self.max   = cast(max) if max != None else None
        self.ro    = ro
        self.opt   = opt
    def cast(self, v, h):
        v = self._cast(v)
        if self.min != None and v < self.min or self.max != None and v > self.max: raise ValueError('value not in range')
        return v

class ImageStackViewHeader(Header):
    """
    The header for an image stack view. Makes it so there are no fields and it cannot be saved.
    """
    _imstack = None
    _fields = None
    _data = None
    def __init__(self, ims): self._imstack = ims; _fields = {}; _data = {};
    def _get_field_name(self, f): return None
    def _update_depth(self, d): pass
    def save(self): raise AttributeError('views cannot be saved')

# Import additional formats
import formats
