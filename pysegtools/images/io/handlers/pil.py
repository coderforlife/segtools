"""PIL (Python Imaging Library) supported image stacks and slices"""
# This uses PIL to read images. For some formats PIL supports image stacks and these are exposed as
# stacks. All formats are exposed as image sources. Some formats require special handling since they
# aren't implemented well. Hopefully in newer versions of PIL these special handlings won't get in
# the way.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import struct
from io import open
from abc import ABCMeta, abstractproperty, abstractmethod

from PIL import Image
from numpy import uint8

from .._stack import FileImageStack, FileImageSlice, FileImageStackHeader, FixedField
from .._single import FileImageSource
from .._util import check_file_obj
from ...types import get_im_dtype_and_nchan
from ..._util import String, _bool

from distutils.version import StrictVersion
if not hasattr(Image, 'PILLOW_VERSION') or StrictVersion(Image.PILLOW_VERSION) < StrictVersion('2.0'):
    raise ImportError

########## PIL dtypes ##########
_native = None
_mode2dtype = None
_dtype2mode = None
def _init_pil():
    from sys import byteorder
    from numpy import sctypes, dtype
    from numpy import bool_, uint8,uint16,uint32, int8,int16,int32, float32,float64 #float16
    from ...types import create_im_dtype as d
    global _native, _mode2dtype, _dtype2mode
    _native = byteorder!='little'
    
    Image.init()
    # Add common extensions for the SPIDER format
    Image.register_extension("SPIDER", ".spi")
    Image.register_extension("SPIDER", ".stk")
    
    # TODO: could use len(PIL.ImageMode.getmode(mode).bands) and PIL.ImageMode.getmode(mode).basetype to auto-generate these conversions
    _mode2dtype = {
        #'P' is a special case
        # Some of these modes will actually never show up because they are raw modes.

        'RGB': d(uint8,False,3), 'RGBX':d(uint8,False,4), # however the fourth one is "padding"
        'RGBA':d(uint8,False,4), 'RGBa':d(uint8,False,4), # non-premultiplied and pre-multiplied
        'CMYK':d(uint8,False,4), 'YCbCr':d(uint8,False,3),
        'LAB': d(uint8,False,3), 'HSV':d(uint8,False,3),
        'LA':  d(uint8,False,2), # grayscale with alpha

        '1':d(bool_),'L':d(uint8),'I':d(int32,_native),
        'I;8':d(uint8),'I;8S':d(int8),
        'I;16':d(uint16),'I;16L':d(uint16),'I;16B':d(uint16,True),'I;16N':d(uint16,_native),
        'I;16S':d(int16),'I;16LS':d(int16),'I;16BS':d(int16,True),'I;16NS':d(int16,_native),
        'I;32':d(uint32),'I;32L':d(uint32),'I;32B':d(uint32,True),'I;32N':d(uint32,_native),
        'I;32S':d(int32),'I;32LS':d(int32),'I;32BS':d(int32,True),'I;32NS':d(int32,_native),

        'F':d(float32,_native),
        #'F;16F':d(float16),'F;16BF':dt(float16,True),'F;16NF':dt(float16,_native),
        'F;32F':d(float32),'F;32BF':d(float32,True),'F;32NF':d(float32,_native),
        'F;64F':d(float64),'F;64BF':d(float64,True),'F;64NF':d(float64,_native),
    }
    _dtype2mode = {
        # mode, rawmode (little endian), rawmode (big endian)
        # Multi-channel and bit images are special cases
        #float16:('F','F;16F','F;16BF'), # F;16 can only come from integers...
    }
    # Build _dtype2mode
    for t in sctypes['uint']:
        nb = dtype(t).itemsize
        if   nb == 1: _dtype2mode[t] = ('L','L','L')
        elif nb == 2: _dtype2mode[t] = ('I;16','I;16','I;16B')
        elif nb == 4: _dtype2mode[t] = ('I','I;32','I;32B')
        else: nb = str(nb*8); _dtype2mode[t] = ('I','I;'+nb,'I;'+nb+'B')
    for t in sctypes['int']:
        nb = dtype(t).itemsize
        if nb == 1: _dtype2mode[t] = ('I','I;8S','I;8S')
        else: nb = str(nb*8); _dtype2mode[t] = ('I','I;'+nb+'S','I;'+nb+'BS')
    for t in sctypes['float']:
        nb = dtype(t).itemsize
        if nb < 4: continue
        nb = str(nb*8); _dtype2mode[t] = ('F','F;'+nb+'F','F;'+nb+'BF')


########## PIL interaction class ##########
def imsrc2pil(im):
    im = im.data
    st, sh = im.strides[0], im.shape[1::-1]
    dt, nchan = get_im_dtype_and_nchan(im)
    if nchan > 1:
        if dt.type != uint8 or nchan > 4: raise ValueError
        mode = ('LA','RGB','RGBA')[nchan-2]
        return Image.frombuffer(mode, sh, im.data, 'raw', mode, st, 1)
    elif dt.kind == 'b':
        # Make sure data is actually saved as 1-bit data (both SciPy and PIL seem to be broken with this)
        im = im * uint8(255)
        return Image.frombuffer('L', sh, im.data, 'raw', 'L', st, 1).convert('1')
    else:
        mode = _dtype2mode.get(dt.type)
        if mode is None: raise ValueError
        return Image.frombuffer(mode[0], sh, im.data, 'raw', mode[2 if _native else 1], st, 1)
def _accept_all(im): return True
def _accept_none(im): return False
class _PILSource(object):
    """
    This is the class that does most of the work interacting with the PIL library. It is
    implemented this way so specific formats can derive from this class to change some of the
    behaviors. When first constructed this simply has references to the format, opener class,
    accepting function, and saving function. Upon opening or creating, a copy is returned which has
    various other attributes.
    """
    def __init__(self, frmt, open, accept, save):
        self.format = frmt
        self._open = open
        self.accept = (_accept_none if open is None else _accept_all) if accept is None else accept
        self._save = save
    @property
    def readable(self): return self._open is not None
    @property
    def writable(self): return self._save is not None
    
    def _open_pil_image(self, f, filename, **options):
        """
        Opens a PIL image object from the file object and filename with the given options. Can be
        overriden by subclasses to handle options. Default implementation does nothing with options.
        """
        return self._open(f, filename)
    def _save_pil_image(self, f, filename, im, palette=False, **options):
        """
        Saves a PIL image object to the file object and filename with the given options. Subclasses
        can override this to deal with the options. Otherwise options are just passed to the image's
        "encoderinfo" (except "palette" which is used to convert the palette of the image).
        """
        if palette is not False:
            im = im.quantize() if palette is True else im.quantize(colors=palette)
        im.encoderinfo = options
        im.encoderconfig = ()
        self._save(im, f, filename)

    def _parse_open_options(self, **options):
        """
        Parse the options list for options this format supports while opening. Return both the
        parsed options and the remaining unused in seperate dictionaries. Make sure to call
        recusively.
        """
        return {}, options

    def _parse_save_options(self, palette=False, **options):
        """
        Parse the options list for options this format supports while saving. Return both the
        parsed options and the remaining unused in seperate dictionaries. Make sure to call
        recusively.
        """
        if palette is not True and palette is not False:
            if isinstance(palette, String):
                pal_lower = palette.lower()
                if   pal_lower in ('true', 't'): palette = True
                elif pal_lower in ('false','f'): palette = False
                elif palette.isdigit():
                    palette = int(palette)
                    if palette < 1 or palette > 256: raise ValueError('Invalid palette')
                else: raise ValueError('Invalid palette')
            elif isinstance(palette, Integral):
                from numbers import Integral
                palette = int(palette)
                if palette < 1 or palette > 256: raise ValueError('Invalid palette')
            else: raise ValueError('Invalid palette')
        return {'palette':palette}, options

    def _parse_options(self, open, save, **options):
        """
        Parse the options list for options this format supports while opening and saving. Do not
        override this function, instead override _parse_open_options and _parse_save_options.
        """
        if open: open_options, options = self._parse_open_options(**options)
        else: open_options = {}
        if save: save_options, options = self._parse_save_options(**options)
        else: save_options = {}
        if len(options) > 0: raise ValueError("Unsupported options given")
        return open_options, save_options

    def _copy(self, im, filename, readonly, open_options, save_options):
        """
        Copies the format, _open, accept, and _save properties of this object into a new object of
        the same type and adds the new properties.
        """
        c = type(self)(self.format, self._open, self.accept, self._save)
        c.im = im
        c.filename = filename
        c.readonly = readonly
        c.open_options = open_options
        c.save_options = save_options
        return c
    
    def open(self, file, readonly, **options):
        """
        Opens a file object/filename image with the given options. Returns a copy of this template
        object that has the image object and other attributes.
        """
        if hasattr(self, 'im'): raise RuntimeError("Not a template PILSource object")
        if not self.readable: raise ValueError("Cannot read from file format")
        if not readonly and not self.writable: raise ValueError("Cannot write to file format")

        open_options, save_options = self._parse_options(True, not readonly, **options)

        if isinstance(file, String):
            filename, f = file, open(file, 'rb')
            try:
                return self._copy(self._open_pil_image(f, filename, **open_options),
                                  filename, readonly, open_options, save_options)
            except StandardError: f.close(); raise
        
        # File-like object
        if not check_file_obj(file, True, not readonly, True): raise ValueError('Unusable file object')
        start = file.tell()
        filename = file.name if (hasattr(file, 'name') and isinstance(file.name, String) and
                                 len(file.name) > 0 and file.name[0] != '<' and file.name[-1] != '>') else ''
        try:
            return self._copy(self._open_pil_image(file, filename, **open_options),
                              filename, readonly, open_options, save_options)
        except StandardError: f.seek(start); raise
    
    def openable(self, filename, prefix, readonly, **options):
        """
        Checks if a file is openable. Prefix is the first 16 bytes from the file. This should not be
        overriden. Instead override _parse_[open_|save_]options or _open_pil_image.
        """
        if not (self.readable and (readonly or self.writable) and self.accept(prefix)): return False
        try: self.open(filename, readonly, **options).close(); return True
        except (SyntaxError, IndexError, TypeError, ValueError, EOFError, struct.error) as ex: return False

    def create(self, filename, im, writeonly, **options):
        """
        Creates a new image file. Returns a copy of this template object that has the image object
        and other attributes. im is an ImageSource.
        """
        if hasattr(self, 'im'): raise RuntimeError("Not a template PILSource object")
        if not self.writable: raise ValueError("Cannot write to file format")
        if not writeonly and not self.readable: raise ValueError("Cannot read from file format")
        if len(options) > 0: raise ValueError("No options are supported")

        open_options, save_options = self._parse_options(not writeonly, True, **options)

        # Save image
        pil = imsrc2pil(im)
        with open(filename, 'wb') as f: self._save_pil_image(f, filename, pil, **save_options)
        
        # Open image
        if writeonly:
            # if writeonly we have to cache the dtype and shape properties
            s = self._copy(None, filename, False, {}, save_options)
            s._dtype = im.dtype
            s._shape = im.shape
            return s
        f = open(filename, 'rb')
        try:
            return self._copy(self._open_pil_image(f, filename, **open_options),
                              filename, False, open_options, save_options)
        except StandardError: f.close(); raise
    
    def creatable(self, writeonly, **options):
        """
        Checks if a file is creatable. UnlikeThis should not be overriden. Instead override
        _parse_[open_|save_]options.
        """
        try: open_options, save_options = self._parse_options(not writeonly, True, **options)
        except StandardError: return False
        return self.writable and (not writeonly or self.readable)

    # Only available after open/create
    def close(self):
        if hasattr(self, 'im'):
            if hasattr(self.im, 'close'): self.im.close()
            del self.im
    @property
    def header_info(self):
        """Gets the header info for a 2D image. In stacks, this gets the current slice's header."""
        if self.im is None:
            h = {'format':self.format}
        else:
            h = {'format':self.im.format}
            h.update(self.im.info)
        return h
    @property
    def is_stack(self): return False
    @property
    def dtype(self):
        dt = self.dtype_raw
        if self.im is not None and self.im.mode=='P' and self.im.info['transparency']:
            # need to add an extra channel for the transparency data
            from numpy import dtype
            dt = dtype((dt.base, 2 if len(dt.shape) == 0 else (dt.shape[0]+1)))
        return dt
    @property
    def dtype_raw(self): return (self._dtype if self.im is None else
                                 _mode2dtype[self.im.palette.mode if self.im.mode=='P' else self.im.mode])
    @property
    def shape(self): return self._shape if self.im is None else tuple(reversed(self.im.size))
    def _get_palette(self, dt):
        from numpy import frombuffer
        pal = self.im.palette
        return frombuffer(pal.tobytes() if pal.rawmode is None else (
            pal.palette if pal.rawmode == pal.mode else self.im.getpalette()), dtype=dt)
    @property
    def data(self): # return ndarray
        from numpy import frombuffer
        dt = self.dtype_raw # the intermediate dtype
        dt_final = self.dtype # the resulting dtype
        if self.im.mode == 'P':
            pal = self._get_palette(dt)
            a = pal.take(frombuffer(self.im.tobytes(), dtype=uint8), axis=0)
            if 'transparency' in self.im.info:
                from numpy import zeros, place, concatenate
                from ...types import get_dtype_max
                if a.ndim == 1: a = a[:,None] # make sure it is 2D
                trans = pal[self.im.info['transparency']]
                alpha = zeros((a.shape[0],1), dtype=dt.base)
                place(alpha, (a!=trans).all(1), get_dtype_max(dt.base))
                a = concatenate((a,alpha), axis=1)
        else:
            a = frombuffer(self.im.tobytes(), dtype=dt)
        return a.reshape(tuple(reversed(dt_final.shape+self.im.size)))
    def set_data(self, im): # im is an ImageSource
        reopen = self.im is not None # if writeonly don't reopen image
        if reopen: self.im.close()
        else:
            self._dtype = im.dtype
            self._shape = im.shape
        im = imsrc2pil(im)
        if self.filename:
            with open(self.filename, 'wb') as f: self._save_pil_image(f, self.filename, im, **self.save_options)
            if reopen:
                f = open(self.filename, 'rb')
                try: self.im = self._open_pil_image(f, self.filename, **self.open_options)
                except StandardError: f.close(); raise
        else:
            self.file.seek(0)
            self._save_pil_image(f, "", im, **self.save_options)
            if reopen:
                self.file.seek(0)
                self.im = self._open_pil_image(self.file, "", **self.open_options)
    def rename(self, renamer, filename):
        if not self.filename: raise ValueError('no filename')
        reopen = self.im is not None
        if reopen: self.im.close()
        self.filename = filename
        renamer(filename)
        if reopen:
            f = open(filename, 'rb')
            try: self.im = self._open_pil_image(f, filename, **self.open_options)
            except StandardError: f.close(); raise

def _get_prefix(file):
    from os import SEEK_CUR
    if isinstance(file, String):
        with open(file, 'rb') as f: return f.read(16)
    data = file.read(16)
    file.seek(-len(data), SEEK_CUR)
    return data
def _open_source(sources, frmt, f, readonly, **options):
    if frmt is not None:
        if frmt not in sources: raise ValueError('Unknown format')
        return sources[frmt].open(f, readonly, **options)
    prefix = _get_prefix(f)
    for s in sources.itervalues():
        if s.readable and (readonly or s.writable) and s.accept(prefix):
            try: return s.open(f, readonly, **options)
            except (SyntaxError, IndexError, TypeError, ValueError, struct.error): pass
    raise ValueError('Unknown format')
def _openable_source(sources, frmt, f, filename, readonly, **options):
    prefix = _get_prefix(f)
    return (any(s.openable(filename, prefix, readonly, **options) for s in sources.itervalues())
            if frmt is None else
            (frmt in sources and sources[frmt].openable(filename, prefix, readonly, **options)))


########## Image Source ##########
# TODO: There are various other options that I am not supporting here due to rarity or being able to
# add them post-saving, but they could be added as needed.
# Some examples of good things to add would be:
#  * saving params dpi and icc-profile (lots of formats support these two options)
#  * GIF/PNG's saving param transparency
#  * JPEG/WebP saving param exif
#  * JPEG2000's opening params mode/reduce/layers and tons of saving params I don't understand
class _EPSSource(_PILSource):
    def _parse_open_options(self, scale=1, **options):
        open_options, options = super(_EPSSource, self)._parse_open_options(**options)
        scale = int(scale)
        if scale < 1: raise ValueError('Invalid scale')
        open_options['scale'] = scale
        return open_options, options
    def _open_pil_image(self, f, filename, scale=1, **options):
        im = super(_EPSSource, self)._open_pil_image(f, filename, **options)
        if scale != 1: im.load(scale=scale)
        return im
class _GIFSource(_PILSource):
    def _parse_open_options(self, local=False, **options):
        open_options, options = super(_GIFSource, self)._parse_open_options(**options)
        open_options['local'] = _bool(local)
        return open_options, options
    def _open_pil_image(self, f, filename, local=False, **options):
        im = super(_GIFSource, self)._open_pil_image(f, filename, **options)
        if local and im.tile[0][0] == 'gif':
            tag, (x0, y0, x1, y1), offset, extra = im.tile[0]
            im.size = (x1-x0, y1-y0)
            im.tile = [(tag, (0,0) + im.size, offset, extra)]
        return im
class _ICNSSource(_PILSource):
    def _parse_open_options(self, size=None, **options):
        open_options, options = super(_ICNSSource, self)._parse_open_options(**options)
        if size is not None:
            if isinstance(size, String): size = size.split(',')
            size = tuple(int(i) for i in size)
            if len(size) == 1: size = (size[0], size[0], 1)
            elif len(size) == 2: size = size + (1,)
            if len(size) != 3 or size[0] < 1 or size[1] < 1 or size[2] not in (1,2): raise ValueError('Invalid size')
            open_options['size'] = size
        return open_options, options
    def _open_pil_image(self, f, filename, size=None, **options):
        im = super(_ICNSSource, self)._open_pil_image(f, filename, **options)
        if size is not None:
            if size not in list(im.info['sizes']): raise ValueError('Size not found in file')
            im.size = size
        return im
class _JPEGSource(_PILSource):
    def _parse_save_options(self, quality=75, optimize=False, progressive=False, **options):
        save_options, options = super(_JPEGSource, self)._parse_save_options(**options)
        quality = int(quality)
        if quality < 1 or quality > 100: raise ValueError('Invalid quality')
        save_options['quality'] = quality
        if _bool(optimize): save_options['optimize'] = True
        if _bool(progressive): save_options['progressive'] = True
        return save_options, options
class _PALMSource(_PILSource):
    def _parse_save_options(self, bpp=4, **options):
        save_options, options = super(_PNGSource, self)._parse_save_options(**options)
        bpp = int(bpp)
        if bpp not in (1, 2, 4): raise ValueError('Invalid bpp')
        save_options['bpp'] = bpp
        return save_options, options
class _PNGSource(_PILSource):
    def _parse_save_options(self, compression=6, optimize=False, **options):
        save_options, options = super(_PNGSource, self)._parse_save_options(**options)
        compression = int(compression)
        if compression < 0 or compression > 9: raise ValueError('Invalid compression')
        save_options['compression'] = compression
        if _bool(optimize): save_options['optimize'] = True
        return save_options, options
    # The text values seem to be already included in info, so don't double add them
    #@property
    #def header_info(self):
    #    h = super(_PNGSource, self).header_info
    #    h.update(self.im.text) 
    #    return h
def _get_tiff_tags(im):
    def _unwrap(x):
        return x[0] if isinstance(x, tuple) and len(x) == 1 else x
    return {k:_unwrap(v) for k,v in im.tag.named().iteritems()} 
class _TIFFSource(_PILSource):
    compressions = {
        "none":"raw",
        "ccitt-1d":"tiff_ccitt",
        "ccitt-group-3":"group3",
        "ccitt-group-4":"group4",
        "lzw":"tiff_lzw",
        #"tiff_jpeg"     # obsolete, rare
        "jpeg":"jpeg",   # uncommon
        "deflate":"tiff_adobe_deflate", # uncommon
        #"tiff_raw_16"   # undocumented? COMPRESSION_CCITTRLEW?
        "packbits":"packbits",
        "thunderscan":"tiff_thunderscan", # rare, proprietary
        #"tiff_deflate"  # obsolete/experimental, uncommon
        #"tiff_sgilog"   # experimental, rare
        #"tiff_sgilog24" # experimental, rare
    }
    def _parse_save_options(self, compression="packbits", **options):
        save_options, options = super(_TIFFSource, self)._parse_save_options(**options)
        compression = compression.lower()
        if compression not in _TIFFSource.compressions: raise ValueError('Invalid compression')
        save_options['compression'] = _TIFFSource.compressions[compression]
        return save_options, options
    @property
    def header_info(self):
        h = super(_TIFFSource, self).header_info
        h.update(_get_tiff_tags(self.im))
        return h
class _WEBPSource(_PILSource):
    def _parse_save_options(self, quality=80, lossless=False, **options):
        save_options, options = super(_WEBPSource, self)._parse_save_options(**options)
        quality = int(quality)
        if quality < 1 or quality > 100: raise ValueError('Invalid quality')
        save_options['quality'] = quality
        if _bool(lossless): save_options['lossless'] = True
        return save_options, options

_sources, _read_formats, _write_formats = None, None, None
def _init_source():
    from types import ClassType
    from PIL.ImageFile import StubImageFile
    global _sources, _read_formats, _write_formats
    if _mode2dtype is None: _init_pil()
    _source_classes = {
            'EPS' : _EPSSource,
            'GIF' : _GIFSource,
            'ICNS': _ICNSSource,
            'JPEG': _JPEGSource,
            'PALM': _PALMSource,
            'PNG' : _PNGSource,
            'TIFF': _TIFFSource,
            'WEBP': _WEBPSource,
        }

    stub_formats = set(frmt for frmt,(clazz,accept) in Image.OPEN.iteritems() if
                       isinstance(clazz,(type,ClassType)) and issubclass(clazz,StubImageFile))
    stub_formats.add('MPEG') # MPEG is not registered properly as a stub
    _read_formats = frozenset(Image.OPEN) - stub_formats
    _write_formats = frozenset(Image.SAVE) - stub_formats
    _sources = {
        frmt:_source_classes.get(frmt,_PILSource)(frmt,clazz,accept,Image.SAVE.get(frmt))
        for frmt,(clazz,accept) in Image.OPEN.iteritems()
        if not isinstance(clazz,(type,ClassType)) or not issubclass(clazz,StubImageFile)
        }
    # Add write-only formats
    _sources.update({frmt:_source_classes.get(frmt,_PILSource)(frmt,None,None,Image.SAVE[frmt])
                     for frmt in (_write_formats-_read_formats)})

class PIL(FileImageSource):
    @staticmethod
    def __parse_opts(slice=None, **options):
        if slice is not None:
            # Slice was given, must be a stack-able type
            if _stacks is None: _init_stacks()
            slice = int(slice)
            if slice < 0: raise ValueError('Slice must be a non-negative integers')
            options['slice'] = slice
            return _stacks, options
        if _sources is None: _init_source()
        return _sources, options
            
    @classmethod
    def open(cls, f, readonly, format=None, **options):
        sources, options = PIL.__parse_opts(**options)
        return PIL(_open_source(sources, format, f, readonly, **options))

    @classmethod
    def _openable(cls, filename, f, readonly, format=None, **options):
        try: sources, options = PIL.__parse_opts(**options)
        except ValueError: return False
        return _openable_source(sources, format, f, filename, readonly, **options)

    @classmethod
    def create(cls, filename, im, writeonly, format=None, **options):
        if _sources is None: _init_source()
        if format is None:
            from os.path import splitext
            format = Image.EXTENSION.get(splitext(filename)[1].lower())
            if format is None: raise ValueError('Unknown file extension')
        return PIL(_sources[format].create(filename, im, writeonly, **options))

    @classmethod
    def _creatable(cls, filename, ext, writeonly, format=None, **options):
        if _sources is None: _init_source()
        if format is None:
            format = Image.EXTENSION.get(ext)
            if format is None: return False
        return _sources[format].creatable(writeonly, **options)

    @classmethod
    def name(cls): return "PIL"

    @classmethod
    def print_help(cls, width):
        from ....imstack import Help
        p = Help(width)
        p.title("Python Imaging Library (PIL) Image Handler")
        p.text("""
PIL is a common library for reading various image formats in Python. Technically we use the PILLOW
fork of PIL which is the standard replacement for PIL. This requires PILLOW v2.0 or newer.

This supports the option 'format' to force one of the supported formats listed below. Some formats
support multiple images in a single file (see 'PIL-Stack' for more information). For these formats
you may specify the option 'slice' to select which frame to use when loading them but not saving
them. If this option is given for a format that doesn't support slices, the slice is out of bounds,
or when saving, the file will fail to load.

When saving, only some formats will internally try to convert the image to a data-type that the
they support (mainly making RGB into palletted). If the format does not support the data-type it
will fail. To force the image to become palletted, use the 'pallette' option, which can be 'true'
to use the PIL function quantize or an integer 1-256 to choose the number of colors that quantize
will use. When reading, palletted images are always converted.

Extensions listed below are used to determine the format to save as if not explicit, during loading
the contents of the file are always used to determine the format.

Details on image formats and options can be found in the PILLOW documentation: http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html

Supported image formats (read/write):""")
        p.list(*sorted(cls.__add_exts(cls.formats(True, True))))
        p.newline()
        p.text("""Supported image formats [read-only]:""")
        p.list(*sorted(cls.formats(True, False)))
        p.newline()
        p.text("""Supported image formats [write-only]:""")
        p.list(*sorted(cls.__add_exts(cls.formats(False, True))))
        p.newline()
        p.text("""
Some formats support additional options when loading or saving. Not all options available in PIL
are available with this plugin.""")
        p.newline()
        p.text("""Supported loading options:""")
        p.list("EPS:  scale (positive integer)",
               "GIF:  local (boolean)",
               "ICNS: size (1-3 comma seperated integers)")
        p.newline()
        p.text("""Supported saving options:""")
        p.list("JPEG: quality (1-100), optimize (bool), progressive (bool)",
               "PALM: bpp (1, 2, or 4)",
               "PNG:  compression (1-9), optimize (bool)",
               "TIFF: compression (one of: none, CCITT-1D, CCITT-Group-3, CCITT-Group-4, LZW, JPEG, deflate, packbits (default), thunderscan)",
               "WEBP: quality (1-100), lossless (bool)")
        p.newline()
        p.text("See also:")
        p.list('PIL-Stack')

    @classmethod
    def __add_exts(cls, formats):
        frmt2exts = {}
        for ext,frmt in Image.EXTENSION.iteritems(): frmt2exts.setdefault(frmt,[]).append(ext)
        return [frmt+((' ('+(', '.join(frmt2exts[frmt])) + ')') if frmt in frmt2exts else '')
                for frmt in formats]

    @classmethod
    def formats(cls, read, write):
        if _sources is None: _init_source()
        if read: return (_read_formats & _write_formats) if write else (_read_formats - _write_formats)
        elif write: return _write_formats - _read_formats
        else: return frozenset()

    def __init__(self, source):
        self._source = source
        super(PIL, self).__init__(source.filename, source.readonly)
    def close(self): self._source.close()
    @property
    def header(self): return self._source.header_info
    def _get_props(self): self._set_props(self._source.dtype, self._source.shape)
    def _get_data(self): return self._source.data
    def _set_data(self, im):
        self._source.set_data(im)
        self._set_props(self._source.dtype, self._source.shape)
    def _set_filename(self, filename):
        self._source.rename(self._rename, filename)

########## Image Stack ##########
# TODO: support local for GIF-stack?
class _PILStack(_PILSource):
    """Class handling general stacks"""
    _z = 0
    _single_slice = False
    def __init__(self, frmt, open, accept, save=None):
        super(_PILStack, self).__init__(frmt, open, accept, None)
    def open(self, file, readonly, slice=None, **options):
        s = super(_PILStack, self).open(file, readonly, **options)
        if not s.is_stack: raise ValueError("File is not a stack")
        if slice is not None:
            s.seek(slice)
            s._single_slice = True
        return s
        
    @property
    def is_stack(self): return True
    @property
    def header_stack_info(self): return self._get_hdr()
    @property
    def header_info(self): return self._get_hdr() if self._single_slice else {}
    def _get_hdr(self):
        """Get the headers for the current slice"""
        h = {'format':self.im.format}
        h.update(self.im.info)
        return h
    def seek(self, z):
        if self._z == z: return
        if self._z > z:
            # we need to go all the way to the end and then reset to the beginning
            #try:
            #    while True:
            #        self._z += 1
            #        self.im.seek(self._z)
            #except (EOFError, ValueError): pass
            self._z = 0
            self.im.seek(0)
        while self._z != z:
            try:
                self._z += 1
                self.im.seek(self._z)
            except (EOFError, ValueError):
                self._z -= 1
                raise ValueError('Slice index out of range')
    def _for_each_slice(self, func):
        lst, start_z = [], self._z
        self.seek(0)
        while True:
            try: lst.append(func()); self.seek(self._z+1)
            except (EOFError, ValueError) as ex: break
        try: self.seek(start_z)
        except StandardError: pass
        return lst
    def slices(self, stack):
        # Default behavior for slices is to read all of them store them. This is pretty bad, but
        # nothing we can really do. The good thing is many formats have a better solution.
        return self._for_each_slice(lambda:_PILSlice(stack, self, self._z))
class _PILSlice(FileImageSlice):
    def __init__(self, stack, pil, z):
        super(_PILSlice, self).__init__(stack, z)
        self._set_props(pil.dtype, pil.shape)
        self._data = pil.data
        self._header = pil.header_info
    @property
    def header(self): return self._header
    def _get_props(self): pass
    def _get_data(self): return self._data
    def _set_data(self): raise RuntimeError() # this can never be called

class _PILStackWithSliceHeaders(_PILStack):
    """A PIL stack that has possibly different headers for each slice."""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def _get_all_hdrs(self):
        """Get a list of all headers for the entire stack."""
        pass
    
    __stk_hdr = None
    __slc_hdrs = None
    @property
    def header_stack_info(self):
        if self.__stk_hdr is None:
            self.__slc_hdrs = self._get_all_hdrs()
            self.__stk_hdr = self.__extract_stack_headers(self.__slc_hdrs)
        return self.__stk_hdr
    @property
    def header_info(self):
        if self._single_slice: return self._get_hdr()
        if self.__slc_hdrs is None:
            self.__slc_hdrs = self._get_all_hdrs()
            self.__stk_hdr = self.__extract_stack_headers(self.__slc_hdrs)
        return self.__slc_hdrs[self._z]
    @staticmethod
    def __extract_stack_headers(headers):
        """
        Extracts the "stack" headers from a list of "slice" headers according to the following:
          * any header itme that is identical across all slices is deemed part of the stack header
            and removed from each slice
          * any header key that only exists for the first slice is moved to the stack header (except
            if there is only one slice)
          * if there are no slices, the current headers are assumed to be for the entire stack.
        Returns the stack header (dict). The slice headers (list of dict) is modified in-place.
        """
        if len(headers) == 0: return self._get_hdr()
        
        from itertools import islice
        def _remove_items(d, keys):
            for k in keys: del d[k]
        def _move_items(d, dsrc, keys):
            for k in keys: d[k] = dsrc[k]; del dsrc[k]
        
        # Calculate the items that all slices have identical and update
        h_stack = set(headers[0].iteritems())
        for h in islice(headers, 1, None): h_stack &= h.viewitems()
        h_stack = dict(h_stack)
        h_stack_keys = h_stack.keys()
        for h in headers: _remove_items(h, h_stack_keys)

        # Calculate keys unique to the first slice and update
        if len(headers) > 1:
            h_stack_keys = set(headers[0])
            for h in islice(headers, 1, None): h_stack_keys -= h.viewkeys()
            _move_items(h_stack, headers[0], h_stack_keys)

        return h_stack

class _GIFStack(_PILStackWithSliceHeaders):
    def _get_all_hdrs(self): return self._for_each_slice(self._get_hdr)
    
class _RandomAccessPILStack(_PILStack):
    """
    A PIL stack for formats that allow random-access of slices. Each format exposes the total
    number of slices differently though, so there are subclasses for that.
    """
    __metaclass__ = ABCMeta
    @abstractproperty
    def depth(self): return 0
    def seek(self, z):
        if self._z == z: return
        if z >= self.depth: raise ValueError('Slice index out of range')
        try: self.im.seek(z)
        except (EOFError, ValueError): raise ValueError('Slice index out of range')
        self._z = z
    def slices(self, stack):
        return [_RandomAccessPILSlice(stack, self, z) for z in xrange(self.depth)]
class _RandomAccessPILSlice(FileImageSlice):
    def __init__(self, stack, pil, z):
        super(_RandomAccessPILSlice, self).__init__(stack, z)
        pil.seek(z)
        self._set_props(pil.dtype, pil.shape)
        self._pil = pil
    @property
    def header(self):
        self._pil.seek(self._z)
        return self._pil.header_info
    def _get_props(self): pass
    def _get_data(self):
        self._pil.seek(self._z)
        return self._pil.data
    def _set_data(self): raise RuntimeError() # this can never be called

class _RandomAccessPILStackWithSliceHeaders(_PILStackWithSliceHeaders, _RandomAccessPILStack):
    def _get_all_hdrs(self):
        d = self.depth
        if d == 0: return []
        headers = [None]*d
        self.im.seek(0)
        for z in xrange(d):
            self.im.seek(z)
            headers[z] = self._get_hdr()
        self.im.seek(self._z)
        return headers

class _IMStack(_RandomAccessPILStack):
    @property
    def depth(self): return self.im.info["File size (no of images)"]
class _SPIDERStack(_RandomAccessPILStackWithSliceHeaders):
    @property
    def depth(self): return self.im.nimages
    @property
    def is_stack(self): return self.im.istack != 0
class _DCXStack(_RandomAccessPILStackWithSliceHeaders):
    @property
    def depth(self): return len(self.im._offset)
class _MICStack(_RandomAccessPILStackWithSliceHeaders):
    @classmethod
    def depth(self): return len(self.im.images)
    @classmethod
    def is_stack(self): return self.im.category == Image.CONTAINER
class _TIFFStack(_RandomAccessPILStackWithSliceHeaders):
    # Not quite random-access because we don't know the depth until we have gone all the way
    # through once. Also, internally, it does use increment and reset but is fast since it can skip
    # all of the image data.
    __depth = None
    @property
    def depth(self):
        if self.__depth is None:
            z = 0
            while True:
                try: self.im.seek(z); z += 1
                except (EOFError, ValueError): break
            self.__depth = z
        return self.__depth
    def _get_hdr(self):
        h = super(_TIFFStack, self)._get_hdr()
        h.update(_get_tiff_tags(self.im))
        return h
class _PSDStack(_RandomAccessPILStack):
    # PIL reads the PSD image "oddly". The entire merged image is in frame=0. The others are the
    # layers in order. However, once you go to any other frame you can never return 0. So what we
    # are going to do is say if the image is opened as a 2D image then we will so the entire merged
    # image. If we open it as a stack we will only go for the other layers (and not the merged image
    # ever). Another problem is that PIL doesn't seem to always be able to extract the layers, but I
    # might be using a too-new version of Photoshop...
    @classmethod
    def depth(self): return len(self.im.layers)
    def seek(self, idx):
        if idx >= self.depth: raise ValueError('Slice index out of range')
        try: self.im.seek(idx+1) # +1 for skipping the entire merged image
        except (EOFError, ValueError): raise ValueError('Slice index out of range')

_stacks = None
def _init_stack():
    from types import ClassType
    global _stacks
    if _mode2dtype is None: _init_pil()
    _stack_classes = {
        'GIF' : _GIFStack,
        'IM'  : _IMStack,
        'SPIDER': _SPIDERStack,
        'TIFF': _TIFFStack,
        'DCX' : _DCXStack,
        'MIC' : _MICStack,
        'PSD' : _PSDStack,
    }
    _stacks = {
        frmt:_stack_classes.get(frmt,_PILStack)(frmt,clazz,accept)
        for frmt,(clazz,accept) in Image.OPEN.iteritems()
        if isinstance(clazz,(type,ClassType)) and clazz.seek != Image.Image.seek
        }

class PILStack(FileImageStack):
    @classmethod
    def open(cls, f, readonly=False, format=None, **options):
        if _stacks is None: _init_stack()
        return PILStack(_open_source(_stacks, format, f, readonly, **options))
    
    @classmethod
    def _openable(cls, filename, f, readonly, format=None, **options):
        if _stacks is None: _init_stack()
        return _openable_source(_stacks, format, f, filename, readonly, **options)

    # TODO: support writing
    # Need to add create/creatable and many other things (start by looking for raise RuntimeError()).
    # Possibly save-able:
    #    IM:     supports "frames" param but I don't see how it actually saves multiple frames
    #    GIF:    see gifmaker.py in the Pillow module
    #    SPIDER: maybe
    #    TIFF:   maybe
    @classmethod
    def _can_write(cls): return False

    @classmethod
    def name(cls): return "PIL-Stack"
    @classmethod
    def print_help(cls, width):
        from ....imstack import Help
        p = Help(width)
        p.title("Python Imaging Library (PIL) Image Stack")
        p.text("""
PIL is a common library for reading various image formats in Python. Technically we use the PILLOW
fork of PIL which is the standard replacement for PIL. This requires PILLOW v2.0 or newer.

PIL is a common library for reading various image formats in Python. Some of those formats support
several image slices in a single file, including TIFF, IM, DCX, and GIF. The PIL formats that
support several image slices can be loaded as a stack. Several of these formats have limitations
such as being only able to read sequentially and may incure higher overheads when not being read
in the manner they were intended to be.

Currently there is no support for writing a PIL-supported image stack format.

This supports the option 'format' to force one of the supported formats listed below. Additional
options are supported as per the non-stack variants. See the PIL topic.

Supported image formats:""")
        p.list(*sorted(cls.formats()))
        p.newline()
        p.text("See also:")
        p.list('PIL')
        
    @classmethod
    def formats(cls):
        if _stacks is None: _init_stack()
        return _stacks.keys()

    def __init__(self, stack):
        self._stack = stack
        super(PILStack, self).__init__(PILHeader(stack), stack.slices(self), True)
    def close(self): self._stack.close()
    def _delete(self, idxs): raise RuntimeError() # not possible since it is always read-only
    def _insert(self, idx, ims): raise RuntimeError()
        
class PILHeader(FileImageStackHeader):
    _fields = None
    def __init__(self, stack, **options):
        data = stack.header_stack_info
        #data['options'] = options
        self._fields = {k:FixedField(lambda x:x,v,False) for k,v in data.iteritems()}
        super(PILHeader, self).__init__(data)
    def save(self):
        if self._imstack._readonly: raise AttributeError('header is readonly')
    def _update_depth(self, d): raise RuntimeError() # not possible since it is always read-only
    def _get_field_name(self, f):
        return f if f in self._fields else None
