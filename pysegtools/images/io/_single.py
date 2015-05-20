"""IO functions for reading, writing and querying 2D image formats, or 'single slices'."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numpy import sctypes, dtype
from numpy import bool_, uint8,uint16,uint32, int8,int16,int32, float32,float64 #float16
from sys import byteorder
from ..types import create_im_dtype as d, check_image, get_im_dtype_and_nchan

__all__ = ['iminfo','imread','imsave']

__native = byteorder!='little'
# TODO: could use len(PIL.ImageMode.getmode(mode).bands) and PIL.ImageMode.getmode(mode).basetype to auto-generate these conversions
__pil_mode_to_dtype = {
    #'P' is a special case
    # Some of these modes will actually never show up because they are raw modes.

    'RGB': d(uint8,False,3), 'RGBX':d(uint8,False,4), # however the fourth one is "padding"
    'RGBA':d(uint8,False,4), 'RGBa':d(uint8,False,4), # non-premultiplied and pre-multiplied
    'CMYK':d(uint8,False,4), 'YCbCr':d(uint8,False,3),
    'LAB': d(uint8,False,3), 'HSV':d(uint8,False,3),
    'LA':  d(uint8,False,2), # grayscale with alpha

    '1':d(bool_),'L':d(uint8),'I':d(int32,__native),
    'I;8':d(uint8),'I;8S':d(int8),
    'I;16':d(uint16),'I;16L':d(uint16),'I;16B':d(uint16,True),'I;16N':d(uint16,__native),
    'I;16S':d(int16),'I;16LS':d(int16),'I;16BS':d(int16,True),'I;16NS':d(int16,__native),
    'I;32':d(uint32),'I;32L':d(uint32),'I;32B':d(uint32,True),'I;32N':d(uint32,__native),
    'I;32S':d(int32),'I;32LS':d(int32),'I;32BS':d(int32,True),'I;32NS':d(int32,__native),

    'F':d(float32,__native),
    #'F;16F':d(float16),'F;16BF':dt(float16,True),'F;16NF':dt(float16,__native),
    'F;32F':d(float32),'F;32BF':d(float32,True),'F;32NF':d(float32,__native),
    'F;64F':d(float64),'F;64BF':d(float64,True),'F;64NF':d(float64,__native),
}
del d
__dtype_to_pil_mode = {
    # mode, rawmode (little endian), rawmode (big endian)
    # Multi-channel and bit images are special cases
    #float16:('F','F;16F','F;16BF'), # F;16 can only come from integers...
}
# Build __dtype_to_pil_mode
for t in sctypes['uint']:
    nb = dtype(t).itemsize
    if   nb == 1: __dtype_to_pil_mode[t] = ('L','L','L')
    elif nb == 2: __dtype_to_pil_mode[t] = ('I;16','I;16','I;16B')
    elif nb == 4: __dtype_to_pil_mode[t] = ('I','I;32','I;32B')
    else: nb = str(nb*8); __dtype_to_pil_mode[t] = ('I','I;'+nb,'I;'+nb+'B')
for t in sctypes['int']:
    nb = dtype(t).itemsize
    if nb == 1: __dtype_to_pil_mode[t] = ('I','I;8S','I;8S')
    else: nb = str(nb*8); __dtype_to_pil_mode[t] = ('I','I;'+nb+'S','I;'+nb+'BS')
for t in sctypes['float']:
    nb = dtype(t).itemsize
    if nb < 4: continue
    nb = str(nb*8); __dtype_to_pil_mode[t] = ('F','F;'+nb+'F','F;'+nb+'BF')

def iminfo(filename):
    """
    Read the basic image information from a file. By default this uses PIL for any formats it
    supports. This attempts to only read the header and not the entire image. Returns the shape
    (H, W) and the dtype.

    Additional formats can be registered by using iminfo.register(...). Python scripts placed in
    images/io/formats will automatically be loaded. A dictionary of additional formats is
    available at iminfo.formats.

    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB [not ZIP compressed]
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?

    See http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html for more details.
    """
    from os.path import splitext
    ext = splitext(filename)[1].lower()
    if ext in iminfo.formats:
        return iminfo.formats[ext](filename)
    else:
        from PIL import Image
        im = Image.open(filename)
        return tuple(reversed(im.size)), __pil_mode_to_dtype[im.palette.mode if im.mode=='P' else im.mode]
def __iminfo_register(ext, info):
    """
    Register a file extension to use a particular info-gathering function. The function needs to
    take one argument: the filename. Setting a new value will overwrite any previous association.
    Setting an extension to None will remove it.
    """
    if ext[0] != '.': ext = '.' + ext
    if info is None:
        try: del iminfo.formats[ext.lower()]
        except KeyError: pass
    elif not callable(info): raise TypeError
    else: iminfo.formats[ext.lower()] = info
iminfo.formats = {}
iminfo.register = __iminfo_register

def imread(filename):
    """
    Read an image from a file. By default this uses PIL for any formats it supports.

    Additional formats can be registered by using imread.register(...). Python scripts placed in
    images/io/formats will automatically be loaded. A dictionary of additional formats is
    available at imread.formats.

    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB [not ZIP compressed]
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?

    See http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html for more details.
    """
    from os.path import splitext
    ext = splitext(filename)[1].lower()
    if ext in imread.formats:
        return imread.formats[ext](filename)
    else:
        from PIL import Image
        from numpy import array
        im = Image.open(filename)
        dt = __pil_mode_to_dtype[im.palette.mode if im.mode=='P' else im.mode]
        a = array(im.getdata(), dtype=dt).reshape(tuple(reversed(dt.shape+im.size)))
        return a
def __imread_register(ext, read):
    """
    Register a file extension to use a particular reading function. The reading function needs to
    take one argument: the filename to read. Setting a new value will overwrite any previous
    association. Setting an extension to None will remove it.
    """
    if ext[0] != '.': ext = '.' + ext
    if read is None:
        try: del imread.formats[ext.lower()]
        except KeyError: pass
    elif not callable(read): raise TypeError
    else: imread.formats[ext.lower()] = read
imread.formats = {}
imread.register = __imread_register

def imsave(filename, im):
    """
    Save an image to a file. By default this uses PIL for any formats it supports.

    Additional formats can be registered by using imsave.register(...). Python scripts placed in
    images/io/formats will automatically be loaded. A dictionary of additional formats is
    available at imsave.formats.

    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?

    See http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html for more details.
    """
    from os.path import splitext
    check_image(im)
    ext = splitext(filename)[1].lower()
    if ext in imsave.formats:
        imsave.formats[ext](filename, im)
    else:
        from PIL import Image
        st, sh = im.strides[0], im.shape[1::-1]
        dt, nchan = get_im_dtype_and_nchan(im)
        if nchan > 1:
            if dt.type != uint8 or nchan > 4: raise ValueError
            mode = ('LA','RGB','RGBA')[nchan-2]
            im = Image.frombuffer(mode, sh, im.data, 'raw', mode, st, 1)
        elif dt.kind == 'b':
            # Make sure data is actually saved as 1-bit data (both SciPy and PIL seem to be broken with this)
            im = im * uint8(255)
            im = Image.frombuffer('L', sh, im.data, 'raw', 'L', st, 1).convert('1')
        else:
            mode = __dtype_to_pil_mode.get(dt.type)
            if mode is None: raise ValueError
            im = Image.frombuffer(mode[0], sh, im.data, 'raw', mode[2 if __native else 1], st, 1)
        im.save(filename)
def _imsave_register(ext, save):
    """
    Register a file extension to use a particular saving function. The saving function needs to
    take two arguments: the filename to save to and the image to save. Setting a new value will
    overwrite any previous association. Setting an extension to None will remove it.
    """
    if ext[0] != '.': ext = '.' + ext
    if save is None:
        try: del imsave.formats[ext.lower()]
        except KeyError: pass
    elif not callable(save): raise TypeError
    else: imsave.formats[ext.lower()] = save
imsave.formats = {}
imsave.register = _imsave_register

# Import additional formats
from . import formats # pylint: disable=unused-import
