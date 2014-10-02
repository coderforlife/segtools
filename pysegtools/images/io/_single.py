"""Base IO functions."""

from ..types import *

__all__ = ['iminfo','imread','imsave']

# Not supported: CMYK, YCbCr, LAB, LA
# Raw modes that need not be supported? (handled internally within PIL?)
#   '1:I':IM_BIT,'1:R':IM_BIT,
#   'L;I':IM_BYTE,
#   'BGR':IM_RGB24*
#   'BGR;15':(unsupported)
#   'BGR;24':IM_RGB24*
#   'RGB;L':IM_RGB24
#   'I;*' are also 'F;*'?
pil_mode_to_dtype = {
    #'P' is a special case
    
    'RGB':IM_RGB24,'RGBX':IM_RGB24,
    'RGBA':IM_RGBA32,'RGBa':IM_RGBA32,

    '1':IM_BIT,'L':IM_UINT8,'I':IM_INT32_NATIVE,
    'I;8':IM_UINT8,'I;8S':IM_INT8,
    'I;16':IM_UINT16,'I;16L':IM_UINT16,'I;16B':IM_UINT16_BE,'I;16N':IM_UINT16_NATIVE,
    'I;16S':IM_INT16,'I;16LS':IM_INT16,'I;16BS':IM_INT16_BE,'I;16NS':IM_INT16_NATIVE,
    'I;32':IM_UINT32,'I;32L':IM_UINT32,'I;32B':IM_UINT32_BE,'I;32N':IM_UINT32_NATIVE,
    'I;32S':IM_INT32,'I;32LS':IM_INT32,'I;32BS':IM_INT32_BE,'I;32NS':IM_INT32_NATIVE,
    
    'F':IM_FLOAT32_NATIVE,
    'F;32F':IM_FLOAT32,'F;32BF':IM_FLOAT32_BE,'F;32NF':IM_FLOAT32_NATIVE,
    'F;64F':IM_FLOAT64,'F;64BF':IM_FLOAT64_BE,'F;64NF':IM_FLOAT64_NATIVE,
    }

def iminfo(filename):
    """
    Read the basic image information from a file. By default this uses SciPy and PIL for any formats
    they support. This attempts to only read the header and not the entire image. Returns the shape
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
        iminfo.formats[ext](filename)
    else:
        from PIL import Image
        im = Image.open(filename)
        return im.size, pil_mode_to_dtype[im.palette.mode if im.mode == 'P' else im.mode]
def _iminfo_register(ext, info):
    """
    Register a file extension to use a particular info-gathering function. The function needs to
    take one argument: the filename. Setting a new value will overwrite any previous association.
    Setting an extension to None will remove it.
    """
    if ext[0] != '.': ext = '.' + ext
    if info == None:
        try: del iminfo.formats[ext.lower()]
        except KeyError: pass
    elif not callable(info): raise TypeError
    else: iminfo.formats[ext.lower()] = info
iminfo.formats = {}
iminfo.register = _iminfo_register

def imread(filename):
    """
    Read an image from a file. By default this uses SciPy and PIL for any formats they support.

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
        imread.formats[ext](filename)
    else:
        # TODO
        from scipy.misc import imread as sp_imread
        im = sp_imread(filename)
        if im.dtype == IM_BIT:
            # Bugs in SciPy cause 1-bit images to be read corrupted
            from PIL import Image
            from numpy import asarray
            im = Image.open(filename)
            im = asarray(im.getdata(), dtype=IM_BIT).reshape(im.size) # TODO: does this need to be im.size[::-1]?
        return im
def _imread_register(ext, read):
    """
    Register a file extension to use a particular reading function. The reading function needs to
    take one argument: the filename to read. Setting a new value will overwrite any previous
    association. Setting an extension to None will remove it.
    """
    if ext[0] != '.': ext = '.' + ext
    if read == None:
        try: del imread.formats[ext.lower()]
        except KeyError: pass
    elif not callable(read): raise TypeError
    else: imread.formats[ext.lower()] = read
imread.formats = {}
imread.register = _imread_register

def imsave(filename, im):
    """
    Save an image to a file. By default this uses SciPy and PIL for any formats they support.

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
    ext = splitext(filename)[1].lower()
    im = im_standardize_dtype(im)
    if ext in imsave.formats:
        imsave.formats[ext](filename, im)
    elif im.dtype == IM_BIT:
        # TODO
        # Make sure data is actually saved as 1-bit data
        from PIL import Image
        im = im * uint8(255)
        Image.frombuffer('L', im.shape, im.data, 'raw', 'L', 0, 1).convert('1').save(filename)
    else:
        # TODO
        from scipy.misc import imsave as sp_imsave
        sp_imsave(filename, im_raw_dtype(im))
def _imsave_register(ext, save):
    """
    Register a file extension to use a particular saving function. The saving function needs to
    take two arguments: the filename to save to and the image to save. Setting a new value will
    overwrite any previous association. Setting an extension to None will remove it.
    """
    if ext[0] != '.': ext = '.' + ext
    if save == None:
        try: del imsave.formats[ext.lower()]
        except KeyError: pass
    elif not callable(save): raise TypeError
    else: imsave.formats[ext.lower()] = save
imsave.formats = {}
imsave.register = _imsave_register

# Import additional formats
import formats
