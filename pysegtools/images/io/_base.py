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
    Read the basic image information using SciPy (actually PIL) for any formats it supports. This
    attempts to only read the header and not the entire image. Returns the shape (H, W) and the
    dtype.

    Additionally, the following extra formats are supported (extension must be right):
        MHA/MHD:    8-bit gray, 16-bit gray, 32-bit gray, 64-bit gray, float, double, 24-bit RGB
        MAT:        all formats, may not be image-like (requires h5py module for newer MAT files)
    Note: both only get the first "image" or data from the file.
    
    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB [not ZIP compressed]
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?
        
    See http://www.pythonware.com/library/pil/handbook/formats.htm for more details
    MHA/MHD code is implemented in the metafile module. MAT code is implemented in this module.
    """
    from os.path import splitext
    ext = splitext(filename)[1].lower()
    if ext == '.mat':
        from matlab import iminfo_mat
        return iminfo_mat(filename)
    elif ext == '.mha':
        from metafile import iminfo_mha
        return iminfo_mha(filename)
    elif ext == '.mhd':
        from metafile import iminfo_mhd
        return iminfo_mhd(filename)
    else:
        from PIL import Image
        im = Image.open(filename)
        return im.size, pil_mode_to_dtype[im.palette.mode if im.mode == 'P' else im.mode]

def imread(filename):
    """
    Read an image using SciPy (actually PIL) for any formats it supports.

    Additionally, the following extra formats are supported (extension must be right):
        MHA/MHD:    8-bit gray, 16-bit gray, 32-bit gray, 64-bit gray, float, double, 24-bit RGB
        MAT:        all formats, may not be image-like (requires h5py module for newer MAT files)
    Note: both only get the first "image" or data from the file.
    
    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB [not ZIP compressed]
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?
        
    See http://www.pythonware.com/library/pil/handbook/formats.htm for more details
    MHA/MHD code is implemented in the metafile module. MAT code is implemented in this module.
    """
    from os.path import splitext
    ext = splitext(filename)[1].lower()
    if ext == '.mat':
        from matlab import imread_mat
        return imread_mat(filename)
    elif ext == '.mha':
        from metafile import imread_mha
        return imread_mha(filename)[1]
    elif ext == '.mhd':
        from metafile import imread_mhd
        return imread_mhd(filename)[1]
    else:
        # TODO
        from scipy.misc import imread
        im = imread(filename)
        if im.dtype == IM_BIT:
            # Bugs in SciPy cause 1-bit images to be read corrupted
            from PIL import Image
            from numpy import asarray
            im = Image.open(filename)
            im = asarray(im.getdata(), dtype=IM_BIT).reshape(im.size) # TODO: does this need to be im.size[::-1]?
        return im

def imsave(filename, im):
    """
    Save an image. It will use SciPy (actually PIL) for any formats it supports.

    Additionally, the following extra formats are supported (extension must be right):
        MHA/MHD:    8-bit gray, 16-bit gray, 32-bit gray, 64-bit gray, float, double, 24-bit RGB
    
    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?
    
    See thtp://www.pythonware.com/library/pil/handbook/formats.htm for more details
    MHA/MHD code is implemented in the metafile module.
    """
    from os.path import splitext
    ext = splitext(filename)[1].lower()
    im = im_standardize_dtype(im)
    if ext == '.mha':
        from metafile import imsave_mha
        imsave_mha(filename, im)
    elif ext == '.mhd':
        from metafile import imsave_mhd
        imsave_mhd(filename, im)
    elif im.dtype == IM_BIT:
        # TODO
        # Make sure data is actually saved as 1-bit data
        from PIL import Image
        im = im * uint8(255)
        Image.frombuffer('L', im.shape, im.data, 'raw', 'L', 0, 1).convert('1').save(filename)
    else:
        # TODO
        from scipy.misc import imsave
        imsave(filename, im_raw_dtype(im))
