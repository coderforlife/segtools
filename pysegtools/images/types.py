from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numbers, re
from numpy import dtype, iinfo, sctypes, promote_types, bool_, ndarray, ascontiguousarray
from ..general import String, sys_endian
from ..imstack import Help

__all__ = [
    'create_im_dtype','get_im_dtype','get_im_dtype_and_nchan','im_dtype_desc','get_dtype_endian',
    'is_image','is_image_desc','check_image','is_single_channel_image','check_image_single_channel',
    'im_rgb_view','im_raw_view','im_complexify','im_decomplexify','im_decomplexify_dtype','im_complexify_dtype',
    'get_im_min_max','get_dtype_min_max','get_dtype_min','get_dtype_max',
    ]

__bit_types = [bool_,bool] # bool8?
__int_types = sctypes['int']+[int,long]
__uint_types = sctypes['uint']
__float_types = sctypes['float']+[float]
__cmplx_types = sctypes['complex']+[complex]
__basic_types = __bit_types+__int_types+__uint_types+__float_types

##### Make numpy types part of the Python ABC heirarchy #####
# Note: in newer version of NumPy this is set by default, so we check
if not issubclass(__int_types[0], numbers.Integral):
    for __i in __int_types:   numbers.Integral.register(__i)
    for __u in __uint_types:  numbers.Integral.register(__u)
    for __f in __float_types: numbers.Real.register(__f)
    for __c in __cmplx_types: numbers.Complex.register(__c)


##### dtype functions #####
__re_im_dtype = re.compile(r'^(([0-9]+\*)?([UIFC])|RGBA?)([0-9]+)(-BE)?$', re.IGNORECASE)
def create_im_dtype(base, big_endian=None, channels=1):
    """
    Creates an image data type given a base type, endian-ness, and number of channels. When creating
    or loading an array, the number of channels will automatically become part of the array shape
    and the array will take on the base data type (so its dtype will lose the channels information).
    Big endian can be specified as True/False or <, > or =. If not given it defaults to the endian
    given in the base dtype. Additionally, this accepts all strings that im_dtype_desc can produce
    in which case the last two arguments are ignored.
    """
    if isinstance(base, String):
        result = __re_im_dtype.search(base)
        if result is None: raise ValueError('Image type not known')
        grps = result.groups()
        bits = int(grps[3])
        if bits == 1 and grps[2] not in 'uU': raise ValueError('Only unsigned integer can have 1 bit')
        if bits != 1 and (bits%8) != 0: raise ValueError('Invalid number of bits, must be 1 or a multiple of 8')
        nbytes = bits//8 # now 0 if originally 1
        big_endian = grps[4] is not None
        if grps[0][0] in 'rR':
            channels = 4 if grps[0][-1] in 'aA' else 3
            if nbytes == 0 or (nbytes%channels) != 0: raise ValueError('Invalid number of bits for RGB/RGBA image')
            base = dtype('u'+str(nbytes//channels))
        else:
            channels = 1 if grps[1] is None else int(grps[1][:-1])
            if channels == 0 or channels > 4: raise ValueError('Invalid number of channels')
            base = bool if nbytes == 0 else dtype(grps[2].lower()+str(nbytes))
    if isinstance(base, dtype):
        if big_endian is None: big_endian = get_dtype_endian(base)
        base = base.type
    elif big_endian is None: big_endian = sys_endian # base is a type without endian information
    if base in __cmplx_types:
        if channels != 1: raise ValueError('Complex types must use 1 channel')
    elif base not in __basic_types: raise ValueError('Image base type not known')
    endian = big_endian if isinstance(big_endian, String) else ('>' if big_endian else '<')
    return dtype((base, channels)).newbyteorder(endian)
def get_im_dtype(im_or_dtype):
    """
    Gets the dtype of an image including the number of channels. This also accepts dtypes (with the
    assumption that missing number of channels means 1 channel) and a tuple of dtype and number of
    channels.
    """
    if isinstance(im_or_dtype, ndarray):
        return dtype((im_or_dtype.dtype, im_or_dtype.shape[2] if im_or_dtype.ndim == 3 else 1))
    elif isinstance(im_or_dtype, dtype):
        return im_or_dtype
    else:
        return dtype(tuple(im_or_dtype[:2]) if len(im_or_dtype) > 1 else im_or_dtype[0])
def get_im_dtype_and_nchan(im_or_dtype):
    """
    Gets the base dtype of an image and the number of channels. This also accepts dtypes (with the
    assumption that missing number of channels means 1 channel) and a tuple of dtype and number of
    channels.
    """
    if isinstance(im_or_dtype, ndarray):
        return im_or_dtype.dtype, im_or_dtype.shape[2] if im_or_dtype.ndim == 3 else 1
    elif isinstance(im_or_dtype, dtype):
        return im_or_dtype.base, im_or_dtype.shape[0] if len(im_or_dtype.shape) else 1
    else:
        return im_or_dtype[0], int(im_or_dtype[1]) if len(im_or_dtype) > 1 else 1
def im_dtype_desc(im_or_dtype):
    """
    Get very brief string description of the data type of an image. The format is:
        [N*]T#[-BE]
    where N is the number of channels (only included if >1), T is the data type (U for unsigned
    integer, I for signed integer, F for floating-point number, and C for complex), # is the number
    of bits, and -BE is included if the numbers are stored in big-endian format.

    If it is a 3 or 4 channel unsigned integer image then RGB[A]#[-BE] will be returned instead.

    This accepts an image, just a dtype (in which case the number of channels is assumed to be 1 if
    the dtype does not include the number of channels), or a sequence of dtype and number of
    channels.
    """
    dt, nchan = get_im_dtype_and_nchan(im_or_dtype)
    kind, bits = dt.kind, dt.itemsize*8
    be = '-BE' if get_dtype_endian(dtype)=='>' else ''
    if nchan in (3,4) and kind == 'u': return ('RGBA%d%s' if nchan == 4 else 'RGB%d%s')%(bits*nchan, be)
    if   kind == 'b': base = 'U1'
    elif kind == 'u': base = 'U%d%s' % (bits, be)
    elif kind == 'i': base = 'I%d%s' % (bits, be)
    elif kind == 'f': base = 'F%d%s' % (bits, be)
    elif kind == 'c': base = 'C%d%s' % (bits, be)
    else: base = str(dt)
    return ('%d*%s'%(nchan,base)) if nchan > 1 else base
def get_dtype_endian(dt):
    """Get a '<' or '>' from a dtype's byteorder (which can be |, =, <, or >)."""
    endian = dtype(dt).byteorder
    if endian == '|': return '<' # | means N/A (single byte), report as little-endian
    elif endian == '=': return sys_endian # is native byte-order
    return endian


##### Image Verification #####
def is_image(im):
    """
    Returns True if `im` is an image, basically it is a ndarray of 2 or 3 dimensions where the 3rd
    dimension length is 1-5 and the data type is a basic data type (integer or float, or complex
    for 2d images). Does not check to see that the image has no zero-length dimensions.
    """
    ndim = 2 if im.ndim == 3 and im.shape[2] == 1 else im.ndim
    return (ndim == 2 and im.dtype.type in __basic_types+__cmplx_types or
            ndim == 3 and 2 <= im.shape[2] <= 5 and im.dtype.type in __basic_types)
def is_image_desc(dt, shape):
    """
    Returns True if the dtype and shape represent an image. Same as is_image but just checks the
    properties instead of needing an array.
    """
    shape = shape + dt.shape
    dt = dt.base.type
    ndim = 2 if len(shape) == 3 and shape[2] == 1 else len(shape)
    return (ndim == 2 and dt in __basic_types+__cmplx_types or
            ndim == 3 and 2 <= shape[2] <= 5 and dt in __basic_types)
def check_image(im):
    """
    Similar to is_image except instead of returning True/False it throws an exception if it isn't
    an image.
    """
    ndim = 2 if im.ndim == 3 and im.shape[2] == 1 else im.ndim
    if not (ndim == 2 and im.dtype.type in __basic_types+__cmplx_types or
            ndim == 3 and 2 <= im.shape[2] <= 5 and im.dtype.type in __basic_types):
        raise ValueError('Unknown image format')
def is_single_channel_image(im):
    """
    Returns True if `im` is a single-channel image, basically it is a ndarray of 2 or 3 dimensions
    where the 3rd dimension length is can only be 1 and the data type is a basic data type (integer
    or float but not complex). Does not check to see that the image has no zero-length dimensions.
    """
    if im.ndim == 3 and im.shape[2] == 1: im = im.squeeze(2)
    return im.ndim == 2 and im.dtype.type in __basic_types
def check_image_single_channel(im):
    """
    Similar to is_single_channel_image except instead of returning True/False it throws an exception
    if it isn't an image. Also, it returns a 2D image (wityh the 3rd dimension, if 1, removed).
    """
    if im.ndim == 3 and im.shape[2] == 1: im = im.squeeze(2)
    if im.ndim != 2 and im.dtype.type not in __basic_types:
        raise ValueError('Not single-channel image format')
    return im


##### View images in different ways #####
def __is_rgb_view(im):
    if im.ndim != 2: return False
    try:
        d = im.dtype
        if any(d[0]!=d[i] for i in xrange(1,len(d))) or ''.join(d.names) not in ('RGB','RGBA','BGR','BGRA'): return False
    except StandardError: return False
    return True
def im_rgb_view(im, bgr=False):
    """
    View an image as an RGB(A) image, returning a view of the image using a flexible data type with
    the fields "R", "B", "G", and possibly "A". Only works with non-complex images with 3 or 4
    channels. If the image is already an RGB view then it is returned as-is. Note that an RGB(A)
    viewed image will not be accepted by most functions and is_image returns False for it.
    """
    if __is_rgb_view(im): return im
    shp = im.shape
    if len(shp) != 3 or shp[2] not in (3,4) or im.dtype.type not in __basic_types: raise ValueError('Not an RGB or RGBA raw image')
    return ascontiguousarray(im).view(dtype=dtype([(C,im.dtype) for C in ('BGRA' if bgr else 'RGBA')[:shp[2]]])).squeeze(2)
def im_raw_view(im):
    """
    Reverse of im_rgb_view. If the image is not an RGB view it is returned as-is. Otherwise it is
    viewed as a 3D ndarray with a third dimension length of 3 or 4.
    """
    if not __is_rgb_view(im): return im
    return ascontiguousarray(im).view(dtype=dtype((im.dtype[0],len(im.dtype))))

__cmplx2float = {__c:dtype('f'+str(dtype(__c).itemsize//2)) for __c in __cmplx_types}
__cmplx2float = {__c:__f.type for __c,__f in __cmplx2float.iteritems() if __f.kind == 'f'}
__float2cmplx = {__f:dtype('c'+str(dtype(__f).itemsize* 2)) for __f in __float_types}
__float2cmplx = {__f:__c.type for __f,__c in __float2cmplx.iteritems() if __c.kind == 'c'}
def im_complexify(im, force=False):
    """
    View an image as complex numbers. The image must be a 2-channel image of one of the supported
    float-point types, otherwise an exception will occur. If the image is already complex, it is
    returned as-is.

    If force is True, then greater effort is taken to make the image a complex image. It still needs
    to be a 2-channel image but the channels can be floats or integers that aren't normally
    supported. There are some differences with the returned data as well:
     * The returned image is a copy, not a view
     * If the input image was integers, the output is scaled so that the values are 0.0 to 1.0
     * im_decomplexify will not restore the image to the same type it came from
     * There is a chance for a loss of precision, e.g. the largest complex type cannot fit two of
       the largest integral types
    """
    shp = im.shape
    if (len(shp) == 3 and shp[2] == 1 or len(shp) == 2) and im.dtype.type in __cmplx_types: return im
    good_shape = len(shp) == 3 and shp[2] == 2
    if not good_shape or im.dtype.type not in __float2cmplx:
        if not force or not good_shape: raise ValueError('Image cannot be represented as a complex type')
        o_dt = im.dtype
        dt = min((promote_types(o_dt, f) for f in __float2cmplx), key=lambda dt:dt.itemsize)
        im = im.astype(dt)
        if   o_dt.kind == 'u': im /= get_dtype_max(o_dt)
        elif o_dt.kind == 'i':
            mn, mx = get_dtype_min_max(o_dt)
            im += mn
            im /= mx
    return ascontiguousarray(im).view(dtype=dtype(__float2cmplx[im.dtype.type]).newbyteorder(im.dtype.byteorder)).squeeze(2)
def im_complexify_dtype(dt, force=False):
    """Same as im_complexify but with just the dtype (which is assumed to have 2 channels)."""
    dt = dt.base # removes channel information, if there
    if dt.type in __cmplx_types: return dt
    if dt.type not in __float2cmplx:
        if not force: raise ValueError('Image cannot be represented as a complex type')
        dt = min((promote_types(dt, f) for f in __float2cmplx), key=lambda dt:dt.itemsize)
    return dtype(__float2cmplx[dt.type]).newbyteorder(dt.byteorder)
    
def im_decomplexify(im):
    """
    Reverse of im_complexify. If the image is a complex image then it will be turned into a
    2-channel floating-point image. Non-complex images are returned as a view with the same dtype.
    """
    return im.view(dtype=im_decomplexify_dtype(im.dtype))
def im_decomplexify_dtype(dt):
    """Same as im_decomplexify but with just the dtype."""
    if dt.type not in __cmplx2float: return dt
    return dtype((__cmplx2float[dt.type],2)).newbyteorder(dt.byteorder)


##### Min/Max for data types #####
__min_max_values = { __t:(iinfo(__t).min,iinfo(__t).max) for __t in __int_types+__uint_types }
for __t in __bit_types:   __min_max_values[__t] = (__t(False),__t(True))
for __t in __float_types: __min_max_values[__t] = (__t('0.0'),__t('1.0'))
def get_dtype_min_max(dt): return __min_max_values[dtype(dt).type]
def get_dtype_min(dt): return __min_max_values[dtype(dt).type][0]
def get_dtype_max(dt): return __min_max_values[dtype(dt).type][1]
def get_im_min_max(im):
    """Gets the min and max values for an image or an image dtype."""
    if not isinstance(im, ndarray): dt, im = dtype(im), None
    else: dt = im.dtype
    if im is None or dt.kind != 'f': return __min_max_values[dt.type]
    mn, mx = im.min(), im.max()
    return (mn, mx) if mn < 0.0 or mx > 1.0 else __min_max_values[dt.type]


##### Command-line help #####
def __dtype_help(width):
    p = Help(width)
    p.title("Data Types")
    p.text("""
When printing out information about a stack or when converting image types, a 'data-type' is given.
Below are the various supported image format data types.

General format string: 
    [N*]T#[-BE] 
where N is the number of channels (only included if >1), T is the data type ('U' for unsigned
integer, 'I' for signed integer, 'F' for floating-point number, and 'C' for complex), # is the
number of bits (a multiple of 8), and -BE is included if the numbers are stored in big-endian
format instead of little-endian. Not all combinations are possible (for example complex images
cannot have any channels).

Additionally, for unsigned integer images with 3 or 4 channels, RGB or RGBA is supported for the
type (and the number of channels is left off). In this case, the number of bits must be a multiple
of 24 or 32 (for RGB and RGBA respectively).

Logical/boolean images/masks are given as U1 and is the only time 1 is allowed for the number of
bits.

The following data types are known by this installation (not including numbers of channels and
endian varieties):
""")
    for types in (__bit_types, __uint_types, __int_types, __float_types, __cmplx_types):
        p.list(*[im_dtype_desc(dt) for dt in sorted(
                set(dtype(t).newbyteorder('<') for t in types),
                key=lambda dt:dt.itemsize)])
    for chans in (3,4):
        p.list(*[im_dtype_desc(dt) for dt in sorted(
                set(dtype((t, chans)).newbyteorder('<') for t in __uint_types),
                key=lambda dt:dt.itemsize)])

    p.newline()
    p.text("See also:")
    p.list("colors")
Help.register(('data-type', 'data-types', 'types'), __dtype_help)
