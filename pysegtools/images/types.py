from numpy import dtype, iinfo, bool_, int8,uint8,int16,int32,int64, uint16,uint32,uint64, float32,float64, complex64,complex128
from sys import byteorder

__all__ = [
    'im_standardize_dtype','imstack_standardize_dtype','im_raw_dtype',
    'get_im_min_max','dtype2desc','get_dtype_endian',
    #'make_rgb24_color','make_rgba32_color',
    'IM_INT_TYPES','IM_FLOAT_TYPES','IM_COMPLEX_TYPES','IM_COLOR_TYPES','IM_COLOR_ALPHA_TYPES','IM_RANGED_TYPES','IM_ALL_TYPES',
    'IM_BIT','IM_INT8','IM_UINT8','IM_RGB24','IM_RGB24_RAW','IM_RGBA32','IM_RGBA32_RAW',
    ]

##### The image types we know about #####
IM_BIT        = dtype(bool_)
IM_INT8       = dtype(int8)
IM_UINT8      = dtype(uint8)
IM_RGB24      = dtype([('R',uint8),('G',uint8),('B',uint8)])
IM_RGB24_RAW  = dtype((uint8,3))
IM_RGBA32     = dtype([('R',uint8),('G',uint8),('B',uint8),('A',uint8)])
IM_RGBA32_RAW = dtype((uint8,4))
_base_types = [
    ( 'INT16',dtype( int16)),( 'INT32',dtype( int32)),( 'INT64',dtype( int64)),
    ('UINT16',dtype(uint16)),('UINT32',dtype(uint32)),('UINT64',dtype(uint64)),
    ('FLOAT32',dtype(float32)),('FLOAT64',dtype(float64)),
    ('COMPLEX64',dtype(complex64)),('COMPLEX128',dtype(complex128)),
    ('INT16_2',dtype([('real',int16),('imag',int16)])),('INT16_2_RAW',dtype((int16,2))),
    ]
_endians = [ ('', '<'), ('_BE', '>'), ('_NATIVE', '<' if byteorder == 'little' else '>')]
_globals = globals()
for k,d in _base_types:
    for end,endian in _endians:
        _globals['IM_'+k+end] = d.newbyteorder(endian)
        __all__.append('IM_'+k+end)
del _base_types, _endians, _globals

# Notes for the multi-channels types (RGB, RGBA, complex, ...):
#  * "RAW" types have d.shape == (channels,)
#  * non-"RAW" types have len(d) == channels
#  * otherwise len(d) is 0 and d.shape == ()

##### Standardize Image Types #####
_standardization = {
        2:{IM_INT16  :IM_INT16_2,   IM_INT16_BE  :IM_INT16_2_BE,
           IM_FLOAT32:IM_COMPLEX64, IM_FLOAT32_BE:IM_COMPLEX64_BE,
           IM_FLOAT64:IM_COMPLEX128,IM_FLOAT64_BE:IM_COMPLEX128_BE,},
        3:{IM_UINT8:IM_RGB24},
        4:{IM_UINT8:IM_RGBA32},
    }
def im_standardize_dtype(im):
    """
    Converts some image array formats to other formats to guarantee that the array is 2D.
    This is done with views and not copying.
    """
    if im.ndim == 3:
        target = _standardization.get(im.shape[2],{}).get(im.dtype,None)
        if target == None: raise ValueError('Image format not known')
        return im.view(dtype=target).squeeze(2)
    elif im.ndim != 2: raise ValueError('Image format not known')
    return im

def imstack_standardize_dtype(im):
    """
    Same as im_standardize_dtype but works on stacks of images.
    """
    if im.ndim == 4:
        target = _standardization.get(im.shape[3],{}).get(im.dtype,None)
        if target == None: raise ValueError('Image format not known')
        return im.view(dtype=target).squeeze(3)
    elif im.ndim != 3: raise ValueError('Image format not known')
    return im

def im_raw_dtype(im):
    """
    Reverse of im_standarize_dtype. Converts some image array formats to other formats to guarantee
    that the array dtype is a native (non-flexible) type. This is done with views and not copying.
    """
    dt = im.dtype
    if len(dt) > 0:
        if any(dt[0] != dt[i] for i in xrange(1, len(dt))): raise ValueError
        return im.view(dtype = dtype((dt[0],len(dt))))
    return im

##### Image dtype coercsion #####
##def _astype(im,t): return im.astype(dtype=t)
##def bit2rgb(im,t):
##    
##
##_coerce = {
##        IM_BIT:(IM_UINT8,IM_INT8,IM_UINT16,IM_UINT16_BE,IM_INT16,IM_INT16_BE,IM_UINT32,IM_UINT32_BE,IM_INT32,IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,
##                (IM_RGB24,x),(IM_RGBA32,x),
##                IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,
##                (IM_INT16_2,x),(IM_INT16_2_BE,x),)
##        IM_UINT8:(IM_UINT16,IM_UINT16_BE,IM_INT16,IM_INT16_BE,IM_UINT32,IM_UINT32_BE,IM_INT32,IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,
##                (IM_RGB24,x),(IM_RGBA32,x),
##                IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,
##                (IM_INT16_2,x),(IM_INT16_2_BE,x),)
##        IM_INT8:(IM_UINT16,IM_UINT16_BE,IM_INT16,IM_INT16_BE,IM_UINT32,IM_UINT32_BE,IM_INT32,IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,(IM_INT16_2,x),(IM_INT16_2_BE,x),)
##        IM_UINT16:(IM_UINT16_BE,IM_UINT32,IM_UINT32_BE,IM_INT32,IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_UINT16_BE:(IM_UINT16,IM_UINT32,IM_UINT32_BE,IM_INT32,IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_INT16:(IM_INT16_BE,IM_UINT32,IM_UINT32_BE,IM_INT32,IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,(IM_INT16_2,x),(IM_INT16_2_BE,x),)
##        IM_INT16_BE:(IM_INT16,IM_UINT32,IM_UINT32_BE,IM_INT32,IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,(IM_INT16_2,x),(IM_INT16_2_BE,x),)
##        IM_UINT32:(IM_UINT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_UINT32_BE:(IM_UINT32,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_INT32:(IM_INT32_BE,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_INT32_BE:(IM_INT32,IM_UINT64,IM_UINT64_BE,IM_INT64,IM_INT64_BE,IM_FLOAT32,IM_FLOAT32_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX64,IM_COMPLEX64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_UINT64:(IM_UINT64_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_UINT64_BE:(IM_UINT64,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_INT64:(IM_INT64_BE,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_INT64_BE:(IM_INT64,IM_FLOAT64,IM_FLOAT64_BE,IM_COMPLEX128,IM_COMPLEX128_BE,)
##        IM_RGB24:((IM_RGBA32,x),),
##        IM_RGBA32:(),
##        
##    }
##
##def im_coerce_dtype(im,targets):
##    im = im_standardize_dtype(im)
##    if im.dtype in targets: return im
##    t = _coerce[im.dtype]
##    for t in _coerce:
##        t,f = t if isinstance(t, tuple) else t,_astype
##        if t in targets:
##            try: return f(im,t)
##            except: pass
##    raise TypeError('Cannot change image type to supported format')

##### Group Image Types #####
IM_INT_TYPES     = (IM_INT8,  IM_INT16,  IM_INT16_BE,  IM_INT32,  IM_INT32_BE,  IM_INT32,  IM_INT32_BE,
                    IM_UINT8, IM_UINT16, IM_UINT16_BE, IM_UINT32, IM_UINT32_BE, IM_UINT64, IM_UINT64_BE)
IM_FLOAT_TYPES   = (IM_FLOAT32, IM_FLOAT64, IM_FLOAT32_BE, IM_FLOAT64_BE)
IM_COMPLEX_TYPES = (IM_COMPLEX64, IM_COMPLEX64_BE, IM_COMPLEX128, IM_COMPLEX128_BE, IM_INT16_2, IM_INT16_2_BE, IM_INT16_2_RAW, IM_INT16_2_RAW_BE)
IM_COLOR_TYPES   = (IM_RGB24, IM_RGB24_RAW, IM_RGBA32, IM_RGBA32_RAW)
IM_COLOR_ALPHA_TYPES = (IM_RGBA32, IM_RGBA32_RAW)
IM_RANGED_TYPES  = (IM_BIT,) + IM_INT_TYPES + IM_FLOAT_TYPES
IM_ALL_TYPES     = IM_RANGED_TYPES + IM_COMPLEX_TYPES + IM_COLOR_TYPES

##### Min/Max for data types #####
_min_max_values = { dt:(iinfo(dt).min,iinfo(dt).max) for dt in IM_INT_TYPES }
_min_max_values[IM_BIT] = (False, True)
_min_max_values[IM_FLOAT32] = (float32(0.0),float32(1.0))
_min_max_values[IM_FLOAT64] = (float64(0.0),float64(1.0))
_min_max_values[IM_FLOAT32_BE] = (float32(0.0),float32(1.0))
_min_max_values[IM_FLOAT64_BE] = (float64(0.0),float64(1.0))
def get_im_min_max(im):
    """Gets the min and max values for an image or an image dtype."""
    if isinstance(im, dtype): dt = im; im = None
    else: dt = im.dtype
    if im == None or dt not in IM_FLOAT_TYPES: return _min_max_values[dt]
    mn, mx = im.min(), im.max()
    return (mn, mx) if mn < 0.0 or mx > 1.0 else _min_max_values[dt]

def get_dtype_endian(dtype):
    """Get a '<' or '>' from a dtype's byteorder (which can be |, =, <, or >)."""
    endian = dtype.byteorder
    if endian == '|': return '<' # | means N/A (single byte), report as little-endian
    elif endian == '=': return '<' if byteorder == 'little' else '>' # is native byte-order
    return endian

def dtype2desc(dtype):
    """Get very brief string description from an image dtype."""
    if   dtype in (IM_RGB24,  IM_RGB24_RAW ): return 'rgb24'
    elif dtype in (IM_RGBA32, IM_RGBA32_RAW): return 'rgba32'
    elif dtype in (IM_INT16_2, IM_INT16_2_BE, IM_INT16_2_RAW, IM_INT16_2_RAW_BE): return 'cs%s32' % (dtype.byteorder)
    elif dtype == IM_BIT:    return 'g1'
    elif dtype == IM_INT8:   return 'gs8'
    elif dtype == IM_UINT8:  return 'g8'
    elif dtype.kind == 'c':  return 'cf%d'   % (dtype.itemsize * 8)
    elif dtype.kind == 'f':  return 'gf%s%d' % (dtype.byteorder, dtype.itemsize * 8)
    elif dtype.kind == 'u':  return 'g%s%d'  % (dtype.byteorder, dtype.itemsize * 8)
    elif dtype.kind == 'i':  return 'gs%s%d' % (dtype.byteorder, dtype.itemsize * 8)
    else:                    return str(dtype)

#def make_rgb24_color (r,g=None,b=None):       return array([ r                           if (g,b)==(None,None) and isinstance(r,tuple) and len(r)==3 else (r,g,b,a)],dtype=IM_RGB24 )[0]
#def make_rgba32_color(r,g=None,b=None,a=255): return array([(r if len(r)==4 else r+(a,)) if (g,b)==(None,None) and isinstance(r,tuple)               else (r,g,b,a)],dtype=IM_RGBA32)[0]
