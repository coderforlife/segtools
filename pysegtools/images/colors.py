"""A library for colors."""

from numpy import void

from types import im_standardize_dtype
from numbers import Integral, Real, Complex
from _util import dtype_cast

__all__ = ['get_color']

_colors = {
    # shades of gray (single float)
    'black':      0.00, 'k':         0.00,
    'dimgray':    0.41, 'dimgrey':   0.41,
    'gray':       0.50, 'grey':      0.50,
    'darkgray':   0.66, 'darkgrey':  0.66,
    'silver':     0.75,
    'lightgray':  0.83, 'lightgrey': 0.83,
    'gainsboro':  0.86,
    'whitesmoke': 0.96,
    'white':      1.00, 'w':         1.00,

    # basic colors (float triples)
    'red':     (1.00, 0.00, 0.00), 'r': (1.00, 0.00, 0.00),
    'maroon':  (0.50, 0.00, 0.00), 
    'orange':  (1.00, 0.65, 0.00), 
    'yellow':  (1.00, 1.00, 0.00), 'y': (0.75, 0.75, 0.00), # not quite the same
    'olive':   (0.50, 0.50, 0.00), 
    'lime':    (0.00, 1.00, 0.00),
    'green':   (0.00, 0.50, 0.00), 'g': (0.00, 0.50, 0.00),
    'cyan':    (0.00, 1.00, 1.00), 'aqua': (0.00, 1.00, 1.00), 'c': (0.00, 0.75, 0.75), # not quite the same
    'teal':    (0.00, 0.50, 0.50),
    'blue':    (0.00, 0.00, 1.00), 'b': (0.00, 0.00, 1.00),
    'navy':    (0.00, 0.00, 0.50), 'navyblue': (0.00, 0.00, 0.50),
    'magenta': (1,00, 0.00, 1.00), 'fuchsia': (1,00, 0.00, 1.00), 'm': (0.75, 0.00, 0.75), # not quite the same
    'purple':  (0.50, 0.00, 0.50),

    # special
    'transparent': (0.0, 0.0, 0.0, 0.0),

    # TODO: rest of the HTML colors
    }

def _color_name_strip(x): return x.strip().replace(' ', '').replace('-', '')

def _get_bit_color(x):
    from numpy import bool_
    if isinstance(x, (bool, bool_)): return bool_(x)
    if isinstance(x, Real):
        if x == 1: return True
        if x == 0: return False
    elif isinstance(x, basestring):
        x = x.strip().lower()
        if x in ('1', 'true',  't'): return True
        if x in ('0', 'false', 'f'): return False
    raise ValueError()
    
def _get_int_color(x, dtype):
    mn, mx = get_im_min_max(dtype)
    if isinstance(x, basestring):
        x = x.strip()
        try: x = long(x)
        except ValueError: x = float(x)
    if isinstance(x, Integral):
        if x >= mn  and x <= mx:  return dtype_cast(x, dtype)
    elif isinstance(x, Real):
        if x >= 0.0 and x <= 1.0: return dtype_cast(x*(mx-mn)+mn, dtype)
    raise ValueError()

def _get_color(x, dtype):
    if dtype is IM_BIT: return _get_bit_color(x)
    if dtype is IM_INT_TYPES: return _get_int_color(x, dtype)
    if dtype is IM_FLOAT_TYPES: return dtype_cast(x, dtype)
    raise ValueError()

##def is_color(x, im = None):
##    """
##    Checks if a value is a color. If an image is not given then it is checked if the value is
##    acceptable as a color for any image type. If an image is given it is checked if the value is
##    acceptable for that specific image type. See get_color for details on "acceptable" colors.
##    """
##    from numpy import void
##    from collections import Iterable
##    im = im_standardize_dtype(im)
##
##    # Basic conversion
##    if isinstance(x, basestring):
##        cn = _color_name_strip(x)
##        if cn in _colors: x = _colors[cn]
##    elif isinstance(x, Iterable):
##        x = tuple(x)
##        if len(x) == 1: x = x[0]
##    elif im is not None and im.dtype.type != void and isinstance(x, im.dtype.type): return True
##
##    # 
##    if im is None:
##        if isinstance(x, basestring):
##            x = x.strip()
##            try: x = long(x)
##            except ValueError:
##                try: x = float(x)
##                except ValueError: return False
##    else:
##        im = im_standardize_dtype(im)
##        if im.dtype is IM_BIT:
##            return isinstance(x, (bool, bool_)) or (isinstance(x, Real) and x in (1, 0)) or (isinstance(x, basestring) and x.strip().lower() in ('1', 'true',  't', '0', 'false', 'f'))
##        elif im.dtype in IM_INT_TYPES:
##            if isinstance(x, basestring):
##                x = x.strip()
##                if len(x) > 1 and x[0] == '#': x = int(x[1:], 16)
##            mn, mx = get_im_min_max(dtype)
##            if isinstance(x, basestring):
##                x = x.strip()
##                try: x = long(x)
##                except ValueError:
##                    try: x = float(x)
##                    except ValueError: return False
##            return (isinstance(x, Integral) and x >= mn and x <= mx) or
##                (isinstance(x, Real) and x >= 0.0 and x <= 1.0)
##        elif im.dtype in IM_FLOAT_TYPES:
##            if isinstance(x, basestring):
##                try: x = float(x.strip())
##                except ValueError: return False
##            return isinstance(x, Real)
##        elif im.dtype in IM_COMPLEX_TYPES:
##            ...
##            if isinstance(x, tuple) and len(x) == 2: x = complex(x[0], x[1])
##            elif isinstance(x, basestring): x = complex(x.strip())
##            return dtype_cast(x, im.dtype)
##        else: # colored types
##            l = len(im.dtype)
##            if isinstance(x, basestring):
##                x = x.strip()
##                if len(x) > 1 and x[0] == '#':
##                    n = im.dtype[0].itemsize * 2
##                    x = tuple(int(x[i:i+n], 16) for i in xrange(1, len(x), n))
##                elif ',' in x: x = tuple(x.split(','))
##                else:          x = tuple(x.split())
##            if ~isinstance(x, tuple): x = (x,)
##            if len(x) == 1: x = x*((l-1) if im.dtype in IM_COLOR_ALPHA_TYPES else l)
##            if im.dtype in IM_COLOR_ALPHA_TYPES and len(x) == l-1: x += (1.0,)
##            if len(x) == l: return tuple(_get_color(v, im.dtype[i]) for i,v in enumerate(x))

def get_color(x, im):
    """
    Gets a color from a value. The color returned is appropiate for the standardized form of the
    given image. Each image type supports and handles values differently.

    Acceptable values:
        IM_BIT:
            bool
            float or int that is either 0 or 1
            string that is one of '0', '1', 'true', 'false', 't', 'f', 'black', 'white', 'k', 'w' (case-insensitive)
            iterable of single value of one of the above
        IM_INT_TYPES:
            float from 0.0 to 1.0 (scaled to range of integer type)
            int from min to max of integer type
            string representing either of the above (if you want float 0.0 or 1.0 you must include the period)
            iterable of single value of one of the above
            string that starts with '#' that is a hex string (converted to integer, must be in range of integer type)
            string color names: 'black', 'k', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke', 'white', 'w', (case-insensitive, space/hyphen-insensitive, grays as grey)
        IM_FLOAT_TYPES:
            float or int from min to max of floating-point type
            string representing the above
            string color names: 'black', 'k', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke', 'white', 'w', (case-insensitive, space/hyphen-insensitive, grays as grey)
            iterable of single value of one of the above
        IM_COMPLEX_TYPES:
            float, int, or complex (float and int assume an imaginary part of 0)
            string representation of the above (complex uses 'j' for imaginary post-fix)
            iterable of single value of one of the above
            iterable of two floats/ints taken as real and imaginary parts
        IM_COLOR_TYPES:
            float from 0.0 to 1.0 (scaled to range of underlying integer type and used for each channel except alpha which is opaque)
            int from min to max of underlying integer type (used for each channel accept except which is opaque)
            string representing either of the above (if you want float 0.0 or 1.0 you must include the period)
            iterable of the above with a number of elements for each channel (except can be skipped, in which case it is made opaque)
            string with floats or ints seperated by commas or spaces, one for each channel (except can be skipped, in which case it is made opaque)
            string that starts with '#' that is a hex string (converted to integer, represents all channels)
            string [basic] HTML color names, Matplotlib single-letter color names, the string 'transparent' if alpha is supported
    """
    from numpy import void
    from collections import Iterable
    im = im_standardize_dtype(im)

    # Basic conversion
    if isinstance(x, basestring):
        x = x.strip()
        cn = _color_name_strip(x)
        if cn in _colors: x = _colors[cn]
    elif isinstance(x, Iterable):
        x = tuple(x)
        if len(x) == 1: x = x[0]
    elif im.dtype.type != void and isinstance(x, im.dtype.type): return x

    # Specialized conversions
    if im.dtype is IM_BIT: return _get_bit_color(x)
    elif im.dtype in IM_INT_TYPES:
        if isinstance(x, basestring) and len(x) > 1 and x[0] == '#': x = long(x[1:], 16)
        return _get_int_color(x, im.dtype)
    elif im.dtype in IM_FLOAT_TYPES: return dtype_cast(x, im.dtype)
    elif im.dtype in IM_COMPLEX_TYPES:
        # TODO: this is likely very broken (e.g. complex is not of the complex128 resolution, what if we have an integer-based complex number like INT16_2?)
        if isinstance(x, tuple) and len(x) == 2: x = complex(x[0], x[1])
        elif isinstance(x, basestring): x = complex(x)
        return dtype_cast(x, im.dtype)
    else: # colored types
        l = len(im.dtype)
        if isinstance(x, basestring):
            if len(x) > 1 and x[0] == '#':
                n = im.dtype[0].itemsize * 2
                x = tuple(long(x[i:i+n], 16) for i in xrange(1, len(x), n))
            elif ',' in x: x = tuple(x.split(','))
            else:          x = tuple(x.split())
        if ~isinstance(x, tuple): x = (x,)
        if len(x) == 1: x *= (l-1) if im.dtype in IM_COLOR_ALPHA_TYPES else l
        if im.dtype in IM_COLOR_ALPHA_TYPES and len(x) == l-1: x += (1.0,)
        if len(x) == l: return tuple(_get_color(v, im.dtype[i]) for i,v in enumerate(x))
