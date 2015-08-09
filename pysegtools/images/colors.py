"""A library for colors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Integral, Real
from collections import Iterable
from numpy import array, bool_

from .types import get_im_dtype_and_nchan, get_im_min_max
from ._util import String
from ..imstack import Help

__all__ = ['get_color','is_color']

__colors = {
    # shades of gray (single float)
    'black':      0.000, 'k':         0.000,
    'dimgray':    0.412, 'dimgrey':   0.412,
    'gray':       0.502, 'grey':      0.502,
    'darkgray':   0.663, 'darkgrey':  0.663,
    'silver':     0.753,
    'lightgray':  0.827, 'lightgrey': 0.827,
    'gainsboro':  0.863,
    'whitesmoke': 0.961,
    'white':      1.000, 'w':         1.000,

    # basic colors (float triples)
    'red':     (1.000, 0.000, 0.000), 'r': (1.000, 0.000, 0.000),
    'maroon':  (0.502, 0.000, 0.000),
    'orange':  (1.000, 0.647, 0.000),
    'yellow':  (1.000, 1.000, 0.000), 'y': (0.753, 0.753, 0.000), # not quite the same
    'olive':   (0.502, 0.502, 0.000),
    'lime':    (0.000, 1.000, 0.000),
    'green':   (0.000, 0.502, 0.000), 'g': (0.000, 0.502, 0.000),
    'cyan':    (0.000, 1.000, 1.000), 'aqua': (0.000, 1.00, 1.000), 'c': (0.000, 0.753, 0.753), # not quite the same
    'teal':    (0.000, 0.502, 0.502),
    'blue':    (0.000, 0.000, 1.000), 'b': (0.000, 0.000, 1.000),
    'navy':    (0.000, 0.000, 0.502), 'navyblue': (0.000, 0.000, 0.502),
    'magenta': (1,000, 0.000, 1.000), 'fuchsia': (1.000, 0.000, 1.000), 'm': (0.753, 0.000, 0.753), # not quite the same
    'purple':  (0.502, 0.000, 0.502),

    # special
    'transparent': (0.000, 0.000, 0.000, 0.000),

    # rest of the HTML colors
    'aliceblue':            (0.941, 0.973, 1.000),
    'antiquewhite':         (0.980, 0.922, 0.843),
    'aquamarine':           (0.498, 1.000, 0.831),
    'azure':                (0.941, 1.000, 1.000),
    'beige':                (0.961, 0.961, 0.863),
    'bisque':               (1.000, 0.894, 0.769),
    'blanchedalmond':       (1.000, 0.922, 0.804),
    'blueviolet':           (0.541, 0.169, 0.886),
    'brown':                (0.647, 0.165, 0.165),
    'burlywood':            (0.871, 0.722, 0.529),
    'cadetblue':            (0.373, 0.620, 0.627),
    'chartreuse':           (0.498, 1.000, 0.000),
    'chocolate':            (0.824, 0.412, 0.118),
    'coral':                (1.000, 0.498, 0.314),
    'cornflowerblue':       (0.392, 0.584, 0.929),
    'cornsilk':             (1.000, 0.973, 0.863),
    'crimson':              (0.863, 0.078, 0.235),
    'darkblue':             (0.000, 0.000, 0.545),
    'darkcyan':             (0.000, 0.545, 0.545),
    'darkgoldenrod':        (0.722, 0.525, 0.043),
    'darkgreen':            (0.000, 0.392, 0.000),
    'darkkhaki':            (0.741, 0.718, 0.420),
    'darkmagenta':          (0.545, 0.000, 0.545),
    'darkolivegreen':       (0.333, 0.420, 0.184),
    'darkorange':           (1.000, 0.549, 0.000),
    'darkorchid':           (0.600, 0.196, 0.800),
    'darkred':              (0.545, 0.000, 0.000),
    'darksalmon':           (0.914, 0.588, 0.478),
    'darkseagreen':         (0.561, 0.737, 0.561),
    'darkslateblue':        (0.282, 0.239, 0.545),
    'darkslategray':        (0.184, 0.310, 0.310), 'darkslategrey': (0.184, 0.310, 0.310),
    'darkturquoise':        (0.000, 0.808, 0.820),
    'darkviolet':           (0.580, 0.000, 0.827),
    'deeppink':             (1.000, 0.078, 0.576),
    'deepskyblue':          (0.000, 0.749, 1.000),
    'dodgerblue':           (0.118, 0.565, 1.000),
    'firebrick':            (0.698, 0.133, 0.133),
    'floralwhite':          (1.000, 0.980, 0.941),
    'forestgreen':          (0.133, 0.545, 0.133),
    'ghostwhite':           (0.973, 0.973, 1.000),
    'gold':                 (1.000, 0.843, 0.000),
    'goldenrod':            (0.855, 0.647, 0.125),
    'greenyellow':          (0.678, 1.000, 0.184),
    'honeydew':             (0.941, 1.000, 0.941),
    'hotpink':              (1.000, 0.412, 0.706),
    'indianred':            (0.804, 0.361, 0.361),
    'indigo':               (0.294, 0.000, 0.510),
    'ivory':                (1.000, 1.000, 0.941),
    'khaki':                (0.941, 0.902, 0.549),
    'lavender':             (0.902, 0.902, 0.980),
    'lavenderblush':        (1.000, 0.941, 0.961),
    'lawngreen':            (0.486, 0.988, 0.000),
    'lemonchiffon':         (1.000, 0.980, 0.804),
    'lightblue':            (0.678, 0.847, 0.902),
    'lightcoral':           (0.941, 0.502, 0.502),
    'lightcyan':            (0.878, 1.000, 1.000),
    'lightgoldenrodyellow': (0.980, 0.980, 0.824),
    'lightgreen':           (0.565, 0.933, 0.565),
    'lightpink':            (1.000, 0.714, 0.757),
    'lightsalmon':          (1.000, 0.627, 0.478),
    'lightseagreen':        (0.125, 0.698, 0.667),
    'lightskyblue':         (0.529, 0.808, 0.980),
    'lightslategray':       (0.467, 0.533, 0.600), 'lightslategrey': (0.467, 0.533, 0.600),
    'lightsteelblue':       (0.690, 0.769, 0.871),
    'lightyellow':          (1.000, 1.000, 0.878),
    'limegreen':            (0.196, 0.804, 0.196),
    'linen':                (0.980, 0.941, 0.902),
    'mediumaquamarine':     (0.400, 0.804, 0.667),
    'mediumblue':           (0.000, 0.000, 0.804),
    'mediumorchid':         (0.729, 0.333, 0.827),
    'mediumpurple':         (0.576, 0.439, 0.859),
    'mediumseagreen':       (0.235, 0.702, 0.443),
    'mediumslateblue':      (0.482, 0.408, 0.933),
    'mediumspringgreen':    (0.000, 0.980, 0.604),
    'mediumturquoise':      (0.282, 0.820, 0.800),
    'mediumvioletred':      (0.780, 0.082, 0.522),
    'midnightblue':         (0.098, 0.098, 0.439),
    'mintcream':            (0.961, 1.000, 0.980),
    'mistyrose':            (1.000, 0.894, 0.882),
    'moccasin':             (1.000, 0.894, 0.710),
    'navajowhite':          (1.000, 0.871, 0.678),
    'oldlace':              (0.992, 0.961, 0.902),
    'olivedrab':            (0.420, 0.557, 0.137),
    'orangered':            (1.000, 0.271, 0.000),
    'orchid':               (0.855, 0.439, 0.839),
    'palegoldenrod':        (0.933, 0.910, 0.667),
    'palegreen':            (0.596, 0.984, 0.596),
    'paleturquoise':        (0.686, 0.933, 0.933),
    'palevioletred':        (0.859, 0.439, 0.576),
    'papayawhip':           (1.000, 0.937, 0.835),
    'peachpuff':            (1.000, 0.855, 0.725),
    'peru':                 (0.804, 0.522, 0.247),
    'pink':                 (1.000, 0.753, 0.796),
    'plum':                 (0.867, 0.627, 0.867),
    'powderblue':           (0.690, 0.878, 0.902),
    'rosybrown':            (0.737, 0.561, 0.561),
    'royalblue':            (0.255, 0.412, 0.882),
    'saddlebrown':          (0.545, 0.271, 0.075),
    'salmon':               (0.980, 0.502, 0.447),
    'sandybrown':           (0.957, 0.643, 0.376),
    'seagreen':             (0.180, 0.545, 0.341),
    'seashell':             (1.000, 0.961, 0.933),
    'sienna':               (0.627, 0.322, 0.176),
    'skyblue':              (0.529, 0.808, 0.922),
    'slateblue':            (0.416, 0.353, 0.804),
    'slategray':            (0.439, 0.502, 0.565), 'slategrey': (0.439, 0.502, 0.565),
    'snow':                 (1.000, 0.980, 0.980),
    'springgreen':          (0.000, 1.000, 0.498),
    'steelblue':            (0.275, 0.510, 0.706),
    'tan':                  (0.824, 0.706, 0.549),
    'thistle':              (0.847, 0.749, 0.847),
    'tomato':               (1.000, 0.388, 0.278),
    'turquoise':            (0.251, 0.878, 0.816),
    'violet':               (0.933, 0.510, 0.933),
    'wheat':                (0.961, 0.871, 0.702),
    'yellowgreen':          (0.604, 0.804, 0.196),
}

def __color_help(width):
    p = Help(width)
    p.title("Colors")
    p.text("""
Many filters take a color as an argument. Below are the supported values for the various image
format types.

General rules for color values:""")
    p.list("all values are case-insensitive",
           "color names are space/hyphen-insensitive",
           "grays can be written as greys")
    p.newline()
    p.text("1-Bit Images: (U1)")
    p.list("on:  1, white, w, true, t",
           "off: 0, black, k, false, f")
    p.newline()
    p.text("Single-Channel Integer Images: (U# or I#)")
    p.list("floating-point value from 0.0 to 1.0 that will be scaled (you must include the period)",
           "integer value",
           "a hex value that starts with #",
           "grayscale color names (black, k, dim gray, gray, dark gray, silver, light gray, gainsboro, white smoke, white, w)")
    p.newline()
    p.text("Single-Channel Floating-Point Images: (F#)")
    p.list("floating-point or integer value",
           "grayscale color names (black, k, dim gray, gray, dark gray, silver, light gray, gainsboro, white smoke, white, w)")
    p.newline()
    p.text("Complex Images: (C#)")
    p.list("floating-point, integer, or complex value (e.g. \'2.5+3.5j\')")
    p.newline()
    p.text("Multi-Channel Images: (#x...)")
    p.list("a single value supported by the base type, repeated for each channel",
           "a comma or space seperated list of supported values of the base type (except color names)",
           "for integral base types, a hex value that starts with # that represents all channels together")
    p.newline()
    p.text("Color Images: (RGB# or RGBA#) [note: alpha is opaque if not explicit]")
    p.list("floating-point value from 0.0 to 1.0 that will be scaled (you must include the period, used for each channel)",
           "integer value (used for each channel)",
           "a comma or space seperated list of 3 or 4 floats/ints as above",
           "a hex value that starts with # that represents all channels together",
           "any Matplotlib single-letter or HTML color names or \'transparent\' if alpha is supported")
    p.newline()
    p.text("See also:")
    p.list("data-types")
Help.register(('color', 'colors', 'colour', 'colours'), __color_help)

def __color_name_strip(x): return x.replace(' ', '').replace('-', '')

def __no_throw(f,x):
    try: f(x); return True
    except StandardError: return False

def __is_color_channel(x):
    if isinstance(x, String):
        x = x.strip().lower()
        return x in ('true', 'false', 't', 'f') or __no_throw(complex,x) or __no_throw(long,x) or __no_throw(float,x)
    return isinstance(x, (bool, bool_, Real, complex))

def is_color(x):
    """
    Checks if a value represents a color for any image type. This may mean that is_color(x) works
    but get_color(x, im) fails for the same value x because im does not support the color value x.
    """
    if __is_color_channel(x): return True
    if isinstance(x, String):
        x = x.strip().lower()
        if __color_name_strip(x) in __colors: return True
        if len(x) > 1 and x[0] == '#': return __no_throw(lambda x:long(x[1:], 16), x)
        x = x.split(',' if ',' in x else None)
    return isinstance(x, Iterable) and all(__is_color_channel(c) for c in x)

def __get_bit_color(x):
    if isinstance(x, (bool, bool_)): return bool_(x)
    if isinstance(x, Real):
        if x == 1: return bool_(True)
        if x == 0: return bool_(False)
    elif isinstance(x, String):
        if x in ('1', 'true',  't'): return bool_(True)
        if x in ('0', 'false', 'f'): return bool_(False)
    raise ValueError()

def __get_int_color(x, dt):
    mn, mx = get_im_min_max(dt)
    if isinstance(x, String):
        try: x = dt.type(x)
        except ValueError: x = float(x)
    if isinstance(x, Integral):
        if x >= mn  and x <= mx:  return dt.type(x)
    elif isinstance(x, Real):
        if x >= 0.0 and x <= 1.0: return dt.type(round(x*(mx-mn)+mn))
    raise ValueError()

def __get_color_channel(x, dt):
    if isinstance(x, dt.type): return x
    if dt.kind == 'b':  return __get_bit_color(x)
    if dt.kind in 'ui': return __get_int_color(x, dt)
    if dt.kind in 'fc': return dt.type(x)
    raise ValueError()

def get_color(x, im_or_dtype):
    """
    Gets a color from a value. The color returned is appropiate for the given image / dtype. Each
    image type supports and handles values differently. The return value is either a NumPy scalar
    for single-channel images or a NumPy array for multi-channel images.

    Acceptable values:
        1-bit / bool:
            bool
            float or int that is either 0 or 1
            string that is one of '0', '1', 'true', 'false', 't', 'f', 'black', 'white', 'k', 'w' (case-insensitive)
            iterable of single value of one of the above
        Integral:
            float from 0.0 to 1.0 (scaled to range of integer type)
            int from min to max of integer type
            string representing either of the above (if you want float 0.0 or 1.0 you must include the period)
            iterable of single value of one of the above
            string that starts with '#' that is a hex string (converted to integer, must be in range of integer type)
            string color names: 'black', 'k', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke', 'white', 'w', (case-insensitive, space/hyphen-insensitive, grays as grey)
        Floating-Point:
            float or int from min to max of floating-point type
            string representing the above
            string color names: 'black', 'k', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke', 'white', 'w', (case-insensitive, space/hyphen-insensitive, grays as grey)
            iterable of single value of one of the above
        Complex:
            float, int, or complex (float and int assume an imaginary part of 0)
            string representation of the above (complex uses 'j' for imaginary post-fix, cannot have spaces)
            iterable of single value of one of the above
            iterable of two floats/ints taken as real and imaginary parts
        Multi-Channel:
            a single value supported by the base type, repeated for each channel
            an iterable of values supported by the base type, except color names (must have one value per channel)
            a comma-seperated string of values supported by the base type (must have one value per channel)
            for integral base types, a hex string that starts with '#' and has all channels together
        Color Types:
            float from 0.0 to 1.0 (scaled to range of underlying integer type and used for each channel except alpha which is opaque)
            int from min to max of underlying integer type (used for each channel accept except which is opaque)
            string representing either of the above (if you want float 0.0 or 1.0 you must include the period)
            iterable of the above with a number of elements for each channel (except alpha can be skipped, in which case it is made opaque)
            string with floats or ints seperated by commas or spaces, one for each channel (except alpha can be skipped, in which case it is made opaque)
            string that starts with '#' that is a hex string (converted to integer, represents all channels)
            string HTML color names, Matplotlib single-letter color names, the string 'transparent' if alpha is supported
    """
    dt, nchan = get_im_dtype_and_nchan(im_or_dtype)

    # Basic conversion, mainly string conversion
    # At the end of this block we have always have a tuple
    if isinstance(x, String):
        x = x.strip().lower()
        cn = __color_name_strip(x)
        if cn in __colors: x = __colors[cn]
        elif len(x) > 1 and x[0] == '#':
            if dt.kind not in 'ui': raise ValueError()
            nhex = dt.itemsize*2
            x = tuple(long(x[i:i+nhex],16) for i in xrange(1, len(x), nhex))
        else:
            x = tuple((x.strip() for x in x.split(',')) if ',' in x else x.split())
    elif isinstance(x, Iterable): x = tuple(x)
    if not isinstance(x, tuple):  x = (x,)

    # Multi-Channel handling
    # At the end of this block the tuple has 1 value per channel, but that value is not converted
    if nchan == 1 and dt.kind == 'c' and isinstance(x, tuple) and len(x) == 2: # Complex is really 2-channel even though it is kind of 1-channel...
        x = complex(x[0], x[1]) # TODO
    elif nchan == 4 and dt.kind == 'u': # RGBA
        if len(x) == 1: x *= 3      # don't fill in alpha channel
        if len(x) == 3: x += (1.0,) # set alpha channel to opaque
    elif len(x) == 1: x *= nchan
    if len(x) != nchan: raise ValueError()

    # Convert channels
    # At the of this block the tuple has 1 properly converted value per channel
    # Note: because NumPy scalars cannot have endian-ness the values might be the wrong endian
    x = tuple(__get_color_channel(c, dt) for c in x)

    # Make scalar/array of color
    return x[0] if nchan == 1 else array(x, dtype=dt)
