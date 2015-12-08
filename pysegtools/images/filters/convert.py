"""Filters that convert the image type."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from fractions import gcd
from itertools import izip
from collections import Sequence

from numpy import empty, zeros, subtract, add, dtype, promote_types

from ..types import check_image, create_im_dtype, im_dtype_desc
from ..types import get_dtype_endian, get_im_dtype_and_nchan, get_im_min_max, get_dtype_min_max
from ._stack import FilteredImageStack, FilteredImageSlice
from .._stack import Homogeneous
from ...imstack import CommandEasy, Opt
from ...general import sys_endian

__all__ = ['threshold', 'raw_convert', 'convert_byte_order', 'scale']

# convert: any single channel types excluding complex
# to:      single channel bool
def threshold(im, thresh=1):
    """
    Convert image to black and white. The threshold is used to determine what is made black and white.
    Every value at or above the threshold will be white and below it will be black.
    """
    check_image(im)
    dt, nchan = get_im_dtype_and_nchan(im)
    if dt.kind == 'c' or nchan != 1: raise ValueError('unsupported image type')
    return im>=thresh

# convert: any type
# to:      a type that has the same itemsize
def raw_convert(im, dt):
    """
    Convert image data type to a data type with the same byte-size directly (for example, from
    signed to unsigned integers of the same size).
    """
    check_image(im)
    if im.dtype.itemsize != dt.itemsize: raise ValueError('must convert to type with same byte size')
    return im.view(dt)
        

__byte_orders = {
    '<': '<', 'l': '<', 'little': '<',
    '>': '>', 'b': '>', 'big': '>',
    '=': sys_endian, '@': sys_endian, 'n': sys_endian, 'native': sys_endian,
    '~': '~', 's': '~', 'swap': '~', # not used by convert_byte_order but used by ConvertByteOrderImageStack
    }
# convert: any type
# to:      same type but with a new byte-order
def convert_byte_order(im, new_byte_order):
    """
    Change image data type's byte-order. The new_byte_order can be:

      '<', 'l', 'little',
      '>', 'b', 'big',
      '=', '@', 'n', 'native',
      '~', 's', 'swap'
    """
    check_image(im)
    if new_byte_order in ('~', 'swap'):
        new_byte_order = '>' if get_dtype_endian(im.dtype) == '<' else '<'
    else:
        new_byte_order = __byte_orders.get(new_byte_order)
        if new_byte_order is None: raise ValueError('invalid new byte order')
    return im.view(im.dtype.newbyteorder(new_byte_order))

# convert: any non-complex type
# to:      any type with the same number of channels
def scale(im, in_scale=None, out_scale=None, dt=None):
    """
    Calculates:
      (im - in_scale[0]) * (out_scale[1] - out_scale[0]) / (in_scale[1] - in_scale[0]) + out_scale[0]
    Taking into account type conversion and int overflows when possible
    im can be multi-channel in which case each channel is done independently
    in_scale[0] < in_scale[1]
    out_scale[0] != out_scale[1]
    """
    
    # Process arguments
    check_image(im)
    if in_scale is None and out_scale is None and dt is None: return im
    cur = im.dtype
    dt = cur if dt is None else dtype(dt)
    if dt.base != dt: raise ValueError('can only accept basic dtypes')
    if cur.kind not in 'biuf' or dt.kind not in 'biuf': raise ValueError('cannot scale complex values')

    # cur_min/max is the min and max of the cur dtype (or possibly 0.0/1.0 for floats)
    # in_min/max is the actual min and max of the image data
    # in_scale is the range of values we want to map to out
    cur_min, cur_max = get_im_min_max(im)
    in_min, in_max = im.min(), im.max()
    if in_scale is None: in_scale = cur_min, cur_max
    elif len(in_scale) != 2: raise ValueError('invalid in_scale')
    else:
        in_scale = cur.type(in_scale[0]), cur.type(in_scale[1])
        if in_scale[0] >= in_scale[1]: raise ValueError('invalid in_scale')
        if in_min > in_scale[0] or in_max < in_scale[1]: raise ValueError('image values outside of in_scale')

    out_min, out_max = get_dtype_min_max(dt)
    if out_scale is None: out_scale = out_min, out_max
    elif len(out_scale) != 2: raise ValueError('invalid out_scale')
    else:
        # TODO: check if out_scale is within the range of dt
        out_scale = dt.type(out_scale[0]), dt.type(out_scale[1])
        if out_scale[0] == out_scale[1]: raise ValueError('invalid out_scale')

    rev = out_scale[0] > out_scale[1]
    if rev: out_scale = out_scale[::-1]

    # Perform conversion
    # b -> any
    if cur.kind == 'b':
        if dt.kind == 'b':
            out = im.astype(dt)
            if rev: out = ~out
        else:
            out = empty(im.shape, dtype=dt)
            out[~im], out[im] = out_scale #pylint: disable=unpacking-non-sequence 
        return out

    # any -> b
    if dt.kind == 'b':
        out = zeros(im.shape, dtype=dt)
        half = (in_scale[1]-in_scale[0])/2 if cur.kind == 'f' else (in_scale[1]-in_scale[0])//2
        out[out<=half if rev else out>half] = True
        return out

    # im - in_scale[0]
    if cur.kind == 'i': im = im.view(dtype(cur.byteorder+'u'+str(cur.itemsize)))
    im = im - in_scale[0]

    # int -> int
    if cur.kind in 'iu' and dt.kind in 'iu':
        # im * out_range / in_range
        u_dt = dtype(dt.byteorder+'u'+str(dt.itemsize))
        in_range  = long(in_scale[1])  - long(in_scale[0])
        out_range = long(out_scale[1]) - long(out_scale[0])
        if in_range != out_range:
            d = gcd(in_range, out_range)
            in_range, out_range = in_range//d, out_range//d
            if in_range == 1:
                im = im.astype(u_dt, copy=False)
                im *= im.u_dt.type(out_range)
            elif out_range == 1:
                im /= im.dtype.type(in_range)
            else:
                im = im.astype(dtype(cur.byteorder+'u'+str(max(cur.itemsize, dt.itemsize))), copy=False)
                big_type = im.dtype.type
                q,r = divmod(im, big_type(in_range))
                q *= big_type(out_range)
                q = q.astype(u_dt, copy=False)
                r *= big_type(out_range)
                r //= big_type(in_range)
                r = r.astype(u_dt, copy=False)
                im = add(q, r, q)

        # reverse values
        if rev: subtract(out_range, im, im)
  
        # im + out_scale[0]
        im = im.astype(u_dt, copy=False)
        im += out_scale[0]
        return im.view(dt)

    # any -> f or f -> any
    im = im.astype(promote_types(cur, dt).newbyteorder(cur.byteorder), copy=False) # always floating-point
    im *= (out_scale[1]-out_scale[0]) / (in_scale[1]-in_scale[0])
    if rev: subtract((out_scale[1]-out_scale[0]), im, im)
    im += out_scale[0]
    return im.astype(dt, copy=False)


########## Image Stacks ##########
class ThresholdImageStack(FilteredImageStack):
    def __init__(self, ims, thresh=1):
        if isinstance(thresh, Sequence):
            if len(thresh) < len(ims):
                thresh = list(thresh) + [thresh[-1]]*(len(ims)-len(thresh))
        else:
            self._dtype, self._homogeneous = bool, Homogeneous.DType
            thresh = [thresh] * len(ims)
        super(ThresholdImageStack, self).__init__(ims,
            [ThresholdImageSlice(im, self, z, t) for z,(im,t) in enumerate(izip(ims, thresh))])
class ThresholdImageSlice(FilteredImageSlice):
    def __init__(self, im, stack, z, thresh):
        super(ThresholdImageSlice, self).__init__(im, stack, z)
        self.__threshold = thresh
        self._set_props(dtype(bool), None)
    def _get_props(self): self._set_props(None, self._input.shape)
    def _get_data(self): return threshold(self._input.data, self.__threshold)

class RawConvertImageStack(FilteredImageStack):
    def __init__(self, ims, dt):
        if isinstance(dt, Sequence):
            if len(dt) < len(ims): dt = list(dt) + [dt[-1]]*(len(ims)-len(dt))
        else:
            self._dtype, self._homogeneous = dt, Homogeneous.DType
            dt = [dt] * len(ims)
        super(RawConvertImageStack, self).__init__(ims,
            [RawConvertImageSlice(im, self, z, dt) for z,(im,dt) in enumerate(izip(ims, dt))])
class RawConvertImageSlice(FilteredImageSlice):
    def __init__(self, im, stack, z, dt):
        super(RawConvertImageSlice, self).__init__(im, stack, z)
        self._set_props(dt, None)
    def _get_props(self): self._set_props(None, self._input.shape)
    def _get_data(self): return raw_convert(self._input.data, self._dtype)

class ConvertByteOrderImageStack(FilteredImageStack):
    def __init__(self, ims, new_byte_order):
        if new_byte_order in __byte_orders:
            new_byte_orders = __byte_orders[new_byte_order] * len(ims)
        else:
            try:
                new_byte_orders = "".join(__byte_orders[nbo] for nbo in new_byte_order)
                if len(new_byte_orders) < len(ims):
                    new_byte_orders += new_byte_orders[-1]*(len(ims)-len(new_byte_orders))
            except KeyError: raise ValueError('invalid new byte order')
        super(ConvertByteOrderImageStack, self).__init__(ims,
            [ConvertByteOrderImageSlice(im, self, z, nbo)
             for z,(im,nbo) in enumerate(izip(ims, new_byte_orders))])
class ConvertByteOrderImageSlice(FilteredImageSlice):
    def __init__(self, im, stack, z, new_byte_order):
        super(ConvertByteOrderImageSlice, self).__init__(im, stack, z)
        self.__new_byte_order = new_byte_order
    def _get_props(self):
        dt = self._input.dtype
        nbo = self.__new_byte_order
        if nbo == '~': nbo = '>' if get_dtype_endian(dt) == '<' else '<'
        self._set_props(dt.newbyteorder(nbo), self._input.shape)
    def _get_data(self): return convert_byte_order(self._input.data, self.__new_byte_order)

class ScaleImageStack(FilteredImageStack):
    def __init__(self, ims, in_scale=None, out_scale=None, dt=None):
        if in_scale not in (None, 'data', 'stack-data') and (len(in_scale) != 2 or in_scale[0] >= in_scale[1]): raise ValueError('invalid in_scale')
        if out_scale is not None and (len(out_scale) != 2 or out_scale[0] == out_scale[1]): raise ValueError('invalid out_scale')
        if dt is not None and dt.base != dt or dt.kind not in 'biuf': raise ValueError('invalid conversion data-type')
        self._dtype = dt
        if dt is not None: self._homogeneous = Homogeneous.DType
        if in_scale == 'data':
            self._scale = lambda im: scale(im, (im.min(),im.max()), out_scale, dt)
        elif in_scale == 'stack-data':
            self._stack_range = None
            self._scale = lambda im: scale(im, self.stack_range, out_scale, dt)
        else:
            self._scale = lambda im: scale(im, in_scale, out_scale, dt)
        super(ScaleImageStack, self).__init__(ims, ScaleImageSlice)
    @property
    def stack_range(self):
        if self._stack_range is None:
            mn, mx = None, None
            for slc in self._ims:
                im = slc.data
                a, b = im.min(), im.max()
                if mn is None or a < mn: mn = a
                if mx is None or b < mx: mx = b
            self._stack_range = mn, mx
        return self._stack_range
class ScaleImageSlice(FilteredImageSlice):
    #pylint: disable=protected-access
    def __init__(self, im, stack, z):
        super(ScaleImageSlice, self).__init__(im, stack, z)
        if stack._dtype is not None: self._set_props(stack._dtype, None)
    def _get_props(self):
        dt = self._stack._dtype if self._stack._dtype is not None else self._input.dtype
        self._set_props(dt, self._input.shape)
    def _get_data(self): return self._stack._scale(self._input.data)


########## Commands ##########
class ThresholdCommand(CommandEasy):
    _threshold = None
    @classmethod
    def name(cls): return 'threshold'
    @classmethod
    def _desc(cls): return """
Threshold a gray-scale image converting it to just black and white. All pixels less than the
threshold will be made black/0 and all pixels greater than or equal to it will be made white/1. By
using a comma-seperated list of thresholds, each slice can have a different threshold applied.
"""
    @classmethod
    def flags(cls): return ('bw', 'threshold', 'thresh', 't')
    @classmethod
    def _opts(cls): return (
        Opt('threshold', 'The threshold value, either an integer, float, or a comma-seperated list of values', Opt.cast_tuple_of(Opt.cast_number())),
        )
    @classmethod
    def _consumes(cls): return ('Grayscale image stack to be thresholded',)
    @classmethod
    def _produces(cls): return ('Thresholded image - a logical/bool image',)
    @classmethod
    def _see_also(cls): return ('invert','scale')
    def __str__(self):
        if len(self._threshold) == 1:
            return ('threshold at %s' % self._threshold)
        else:
            return 'threshold at [%s]' % (",".join(str(t) for t in self._threshold))
    def execute(self, stack): stack.push(ThresholdImageStack(stack.pop(), self._threshold))

class RawConvertCommand(CommandEasy):
    _dtype = None
    @classmethod
    def name(cls): return 'raw conversion'
    @classmethod
    def _desc(cls): return """
Convert the data type of images to a data type with the same number of bits directly (for example,
from signed to unsigned integers of the same size).
"""
    @classmethod
    def flags(cls): return ('rc', 'raw-convert')
    @classmethod
    def _opts(cls): return (
        Opt('dtype', 'The new data type (see --help data-types) or a comma-seperated list of values', Opt.cast_tuple_of(create_im_dtype)),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be converted',)
    @classmethod
    def _produces(cls): return ('Converted image stack',)
    @classmethod
    def _see_also(cls): return ('scale','byte-order-convert','data-types')
    def __str__(self):
        if len(self._dtype) == 1:
            return 'raw-convert to %s' % im_dtype_desc(self._dtype)
        else:
            return 'raw-convert to [%s]' % (",".join(im_dtype_desc(dt) for dt in self._dtype))
    def execute(self, stack): stack.push(RawConvertImageStack(stack.pop(), self._dtype))

class ByteOrderConvertCommand(CommandEasy):
    _new = None
    @staticmethod
    def _cast_endian(x):
        if len(x) == 0: raise ValueError
        try: return "".join(__byte_orders[x] for x in x)
        except KeyError: raise ValueError

    @classmethod
    def name(cls): return 'byte-order conversion'
    @classmethod
    def _desc(cls): return """
Convert the byte-order of the data of images.

The new byte order is specified as one of the following: 
  * little-endian: <, l, little 
  * big-endian:    >, b, big 
  * native:        =, @, n, native (%s on this machine) 
  * swap:          ~, s, swap      (big to little, little to big)

If you wish to have different values for each image slice, then give a value composed of many
single-character symbols (e.g. '<><><>').
""" % ('little' if sys_endian == '<' else 'big')
    @classmethod
    def flags(cls): return ('boc', 'byte-order-convert', 'endian')
    @classmethod
    def _opts(cls): return (
        Opt('new', 'The single new byte order or a many single-character ones', ByteOrderConvertCommand._cast_endian),
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be converted',)
    @classmethod
    def _produces(cls): return ('Converted image stack',)
    @classmethod
    def _see_also(cls): return ('raw-convert',)
    def __str__(self):
        if len(self._new) == 1:
            return 'byte-order-convert to %s' % self._new
        else:
            return 'byte-order-convert to [%s]' % self._new
    def execute(self, stack): stack.push(ConvertByteOrderImageStack(stack.pop(), self._new))

class ConvertScaleConvertCommand(CommandEasy):
    _in = None
    _out = None
    _dt = None
    @staticmethod
    def _cast_in_scale(x):
        if x == 'data-type': return None
        if x in ('data', 'stack-data'): return x
        if len(x) == 0: raise ValueError
        x = Opt.cast_tuple_of(Opt.cast_number(), 2, 2)(x)
        if isinstance(x, complex) or x[0] >= x[1]: raise ValueError
        return x
    @staticmethod
    def _cast_out_scale(x):
        if x == 'data-type': return None
        if len(x) == 0: raise ValueError
        x = Opt.cast_tuple_of(Opt.cast_number(), 2, 2)(x)
        if isinstance(x, complex) or x[0] == x[1]: raise ValueError
        return x
    @staticmethod
    def _cast_dtype(x):
        if x == 'same': return None
        x = create_im_dtype(x)
        if x.base != x or x.kind not in 'biuf': raise ValueError
        return x
    @classmethod
    def name(cls): return 'scale and convert'
    @classmethod
    def _desc(cls): return """
Scales and/or converts image data. If no arguments are provided this will do nothing. To scale data,
specify at least one of 'in' or 'out' which default to the data-types. To convert data, specify
'dt'. Specifying both preforms the scaling and conversion together.

This ends up solving the following for every channel of every pixel: 
   (X-in[0]) * (out[1]-out[0]) / (in[1]-in[0]) + out[0] 
while handling type conversions and some overflow conditions.

The 'in' and 'out' arguments effect scaling. They default to the range of values supported by the
data-type of the input or output. The ranges for common types are: 
  * U1:  0 to 1 or false to true 
  * U8:  0 to 255 
  * U16: 0 to 65535 
  * U32: 0 to 4294967295 
  * I8:  -128 to 127 
  * I16: -32768 to 32767 
  * I32: -2147483648 to 2147483647 
  * F16: 0.0 to 1.0 
  * F32: 0.0 to 1.0 
  * F64: 0.0 to 1.0 
One important note is that floating-point values default to a range from 0.0 to 1.0 instead of the
available range for the type. Also, complex values are not supported so decomplexify first.

Besides using the ranges of the data-types the input scaling can also be specified to use the actual
range of the data, either based on the current slice or the entire stack. Additionally for either
the input or output scale a custom range can be given with two values seperated by a comma.

The 'dt' argument supports any non-complex data-type without channel information. The number of
channels is always kept the same. It also supports the value 'same' which is the default and means
the data-type won't be changed.
"""
    @classmethod
    def flags(cls): return ('scale', 'convert')
    @classmethod
    def _opts(cls): return (
        Opt('in', 'The range to scale from, one of "data-type", "data", "stack-data", or two comma-seperated values for lower and upper',
            ConvertScaleConvertCommand._cast_in_scale, None, 'data-type'),
        Opt('out', 'The range to scale to, one of "data-type" or two comma-seperated values for lower and upper (if reversed data will be reversed)',
            ConvertScaleConvertCommand._cast_out_scale, None, 'data-type'),
        Opt('dt', 'The new data-type (see --help data-types) or "same" to not convert',
            ConvertScaleConvertCommand._cast_dtype, None, 'same')
        )
    @classmethod
    def _consumes(cls): return ('Image stack to be scaled and/or converted',)
    @classmethod
    def _produces(cls): return ('Scaled and/or converted image stack',)
    @classmethod
    def _see_also(cls): return ('raw-convert','threshold','data-types')
    def __str__(self):
        if self._in is None and self._out is None:
            if self._dt is None: return 'do nothing'
            return 'convert to %s' % im_dtype_desc(self._dt)
        s = 'scale'
        if self._in is not None: s += (' from [%s-%s]' if len(self._in)==2 else ' %s') % self._in
        if self._out is not None: s += ' to [%s-%s]' % self._out
        if self._dt is not None: s += ' converting to %s' % im_dtype_desc(self._dt)
        return s
    def execute(self, stack): stack.push(ScaleImageStack(stack.pop(), self._in, self._out, self._dt))
