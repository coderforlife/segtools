"""Utilities for library use."""

import numpy
import numbers
from operator import mul
from collections import Iterable
from itertools import islice, repeat


##### make numpy types part of the Python ABC heirarchy #####
# Note: in newer version of NumPy this is set by default, so we check
if not issubclass(numpy.int8, numbers.Integral):
    numbers.Integral.register(numpy.int_)
    numbers.Integral.register(numpy.intc)
    numbers.Integral.register(numpy.intp)
    numbers.Integral.register(numpy.int8)
    numbers.Integral.register(numpy.int16)
    numbers.Integral.register(numpy.int32)
    numbers.Integral.register(numpy.int64)
    numbers.Integral.register(numpy.uint8)
    numbers.Integral.register(numpy.uint16)
    numbers.Integral.register(numpy.uint32)
    numbers.Integral.register(numpy.uint64)
    numbers.Real.register(numpy.float_)
    numbers.Real.register(numpy.float16)
    numbers.Real.register(numpy.float32)
    numbers.Real.register(numpy.float64)
    numbers.Complex.register(numpy.complex_)
    numbers.Complex.register(numpy.complex64)
    numbers.Complex.register(numpy.complex128)


##### numpy-like functions for iterators #####
def prod(itr): return reduce(mul, itr, 1)
def ravel(itr): return (x for i in itr for x in (ravel(i) if isinstance(i, Iterable) and not isinstance(i, basestring) else (i,)))
def __reshape(itr, shape, otype): return otype((itr.next() for _ in xrange(shape[0])) if len(shape) == 1 else (reshape(itr, shape[1:], otype) for _ in xrange(shape[0])))
def reshape(itr, shape, otype=list): return __reshape(iter(itr), tuple(shape) if isinstance(shape, Iterable) else (shape,), otype) # otype can be list or tuple


##### string and casting utilities #####
def itr2str(itr, sep=' '): return sep.join(str(x) for x in itr)
def splitstr(s, cast=str): return [cast(x) for x in s.split()]
def get_list(data, shape, cast=int, sep=None, otype=list):
    """
    Convert a string of values to a list of a particular data type. The data can also come from an
    iterable in which case all elements are ensured then to be the right type. You can specify
    either a single value or a tuple for the shape. The dtype defaults to int, but can be others.
    The seperator in the string defaults to all whitespace. The output type defaults to a list, but
    you can also set it to tuple to get an imutable output.
    """
    shape = tuple(shape) if isinstance(shape, Iterable) else (shape,)
    data = ((cast(x) for x in data.split(sep)) if isinstance(v, basestring) else ravel(data)) if isinstance(data, Iterable) else repeat(cast(data), prod(shape))
    return __reshape(data, shape, otype)
def _bool(x):
    """
    Casts a value to a bool taking into acount the string values "true", "false", "t", "f", "1",
    and "0" (not case-sensitive)
    """
    if isinstance(x, bool): return x
    if isinstance(x, basestring):
        if x.lower() in ('false', 'f', '0'): return False
        if x.lower() in ('true',  't', '1'): return True
    return bool(x)
def dtype_cast(x, dtype):
    """Casts a value using a dtype specification."""
    a = numpy.array([x], dtype.base)
    if a.shape[1:] != dtype.shape: raise ValueError('Invalid value "' + v + '" for tag "' + k_ + '"')
    return a[0] if a.ndim == 1 else tuple(a[0])
