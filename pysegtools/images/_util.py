"""Utilities for library use."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, struct, functools
from operator import mul
from collections import Iterable
from itertools import repeat

##### numpy-like functions for iterators #####
def prod(itr): return functools.reduce(mul, itr, 1)
def ravel(itr): return (x for i in itr for x in (ravel(i) if isinstance(i, Iterable) and not isinstance(i, String) else (i,)))
def __reshape(itr, shape, otype): return otype((next(itr) for _ in xrange(shape[0])) if len(shape) == 1 else (reshape(itr, shape[1:], otype) for _ in xrange(shape[0])))
def reshape(itr, shape, otype=list): return __reshape(iter(itr), tuple(shape) if isinstance(shape, Iterable) else (shape,), otype) # otype can be list or tuple


##### string and casting utilities #####
__is_py3 = sys.version_info[0] == 3
String = str if __is_py3 else basestring
Unicode = str if __is_py3 else unicode
def re_search(re, s):
    re_search.match = m = re.search(s)
    return m is not None
def itr2str(itr, sep=' '): return sep.join(type(sep)(x) for x in itr)
def splitstr(s, cast=lambda x:x, sep=None): return [cast(x) for x in s.split(sep)]
def get_list(data, shape, cast=int, sep=None, otype=list):
    """
    Convert a string of values to a list of a particular data type. The data can also come from an
    iterable in which case all elements are ensured then to be the right type. You can specify
    either a single value or a tuple for the shape. The dtype defaults to int, but can be others.
    The seperator in the string defaults to all whitespace. The output type defaults to a list, but
    you can also set it to tuple to get an imutable output.
    """
    shape = tuple(shape) if isinstance(shape, Iterable) else (shape,)
    data = ((cast(x) for x in data.split(sep)) if isinstance(data, String) else ravel(data)) if isinstance(data, Iterable) else repeat(cast(data), prod(shape))
    return __reshape(data, shape, otype)
def _bool(x):
    """
    Casts a value to a bool taking into acount the string values "true", "false", "t", "f", "1",
    and "0" (not case-sensitive)
    """
    if isinstance(x, String):
        if x.lower() in ('false', 'f', '0'): return False
        if x.lower() in ('true',  't', '1'): return True
    return bool(x)
def dtype_cast(x, dtype):
    """Casts a value using a dtype specification."""
    a = array([x], dtype.base)
    if a.shape[1:] != dtype.shape: raise ValueError('Invalid value "' + v + '" for tag "' + k_ + '"')
    return a[0] if a.ndim == 1 else tuple(a[0])


##### struct utilities #####
def pack(fmt, *v): return struct.pack(str(fmt), *v)
def unpack(fmt, b): return struct.unpack(str(fmt), b)
def unpack1(fmt, b): return struct.unpack(str(fmt), b)[0]
