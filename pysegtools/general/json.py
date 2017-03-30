"""
JSON saver and loader that support NumPy arrays and additional objects beyond the builtin json
library (which this uses internally).

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function
        
__all__ = ['save', 'load', 'add_class_name']

def save(path, data):
    """
    Saves the data to a gzipped JSON file supporting NumPy arrays and other objects in addition to
    the standard dictionaries, lists, strings, numbers, and booleans.
    """
    from pysegtools.general import GzipFile
    from json import dump
    with GzipFile(path, 'wb') as f:
        dump(data, f, separators=(',',':'), default=__json_save_hook)
        
def load(path):
    """
    Loads given gzipped JSON file supporting NumPy arrays and other objects in addition to the
    standard dictionaries, lists, strings, numbers, and booleans.
    """
    from pysegtools.general import GzipFile
    from json import load
    with GzipFile(path, 'rb') as f:
        return load(f, object_hook=__json_load_hook)

def add_class_name(name, cls):
    """
    Adds a class name that will map to a base class, e.g. '__submodel__' will map to SubModel.
    """
    if name in __json_class_names and __json_class_names[name] != cls:
        from warnings import warn
        warn('overriding classname already registered in JSON save/load')
    __json_class_names[name] = cls
__json_class_names = {}
    
def __json_save_hook(o):
    """
    Function to use as default in json.dump/json.dumps for encoding a Model, SubModels, Filters,
    and Classifier objects into JSON. Besides the standard objects the JSON encoder can handle this
    is able to encode SubModel, Filter, and Classifier objects by encoding their FQN and their
    __dict__ or the result of __getstate__. Also this is able to encode NumPy arrays. Arrays with
    less than 1024 elements are saved directly into the JSON as lists. Otherwise arrays are saved
    in base-64 encoded raw binary data.
    """
    # Weighting the difference between using lists and base-64 encoded data
    # Storing arrays as lists:
    # * Human readable
    # * No byte order issues
    # * Data readable in any environment with a JSON decoder 
    # * Integer arrays likely to be much smaller
    # * After compression close to the same size even for floating-point numbers
    # * Might be closer in time in Python 3.x as it uses an improved json module
    # Storing arrays as base-64 encoded binary data:
    # * 30x faster for my example data (only ~150ms for all weights compared to ~4.2 secs)
    # * 3.5x faster during compression example data (~1.5 secs compared to ~4.5 secs)
    # * Significantly faster to read, compressed or decompressed
    # * 50% of the size when uncompressed
    # * 85% of the size when compressed
    from numpy import ndarray
    if isinstance(o, ndarray):
        if o.size <= 1024:
            # Arrays with only 1024 elements are saved as lists
            return {'__ndarray__':o.tolist(),'dtype':o.dtype.str}
        # Embed raw data as a single base-64 string
        from base64 import b64encode
        return {'__ndarray__': b64encode(o.tobytes()),'dtype':o.dtype.str,'shape':o.shape}
    for name,base in __json_class_names.iteritems():
        if isinstance(o, base): return {name: __json_save_obj(o)}
    raise TypeError('cannot convert object of '+type(o)+' to JSON')

def __json_save_obj(o):
    """For saving one of the object types registered."""
    cls = o.__class__
    name = getattr(cls, '__qualname__', cls.__name__)
    attr = getattr(o, '__getstate__', lambda:o.__dict__)()
    return [cls.__module__, name, attr]

def __json_load_hook(o):
    """
    A JSON load object_hook that supports SubModel, Filter, Classifer, and NumPy array objects.
    This matches the __json_save_hook function.
    """
    if '__ndarray__' in o:
        from numpy import array, fromstring, dtype
        dt = dtype(o.get('dtype', '<f8'))
        dt_native = dt.newbyteorder('=')
        data = o['__ndarray__']
        if isinstance(data, list): return array(data, dt_native)
        from base64 import b64decode
        data = fromstring(b64decode(data), dt_native).reshape(o['shape'])
        if dt.byteorder != '=': data = data.byteswap(True)
        return data
    return next((__json_load_obj(base, *o[name]) for name,base in __json_class_names.iteritems() if name in o), o)

def __json_load_obj(base, mod, cls, attr):
    """For loading one of the object types registered."""
    from importlib import import_module
    c = import_module(mod)
    for cn in cls.split('.'): c = getattr(c, cn)
    if not issubclass(c, base): raise TypeError('Unknown or bad '+base.__name__+' in model')
    o = c.__new__(c)
    getattr(o, '__setstate__', o.__dict__.update)(attr)
    return o
