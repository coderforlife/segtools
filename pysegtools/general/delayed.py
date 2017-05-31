"""
Simple delay-loaded objects. Essentially this calls a function to generate the object the first time
any attribute of the object is accesssed.

This supports many object types. Some types it does not support are (at least not fully):
 * objects with a modified __getattribute__
 * objects with __slots__
 * objects with __value__
 * descriptor objects will not work as descriptors
 * buffer-able objects will not work with buffer

However, it does work on int, long, float, bool, complex, list, dict, tuple, set, and many others.
It even supports isinstance(x, int) and so forth.

The delayed object masks itself pretty well. However, type(x) will always report a type like
<class delayed.object> or <class loaded.list> (after it is loaded) even thouh x.__class__ reports
the underlying class). You can use delayed.unwrap(x) to unwrap an object (if it is not a delayed
object, it is returned as-is).

When creating the delayed object, you can also specify the excepted loaded type (which must be a
base class of the actual loaded type). This can help the object look more like the eventually loaded
object before it actually loads. For example, if you know it is a list, providing list will make
sure that d.__hash__ is set to None indicating it is unhashable. This is always taken care of once
the object is loaded.
"""

from __future__ import absolute_import

__all__ = ['delayed']

import math
import sys
import operator as op

__specials = set(('__new__', '__init__', '__class__', '__dict__', '__doc__',
                  '__module__', '__weakref__', '__subclasshook__'))

def __get_methods(clazz):
    """
    Get the magic method wrappers that for a specific class. This also adds the reflected methods
    if appropiate and possibly sets __hash__ to None.
    """
    cls = set(name for name in dir(clazz)
              if name[:2] == '__' and name[-2:] == '__')
    cls.update(('__getattr__', '__setattr__', '__delattr__'))
    methods = {name:meth for name, meth in __methods.iteritems()
               if name in cls}
    methods.update({name:meth for name, meth in __reflected.iteritems()
                    if name[:2]+name[3:] in cls})
    if '__hash__' in cls and clazz.__hash__ is None:
        methods['__hash__'] = None
    return methods

def delayed(load, base=object):
    """
    Create a delay-loaded object. Upon being used for the first time, the function `load` is called
    and this object will then act like the returned value.

    The `base` can be set to the known return value type (or a super-class of it) to make this
    object look more like the returned value before the value is loaded.
    """
    if not callable(load): raise TypeError('load must be callable')
    if not isinstance(base, type): raise TypeError('base must be a type')

    clazz = None
    def value(_self):
        """
        Get the underlying value of this object, possibly loading it if it hasn't been loaded yet.
        """
        try: val = load()
        except StandardError as ex: raise RuntimeError(ex)
        assert isinstance(val, base)
        for name in set(clazz.__dict__) - __specials:
            delattr(clazz, name)
        for name, meth in __get_methods(val.__class__).iteritems():
            setattr(clazz, name, meth)
        clazz.__name__ = "loaded."+val.__class__.__name__
        clazz.__value__ = property(lambda self: val)
        return val

    if base is object:
        methods = __methods.copy()
        methods.update(__reflected)
    else:
        methods = __get_methods(base)
    methods['__value__'] = property(value)
    methods['__class__'] = property(lambda self: self.__value__.__class__)
    clazz = type("delayed."+base.__name__, (object,), methods)
    return clazz()

def unwrap(obj):
    """Returns the underlying value for a delay-loaded object or returns the given object as-is"""
    return obj.__value__ if hasattr(obj, '__value__') else obj

delayed.unwrap = unwrap

def __direct(name, nargs):
    """Create a method wrapper for the function name that takes nargs argments."""
    if nargs == 0: return name, lambda self: getattr(self.__value__, name)()
    if nargs == 1: return name, lambda self, a: getattr(self.__value__, name)(a)
    if nargs == 2: return name, lambda self, a, b: getattr(self.__value__, name)(a, b)
    if nargs == 3: return name, lambda self, a, b, c: getattr(self.__value__, name)(a, b, c)
    return name, lambda self, *args: getattr(self.__value__, name)(*args)
def __unary_wrap(func):
    """Create a method wrapper for the unary operator defined by func."""
    return '__'+func.__name__+'__', lambda self: func(self.__value__)
def __unary_plus_wrap(func, nargs):
    """Create a method wrapper for the unary operator defined by func with nargs extra arguments."""
    name = '__'+func.__name__+'__'
    if nargs == 0: return name, lambda self: func(self.__value__)
    if nargs == 1: return name, lambda self, a: func(self.__value__, a)
    if nargs == 2: return name, lambda self, a, b: func(self.__value__, a, b)
    if nargs == 3: return name, lambda self, a, b, c: func(self.__value__, a, b, c)
    return name, lambda self, *args: func(self.__value__, *args)
def __binary_wrap(func):
    """Create a method wrapper for the binary operator defined by func."""
    return ('__'+func.__name__.rstrip('_')+'__',
            lambda self, other: func(self.__value__, unwrap(other)))
def __binary_wrap_ref(func):
    """Create a reflected method wrapper for the binary operator defined by func."""
    return ('__r'+func.__name__.rstrip('_')+'__',
            lambda self, other: func(unwrap(other), self.__value__))

__unary = [ # methods that only take self
    bool, int, op.index, float, complex, bytes, str, repr, oct, hex, dir, hash, len, iter, reversed,
    abs, op.pos, op.neg, op.invert, math.trunc, math.floor, math.ceil, # op.abs vs abs?
]
__unary_plus = [ # methods that take self and some other non-delayed argument
    (getattr, 1), (setattr, 2), (delattr, 1),
    (op.getitem, 1), (op.setitem, 2), (op.delitem, 1), (op.contains, 1),
    (op.getslice, 2), (op.setslice, 3), (op.delslice, 2),
]
__binary = [ # methods that take self and another possibly delayed object
    op.eq, op.ne, op.ge, op.gt, op.le, op.lt,
    op.iadd, op.isub, op.imul, op.ipow, op.itruediv, op.ifloordiv, op.imod,
    op.iand, op.ior, op.ixor, op.ilshift, op.irshift,
]
__binary_reflectable = [ # binary methods that also have reflected versions
    op.add, op.sub, op.mul, op.pow, op.truediv, op.floordiv, op.mod, divmod,
    op.and_, op.or_, op.xor, op.lshift, op.rshift,
]
__others = [ # and other functions that we won't wrap
    ('__round__', lambda self, ndigits=0: round(self.__value__, ndigits)),
    ('__sizeof__', lambda self: sys.getsizeof(self.__value__)),
    ('__call__', lambda self, *args, **kwargs: (self.__value__(*args, **kwargs))),
    ('__pow__',
     lambda self, other, modulus=None:
         op.pow(self.__value__, unwrap(other)) if modulus is None else
         pow(self.__value__, unwrap(other), modulus)),
    __direct('__format__', 1),
    __direct('__missing__', 1),
    __direct('__enter__', 0), __direct('__exit__', 3),
]
if sys.version_info[0] < 3: # handle Python 2 differences
    __unary += [long, unicode]
    __binary += [coerce, cmp, op.idiv]
    __binary_reflectable.append(op.div)
    __others.append(('__nonzero__', lambda self: bool(self.__value__)))

__methods = dict(__unary_wrap(func) for func in __unary)
__methods.update(__unary_plus_wrap(func, nargs) for func, nargs in __unary_plus)
__methods.update(__binary_wrap(func) for func in __binary)
__methods.update(__binary_wrap(func) for func in __binary_reflectable)
__methods.update(__others)
__reflected = dict(__binary_wrap_ref(f) for f in __binary_reflectable)

del __direct, __unary_wrap, __unary_plus_wrap, __binary_wrap, __binary_wrap_ref
del __unary, __unary_plus, __binary, __binary_reflectable, __others

if __name__ == "__main__":
    pass
