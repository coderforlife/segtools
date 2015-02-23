"""
Provides a collection of data-object wrappers to ease implementing them elsewhere. Unlike the real
list/dict objects, you can change the definition of functions (and for the most part only need to
change a few since they are implemented in terms of each other) and they are wrappers of the
underlying objects, which means that you can have a read-only view of a dictionary that stays
up-to-date if the real dictionary is modified somewhere else.

 * DictionaryWrapper extends dict and stores its actual dictionary in self._data.
 * DictionaryWrapperWithAttr extends DictionaryWrapper and adds using dictionary keys as attributes
   which is easier for the end-user but more difficult for the implementor. See notes in class for
   caveats.
 * ReadOnlyDictionaryWrapper extends DictionaryWrapper and throws exceptions in all mutating
   functions.
 * ListWrapper extends list and stores its actual list in self._data.
 * ReadOnlyListWrapper extends ListWrapper and and throws exceptions in all mutating functions.

As stated before, the wrappers try to re-use functions so that you only have to re-implement a
minimal set of functions to change behavior. The "raw" functions (ones that are directly implemented
and not implemented in terms of other functions). Below are listed the raw functions. For details on
which functions are derived from these functions, see the classes themselves.

For dictionary:
    __len__
    __getitem__
    __contains__
    __iter__
    itervalues
    __delitem__
    __setitem__
    __eq__
    copy

For list:
    __len__
    __getitem__
    __delitem__
    __setitem__
    insert
    extend
    __eq__
    __lt__

"""

#pylint: disable=too-few-public-methods,too-many-ancestors,super-init-not-called

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
from sys import maxsize
from itertools import izip
from numbers import Integral
from abc import ABCMeta, abstractmethod

__all__ = ['DictionaryWrapper', 'DictionaryWrapperWithAttr', 'ReadOnlyDictionaryWrapper',
           'ListWrapper', 'ReadOnlyListWrapper']

_marker = object() # unique object for detecting if argument is passed

class DictionaryWrapper(dict):
    # Raw Functions     Derived Functions
    #  __len__      <=
    #  __getitem__  <=           get, setdefault, pop, popitem
    #  __contains__ <=  has_key, get, setdefault, pop
    #  __iter__     <=  popitem, clear, keys, iterkeys, iteritems, items, __repr__
    #  itervalues   <=  values, iteritems, items, __repr__
    #
    #  __delitem__  <=  pop, popitem, clear
    #  __setitem__  <=  setdefault, update
    #
    #  __eq__       <=  __ne__
    # copy          <=

    _marker = _marker

    def __init__(self, d): self._data = d

    # Deleting functions
    def __delitem__(self, key): del self._data[key]
    def clear(self):
        for k in list(self): del self[k]
    def pop(self, key, default = _marker):
        if key in self:
            value = self[key]
            del self[key]
            return value
        elif default is _marker: raise KeyError
        return default
    def popitem(self):
        try: key = next(iter(self))
        except StopIteration: raise KeyError
        value = self[key]
        del self[key]
        return key, value

    # Setting/inserting functions
    def __setitem__(self, key, value): self._data[key] = value
    def setdefault(self, key, default=None):
        if key in self: return self[key]
        self[key] = default
        return default
    def update(self, *args, **kwargs):
        if len(args) == 1:
            other = args[0]
            if isinstance(other, collections.Mapping):
                try: other = other.iteritems()
                except AttributeError: other = ((k,other[k]) for k in other)
            for key, val in other: self[key] = val
        elif len(args) != 0: raise TypeError("update expected at most 1 positional arguments, got {}".format(len(args)+1))
        for key, val in kwargs.iteritems(): self[key] = val

    # Direct retrieval, iteration, and length
    def __getitem__(self, key): return self._data[key]
    def get(self, key, default=None): return self[key] if key in self else default
    def __contains__(self, key): return key in self._data    # as derived:  try: self[key]; except KeyError: return False; else: return True;
    def has_key(self, key): return key in self
    def __iter__(self): return iter(self._data)
    def __len__(self): return len(self._data)

    # Values, keys, and items
    def iterkeys(self):   return iter(self)
    def itervalues(self): return self._data.itervalues()       # as derived:  (self[k] for k in self)
    def iteritems(self):  return izip(self, self.itervalues()) # as derived:  ((k,self[k]) for k in self)
    def keys(self):   return list(self)
    def values(self): return list(self.itervalues())
    def items(self):  return zip(self, self.itervalues())
    def viewkeys(self): return _DictViewKeys(self)
    def viewvalues(self): return _DictViewValues(self)
    def viewitems(self): return _DictViewItems(self)

    # Comparison operators
    def __eq__(self, other): return self._data == other
    def __ne__(self, other): return not (self == other)
    # Other comparisons are meaningless so force them to not exist (they exist on dict but not MutableMapping)
    __lt__ = property()
    __le__ = property()
    __gt__ = property()
    __ge__ = property()

    # Other
    def copy(self): return self._data.copy()
    def __repr__(self): return '{' + (', '.join('%s: %s'%kv for kv in izip(self, self.itervalues()))) + '}'

class _DictView(object):
    __metaclass__ = ABCMeta
    def __init__(self, d): self._d = d
    def __len__(self):  return len(self._d)
    def __repr__(self): return "%s(%s)"%(type(self).__name__, repr(list(self)))
    @abstractmethod
    def __iter__(self): pass
    @abstractmethod
    def __contains__(self, item): pass

class _DictViewSetLike(_DictView):
    def __init__(self, d): super(_DictViewSetLike, self).__init__(d)
    @abstractmethod
    def __iter__(self): pass
    @abstractmethod
    def __contains__(self, item): pass

    @staticmethod
    def __all_contained_in(a, b):
        return all(x in b for x in a)
    def __eq__(self, other):
        if not isinstance(other, collections.Set): return NotImplemented
        return len(self) == len(other) and _DictViewSetLike.__all_contained_in(self, other)
    def __lt__(self, other):
        if not isinstance(other, collections.Set): return NotImplemented
        return len(self) < len(other) and _DictViewSetLike.__all_contained_in(self, other)
    def __le__(self, other):
        if not isinstance(other, collections.Set): return NotImplemented
        return len(self) <= len(other) and _DictViewSetLike.__all_contained_in(self, other)
    def __ne__(self, other): return not (self == other)
    def __gt__(self, other): return not (self <= other)
    def __ge__(self, other): return not (self < other)

    @staticmethod
    def __set_op(a, b, op):
        s = set(a)
        op(s, b)
        return s
    def __and__(self, other): return _DictViewSetLike.__set_op(self, other, set.intersection_update)
    def __or__(self, other): return _DictViewSetLike.__set_op(self, other, set.update)
    def __sub__(self, other): return _DictViewSetLike.__set_op(self, other, set.difference_update)
    def __xor__(self, other): return _DictViewSetLike.__set_op(self, other, set.symmetric_difference_update)

class _DictViewKeys(_DictViewSetLike, collections.KeysView):
    def __init__(self, d): super(_DictViewKeys, self).__init__(d)
    def __iter__(self): return self._d.iterkeys()
    def __contains__(self, key): return key in self._d

class _DictViewItems(_DictViewSetLike, collections.ItemsView):
    def __init__(self, d): super(_DictViewItems, self).__init__(d)
    def __iter__(self): return self._d.iteritems()
    def __contains__(self, item):
        if not isinstance(item, tuple) or len(item) != 2: return False
        key,value = item
        return key in self._d and self._d[key] == value

class _DictViewValues(_DictView, collections.ValuesView):
    def __init__(self, d): super(_DictViewValues, self).__init__(d)
    def __iter__(self): return self._d.itervalues()
    def __contains__(self, value): return any(x == value for x in self)

class DictionaryWrapperWithAttr(DictionaryWrapper):
    """
    Useful for dictionaries with string keys. Any dictionary keys that share names with instance
    attributes will not be accessible/settable/removable as attributes. Also, until something
    exists as an instance attribute it will try to use it as a key of the dictionary (this can be
    circumvented using self.__dict__['attr'] = x the first time).
    """
    _data = None
    def __init__(self, d): super(DictionaryWrapperWithAttr, self).__init__(d)
    def __setattr__(self, key, value):
        x = self
        while x != type:
            if key in x.__dict__: self.__dict__[key] = value; break
            x = x.__class__
        else:
            self[key] = value
    def __getattr__(self, key): return self[key]
    def __delattr__(self, key): del self[key]

class ReadOnlyDictionaryWrapper(DictionaryWrapper):
    def __init__(self, d): super(ReadOnlyDictionaryWrapper, self).__init__(d)

    # Mutating functions only raise errors
    def __delitem__(self, key): raise AttributeError('Illegal action for readonly dictionary')
    def clear(self): raise AttributeError('Illegal action for readonly dictionary')
    def pop(self, key, default = _marker): raise AttributeError('Illegal action for readonly dictionary')
    def popitem(self): raise AttributeError('Illegal action for readonly dictionary')
    def __setitem__(self, key, value): raise AttributeError('Illegal action for readonly dictionary')
    def setdefault(self, key, default=None): raise AttributeError('Illegal action for readonly dictionary')
    def update(self, *args, **kwargs): raise AttributeError('Illegal action for readonly dictionary')

class ListWrapper(list):
    # Raw Functions     Derived Functions
    #  __len__      <=                __iter__, __contains__, index, remove, count, reverse, __reversed__, __repr__, sort, append
    #  __getitem__  <=  __getslice__, __iter__, __contains__, index, remove, count, reverse, __reversed__, __repr__, sort, pop, __imul__, __mul__, __add__
    #
    #  __delitem__  <=  __delslice__, remove, pop, __imul__
    #  __setitem__  <=  __setslice__, reverse, sort
    #  insert       <=  append
    #  extend       <=  __iadd__, __imul__
    #
    #  __eq__       <=  __ne__, __le__, __gt__
    #  __lt__       <=  __le__, __ge__, __gt__


    def __init__(self, l): self._data = l

    # Slice functions are only defined to cover-up the ones defined in list
    # They can be completely ignored
    def __delslice__(self, i, j): del self[slice(i,j)]
    def __setslice__(self, i, j, value): self[slice(i,j)] = value
    def __getslice__(self, i, j): return self[slice(i,j)]

    # Deleting functions
    def __delitem__(self, i): del self._data[i]
    def remove(self, value):
        for i in xrange(len(self)):
            if self[i] == value: del self[i]; return
        raise ValueError('%s is not in list' % value)
    def pop(self, i=-1):
        if not isinstance(i, Integral): raise TypeError('an integer is required')
        x = self[i]
        del self[i]
        return x

    # Setting/inserting functions
    def __setitem__(self, i, value): self._data[i] = value
    def insert(self, i, value): self._data.insert(i, value)
    def extend(self, itr):      self._data.extend(itr)
    def append(self, value):    self.insert(len(self), value)
    def __iadd__(self, other):
        self.extend(other)
        return self
    def __imul__(self, n): # much slower than raw
        if not isinstance(n, Integral): raise TypeError('an integer is required')
        if n <= 0: del self[:] # become empty
        elif n == 1: return    # stay the same
        else:
            if hasattr(n, 'bit_length'):
                bl = n.bit_length() -1
            else:
                nx, bl = n, -1
                while nx: nx >>= 1; bl += 1
            for _ in xrange(bl): self.extend(self._data) # double the length a bunch of times (use _data here just to make sure it isn't using the iterable)
            n &= (1 << bl) - 1
            self.extend(self[:n]) # remainder
        return self

    # Rearranging functions
    def reverse(self): # much slower than raw
        half = len(self)//2
        self[:half], self[-half:] = self[-1:-half-1:-1], self[half-1::-1]
    def sort(self, cmp=None, key=None, reverse=False): #pylint: disable=redefined-builtin
        # much slower than raw
        if cmp is not None:
            if key is not None: raise ValueError('cannot specify both cmp and key')
            from functools import cmp_to_key
            key = cmp_to_key(cmp)
        decorated = [(self[i],i,self[i]) for i in xrange(len(self))] if key is None else [(key(self[i]),i,self[i]) for i in xrange(len(self))]
        decorated.sort(reverse=reverse)
        for i,(_,_,v) in enumerate(decorated): self[i] = v
        #swaps = [(old_i,new_i) for new_i,(k,old_i) in enumerate(keys)]
        # TODO: how to use swaps? is it more efficient?

    # Direct retrieval, iteration, and length
    def __len__(self):            return len(self._data)
    def __getitem__(self, i):     return self._data[i]
    def __iter__(self):
        for i in xrange(len(self)): yield self[i]
    def __contains__(self, value): return any(self[i] == value for i in xrange(len(self)))
    def index(self, value, start=0, stop=maxsize):
        if not isinstance(start, Integral) or not isinstance(stop, Integral): raise TypeError('an integer is required')
        for i in xrange(start, min(stop, len(self))):
            if self[i] == value: return i
        raise ValueError('%s is not in list' % value)
    def count(self, value): return sum(1 for i in xrange(len(self)) if self[i] == value) # much slower than raw

    # Addition and multiplication operators
    def __add__(self, other):
        lst = self[:]
        lst.extend(other)
        return lst
    def __mul__(self, n):
        if not isinstance(n, Integral): raise TypeError('an integer is required')
        if n <= 0: return self[0:0] # empty list of same type
        elif n == 1: return self[:] # copy
        else:
            lst = self[:]
            if hasattr(n, 'bit_length'):
                bl = n.bit_length() -1
            else:
                nx, bl = n, -1
                while nx: nx >>= 1; bl += 1
            for _ in xrange(bl): lst.extend(lst) # double the length a bunch of times
            n &= (1 << bl) - 1
            lst.extend(lst[:n]) # remainder
            return lst
    __rmul__ = __mul__ # exact same function

    # Comparison operators
    def __eq__(self, other): return self._data == other
    def __ne__(self, other): return not (self == other)
    def __lt__(self, other): return self._data < other
    def __le__(self, other): return (self < other) or (self == other)
    def __gt__(self, other): return not ((self < other) or (self == other))
    def __ge__(self, other): return not (self._data < other)

    # Other
    def __repr__(self): return '[' + (', '.join(str(self[i]) for i in xrange(len(self)))) + ']'
    def __reversed__(self):
        for i in xrange(len(self)-1, -1, -1): yield self[i]

class ReadOnlyListWrapper(ListWrapper):
    def __init__(self, l=None, *args): super(ReadOnlyListWrapper, self).__init__(l or args)

    # Mutating functions only raise errors
    def __delitem__(self, i): raise AttributeError('Illegal action for readonly list')
    def __setitem__(self, i, value): raise AttributeError('Illegal action for readonly list')
    def extend(self, itr): raise AttributeError('Illegal action for readonly list')
    def insert(self, i, value): raise AttributeError('Illegal action for readonly list')

    # Even though these are defined in terms of other mutating functions, they have the opportunity to quiettly do nothing in same cases (e.g. lst *= 1)
    def __imul__(self, n): raise AttributeError('Illegal action for readonly list')
    def reverse(self): raise AttributeError('Illegal action for readonly list')
    def sort(self, cmp=None, key=None, reverse=False): #pylint: disable=redefined-builtin
        raise AttributeError('Illegal action for readonly list')
