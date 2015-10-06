"""
Back ported from Python 3. Some features were lost including keeping names in
order and automatic checking for duplicate names. Also required was a trimmed
copy of DynamicClassAttribute (difficult to find) and a new MappingProxyType
(which I call ReadOnlyDictionaryWrapper) because in Python 3 that one simply
is the built-in dictproxy with an available constructor.
"""

# pylint: disable=protected-access, no-member

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from collections import Sequence
String = str if sys.version_info[0] == 3 else basestring

from .datawrapper import ReadOnlyDictionaryWrapper

__all__ = ['Enum', 'IntEnum', 'unique_enum_values', 'Flags', 'IntFlags']

class DynamicClassAttribute(object):
    """Route attribute access on a class to __getattr__.

    This is a descriptor, used to define attributes that act differently when
    accessed through an instance and through a class.  Instance access remains
    normal, but access to an attribute through a class will be routed to the
    class's __getattr__ method; this is done by raising AttributeError.

    This allows one to have properties active on an instance, and have virtual
    attributes on the class with the same name (see Enum for an example).
    """
    def __init__(self, fget=None):
        self.fget = fget
        self.__doc__ = fget.__doc__
    def __get__(self, instance, ownerclass=None):
        if instance is None or self.fget is None: raise AttributeError("unreadable attribute")
        return self.fget(instance)
    def __set__(self, instance, value): raise AttributeError("can't set attribute")
    def getter(self, fget): return type(self)(fget)

def _is_descriptor(obj):
    """Returns True if obj is a descriptor, False otherwise."""
    return (hasattr(obj, '__get__') or
            hasattr(obj, '__set__') or
            hasattr(obj, '__delete__'))

def _is_dunder(name):
    """Returns True if a __dunder__ name, False otherwise."""
    return (name[:2] == name[-2:] == '__' and
            name[2:3] != '_' and
            name[-3:-2] != '_' and
            len(name) > 4)

def _is_sunder(name):
    """Returns True if a _sunder_ name, False otherwise."""
    return (name[0] == name[-1] == '_' and
            name[1:2] != '_' and
            name[-2:-1] != '_' and
            len(name) > 2)

def _make_class_unpicklable(cls):
    """Make the given class un-picklable."""
    def _break_on_call_reduce(self):
        raise TypeError('%r cannot be pickled' % self)
    cls.__reduce__ = _break_on_call_reduce
    cls.__module__ = '<unknown>'

# Dummy value for Enum as EnumMeta explicitly checks for it, but of course
# until EnumMeta finishes running the first time the Enum class doesn't exist.
# This is also why there are checks in EnumMeta like `if Enum is not None`
Enum = None

class EnumMeta(type):
    """Metaclass for Enum"""
    def __new__(metacls, cls, bases, classdict): # pylint: disable=method-hidden
        # an Enum class is final once enumeration items have been defined; it
        # cannot be mixed with other types (int, float, etc.) if it has an
        # inherited __new__ unless a new __new__ is defined (or the resulting
        # class will fail).
        member_type, first_enum = metacls._get_mixins_(bases, Enum)
        __new__, save_new, use_args = metacls._find_new_(classdict, member_type,
                                                         first_enum, Enum)

        # save enum items into separate mapping so they don't get baked into
        # the new class
        members = {k:v for k,v in classdict.iteritems() if not _is_descriptor(v) and not _is_dunder(k) and not _is_sunder(k)}
        for name in members:
            del classdict[name]

        # check for illegal enum names (any others?)
        invalid_names = set(members) & {'mro', }
        if invalid_names:
            raise ValueError('Invalid enum member name: {0}'.format(
                ','.join(invalid_names)))

        # create our new Enum type
        enum_class = type.__new__(metacls, cls, bases, classdict)
        enum_class._member_names_ = []               # names in definition order
        enum_class._member_map_ = {}                 # name->value map
        enum_class._member_type_ = member_type

        # Reverse value->name map for hashable values.
        enum_class._value2member_map_ = {}

        # check for a __getnewargs__, and if not present sabotage
        # pickling, since it won't work anyway
        if member_type is not object and member_type.__dict__.get('__getnewargs__') is None:
            _make_class_unpicklable(enum_class)

        # instantiate them, checking for duplicates as we go
        # we instantiate first instead of checking for duplicates first in case
        # a custom __new__ is doing something funky with the values -- such as
        # auto-numbering ;)
        metacls._instantiate_members_(enum_class, members, member_type, __new__, use_args)

        # double check that repr and friends are not the mixin's or various
        # things break (such as pickle)
        for name in ('__repr__', '__str__', '__format__', '__getnewargs__'):
            class_method = getattr(enum_class, name)
            obj_method = getattr(member_type, name, None)
            enum_method = getattr(first_enum, name, None)
            if obj_method is not None and obj_method is class_method:
                setattr(enum_class, name, enum_method)

        # replace any other __new__ with our own (as long as Enum is not None,
        # anyway) -- again, this is to support pickle
        if Enum is not None:
            # if the user defined their own __new__, save it before it gets
            # clobbered in case they subclass later
            if save_new:
                enum_class.__new_member__ = __new__
            enum_class.__new__ = Enum.__new__
        return enum_class

    @staticmethod
    def _instantiate_members_(enum_class, members, member_type, __new__, use_args):
        func_type = type(lambda:None)
        for member_name,value in members.iteritems():
            if isinstance(value, func_type): continue
            args = (value,) if not isinstance(value, tuple) else value
            if member_type is tuple: # special case for tuple enums
                args = (args,)       # wrap it one more time
            if not use_args:
                enum_member = __new__(enum_class)
                if not hasattr(enum_member, '_value_'):
                    enum_member._value_ = value
            else:
                enum_member = __new__(enum_class, *args)
                if not hasattr(enum_member, '_value_'):
                    enum_member._value_ = member_type(*args)
            value = enum_member._value_
            enum_member._name_ = member_name
            enum_member.__objclass__ = enum_class
            enum_member.__init__(*args)
            # If another member with the same value was already defined, the
            # new member becomes an alias to the existing one.
            for canonical_member in enum_class._member_map_.itervalues():
                if canonical_member.value == enum_member._value_:
                    enum_member = canonical_member
                    break
            else:
                # Aliases don't appear in member names (only in __members__).
                enum_class._member_names_.append(member_name)
            enum_class._member_map_[member_name] = enum_member
            try:
                # This may fail if value is not hashable. We can't add the value
                # to the map, and by-value lookups for this value will be
                # linear.
                enum_class._value2member_map_[value] = enum_member
            except TypeError:
                pass

    def __call__(cls, value, names=None, module=None, typ=None):
        """Either returns an existing member, or creates a new enum class.

        This method is used both when an enum class is given a value to match
        to an enumeration member (i.e. Color(3)) and for the functional API
        (i.e. Color = Enum('Color', names='red green blue')).

        When used for the functional API: `module`, if set, will be stored in
        the new class' __module__ attribute; `type`, if set, will be mixed in
        as the first base class.

        Note: if `module` is not set this routine will attempt to discover the
        calling module by walking the frame stack; if this is unsuccessful
        the resulting class will not be pickleable.

        """
        if names is None:  # simple value lookup
            return cls._member_map_[value] if value in cls._member_map_ else cls._value2member_map_[value]
        # otherwise, functional API: we're creating a new Enum type
        return cls._create_(value, names, module=module, typ=typ)

    def __contains__(cls, member):
        return isinstance(member, cls) and member.name in cls._member_map_ or member in cls._value2member_map_

    def __delattr__(cls, attr):
        # nicer error message when someone tries to delete an attribute
        # (see issue19025).
        if attr in cls._member_map_:
            raise AttributeError("%s: cannot delete Enum member." % cls.__name__)
        super(EnumMeta, cls).__delattr__(attr)

    def __dir__(self):
        return (['__class__', '__doc__', '__members__', '__module__'] + self._member_names_)

    def __getattr__(cls, name):
        """Return the enum member matching `name`

        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.

        """
        if _is_dunder(name):
            raise AttributeError(name)
        try:
            return cls._member_map_[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(cls, name):
        return cls._member_map_[name]

    def __iter__(cls):
        return (cls._member_map_[name] for name in cls._member_names_)

    def __len__(cls):
        return len(cls._member_names_)

    @property
    def __members__(cls):
        """Returns a mapping of member name->value.

        This mapping lists all enum members, including aliases. Note that this
        is a read-only view of the internal mapping.

        """
        return ReadOnlyDictionaryWrapper(cls._member_map_)

    def __repr__(cls):
        return "<enum %r>" % cls.__name__

    def __reversed__(cls):
        return (cls._member_map_[name] for name in reversed(cls._member_names_))

    def __setattr__(cls, name, value):
        """Block attempts to reassign Enum members.

        A simple assignment to the class namespace only changes one of the
        several possible ways to get an Enum member from the Enum class,
        resulting in an inconsistent Enumeration.

        """
        member_map = cls.__dict__.get('_member_map_', {})
        if name in member_map:
            raise AttributeError('Cannot reassign members.')
        super(EnumMeta, cls).__setattr__(name, value)

    def _create_(cls, class_name, names=None, module=None, typ=None):
        """Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are auto-numbered from 1.
        * An iterable of member names.  Values are auto-numbered from 1.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value.

        """
        metacls = cls.__class__
        bases = (cls, ) if typ is None else (typ, cls)
        classdict = dict()

        # special processing needed for names?
        if isinstance(names, String):
            names = names.replace(',', ' ').split()
        if isinstance(names, Sequence) and isinstance(names[0], String):
            names = [(e, i) for (i, e) in enumerate(names, 1)]

        # Here, names is either an iterable of (name, value) or a mapping.
        for item in names:
            if isinstance(item, String):
                member_name, member_value = item, names[item]
            else:
                member_name, member_value = item
            classdict[member_name] = member_value
        enum_class = metacls.__new__(metacls, str(class_name), bases, classdict)

        # TODO: replace the frame hack if a blessed way to know the calling
        # module is ever developed
        if module is None:
            try:
                from sys import _getframe
                module = _getframe(2).f_globals['__name__']
            except (AttributeError, ValueError):
                pass
        if module is None:
            _make_class_unpicklable(enum_class)
        else:
            enum_class.__module__ = module # pylint: disable=attribute-defined-outside-init

        return enum_class

    @staticmethod
    def _get_mixins_(bases, base_class):
        """Returns the type for creating enum members, and the first inherited
        enum class.

        bases: the tuple of bases that was given to __new__
        base_class: Enum or Flags

        """
        if not bases or base_class is None:
            return object, base_class

        # double check that we are not subclassing a class with existing
        # enumeration members; while we're at it, see if any other data
        # type has been mixed in so we can use the correct __new__
        member_type = first_enum = None
        if any(base is not base_class and issubclass(base, base_class) and base._member_names_
               for base in bases): raise TypeError("Cannot extend enumerations")
        # base is now the last base in bases
        if not issubclass(bases[-1], base_class):
            raise TypeError("new enumerations must be created as "
                            "`ClassName([mixin_type,] enum_type)`")

        # get correct mix-in type (either mix-in type of Enum subclass, or
        # first base if last base is Enum)
        if not issubclass(bases[0], base_class):
            member_type = bases[0]  # first data type
            first_enum = bases[-1]  # enum type
        else:
            for base in bases[0].__mro__:
                # most common: (IntEnum, IntFlags, int, Enum, Flags, object)
                # possible:    (<Enum 'AutoIntEnum'>, <Enum 'IntEnum'>, (or Flags variants)
                #               <class 'int'>, <Enum 'Enum'>, <Flags 'Flags'>,
                #               <class 'object'>)
                if issubclass(base, base_class):
                    if first_enum is None:
                        first_enum = base
                else:
                    if member_type is None:
                        member_type = base

        return member_type, first_enum

    @staticmethod
    def _find_new_(classdict, member_type, first_enum, base_type):
        """Returns the __new__ to be used for creating the enum members.

        classdict: the class dictionary given to __new__
        member_type: the data type whose __new__ will be used by default
        first_enum: enumeration to check for an overriding __new__
        base_type: Enum or Flags

        """
        # now find the correct __new__, checking to see of one was defined
        # by the user; also check earlier enum classes in case a __new__ was
        # saved as __new_member__
        __new__ = classdict.get('__new__', None)

        # should __new__ be saved as __new_member__ later?
        save_new = __new__ is not None

        if __new__ is None:
            # check all possibles for __new_member__ before falling back to
            # __new__
            for method in ('__new_member__', '__new__'):
                for possible in (member_type, first_enum):
                    target = getattr(possible, method, None)
                    if target not in {
                            None,
                            None.__new__,
                            object.__new__,
                            base_type.__new__}:
                        __new__ = target
                        break
                if __new__ is not None:
                    break
            else:
                __new__ = object.__new__

        # if a non-object.__new__ is used then whatever value/tuple was
        # assigned to the enum member name will be passed to __new__ and to the
        # new enum member's __init__
        if __new__ is object.__new__:
            use_args = False
        else:
            use_args = True

        return __new__, save_new, use_args


class Enum: # pylint: disable=function-redefined, no-init
    __metaclass__ = EnumMeta
    """Generic enumeration.

    Derive from this class to define new enumerations.

    """
    def __new__(cls, value):
        # all enum instances are actually created during class construction
        # without calling this method; this method is called by the metaclass'
        # __call__ (i.e. Color(3) ), and by pickle
        if type(value) is cls: #pylint: disable=unidiomatic-typecheck
            # For lookups like Color(Color.red)
            return value
        # by-value search for a matching enum member
        # see if it's in the reverse mapping (for hashable values)
        try:
            if value in cls._value2member_map_:
                return cls._value2member_map_[value]
        except TypeError:
            # not there, now do long search -- O(n) behavior
            for member in cls._member_map_.values():
                if member.value == value:
                    return member
        raise ValueError("%s is not a valid %s" % (value, cls.__name__))

    def __repr__(self):
        return "<%s.%s: %r>" % (self.__class__.__name__, self._name_, self._value_)

    def __str__(self):
        return "%s.%s" % (self.__class__.__name__, self._name_)

    def __dir__(self):
        added_behavior = [m for m in self.__class__.__dict__ if m[0] != '_']
        return (['__class__', '__doc__', '__module__', 'name', 'value'] + added_behavior)

    def __format__(self, format_spec):
        # mixed-in Enums should use the mixed-in type's __format__, otherwise
        # we can get strange results with the Enum name showing up instead of
        # the value

        # pure Enum branch
        if self._member_type_ is object:
            cls = str
            val = str(self)
        # mix-in branch
        else:
            cls = self._member_type_
            val = self.value
        return cls.__format__(val, format_spec)

    def __getnewargs__(self):
        return (self._value_, )

    def __hash__(self):
        return hash(self._name_)

    # DynamicClassAttribute is used to provide access to the `name` and
    # `value` properties of enum members while keeping some measure of
    # protection from modification, while still allowing for an enumeration
    # to have members named `name` and `value`.  This works because enumeration
    # members are not set directly on the enum class -- __getattr__ is
    # used to look them up.

    @DynamicClassAttribute
    def name(self):
        """The name of the Enum member."""
        return self._name_

    @DynamicClassAttribute
    def value(self):
        """The value of the Enum member."""
        return self._value_


class IntEnum(int, Enum):
    """Enum where members are also (and must be) ints"""


def unique_enum_values(enumeration):
    """Class decorator for enumerations ensuring unique member values."""
    duplicates = []
    for name, member in enumeration.__members__.items():
        if name != member.name:
            duplicates.append((name, member.name))
    if duplicates:
        alias_details = ', '.join(["%s -> %s" % (alias, name) for (alias, name) in duplicates])
        raise ValueError('duplicate values found in %r: %s' % (enumeration, alias_details))
    return enumeration



# Derived from Enum. It is a collection of flags. The flags required being mixed
# in with an Integral type that is hashable. Flags support all the features of
# Enum with the additional features of bitwise manipulation and checking and
# implicit values for bitwise combinations that are not explicity declared.

from numbers import Integral
#from math import floor, log

def _get_bits_set(i):
    if i < 0:
        # Doesn't have an actual length so there is really nothing we can do
        # hiBit = i.bit_length() if hasattr(i, 'bit_length') else floor(log(i, 2))
        raise ValueError('Negative values are not supported. You may have to use a larger underlying type to support values that utilize the highest bit of the type')
    else:
        p = 0
        l = []
        while i:
            if i & 1: l.append(p)
            i >>= 1
            p += 1
        return frozenset(l)

# Dummy value for Flags as FlagsMeta explicitly checks for it, but of course
# until FlagsMeta finishes running the first time the Flags class doesn't exist.
# This is also why there are checks in FlagsMeta like `if Flags is not None`
Flags = None

class FlagsMeta(EnumMeta):
    """Metaclass for Flags"""
    def __new__(metacls, cls, bases, classdict): # pylint: disable=method-hidden
        # a Flags class is final once flag items have been defined; it
        # cannot be mixed with other types (int, float, etc.) if it has an
        # inherited __new__ unless a new __new__ is defined (or the resulting
        # class will fail).
        member_type, first_flags = metacls._get_mixins_(bases, Flags)
        __new__, _, _ = metacls._find_new_(classdict, member_type, first_flags, Flags)

        # save flags items into separate mapping so they don't get baked into
        # the new class
        members = {k:v for k,v in classdict.iteritems() if not _is_descriptor(v) and not _is_dunder(k) and not _is_sunder(k)}
        for name in members:
            del classdict[name]

        # check for illegal flag names (any others?)
        invalid_names = set(members) & {'mro', }
        if invalid_names:
            raise ValueError('Invalid flag member name: {0}'.format(
                ','.join(invalid_names)))
        if any(m.find('|') >= 0 or m.find('~') >= 0 for m in members):
            raise ValueError('Invalid flag member name: {0}'.format(
                ','.join(m for m in members if m.find('|') >= 0 or m.find('~') >= 0)))

        # create our new Flags type
        flags_class = type.__new__(metacls, cls, bases, classdict)
        flags_class._member_names_ = []               # names in definition order
        flags_class._member_map_ = {}                 # name->value map
        flags_class._no_bits_ = None                  # item that has no bits set
        flags_class._member_type_ = member_type
        flags_class._mask_ = member_type(0) if member_type is not object else None

        # Reverse value->name map for hashable values.
        flags_class._value2member_map_ = {}

        # replace any other __new__ with our own (as long as Flags is not None,
        # anyway) -- again, this is to support pickle
        if Flags is not None:
            flags_class.__new_member__ = __new__
            flags_class.__new__ = Flags.__new__

        # check for a __getnewargs__, and if not present sabotage
        # pickling, since it won't work anyway
        if (member_type is not object and member_type.__dict__.get('__getnewargs__') is None):
            _make_class_unpicklable(flags_class)

        # instantiate them, checking for duplicates as we go
        # we instantiate first instead of checking for duplicates first in case
        # a custom __new__ is doing something funky with the values -- such as
        # auto-numbering ;)
        for member_name,value in members.iteritems():
            flags_member = metacls._create_member_(member_name, value, flags_class, True)
            if len(flags_member._bits_set_) == 0: flags_class._no_bits_ = flags_member
            else:                                 flags_class._mask_ |= flags_member._value_
        if member_type is not object and flags_class._no_bits_ is None:
            name = next((name for name in ('None','NoFlags','NoneSet','None_','_None')
                         if name not in members), None)
            if name is not None:
                flags_class._no_bits_ = metacls._create_member_(name, 0, flags_class, False)

        # double check that repr and friends are not the mixin's or various
        # things break (such as pickle)
        for name in ('__repr__', '__str__', '__format__', '__getnewargs__',
                     '__contains__', '__invert__',
                     '__and__', '__or__', '__xor__',
                     '__rand__', '__ror__', '__rxor__'):
            class_method = getattr(flags_class, name)
            obj_method = getattr(member_type, name, None)
            flags_method = getattr(first_flags, name, None)
            if obj_method is not None and obj_method is class_method:
                setattr(flags_class, name, flags_method)

        return flags_class

    def __call__(cls, value, names=None, module=None, typ=None):
        """Either returns an existing member, or creates a new flags class.

        This method is used both when a flags class is given a value to match
        to an flags member (i.e. Color(3)) and for the functional API
        (i.e. Color = Flags('Color', names='red green blue')).

        When matching, it we will look for possible divided matches (for example
        the value 3 might be divided into 1 and 2 if they exist).

        When used for the functional API: `module`, if set, will be stored in
        the new class' __module__ attribute; `type`, if set, will be mixed in
        as the first base class.

        Note: if `module` is not set this routine will attempt to discover the
        calling module by walking the frame stack; if this is unsuccessful
        the resulting class will not be pickleable.

        """

        if names is None: # simple value lookup
            if isinstance(value, Integral):
                # Check if we have an exact match
                if value in cls._value2member_map_: return cls._value2member_map_[value]

                # Make sure we have a valid solution
                if (value & cls._mask_) != value: raise ValueError('Value does not exist in %s' % cls.__name__)
                bits_set = _get_bits_set(value)
                sets = {f._bits_set_:f for f in cls._member_map_.itervalues() if f._bits_set_.issubset(bits_set)} # set(...)->Flags obj
                if not bits_set.issubset(set().union(*sets)): raise ValueError('Value does not exist in %s' % cls.__name__)

                ### Calculate flags ###
                flags = {}
                missing = set()
                # Start with one-sized sets
                for bs in bits_set:
                    f = sets.pop(frozenset((bs,)), None)
                    if f is not None: flags[f._bits_set_] = f
                    else:             missing.add(bs)
                # While there are still some missing, move up in sizes
                while len(missing) > 0:
                    # Update sets to only those with having at least one vertex in missing
                    sets = {fbs:f for fbs,f in sets.iteritems() if not fbs.isdisjoint(missing)}
                    # Select the flag that has the least number of vertices then the most in missing
                    fbs = min(sets, key=lambda fbs: (len(fbs), len(fbs-missing)))
                    # Update missing
                    missing -= fbs
                    # Remove flags from the selected set if they are covered by the addition of the new edge
                    for gbs in flags.keys():
                        if gbs.issubset(fbs.union(*(flags.viewkeys()-set((gbs,))))):
                            del flags[gbs]
                    # Add the new flag to the set
                    flags[fbs] = sets[fbs]
                # Create the new member
                return FlagsMeta._create_member_('|'.join(f._name_ for f in flags.itervalues()), cls._member_type_(value), cls, False)
            else:
                return cls._member_map_[value]
        else:
            # otherwise, functional API: we're creating a new Flags type
            return cls._create_(value, names, module=module, typ=typ)
    def __contains__(cls, member): return isinstance(member, cls) and member.name in cls._member_map_ or (member & cls._mask_) == member
    def __dir__(self): return ['__mask__'] + super(FlagsMeta, self).__dir__()

    @property
    def __mask__(cls):
        """Returns a bit mask for this set of Flags."""
        return cls._mask_
    def __repr__(cls): return "<flags %r>" % cls.__name__

    @staticmethod
    def _create_member_(member_name, value, flags_class, visible):
        __new__ = flags_class.__new_member__
        if not isinstance(value, tuple):
            args = (value, )
        else:
            args = value
        if flags_class._member_type_ is tuple:   # special case for tuple flags
            args = (args, )                      # wrap it one more time
        if __new__ is object.__new__:
            flags_member = __new__(flags_class)
            if not hasattr(flags_member, '_value_'):
                flags_member._value_ = value
        else:
            flags_member = __new__(flags_class, *args)
            if not hasattr(flags_member, '_value_'):
                flags_member._value_ = flags_class._member_type_(*args)
        flags_member._name_ = member_name
        flags_member._bits_set_ = _get_bits_set(flags_member._value_)
        flags_member.__objclass__ = flags_class
        flags_member.__init__(*args)
        # If another member with the same value was already defined, the
        # new member becomes an alias to the existing one.
        for canonical_member in flags_class._member_map_.itervalues():
            if canonical_member.value == flags_member._value_:
                flags_member = canonical_member
                break
        else: # Aliases don't appear in member names (only in __members__).
            if visible: flags_class._member_names_.append(member_name)
        flags_class._member_map_[member_name] = flags_member
        flags_class._value2member_map_[value] = flags_member
        return flags_member

class Flags: # pylint: disable=function-redefined, no-init
    __metaclass__ = FlagsMeta
    """Generic set of flags.

    Derive from this class to define new flag sets."""
    def __new__(cls, value):
        if type(value) is cls: return value #pylint: disable=unidiomatic-typecheck
        if value in cls._value2member_map_: return cls._value2member_map_[value]
        raise ValueError("%s is not a valid %s" % (value, cls.__name__))
    def __repr__(self): return "<%s.%s: %r>" % (self.__class__.__name__, self._name_, self._value_)
    def __str__(self): return "%s.%s" % (self.__class__.__name__, self._name_)
    def __dir__(self): return ['__class__', '__doc__', '__module__', 'name', 'value'] + [m for m in self.__class__.__dict__ if m[0] != '_']
    def __format__(self, format_spec):
        if self._member_type_ is object:
            cls = str
            val = str(self)
        else:
            cls = self._member_type_
            val = self.value
        return cls.__format__(val, format_spec)
    def __getnewargs__(self): return (self._value_, )
    def __hash__(self): return hash(self._name_)
    def __and__ (self, other): return type(self)(self._value_ & other._value_) if isinstance(other, Flags) else type(self)(self._value_ & other)
    def __or__  (self, other): return type(self)(self._value_ | other._value_) if isinstance(other, Flags) else type(self)(self._value_ | other)
    def __xor__ (self, other): return type(self)(self._value_ ^ other._value_) if isinstance(other, Flags) else type(self)(self._value_ ^ other)
    def __rand__(self, other): return type(self)(other._value_ & self._value_) if isinstance(other, Flags) else type(self)(other & self._value_)
    def __ror__ (self, other): return type(self)(other._value_ | self._value_) if isinstance(other, Flags) else type(self)(other | self._value_)
    def __rxor__(self, other): return type(self)(other._value_ ^ self._value_) if isinstance(other, Flags) else type(self)(other ^ self._value_)
    def __invert__(self): return type(self)(~self._value_ & self._mask_)
    def __contains__(self, value): return (self & value) == value

    @DynamicClassAttribute
    def name(self):
        """The name of the Flags member."""
        return self._name_

    @DynamicClassAttribute
    def value(self):
        """The value of the Flags member."""
        return self._value_


class IntFlags(int, Flags):
    """Flags where members are also (and must be) ints"""
