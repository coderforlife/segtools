"""Filtered Image Stack Classes."""

from abc import ABCMeta, abstractmethod

from ..source import ImageSource
from .._stack import ImageStack, ImageSlice

__all__ = ['FilteredImageStack','FilteredImageSlice',
           'UnchangingFilteredImageStack','UnchangingFilteredImageSlice',
           'FilterOption']

_filter_names = {}
_filter_flags = {}
def _get_filter_class(nf): return _filter_flags.get(nf, None) or _filter_names.get(nf.lower(), None)

class FilteredImageStack(ImageStack):
    __metaclass__ = ABCMeta

    @classmethod
    def _get_filter_class(cls, name_or_flag):
        fcls = _get_filter_class(name_or_flag)
        if fcls is not None: return fcls
        for fcls in cls._get_all_subclasses():
            n, f = fcls._name(), fcls._flags()
            if n is not None: _filter_names[n.lower()] = fcls
            if f is not None: _filter_flags.update({F:fcls for F in f})
        return _get_filter_class(name_or_flag)

    @classmethod
    def create(cls, ims, flt, *args, **kwargs):
        """
        Create a filtered image stack from the stack `ims` that uses the filter name `flt` passing
        in positional and keyword agruments. The filter name can be the case-insensitive name of the
        filter or a case-sensitive flag name (without leading -/--).
        """
        fcls = cls._get_filter_class(flt)
        if fcls is None: raise ValueError("'%s' is not a known filter" % flt)
        opts = fcls._opts()
        if len(opts) < len(args): raise ValueError()
        args = [opt.cast(val) for opt,val in zip(opts, args)] # positional arguments
        opts = {opt.name:opt for opt in opts[len(args):]} # remove the positional arguments
        kwargs = {key:opts[key].cast(val) for key,val in kwargs.iteritems()} # keyword arguments
        opts = [opt for opt in opts.itervalues() if opt.name not in kwargs] # remove the keyword arguments
        if any(opt.has_no_default for opt in opts): raise ValueError() # check all non-optional arguments
        for opt in opts: kwargs[opt.name] = opt.default # assign defaults to all optional arguments
        return fcls(ims, *args, **kwargs)

    @classmethod
    def parse_cmd_line(cls, cmd):
        """
        Parse a command line, a list of arguments, where the first item is the filter name (with or
        without -/--) and the positional and keyword arguments as the other items (keyword args use
        name=value).
        """
        flt, cmd = cmd[0].lstrip('-'), cmd[1:]
        fcls = cls._get_filter_class(flt)
        if fcls is None: raise ValueError("'%s' is not a known filter" % flt)
        flt = fcls._name()
        opts = fcls._opts()
        if len(opts) < len(cmd): raise ValueError("too many arguments for the '%s' filter" % flt)

        # Position arguments
        npos = next((i for i,x in enumerate(cmd) if '=' in x), len(cmd)) # number of positional arguments
        if any('=' not in x for x in cmd[npos:]): raise ValueError("cannot include non-named arguments after named arguments (ones that have '=' in them)")
        try: args = [opt.cast(val) for opt,val in zip(opts[:npos], cmd)]
        except: raise ValueError("argument '%s' of '%s' filter does not support value '%s'" % (opt.name, flt, val))

        # Named arguments
        opts = {opt.name:opt for opt in opts[npos:]} # remove the positional arguments
        try: kwargs = {key:opts.pop(key).cast(val) for key,val in (arg.split('=',1) for arg in cmd[npos:])}
        except KeyError: raise ValueError("invalid named argument '%s' of '%s' filter" % (key, flt))
        except ValueError, TypeError: raise ValueError("argument '%s' of '%s' filter does not support value '%s'" % (opt.name, flt, val))

        # Default Checks
        opts = [opt for opt in opts.itervalues() if opt.name not in kwargs] # remove the keyword arguments
        if any(opt.has_no_default for opt in opts): raise ValueError("value required for argument '%s' of '%s' filter" % (opt.name, flt)) # check all non-optional arguments
        for opt in opts: kwargs[opt.name] = opt.default # assign defaults to all optional arguments

        return flt, args, kwargs

    @classmethod
    def filter_names(cls):
        """Gets a list of all known filter names"""
        names = []
        for cls in cls._get_all_subclasses():
            n = cls._name()
            if n is not None: names.append(n)
        return names
    
    @classmethod
    def description(cls, name, width=None):
        """
        Gets a detailed description of a particular filter (from the name or a flag without leading
        -/--). If `width` is provided and not None then the text will be nicely wrapped to that
        width.
        """
        name = name.lower()
        for fcls in cls._get_all_subclasses():
            n = fcls._name()
            if n is not None and n.lower() == name: return fcls._description(width)
        return None

    @classmethod
    def _description(cls, width=None):
        """Generates a detailed description of a particular filter"""
        d = cls._desc()
        fs = cls._flags()
        os = cls._opts()
        ex = cls._example()
        if d is None or fs is None or len(fs) == 0: return None
        have_opts = os is not None and len(os) > 0
        if have_opts: opt_name_len = min(max(max(len(o.name) for o in os)+1, 8), 16)

        if width is not None:
            from textwrap import fill, TextWrapper
            flg_fill = TextWrapper(width=width, initial_indent='  ').fill
            opt_fill = TextWrapper(width=width, initial_indent='  ', subsequent_indent=' '*(2+opt_name_len)).fill
        else:
            fill     = lambda l,w: l
            flg_fill = lambda l,w: '  '+l
            opt_fill = lambda l,w: '  '+l
        
        s = "\n".join(fill(l,width) for l in d.splitlines())
        
        s += "\n\n"+fill("Supported flags:",width)+"\n"
        s += "\n".join(flg_fill(('-' if len(f) == 1 else '--') + f) for f in fs)
        
        if have_opts:
            s += "\n\nArguments:\n"
            s += "\n".join(opt_fill(o.name+(' '*max(opt_name_len-len(o.name), 1))+o.full_desc) for o in os)
            
        if ex is not None: s += "\n\n" + "\n".join(fill(l,width) for l in ex.splitlines())
        return s
    
    @classmethod
    def _name(cls):
        """The name of the filter"""
        return None
    @classmethod
    def _desc(cls):
        """The description of the filter"""
        return None
    @classmethod
    def _flags(cls):
        """List of acceptable flags to recognize the filter without - or --"""
        return None
    @classmethod
    def _opts(cls):
        """Optional list of FilterOption objects for the options/arguments of the filter"""
        return None
    @classmethod
    def _example(cls):
        """Optional text to display an example, written after the list of arguments"""
        return None
    @classmethod
    def _supported(cls, dtype):
        """Returns true if the given dtype is supported as an input image data type"""
        return False
    @abstractmethod
    def __str__(self):
        """Returns a short description of the filter given the arguments supplied to"""
        return None
    def __init__(self, ims, slices, *args, **kwargs):
        if isinstance(slices, type): slices = [slices(im,self,z,*args,**kwargs) for z,im in enumerate(ims)]
        super(FilteredImageStack, self).__init__(slices)
        self._ims = ImageStack.as_image_stack(ims)

class FilteredImageSlice(ImageSlice):
    def __init__(self, image, stack, z):
        super(FilteredImageSlice, self).__init__(stack, z)
        self._input = ImageSource.as_image_source(image)

class UnchangingFilteredImageStack(FilteredImageStack):
    def _get_homogeneous_info(self): return self._ims._get_homogeneous_info()

class UnchangingFilteredImageSlice(FilteredImageSlice):
    def __init__(self, image, stack, z):
        super(UnchangingFilteredImageSlice, self).__init__(image, stack, z)
        self._set_props(image.dtype, image.shape)
    def _get_props(self): pass

NoDefault = object()
class FilterOption:
    def __init__(self, name, desc, cast, default = NoDefault):
        """
        An option for a filter. The `name` and `desc` are for printing information about the option.
        The `cast` is a function that takes a single argument and returns an apporpiately casted
        value to pass to the filter constructor. It should be able to take strings and actual
        values, throwing a ValueError or TypeError if the cast cannot be completed. If the option
        is truly optional, then a default value is specified. This value is not sent through the
        cast function.
        """
        self._name = name
        self._desc = desc
        self._cast = cast
        self._def = default
    @property
    def name(self): return self._name
    @property
    def description(self): return self._desc
    @property
    def default(self): return self._def
    @property
    def has_no_default(self): return self._def is NoDefault
    @property
    def full_desc(self): return self.description+("" if self.has_no_default else " (default: "+str(self.default)+")")
    def cast(self, x): return self._cast(x)

    @staticmethod
    def cast_from_dict(d):
        def _cast_from_dict(x):
            try: return d[x]
            except KeyError: raise ValueError
        return _cast_from_dict
    @staticmethod
    def cast_from_list(l):
        def _cast_from_list(x):
            if x in l: return x
            raise ValueError
        return _cast_from_list
    @staticmethod
    def cast_int(pred=lambda x:True):
        def _cast_int(x):
            x = int(x)
            if not pred(x): raise ValueError
            return x
        return _cast_int
    @staticmethod
    def cast_float(pred=lambda x:True):
        def _cast_float(x):
            from math import isnan
            x = float(x)
            if isnan(x) or not pred(x): raise ValueError
            return x
        return _cast_float
