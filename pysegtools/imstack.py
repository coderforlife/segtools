"""
Command line program to convert an image stack to another stack by processing each slice. This is
designed as a modular system. To create a new command for the program simply implement Command or
CommandEasy. For non-command help topics (for general things like colors and image types) you call
Help.register. The Help class can also be instantiated for a help-printing utility class. The Opt
class is useful for dealing with command lines as there are functions for using them to
automatically parse the command line or format help.
"""

__all__ = ["main","Help","Opt","Command","CommandEasy"]

from abc import ABCMeta, abstractmethod
from numbers import Integral
from collections import Sequence
from textwrap import TextWrapper
from .general.utils import get_terminal_width
out_width = max(get_terminal_width()-1,24)
fill = TextWrapper(width=out_width).fill
stack_status_fill = TextWrapper(width=out_width,subsequent_indent=2).fill

class Stack:
    """
    The stack of image stacks that commands consume from and produce to. In general a command should
    only ever need pop and push, and possible len(...) and empty. The select and remove functions'
    are meant for the internal "select" and "remove" commands.
    """
    def __init__(self, verbose=False):
        self._stack = []
        self._verbose = verbose
    @property
    def empty(self): return len(self._stack) == 0
    def __len__(self): return len(self._stack)
    def pop(self):
        if len(self._stack) == 0: raise ValueError('No image stack to consume')
        ims = self._stack.pop()
        if self._verbose: print stack_status_fill("- Consuming image stack '%s'"%ims)
        return ims
    def push(self, ims):
        from .images import ImageStack
        if not isinstance(ims, ImageStack): raise ValueError('Illegal call to Stack.push')
        if self._verbose:
            print stack_status_fill("+ Produced image stack '%s'"%ims)
            ims.print_detailed_info()
            print "-"*out_width
        self._stack.append(ims)
    def _inv_inds(self, inds):
        inds,l = [(-i-1) for i in inds], len(self._stack) # TODO: resolve negative values
        if any(not (-l<=i<l) for i in inds): raise ValueError('No image stack #%d', i)
        return inds
    def select(self, inds):
        inds = self._inv_inds(inds)
        ims = [self._stack[i] for i in inds]
        if self._verbose:
            for i in ims: print stack_status_fill("* Moving image stack '%s' to top"%i)
        for i in sorted(set(inds), reverse=True): del self._stack[i]
        self._stack.extend(ims)
    def remove(self, inds):
        inds = list(sorted(set(self._inv_inds(inds)), reverse=True))
        if self._verbose:
            for i in inds: print stack_status_fill("* Removing image stack '%s'"%self._stack[i])
        for i in inds: del self._stack[i]
        
class PseudoStack(Stack):
    """
    A class that almost looks like Stack except that it doesn't actually save image stacks (and
    doesn't accept or return them). However it does record the number of items on the stack and
    checks to make sure the stack is never under-flowed.
    """
    def __init__(self, verbose=False): self._stack_count = 0
    @property
    def empty(self): return self._stack_count == 0
    def __len__(self): return self._stack_count
    def pop(self):
        if self._stack_count == 0: raise ValueError('No image stack to consume', i)
        self._stack_count -= 1
    def push(self, ims=None): self._stack_count += 1
    def _inv_inds(self, inds):
        inds,l = [(-i-1) for i in inds], self._stack_count # TODO: resolve negative values
        if any(not (-l<=i<l) for i in inds): raise ValueError('No image stack #%d', i)
        return inds
    def select(self, inds):
        inds = self._inv_inds(inds)
        self._stack_count += len(inds) - len(set(inds))
    def remove(self, inds):
        inds = self._inv_inds(inds)
        self._stack_count -= len(set(inds))

    
class Args:
    """
    A collection of command arguments which can include both positional and named arguments. This
    class acts mostly like a dictionary except the keys can be ints (for positional arguments),
    strings (for named arguments), or the preferred tuple-of-int-and-string for an argument that
    is possibly both positional and named.

    Getting and deleting items also accept slices but this will only effect positional arguments.
    This is useful if you have a variable number of positional arguments first and want to remove
    them before analyzing the remainder of the arguments.
    """
    def __init__(self, args):
        self._cmd = args[0]
        args = args[1:]
        npos = next((i for i,x in enumerate(args) if '=' in x), len(args)) # number of positional arguments
        if any('=' not in x for x in args[npos:]): raise ValueError("Cannot include positional arguments after named arguments (ones that have '=' in them)")
        self._kwargs = {key:val for key,val in (arg.split('=',1) for arg in args[npos:])}
        if len(self._kwargs) != len(args)-npos: raise ValueError("Multiple values given for a single named argument")
        self._args = args[:npos]
    @property
    def cmd(self): return self._cmd
    def __get(self, key):
        """
        Looks up a key, which is either an integer, string, or tuple or int and string. If the key
        is found, then the "translated" key and value are returned (key is no longer a tuple). If
        it is not found then None and the "translated" key are returned (notice that the key has
        changed position!).
        """
        if   isinstance(key, Integral)   and key < len(self._args): return key, self._args[key]
        elif isinstance(key, basestring) and key in self.kwargs:    return key, self._kwargs[key]
        elif isinstance(key, Sequence) and len(key) == 2 and isinstance(key[0], Integral) and isinstance(key[1], basestring):
            pos, name = key
            possed, named = pos < len(self._args), name in self._kwargs
            if possed != named: return pos, self._args[possed] if possed else name, self._kwargs[name]
            if possed and named: raise ValueError('Argument "%s" was provided as both a positional and named argument' % name)
            return None, name
        return None, key
    def __getitem__(self, key):
        if isinstance(key, slice): return self._args[key]
        key, val = self.__get(key)
        if key is None: raise KeyError('Value required for argument "%s"' % val)
        return val
    def get(self, key, default=None):
        key, val = self.__get(key)
        return default if key is None else val
    def __get_all(self, *opts):
        """
        Get all arguments as an iterator of key,val pairs like __get would return. Performs numerous
        checks on the options/arguments.
        """
        if len(self._args)+len(self._kwargs) > len(opts): raise ValueError('Too many arguments given')
        names = [o.name for o in opts]
        bad = self._kwargs.viewkeys() - set(names)
        if len(bad) > 0: raise ValueError('Invalid named argument "%s"', bad[0])
        bad = self._kwargs.viewkeys() & set(names[:len(self._args)])
        if len(bad) > 0: raise ValueError('Argument "%s" was given both as positional and named' % bad[0])
        return (self.__get((i,o.name))+(o,) for i,o in enumerate(opts))
    def get_all(self, *opts):
        """
        Gets the options (as Opt objects) listed from the arguments and keyword arguments. Their
        order determines the order in the positional list they are looked for. This returns the
        values in the same order requested. The values are casted. This assumes that all arguments
        should fit into one of the options and raises exceptions if there are extra values.
        """
        return [o.default if key is None else o.cast(val) for key,val,o in self.__get_all(*opts)]
    def get_all_kw(self, *opts):
        """Like get_all except it returns a dictionary of name:value pairs."""
        return {o.name:(o.default if key is None else o.cast(val)) for key,val in self.__get_all(*opts)}
    def __contains__(self, key): return self.__get(key)[0] is not None
    def has_key(self, key): return key in self
    def __len__(self): return len(self._args)+len(self._kwargs)
    def __delitem__(self, key):
        if isinstance(key, slice): del self._args[key]
        else:
            key,val = self.__get(key)
            if key is None: raise KeyError(key)
            if isinstance(key, basestring): del self._kwargs[key]
            else: del self._args[key]
    def clear(self):
        del self._args[:]
        self._kwargs.clear()
    @property
    def positional(self): return self._args
    @property
    def named(self):      return self._kwargs
    def __iter__(self):
        for x in xrange(len(self._args)): yield x
        for x in self._kwargs:            yield x
    def iterkeys(self):   return iter(self)
    def itervalues(self):
        for x in self._args:                yield x
        for x in self._kwargs.itervalues(): yield x
    def iteritems(self):
        for x in enumerate(self._args):    yield x
        for x in self._kwargs.iteritems(): yield x
    def keys(self):   return range(len(self._args)) + self._kwargs.keys()
    def values(self): return self._args + self._kwargs.values()
    def items(self):  return list(enumerate(self._args)) + self._kwargs.items()


NoDefault = object()
class Opt:
    """
    An option for a command. The `name` is used to look for named arguments. The `desc` is for
    printing information about the option. The `cast` is a function that takes a single argument
    and returns an apporpiately casted value, it should be able to take strings and actual values,
    throwing a ValueError or TypeError if the cast cannot be completed. If the option is truly
    optional, then a default value is specified. This value is not sent through the cast function.

    Also included in this class are various general casting functions. They all return the actual
    casting function, possibly customized for a particular purpose.
    """
    def __init__(self, name, desc, cast, default = NoDefault):
        self._name = name
        self._desc = desc
        self._cast = cast
        self._def = default
    @property
    def name(self): return self._name
    @property
    def description(self): return self._desc
    @property
    def default(self):
        if self._def is NoDefault: raise ValueError('Value required for "%s"' % self._name)
        return self._def
    @property
    def has_default(self): return self._def is not NoDefault
    @property
    def full_desc(self): return self.description+("" if self._def is NoDefault else " (default: "+str(self.default)+")")
    def cast(self, x): return self._cast(x)

    # Various casting functions
    @staticmethod
    def cast_or(*casts):
        """
        Takes other casting functions and calls them in order until one does not raise a ValueError
        or TypeError and returns that value. If all raise errors, a ValueError is raised.
        """
        def _cast_or(x):
            for c in casts:
                try: return c(x)
                except (ValueError, TypeError): pass
            raise ValueError
        return _cast_or
    @staticmethod
    def cast_check(pred):
        """
        Sends the value to a predicate to be checked. If the predicate returns False, a ValueError
        is raised, if it returns True the value is returned unchanged.
        """
        def _cast_check(x):
            if not pred(x): raise ValueError
            return x
        return _cast_check
    @staticmethod
    def cast_equal(val):
        """
        Checks that the value is equal to the given value, or it throws a ValueError. This is
        useful for use with cast_or with something that has a special value or something more common
        like an integer.
        """
        def _cast_equal(x):
            if x!=val: raise ValueError
            return x
        return _cast_equal
    @staticmethod
    def cast_lookup(d):
        """
        Casts the value by giving the value as a key to a dictionary. If it isn't a key, a
        ValueError is raised.
        """
        def _cast_lookup(x):
            try: return d[x]
            except KeyError: raise ValueError
        return _cast_lookup
    @staticmethod
    def cast_in(*l):
        """
        Checks the value by making sure it is in the given list (given as multiple arguments),
        raising a ValueError if it is not in the list.
        """
        def _cast_in_list(x):
            if x in l: return x
            raise ValueError
        return _cast_in_list
    @staticmethod
    def cast_int(pred=lambda x:True):
        """
        Casts the value to an int using the int() function and then checks it with the optional
        predicate (which, by default, allows all integers). If it cannot be cast to an int or the
        predicate returns False, a ValueError is raised.
        """
        def _cast_int(x):
            x = int(x)
            if not pred(x): raise ValueError
            return x
        return _cast_int
    @staticmethod
    def cast_float(pred=lambda x:True):
        """
        Casts the value to a float using the float() function and then checks it with the optional
        predicate (which, by default, allows all integers). If it cannot be cast to an int or the
        predicate returns False, a ValueError is raised. Additionally, NaN values are never allowed.
        """
        def _cast_float(x):
            from math import isnan
            x = float(x)
            if isnan(x) or not pred(x): raise ValueError
            return x
        return _cast_float
    @staticmethod
    def cast_writable_file():
        # TODO: fix
        """
        A cast that does some simple checks to see if the file is really not writable. No guarntees
        that it isn't actually writable, but should catch some problems. Also improves the results
        of cast_readable_file.
        """
        import os
        def _cast_writable_file(x):
            if x == '': raise ValueError
            x = os.path.abspath(x)
            if os.path.exists(x):
                if os.path.isdir(x) or not os.access(x, os.W_OK): raise ValueError
            elif not os.access(os.path.dirname(x), os.W_OK|os.X_OK): raise ValueError
            _files_to_be_written.append(x)
            return x
        return _cast_writable_file
    @staticmethod
    def cast_readable_file():
        # TODO: fix
        """
        A cast that does some simple checks to see if the file is really not readable. No guarntees
        that it isn't actually readable, but should catch some problems. The file should already
        exist or be a file that will be written by cast_writable_file().
        """
        import os
        def _cast_readable_file(x):
            if x == '': raise ValueError
            x = os.path.abspath(x)
            if not os.path.exists(x):
                if x not in _files_to_be_written: raise ValueError
            elif not (os.path.isfile(x) and os.access(x, os.R_OK)): raise ValueError
            return x
        return _cast_readable_file

class _CommandMeta(ABCMeta):
    """The meta-class for commands, which extends ABCMeta and calls Help.register if applicable"""
    def __new__(cls, clsname, bases, dct):
        c = super(_CommandMeta, cls).__new__(cls, clsname, bases, dct)
        fs = c.flags()
        if fs is not None and len(fs) > 0: Help.register((c.name(),)+fs, c.print_help)
        return c

_cmd_names = {}
_cmd_flags = {}
def _get_cmd_class(nf): return _cmd_flags.get(nf, None) or _cmd_names.get(nf.lower(), None)
class Command:
    """
    A command that can be executed from the command line. It is looked up with its name or the
    flags. it is run in two seperate phases: parsing (in the __init__ function) and the executing.
    The __init__ function does much of the logic work and once called the command should know
    everything it needs to do during execution. The exception is with files which may not be saved
    yet when the parsing happens but may exist (because another command ran) when executing.

    Implementors must implement the followng class methods:
        name(cls)
        flags(cls)
        print_help(cls, width)
        
    And the following methods:
        __init__(self, args, stack)
        __str__(self)
        execute(self, stack)
    """
    __metaclass__ = _CommandMeta
    
    @classmethod
    def _get_all_subclasses(cls):
        subcls = cls.__subclasses__()
        for sc in list(subcls): subcls.extend(sc._get_all_subclasses())
        return subcls

    @classmethod
    def __get_cmd_class(cls, name_or_flag):
        ccls = _get_cmd_class(name_or_flag)
        if ccls is not None: return ccls
        for ccls in cls._get_all_subclasses():
            n, f = ccls.name(), ccls.flags()
            if n is not None: _cmd_names[n.lower()] = ccls
            if f is not None: _cmd_flags.update({F:ccls for F in f})
        return _get_cmd_class(name_or_flag)

    @classmethod
    def create(cls, name_or_flag, args, stack):
        ccls = cls.__get_cmd_class(name_or_flag.lstrip('-'))
        if ccls is None: raise ValueError("'%s' is not a known command" % name_or_flag)
        return ccls(args, stack)
    
    @classmethod
    def name(cls):
        """
        The name of the command. This is another string that the command can be looked up with and
        the text that is displayed in the list of available commands.
        """
        return None
    
    @classmethod
    def flags(cls):
        """List of acceptable flags to recognize the command, without - or --"""
        return None

    @classmethod
    def print_help(cls, width):
        """
        Print the generic help page for this command using the given width. There are many functions
        in the Help class to ease printing.
        """
        pass

    @abstractmethod
    def __init__(self, args, stack):
        """
        Parse the command arguments, given from the Args object. The stack is a psuedo-stack object
        that has the same interface as the stack given to execute except no image stacks are taken
        or returned. You must call pop/push on the stack so that an accurate count of items on the
        stack is kept to check for errors.
        """
        pass
    
    @abstractmethod
    def __str__(self):
        """Get a brief, one-line, description of this command."""
        pass
    
    @abstractmethod
    def execute(self, stack):
        """Execute the command from the previously parsed arguments."""
        pass

class CommandEasy(Command):
    """
    A Command that does some extra work for you but restricts how options are parsed. All options
    must be defined in _opts, there cannot be any inter-dependence of options, and there cannot be
    any in positional-only arguments or variable-count arguments. Additionally, it is assumed that
    you will pop all image stacks you list before pushing any image stacks, and there can be no
    optionally consumed or produced image stacks.
    
    A dictionary of arguments is available as the _vals attribute and every option is also made into
    a field starting with an underscore than the name of the field. The print-help function is also
    made for you (as long as you fill out the other functiosn, like _title and _desc).
    """
    @classmethod
    def _title(cls):
        """The title of the easy command, default is the name with Title Case."""
        return " ".join(w.capitalize() for w in cls.name.split())
    @classmethod
    def _desc(cls):
        """The description of the easy command"""
        return None
    @classmethod
    def _opts(cls):
        """Optional list of Option objects for the options/arguments of the easy command"""
        return None
    @classmethod
    def _consumed(cls):
        """Return a list that names all consumed image stacks"""
        return ()
    @classmethod
    def _produced(cls):
        """Return a list that names all produced image stacks"""
        return ()
    @classmethod
    def _examples(cls):
        """Optional list of examples"""
        return None
    @classmethod
    def _see_also(cls):
        """Optional list similar topics"""
        return None
    @classmethod
    def print_help(cls, width):
        t, d, fs, os, c, p, ex, sa = (
            cls._title(), cls._desc(), cls.flags(), cls._opts(),
            cls._consumed(), cls._produced(), cls._examples(), cls._see_also())
        if fs is None or len(fs) == 0: return
        p = Help(width)
        p.title(t)
        if d: p.text(d); print ""
        p.flags(fs)
        p.text("Command format:")
        s = next(sorted(fs, key=len, reverse=True))
        s = ('--' if len(flg) > 1 else '-')+s
        if os: s += " " + (" ".join('['+o.name+']' if opt.has_default else o.name for o in os))
        p.cmds(s)
        if os: print "\nOptions:"; p.opts(os)
        if c or p: print ""; p.stack_changes(consumed=c, produced=p)
        if ex: print ""; p.list(*ex)
        if sa: print ""; p.list(*sa)
    def __init__(self, args, stack):
        for i in xrange(len(type(self)._consumed())): stack.pop()
        for i in xrange(len(type(self)._produced())): stack.push()
        self._vals = args.get_all_kw(type(self)._opts()).iteritems()
        for name,val in self._vals:
            setattr(self, '_'+name, val)

class Help:
    """
    General methods for display help. For printing, instantiate this class with a width and then use
    all the various utility printing functions.
    """

    _topics = {}
    @staticmethod
    def register(topics, content):
        """
        Registers a help topic. The `topics` is a list of strings which the content can be looked up
        with. The `content` is either a string to be displayed or a function that takes a width that
        will prints out the content. The content should start with a header (e.g. ===== Topic =====...).
        The Help class has many utility print functions for easy printing using a standard format.
        Also note that every Command subclass automatically has it's print_help registered under its
        flag and names.
        """
        if isinstance(topics, basestring): topics = (topics,)
        for t in topics: Help._topics[t.strip()] = content

    @staticmethod
    def show(topic=None):
        from sys import exit
        if topic == None: Help.__msg()
        else:
            content = Help._topics.get(topic.strip())
            if content is None: __err_msg("Help topic not found.")
            if isinstance(content, basestring):
                for l in desc.splitlines(): print fill(l)
            else: content(out_width)
        exit(0)
    
    @staticmethod
    def __msg():
        from os.path import basename
        from sys import argv
        f12 = TextWrapper(width=out_width, subsequent_indent=' '*11).fill
        f18 = TextWrapper(width=out_width, subsequent_indent=' '*18).fill
        print fill("===== Image Stack Reader and Converter Tool " + ("="*max(0,out_width-44)))
        print f18("%s [basic args] [--cmd1 args...] [--cmd2 args...] ..." % basename(argv[0]))
        print fill("Basic arguments:")
        print f18("  -h  --help [x]  display help about a command, filter, or format (all other arguments will be ignored)")
        print f18("  -v  --verbose   display all information about processing of commands")
        print ""
        print fill("You may specify any set of commands and arguments via a file using @argfile which will read that file in as POSIX-style command arguments (including supporting # for comments).")
        print ""
        print fill("The basic idea of this program is it runs a series of commands where each command can consume and/or produce image stacks. Some examples are 'loading' doesn't consume any stacks and produces one stack - to be consumed by the next command. The list of image stack is done in a LIFO ordering.")
        print ""
        print fill("All commands start with - or -- and the arguments that follow the command can be provided in order or using named agruments such as name=value.")
        print ""
        print fill("For the available commands, see the following help topics:")
        print f12("  load      loading an image stack")
        print f12("  save      saving an image stack")
        print f12("  select    choose which image stacks will be used next")
        print f12("  commands  a list of available commands and some other non-command topics (like colors)")

    def __init__(self, width=out_width):
        self._width = width
        self._title_wrp = TextWrapper(width=width-6, initial_indent='===== ', subsequent_indent='===== ').wrap
        self._subtitle_wrp = TextWrapper(width=width-6,initial_indent='----- ', subsequent_indent='----- ').wrap
        self._cmd_fll = TextWrapper(width=width, initial_indent='  ', subsequent_indent='        ').fill
        self._text_fll = TextWrapper(width=width).fill
        self._flag_fll = TextWrapper(width=width, initial_indent='  ').fill
        self._stack_ch_fll = TextWrapper(width=width, subsequent_indent=' '*10).fill
        self._list_fll = TextWrapper(width=width, initial_indent=' * ', subsequent_indent='   ').fill
    def title(self, text):
        """General help page printing of a title."""
        text = text.strip()
        lines = self._title_wrp(text)
        for l in lines[:-1]: print l + (" "*(self._width-len(l)-5)) + '====='
        print lines[-1] + ' ' + ('='*(self._width-len(lines[-1])-1))
    def subtitle(self, text):
        """General help page printing of a title. Comes with an extra newline before it."""
        text = text.strip()
        lines = self._subtitle_wrp(text)
        print ''
        for l in lines[:-1]: print l + (" "*(self._width-len(l)-5)) + '-----'
        print lines[-1] + ' ' + ('-'*(self._width-len(lines[-1])-1))
    def cmds(self, *cmds):
        """Print a list of command lines"""
        for c in cmds: print self._cmd_fll(c)
    def text(self, text):
        """
        General help page printing of a block of text. In that block, double-newlines form a new
        paragraph and a newline followed by a space is a literal single newline. Newlines that
        aren't followed by a space or another newline are removed. Additionally, whitespace at the
        beginning and end is removed.
        """
        lines = ['']
        for line in text.strip().splitlines():
            if line == '':             lines.extend(('','')) # paragraph break adds an empty line
            elif len(lines[-1]) == 0:  lines[-1] += line     # beginning of a new line
            elif lines[-1][-1] == ' ': lines.append(line)    # literal new line starts a new line
            else:                      lines[-1] += ' '+line # add it to the last line with a space
        for l in lines: print self._text_fll(l)
    def flags(self, flags):
        """General help page printing of the command flags."""
        if flags is None or len(flags) == 0: return
        print self._text_fll("Command names:")
        for f in flags: print self._flag_fll(('-' if len(f) == 1 else '--') + f)
    def __stack_changes(self, name, req, opt):
        if len(req) + len(opt) > 0:
            names = (", ".join(req))+(", " if len(req) != 0 and len(opt) != 0 else "")+(", ".join(opt))
            s = "s" if len(req)+len(opt) > 1 else ""
            up = "" if len(opt) == 0 else "-%d"%(len(req)+len(opt))
            print self._stack_ch_fll("%s %d%s image stack%s (%s)"%(name, len(req), up, s, names))
    def stack_changes(self, consumed=(), opt_consumed=(), produced=(), opt_produced=()):
        """
        General help page printing of how the command effects the stack of images. The `consumed` and
        `produced` arguments are lists of image stack names that are always taken/created (such as
        'mask'). The `opt_consumed` and `opt_generated` are ones that may be taken/created.
        """
        self.__stack_changes("Consumes: ", consumed, opt_consumed)
        self.__stack_changes("Produces: ", produced, opt_produced)
    def opts(self, *opts):
        """General help page printing of a set of command options."""
        l = min(max(max(len(o.name) for o in opts)+1, 8), 16)
        fll = TextWrapper(width=self._width, initial_indent='  ', subsequent_indent=' '*(2+l)).fill
        for o in opts: print fll(o.name+(' '*max(l-len(o.name), 1))+o.full_desc)
    def list(self, *items):
        """General help page printing of a list of strings."""
        for x in items: print self._list_fll(x)
        
def __topics_help(width):
    p = Help(width)
    p.title("Topics / Commands")
    topics = {}
    for t,f in Help._topics.iteritems(): topics.setdefault(f,[]).append(t)
    topics = [list(sorted(ts, key=len, reverse=True)) for f,ts in topics.iteritems()]
    topics.sort(key=lambda l:l[0].lower())
    p.list(*[", ".join(ts) for ts in topics])
Help.register(('topic','topics','commands'),__topics_help)

__debug = False
def __err_msg(msg, ex=None):
    from sys import stderr, exit
    print >> stderr, fill(msg)+'\n'
    print __debug
    if ex is not None and __debug:
        from sys import exc_info
        from traceback import print_exception
        typ, val, tb = exc_info()
        print_exception(type(ex), ex, tb if val is ex else None)
        del tb
    exit(1)

##def get_opts(s):
##    x = s[2:].rsplit(':',1)
##    return s[:2]+x[0], dict(x.split('=',1) for x in x[1].split(',')) if len(x) == 2 else {}

def __split_args(args):
    """
    Splits an argument list into groups starting with either -x or --long_name followed by any
    values that follow them (values are anything that does not start with - (except -# where # is .
    or a number).
    """
    out = []
    for a in args:
        if len(a) >= 2 and a[0] == '-' and a[1] not in '.0123456789':
            if a[1]!='-': out.extend(['-'+x] for x in a[1:])
            else:         out.append([a])
        else:             out[-1].append(a)
    return out

def main(debug=False):
    from sys import argv
    import shlex
    global __debug
    __debug = debug
    
    ##### Parse Arguments #####
    args = argv[1:]
    append = False

    ## Find and expand @argfile arguments
    # This loop goes through all @ arguments in reverse order providing the index of the @ argument
    for i in reversed([i for i,a in enumerate(args) if len(a)>0 and a[0] == '@']):
        try:
            with open(args[i][1:], 'r') as argfile: args[i:i+1] = shlex.split(argfile.read(), True) # read the file, split it, and insert in place of the @ argument
        except IOError as ex:   __err_msg('Unable to read arg-file "%s": %s' % (args[i][1:],ex), ex)
        except Exception as ex: __err_msg('Invalid contect of arg-file "%s": %s' % (args[i][1:],ex), ex)
    args = __split_args(args)

    # Basic arguments
    verbose = False
    num_basic_args = 0
    for cmd in args:
        if cmd[0] in ('-v', '--verbose'):
            if verbose: __err_msg("Can only be one -v/--verbose argument.")
            if len(cmd) > 1: __err_msg("-v/--verbose does not take any values")
            verbose = True
            num_basic_args += 1
        elif cmd[0] in ('-h', '--help'):
            if len(cmd) > 2: __err_msg("-h/--help can take at most more value")
            Help.show(cmd[1] if len(cmd) == 2 else None)
            # unreachable: num_basic_args += 1
        else: break
    del args[:num_basic_args]
    if len(args) == 0: Help.show()

    # Parse all of the commands
    stack = PseudoStack(verbose)
    cmds = []
    for cmd_arg in args:
        try: cmds.append(Command.create(cmd_arg[0].lstrip('-'),Args(cmd_arg),stack))
        except Exception as ex: __err_msg('%s: %s'%(" ".join(cmd_arg),ex), ex)

    # Execute all of the commands
    stack = Stack(verbose)
    for cmd in cmds:
        if verbose:
            print "="*50
            print "> %s"%cmd
        try: cmd.execute(stack)
        except Exception as ex: __err_msg('%s: %s'%(cmd,ex), ex)

    # TODO: if there is something left in stack, output it?

import _imstack
import images # make sure everything is loaded to get commands and help topics registered
