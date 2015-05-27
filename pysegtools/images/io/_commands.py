"""The commands for I/O: load and save to image stacks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
numeric_pattern = re.compile("(^[^#]*)(#+)([^#]*[.][^#]+)$")

from ...imstack import Command, Opt, Help
from ._stack import FileImageStack
from ._single import FileImageSource
from .._util import re_search

@staticmethod
def cast_num_pattern(s):
    # NOTE: this cannot be called on an already casted value
    if not re_search(numeric_pattern, s): raise ValueError('pattern must have a single set of number signs denoting when the slice number will be placed')
    before, digits, after = re_search.match.groups()
    return before+"%0"+str(len(digits))+"d"+after

oPattern = Opt('pattern','numerical pattern with number signs (#) to be replaced by integers',cast_num_pattern)
oStart = Opt('start','non-negative integer to start counting at',Opt.cast_int(lambda x:x>=0),0)
oStep = Opt('step','positive integer to count by',Opt.cast_int(lambda x:x>=1),1)
oStop = Opt('stop','non-negative integer to stop counting at (it is included)',Opt.cast_int(lambda x:x>=0),None)

def _pattern_desc(pattern,start,step,stop=None):
    if start==0 and step==1 and stop is None: return "'%s'" % pattern
    return "'%s' starting at %d%s%s"%(pattern,start,
        ('and stepping by %d'%step) if step!=1 else '',
        ('up to %d'%stop) if stop is not None else '')

class LoadCommand(Command):
    @classmethod
    def name(cls): return 'load'
    @classmethod
    def flags(cls): return ('L', 'load')
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Loading an Image Stack")
        p.text("""
Image stacks can be loaded from individual 3D image files or a collection of 2D image files. In all
cases filenames with '=' in them, or are only '[' or ']', are not supported.
""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("Command formats:")
        p.cmds("--load filepath [options...]",
               "--load pattern [start] [step] [stop] [options...]",
               "--load [ file1 file2 ... ] [options...]")
        p.newline()
        p.stack_changes(produces=('loaded image stack',))
        p.newline()
        p.text("See also:")
        p.list('save')

        p.newline()
        p.subtitle("3D Image Files")
        p.text("""Command Format: --load filepath [options...]
Supported handlers are:""")
        p.list(*sorted(FileImageStack.handlers(True)))
        p.text("""
The options effect how images are loaded and must be given as named options. The option 'handler'
forces a specific handler to be used, which is useful if two handlers support the same file format
but in different ways (such as MRC files being supported by the 'MRC' and 'bio-formats' handlers).

Some handlers support additional options as well. To get more information about a specific handler
and its options, use --help {format}, for example --help MRC.""")

        p.newline()
        p.subtitle("2D Image Files")
        p.text("""
For collections of 2D image files (including PNG, TIFF, BMP, JPEG, MHA/MHA, IM, and MAT) specify
either a numeric pattern or a list of files. Supported handlers are:""")
        p.list(*sorted(FileImageSource.handlers(True)))
        p.text("""
Additional options can be applied to loading files and must be given as named options. In all cases
the option 'handler' specifies which of the above handlers to use. Other options are specific to
each handler, see the help page for each handler for more information.""")

        p.subtitle("Numerical Pattern of 2D Image Files")
        p.text("Command Format: --load pattern [start] [step] [stop] [options...]")
        p.opts(oPattern, oStart, oStep, oStop)
        p.text("""
When loading, the number of number signs in the pattern does not matter. It will use all filenames
with any number of digits in place of the # signs and use them in numerical order. The start, step,
and stop options put restrictions on the values allowed in place of the number signs.

The additional options apply to all images.

Examples:""")
        p.list("--load file###.png", "-L file###.png 3 2 15")

        p.subtitle("List of 2D Image Files")
        p.text("""Command Format: --load [ file1 file2 ... ] [options...] 
A list of files is given between literal [ and ]. Glob patterns are also accepted within the
brackets. Options at the end apply to all images listed. There is currently no way to set the
options for individual files.

Examples:""")
        p.list("[ file1.png file2.png file3.png ]", "[ dir/*.png ]")

    def __str__(self): return "Loading from %s"%self._name
    def __init__(self, args, stack):
        from os.path import abspath, isfile
        if len(args.positional) == 0: raise ValueError("No file given to load")
        stack.push()
        self._name = args.pop(0)
        self._handler = args.pop('handler', None)
        stack_based = False # if we are loading a stack directly (True) or a series of files (False)
        path = abspath(self._name)

        if self._name == "[": # File image list
            try: end = args.positional.index("]")
            except ValueError: raise ValueError("No terminating ']' in filename list")
            if len(args.positional) > end+1: raise ValueError("All options must be specified as named options")
            self._file = [abspath(f) for f in args[:end]]
            self._name = ("'"+("', '".join(args[:end]))+"'") if end else '<no files>'
            self._kwargs = args.named

        elif re_search(numeric_pattern, path): # Numeric Pattern
            self._kwargs = {k:args.pop(k) for k in args.named.keys() if k not in ('start','step','stop')}
            before, digits, after = re_search.match.groups()
            start, step, stop = args.get_all(oStart, oStep, oStop)
            self._file = (before, after, start, step, stop)
            self._name = _pattern_desc(self._name, start, step, stop)

        else: # 3D Image File
            if len(args.positional)>0: raise ValueError('All options must be specified as named options')
            if isfile(path) and not FileImageStack.openable(path, True, self._handler, **args.named): #pylint: disable=star-args
                # if the file does not yet exist, it may after some other operation, so only check if it exists now
                raise ValueError("Unable to open '%s' with given options" % path)
            self._file   = path
            self._name = "'%s'" % self._name
            self._kwargs = args.named
            stack_based = True

        if self._handler is not None:
            if stack_based: FileImageStack.is_handler(self._handler, True)
            else:           FileImageSource.is_handler(self._handler, True)
            self._name += ' using handler "'+self._handler+'"'
        if len(self._kwargs)>0:
            self._name += " with options " + (", ".join("%s=%s"%(k,v) for k,v in self._kwargs.iteritems()))

    def _get_files(self):
        from glob import glob, iglob
        from os.path import isfile

        if isinstance(self._file, list): # File image list
            files = []
            for f in self._file:
                if not isfile(f):
                    if any(c in f for c in '*?['):
                        fg = glob(f)
                        if len(fg) > 0: files += fg; continue
                    raise IOError("File %s does not exist" % f)
                files.append(f)
            return files

        elif isinstance(self._file, tuple): # Numeric Pattern
            before, after, start, step, stop = self._file
            files = []
            # TODO: could force number of digits using '[0-9]'*len(digits) in glob pattern instead of '*'
            for f in iglob(before+'*'+after):
                if not f.startswith(before) or not f.endswith(after): continue # possible?
                num = f[len(before):-len(after)]
                if not num.isdigit(): continue
                i = int(num)
                if i<start or (stop is not None and i>stop) or ((i-start)%step)!=0: continue
                files.append((i, f))
            return [f for i,f in sorted(files)]

        else: return self._file # 3D Image File

    def execute(self, stack):
        stack.push(FileImageStack.open(self._get_files(), True, self._handler, **self._kwargs))


class SaveCommand(Command):
    @classmethod
    def name(cls): return 'save'
    @classmethod
    def flags(cls): return ('S', 'save')
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Saving an Image Stack")
        p.text("""
Image stacks can be saved to individual 3D image files or a collection of 2D image files. In all
cases filenames with '=' in them, or are only '[' or ']', are not supported.
""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("Command formats:")
        p.cmds("--save filepath [options...]",
               "--save pattern [start] [step] [options...]",
               "--save [ file1 file2 ... ] [pattern [start] [step]] [options...]")
        p.newline()
        p.stack_changes(consumes=('image stack to be saved',))
        p.newline()
        p.text("See also:")
        p.list('load')

        p.newline()
        p.subtitle("3D Image Files")
        p.text("""Command Format: --load filepath [options...]
Supported handlers are:""")
        p.list(*sorted(FileImageStack.handlers(False)))
        p.text("""
The options effect how images are saves and must be given as named options. The option 'handler'
forces a specific handler to be used, which is useful if two handlers support the same file
extension but in different ways (such as MRC files being supported by the 'MRC' and 'bio-formats'
handlers) or for formats which don't have a consistent file extension.

Some handlers support additional options as well. To get more information about a specific handler
and its options, use --help {format}, for example --help MRC.""")

        p.newline()
        p.subtitle("2D Image Files")
        p.text("""
For collections of 2D image files (including PNG, TIFF, BMP, JPEG, MHA/MHA, and IM) specify either
a numeric pattern or a list of files. Supported handlers are:""")
        p.list(*sorted(FileImageSource.handlers(False)))
        p.text("""
Additional options can be applied to saving files and must be given as named options. In all cases
the option 'handler' specifies which of the above handlers to use. Other options are specific to
each handler, see the help page for each handler for more information.""")

        p.subtitle("Numerical Pattern of 2D Image Files")
        p.text("Command Format: --save pattern [start] [step] [options...]")
        p.opts(oPattern, oStart, oStep)
        p.text("""
When saving, the number of number signs will determine how to pad the numbers with leading zeros.
The start and step values dictate the numbers to use to save the files. Unlike when loading, there
is no "stop" value as it is determined by the number of image slices being saved.

The additional options apply to all images.

Examples:""")
        p.list("--save file###.png", "-S file###.png 3 2")

        p.subtitle("List of 2D Image Files")
        p.text("""Command Format: --save [ file1 file2 ... ] [pattern [start] [step]] [options...] 
A list of files is given between literal [ and ]. Glob patterns are not accepted. If there are more
slices than filenames listed and a pattern is given, that pattern will be used to generate the
remaining filenames. See above for the definition of those options.

Options at the end apply to all images listed. There is currently no way to set the options for
individual files.

Examples:""")
        p.list("-S [ file1.png file2.png file3.png ]")
    def __str__(self): return "Saving to %s"%self._name
    def __init__(self, args, stack):
        from os.path import abspath
        if len(args.positional) == 0: raise ValueError("No file given to save to")
        stack.pop()
        self._name = args.pop(0)
        self._handler = args.pop('handler', None)
        stack_based = False # if we are loading a stack directly (True) or a series of files (False)
        path = abspath(self._name)

        if self._name == "[": # File image list
            try: end = args.positional.index("]")
            except ValueError: raise ValueError("No terminating ']' in filename list")
            self._file = [abspath(f) for f in args[:end]]
            self._name = ("'"+("', '".join(args[:end]))+"'") if end else '<no files>'
            del args[:end+1]
            self._kwargs = {k:args.pop(k) for k in args.named.keys() if k not in ('pattern','start','step')}
            if len(args) > 0:
                raw_pattern = args[(0,'pattern')]
                pattern,start,step = args.get_all(oPattern, oStart, oStep)
                self._kwargs.update(pattern=abspath(pattern),start=start,step=step)
                self._name = ((self._name+' then using ') if end else '')+_pattern_desc(raw_pattern,start+end,step)
                # TODO: for name's sake, pattern,start,step are both already in name and in kwargs

        elif re_search(numeric_pattern, path): # Numeric Pattern
            self._kwargs = {k:args.pop(k) for k in args.named.keys() if k not in ('start','step')}
            before, digits, after = re_search.match.groups()
            pattern = before+("%%0%dd"%len(digits))+after
            start, step = args.get_all(oStart, oStep)
            self._file = (pattern, start, step)
            self._name = _pattern_desc(self._name, start, step)
            self._kwargs.update(pattern=pattern,start=start,step=step)

        else: # 3D Image File
            if len(args.positional)>0: raise ValueError('You must provide all options as named options.')
            if not FileImageStack.creatable(path, **args.named): raise ValueError("Unable to create '%s' with given options" % path) #pylint: disable=star-args
            self._file   = path
            self._kwargs = args.named # arguments get passed straight to the iamge stack creator
            self._name = "'%s'" % self._name
            stack_based = True

        if self._handler is not None:
            if stack_based: FileImageStack.is_handler(self._handler, False)
            else:           FileImageSource.is_handler(self._handler, False)
            self._name += ' using handler "'+self._handler+'"'
        if len(self._kwargs)>0:
            self._name += " with options " + (", ".join("%s=%s"%(k,v)
                for k,v in self._kwargs.iteritems()
                if stack_based or k not in ('pattern','start','step')))

    def execute(self, stack):
        from os.path import dirname
        from ...general.utils import make_dir
        ims = stack.pop()
        files = self._file
        if isinstance(files, tuple):
            pattern, start, step = self._file
            files = [pattern%(i*step+start) for i in xrange(len(ims))]
        if any(not make_dir(dirname(f)) for f in (files if isinstance(files, list) else [files])): raise ValueError("Failed to create ouput directories")
        ims = FileImageStack.create(files, ims, True, self._handler, **self._kwargs)
        ims.save()
        Help.print_stack(ims)
        ims.close()
