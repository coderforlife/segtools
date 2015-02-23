"""The commands for I/O: load, save, and append to image stacks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
numeric_pattern = re.compile("(^[^#]*)(#+)([^#]*[.][^#]+)$")

from ...imstack import Command, Opt, Help
from ._stack import FileImageStack
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
oStop = Opt('stop','non-negative integer to stop counting at (it is included)',Opt.cast_or(Opt.cast_equal(None),Opt.cast_int(lambda x:x>=0)),None)

def _pattern_desc(pattern,start,step,stop=None):
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
        p.text("Image stacks can be loaded from individual 3D image files or a collection of 2D image files.")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("Command formats:")
        p.cmds("--load filepath [format-specific-options]",
               "--load pattern [start] [step] [stop]",
               "--load [ file1 file2 ... ]   (literal brackets)")
        p.newline()
        p.stack_changes(produces=('loaded image stack',))
        p.newline()
        p.text("See also:")
        p.list('save', 'append')

        p.subtitle("3D Image Files")
        p.text("""Command Format: --load filepath [format-specific-options]
Supported formats are:""")
        p.list(*FileImageStack.formats(True))
        p.text("""
Some file formats support additional options which can be specified after the path. To get more
information about a format and its options, use --help {format}, for example --help MRC. All options
to these files must be given as named options.""")

        p.subtitle("2D Image Files")
        p.text("""
For collections of 2D image files (including PNG, TIFF, BMP, JPEG, MHA/MHA, IM, and MAT) specify
either a numeric pattern or a list of files.""")

        p.subtitle("Numerical Pattern of 2D Image Files")
        p.text("Command Format: --load pattern [start] [step] [stop]")
        p.opts(oPattern, oStart, oStep, oStop)
        p.text("""
When loading, the number of number signs in the pattern does not matter. It will use all filenames
with any number of digits in place of the # signs and use them in numerical order. The start, step,
and stop options put restrictions on the values allowed in place of the number signs.

Examples:""")
        p.list("--load file###.png", "-L file###.png 3 2 15")

        p.subtitle("List of 2D Image Files")
        p.text("""Command Format: --load [ file1 file2 file3 ]   (literal brackets)
A list of files is given between literal [ and ]. Glob patterns are also accepted within the
brackets.

Examples:""")
        p.list("[ file1.png file2.png file3.png ]", "[ dir/*.png ]")

    def __str__(self): return "Loading from %s"%self._name
    def __init__(self, args, stack, appending=False):
        from os.path import abspath, isfile
        if len(args.positional) == 0: raise ValueError("No file given to load")
        if appending: stack.pop()
        else: stack.push()
        self._name = args[0]
        self._args = ()
        self._kwargs = {}
        del args[0]
        path = abspath(self._name)

        if self._name == "[": # File image list
            try: end = args.positional.index("]")
            except ValueError: raise ValueError("No terminating ']' in filename list")
            self._file = [abspath(f) for f in args[:end]]
            self._name = ("'"+("', '".join(args[:end]))+"'") if end else '<no files>'
            del args[:end+1]
            if len(args) > 0:
                if not appending: raise ValueError("Loading from a file list does not support any extra options")
                try:
                    raw_pattern = args[(0,'pattern')]
                    pattern,start,step = args.get_all(oPattern, oStart, oStep)
                    self._kwargs = {'pattern':abspath(pattern),'start':start,'step':step}
                    self._name = ((self._name+', then using ') if end else '')+_pattern_desc(raw_pattern,start+end,step)
                except KeyError: raise ValueError("Appending to a file list does not support any extra options without pattern")

        elif re_search(numeric_pattern, path): # Numeric Pattern
            before, digits, after = re_search.match.groups()
            start, step, stop = args.get_all(oStart, oStep, oStop)
            self._file = (before, after, start, step, stop)
            self._name = _pattern_desc(self._name, start, step, stop)
            if appending: self._kwargs = {'pattern':before+("%%0%dd"%len(digits))+after,'start':start,'step':step}

        else: # 3D Image File
            if len(args.positional)>0: raise ValueError('You must provide all file-format options as named options.')
            args = args.named
            if isfile(path) and not FileImageStack.openable(path, not appending, **args): #pylint: disable=star-args
                # if the file does not yet exist, it may after some other operation, so only check if it exists now
                raise ValueError("Unable to open '%s' with given options" % path)
            self._file   = path
            self._kwargs = args # arguments get passed straight to the iamge stack creator
            self._name = "'%s'" % self._name
            if len(args)>0: self._name += " with options " + (", ".join("%s=%s"%(k,v) for k,v in args.iteritems()))

    def _get_files(self, appending=False):
        from glob import glob, iglob
        from os.path import isfile

        if isinstance(self._file, list): # File image list
            files = []
            for f in self._file:
                if not isfile(f):
                    if any(c in f for c in '*?['):
                        fg = glob(f)
                        if len(fg) > 0: files += fg; continue
                    if not appending: raise IOError("File %s does not exist" % f)
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
        stack.push(FileImageStack.open(self._get_files(), True, **self._kwargs))


class AppendCommand(LoadCommand):
    @classmethod
    def name(cls): return 'append'
    @classmethod
    def flags(cls): return ('A', 'append')
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Appending to an Image Stack")
        p.text("Appending to an image stack is very similar to loading and saving an image stack.")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("Command formats:")
        p.cmds("--append filepath [format-specific-options]",
               "--append pattern [start] [step] [stop]",
               "--append [ file1 ... ] [pattern [start] [step]]  (literal brackets around list)")
        p.newline()
        p.stack_changes(consumes=('image stack to be saved',))
        p.newline()
        p.text("See also:")
        p.list('load', 'save')

        p.subtitle("3D Image Files")
        p.text("""Command Format: --load filepath [format-specific-options]
Supported formats are:""")
        p.list(*FileImageStack.formats(False))
        p.text("""
Some file formats support additional options which can be specified after the path. To get more
information about a format and its options, use --help {format}, for example --help MRC. All options
to these files must be given as named options.""")

        p.subtitle("2D Image Files")
        p.text("""
For collections of 2D image files (including PNG, TIFF, BMP, JPEG, MHA/MHA, and IM) specify either
a numeric pattern or a list of files.""")

        p.subtitle("Numerical Pattern of 2D Image Files")
        p.text("Command Format: --append pattern [start] [step] [stop]")
        p.opts(oPattern, oStart, oStep, oStop)
        p.text("""
Files that already exist that follow the pattern will not be overwritten and the number of number
signs do not matter for finding these files (like during loading). Following start and step, the
first number not found will be the start of the saving of new slices. No files will be loaded after
the stop value. When writing, the number of number signs will determine how to pad the numbers with
leading zeros.

Examples:""")
        p.list("--append file###.png", "-A file###.png 3 2 15")

        p.subtitle("List of 2D Image Files")
        p.text("""Command Format: --append [ file1 ... ] [pattern [start] [step]]  (literal brackets around list)
A list of files is given between literal [ and ]. Glob patterns are also accepted within the
brackets but only apply to loading images. The files are loaded as long as they exist. Once we
reach a file that doesn't exist we start saving. If there are more slices than filenames listed and
a pattern is given, that pattern will be used to generate the remaining filenames. See above for the
definition of those options.

Examples:""")
        p.list("-A [ file1.png file2.png file3.png ]")
    def __str__(self): return "Appending to %s"%self._name
    def __init__(self, args, stack): super(AppendCommand, self).__init__(args, stack, True)
    def execute(self, stack):
        from os.path import dirname
        from ...general.utils import make_dir
        files = self._get_files(True)
        if any(not make_dir(dirname(f)) for f in (files if isinstance(files, list) else [files])): raise ValueError("Failed to create output directories")
        ims = FileImageStack.open(files, False, **self._kwargs)
        ims.extend(stack.pop())
        ims.save()
        Help.print_stack(ims)
        ims.close()


class SaveCommand(Command):
    @classmethod
    def name(cls): return 'save'
    @classmethod
    def flags(cls): return ('S', 'save')
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Saving an Image Stack")
        p.text("Image stacks can be saved to individual 3D image files or a collection of 2D image files.")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("Command formats:")
        p.cmds("--save filepath [format-specific-options]",
               "--save pattern [start] [step]",
               "--save [ file1 ... ] [pattern [start] [step]]  (literal brackets around list)")
        p.newline()
        p.stack_changes(consumes=('image stack to be saved',))
        p.newline()
        p.text("See also:")
        p.list('load', 'append')

        p.subtitle("3D Image Files")
        p.text("""Command Format: --load filepath [format-specific-options]
Supported formats are:""")
        p.list(*FileImageStack.formats(False))
        p.text("""
Some file formats support additional options which can be specified after the path. To get more
information about a format and its options, use --help {format}, for example --help MRC. All options
to these files must be given as named options.""")

        p.subtitle("2D Image Files")
        p.text("""
For collections of 2D image files (including PNG, TIFF, BMP, JPEG, MHA/MHA, and IM) specify either
a numeric pattern or a list of files.""")

        p.subtitle("Numerical Pattern of 2D Image Files")
        p.text("Command Format: --save pattern [start] [step]")
        p.opts(oPattern, oStart, oStep)
        p.text("""
When saving, the number of number signs will determine how to pad the numbers with leading zeros.
The start and stop values dictate the numbers to use to save the files.

Examples:""")
        p.list("--save file###.png", "-S file###.png 3 2")

        p.subtitle("List of 2D Image Files")
        p.text("""Command Format: --save [ file1 ... ] [pattern [start] [step]]  (literal brackets around list)
A list of files is given between literal [ and ]. Glob patterns are not accepted. If there are more
slices than filenames listed and a pattern is given, that pattern will be used to generate the
remaining filenames. See above for the definition of those options.

Examples:""")
        p.list("-S [ file1.png file2.png file3.png ]")
    def __str__(self): return "Saving to %s"%self._name

    def __init__(self, args, stack):
        from os.path import abspath
        if len(args.positional) == 0: raise ValueError("No file given to save to")
        stack.pop()
        self._name = args[0]
        self._kwargs = {}
        del args[0]
        path = abspath(self._name)

        if self._name == "[": # File image list
            try: end = args.positional.index("]")
            except ValueError: raise ValueError("No terminating ']' in filename list")
            self._file = [abspath(f) for f in args[:end]]
            self._name = ("'"+("', '".join(args[:end]))+"'") if end else '<no files>'
            del args[:end+1]
            if len(args) > 0:
                try:
                    raw_pattern = args[(0,'pattern')]
                    pattern,start,step = args.get_all(oPattern, oStart, oStep)
                    self._kwargs = {'pattern':abspath(pattern),'start':start,'step':step}
                    self._name = ((self._name+', then using ') if end else '')+_pattern_desc(raw_pattern,start+end,step)
                except KeyError: raise ValueError("Saving to a file list does not support any extra options without pattern")

        elif re_search(numeric_pattern, path): # Numeric Pattern
            before, digits, after = re_search.match.groups()
            pattern = before+("%%0%dd"%len(digits))+after
            start, step = args.get_all(oStart, oStep)
            self._file = (pattern, start, step)
            self._name = _pattern_desc(self._name, start, step)
            self._kwargs = {'pattern':pattern,'start':start,'step':step}

        else: # 3D Image File
            if len(args.positional)>0: raise ValueError('You must provide all file-format options as named options.')
            args = args.named
            if not FileImageStack.creatable(path, **args): raise ValueError("Unable to create '%s' with given options" % path) #pylint: disable=star-args
            self._file   = path
            self._kwargs = args # arguments get passed straight to the iamge stack creator
            self._name = "'%s'" % self._name
            if len(args)>0: self._name += " with options " + (", ".join("%s=%s"%(k,v) for k,v in args.iteritems()))

    def execute(self, stack):
        from os.path import dirname
        from ...general.utils import make_dir
        ims = stack.pop()
        files = self._file
        if isinstance(files, tuple):
            pattern, start, step = self._file
            files = [pattern%(i*step+start) for i in xrange(len(ims))]
        if any(not make_dir(dirname(f)) for f in (files if isinstance(files, list) else [files])): raise ValueError("Failed to create ouput directories")
        ims = FileImageStack.create(files, ims, **self._kwargs)
        ims.save()
        Help.print_stack(ims)
        ims.close()
