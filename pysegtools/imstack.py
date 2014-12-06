#!/usr/bin/env python

"""Command line program to convert an image stack to another stack by processing each slice."""

__all__ = ["main"]

from .general.utils import check_reqs
check_reqs()

import re

number_range = "([0-9]+)(-[0-9]+)?(@[0-9]+)?"
numeric_pattern = re.compile("(^[^#]*)(#+)([^#]*[.][^#]+)(?:"+number_range+")?$")
number_range = re.compile("^"+number_range+"$")

def help_msg(err = 0, msg = None):
    from os.path import basename
    from sys import stderr, argv, exit
    from textwrap import fill, TextWrapper
    from .general.utils import get_terminal_width
    w = max(get_terminal_width()-1, 24)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None:
        print >> stderr, fill(msg, w)+'\n'
        print '-' * w
        print
    print fill("Image Stack Reader and Converter Tool", w)
    print ""
    print tw.fill("%s [args] -i input [[filters] -o [-a] output]" % basename(argv[0]))
    print tw.fill("  -h  --help [x]  Display this help or information about a filter or format")
    print tw.fill("  -v  --verbose   Display all information about image stack(s)")
    print tw.fill("  -a  --append    Append data to the output-stack instead of creating a new output-stack, single-file output stacks must exist")
    print ""
    print fill("You may provide any arguments via a file using @argfile which will read that file in as POSIX-style command arguments (including supporting # for comments).", w)
    print ""
    print fill("For more information on the input and output arguemnts, see --help input or --help output. For a list of available filters see --help filters.", w)
    print ""
    print fill("===== Identify Usage =====", w)
    print fill("If output is not provided then information about the input stack is printed out.", w)
    print ""
    print fill("===== Convert Usage =====", w)
    print fill("Convert the input to the output possibly running it through image filters.", w)
    exit(err)

def help_adv(topic):
    from sys import exit
    from textwrap import fill, TextWrapper
    from .general.utils import get_terminal_width
    from .images.io import FileImageStack
    from .images import FilteredImageStack
    
    w = max(get_terminal_width()-1, 24)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    lst_itm = TextWrapper(width = w, initial_indent = ' * ', subsequent_indent = ' '*3)
    topic_lower = topic.lower()
    formats = {f.lower():f for f in FileImageStack.formats()}
    filters = {f.lower():f for f in FilteredImageStack.filter_names()}
    
    if topic_lower in ("input", "output", "formats"):
        print "===== Input and Output Stack Specifications and Formats ====="       
        print fill("The input and output stacks can either be a single 3D image file or a collection of 2D image files.", w)
        print ""
        print fill("For image formats that support 3D images simply give the filename after -i or -o. The supported formats for this are:", w)
        for l in formats.itervalues(): print lst_itm.fill(l)
        print fill("Some formats take extra options, to use them put a colon (:) after the filename and seperate each option with a comma. To get more information about a format and its options, use --help {format}, for example --help MRC.", w)
        print ""
        print fill("For collections of 2D image files (including PNG, TIFF, BMP, JPEG, MHA/MHA, IM, and MAT) specify either a numeric pattern or a list of files:", w)
        print lst_itm.fill("A numeric pattern is a filepath that uses number signs (#) in place of a number. The numeric pattern may be followed by a colon (:) and a number range*. For input-stacks it will take all filenames with any number of digits in place of the # signs and use them in numerical order. If a number range is provided it limits the values allowed in place of the # signs. For output-stacks the number of # signs dictactes how numbers will be padded with 0s. Output-stacks start at value 0 by default. Examples:")
        print "     file###.png"
        print "     file###.png:2-10@2"
        print lst_itm.fill("A list of files is given between [ and ] as individual arguments. Examples:")
        print "     [ file1.png file2.png file3.png ]"
        print "     [ dir/*.png ]"
        print ""
        print "* Number ranges are specified as one of <start>-<stop>, <start>-<stop>@<step>, <start>, or <start>@<step> where start and stop are non-negative integers and step is a positive integer. A missing stop value means go till the end and step defaults to 1."
    elif topic_lower == "filters":
        print "===== Filters ====="
        print fill("The filters applied to the images can change the shape and type of the data along with the image content itself. You may need to use some filters just to get the data into a format that the output image stack can handle.")
        print ""
        print fill("The support filters are:")
        for l in filters.itervalues(): print lst_itm.fill(l)
        print fill("To lookup more information about a specific filter, do --help {filter}, for example --help \"Median Blur\".")
        print ""
        print fill("Most have arguments that you can supply after the filter name and some arguments are even required. To specify arguments after the filter name seperated by spaces either listing out the values in order they are specified or giving the argument name, an equal sign, and then the value (allowing you to skip optional arguments). Some examples:")
        print fill("--median-blur 3")
        print fill("--median-blur size=3")
        print fill("-G sigma=1.5")
    elif topic_lower in formats:
        print "===== Format: %s =====" % formats[topic_lower]
        desc = FileImageStack.description(topic_lower)
        if desc is None: desc = "Sorry, no description is currently available for this format."
        for l in desc.splitlines(): print fill(l, w)
    elif topic_lower in filters:
        print "===== Filter: %s =====" % filters[topic_lower]
        print FilteredImageStack.description(topic_lower, w) or "Sorry, no description is currently available for this filter."
    else: help_msg(2, "Help topic not found.") 
    exit(0)

def get_opts(s):
    x = s[2:].rsplit(':',1)
    return s[:2]+x[0], dict(x.split('=',1) for x in x[1].split(',')) if len(x) == 2 else {}

def get_input(args, readonly=True):
    """
    Gets the input stack from the arguments after -i and returns the ImageStack instance and a
    textual version. The args have already been "pre-parsed" in that the [] argument is removed,
    and if not a list only the filename is given.
    """
    from .images.io import FileImageStack
    from glob import iglob
    from os.path import isfile
    # TODO: check readonly=False use... especially file list which may now support it
    if isinstance(args, list):
        if not readonly: help_msg(2, "You cannot append to an image stack that is a fixed list of files, you must use a numeric pattern instead.")
        files = []
        for i in args:
            if not isfile(i): help_msg(3, "File '%s' does not exist." % i)
            files.append(i)
        name = " ".join(files)
        try: return FileImageStack.open(files, readonly), name
        except Exception as e: help_msg(2, "Failed to open image-stack '"+name+"': "+str(e))
    else: #isinstance(args, basestring)
        name = args
        m = numeric_pattern.search(name)
        if m == None:
            name, options = get_opts(name)
            try: return FileImageStack.open(name, readonly, **options), name
            except Exception as e: raise; help_msg(2, "Failed to open image-stack '"+name+"': "+str(e))
        g = m.groups()
        before, digits, after = g[:3]
        start = int(g[3] or 0)
        stop = None if g[4]==None else int(g[4]) # inclusive, None for infinite
        step = int(g[5] or 1) # if it was not provided or given as 0 then we get 1
        # TODO: could force number of digits using '[0-9]'*len(digits) in glob pattern instead of '*'
        files = []
        for f in iglob(before+'*'+after):
            if not f.startswith(before) or not f.endswith(after): continue # possible?
            num = f[len(before):-len(after)]
            if not num.isdigit(): continue
            i = int(num)
            if i<start or stop!=None and i>stop or ((i-start)%step)!=0: continue
            files.append((i, f))
        files.sort()
        options = {}
        if not readonly:
            # TODO: check if stop is given? if stop != None: 
            options['pattern'] = before+("%%0%dd"%len(digits))+after
            options['start'] = start
            options['step'] = step
        try: return FileImageStack.open([f for i,f in files], readonly, **options), name
        except Exception as e: help_msg(2, "Failed to open image-stack '"+name+"': "+str(e))
        
def get_output(args, ims, append):
    """
    Gets the output stack from the arguments after -o till the end and returns the ImageStack
    instance and a textual version. The args have already been "pre-parsed" in that the -a and []
    arguments have been removed, and if not a list only the filename is given.
    """
    import os.path
    from .general.utils import make_dir
    from .images.io import FileImageStack
    if args == None: return None, None
    elif append:
        return get_input(args, False)
    elif isinstance(args, list):
        files = args
        if len(files) != len(ims): help_msg(2, "When using a list of filenames for the output stack it must have exactly one filename for each slice in the input-stack.")
        if any(not make_dir(os.path.dirname(f)) for f in files if os.path.dirname(f) != ''): help_msg(2, "Failed to create new image-stack because the file directories could not be created.")
        name =  " ".join(files)
        try: return FileImageStack.create(files, ims), name
        except Exception as e: raise; help_msg(2, "Failed to create new image-stack '"+name+"': "+str(e))
    else: #isinstance(args, basestring)
        name = args
        m = numeric_pattern.search(name)
        if m == None:
            name, options = get_opts(name)
            try: return FileImageStack.create(name, ims, **options), name
            except Exception as e: raise; help_msg(2, "Failed to create new image-stack '"+name+"': "+str(e))
        g = m.groups()
        before, digits, after = g[:3]
        ndigits = len(digits)
        start = g[3] or 0
        stop = g[4] # inclusive, None for infinite
        step = g[5] or 1 # if it was not provided or given as 0 then we get 1
        if stop != None and (stop-start)//step != len(ims): help_msg(2, "When using numerical pattern with a range it must cover the exact number of slices in the input stack (easiest to just leave off the upper bound).")
        files = [before + str(i*step+start).zfill(ndigits) + after for i in xrange(len(ims))]
        if any(not make_dir(os.path.dirname(f)) for f in files if os.path.dirname(f) != ''): help_msg(2, "Failed to create new image-stack because the file directories could not be created.")
        try: return FileImageStack.create(files, ims, pattern=before+("%%0%dd"%ndigits)+after, start=start, step=step), name
        except Exception as e: raise; help_msg(2, "Failed to create new image-stack '"+name+"': "+str(e))

def split_args(args):
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

def extract_args(args, names):
    """
    Extract arguments from an un-ordered argument matrix (as returned by split_args). The names
    should be like -x or --long_name. This returns all matching rows and removes those rows from
    the arguments matrix.
    """
    matches = [i for i,row in enumerate(args) if row[0] in names]
    extracted = [args[i] for i in matches]
    for i in reversed(matches): del args[i]
    return extracted

def main():
    from os.path import realpath, isfile
    from sys import argv, exit
    from glob import iglob
    import shlex
    from .images import FilteredImageStack
    
    ##### Parse Arguments #####
    # Seperate out the arguments into 4 sections:
    #   general - the arguments that come before the input stack
    #   input   - the arguments that define the input stack (the -i and [] are removed)
    #   filter  - the arguments that define the filters to be run
    #   output  - ths arguments that define the output stack (the -o, [], and -a are removed)
    args = argv[1:]
    append = False

    ## Find and expand @argfile arguments
    # This loop goes through all @ arguments in reverse order providing the index of the @ argument
    for i in reversed([i for i,a in enumerate(args) if len(a)>0 and a[0] == '@']):
        with open(args[i][1:], 'r') as argfile: args[i:i+1] = shlex.split(argfile.read(), True) # read the file, split it, and insert in place of the @ argument

    if len(args) == 0: help_msg()
    
    # Input stack
    try:
        i = args.index("-i")
        if i == len(args) - 1: help_msg(2, "Must include input stack after -i.")
        general_args = args[:i]
        input_args = args[i+1:]
        if input_args[0] == "[":
            try:
                end = input_args.index("]")
                if end ==  1: help_msg(2, "No files given in input filename list.")
                input_args, filter_args = input_args[1:end], input_args[end+1:]
            except ValueError: help_msg(2, "No terminating ']' for input filename list.")
        else:
            input_args, filter_args = input_args[0], input_args[1:]
    except ValueError:
        general_args = args
        input_args = None
        filter_args = []
            
    # Output stack
    try:
        i = filter_args.index("-o")
        if i == len(filter_args) - 1: help_msg(2, "Must include output stack after -o.")
        filter_args, output_args = filter_args[:i], filter_args[i+1:]
        if output_args[0] in ("-a", "--append"):
            append = True
            del output_args[0]
        if output_args[0] in ("["):
            try:
                end = output_args.index("]")
                if end ==  1: help_msg(2, "No files given in output filename list.")
                if end != len(output_args) - 1: help_msg(2, "The output stack must be the last argument.")
                output_args = output_args[1:-1]
            except ValueError: help_msg(2, "No terminating ']' for output filename list.")
        elif len(output_args) != 1: help_msg(2, "The output stack must be the last argument.")
        else: output_args = output_args[0]
    except ValueError:
        output_args = None
        if len(filter_args) > 0: help_msg(2, "Filters cannot be used unless outputting to a file.")


    # General arguments
    general_args = split_args(general_args)

    help_args = extract_args(general_args, ('-h', '--help'))
    if len(help_args) > 1: help_msg(2, "Can only be one -h/--help argument.")
    if len(help_args) == 1:
        if len(help_args[0]) > 2: help_msg(2, "-h/--help can take at most more value")
        if len(help_args[0]) == 2: help_adv(help_args[0][1])
        help_msg()
    
    verbose_args = extract_args(general_args, ('-v', '--verbose'))
    if len(verbose_args) > 1: help_msg(2, "Can only be one -v/--verbose argument.")
    verbose = len(verbose_args) == 1
    if verbose and len(verbose_args[0]) > 1: help_msg(2, "-v/--verbose does not take any values")

    if len(general_args) > 0: help_msg(2, "Unknown general arguments provided: " + ", ".join(a[0] for a in general_args))
    

    # Filters
    filters = [FilteredImageStack.parse_cmd_line(f) for f in split_args(filter_args)]


    ##### Open Input Stack #####
    if args == None: help_msg(2, "You must provide an input image stack.")
    ims, in_name = get_input(input_args)
    if verbose:
        print "----------------------------------------"
        print "Input Image Stack: %s" % in_name
        ims.print_detailed_info()
        print "----------------------------------------"
        if output_args == None: exit(0)
    elif output_args == None:
        print str(ims)
        exit(0)
    
    ##### Process Filters #####
    for flt,args,kwargs in filters:
        ims = FilteredImageStack.create(ims, flt, *args, **kwargs)
        if verbose: print str(ims) or "<filter without description>"

    ##### Save Output Stack #####
    ims, out_name = get_output(output_args, ims, append)
    ims.save()
    
    if verbose:
        print "----------------------------------------"
        print "Output Image Stack: %s" % out_name
        if append: print "Appended data to output instead of overwriting"
        ims.print_detailed_info()
        print "----------------------------------------"

if __name__ == "__main__": main()
