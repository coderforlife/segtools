#!/usr/bin/env python

"""Command line program to convert an image stack to another stack by processing each slice."""

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
    from general.utils import get_terminal_width
    from .images.io import ImageStack
    # TODO: from .images import imfilter_util
    w = max(get_terminal_width()-1, 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    lst_itm = TextWrapper(width = w, initial_indent = ' * ', subsequent_indent = ' '*3)
    if msg != None: print >> stderr, fill(msg, w)
    print fill("Image Stack Reader and Converter Tool", w)
    print ""
    print "===== General Usage ====="
    print tw.fill("%s [args] -i [-l] input [filters] [-o [-l] output]" % basename(argv[0]))
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -v  --verbose   Display all information about image stack(s)")
    print tw.fill("  -z indices      The slice indices to use from the input-stack, a comma-seperated list of single indicies or number ranges* with stops")
    print tw.fill("  -a  --append    Append data to the output-stack instead of creating a new output-stack, single-file output stacks must exist")
    print ""
    print fill("The input and output stacks can either be a single 3D image file or a collection of same-sized 2D image files.", w)
    print ""
    print fill("For image formats that support 3D images, simply give the filename after -i or -o. The supported formats for this are:", w)
    for l in ImageStack.supported_list(): print lst_itm.fill(l)
    print fill("To specify the options, put a colon (:) after the filename and seperate each option with a comma.", w)
    print ""
    print fill("For collections of 2D image files (including PNG, TIFF, BMP, JPEG, MHA/MHA, IM, and MAT) specify either a numeric pattern or a list of files. If using a list of files you must start the list with -l.", w)
    print ""
    print fill("To use a numeric pattern use number signs (#) in place of the slice number. The filename may be followed by a colon (:) and a number range*. For input-stacks it will take all filenames with any number of digits in place of the # signs and use them in numerical order. If a number range is provided it limits the values allowed in place of the # signs. For output-stacks the number of # signs dictactes how numbers will be padded with 0s. Output-stacks start at value 0 by default.", w)
    print ""
    print fill("===== Identify Usage =====", w)
    print fill("If an output-stack is not provided then information about the input stack is printed out.", w)
    print ""
# TODO:
#    print fill("===== Convert Usage =====", w)
#    print fill("Convert the input-stack to the output-stack possibly running image filters on the slices.", w)
#    for l in imfilter_util.usage: print tw.fill(l) if len(l) > 20 and l[0] == ' ' else fill(l, w)
#    print ""
    print "* Number ranges are specified as one of <start>-<stop>, <start>-<stop>@<step>, <start>, or <start>@<step> where start and stop are non-negative integers and step is a positive integer. A missing stop value means go till the end and step defaults to 1."
    exit(err)

def get_opts(s):
    x = s[2:].rsplit(':',1)
    return s[:2]+x[0], dict(x.split('=',1) for x in x[1].split(',')) if len(x) == 2 else {}

def get_input(args, readonly=True):
    """
    Gets the input stack from the arguments after -i before -o (or the end) and returns the ImageStack instance and a textual version.
    """
    from .images import ImageStack
    if args == None or len(args) == 0: help_msg(2, "You must provide an input image stack.")
    elif args[0] == '-l':
        if not readonly: help_msg(2, "You cannot append to an image stack that is a fixed list of files, you must use a numeric pattern instead.")
        files = []
        for i in args[1:]:
            if not isfile(i): help_msg(3, "File '%s' does not exist." % i)
            files.append(i)
        if len(files) == 0: help_msg(2, "You need to provide at least one image after -l.")
        try: return ImageStack.open(files, readonly), " ".join(files)
        except Exception as e: help_msg(2, "Failed to open input image-stack '"+(" ".join(files))+"': "+str(e))
    elif len(args) != 1: help_msg(2, "You must provide only one argument after -i if not using -l.")
    else:
        name = args[0]
        m = numeric_pattern.search(name)
        if m == None:
            name, options = get_opts(name)
            try: return ImageStack.open(name, readonly, **options), name
            except Exception as e: raise; help_msg(2, "Failed to open input image-stack '"+name+"': "+str(e))
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
        try: return ImageStack.open([f for i,f in files], readonly, **options), name
        except Exception as e: help_msg(2, "Failed to open input image-stack '"+name+"': "+str(e))
        
def get_output(args, in_stack, append=False):
    """
    Gets the output stack from the arguments after -o till the end and returns the ImageStack instance and a textual version.
    """
    import os.path
    from .general.utils import make_dir
    from .images import ImageStack
    if args == None or len(args) == 0: return None, None
    elif append: return get_input(args, False)
    elif args[0] == '-l':
        files = arg[1:]
        if len(files) != len(in_stack): help_msg(2, "When using a file list for the output stack it must have exactly one file for each slice in the input-stack.")
        if any(not make_dir(os.path.dirname(f)) for f in files if os.path.dirname(f) != ''): help_msg(2, "Failed to create new image-stack because the file directories could not be created.")
        try: return ImageStack.create(files, (in_stack.h, in_stack.w), in_stack.dtype), " ".join(files)
        except Exception as e: help_msg(2, "Failed to create new image-stack '"+(" ".join(files))+"': "+str(e))
    elif len(args) != 1: help_msg(2, "You must provide only one argument after -o if not using -l.")
    else:
        name = args[0]
        m = numeric_pattern.search(name)
        if m == None:
            name, options = get_opts(name)
            try: return ImageStack.create(name, (in_stack.h, in_stack.w), in_stack.dtype, **options), name
            except Exception as e: help_msg(2, "Failed to create new image-stack '"+name+"': "+str(e))
        g = m.groups()
        before, digits, after = g[:3]
        ndigits = len(digits)
        start = g[3] or 0
        stop = g[4] # inclusive, None for infinite
        step = g[5] or 1 # if it was not provided or given as 0 then we get 1
        if stop != None and (stop-start)//step != len(in_stack): help_msg(2, "When using numerical pattern with a file range it must cover the exact number of slices in the input stack (easiest to just leave of the upper bound).")
        files = [before + str(i*step+start).zfill(ndigits) + after for i in xrange(len(in_stack))]
        if any(not make_dir(os.path.dirname(f)) for f in files if os.path.dirname(f) != ''): help_msg(2, "Failed to create new image-stack because the file directories could not be created.")
        try: return ImageStack.create(files, (in_stack.h, in_stack.w), in_stack.dtype, pattern=before+("%%0%dd"%ndigits)+after, start=start, step=step), name
        except Exception as e: help_msg(2, "Failed to create new image-stack '"+name+"': "+str(e))

if __name__ == "__main__":
    from os.path import realpath, isfile
    from sys import argv, exit
    from getopt import getopt, GetoptError
    from glob import iglob
    
    from .images import dtype2desc #imfilter_util
    
    args = argv[1:]
    try:
        i = args.index("-o")
        args, output = args[:i], args[i+1:]
    except ValueError: output = None
    try:
        i = args.index("-i")
        args, input = args[:i], args[i+1:]
    except ValueError: input = None
    
    try: opts, args = getopt(args, "hvz:l", ["help","verbose","append"]) #+imfilter_util.getopt_short, +imfilter_util.getopt_long
    except GetoptError as err: help_msg(2, str(err))

    ##### Parse arguments #####
    z = []
    imfilters = []
    append = False
    verbose = False
    for i,(o,a) in enumerate(opts):
        if o == "-h" or o == "--help": help_msg()
        elif o == "-a" or o == "--append": append = True
        elif o == "-v" or o == "--verbose": verbose = True
        elif o == "-z":
            for p in a.split(","):
                if p.isdigit(): z.append(int(p)) # single digit
                else: # range of numbers
                    m = number_range.search(p)
                    if m == None: help_msg(2, "Invalid z argument supplied.")
                    g = m.groups()
                    start = g[0] or 0
                    stop = g[1] # inclusive, None for infinite (not allowed for z argument)
                    if stop == None: help_msg(2, "Invalid z argument supplied.")
                    step = g[2] or 1 # if it was not provided or given as 0 then we get 1
                    z.extend(range(int(start), int(stop)+1, int(step)))
        elif o == "-l": help_msg(2, "-l can only be used immediately after -i or -o.")
        # TODO: else: imfilters.append(imfilter_util.parse_opt(o,a,help_msg))
        else: help_msg(2, "currently no other ooptions are supported.")
    # TODO: imf, imf_names = imfilter_util.list2imfilter(imfilters)
    if len(args) != 0: help_msg(2, "Filenames can only come after -i or -o.")

    # Open input stack
    in_stack, in_name = get_input(input)
    if z:
        if min(z) < 0 or max(z) >= len(in_stack): help_msg(2, "z argument supplied is out of range for given input-stack.")
        in_stack = in_stack[z]

    # Create output stack
    # TODO: get output type from filters
    out_stack, out_name = get_output(output, in_stack, append)

    ##### Print information #####
    if verbose:
        print "----------------------------------------"
        print "Input Image Stack: %s" % in_name
        print "Dimensions (WxHxD): %d x %d x %d" % (in_stack.w, in_stack.h, in_stack.d)
        print "Data Type:   %s" % dtype2desc(in_stack.dtype)
        sec_bytes = in_stack.w * in_stack.h * in_stack.dtype.itemsize
        print "Bytes/Slice: %d" % sec_bytes
        print "Total Bytes: %d" % (in_stack.d * sec_bytes)
        print "Handler:     %s" % type(in_stack).__name__
        if len(in_stack.header) == 0: print "No header information"
        else:
            print "Header:"
            for k,v in in_stack.header.iteritems(): print "  %s = %s" % (k,v)
        print "----------------------------------------"
        print "Using Slices: %s" % ("(all)" if len(z) == 0 else (", ".join(z)))
        # TODO: print "Image Filters: %s" % ("(none)" if len(imf_names) == 0 else ", ".join(imf_names))
        print "----------------------------------------"
        if output != None:
            print "Output Image Stack: %s" % out_name
            if append: print "Appending data to output instead of overwriting"
            print "Dimensions (WxHxD): %d x %d x %d" % (out_stack.w, out_stack.h, len(in_stack))
            print "Data Type:   %s" % dtype2desc(out_stack.dtype)
            sec_bytes = out_stack.w * out_stack.h * out_stack.dtype.itemsize
            print "Bytes/Slice: %d" % sec_bytes
            print "Total Bytes: %d" % (len(in_stack) * sec_bytes)
            print "Handler:     %s" % type(out_stack).__name__
            if len(out_stack.header) == 0: print "No header information"
            else:
                print "Header:"
                for k,v in out_stack.header.iteritems(): print "  %s = %s" % (k,v)
            print "----------------------------------------"
        else: exit(0)
    elif output == None:
        print "%dx%dx%d %s" % (in_stack.w, in_stack.h, in_stack.d, dtype2desc(in_stack.dtype))
        exit(0)

    ##### Convert all slices #####
    # TODO: for im in iter(in_stack): out_stack.append(imf(im))
    for im in iter(in_stack): out_stack.append(im)
