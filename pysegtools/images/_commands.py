"""
These are the basic image-stack processing commands. They aren't really filters as they don't change
the data in the image slices but instead work on changing which slices we are using. They generally
create ImageStackCollection image stacks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import chain, izip_longest, islice

from ..general import itr2str
from ._stack import ImageStack, ImageSlice, ImageStackArray, ImageStackCollection, Homogeneous
from ..imstack import Command, Help, Opt

class ZCommand(Command):
    @classmethod
    def name(cls): return 'z'
    @classmethod
    def flags(cls): return ('z',)
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Select Image Slices")
        p.text("""Select, reorder, and duplicate Z slices from a stack.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.stack_changes(consumes=("Image stack to reorganize",), produces=("Reorganized image stack",))
        p.newline()
        p.text("Command format:")
        p.cmds("-z image-slice-numbers-or-slices ...")
        p.newline()
        p.text("""
The command is simply followed by a series of numbers or slices. These are the image slices that you
wish to keep and their order. If the integers are negative then they are relative to the last image
slice.

Instead of integers you can also use slices, which are specified as [first][:step]:[last] which will
expand into all integers from first to last (inclusive of first and last) that increment by step.
Each part is optional, with step defaulting to one, and first and last defaulting to the ends of the
stack - for whatever makes sense for the step.

If a particular slice is specified twice, it will be duplicated.

Examples:""")
        p.list("-z 0 1 2    # keep first three image slices",
               "-z 0:2      # same as above",
               "-z 0 0      # have a copy of the top image slices and nothing else",
               "-z -1       # keep last image slice only",
               "-z :-1:     # reverse the order of the image stack (same as --flip z)",
               "-z :2:      # keep all every other slice",
               "-z :2: 1:2: # interleave even and odd slices")
        p.newline()
        p.text("See also:")
        p.list('flip', 'split', 'combine')

    @staticmethod
    def __fmt(idx):
        if isinstance(idx, int): return idx
        start, stop, step = idx.start, idx.stop, idx.step
        return (('' if start is None else str(start)) +
                ('' if step==1 else ':'+str(step)) + ':' +
                ('' if stop is None else str(stop)))
    def __str__(self): return "selecting slices "+itr2str((ZCommand.__fmt(i) for i in self.__inds),", ")
    def __init__(self, args, stack):
        if len(args.positional) == 0: raise ValueError("No image slices selected")
        if len(args.named) != 0: raise ValueError("No named options are accepted")
        stack.pop()
        stack.push()
        self.__inds = []
        try:
            for arg in args.positional:
                arg = arg.strip()
                if ':' in arg:
                    # slice: different order from how Python normally operates
                    parts = [int(i) if len(i) > 0 else None for i in arg.split(':', 2)]
                    start, stop, step = parts[0], parts[-1], ((parts[1] if len(parts) == 2 else None) or 1)
                    self.__inds.extend(slice(start, stop, step)) # not a real slice since stop is inclusive
                else: self.__inds.append(int(arg))
        except ValueError: raise ValueError("Only integers and slices ([first][:step]:[last]) are allowed")
    def execute(self, stack):
        ims = stack.pop()
        last_ind = len(ims)-1
        inds = []
        #all_neg = True
        for i in self.__inds:
            if isinstance(i, slice):
                # slice: stop is inclusive (not normally in Python)
                start, stop, step = i.start, i.stop, i.step
                if step > 0:
                    start = 0         if start is None else ((last_ind+start) if start < 0 else start)
                    stop  = (last_ind if stop  is None else ((last_ind+stop)  if stop  < 0 else stop))+1
                    #all_neg = False
                else: # step < 0
                    start = last_ind if start is None else ((last_ind+start) if start < 0 else start)
                    stop  = (0       if stop  is None else ((last_ind+stop)  if stop  < 0 else stop))-1
                inds.extend(xrange(start, stop, step))
            else:
                inds.append((i+last_ind) if i < 0 else i)
                #all_neg = False
        out = ImageStackCollection(ims[inds])
        #pylint: disable=protected-access
        out._ims = ims # make sure we save a reference to the image stack so it doesn't get cleaned up
        stack.push(out)

class SplitCommand(Command):
    @classmethod
    def name(cls): return 'split'
    @classmethod
    def flags(cls): return ('split',)
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Split Image Stack")
        p.text("""Split a stack into two stacks.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.stack_changes(consumes=("Image stack to split",), produces=("Selected image slices","Complementary image stack"))
        p.newline()
        p.text("Command format:")
        p.cmds("--split [first][:step]:[last]")
        p.newline()
        p.text("""
The command is followed by a 'slice'. This desribes the image slices to put into one of the stacks.
The other stack is formed from the complement of that stack. The 'slice' describes all slices to
use, including first and last that are multiples of step from first. Each part is option, with step
defaulting to 1, and first and last defaulting to the ends of the stack - for whatever makes sense
for the step. At least one of the parts must be given a non-default value. Additionally the variable
n can be used in as a substitute for the number of image slices in the source stack and used with
simple math.

Examples:""")
        p.list("--split :2:  # split into stacks with all even slices and with all odd slices",
               "--split :n/2 # split into into first half and second")
        p.newline()
        p.text("See also:")
        p.list('z', 'combine')

    def __str__(self): return "spliting stack into %s and the inverse"%self.__raw

    @staticmethod
    def __safe_eval(x, n):
        # At this point we have also already sanitized and made sure there are only digits, operators, and n
        return eval(x, globals={'__builtins__':{}}, locals={'n':n}) #pylint: disable=eval-used

    @staticmethod
    def __cast(x):
        if 'n' not in x: return int(x)
        if any((c not in '0123456789*/+-()n') for c in x): raise ValueError
        try: SplitCommand.__safe_eval(x, 1)
        except: raise ValueError()
        return x

    def __init__(self, args, stack):
        if len(args.positional) != 1: raise ValueError("Exaclty one option required")
        if len(args.named) != 0: raise ValueError("No named options are accepted")
        stack.pop()
        stack.push()
        stack.push()
        try:
            self.__raw = arg = args[0].strip()
            if ':' not in arg: raise ValueError
            # slice: different order from how Python normally operates and can have 'n'
            parts = [SplitCommand.__cast(i) if len(i) > 0 else None for i in arg.split(':', 2)]
            self.__start, self.__stop, self.__step = parts[0], parts[-1], ((parts[1] if len(parts) == 2 else None) or 1)
        except ValueError: raise ValueError("Option must be a single slice ([first][:step]:[last]) with at least one non-default value set")
    def execute(self, stack):
        ims = stack.pop()
        last_ind = len(ims)-1
        inds = []
        # slice: stop is inclusive (not normally in Python) and can include 'n'
        start, stop, step = (SplitCommand.__safe_eval(self.__start, last_ind),
                             SplitCommand.__safe_eval(self.__stop, last_ind),
                             SplitCommand.__safe_eval(self.__step, last_ind))
        forward = step > 0
        if forward:
            start = 0         if start is None else ((last_ind+start) if start < 0 else start)
            stop  = (last_ind if stop  is None else ((last_ind+stop)  if stop  < 0 else stop))+1
        else: # step < 0
            start = last_ind if start is None else ((last_ind+start) if start < 0 else start)
            stop  = (0       if stop  is None else ((last_ind+stop)  if stop  < 0 else stop))-1
        inds.extend(xrange(start, stop, step))
        u_inds = frozenset(inds)
        c_inds = xrange(0, last_ind+1, 1) if forward else xrange(last_ind, -1, -1)
        c_inds = (i for i in c_inds if i not in u_inds)
        stack.push(ImageStackCollection(ims[c_inds]))
        stack.push(ImageStackCollection(ims[inds]))

class CombineCommand(Command):
    @classmethod
    def name(cls): return 'combine'
    @classmethod
    def flags(cls): return ('C', 'combine')
    @classmethod
    def _opts(cls): return (
            Opt('nstacks',    'The number of stacks to combine', Opt.cast_int(lambda x:x>=2), 2),
            Opt('interleave', 'If the combined stacks should be interleaved instead of concatenated', Opt.cast_bool(), False),
            )
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Combine Image Stacks")
        p.text("""
Combines two or more image stacks into a single stack. The next image stack is placed on top,
followed by the second-to-next, and so forth.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("""
Consumes:  2+ image stacks 
Produces:  1 image stack""")
        p.newline()
        p.text("Command format:")
        p.cmds("-C [nstacks] [interleave]")
        p.newline()
        p.text("Options:")
        p.opts(*cls._opts())
        p.newline()
        p.text("See also:")
        p.list('z', 'split')
    def __str__(self): return ("interleaving" if self.__interleave else "combining")+(" %d image stacks"%self.__nstacks)
    def __init__(self, args, stack):
        self.__nstacks, self.__interleave = args.get_all(*CombineCommand._opts())
        for _ in xrange(self.__nstacks): stack.pop()
        stack.push()
    def execute(self, stack):
        ims = [stack.pop() for _ in xrange(self.__nstacks)]
        if self.__interleave:
            itr = (x for x in chain.from_iterable(izip_longest(*ims)) if x is not None)
        else:
            itr = chain(*ims)
        stack.push(ImageStackCollection(itr))

class MemCacheCommand(Command):
    @classmethod
    def name(cls): return 'memcache'
    @classmethod
    def flags(cls): return ('M', 'memcache')
    @classmethod
    def _opts(cls):
        from tempfile import gettempdir
        return (
            Opt('memmap', 'Use a memory-mapped physical backing store for the cache', Opt.cast_bool(), False),
            Opt('tmp', 'Temporary directory to put memory-mapped files in', Opt.cast_writable_dir(), gettempdir()),
            )
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Memory Cache")
        p.text("""
This commands reads all available image stacks (causing them read from disk, compute, etc) into
cached image stacks so that they are never re-read or re-computed if they are used in multiple
places (which may not be explicit, for example a crop command may double-read data by
requesting it once to compute the padding and once to actually crop). This command can greatly speed
up future commands at the cost of using more memory.

If you are likely to run out of virtual memory using these caches, you can provide the memmap
argument to have the caches backed by physical storage via memory-mapped files. They will only be
written to disk as the OS deems necessary so it is unlikely to slow things down too much. Note that
on Linux the system-default for tmp is frequently on a tmpfs or shmfs filesystem which is simply
backed by swap, so specifying that will not help reduce swap load.

After things are cached, the garbage collector is run and all previous image stacks should be
removed from memory, causing release of memory. This means when using multiple memory cache commands
only the last two need to fit in memory (along with the intermediate steps).

One downside of using memory caches is that the some features of the original image stacks will be
lost, such as headers for image stacks on disk. The features lost are mainly useful for debugging
with the info command, which can be placed immediately before this command if needed.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("Command format:")
        p.cmds("-M [memmap] [tmp]")
        p.newline()
        p.text("Options:")
        p.opts(*cls._opts())
    def __str__(self): return "memory-mapped caching" if self.__memmap else "memory caching"
    def __init__(self, args, _):
        self.__memmap, self.__tmpdir = args.get_all(*MemCacheCommand._opts())
    def execute(self, stack):
        import gc
        imss = [stack.pop() for _ in xrange(len(stack))] # pop all image stacks
        imss.reverse() # make sure that they will stay the same order
        if self.__memmap:
            for ims in imss: stack.push(MemMapCacheImageStack(ims, tmpdir=self.__tmpdir))
        else:
            from numpy import empty
            for ims in imss:
                info = [(im.shape, im.dtype) for im in ims]
                if all(i == info[0] for i in islice(info, 1, None)):
                    # Homogeneous - single 3D array
                    sh, dt = info[0]
                    stk = empty((ims.d,) + sh, dtype=dt)
                    for z, slc in enumerate(ims): stk[z,:,:,...] = slc.data
                    stack.push(ImageStackArray(stk))
                else:
                    # Heterogeneous - one array per slice
                    stack.push(ImageStackCollection(slc.data for slc in ims))
        gc.collect() # TODO: run some tests to make sure we aren't leaking objects (unreachable cycle with __del__)
        
class MemMapCacheImageStack(ImageStack):
    __file = None
    def __init__(self, ims, tmpdir=None):
        from numpy import ndarray, cumsum
        slices = ims[:]
        info = [(im.dtype, im.shape) for im in slices]
        d = len(slices)
        if d == 0: return
        homogeneous = all(i == info[0] for i in islice(info, 1, None))

        if homogeneous:
            dt,sh = info[0]
            self._arr = ndarray((d,)+sh, dt, self.__open((sh[0]*sh[1]*dt.itemsize)*d, tmpdir))
            for z, slc in enumerate(slices): self._arr[z,:,:,...] = slc.data
            self._arr.flags.writeable = False
            super(MemMapCacheImageStack, self).__init__(
                [MemMapImageSlice(self,z,im,dt,sh) for z,im in enumerate(self._arr)])
            self._h, self._w = sh
            self._shape = sh
            self._dtype = dt
            self._homogeneous = Homogeneous.All
        else:
            nbytes = [sh[0]*sh[1]*dt.itemsize for dt,sh in info]
            nbytes_aligned = [(x - x % -4) for x in nbytes]
            offsets = [0] + cumsum(nbytes_aligned).tolist()
            mm = self.__open(offsets.pop(), tmpdir)
            ims = [ndarray(sh, dt, mm, off) for (dt,sh),off in zip(info,offsets)]
            for im,slc in zip(ims, slices):
                im[:,:,...] = slc.data
                im.flags.writeable = False
            super(MemMapCacheImageStack, self).__init__(
                [MemMapImageSlice(self,z,im,dt,sh) for (z,im),(dt,sh) in zip(enumerate(ims),info)])

    def __open(self, size, tmpdir):
        from os import name
        from tempfile import TemporaryFile
        from mmap import mmap, ACCESS_WRITE
        self.__file = TemporaryFile('wb', 0, dir=tmpdir)
        if name == 'nt':
            return mmap(self.__file.fileno(), size, access=ACCESS_WRITE)
        else:
            self.__file.truncate(size)
            return mmap(self.__file.fileno(), size, access=ACCESS_WRITE)

    @ImageStack.cache_size.setter
    def cache_size(self, value): pass # prevent built-in caching - this is a cache! #pylint: disable=arguments-differ
    def close(self):
        self.__file.close()
        self.__file = None
    def __delete__(self): self.close()
    @property
    def stack(self): return self._arr # only if homogeneous

class MemMapImageSlice(ImageSlice):
    def __init__(self, stack, z, im, dt, sh):
        super(MemMapImageSlice, self).__init__(stack, z)
        self._set_props(dt, sh)
        self._im = im
    def _get_props(self): pass
    def _get_data(self): return self._im
