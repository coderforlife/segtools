"""Here are the "built-in" commands for imstack: select and remove"""

from imstack import Command, Help

class SelectCommand(Command):
    @classmethod
    def name(cls): return 'select'
    @classmethod
    def flags(cls): return ('s', 'select')
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Selecting Image Stacks")
        p.text("""
Since numerous image stacks may be open simultaneously and you may want to not work with last image
stack produce you need a way to select which image stack to work with. With this command you can
also duplicate immage stacks.""")
        print ""
        p.flags(cls.flags())
        print ""
        p.text("Command format:")
        p.cmds("--select image-stack-numbers-or-slices ...")
        print ""
        p.text("""
The command is simply followed by a series or numbers or slices. These are the image stacks that you
wish to move to the front of the line, to be worked with next. A value of 0 would be the image stack
that is already going to be next, 1 is second-to-next, and so forth. You can also use negative
values, where -1 would be the image stack that is last-to-be-used, -2 would be penultimate, etc. The
last number listed will end up being the next stack used.

Instead of integers you can also use slices, which are specified as [first][:step]:[last] which will
expand into all integers from first to last (inclusive of first and last) that increment by step.
Each part is optional, with step defaulting to one, and first and last defaulting to the ends of the
stack - for whatever makes sense for the step.

If a number is repeated then it will be placed twice, resulting in a duplicate.

Examples:""")
        p.list("--select 0 1 2  # top three image stacks to switch",
               "--select 0:2    # same as above",
               "--select 0 0    # adds a copy of the top image stack to the top",
               "--select -1     # bottom image stack goes to the top",
               "--select :-2:   # every other image stack is brought to the top")
        print ""
        p.text("See also:")
        p.list('remove')
    def __str__(self): return "Selecting "+(", ".join(self._inds))
    def __init__(self, args, stack):
        if len(args.positional) == 0: raise ValueError("No image stacks selected")
        if len(args.named) != 0: raise ValueError("No named options accepted")
        last_ind = len(stack)-1
        self._inds = []
        try:
            for arg in args.positional:
                arg = arg.strip()
                if ':' in arg:
                    # slice: stop is inclusive and different order from how Python normally operates
                    parts = [int(i) if len(i) > 0 else None for i in arg.split(':', 2)]
                    start, stop, step = parts[0], parts[-1], ((parts[2] if len(parts) == 2 else None) or 1)
                    if step > 0:
                        start = 0         if start is None else ((ls+start) if start < 0 else start)
                        stop  = (last_ind if stop  is None else ((ls+stop)  if stop  < 0 else stop))+1
                    else: # step < 0
                        start = last_ind if start is None else ((ls+start) if start < 0 else start)
                        stop  = (0       if stop  is None else ((ls+stop)  if stop  < 0 else stop))-1
                    self._inds.extend(xrange(start, stop, step))
                else:
                    i = int(arg)
                    if i < 0: i += last_ind
                    self._inds.append(i)
        except ValueError: raise ValueError("Only integers and slices ([first][:step]:[last]) are allowed")
        self.execute(stack)
    def execute(self, stack): stack.select(self._inds)

class RemoveCommand(SelectCommand):
    @classmethod
    def name(cls): return 'remove'
    @classmethod
    def flags(cls): return ('r', 'remove')
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Removing Image Stacks")
        p.text("Sometimes commands produce image stacks that you don't actually want to use and this command allows you to remove them.")
        print ""
        p.flags(cls.flags())
        print ""
        p.text("Command format:")
        p.cmds("--remove image-stack-numbers-or-slices ...")
        print ""
        p.text("""
The command is simply followed by a series or numbers or slices. These are the image stacks that you
wish to remove from being processed. A value of 0 would be the image stack that is going to be next,
1 is second-to-next, and so forth. You can also use negative values, where -1 would be the image
stack that is last-to-be-used, -2 would be penultimate, etc.

Instead of integers you can also use slices, which are specified as [first][:step]:[last] which will
expand into all integers from first to last (inclusive of first and last) that increment by step.
Each part is optional, with step defaulting to one, and first and last defaulting to the ends of the
stack - for whatever makes sense for the step.

Examples:""")
        p.list("--remove 0 1 2  # remove next three image stacks",
               "--remove 0:2    # same as above",
               "--remove -1     # remove the bottom image stack",
               "--remove :2:    # remove every other image stack")
        print ""
        p.text("See also:")
        p.list('select')
    def __str__(self): return "Removing "+(", ".join(self._inds))
    def __init__(self, args, stack):
        if len(args.positional) == 0: raise ValueError("No image stacks to remove")
        super(RemoveCommand, stack).__init__(args, stack)
        self._inds = list(sorted(set(self._inds)))
    def execute(self, stack): stack.remove(self._inds)
