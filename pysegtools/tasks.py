"""
Provides a Tasks class that creates a logical tree of tasks, indicating which tasks need to wait for
other tasks to finish before they can start (based on input and output files). It will run the whole
tree as efficiently as possible, with many processes at once if able.

The exact ordering of starting/completing tasks cannot be guarnteed, only that tasks that depend on
the output of other tasks will not start until the outputing tasks are done.

While tasks are running, the program listens to SIGUSR1. When recieved, the status of the tasks is
output, including the memory load, expected memory load, tasks running, done, and total, and a list
of all tasks that are "ready to go" (have all prerequistes complete but need either task slots or
memory to run).

On *nix and Windows systems resource usage can be obtained and saved to a log. Each line is first
the name of the task then the rusage fields (see http://docs.python.org/2/library/resource.html#resource-usage
and man 2 getrusage for more information). It will not record Python function tasks that do not run
in a seperate process. On some forms of *nix the ru_maxrss and other fields will always be 0.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Only the Tasks class along with the byte-size constants are exported
__all__ = ['Tasks', 'KB', 'MB', 'GB', 'TB', 'STDOUT', 'DEVNULL']

from abc import ABCMeta, abstractmethod
from functools import total_ordering
from collections import Sequence, Iterable
from numbers import Integral, Real

import sys, os, math
from os import getcwdu as getcwd, getpid, chdir
from os.path import exists, getmtime, join, normpath
join_norm = lambda path,*paths: normpath(join(path,*paths))

from heapq import heapify, heappop, heappush

from multiprocessing import cpu_count, Process as PyProcess
if sys.version_info[0] < 3:
    try:
        from subprocess32 import Popen, CalledProcessError, STDOUT
    except ImportError:
        from subprocess import Popen, CalledProcessError, STDOUT
        if os.name == 'posix':
            import warnings
            warnings.warn('Using the built-in subprocess module on POSIX in Python v2.7 is unreliable with tasks. Please install subprocess32.')
from threading import Condition, Thread
from pipes import quote

from calendar import timegm
from time import gmtime, sleep, strftime, strptime, time
from datetime import datetime
import re

from psutil import Process, virtual_memory, Error as ProcessError # not a built-in library
try: import saga # not a built-in library, but only required when using clustering
except ImportError: saga = None

# TODO: psutil now support futures, we don't need to do all the extra process management stuff

#STDOUT = STDOUT
DEVNULL = STDOUT-1

String = str if (sys.version_info[0] == 3) else basestring

this_proc = Process(getpid())
time_format = '%Y-%m-%d %H:%M:%S' # static, constant

def get_mem_used_by_tree(proc = this_proc):
    """
    Gets the memory used by a process and all its children (RSS). If the process is not provided,
    this process is used. The argument must be a pid or a psutils.Process object. Return value is in
    bytes.
    """
    # This would be nice, but it turns out it crashes the whole program if the process finished between creating the list of children and getting the memory usage
    # Adding "if p.is_running()" would help but still have a window for the process to finish before getting the memory usage
    #return sum((p.get_memory_info()[0] for p in proc.get_children(True)), proc.get_memory_info()[0])
    if isinstance(proc, Integral): proc = Process(proc) # was given a PID
    mem = proc.memory_info()[0]
    for p in proc.children(True):
        try:
            if p.is_running():
                mem += p.memory_info()[0]
        except ProcessError: pass
    return mem

def get_time_used_by_tree(proc = this_proc):
    """
    Gets the CPU time used by a process and all its children (user+sys). If the process is not
    provided, this process is used. The argument must be a pid or a psutils.Process object. Return
    value is in seconds.
    """
    if isinstance(proc, Integral): proc = Process(proc) # was given a PID
    t = sum(proc.cpu_times())
    for p in proc.children(True):
        try:
            if p.is_running():
                t += sum(p.cpu_times())
        except ProcessError: pass
    return t

def write_error(s):
    """
    Writes out an error message to stderr in red text. This is done so that the error messages from
    the Tasks system can be easily distinguished from the errors from the underlying commands being
    run. If we cannot change the text color (not supported by OS or redirecting to a file) then just
    the string is written.
    """
    is_tty = False
    try: is_tty = sys.stderr.isatty()
    except AttributeError: pass
    
    if is_tty:
        if os.name == "posix": sys.stderr.write("\x1b[1;31m")
        elif os.name == "nt":
            from ctypes import windll, Structure, c_short, c_ushort, byref
            k32 = windll.kernel32
            handle = k32.GetStdHandle(-12)
            class COORD(Structure):      _fields_ = [("X", c_short), ("Y", c_short)]
            class SMALL_RECT(Structure): _fields_ = [("L", c_short), ("T", c_short), ("R", c_short), ("B", c_short)]
            class CONSOLE_SCREEN_BUFFER_INFO(Structure): _fields_ = [("Size", COORD), ("CursorPos", COORD), ("Color", c_ushort), ("Rect", SMALL_RECT), ("MaxSize", COORD)]
            csbi = CONSOLE_SCREEN_BUFFER_INFO()
            k32.GetConsoleScreenBufferInfo(handle, byref(csbi))
            prev = csbi.Color
            k32.SetConsoleTextAttribute(handle, 12)
    sys.stderr.write(s)
    if is_tty:
        if os.name == "posix": sys.stderr.write("\x1b[0m")
        elif os.name == "nt":  k32.SetConsoleTextAttribute(handle, prev)
    sys.stderr.write("\n")

def DFS(itr, above=()):
    """
    Returns the depth-first-search ordering of an iterable of iterables, emitting each of the
    non-iterable nodes and strings (so in this case strings are considered non-iterable). Does not
    support situations when an iterable contains itself, directly or indirectly (this would lead to
    a stack overflow).
    """
    if any(itr is a for a in above): raise ValueError('recursive iterables not supported')
    above += (itr,)
    return (x for i in itr for x in (DFS(i, above) if isinstance(i, Iterable) and not isinstance(i, String) else (i,)))

# These constants are for when giving a certain amount of memory pressure to a
# task. So 1 GB can be easily written as 1*GB.
KB = 1024
MB = 1024*1024
GB = 1024*1024*1024
TB = 1024*1024*1024*1024

@total_ordering
class Task(object):
    """
    Abstract Task class representing a single Task to be run. See Tasks.add for an explanation
    of the arguments.
    """
    __metaclass__ = ABCMeta
    done = False # not done yet
    _process = None # the current running process
    __all_after = None # the cache for the all_after function
    def __init__(self, name, inputs=(), outputs=(), settings=(), wd=None,
                 stdin=None, stdout=None, stderr=None, stdout_append=False, stderr_append=False,
                 mem=1*MB, cpu=1):
        #if len(outputs) == 0: raise ValueError('Each task must output at least one file')
        self.name = name         # name of this task
        if isinstance(inputs, String): inputs = (inputs,)
        if isinstance(outputs, String): outputs = (outputs,)
         # relative inputs / outputs / working directory
        self.inputs = frozenset(normpath(f) for f in inputs)
        self.outputs = frozenset(normpath(f) for f in outputs)
        self.wd = wd
        self.settings = frozenset((settings,) if isinstance(settings, String) else settings)
        self.before = set()       # tasks that come before this task
        self.after = set()        # tasks that come after this task
        # Standard streams
        self.stdin  = self.__get_std(wd, stdin )
        self.stdout = self.__get_std(wd, stdout)
        self.stderr = self.__get_std(wd, stderr, True)
        self.stdout_append = stdout_append
        self.stderr_append = stderr_append
        # CPU and memory pressure
        if isinstance(cpu, Integral):
            self._cpu = int(cpu)
            if self._cpu <= 0: raise ValueError('Number of used CPUs must be positive')
        else:
            self._cpu = float(cpu)
            if self._cpu <= 0.0 or self._cpu > 1.0: raise ValueError('Fraction of number of used CPUs must be [0.0, 1.0)')
        self._mem = int(mem)
        if self._mem < 0: raise ValueError('Amount of used memory must be non-negative')
    @staticmethod
    def __get_std(wd, std, is_stderr=False):
        if std is None or std is DEVNULL or (std is STDOUT and is_stderr): return std
        if isinstance(std, String): return normpath(std) if wd is None else join_norm(wd, std)
        raise TypeError('standard stream must be either a filename, None, or subprocess.STDOUT (stderr only)')
    def __eq__(self, other): return type(self) == type(other) and self.name == other.name #pylint: disable=unidiomatic-typecheck
    def __lt__(self, other): return type(self) <  type(other) or  type(self) == type(other) and self.name < other.name #pylint: disable=unidiomatic-typecheck
    def __hash__(self):      return hash(self.name+str(type(self)))
    def __str__(self):       return self.name
    def __repr__(self):      return '<%s task "%s">' % (self.__class__.__name__, self.name)

    def cpu(self, running):
        return int(math.ceil(self._cpu*running.cores)) if isinstance(self._cpu, float) else \
               min(running.cores, self._cpu)
    def mem(self, running):
        #pylint: disable=unused-argument
        return self._mem
    def workingdir(self, running):
        return running.workingdir if self.wd is None else join_norm(running.workingdir, self.wd)
    @abstractmethod
    def run(self, running):
        """
        Starts the task and waits, throws exceptions if something goes wrong. The argument is the
        current run, a RunningTasks object. This should call _run_proc, if not, is_running,
        current_usage, terminate, and kill must all be overridden.

        Note that this does not mark this task as done as that should be done in a controlled,
        thread-safe, manner.
        """
        pass

    def _run_proc(self, pid, wait, running):
        """
        The basic run method for seperate processes. The pid is the process number and wait is a
        function that takes no arguments, waits for the process to terminate, and returns the exit
        code. The running argument is the RunningTasks object given to run().
        """
        self._process = Process(pid)
        if running.rusagelog:
            from .general.os_ext import wait4
            _, exitcode, ru = wait4(pid, 0)
            self._process = None
            if exitcode: raise CalledProcessError(exitcode, str(self))
            running.rusagelog.write('%s %f %f %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n' % (str(self), 
                ru.ru_utime, ru.ru_stime, ru.ru_maxrss, ru.ru_ixrss, ru.ru_idrss, ru.ru_isrss,
                ru.ru_minflt, ru.ru_majflt, ru.ru_nswap, ru.ru_inblock, ru.ru_oublock,
                ru.ru_msgsnd, ru.ru_msgrcv, ru.ru_nsignals, ru.ru_nvcsw, ru.ru_nivcsw))
        else:
            exitcode = wait()
            self._process = None
            if exitcode: raise CalledProcessError(exitcode, str(self))
    @property
    def is_running(self): return self._process is not None

    # The following 3 methods only work when we are running
    def current_usage(self):
        """
        Gets the current memory (bytes) and total CPU usage (seconds) by this task. Throws
        exceptions in many cases, including if the task does not support this operation.
        """
        return get_mem_used_by_tree(self._process), get_time_used_by_tree(self._process)
    def terminate(self): self._process.terminate()
    def kill(self): self._process.kill()

    def all_after(self, back_stack=frozenset()):
        """
        Get a set of all tasks that come after this task while performing a test for cycles.
        
        This can be an expensive operation but is cached so multiple calls to it are fast. The cache
        is cleared after any tasks are added to the tree though.
        """
        if self.__all_after is None:
            if back_stack is None: back_stack = set()
            if self in back_stack: raise ValueError('Tasks are cyclic')
            back_stack = back_stack.union((self,))
            after = self.after.copy()
            for a in self.after: after.update(a.all_after(), back_stack)
            self.__all_after = frozenset(after)
        return self.__all_after
    def add_after(self, task):
        """
        Adds a task or tasks (if a sequence) to be after this task. This makes sure that this task
        is added before the given task/tasks and does some internal bookkeeping. The after and
        before sets should not be modified directly.
        """
        if isinstance(task, Task):
            self.after.add(task)
            task.before.add(self)
        else:
            self.after.update(task)
            for t in task: t.before.add(self)
        self._clear_cached_all_after()
    def _clear_cached_all_after(self):
        """Clears the cached results of "all_after" recursively."""
        if self.__all_after is not None:
            self.__all_after = None
            for b in self.before: b._clear_cached_all_after() #pylint: disable=protected-access
    def mark_as_done(self):
        """
        Marks a task and all the tasks before it as done. This means when a task tree runs that
        includes them they will not be run.
        """
        self.done = True
        for t in self.before:
            if not t.done: t.mark_as_done()
    
class TaskUsingProcess(Task):
    """
    A single Task that runs using a seperate process. 
    """
    def __init__(self, cmd, **kwargs):
        """
        Create a new Task that will run a command line using a process. The cmd can either be a
        command-line string (parseable with shlex.split) or an iterable of command-line parts that
        are converted to strings (nested iterables are expanded). Other arguments are passed
        directly to the Task constructor.
        """
        if isinstance(cmd, String):
            import shlex
            cmd = shlex.split(cmd)
        else:
            cmd = [unicode(arg) for arg in DFS(cmd)]
        self.cmd = cmd
        name = "`%s`" % " ".join(quote(s) for s in cmd)
        super(TaskUsingProcess, self).__init__(name, **kwargs)
    def run(self, running):
        owns_in, owns_out, owns_err = False, False, False
        wd = self.workingdir(running)
        try:
            def open_std(std, mode, bufsize):
                if isinstance(std, String): return open(join_norm(wd, std), mode, bufsize), True
                if std is DEVNULL: return open(os.devnull, mode, bufsize), True
                return std, False # either None or STDOUT
            stdin,  owns_in  = open_std(self.stdin,  'r', 1)
            stdout, owns_out = open_std(self.stdout, 'a' if self.stdout_append else 'w', 1)
            stderr, owns_err = open_std(self.stderr, 'a' if self.stderr_append else 'w', 0)
            p = Popen(self.cmd, cwd=wd, stdin=stdin, stdout=stdout, stderr=stderr)
            self._run_proc(p.pid, p.wait, running)
        finally:
            if owns_in:  stdin.close()
            if owns_out: stdout.close()
            if owns_err: stderr.close()

class TaskUsingCluster(TaskUsingProcess):
    """
    A single Task that runs using a seperate process, either locally or on a cluster.
    TODO: THIS IS UNTESTED (and incomplete)
    """
    def __init__(self, cmd, **kwargs):
        """See Task and TaskUsingProcess for arguments."""
        self._job = None
        super(TaskUsingCluster, self).__init__(cmd, **kwargs)
    def run(self, running):
        if not running.cluster: return super(TaskUsingCluster, self).run(running)
        
        # TODO: rusagelog
        # TODO: SGE properties: name queue project

        # Set the command to be run
        desc = saga.job.Description()
        desc.executable = self.cmd[0]
        desc.arguments = self.cmd[1:]
        #desc.environment = TODO

        # TODO: when None (redirect to current) what should we do?
        # TODO: what about appending?
        if isinstance(self.stdin, String): desc.input = self.stdin
        elif self.stdin is not DEVNULL: raise ValueError("Commands running on a cluster only support filename and DEVNULL for STDIN")
        if isinstance(self.stdout, String): desc.output = self.stdout
        elif self.stdout is not DEVNULL: raise ValueError("Commands running on a cluster only support filename and DEVNULL for STDOUT")
        if isinstance(self.stderr, String): desc.error = self.stderr
        elif self.stderr is STDOUT: pass # TODO: support stdout redirection
        elif self.stderr is not DEVNULL: raise ValueError("Commands running on a cluster only support filename, DEVUNLL, and STDOUT for STDERR")

        #desc.working_directory = self.wd # TODO

        # Set the CPU and memory hints
        desc.total_cpu_count = self._cpu # TODO: determine target's CPU capabilities and deal with fractional amounts
        if self._mem > 1*MB: desc.total_physical_memory = self._mem // MB

        # Set inputs and outputs (TODO: support NOT copying files, either because we know the files are on some shared setup or because another job produced them)
        desc.file_transfer = ([join_norm(running.workingdir, i)+" > "+i for i in self.inputs ] +
                              [join_norm(running.workingdir, o)+" < "+o for o in self.outputs])
        desc.cleanup = True

        # TODO: are the stdin/stdout/stderr files copied automatically?

        self._job = running.cluster.service.create_job(desc)
        try:
            self._job.run()
            self._job.wait()
            exitcode = self._job.exit_code
        finally:
            self._job = None
        if exitcode: raise CalledProcessError(exitcode, str(self))
    # Clustered tasks don't effect our local resource usage
    def cpu(self, running): return 0 if running.cluster else super(TaskUsingCluster, self).cpu(running)
    def mem(self, running): return 0 if running.cluster else super(TaskUsingCluster, self).mem(running)
    def current_usage(self):
        if self._job is None: return super(TaskUsingCluster, self).current_usage()
        return 0, ((time() - TaskUsingCluster._get_time(self._job.started))
                   if self._job.state == saga.RUNNING else 0)
    @property
    def is_running(self): return (self._job or self._process) is not None
    @staticmethod
    def _get_time(x):
        if   isinstance(x, Real):     return float(x)
        elif isinstance(x, String):   return strptime(x)
        elif isinstance(x, datetime): return (x-datetime.utcfromtimestamp(0)).total_seconds()
        else: raise ValueError()
    def terminate(self):
        if self._job: self._job.cancel(1)
        else: super(TaskUsingCluster, self).terminate()
    def kill(self):
        if self._job: self._job.cancel()
        else: super(TaskUsingCluster, self).kill()

class TaskUsingPythonProcess(Task):
    """
    Create a new Task that calls a Python function using a seperate process.

    Functions are wrapped to suppport changing the working directory and the standard streams along
    with also supporting callable objects that aren't functions but are pickle-able. That code is in
    _task_wrapper.
    """
    def __init__(self, func, args=(), kwargs={}, **kwargs_task): #pylint: disable=dangerous-default-value
        """
        The args and kwargs are the arguments passed to the target function. Other arguments are
        passed directly to the Task constructor.
        """
        if not callable(func): raise ValueError('not callable')
        self.func   = func
        self.args   = list(args)
        self.kwargs = dict(kwargs)
        args = [repr(a) for a in args] + ["%s=%s" % (str(k), repr(v)) for k,v in kwargs.iteritems()]
        name = "%s.%s(%s)" % (func.__module__, func.__name__, ", ".join(args))
        super(TaskUsingPythonProcess, self).__init__(name, **kwargs_task)

    def run(self, running):
        from ._task_wrapper import _pyproc
        wd = self.workingdir(running)
        def get_std(std):
            if isinstance(std, String): return join_norm(wd, std)
            if std is DEVNULL: return os.devnull
            if std is STDOUT: return -2 # the value of STDOUT, but need some known and controled constant
            return None # only option left
        p = PyProcess(target=_pyproc,
                      args=(self.func, self.args, self.kwargs, wd,
                            get_std(self.stdin),
                            get_std(self.stdout), self.stdout_append,
                            get_std(self.stderr), self.stderr_append))
        p.daemon = True
        p.start()
        def _wait(): p.join(); return p.exitcode
        self._run_proc(p.pid, _wait, running)

class Tasks(object):
    """
    Represents a set of tasks that need to be run, possibly with dependencies on each other. The
    tasks are run as efficiently as possible.
    """

    def __init__(self):
        """
        Create a new set of Tasks.
        """
        self.settings   = set() # list of setting names
        self.outputs    = {}    # key is a relative filename, value is a task that outputs that file
        self.inputs     = {}    # key is a relative filename, value is a list of tasks that need that file as input
        self.generators = set() # list of tasks that have no input files
        self.cleanups   = set() # list of tasks that have no output files
        self.all_tasks  = {}    # all tasks, indexed by their string representation
        self.__running = None

    def add(self, task, **kwargs):
        """
        Adds a new task. The task does not run until sometime after run() is called. The added task
        is retruned.

        The main argument 'task' is a Task object or one of the following which can be converted
        into a task object:
            * list/tuple - converted into a TaskUsingProcess or TaskUsingCluster
            * function - converted into a TaskUsingPythonFunction or TaskUsingPythonProcess

        If task is a Task object, there are no additional arguments. Otherwise, the arguments
        are given to the underlying Task constructor. They must all be given as keyword arguments.

        For list/tuple tasks, a seperate process is started, either locally or on a 'cluster'. The
        list/tuple is the command line to run with the first argument being the command and the
        others being the arguments. Unlike Python's subprocess module, the arguments do no need to
        be strings. If any argument is a tuple or list, it is expanded in-place. Everything else is
        converted to a string with str. If you want the process to run remotely, set the option
        'run_on_cluster' to True.

        For function tasks, a callable object is needed. Additionally, that callable can take a set
        of arguments given by the options 'args' and 'kwargs'. The function is run in a seperate
        process. This is done using multiprocessing.Process so the callable and the arguments must
        be picklable (if a function, it must be a top-level function and importable).

        Additional arguments for creation of tasks:
          inputs=(), outputs=(), settings=()
            Lists/tuples of files or names of settings, used to determine dependencies between
            individual tasks and if individual tasks can be skipped or not; each can be empty if the
            task generates files without any file input, performs cleanup tasks, or only uses files
            (note: if a string is given directly, it is automatically wrapped in a tuple); the
            inputs and outputs should be relative file names to the overall working directory - not
            the task's working directory
          wd
            the working directory of the task, relative to the working directory given to all tasks;
            by default it is the directory of the whole set of tasks (which defaults to the current
            working directory); the path to the executable (if running a comand line) should always
            be absolute as the actual current directory when looking for the executable is
            indeterminate
          stdin, stdout, stderr
            set the standard input and outputs for the task, must be a filename (relative to the
            task working directory) or tasks.DEVNULL for no input/output, stderr can also be
            tasks.STDOUT to redirect it to stdout; by default they are the same as this process
          stdout_append=False, stderr_append=False
            if stdout or stderr is a filename and the correspending append value is set to True then
            the output is appended to the file instead of replacing the file (relative to the task's
            working directory)
          mem=1*MB, cpu=1
            Sets the expected maximal amount of memory and number of CPUs this task will consume
            while running. These can effect how many tasks are run simultaneously or can determine
            which cluster a process is sent to; the memory is an ammount in bytes and the CPU is
            either a total number of CPUs (if an int) or a fraction of CPUs (if a float, rounded
            up); if a task is running locally and there is not enough memory, it will never run,
            however if there not enough cores a local task will still run by itself
        """
        if isinstance(task, Task):
            if len(kwargs) > 0: raise ValueError('Adding a task object accepts no other arguments')
        else:
            if kwargs.get('wd') == '.': del kwargs['wd']
            if callable(task):
                task = TaskUsingPythonProcess(task, **kwargs)
            elif isinstance(task, Sequence):
                task = TaskUsingCluster(task, **kwargs) if kwargs.pop('run_on_cluster', False) else \
                       TaskUsingProcess(task, **kwargs)
            else:
                raise ValueError('%s could not be added because we don\'t know how to make it a task' % task)
        return self.__add(task)
    def __add(self, task):
        """
        Actual add function. Checks the task and makes sure the task is valid for this set of tasks.
        Updates the task graph (before and after links for this and other tasks).
        """
        # Processes the input and output information from the task
        # Updates all before and after lists as well
        if not task.inputs.isdisjoint(task.outputs): raise ValueError('A task cannot output a file that it needs for input')
        # A "generator" task is one with only inputs, a "cleanup" task is one with only outputs
        is_generator, is_cleanup = len(task.inputs) == 0, len(task.outputs) == 0
        new_inputs  = task.inputs  | (self.overall_inputs() - task.outputs) # input files never seen before
        new_outputs = task.outputs | (self.overall_outputs() - task.inputs) # output files never seen before
        # Check for the creation of a cyclic dependency in tasks
        if not new_inputs.isdisjoint(new_outputs) or \
            (len(new_inputs) == 0 and not is_generator and len(self.generators) == 0) or \
            (len(new_outputs) == 0 and not is_cleanup and len(self.cleanups) == 0): raise ValueError('Task addition will cause a cycle in dependencies')
        # Add the task to the graph
        if is_cleanup: self.cleanups.add(task)
        else:
            if any(o in self.outputs for o in task.outputs): raise ValueError('Each file can only be output by one task')
            for o in task.outputs:
                self.outputs[o] = task
                if o in self.inputs: task.add_after(self.inputs[o])
        if is_generator: self.generators.add(task)
        else:
            for i in task.inputs:
                self.inputs.setdefault(i, []).append(task)
                if i in self.outputs: self.outputs[i].add_after(task)
        self.settings |= task.settings
        self.all_tasks[str(task)] = task
        return task
    def find(self, name):
        """Find a task from the string representation of the task."""
        return self.all_tasks.get(name)
    def overall_inputs(self):
        """Get the overall inputs required from the entire set of tasks."""
        return self.inputs.viewkeys() - self.outputs.viewkeys() #set(self.inputs.iterkeys()) - set(self.outputs.iterkeys())
    def overall_outputs(self):
        """Get the overall outputs generated from the entire set of tasks."""
        return self.outputs.viewkeys() - self.inputs.viewkeys() #set(self.outputs.iterkeys()) - set(self.inputs.iterkeys())
    def __check_acyclic(self):
        """Run a thorough check for cyclic dependencies. Not actually used anywhere."""
        if len(self.outputs) == 0 and len(self.inputs) == 0: return
        overall_inputs  = {t for f in self.overall_inputs() for t in self.inputs[f]}
        if (len(overall_inputs) == 0 and len(self.generators) == 0) or (len(self.overall_outputs()) == 0 and len(self.cleanups) == 0): raise ValueError('Tasks are cyclic')
        for t in overall_inputs: t.all_after()
        for t in self.generators: t.all_after()

    def display_stats(self, _signum=0, _frame=None):
        """
        Writes to standard out a whole bunch of statistics about the current status of the tasks. Do
        not call this except while the tasks are running. It is automatically registered to the USR1
        signal on POSIX systems. The signum and frame arguments are not used but are required to be
        present for the signal handler.
        """
        if self.__running is None: raise RuntimeError()
        
        print('=' * 80)

        rnng = self.__running
        mem_sys = virtual_memory()
        mem_task = get_mem_used_by_tree()
        mem_press = rnng.mem_pressure
        mem_avail = mem_sys.available - max(mem_press - mem_task, 0)
        print('Memory (GB): System: %d / %d    Tasks: %d [%d], Avail: %d' % (
            int(round((mem_sys.total - mem_sys.available) / GB)), int(round(mem_sys.total / GB)),
            int(round(mem_task / GB)), int(round(mem_press / GB)), int(round(mem_avail / GB))))

        task_done  = sum(1 for t in self.all_tasks.itervalues() if t.done)
        task_next  = len(rnng.next)
        task_run   = len(rnng.running)
        task_total = len(self.all_tasks)
        task_press = rnng.cpu_pressure
        task_max   = rnng.cores
        print('Tasks:       Running: %d [%d] / %d, Done: %d / %d, Queue: %d' % (task_run, task_press, task_max, task_done, task_total, task_next))

        print('-' * 80)
        if task_run == 0:
            print('Running: none (probably waiting for more memory)')
        else:
            print('Running:')
            for task in sorted(rnng.running):
                text = str(task)
                if len(text) > 60: text = text[:56] + '...' + text[-1]
                real_mem = '? '
                timing = ''
                try:
                    real_mem, t = task.current_usage()
                    real_mem = str(int(round(real_mem / GB)))
                    t = int(round(t))
                    hours, mins, secs = t // (60*60), t // 60, t % 60
                    timing = ('%d:%02d:%02d' % (hours, mins - hours * 60, secs)) if hours > 0 else ('%d:%02d' % (mins, secs))
                except StandardError: pass
                mem = str(int(round(task.mem(rnng) / GB)))
                print('%-60s %3sGB [%3s] %7s' % (text, real_mem, mem, timing))

        print('-' * 80)
        if task_next == 0:
            print('Queue: none')
        else:
            print('Queue:')
            for _, task in sorted(rnng.next):
                text = str(task)
                if len(text) > 60: text = text[:56] + '...' + text[-1]
                task_mem = task.mem(rnng)
                mem = str(int(round(task_mem / GB))) + 'GB' if task_mem >= 0.5*GB else ''
                if task_mem <= mem_avail: mem += '*'
                task_cpu = task.cpu(rnng)
                cpu = str(task_cpu) + 'x' if task_cpu >= 2 else ''
                if task_cpu <= (task_max - task_press): cpu += '*'
                print('%4d %-60s %5s %4s' % (len(task.all_after()), text, mem, cpu))

        print('=' * 80)
    
    def __run(self, task):
        """
        Actually runs a task. This function is called as a seperate thread. Waits for the task to
        complete and then updates the information about errors, next, last, pressure, and running.
        """
        # Run the task and wait for it to finish
        err = None
        rnng = self.__running
        try:
            task.run(rnng)
        except (StandardError, CalledProcessError) as e:
            err = e
        except:
            # Even with critical errors we still need to do some cleanup otherwise the process
            # will just hang
            rnng.error = True
            with rnng.conditional:
                rnng.running.remove(task)
                rnng.conditional.notify()
            raise

        with rnng.conditional:
            if not rnng.killing:
                if err:
                    write_error("Error in task: " + str(err))
                    rnng.error = True
                else:
                    task.done = True # done must be marked in a locked region to avoid race conditions
                    # Update subsequent tasks
                    for t in task.after:
                        if not t.done and all(b.done for b in t.before):
                            heappush(rnng.next, (len(self.all_tasks) - len(t.all_after()), t))
                    rnng.last.discard(task)
                    # Log completion
                    rnng.log.write(strftime(time_format, gmtime(time()+1))+" "+str(task)+" \n") # add one second for slightly more reliability in determing if outputs are legal
            # Remove CPU and memory pressures of this task
            rnng.cpu_pressure -= task.cpu(rnng)
            rnng.mem_pressure -= task.mem(rnng)
            # This task is no longer running
            rnng.running.remove(task)
            # Notify waiting threads
            rnng.conditional.notify()

    def _calc_next(self):
        """
        Calculate the list of tasks that have all prerequisites completed. This also verifies that
        the tasks are truly acyclic (the add() function only does a minimal check). Must be called
        when self.__running.conditional is acquired or before any tasks are started.

        The next list returned, and should be stored to self.__running.next.

        We recalculate the next list at the very beginning and periodically when going through the
        list of tasks just to make sure the list didn't get corrupted or something.
        """
        first = {t for f in self.overall_inputs() for t in self.inputs[f]}
        first |= self.generators
        if len(first) == 0: raise ValueError('Tasks are cyclic')
        for task in first: task.all_after() # precompute these (while also checking for cyclic-ness)
        changed = True
        while changed:
            changed = False
            for task in first.copy():
                if task.done:
                    first.remove(task)
                    first |= {a for a in task.after if all(b.done for b in a.before)}
                    changed = True
        num_tasks = len(self.all_tasks)
        nxt = [(num_tasks - len(t.all_after()), t) for t in first if all(b.done for b in t.before)]
        heapify(nxt)
        return nxt

    def __next_task(self):
        """
        Get the next task to be run based on priority and memory/CPU usage. Updates the CPU and
        memory pressures assuming that task will be run.

        Must be called while self.__running.conditional is acquired.
        """

        rnng = self.__running
        if len(rnng.next) == 0 and len(rnng.running) == 0:
            # Something went wrong... we have nothing running and nothing upcoming... recalulate the next list
            rnng.next = self._calc_next()
        if len(rnng.next) == 0 or rnng.cores == rnng.cpu_pressure: return None

        # Get available CPU and memory
        avail_cpu = rnng.cores - rnng.cpu_pressure
        avail_mem = virtual_memory().available - max(rnng.mem_pressure - get_mem_used_by_tree(), 0)

        # First do a fast check to see if the very next task is doable
        # This should be very fast and will commonly be where the checking ends
        _, task = rnng.next[0]
        needed_cpu, needed_mem = task.cpu(rnng), task.mem(rnng)
        if needed_cpu <= avail_cpu and needed_mem <= avail_mem:
            heappop(rnng.next)
            rnng.cpu_pressure += needed_cpu
            rnng.mem_pressure += needed_mem
            return task

        # Second do a slow check of all upcoming tasks
        # This can be quite slow if the number of upcoming processes is long
        try:
            _, task, i = min((priority, task, i) for i, (priority, task) in enumerate(rnng.next)
                             if task.cpu(rnng) <= avail_cpu and task.mem(rnng) <= avail_mem)
            if i == len(rnng.next) - 1:
                rnng.next.pop()
            else:
                rnng.next[i] = rnng.next.pop() # O(1)
                heapify(rnng.next) # O(n) [TODO: could be made O(log(N)) with undocumented _siftup/_siftdown]
            rnng.cpu_pressure += task.cpu(rnng)
            rnng.mem_pressure += task.mem(rnng)
            return task
        except (ValueError, LookupError): pass

        return None

    def run(self, log, rusage_log=None, verbose=False, settings=None, workingdir=None,
            cores=cpu_count(), cluster=None):
        """
        Runs all the tasks in a smart order with many at once. Will not return until all tasks are
        done.

          log
            the filepath to a file where to read/write the log of completed tasks to
          rusage_log
            if provided the memory and time usage of every task will be logged to the given file
            (only provide if on *nix or Windows)
          verbose
            if set to True will cause the time and the command to print to stdout whenever a new
            command is about to start
          settings
            the setting names and their current values for this run as a dictionary
          workingdir
            the default working directory for all of the tasks, defaults to the current working
            directory
          cores
            the number of local cores to use, defaults to the number of processors available
          cluster
            a Cluster object that is used for any tasks that are added with run_on_cluster=True
        """

        # Checks
        if self.__running is not None: raise ValueError('Tasks already running')
        if  len(self.inputs) == 0 and len(self.generators) == 0  and len(self.outputs) == 0 and len(self.cleanups) == 0 : return
        if (len(self.inputs) == 0 and len(self.generators) == 0) or (len(self.outputs) == 0 and len(self.cleanups) == 0): raise ValueError('Invalid set of tasks (likely cyclic)')
        prev_signal = None

        try:
            self.__running = rnng = RunningTasks(self, log, rusage_log, verbose, settings, workingdir,
                                                 cores, cluster)

            # Set a signal handler
            try:
                from signal import signal, SIGUSR1 #pylint: disable=no-name-in-module
                prev_signal = signal(SIGUSR1, self.display_stats)
            except (ImportError, ValueError): pass

            # Keep running tasks in the tree until we have completed the root (which is self)
            with rnng.conditional:
                while len(rnng.last) != 0:
                    # Get next task (or wait until all tasks or finished or an error is generated)
                    while len(rnng.last) > 0 and not rnng.error:
                        task = self.__next_task()
                        if task is not None: break
                        # Wait until we have some available [without the timeout CTRL+C does not work and we cannot see if memory is freed up on the system]
                        rnng.conditional.wait(30)
                    if len(rnng.last) == 0 or rnng.error: break

                    # Run it
                    rnng.running.add(task)
                    t = Thread(target=self.__run, args=(task,))
                    t.daemon = True
                    if verbose: print(strftime(time_format) + " Running " + str(task))
                    t.start()
                    sleep(0) # make sure it starts

                # There was an error, let running tasks finish
                if rnng.error and len(rnng.running) > 0:
                    write_error("Waiting for other tasks to finish running.\nYou can terminate them by doing a CTRL+C.")
                    while len(rnng.running) > 0:
                        # Wait until a task stops [without the timeout CTRL+C does not work]
                        rnng.conditional.wait(60)

        except KeyboardInterrupt:

            # Terminate and kill tasks
            write_error("Terminating running tasks")
            with self.__running.conditional:
                self.__running.killing = True
                for t in self.__running.running:
                    try: t.terminate()
                    except StandardError: pass
                secs = 0
                while len(self.__running.running) > 0 and secs < 10:
                    self.__running.conditional.wait(1)
                    secs += 1
                for t in self.__running.running:
                    try: t.kill()
                    except StandardError: pass

        finally:
            # Cleanup
            if prev_signal: signal(SIGUSR1, prev_signal)
            if self.__running is not None: self.__running.close()

class RunningTasks(object):
    def __init__(self, tasks, log, rusage_log, verbose, settings, wd, cores, cluster):
        # Create basic variables and lock
        if wd:
            self._original_wd = getcwd()
            wd = normpath(wd)
            chdir(wd)
        else:
            wd = getcwd()
        self.workingdir = wd
        self.cores = max(int(cores), 1)
        self.cluster = cluster
        self.error = False
        self.killing = False
        self.conditional = Condition() # for locking access to Task.done, cpu_pressure, mem_pressure, next, last, log, and error
        self.running = set()

        # Setup log
        self.settings = settings
        if len(tasks.settings - settings.viewkeys()) > 0: raise ValueError('Not all settings given values')
        log = join_norm(wd, log)
        done_tasks = self.__process_log(tasks.all_tasks, log) if exists(log) else ()
        self.log = open(log, 'w', 0)
        for k,v in settings.iteritems(): self.log.write("*"+k+"="+str(v)+"\n")
        # TODO: log overall inputs and outputs
        for dc in done_tasks:
            if verbose: print("Skipping " + dc[20:].strip())
            self.log.write(dc+"\n")
        if verbose and len(done_tasks) > 0: print('-' * 80)
        self.rusagelog = open(join_norm(wd, rusage_log), 'a', 1) if rusage_log else None

        # Calcualte the set of first and last tasks
        self.next = tasks._calc_next() # These are the first tasks #pylint: disable=protected-access
        last = {tasks.outputs[f] for f in tasks.overall_outputs()}
        last |= tasks.cleanups
        if len(last) == 0: raise ValueError('Tasks are cyclic')
        self.last = {t for t in last if not t.done}

        # Get the initial pressures
        self.cpu_pressure = 0
        self.mem_pressure = get_mem_used_by_tree() + 1*MB # assume that the creation of threads and everything will add some extra pressure

    def __process_log(self, all_tasks, log):
        """
        This looks at the previous log file and determines which commands do not need to be run this
        time through. This checks for changes in the commands themselves, when the commands were run
        relative to their output files, and more.
        """
        with open(log, 'r+') as log: lines = [line.strip() for line in log] #pylint: disable=redefined-argument-from-local
        lines = [line for line in lines if len(line) != 0]
        #comments = [line for line in lines if line[0] == '#']
        # Note: this will take the last found setting/command with a given and silently drop the others
        settings = {s[0].strip():s[1].strip() for s in (line[1:].split('=',1) for line in lines if line[0] == '*')} # setting => value
        tasks = {line[20:].strip():line[:19] for line in lines if re.match(r'\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\s', line)} # task string => date/time string
        #if len(lines) != len(comments) + len(settings) + len(commands): raise ValueError('Invalid file format for tasks log')

        # Check Settings
        changed_settings = self.settings.viewkeys() - settings.viewkeys() # new settings
        changed_settings.update(k for k in (self.settings.viewkeys() & settings.viewkeys()) if str(self.settings[k]).strip() != settings[k]) # all previous settings that changed value

        # Check Tasks / Files
        changed = all_tasks.viewkeys() - tasks.viewkeys() # new tasks are not done
        for n,dt in tasks.items(): # not iteritems() since we may remove elements
            t = all_tasks.get(n)
            if not t: del tasks[n] # task no longer exists
            elif not t.settings.isdisjoint(changed_settings): changed.add(n) # settings changed
            else:
                date_time = timegm(strptime(dt, time_format))
                if any((exists(f) and getmtime(f) >= date_time for f in t.inputs)) or any(not exists(f) for f in t.outputs):
                    changed.add(n)
        for n in changed.copy(): changed.update(str(t) for t in all_tasks.get(n).all_after()) # add every task that comes after a changed task

        # Mark as Done
        done_tasks = tasks.viewkeys() - changed
        for n in done_tasks: all_tasks.get(n).done = True
        return sorted((tasks[n] + " " + n) for n in done_tasks)

    def close(self):
        if hasattr(self, '_original_wd'): chdir(self._original_wd)
        if hasattr(self, 'log'): self.log.close()
        if hasattr(self, 'rusagelog') and self.rusagelog: self.rusagelog.close()
