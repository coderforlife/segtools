"""
This is used to wrap Python function tasks so that their process has the proper working directory
and standard streams. Making it a seperate module helps the multiprocessing module on Windows not
have to import the entire actual module.
"""

import sys, os

__modes = [os.O_RDONLY, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, os.O_WRONLY|os.O_APPEND]
def __reopen(std, path, mode):
    fd = os.open(path, __modes[mode])
    os.dup2(fd, std.fileno())
    os.close(fd)

#pylint: disable=too-many-arguments
def _pyproc(func, args, kwargs, wd, stdin, stdout, stdout_append, stderr, stderr_append):
    os.chdir(wd)
    if stdin:  __reopen(sys.stdin, stdin, 0)
    if stdout: __reopen(sys.stdout, stdout, 1 + stdout_append)
    if stderr == -2: os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
    elif stderr: __reopen(sys.stderr, stderr, 1 + stderr_append)
    return func(*args, **kwargs)
