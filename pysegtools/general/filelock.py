"""
This implements a simple file locking mechanism that uses a context manager and is cross-platform.
Probably not completely robust, but good enough for most situations. It includes a basic facility
for cleaning up stale locks as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = ['FileLock']

class FileLock(object):
    """
    A file locking mechanism that uses a context-manager support so it can be used in a with
    statement which increases robustness.
    """
    
    def __init__(self, filename):
        import os.path
        self.__file = os.path.abspath(filename)
        self.__lockfile = None
        # TODO: should the acquire and release method use a standard lock?
        #from threading import Lock
        #self.__lock = Lock()

    @property
    def acquired(self): return self.__lockfile is not None
    @property
    def file(self): return self.__file
        
    def __check_stale(self):
        """
        Checks for a possible stale lock file. The lock file contains a pid and it is checked if
        a process with that PID is still running. If not then the file is deleted. This only works
        if psutil module is available.
        """
        try:
            import psutil, os
            with open(self.__file, 'rb') as f: pid = int(f.read())
            try: psutil.Process(pid)
            except psutil.NoSuchProcess: os.unlink(self.__file)
        except (ImportError, IOError, OSError, ValueError, TypeError): pass
        
    def acquire(self, timeout=None, delay=0.01):
        """
        Acquire the lock, creating the lockfile. If the lockfile already exists this tries to
        acquire the lock every `delay` seconds (default 0.01) until `timeout` has been reached
        (default is to never timeout). When the timeout does occurs an OSError is raised.
        Additionally after every 5 attempts to acquire the lock this will attempt to detect a
        stale lock file and delete it. This feature is only supported if psutil is installed.
        Returns `self` so that this can be used as a context manager while also setting the
        timeout and delay.
        """
        import os
        from errno import EEXIST
        from time import time, sleep
        delay, attempts = float(delay), 0
        if timeout is not None: timeout, start = float(timeout), time()
        while self.__lockfile is None:
            try:
                self.__lockfile = os.open(self.__file, os.O_CREAT|os.O_EXCL|os.O_RDWR, 0o666)
                pid = str(os.getpid())
                n = os.write(self.__lockfile, pid)
                if n != len(pid):
                    from warnings import warn
                    warn("lockfile not written properly, wrote %d bytes instead of %d" % (n,len(pid)), category)
                os.fsync(self.__lockfile)
            except OSError as e:
                if e.errno != EEXIST: self.release(); raise
                if timeout is not None and time() - start >= timeout: raise OSError("timeout occured")
                attempts += 1
                if attempts == 5:
                    self.__check_stale()
                    attempts = 0
                sleep(delay)
        return self
    
    def release(self):
        """Release the lock, deleting the lockfile."""
        if self.__lockfile is not None:
            import os
            os.close(self.__lockfile)
            os.unlink(self.__file)
        self.__lockfile = None

    def __enter__(self): return self.acquire()
    def __exit__(self, typ, value, traceback): self.release()
    def __del__(self): self.release() # make sure the lock is not kept around
