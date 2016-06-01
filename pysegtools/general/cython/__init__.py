"""
This implements a custom module importer that finds PYX modules and loads them. There are
differences from pyximport though, including that PYD/SO files are always used if next to the PYX
file (unless they are older) even if not running 'inplace'. This allows one to distribute the
compiled code in some situations where inplace cannot be used. Also, it allows falling-back to pure
Python code if loading the PYX fails (e.g. if Cython is not installed). The fallback document has
the same name with the extension .py. The final new feature is that the module is actually
delay-loaded. This means that until an attribute is requested, the module is not actually loaded.

Besides adding the above features, it lacks a few features the pyximport has. These include support
for reloading modules and loading packages. These were removed for simplicity.

This sets a few different defaults than the pyximport module. It sets a few compiler flags and
options (language is C++, compile args that optimize code and disable NumPy deprecated API, and some
other tweaks), and adds the many include directories (NumPy include directory, this module's
directory, and the PYX file's directory).

Note: internally this uses pyximport to build the Cython modules. If it is missing, the fallback
code will always be used.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = ['install', 'get_include']

import sys, imp, types, shutil, re, os, warnings
from distutils.ccompiler import get_default_compiler

try:
    import pyximport
    _have_pyximport = True
except ImportError:
    _have_pyximport = False

try:
    import numpy
    _have_numpy = True
except ImportError:
    _have_numpy = False

so_suffix = next((s for s,_,t in imp.get_suffixes() if t == imp.C_EXTENSION), '.so')
cy_suffix = '.pyx'
fb_suffix = '.py'

re_sources = re.compile(r'\s*#\s*distutils:\s+sources=([^#]+)')
re_include = re.compile(r'\s*include\s+([\'"])([^\'"]+)(?:\1)')
re_extern  = re.compile(r'cdef\s+extern\s+from\s+([\'"])([^*\'"]+)(?:\1)(?:|\s+nogil)\s*:')
re_cimport = re.compile(r'(from\s+npy_helper\s+cimport\s+|cimport\s+npy_helper($|\s|#))')

thisdir = os.path.dirname(__file__)
npy_helper_pxd_dep = [os.path.join(thisdir, __fn) for __fn in ('npy_helper.h', 'npy_helper.pxd')]
npy_helper_pxi_dep = npy_helper_pxd_dep + [os.path.join(thisdir, 'npy_helper.pxi')]
fused_pxi_dep      = npy_helper_pxd_dep + [os.path.join(thisdir, 'fused.pxi')]

default_compiler = get_default_compiler()  # TODO: this isn't the compiler that will necessarily be used, but is a good guess...
includes = ('.', thisdir, numpy.get_include()) if _have_numpy else ('.', thisdir,)
compiler_opts = {
        'msvc'    : ['/D_SCL_SECURE_NO_WARNINGS','/EHsc','/O2','/DNPY_NO_DEPRECATED_API=7','/bigobj','/openmp'],
        'unix'    : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'], # gcc/clang (whatever is system default)
        'mingw32' : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
        'cygwin'  : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
    }.get(default_compiler, [])
link_opts = {
        'msvc'    : [],
        'unix'    : ['-fopenmp'], # gcc/clang (whatever is system default)
        'mingw32' : ['-fopenmp'],
        'cygwin'  : ['-fopenmp'],
    }.get(default_compiler, [])

warnings.filterwarnings('ignore', # stupid warning because Cython is confused...
                        'numpy[.]ndarray size changed, may indicate binary incompatibility',
                        RuntimeWarning)

class CythonFallbackImporter(object):
    def __init__(self, build_dir=None):
        self.build_dir = build_dir
    def find_module(self, fullname, package_path=None):
        if fullname in sys.modules: return None  # only here when reload()
        module_name_pyx = fullname[fullname.rfind('.')+1:] + cy_suffix
        for path in (package_path or sys.path):
            if not path: path = os.getcwd()
            elif not os.path.isabs(path): path = os.path.abspath(path)
            pyx = os.path.join(path, module_name_pyx)
            if os.path.isfile(pyx): return CythonFallbackLoader(fullname, pyx, self.build_dir)
        return None

def load_module(fullname, path, build_dir):
    #pylint: disable=protected-access
    so_file = path + so_suffix
    cy_file = path + cy_suffix
    fb_file = path + fb_suffix

    deps = None
    if os.path.isfile(so_file):
        try:
            deps = __get_dependencies(cy_file)
            if all(os.path.getmtime(so_file) >= os.path.getmtime(f) for f in deps):
                try: return imp.load_dynamic(fullname, so_file)
                except ImportError: pass
            else:
                os.remove(so_file)
        except OSError: pass
    
    if os.path.isfile(cy_file) and _have_pyximport:
        try:
            # Wrap the 'get_distutils_extension' to apply our defaults
            #pylint: disable=no-member
            if deps is None: deps = __get_dependencies(cy_file)
            pyximport.build_module.__globals__['get_distutils_extension'] = __get_distutils_extension_wrap(deps)
            new_so_file = pyximport.build_module(fullname, cy_file, build_dir)
            pyximport.build_module.__globals__['get_distutils_extension'] = pyximport.get_distutils_extension
            try:
                shutil.copy2(new_so_file, so_file)
                new_so_file = so_file
            except IOError: pass
            return imp.load_dynamic(fullname, new_so_file)
        except Exception as ex: #pylint: disable=broad-except
            print(ex)
    
    if os.path.isfile(fb_file):
        warnings.warn("Cannot load Cython-optimized functions for module '%s', using fallback functions which might be slow." % fullname, RuntimeWarning)
        try: return imp.load_source(fullname, fb_file)
        except ImportError: pass
    
    raise ImportError("Unable to load module %s from %s, %s, or %s" % (fullname, so_file, cy_file, fb_file))

def __get_dependencies(filename):
    """
    Finds the dependencies of a pyx file by reading the file and looking for '#distutil: sources=',
    'include "*.pxi"', 'cdef extern from "*.h"', and cimport npy_helper lines. The PXI files are
    recursively searched. Also the contents of the pyxdep file are added. Does not recursively go
    through .h, .c, .cpp or other files.
    """
    dirname = os.path.dirname(filename)
    def fullpath(filename): return os.path.normpath(os.path.join(dirname, filename))
    files = [fullpath(filename)]
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f:
                m = re_sources.match(line)
                if m is not None: files += [fullpath(fn) for fn in m.group(1).split()]; continue

                m = re_cimport.match(line)
                if m is not None: files += npy_helper_pxd_dep; continue

                m = re_include.match(line)
                if m is not None:
                    fn = m.group(2)
                    if fn == 'npy_helper.pxi': files += npy_helper_pxi_dep
                    elif fn == 'fused.pxi':    files += fused_pxi_dep
                    else:                      files += __get_dependencies(fullpath(fn))
                    continue

                m = re_extern.match(line)
                if m is not None: files.append(fullpath(m.group(2))); continue

    if os.path.isfile(filename+'dep'):
        with open(filename+'dep', 'r') as f:
            for line in f:
                if   line == 'npy_helper.pxd': files += npy_helper_pxd_dep
                elif line == 'npy_helper.pxi': files += npy_helper_pxi_dep
                elif line == 'fused.pxi':      files += fused_pxi_dep
                elif line.endswith('.pxi') or line.endswith('.pxd'): files += __get_dependencies(fullpath(line))
                else: files.append(fullpath(line))

    return [fn for fn in set(files) if os.path.isfile(fn)] # make the files unique and makes sure they exist

def __get_distutils_extension_wrap(depends):
    def get_distutils_extension(modname, pyxfilename, *args):
        extension_mod,setup_args = pyximport.get_distutils_extension(modname, pyxfilename, *args)
        extension_mod.include_dirs.extend(includes)
        extension_mod.include_dirs.append(os.path.dirname(pyxfilename))
        if extension_mod.language is None:
            extension_mod.language = "c++"
        extension_mod.extra_compile_args = compiler_opts + extension_mod.extra_compile_args
        extension_mod.extra_link_args = link_opts + extension_mod.extra_link_args
        extension_mod.depends = extension_mod.depends + depends
        return extension_mod,setup_args
    return get_distutils_extension

def _load_mod(mod, path, build_dir):
    import time, errno
    fullname = mod.__name__

    # Obtain a lock for the path
    lockpath = path + '.lock'
    while True:
        try:
            lock = os.open(lockpath, os.O_CREAT|os.O_EXCL|os.O_RDWR)
            break
        except OSError as e:
            if e.errno != errno.EEXIST: raise
        time.sleep(0.01) # re-check every 10 ms

    # Check that the module wasn't already loaded while waiting to obtain the lock
    if sys.modules[fullname] is not mod:
        return sys.modules[fullname]

    # Perform the actual load
    try:
        del mod.__class__.__repr__
        del mod.__class__.__dir__
        del mod.__class__.__getattr__
        #del mod.__class__.__setattr__
        del mod.__class__.__delattr__
        del sys.modules[fullname]
        new = load_module(fullname, path, build_dir)
        mod.__dict__.update(new.__dict__)
    finally:
        # Unlock the path
        os.close(lock)
        os.unlink(lockpath)
    
    return new

class CythonFallbackLoader(object):
    def __init__(self, fullname, path, build_dir):
        self.fullname = str(fullname)
        self.path = str(path[:path.rfind('.')])
        self.build_dir = build_dir
    def load_module(self, fullname):
        # This returns a dummy module that upon being accessed loads the real module
        assert self.fullname == fullname, ("invalid module, expected %s, got %s" % (self.fullname, fullname))
        load = lambda mod:_load_mod(mod, self.path, self.build_dir)
        mod_type = type(str("module."+fullname), (types.ModuleType,),
                     {
                         '__repr__':   lambda self:'<module '+self.__name__+' (delay loaded)>',
                         '__dir__':    lambda self:         dir(    load(self)),
                         '__getattr__':lambda self,name:    getattr(load(self), name),
                         #'__setattr__':lambda self,name,val:setattr(load(self), name, val),
                         '__delattr__':lambda self,name:    delattr(load(self), name),
                     })
        mod = mod_type(fullname)
        mod.__builtins__ = __builtins__
        mod.__file__ = self.path + cy_suffix
        mod.__package__ = None
        sys.modules[fullname] = mod
        return mod

def install(build_dir=None, build_in_temp=True, setup_args=None):
    #pylint: disable=protected-access
    if not build_dir:
        build_dir = os.path.join(os.path.expanduser('~'), '.pyxbld')
    build_dir = str(build_dir)
    if _have_pyximport:
        # We have to install and uninstall the pyximporter so we can set the given options
        pyximport.uninstall(*pyximport.install(build_dir=build_dir, build_in_temp=build_in_temp, setup_args=setup_args))
    if not any(isinstance(i, CythonFallbackImporter) for i in sys.meta_path):
        # Only install if we aren't already installed
        sys.meta_path.append(CythonFallbackImporter(build_dir))

def get_include(): return thisdir
