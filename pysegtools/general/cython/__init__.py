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

import sys, imp, types, shutil, os.path, warnings
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

default_compiler = get_default_compiler()  # TODO: this isn't the compiler that will necessarily be used, but is a good guess...
includes = (os.path.dirname(__file__), numpy.get_include()) if _have_numpy else (os.path.dirname(__file__),)
compiler_opts = {
        'msvc'    : ['/D_SCL_SECURE_NO_WARNINGS','/EHsc','/O2','/DNPY_NO_DEPRECATED_API=7','/bigobj','/openmp'],
        'unix'    : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'], # gcc/clang (whatever is system default)
        'mingw32' : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
        'cygwin'  : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
    }.get(default_compiler, [])
linker_opts = {
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
    
    if os.path.isfile(so_file):
        try: newer = os.path.getmtime(so_file) >= os.path.getmtime(cy_file)
        except OSError: newer = False
        if newer:
            try: return imp.load_dynamic(fullname, so_file)
            except ImportError: pass
    
    if os.path.isfile(cy_file) and _have_pyximport:
        try:
            # Wrap the 'get_distutils_extension' to apply our defaults
            #pylint: disable=no-member
            pyximport.build_module.__globals__['get_distutils_extension'] = __get_distutils_extension_wrap
            new_so_file = pyximport.build_module(fullname, cy_file, build_dir)
            pyximport.build_module.__globals__['get_distutils_extension'] = pyximport.get_distutils_extension
            try:
                shutil.copy2(new_so_file, so_file)
                new_so_file = so_file
            except IOError: pass
            return imp.load_dynamic(fullname, new_so_file)
        except Exception as ex: #pylint: disable=broad-except
            print(ex)
    
    warnings.warn("Cannot load Cython-optimized functions for module '%s', using fallback functions which might be slow." % fullname, RuntimeWarning)

    if os.path.isfile(fb_file):
        try: return imp.load_source(fullname, fb_file)
        except ImportError: pass
    
    raise ImportError("Unable to load module %s from %s, %s, or %s" % (fullname, so_file, cy_file, fb_file))

def __get_distutils_extension_wrap(modname, pyxfilename, *args):
    extension_mod,setup_args = pyximport.get_distutils_extension(modname, pyxfilename, *args)
    extension_mod.include_dirs.extend(includes)
    extension_mod.include_dirs.append(os.path.dirname(pyxfilename))
    if extension_mod.language is None:
        extension_mod.language = "c++"
    extension_mod.extra_compile_args = compiler_opts + extension_mod.extra_compile_args
    extension_mod.extra_linker_args = linker_opts + extension_mod.extra_linker_args
    return extension_mod,setup_args

def _load_mod(mod, path, build_dir):
    fullname = mod.__name__
    del mod.__class__.__repr__
    del mod.__class__.__dir__
    del mod.__class__.__getattr__
    #del mod.__class__.__setattr__
    del mod.__class__.__delattr__
    del sys.modules[fullname]
    new = load_module(fullname, path, build_dir)
    mod.__dict__.update(new.__dict__)
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
