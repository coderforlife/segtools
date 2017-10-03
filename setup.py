#!/usr/bin/env python

import os, sys
from os.path import join

# We require Python v2.7 or newer
if sys.version_info[:2] < (2,7): raise RuntimeError("This requires Python v2.7 or newer")
    
# If this is True we need to have subprocess32 installed
# Only needed on POSIX systems using Python < 3.2 due to a bug in the built-in subprocess module
need_sp32 = (os.name == 'posix') and (sys.version_info[:2] < (3,2))

# Prepare for compiling the source code
# NOTE: this no longer auto-adds the extra dependencies as follows: (the old cython module did)
#npy_helper_pxd_dep = [os.path.join(thisdir, __fn) for __fn in ('npy_helper.h', 'npy_helper.pxd')]
#npy_helper_pxi_dep = npy_helper_pxd_dep + [os.path.join(thisdir, 'npy_helper.pxi')]
#fused_pxi_dep      = npy_helper_pxd_dep + [os.path.join(thisdir, 'fused.pxi')]
# There were other nice sophisticated systems in the old general/cython module as well that could be re-implemented
from distutils.ccompiler import get_default_compiler
import numpy
compiler_name = get_default_compiler() # TODO: this isn't the compiler that will necessarily be used, but is a good guess...
compiler_opt = {
    'msvc'    : ['/D_SCL_SECURE_NO_WARNINGS','/EHsc','/O2','/DNPY_NO_DEPRECATED_API=7','/bigobj','/openmp'],
    # TODO: older versions of gcc need -std=c++0x instead of -std=c++11
    'unix'    : ['-std=c++11','-O3','-DNPY_NO_DEPRECATED_API=7','-fopenmp'], # gcc/clang (whatever is system default)
    'mingw32' : ['-std=c++11','-O3','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
    'cygwin'  : ['-std=c++11','-O3','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
}
linker_opt = {
    'msvc'    : [],
    'unix'    : ['-fopenmp'], # gcc/clang (whatever is system default)
    'mingw32' : ['-fopenmp'],
    'cygwin'  : ['-fopenmp'],
}
np_inc = numpy.get_include()
cy_inc = join(os.path.dirname(__file__), 'pysegtools', 'general', 'cython') # TODO
src_ext = '.cpp'
def create_ext(name, dep=[], src=[], inc=[], lib=[], objs=[]):
    from distutils.extension import Extension
    return Extension(
        name=name,
        depends=dep,
        sources=[join(*name.split('.'))+src_ext]+src,
        define_macros=[('NPY_NO_DEPRECATED_API','7'),],
        include_dirs=[np_inc,cy_inc]+inc,
        library_dirs=lib,
        extra_objects=objs,
        extra_compile_args=compiler_opt.get(compiler_name, []),
        extra_link_args=linker_opt.get(compiler_name, []),
        language='c++',
    )

# Find and use Cython if available
try:
    from distutils.version import StrictVersion
    import Cython.Build
    if StrictVersion(Cython.__version__) >= StrictVersion('0.22'):
        src_ext = '.pyx'
        def cythonize(*args, **kwargs):
            kwargs.setdefault('include_path', []).append(cy_inc)
            return Cython.Build.cythonize(*args, **kwargs)
except ImportError:
    def cythonize(exts, *args, **kwargs): return exts

# Finally we get to run setup
try: from setuptools import setup
except ImportError: from distutils.core import setup
setup(name='pysegtools',
      version='0.1',
      description='Python Segmentation Tools',
      long_description=open('README.md').read(),
      author='Jeffrey Bush',
      author_email='jeff@coderforlife.com',
      url='https://cellsegmentation.org/',
      scripts=['imstack.bat' if os.name == 'nt' else 'imstack'],
      packages=['pysegtools', 'pysegtools.general',
                'pysegtools.general.cython',
                'pysegtools.images',
                'pysegtools.images.filters',
                'pysegtools.images.io',
                'pysegtools.images.io.handlers'],
      use_2to3=True, # the code *should* support Python 3 once run through 2to3 but this isn't tested
      zip_safe=False, # I don't think this code would work when running from inside a zip file due to the dynamic-load system
      package_data = { '': ['*.pyx', '*.pyxdep', '*.pxi', '*.pxd', '*.h', '*.txt'], }, # Make sure all Cython files are wrapped up with the code
      setup_requires=['numpy>=1.7'],
      install_requires=['numpy>=1.7','scipy>=0.16','psutil>=2.0','cython>=0.22'] + (['subprocess32>=3.2.6'] if need_sp32 else []),
      extras_require={
            'OPT': ['pyfftw>=0.9.2'],
            'PIL': ['pillow>=2.0'],
            'MATLAB': ['h5py>=2.0']
      },
      ext_modules = cythonize([
          create_ext('pysegtools.images.filters._frangi', dep=[join(cy_inc, 'npy_helper.pxd'), join(cy_inc, 'npy_helper.h')]),
          create_ext('pysegtools.images.filters._label', dep=[join(cy_inc, 'fused.pxi'), join(cy_inc, 'npy_helper.pxi'), join(cy_inc, 'npy_helper.pxd'), join(cy_inc, 'npy_helper.h')]),
      ])
)
