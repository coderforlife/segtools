#!/usr/bin/env python3
# pylint: disable=invalid-name

import os
import sys
from os.path import join
from distutils.ccompiler import get_default_compiler
from setuptools import setup

import numpy

# We require Python v3.6 or newer
if sys.version_info[:2] < (3, 6): raise RuntimeError("This requires Python v3.6 or newer")

# Prepare for compiling the source code
compiler_name = get_default_compiler() # TODO: this isn't the compiler that will necessarily be used, but is a good guess...
compiler_opt = {
    'msvc'    : ['/D_SCL_SECURE_NO_WARNINGS', '/EHsc', '/O2', '/DNPY_NO_DEPRECATED_API=7', '/bigobj', '/openmp'],
    'unix'    : ['-std=c++11', '-O3', '-DNPY_NO_DEPRECATED_API=7', '-fopenmp'], # gcc/clang (whatever is system default)
    'mingw32' : ['-std=c++11', '-O3', '-DNPY_NO_DEPRECATED_API=7', '-fopenmp'],
    'cygwin'  : ['-std=c++11', '-O3', '-DNPY_NO_DEPRECATED_API=7', '-fopenmp'],
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
def create_ext(name, dep=None):
    """Creates a Python extension dependent on numpy."""
    from distutils.extension import Extension
    if dep is None: dep = []
    return Extension(
        name=name,
        depends=dep,
        sources=[join(*name.split('.'))+src_ext],
        define_macros=[('NPY_NO_DEPRECATED_API', '7'),],
        include_dirs=[np_inc, cy_inc],
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
        def cythonize(*args, **kwargs): # pylint: disable=missing-docstring
            kwargs.setdefault('include_path', []).append(cy_inc)
            return Cython.Build.cythonize(*args, **kwargs)
except ImportError:
    def cythonize(exts, *args, **kwargs): # pylint: disable=unused-argument, missing-docstring
        return exts

# Finally we get to run setup
setup(name='pysegtools',
      version='0.1',
      description='Python Segmentation Tools',
      long_description=open('README.md').read(),
      author='Jeffrey Bush',
      author_email='jeff@coderforlife.com',
      url='https://cellsegmentation.org/',
      entry_points={
          'console_scripts': [
              'imstack = pysegtools.imstack_main:main_func',
          ],
      },
      packages=['pysegtools', 'pysegtools.general',
                'pysegtools.general.cython',
                'pysegtools.images',
                'pysegtools.images.filters',
                'pysegtools.images.io',
                'pysegtools.images.io.handlers'],
      zip_safe=False, # dynamic-load system prevents this
      package_data={'': ['*.pyx', '*.pyxdep', '*.pxi', '*.pxd', '*.h', '*.txt']}, # Cython files
      setup_requires=['numpy>=1.7'],
      install_requires=['numpy>=1.7', 'scipy>=0.16', 'cython>=0.22',
                        'psutil>=2.0', 'python-intervals>=1.5.1'],
      extras_require={
          'OPT': ['pyfftw>=0.9.2'],
          'PIL': ['pillow>=2.0'],
          'MATLAB': ['h5py>=2.0']
      },
      ext_modules=cythonize([
          create_ext('pysegtools.images.filters._frangi',
                     dep=[join(cy_inc, 'npy_helper.pxd'), join(cy_inc, 'npy_helper.h')]),
          create_ext('pysegtools.images.filters._label',
                     dep=[join(cy_inc, 'fused.pxi'), join(cy_inc, 'npy_helper.pxi'),
                          join(cy_inc, 'npy_helper.pxd'), join(cy_inc, 'npy_helper.h')]),
      ])
     )
