#!/usr/bin/env python

"""
Command line program to convert an image stack to another stack by processing
each slice. This module simply calls the main function from imstack, which is
necessary to prevent double-import. It also does checks on the runtime
environment to make sure things will work (before trying to import stuff).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

def __main():
    from .imstack import main
    main()

def __version_getter(name):
    # Gets a function that gets the version of a module (assuming the version is in __version__) or
    # False if the module cannot be loaded.
    def __get_version():
        import importlib
        try: return importlib.import_module(name).__version__
        except ImportError: return False
    return __get_version
def __get_python_version():
    return '.'.join(str(x) for x in sys.version_info[0:3])
def __get_pillow_version():
    try: import PIL; return PIL.PILLOW_VERSION if hasattr(PIL, 'PILLOW_VERSION') else False
    except ImportError: return False

__req_modules = ( # module, target version, actual version (or False)
    ('numpy', '1.7', __version_getter('numpy')()),
    ('scipy', '0.12', __version_getter('scipy')()),
    )
__opt_modules = ( # module, target version, version getter, description (or None)
    ('cython', '0.19', __version_getter('cython'), 'for some optimized libraries'),
    ('pillow', '2.0', __get_pillow_version, 'for loading common image formats'),
    # TODO: bioformats
    ('h5py',   '2.0', __version_getter('h5py'), 'for loading MATLAB files'),
    ('psutil', '2.0', __version_getter('psutil'), None),
    )

def __basic_check():
    # This checks only the required modules. The optional modules are checked by individual
    # 'plugins' or if --check is given.
    if sys.version_info[0:2] != (2,7):
        print("Python v2.7 is required and you are currently running v%s" % (__get_python_version()), file=sys.stderr)
        print("Currently no other versions (including v3+) are supported", file=sys.stderr)
        sys.exit(1)
    from distutils.version import StrictVersion as Vers
    for name, target, vers in __req_modules:
        if vers is False:
            print("%s v%s or higher is required and could not be imported" % (name, target), file=sys.stderr)
            print("Try running `pip install %s` or `easy_install %s` to install it" % (name, name), file=sys.stderr)
            sys.exit(1)
        elif Vers(target) > Vers(vers):
            print("%s v%s or higher is required and currently v%s is installed" % (name, target, vers), file=sys.stderr)
            print("Try running `pip install -U %s` or `easy_install -U %s` to upgrade it" % (name, name), file=sys.stderr)
            sys.exit(1)

def __check():
    from distutils.version import StrictVersion as Vers
    checkbox = '\u2713' if sys.stdout.encoding in ('UTF-8','UTF-16') else '+'
    print("Module    Required  Installed")
    print("Python    v2.7      {0:<9}  {1}".format(__get_python_version(), 'x' if sys.version_info[0:2] != (2,7) else checkbox))
    for name, target, vers in __req_modules:
        if vers is False: vers, mark = '-'*8, 'x'
        else: vers, mark = 'v'+vers, ('x' if Vers(target) > Vers(vers) else checkbox)
        print("{0:<9} v{1:<8} {2:<9}  {3}".format(name, target, vers, mark))
    print("")
    print("Optional:")
    for name, target, vers, desc in __opt_modules:
        vers = vers()
        if vers is False: vers, mark = '-'*8, 'x'
        else: vers, mark = 'v'+vers, ('x' if Vers(target) > Vers(vers) else checkbox)
        desc = ('('+desc+')') if desc else ''
        print("{0:<9} v{1:<8} {2:<9}  {3} {4}".format(name, target, vers, mark, desc))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        __check()
    else:
        __basic_check()
        __main()
