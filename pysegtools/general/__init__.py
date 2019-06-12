# Load the most common classes and methods directly
from .datawrapper import * #pylint: disable=wildcard-import
from .delayed import delayed
from .gzip import GzipFile, compress, decompress
from .utils import sys_endian, sys_64bit, pairwise, prod, ravel, re_search, itr2str, splitstr, _bool

# Also alias many of the module names
from . import json
from . import io
from . import os_ext
from . import utils
