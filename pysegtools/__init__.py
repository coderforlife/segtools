from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try: # this try-except is to allow --check to work without requiring imports to work
    # The following don't work, but after we have the rough equivalent
    #from . import general
    #from . import images
    #from . import tasks
    __import__(__name__+'.general', globals(), locals())
    __import__(__name__+'.images', globals(), locals())
    #__import__(__name__+'.tasks', globals(), locals())
except ImportError as ex:
    # We will still warn though
    from warnings import warn
    warn(ex.message, RuntimeWarning, 2)
    del warn
