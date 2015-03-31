from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# The following don't work, but after we have the rough equivalent
#from . import general
#from . import images
#from . import tasks
__import__(__name__+'.general', globals(), locals())
__import__(__name__+'.images', globals(), locals())
#__import__(__name__+'.tasks', globals(), locals())
