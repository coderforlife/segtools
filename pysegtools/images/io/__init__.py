"""Image IO functions. All images are numpy arrays in memory."""

__all__ = ['iminfo', 'imread', 'imsave', 'ImageStack', 'FileStack']

from _single import iminfo, imread, imsave
from _stack import ImageStack
from _files import FileStack

import formats
