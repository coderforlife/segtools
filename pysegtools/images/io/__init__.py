"""Image IO functions. All images are numpy arrays in memory."""

__all__ = ['iminfo', 'imread', 'imsave', 'ImageStack', 'MemoryImageStack', 'ImageStackCollection']

from _single import iminfo, imread, imsave

from _stack import ImageStack
from _memory import MemoryImageStack
from _collection import ImageCollectionStack

import formats
