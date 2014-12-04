"""Image IO functions. All images are numpy arrays in memory."""

__all__ = ['iminfo', 'imread', 'imsave', 'FileImageStack', 'FileCollectionStack']

from _single import iminfo, imread, imsave
from _stack import FileImageStack
from _collection import FileCollectionStack

import formats
