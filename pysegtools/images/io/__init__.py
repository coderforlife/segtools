"""Image IO functions. All images are numpy arrays in memory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = ['FileImageSource', 'FileImageStack', 'FileCollectionStack']

from ._single import FileImageSource
from ._stack import FileImageStack
from ._collection import FileCollectionStack

from . import handlers
