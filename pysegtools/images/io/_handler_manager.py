"""
Defines the handler-manager which manages handlers for a class. This is used by the FileImageStack
and FileImageSource classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta
from io import open #pylint: disable=redefined-builtin

from ...imstack import Help
from ...general.utils import all_subclasses

class _HandlerManagerMeta(ABCMeta):
    """
    The meta-class for the handler-manager, which extends ABCMeta to call Help.register if
    applicable.
    """
    def __new__(cls, clsname, bases, dct):
        c = super(_HandlerManagerMeta, cls).__new__(cls, clsname, bases, dct)
        n = c.name()
        if n is not None:
            names = (n,c.__name__) + tuple(ext.lstrip('.').lower() for ext in c.exts())
            Help.register(names, c.print_help)
        return c

class HandlerManager(object):
    __metaclass__ = _HandlerManagerMeta

    @classmethod
    def is_handler(cls, handler, read=True):
        """Checks that the given string is a valid handler for this type."""
        #pylint: disable=protected-access
        assert cls != HandlerManager
        return any(handler == sub.name() and (read and sub._can_read() or not read and sub._can_write())
                   for sub in all_subclasses(cls))

    @classmethod
    def handlers(cls, read=True):
        """Get a list of all handlers of this type."""
        #pylint: disable=protected-access
        assert cls != HandlerManager
        handlers = []
        for sub in all_subclasses(cls):
            h = sub.name()
            if h is not None and (read and sub._can_read() or not read and sub._can_write()):
                handlers.append(h)
        return handlers

    @classmethod
    def __openable_by(cls, filename, readonly=False, handler=None, **options):
        #pylint: disable=protected-access
        handlers = (h for h in all_subclasses(cls) if h._can_read() and (readonly or h._can_write()))
        if handler is not None:
            for h in handlers:
                if handler == h.name(): return h
            raise ValueError('No handler named "%s"' % handler)
        for h in handlers:
            with open(filename, 'rb') as f:
                try:
                    if h._openable(filename, f, readonly, **options): return h
                except StandardError: pass
        raise ValueError('Unable to find handler for opening file "%s"' % filename)

    @classmethod
    def open(cls, filename, readonly=False, handler=None, **options):
        """
        Opens an existing image file. Extra options are only supported by some file handlers.
        """
        assert cls != HandlerManager
        return cls.__openable_by(filename, readonly, handler, **options). \
               open(filename, readonly, **options)

    @classmethod
    def openable(cls, filename, readonly=False, handler=None, **options):
        """
        Checks if an existing image file can be opened with the given arguments. Extra options are
        only supported by some file handlers.
        """
        assert cls != HandlerManager
        try: cls.__openable_by(filename, readonly, handler, **options); return True
        except StandardError: return False

    @classmethod
    def __creatable_by(cls, filename, writeonly=False, handler=None, **options):
        #pylint: disable=protected-access
        from os.path import splitext
        ext = splitext(filename)[1].lower()
        handlers = (h for h in all_subclasses(cls) if h._can_write() and (writeonly or h._can_read()))
        if handler is not None:
            for h in handlers:
                if handler == cls.name(): return h
            raise ValueError('No image source handler named "'+handler+'" for creating files')
        for h in handlers:
            try:
                if h._creatable(filename, ext, writeonly, **options): return h
            except StandardError: pass
        raise ValueError('Unable to find image source handler for creating file "'+filename+'"')

    @classmethod
    def _create_trans(cls, im): return im

    @classmethod
    def create(cls, filename, im, writeonly=False, handler=None, **options):
        """
        Creates an image file. Extra options are only supported by some file handlers.

        Selection of a handler and format is purely on file extension and options given.

        Note that the "writeonly" flag is only used for optimization and may not always been
        honored. It is your word that you will not use any functions that get data from the
        stack.
        """
        assert cls != HandlerManager
        return cls.__creatable_by(filename, writeonly, handler, **options). \
               create(filename, cls._create_trans(im), writeonly, **options)

    @classmethod
    def creatable(cls, filename, writeonly=False, handler=None, **options):
        """
        Checks if a filename can written to as a new image file. Extra options are only supported by
        some file handlers.
        """
        assert cls != HandlerManager
        try: cls.__creatable_by(filename, writeonly, handler, **options); return True
        except StandardError: return False

    @classmethod
    def _openable(cls, filename, f, readonly, **opts): #pylint: disable=unused-argument
        """
        [To be implemented by handler, default is nothing is openable]

        Return if a file is openable given the filename, file object, and dictionary of options. If
        this returns True then the class must provide a static/class method like:
            `open(filename, readonly, **options)`
        Option keys are always strings, values can be either strings or other values (but strings
        must be accepted for any value and you must convert, if possible). While _openable should
        return False if there any unknown option keys or option values cannot be used, open should
        throw exceptions.
        """
        return False

    @classmethod
    def _creatable(cls, filename, ext, writeonly, **opts): #pylint: disable=unused-argument
        """
        [To be implemented by handler, default is nothing is creatable]

        Return if a filename/ext (ext always lowercase and includes .) is creatable as given the
        dictionary of options. If this returns True then the class must provide a static/class
        method like:
            `create(filename, IMAGE, writeonly, **options)`
        Option keys are always strings, values can be either strings or other values (but strings
        must be accepted for any value and you must convert, if possible). While _creatable should
        return False if there any unknown option keys or option values cannot be used, create should
        throw exceptions.

        The IMAGE is either an ImageSource for source handlers or an ImageStack for stack handlers.

        Note that the "writeonly" flag is only used for optimization and may not always been
        honored. It is the word of the caller they will not use any functions that get data from
        the stack. The handler may ignore this and treat it as read/write.
        """
        return False

    @classmethod
    def _can_read(cls):
        """
        [To be implemented by handler, default is readable]

        Returns True if this handler can, under any circumstances, read images.
        """
        return True

    @classmethod
    def _can_write(cls):
        """
        [To be implemented by handler, default is writable]

        Returns True if this handler can, under any circumstances, write images.
        """
        return True

    @classmethod
    def name(cls):
        """
        [To be implemented by handler, default causes the handler to not have a help page, be
        unusable by name, and not be listed, but still can handle things]

        Return the name of this image handler to be displayed in help outputs.
        """
        return None

    @classmethod
    def exts(cls):
        """
        [To be implemented by handler, default returns empty tuple the handler to not have any extra
        help page names]

        Return a tuple of lower-case exts including the . that this image handler recongnizes for
        writing (and common extensions for readable types). These are added as help pages. In some
        cases it may make to not return any extensions even if extensions are used to determine if
        a file can be created.
        """
        return ()

    @classmethod
    def print_help(cls, width):
        """
        [To be implemented by handler, default prints nothing]

        Prints the help page of this image handler.
        """
        pass
