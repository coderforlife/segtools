"""Image Filters for Complex Images"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import repeat

from numpy import float64, complex128, real_if_close, dstack, zeros_like
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from ..types import check_image, check_image_single_channel, im_decomplexify, im_decomplexify_dtype, im_complexify_dtype
from ..types import get_dtype_min_max, get_dtype_max, get_im_dtype_and_nchan
from ._stack import FilteredImageStack, FilteredImageSlice
from ...imstack import Command, CommandEasy, Opt, Help

__all__ = ['real', 'imag', 'im_decomplexify', 'im_complexify', 'fft', 'ifft',
           'RealImageStack', 'ImagImageStack', 'DecomplexifyImageStack', 'ComplexifyImageStack',
           'FFTImageStack', 'IFFTImageStack']

def real(im):
    check_image(im)
    im = im_decomplexify(im)
    if im.ndim != 3 or im.shape[2] != 2: raise ValueError('Unsupported image type')
    return im[:,:,0]

def imag(im):
    check_image(im)
    im = im_decomplexify(im)
    if im.ndim != 3 or im.shape[2] != 2: raise ValueError('Unsupported image type')
    return im[:,:,1]

def im_complexify(real, imag=None, force=True):
    from ..types import im_complexify
    if imag is None:
        if real.dtype.kind == 'c' or real.ndim == 3 and real.shape[2] == 2:
            check_image(real)
            return im_complexify(real, force)
        real = check_image_single_channel(real)
        imag = zeros_like(real)
    else:
        real = check_image_single_channel(real)
        imag = check_image_single_channel(imag)
        if real.dtype != imag.dtype: raise ValueError('Real and imaginary data types must be the same')
    return im_complexify(dstack(real, imag), force)

def fft(im, shift=True):
    o_dt = im.dtype
    if o_dt.kind in 'ui':
        mn, mx = get_dtype_min_max(o_dt)
        im = im.astype(float64)
        if o_dt.kind == 'i': im += mn
        im /= mx
    im = fft2(check_image_single_channel(im))
    return fftshift(im) if shift else im
    
def ifft(im, shift=True):
    check_image(im)
    if im.dtype.kind != 'c': raise ValueError('Unsupported image type')
    return real_if_close(ifft2(ifftshift(im) if shift else im)) # should always end up real


##### Image Stacks #####
class __SameShapeImageSlice(FilteredImageSlice):
    def _get_props(self): self._set_props(None, self._input.shape)

class RealImageStack(FilteredImageStack):
    def __init__(self, ims): super(RealImageStack, self).__init__(ims, RealImageSlice)
class RealImageSlice(__SameShapeImageSlice):
    def __init__(self, stack, z, im):
        super(RealImageSlice, self).__init__(stack, z, im)
        dt = im.dtype
        if dt.kind != 'c' and dt.shape != (2,): raise ValueError('Not a complex image')
        self._set_props(create_im_dtype(im_decomplexify_dtype(dt).base, dt.byteorder, 1), None)
    def _get_data(self): return real(self._input.data)
class ImagImageStack(FilteredImageStack):
    def __init__(self, ims): super(ImagImageStack, self).__init__(ims, ImagImageSlice)
class ImagImageSlice(RealImageSlice):
    def _get_data(self): return imag(self._input.data)

class DecomplexifyImageStack(FilteredImageStack):
    def __init__(self, ims): super(DecomplexifyImageStack, self).__init__(ims, DecomplexifyImageSlice)
class DecomplexifyImageSlice(__SameShapeImageSlice):
    def __init__(self, stack, z, im):
        super(DecomplexifyImageSlice, self).__init__(stack, z, im)
        dt = im.dtype
        if dt.kind != 'c': raise ValueError('Not a complex image')
        self._set_props(create_im_dtype(im_decomplexify_dtype(dt).base, dt.byteorder, 2), None)
    def _get_data(self): return im_decomplexify(self._input.data)

class ComplexifyImageStack(FilteredImageStack):
    def __init__(self, real, imag=None, force=True):
        self._force = force
        real = ImageStack.as_image_stack(real)
        if imag is not None:
            imag = ImageStack.as_image_stack(imag)
            if len(real) != len(imag): raise ValueError('Real and imaginary parts must be the same shape and data type')
        else:
            imag = repeat(None)
        super(InvertImageStack, self).__init__(real,
            [ComplexifyImageSlice(self,z,R,I) for z,(R,I) in enumerate(zip(real,imag))])
class ComplexifyImageSlice(FilteredImageSlice):
    def __init__(self, stack, z, R, I):
        super(ComplexifyImageSlice, self).__init__(stack, z, R)
        self.__imag = I
        dt, sh = R.dtype, R.shape
        if I is None:
            dt, nchan = get_im_dtype_and_nchan()
            if dt.kind == 'c' or nchan > 2: raise ValueError()
        else:
            if dt.kind == 'c' or len(dtype.shape): raise ValueError()
            if I.dtype != dt or I.shape != sh: raise ValueError('Real and imaginary parts must be the same shape and data type')
        self._set_props(im_complexify_dtype(dt, self._stack._force), sh)
    def _get_data(self): return im_complexify(self._input.data, self.__imag.data, self._stack._force)


class FFTImageStack(FilteredImageStack):
    def __init__(self, ims, shift=True):
        super(FFTImageStack, self).__init__(ims, FFTImageSlice)
        self._shift = shift
class FFTImageSlice(__SameShapeImageSlice):
    def __init__(self, stack, z, im):
        super(FFTImageSlice, self).__init__(stack, z, im)
        if im.dtype.kind == 'c' or len(im.dtype.shape): raise ValueError('Not a single-channel image')
        self._set_props(complex128, None)
    def _get_data(self): return fft(self._input.data, self._stack._shift)

class IFFTImageStack(FilteredImageStack):
    def __init__(self, ims, shift=True):
        super(IFFTImageStack, self).__init__(ims, IFFTImageSlice)
        self._shift = shift
class IFFTImageSlice(__SameShapeImageSlice):
    def __init__(self, stack, z, im):
        super(IFFTImageSlice, self).__init__(stack, z, im)
        if im.dtype.kind != 'c': raise ValueError('Not a complex image')
        self._set_props(float64, None)
    def _get_data(self): return ifft(self._input.data, self._stack._shift)


##### Commands #####
class RealCommand(CommandEasy):
    @classmethod
    def name(cls): return 'real'
    @classmethod
    def _desc(cls): return 'Extracts the real component of a complex or complex-like image.'
    @classmethod
    def flags(cls): return ('real',)
    @classmethod
    def _consumes(cls): return ('Complex image',)
    @classmethod
    def _produces(cls): return ('Real component',)
    @classmethod
    def _see_also(cls): return ('imag','decomplexify','complexify')
    def __str__(self): return 'real'
    def execute(self, stack): stack.push(RealImageStack(stack.pop()))
class ImagCommand(CommandEasy):
    @classmethod
    def name(cls): return 'imag'
    @classmethod
    def _desc(cls): return 'Extracts the imaginary component of a complex or complex-like image.'
    @classmethod
    def flags(cls): return ('imag',)
    @classmethod
    def _consumes(cls): return ('Complex image',)
    @classmethod
    def _produces(cls): return ('Imaginary component',)
    @classmethod
    def _see_also(cls): return ('real','decomplexify','complexify')
    def __str__(self): return 'imag'
    def execute(self, stack): stack.push(ImagImageStack(stack.pop()))
class DecomplexifyCommand(CommandEasy):
    @classmethod
    def name(cls): return 'decomplexify'
    @classmethod
    def _desc(cls): return 'Converts a complex image into a 2-channel image.'
    @classmethod
    def flags(cls): return ('decomplexify',)
    @classmethod
    def _consumes(cls): return ('Complex image',)
    @classmethod
    def _produces(cls): return ('2-channel image with real and imaginary parts in the two channels',)
    @classmethod
    def _see_also(cls): return ('real','imag','complexify')
    def __str__(self): return 'decomplexify'
    def execute(self, stack): stack.push(DecomplexifyImageStack(stack.pop()))
class ComplexifyCommand(Command):
    @classmethod
    def name(cls): return 'complexify'
    @classmethod
    def flags(cls): return ('complexify',)
    @classmethod
    def _opts(cls): return (
            Opt('two_stacks', 'If true, use a second stack as the imaginary part', Opt.cast_bool(), False),
            )
    @classmethod
    def print_help(cls, width):
        p = Help(width)
        p.title("Complexify")
        p.text("""
Create a complex image by either combining two image stacks for the real and imaginary parts,
combining two channels from a single image stack, or by using a single image stack for the real
part and setting the imaginary part to 0.""")
        p.newline()
        p.flags(cls.flags())
        p.newline()
        p.text("""
Consumes:  1 single/double channel image stack or 2 single channel image stacks 
Produces:  1 combined image stack""")
        p.newline()
        p.text("Command format:")
        p.cmds("--complexify [two_stacks]")
        p.newline()
        p.text("Options:")
        p.opts(*cls._opts())
        p.newline()
        p.text("""
If two_stacks is true, then two image stacks are used to create the new complex image, taking the
real part from the first one and the imaginary part from the second one. Otherwise they come from
the two channels of the one stack read (and if there is only one channel, then the imaginary part
is filled in with 0).
""")
        p.newline()
        p.text("See also:")
        p.list('real','imag','decomplexify')
    def __str__(self): 'complexify'+(' (using 2 stacks)' if self.__two_stacks else '')
    def __init__(self, args, stack):
        self.__two_stacks, = args.get_all(*ComplexifyCommand._opts())
        stack.pop()
        if self.__two_stacks: stack.pop()
        stack.push()
    def execute(self, stack):
        stack.push(ComplexifyImageStack(stack.pop(), stack.pop() if self.__two_stacks else None))

class FFTCommand(CommandEasy):
    _shift = None
    @classmethod
    def name(cls): return 'Fast Fourier Transform'
    @classmethod
    def _desc(cls): return 'Takes the fast Fourier Transform of an image.'
    @classmethod
    def _opts(cls): return (
        Opt('shift', 'Shift the Fourier-space image so that the 0 frequency component is in the center of the image', Opt.cast_bool, False),
        )
    @classmethod
    def flags(cls): return ('fft',)
    @classmethod
    def _consumes(cls): return ('Grayscale image',)
    @classmethod
    def _produces(cls): return ('Complex image representing Fourier space',)
    @classmethod
    def _see_also(cls): return ('ifft',)
    def __str__(self): return 'fft'+(' (with shift)' if self._shift else '')
    def execute(self, stack): stack.push(FFTImageStack(stack.pop(), self._shift))
class IFFTCommand(CommandEasy):
    _shift = None
    @classmethod
    def name(cls): return 'Inverse Fast Fourier Transform'
    @classmethod
    def _desc(cls): return 'Takes the inverse fast Fourier Transform of a complex image.'
    @classmethod
    def _opts(cls): return (
        Opt('shift', 'Un-shift the Fourier-space image so that the 0 frequency component is in the top-left corner', Opt.cast_bool, False),
        )
    @classmethod
    def flags(cls): return ('ifft',)
    @classmethod
    def _consumes(cls): return ('Complex image representing Fourier space',)
    @classmethod
    def _produces(cls): return ('Grayscale image',)
    @classmethod
    def _see_also(cls): return ('fft',)
    def __str__(self): return 'ifft'+(' (with shift)' if self._shift else '')
    def execute(self, stack): stack.push(IFFTImageStack(stack.pop(), self._shift))
