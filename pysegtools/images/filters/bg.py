"""Filters that change, remove, or add a background to the image."""

from sys import stdin, stdout
from io import open
from itertools import islice
from re import compile

from _stack import FilteredImageStack, FilteredImageSlice, UnchangingFilteredImageStack, UnchangingFilteredImageSlice, FilterOption as Opt
from ..types import im_standardize_dtype
from ..colors import get_color, is_color

__all__ = ['get_bg_color','get_bg_padding','bgfill','crop','pad']

def get_bg_color(im, bg=None):
    """Get the color of the background by looking at the edges of the image."""
    # Calculate bg color using solid strips on top, bottom, left, or right
    im = im_standardize_dtype(im)
    if (im[0,:] == im[0,0]).all() or (im[:,0] == im[0,0]).all():
        return im[0,0]
    elif (im[-1,:] == im[-1,-1]).all() or (im[:,-1] == im[-1,-1]).all():
        return im[-1,-1]
    else: return None

def get_bg_padding(im, bg=None):
    """
    Get the area of the background as the amount on the top, left, bottom, and right. If bg is not
    given, it is calculated from the edges of the image.
    """
    im = im_standardize_dtype(im)
    if bg == None:
        bg = get_bg_color(im)
        if bg == None: return (0,0,0,0) # no discoverable bg color, no paddding
    else:
        bg = get_color(bg, im)
    shp = im.shape
    w,h = shp[1]-1, shp[0]-1
    t,l,b,r = 0, 0, h, w
    while t < h and (im[t,:] == bg).all(): t += 1
    while b > t and (im[b,:] == bg).all(): b -= 1
    while l < w and (im[:,l] == bg).all(): l += 1
    while r > l and (im[:,r] == bg).all(): r -= 1
    return (t,l,h-b,w-r)

def bgfill(im, padding=None, bg='black'):
    """
    Fills the 'background' of the image with a solid color or mirror. The foreground is given by the
    amount of padding (top, left, bottom, right). If the padding is not given, it is calculated with
    get_bg_padding.

    The background can be 'mirror', 'mean', or any value supported by get_color for the image type.
    If 'mirror' then the background is filled with a reflection of the foreground (currently when
    using reflection, the foreground must be wider/taller than the background). If bg 'mean' then
    the background is filled with the average foreground color. This is only supported for grayscale
    images.

    Operates on the array directly and does not copy it.
    """
    from numpy import mean
    im = im_standardize_dtype(im)
    if padding == None: padding = get_bg_padding(im)
    elif len(padding) != 4 or any(not isinstance(x, (int, long)) for x in padding): raise ValueError
    t,l,b,r = padding
    h,w = im.shape[:2]
    B,R = h-b,w-r
    if   bg == 'mean':   bg = mean(im[t:b+1,l:r+1])
    elif bg != 'mirror': bg = get_color(bg, im)
    if bg == 'mirror':
        im[ :t, : ] = im[2*t-1:t-1:-1,:]
        im[ : , :l] = im[:,2*l-1:l-1:-1]
        im[B: , : ] = im[-b-1:-2*b-1:-1,:]
        im[ : ,R: ] = im[:,-r-1:-2*r-1:-1]
    else:
        im[ :t, : ] = bg
        im[ : , :l] = bg
        im[B: , : ] = bg
        im[ : ,R: ] = bg
    return im

def crop(im, padding=None):
    """
    Crops an image, removing the padding from each side (top, left, bottom, right). If the padding
    is not given, it is calculated with get_bg_padding. Returns a view, not a copy.
    """
    im = im_standardize_dtype(im)
    if padding == None: padding = get_bg_padding(im)
    elif len(padding) != 4 or any(not isinstance(x, (int, long)) for x in padding): raise ValueError
    t,l,b,r = padding
    h,w = im.shape[:2]
    return im[t:h-b, l:w-r]

def pad(im, padding):
    """Pad an image with 0s. The amount of padding is given by top, left, bottom, and right."""
    # TODO: could use "pad" function instead
    from numpy import zeros
    im = im_standardize_dtype(im)
    if len(padding) != 4 or any(not isinstance(x, (int, long)) for x in padding): raise ValueError
    t,l,b,r = padding
    h,w = im.shape[:2]
    im_out = zeros((h+b+t,w+r+l),dtype=im.dtype)
    im_out[t:t+h,l:l+w] = im
    return im_out


##### Stack Classes #####

_re_4_ints = compile("^\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*$")

class SaveBackgroundPadding(UnchangingFilteredImageStack):
    @classmethod
    def _name(cls): return 'Save Background Padding'
    @classmethod
    def _desc(cls): return 'Save the background padding area, calculated from solid strips of color on the sides of the images. This color can be calucated from the image slices themselves or a predetermined color.'
    @classmethod
    def _flags(cls): return ('save-padding',)
    @classmethod
    def _opts(cls): return (
        Opt('file', 'The file name to save to or \'-\' for standard out', Opt.cast_or(Opt.cast_equal('-'), Opt.cast_writable_file())),
        Opt('color', 'The current background color (see --help color) or \'auto\' to calculate it per slice', Opt.cast_or(Opt.cast_equal('auto'), Opt.cast_check(is_color)), 'auto')
        )
    @classmethod
    def _supported(cls, dtype): return True
    def description(self): return 'save background padding %sto %s'%(
        ('of color %s '%self._color) if self._color!='auto' else '',
        '<standard out>' if self._filename is None else self._filename,
        )
    def __init__(self, ims, file, color='auto'):
        self._file = None
        self._paddings_saved = 0
        self._filename = None if file=='-' else file
        self._color = None if color=='auto' else color
        super(SaveBackgroundPadding, self).__init__(ims, SaveBackgroundPaddingImageSlice)
    def _calc_padding(self, z):
        if self._paddings_saved == self._d: return self._slices[z]._input.data
        if self._file is None: self._file = stdout if self._filename is None else open(self._filename, 'w')
        for im in islice(self._slices, self._paddings_saved, z+1):
            data = im._input.data
            self._file.write(u"%d %d %d %d\n"%get_bg_padding(data, self._color))
        self._paddings_saved = z + 1
        if self._paddings_saved == self._d and self._filename is not None: self._file.close()
        return data

class SaveBackgroundPaddingImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self): return self._stack._calc_padding(self._z)

class _BackgroundPaddingImageStack(FilteredImageStack):
    @classmethod
    def _opts(cls): return (
        Opt('padding', 'Color to use to use to detect the background color or \'auto\' to calculate the padding color automatically', Opt.cast_or(Opt.cast_equal('auto'), Opt.cast_check(is_color)), 'auto'),
        Opt('file', 'File to read (or \'-\' for standard in) with lines of four numbers for top/left/bottom/right padding values', Opt.cast_or(Opt.cast_in('-',None), Opt.cast_readable_file()), None),
        )
    @classmethod
    def _supported(cls, dtype): return True
    def __init__(self, ims, slc_cls, padding='auto', file=None):
        super(_BackgroundPaddingImageStack, self).__init__(ims, slc_cls)
        self._use_padding = file is None
        if self._use_padding:
            self._padding = None if padding == 'auto' else padding
        else:
            self._filename = None if file == '-' else file
            self._file = None
            self._paddings = []
    def _get_padding(self, im, z):
        if self._use_padding: return get_bg_padding(im, self._padding) # TODO: cache this value like we cache read-in values?
        if len(self._paddings) <= z:
            if self._file is None: self._file = stdin if self._filename is None else open(self._filename, 'r')
            while len(self._paddings) <= z:
                result = _re_4_ints.search(self._file.readline())
                if result is None: # file ended prematurely or there was an invalid line
                    last = self._padding[-1] if len(self._padding) else (0,0,0,0)
                    self._padding.extend(last for i in xrange(self._d-len(self._paddings)))
                    break
                else:
                    self._paddings.append(tuple(int(x) for x in result.groups()))
            if len(self._paddings) == self._d and self._filename is not None: self._file.close()
        return self._paddings[z]
    
class FillBackgroundPaddingImageStack(_BackgroundPaddingImageStack):
    @classmethod
    def _name(cls): return 'Fill Background Padding'
    @classmethod
    def _desc(cls): return 'Fill the background padding area, defined either from predefined values in a file or is calculated from solid strips of color on the sides of the images (this color can be calculated from the image slices themselves or a predetermined color). The fill color can be the mean of the foreground, a mirror of the foreground, or a specific solid color.'
    @classmethod
    def _flags(cls): return ('fill-padding','fill-bg','fill-background')
    @classmethod
    def _opts(cls): return (
        Opt('fill', 'The name of a color to use to fill with (see --help color), \'mean\' to use the average foreground color, or \'mirror\' to mirror the foreground onto the background', Opt.cast_check(is_color), 'black'),
        ) + _BackgroundPaddingImageStack._opts()
    @classmethod
    def _example(cls): return """You should only specify one of \'padding\' or \'file\'. If the file has less lines than the images in the stack, the last one is repeated as necessary. The numbers can be seperated by spaces, tabs, or commas.

Examples:
  fill calculated bg with black: --fill-padding
  fill calculated bg with mirror: --fill-padding mirror
  fill bg defined by path with black: --fill-padding file=path
  fill a black bg with white: --fill-padding white padding=black"""
    def description(self): return 'fill background%s with %s'%(
        ('' if self._padding is None else ' of %s'%self._padding) if self._use_padding else (' defined by %s'%('<standard in>' if self._filename is None else self._filename)),
        ('mean foreground color' if self._color=='mean' else ('mirror of foreground' if self._color=='mirror' else self._color)),
        )
    def __init__(self, ims, fill='black', padding='auto', file=None):
        self._fill = fill
        super(FillBackgroundPaddingImageStack, self).__init__(ims, FillBackgroundPaddingImageSlice, padding, file)
    def _get_homogeneous_info(self): return self._ims._get_homogeneous_info() # from UnchangingFilteredImageStack

class FillBackgroundPaddingImageSlice(UnchangingFilteredImageSlice):
    def _get_data(self):
        im = self._input.data
        return bgfill(im, self._stack._get_padding(im, self._z), self._stack._fill)

class CropBackgroundPaddingImageStack(_BackgroundPaddingImageStack):
    @classmethod
    def _name(cls): return 'Crop Background Padding'
    @classmethod
    def _desc(cls): return 'Crop the background padding area, defined either from predefined values in a file or is calculated from solid strips of color on the sides of the images (this color can be calculated from the image slices themselves or a predetermined color).'
    @classmethod
    def _flags(cls): return ('c','crop','crop-padding','crop-bg','crop-background',)
    @classmethod
    def _example(cls): return """You should only specify one of \'padding\' or \'file\'. If the file has less lines than the images in the stack, the last one is repeated as necessary. The numbers can be seperated by spaces, tabs, or commas.

Examples:
  crop calculated bg: --crop
  crop bg defined by path: --crop file=path
  crop a black bg: --crop padding=black"""
    @classmethod
    def _supported(cls, dtype): return True
    def description(self): return 'crop background%s'%('' if self._padding is None else ' of %s'%self._padding) if self._use_padding else (' defined by %s'%('<standard in>' if self._filename is None else self._filename))
    def __init__(self, ims, padding='auto', file=None): super(CropBackgroundPaddingImageStack, self).__init__(ims, CropBackgroundPaddingImageSlice, padding, file)

class CropBackgroundPaddingImageSlice(FilteredImageSlice):
    def _get_padding(self):
        im = self._input.data # TODO: this re-gets image... and if we are doing file-based we don't need the image data at all
        p = self._stack._get_padding(im, self._z)
        s = self._input.shape
        self._set_props(self._input.dtype, s[1]-p[1]-p[3], s[0]-p[0]-p[2])
        return im, p
    def _get_props(self): self._get_padding()
    def _get_data(self): return crop(*self._get_padding())

class AddBackgroundPaddingImageStack(_BackgroundPaddingImageStack):
    @classmethod
    def _name(cls): return 'Add Background Padding'
    @classmethod
    def _desc(cls): return 'Add background padding defined from predefined values in a file. The padding area is always solid black (or whatever 0 is for the image type).'
    @classmethod
    def _flags(cls): return ('p','pad','add-padding',)
    @classmethod
    def _opts(cls): return (
        Opt('file', 'File to read (or \'-\' for standard in) with lines of four numbers for top/left/bottom/right padding values', Opt.cast_or(Opt.cast_equal('-'), Opt.cast_readable_file())),
        )
    @classmethod
    def _example(cls): return """If the file has less lines than the images in the stack, the last one is repeated as necessary. The numbers can be seperated by spaces, tabs, or commas.

Examples:
  add padding from a file: --pad path"""
    @classmethod
    def _supported(cls, dtype): return True
    def description(self): return 'add padding defined by %s'%('<standard in>' if self._filename is None else self._filename)
    def __init__(self, ims, file): super(AddBackgroundPaddingImageStack, self).__init__(ims, AddBackgroundPaddingImageSlice, None, file)

class AddBackgroundPaddingImageSlice(FilteredImageSlice):
    def _get_padding(self):
        p = self._stack._get_padding(None, self._z) # can use im as None since it is never calculated from the image
        s = self._input.shape
        self._set_props(self._input.dtype, s[1]+p[1]+p[3], s[0]+p[0]+p[2])
        return p
    def _get_props(self): self._get_padding()
    def _get_data(self):  return pad(self._input.data, self._get_padding())
