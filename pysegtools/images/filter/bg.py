"""Filters that change, remove, or add a background to the image."""

from ..types import im_standardize_dtype
from ..colors import get_color

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
    elif len(padding) != 4 or all(isinstance(x, (int, long)) for x in padding): raise ValueError
    t,l,b,r = padding
    if bg == 'mean':     bg = mean(im[t:b+1,l:r+1])
    elif bg != 'mirror': bg = get_color(bg, im)
    if bg == 'mirror':
        im[:t,:]  = im[2*t-1:t-1:-1,:]
        im[:,:l]  = im[:,2*l-1:l-1:-1]
        im[-b:,:] = im[-b-1:-2*b-1:-1,:]
        im[:,-r:] = im[:,-r-1:-2*r-1:-1]
    else:
        im[:t,:]  = bg
        im[:,:l]  = bg
        im[-b:,:] = bg
        im[:,-r:] = bg
    return im

def crop(im, padding=None):
    """
    Crops an image, removing the padding from each side (top, left, bottom, right). If the padding
    is not given, it is calculated with get_bg_padding. Returns a view, not a copy.
    """
    im = im_standardize_dtype(im)
    if padding == None: padding = get_bg_padding(im)
    elif len(padding) != 4 or all(isinstance(x, (int, long)) for x in padding): raise ValueError
    return im[t:-b, l:-r]

def pad(im, padding):
    """Pad an image with 0s. The amount of padding is given by top, left, bottom, and right."""
    # TODO: could use "pad" function instead
    from numpy import zeros
    im = im_standardize_dtype(im)
    h,w = im.shape[:2]
    im_out = zeros((h+b+t,w+r+l),dtype=im.dtype)
    im_out[t:t+h,l:l+w] = im
    return im_out
