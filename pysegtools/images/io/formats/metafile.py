from collections import Iterable, Sequence
from itertools import product
from functools import partial
from numpy import array, empty
import re, os

from ....general.gzip import GzipFile
from ....general.enum import Enum
from ..._util import prod, dtype_cast, itr2str, splitstr, get_list, _bool

from ...types import *
from .._single import iminfo, imread, imsave
from .._stack import ImageStack, Header, Field, FixedField
from .._util import openfile, get_file_size, file_shift_contents, imread_raw, imsave_raw, imread_ascii_raw, imsave_ascii_raw

__all__ = ['iminfo_mha', 'iminfo_mhd', 'imread_mha', 'imread_mhd', 'imsave_mha', 'imsave_mhd']
 # TODO: 'Metafile'

dtype2met = {
    # handle these specially
    #IM_RGB24  : 'MET_UCHAR_ARRAY',
    #IM_RGBA32 : 'MET_UCHAR_ARRAY',
    IM_UINT8  : 'MET_UCHAR',
    IM_INT8   : 'MET_CHAR',
    IM_INT16  : 'MET_SHORT',      IM_INT16_BE  : 'MET_SHORT',
    IM_UINT16 : 'MET_USHORT',     IM_UINT16_BE : 'MET_USHORT',
    IM_INT32  : 'MET_INT',        IM_INT32_BE  : 'MET_INT',
    IM_UINT32 : 'MET_UINT',       IM_UINT32_BE : 'MET_UINT',
    IM_INT64  : 'MET_LONG_LONG',  IM_INT64_BE  : 'MET_LONG_LONG',
    IM_UINT64 : 'MET_ULONG_LONG', IM_UINT64_BE : 'MET_ULONG_LONG',
    IM_FLOAT32 : 'MET_FLOAT',
    IM_FLOAT64 : 'MET_DOUBLE',
}
met2dtype = { v:k for k,v in dtype2met.iteritems() }
met2dtype['MET_LONG']  = IM_INT32  # synonyms
met2dtype['MET_ULONG'] = IM_UINT32
# Note: MET_*_ARRAY is equivilient to MET_*, we standardize while reading the header

req_keys = ('ObjectType','NDims','DimSize','ElementType','ElementDataFile')
synonyms = {
    'ElementByteOrderMSB':'BinaryDataByteOrderMSB',
    'Position':'Offset','Origin':'Offset',
    'Orientation':'TransformMatrix','Rotation':'TransformMatrix',
    }

# TODO: the ':' seems to be a comment in the header according to source
header_line  = re.compile('^[A-Za-z0-9_-]+\s*=')
file_list    = re.compile('^LIST(?:\s+(\d+)D?)?$')
file_pattern = re.compile('^(\S*%[#0-9 +.-]*[hlL]?[dD]\S*)(?:\s+(\d+)(?:\s+(\d+)(?:\s+(\d+))?)?)?$')

# TODO: use ElementToIntensityFunctionSlope and ElementToIntensityFunctionOffset?

def parse_pattern(m, depth):
    pattern, start, stop, step = m.groups(1)
    start, step = int(start), int(step)
    stop = start+step*(depth-1) if m.lastindex < 3 else int(stop)
    if step <= 0 or depth != (stop-start)//step+1: raise ValueError('MHA/MHD invalid ElementDataFile pattern')
    return (pattern, start, stop, step)


####### Reading ################################################################

def read_data_file(datafile, off, datasize, imread, compression, dtype, shape):
    if off == -1: off = get_file_size(datafile) - datasize
    if off < 0: raise ValueError('MHA/MHD file data file is not large enough for the pixel data')
    with openfile(datafile, 'rb', compression, off=off) as f: return imread(f, shape, dtype, 'C')

def re_search(re, s):
    re_search.last_match = m = re.search(s)
    return m != None

#def is_text(s): return all(32<=ord(c)<=126 or c in '\t\n\r' for c in s)

def read_mha_header(f):
    # Read Header
    h = {}
    line = f.readline(256)
    while len(line) > 0 and header_line.search(line) != None:
        # TODO: types that are array types can have data on mutiple lines
        if line[-1] != '\n': line += f.readline() # read the rest of the line
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip()
        k = synonyms.get(k, k)
        h[k] = v
        if k == 'ObjectType' and v != 'Image': raise ValueError('Non-image MHA/MHD files are not supported')
        if k == 'ElementDataFile': break
        line = f.readline(256)
    else: raise ValueError('MHA/MHD file header does not have required field \'ElementDataFile\'')
    off = f.tell()

    # Check/Parse Header
    if any(k not in h for k in req_keys): raise ValueError('MHA/MHD file header does not have required field \''+k+'\'')
    ndims = int(h['NDims'])
    shape = splitstr(h['DimSize'], int)
    if ndims < 1 or len(shape) != ndims or any(x <= 0 for x in shape): raise ValueError('Invalid dimension sizes in MHA/MHD file')
    headersize = int(h.get('HeaderSize', 0))
    if headersize < -1: raise ValueError('MHA/MHD HeaderSize is invalid')
    binary = _bool(h.get('BinaryData', True))
    compressed = _bool(h.get('CompressedData', False))
    has_compressed_data_size = int(h.get('CompressDataSize', 0)) != 0
    if compressed and not binary: raise ValueError('MHA/MHD CompressedData cannot be True when BinaryData is False')
    if headersize == -1 and compressed and not has_compressed_data_size: raise ValueError('MHA/MHD HeaderSize cannot be -1 when CompressedData is True or CompressedDataSize is 0 or not given')
    if not compressed and has_compressed_data_size: raise ValueError('MHA/MHD CompressedDataSize cannot be given when CompressedData is False')
    if headersize == -1 and not binary: raise ValueError('MHA/MHD HeaderSize cannot be -1 when BinaryData is False')
    datafile = h['ElementDataFile']
    if datafile in ('LOCAL', 'Local', 'local'):
        h['ElementDataFile'] = 'LOCAL'
        datafile = filename
        if 'HeaderSize' not in h: headersize = off
    elif re_search(file_list, datafiles): # we have a list of files
        file_ndims = int(re_search.last_match.group(1)) if re_search.last_match.lastindex == 1 else ndims - 1
        if file_ndims == 0 or file_ndims > ndims: raise ValueError('Invalid ElementDataFile in MHA/MHD header')
        datafile = [f.readline().strip() for _ in xrange(prod(shape[file_ndims:]))]
    elif re_search(file_pattern, datafiles): # we have a pattern of files
        datafile = parse_pattern(re_search.last_match, shape[-1])
    
    # Check/Parse Element Type Header Keys
    etype = h['ElementType']
    if etype.endswith('_ARRAY'): etype = etype[:-6]
    echns = int(h.get('ElementNumberOfChannels', 1))
    endian = '>' if _bool(h.get('BinaryDataByteOrderMSB', False)) else '<'
    if echns == 4 and etype == 'MET_UCHAR': dtype = IM_RGBA32; h['ElementType'] = 'MET_UCHAR_ARRAY'
    if echns == 3 and etype == 'MET_UCHAR': dtype = IM_RGB24; h['ElementType'] = 'MET_UCHAR_ARRAY'
    elif echns == 1 and etype in met2dtype: dtype = met2dtype[etype].newbyteorder(endian); h['ElementType'] = etype
    else: raise ValueError('MHA/MHD file image type not supported')
    
    return h, dtype, datafile, headersize

def iminfo_mha(filename):
    """Equivilent to iminfo_mhd, MHD vs MHA files are determined based on the header in the file"""
    return iminfo_mhd(filename)
def iminfo_mhd(filename):
    """
    Read the header from an MHA/MHD image file and return shape (H, W) and dtype.
    """
    with openfile(filename, 'rb') as f:
        h, dtype, datafile, headersize = read_mha_header(f)
    shape = splitstr(h['DimSize'], int)
    shape.reverse()
    return shape, dtype
iminfo.register('.mha', iminfo_mha)
iminfo.register('.mhd', iminfo_mhd)

def imread_mha(filename):
    """Equivilent to imread_mhd, MHD vs MHA files are determined based on the header in the file"""
    return imread_mhd(filename)
def imread_mhd(filename):
    """
    Read an MHA/MHD image file. Returns both the header dictionary and the image data.

    Supported image data formats:
        Single channel unsigned and signed integral data types up to 8 bytes each
        Single channel float and double
        3-channel unsigned byte (MET_UCHAR_ARRAY[3])
        4-channel unsigned byte (MET_UCHAR_ARRAY[4])

    Unsupported features:
        Non-image files
        HeaderSize of -1 when data is ASCII or data is compressed without knowing the compressed data size
        Many image data formats
        Many fields are simply ignored (e.g. TransformationMatrix) but they are returned
    """
    # Read/Check/Parse Header
    with openfile(filename, 'rb') as f:
        h, dtype, datafile, headersize = read_mha_header(f)
    shape = splitstr(h['DimSize'], int)
    shape.reverse()
    imread = imread_raw if _bool(h.get('BinaryData', True)) else imread_ascii_raw
    compressed = _bool(h.get('CompressedData', False))
    compression = 'auto' if compressed else None

    # Read data
    if isinstance(datafile, list):
        # a list of files
        directory = os.path.dirname(os.path.realpath(filename))
        im = empty(shape, dtype)
        ind = [slice(None)] * len(shape)
        file_ndims = ndims
        while prod(shape[:-file_ndims]) != len(datafile): file_ndims -= 1
        inds = product(*[xrange(x) for x in shape[:-file_ndims]])
        shape = shape[-file_ndims:]
        datasize = h.get('CompressedDataSize', None) if compressed else prod(shape)*dtype.itemsize
        for f in datafile:
            ind[:-file_ndims] = inds.next()
            im[ind] = read_data_file(os.path.realpath(os.path.join(directory, f)), headersize, datasize, imread, compression, dtype, shape)
        return h, im
    elif isinstance(datafile, tuple):
        # follow a pattern to find all the files to be loaded
        directory = os.path.dirname(os.path.realpath(filename))
        im = empty(shape, dtype)
        shape = shape[1:]
        datasize = h.get('CompressedDataSize', None) if compressed else prod(shape)*dtype.itemsize
        pattern, start, stop, step = datafile
        for i in xrange(start, stop + 1, step):
            im[i,...] = read_data_file(os.path.realpath(os.path.join(directory, pattern % i)), headersize, datasize, imread, compression, dtype, shape)
        return h, im
    else:
        # datafile is just a file, starting at headersize
        datasize = h.get('CompressedDataSize', None) if compressed else prod(shape)*dtype.itemsize
        return h, read_data_file(datafile, headersize, datasize, imread, compression, dtype, shape)
imread.register('.mha', lambda filename: imread_mha(filename)[1])
imread.register('.mhd', lambda filename: imread_mhd(filename)[1])

####### Saving #################################################################

def imsave_mha(filename, im, CompressedData=False, BinaryData=True, **tags):
    """Save an image as an MHA image (data embeded in the metadata file). See imsave_mhd for more information."""
    imsave_mhd(filename, im, 'LOCAL', CompressedData, BinaryData, **tags)
def imsave_mhd(filename, im, datafile=None, CompressedData=False, BinaryData=True, **tags):
    """
    Save an image as an MHD image.
    
    If the datafile name is not given it will be automatically generated. If it is 'LOCAL' than the
    data will be embeded in the main file (MHA-mode). If an iterable of files is given, each will
    contain a slice of the data. The number of files must be equal to the length of the highest
    dimension. The datafile cannot be the string 'LIST', start with 'LIST #', or contain a %d.

    If you set CompressedData to True it will cause the image data to be compressed. This will slow
    down saving but in many cases will result in significantly smaller files.

    If you set BinaryData to False it will cause the image data to be written in ASCII. This will
    slow down saving and will result in significantly larger file, but they are human-readable.
    
    You may specify extra tags to be saved in the image header. These are ignored for the most part
    and simply copied into the header. Known tags are checked and possibly corrected. For any tag
    that requires one value per dimension and only a single value is provided it will automatically
    be copied for each dimension.

    Currently many features of MHA/MHD files are not supported, including:
        Not all data types are understood
        Cannot output data listed datafiles that don't contain exactly 1 less dimension than the overall number of dimensions
        Cannot save non-image objects
        Cannot create a datafile with a non-MHA/MHD header (HeaderSize) (partially allowed in the Metafile ImageStack)
    Attempting to force these features through tags will raise errors.
    """

    from sys import byteorder

    # TODO: the line below is the only reason this function only works on 2D images
    # If it was "smarter" then this could work on any dimensional data
    im = im_standardize_dtype(im.dtype)
    BinaryData = _bool(BinaryData)
    CompressedData = _bool(CompressedData)
    if not BinaryData and CompressedData: raise ValueError('Cannot compress non-binary data')

    ndims = im.ndim
    shape = list(im.shape)
    shape.reverse()
    filename = os.path.realpath(filename.strip())
    directory = os.path.dirname(filename)
    if datafile == None:
        ext = ('.gz' if CompressedData else '.bin' if BinaryData else '.txt')
        datafile = os.path.splitext(filename)[0] + ext
        if os.path.exists(datafile):
            from tempfile import mktemp
            datafile = mktemp(ext, os.path.basename(filename), directory)
        elem_data_file = os.path.relpath(datafile, directory)
    elif isinstance(datafile, basestring):
        datafile = datafile.strip()
        if datafile in ('LOCAL', 'Local', 'local'):
            elem_data_file = datafile = 'LOCAL'
        elif re_search(file_pattern, datafile):
            p = parse_pattern(re_search.last_match, shape[-1])
            elem_data_file = itr2str(p)
            datafile = [os.path.realpath(os.path.join(directory,p[0]%i)) for i in xrange(p[1],p[2]+1,p[3])]
            datafile_list = []
        elif file_list.search(datafile) != None:
            raise ValueError('Datafile "'+datafile+'" is a reserved name, provide a list of files instead')
        else:
            elem_data_file = os.path.relpath(datafile, directory)
    elif isinstance(datafile, Sequence) and re_search(file_pattern, itr2str(datafile)):
        p = parse_pattern(re_search.last_match, shape[-1])
        elem_data_file = itr2str(p)
        datafile = [os.path.realpath(os.path.join(directory,p[0]%i)) for i in xrange(p[1],p[2]+1,p[3])]
        datafile_list = []
    elif isinstance(datafile, Iterable):
        datafile = [os.path.realpath(os.path.join(directory, x)) for x in datafile]
        datafile_list = [os.path.relpath(x, directory) for x in datafile]
        if len(datafile) != shape[-1]: raise ValueError('When using list datafiles there must be one file for each entry in the highest dimension')
        elem_data_file = 'LIST'

    # Setup tags that we compute from the image or other given options
    alltags = [
        ('ObjectType', 'Image'),
        ('NDims', str(ndims)),
        ('DimSize', itr2str(shape)),
        ('BinaryData', str(BinaryData)),
        ('BinaryDataByteOrderMSB', str(im.dtype.byteorder == '>' or im.dtype.byteorder == '=' and byteorder != 'little')),
        ('CompressedData', str(CompressedData)),
        ]
    if im.dtype == IM_RGB24 or im.dtype == IM_RGBA32:
        alltags.append(('ElementType', 'MET_UCHAR_ARRAY'))
        alltags.append(('ElementNumberOfChannels', '3' if im.dtype == IM_RGB24 else '4'))
    else:
        if im.dtype not in dtype2met: raise ValueError('Format of image is not supported')
        alltags.append(('ElementType', dtype2met[im.dtype]))
    alltags_dict = dict(alltags)
    alltags_dict['ElementDataFile'] = elem_data_file

    # These tags are not allowed in the extra tags as they are either implied from the image data or have required values
    not_allowed = req_keys + ('ObjectSubType', 'ElementNumberOfChannels', 'HeaderSize', # used to specify how data is stored, int >= -1 with -1 as special
                   'CompressedData', 'CompressedDataSize',
                   'BinaryData', 'BinaryDataByteOrderMSB', 'ElementByteOrderMSB')
    string    = ('Name', 'Comment', 'TransformType', 'AcquisitionDate')
    enums     = {'Modality':METModality, 'DistanceUnits':METDistanceUnits}
    ints      = ('ID', 'ParentID', 'ElementNBits')
    floats    = ('ElementToIntensityFunctionSlope', 'ElementToIntensityFunctionOffset')
    sames     = ('ElementMin', 'ElementMax') # values that must be the same as the data values
    n_floats  = ('ElementSize', 'ElementSpacing', 'Offset', 'Position', 'Origin', 'ImagePosition', 'CenterOfRotation')
    n2_floats = ('Rotation', 'Orientation', 'TransformMatrix')

    # These are synonyms, only one of each is allowed
    only_one = (
        ('BinaryDataByteOrderMSB', 'ElementByteOrderMSB'),
        ('Offset', 'Position', 'Origin'),
        ('Rotation', 'Orientation', 'TransformMatrix'),
        )

    # Process items
    for k,v in tags.items():
        if k in not_allowed:
            if k in alltags_dict and alltags_dict[k] == v: continue
            raise ValueError('Setting the tag "' + k_ + '" is not allowed')
        elif k in string:       v = str_nl_check(v)
        elif k in enums:        v = enums[k](v).name
        elif k in ints:         v = str(int(v))
        elif k in floats:       v = str(float(v))
        elif k in sames:        v = dtype_cast(v,im.dtype.base); v = itr2str(v) if isinstance(v, Iterable) else v
        elif k in n_floats:     v = itr2str(get_list(v, ndims, float))
        elif k in n2_floats:    v = itr2str(get_list(v, (ndims, ndims), float))
        elif k == 'Color':      v = itr2str(get_list(v, 4, float))
        elif k == 'SequenceID': v = itr2str(get_list(v, 4, int))
        elif k == 'AnatomicalOrientation': v = anatomical_orientation(v, ndims)
        elif re.search('[^a-z0-9_-]', k) != None:
            raise ValueError('Invalid key name "' + k + '"')
        else: continue
        tags[k] = v
    for name_set in only_one:
        if len([1 for n in name_set if n in tags]) > 1: raise ValueError('There can only be one tag from the set '+(', '.join(only_one)))
    alltags.extend(tags.iteritems())

    if datafile == 'LOCAL':
        hdr = ''.join((name+' = '+value+'\n') for name, value in alltags)
        if CompressedData: hdr += 'CompressedDataSize = '; cds_off = len(hdr); hdr += '0              \n'
        hdr += 'ElementDataFile = LOCAL\n'
        with openfile(filename, 'wb') as f:
            f.write(hdr)
            with openfile(f, 'wb', 'zlib' if CompressedData else None, comp_level=6) as df:
                imsave_raw(df, im) if BinaryData else imsave_ascii_raw(df, im)
            if CompressedData:
                cds = f.tell() - len(hdr)
                f.seek(cds_off)
                f.write(str(cds)) # TODO: make sure there is enough space
    elif isinstance(datafile, list):
        for i,f in enumerate(datafile):
            with openfile(f, 'wb', 'gzip' if CompressedData else None, comp_level=6) as df:
                imsave_raw(df, im[i,...]) if BinaryData else imsave_ascii_raw(df, im[i,...])
        hdr = ''.join((name+' = '+value+'\n') for name, value in alltags)
        hdr += 'ElementDataFile = '+elem_data_file+'\n'+'\n'.join(datafile_list)
        with openfile(filename, 'wb') as f: f.write(hdr)
    else:
        with openfile(datafile, 'wb', 'gzip' if CompressedData else None, comp_level=6) as df:
            imsave_raw(df, im) if BinaryData else imsave_ascii_raw(df, im)
        hdr = ''.join((name+' = '+value+'\n') for name, value in alltags)
        if CompressedData: hdr += 'CompressedDataSize = '+str(get_file_size(datafile))+'\n'
        hdr += 'ElementDataFile = '+elem_data_file
        with openfile(filename, 'wb') as f: f.write(hdr)
imsave.register('.mha', imsave_mha)
imsave.register('.mhd', imsave_mhd)
