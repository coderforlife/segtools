from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Iterable, Sequence, OrderedDict
from itertools import product
from numpy import empty, uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float64
import re, os

from ....general.enum import Enum
from ..._util import prod, itr2str, splitstr, get_list, _bool

from ...types import create_im_dtype, im_decomplexify
from ..._util import String, Unicode, re_search
from .._util import openfile, get_file_size, imread_raw, imsave_raw, imread_ascii_raw, imsave_ascii_raw
from .. import _single

##from functools import partial
##from numpy import array
##from ....general.gzip import GzipFile
##from .._stack import FileImageStack, FileImageStackHeader, Field, FixedField
##from .._util import copy_data


__all__ = ['iminfo_mha', 'iminfo_mhd', 'imread_mha', 'imread_mhd', 'imsave_mha', 'imsave_mhd']
 # TODO: 'Metafile'

dtype2met = {
    uint8  : 'MET_UCHAR',
    int8   : 'MET_CHAR',
    uint16 : 'MET_USHORT',
    int16  : 'MET_SHORT',
    uint32 : 'MET_UINT',
    int32  : 'MET_INT',
    uint64 : 'MET_ULONG_LONG',
    int64  : 'MET_LONG_LONG',
    float32 : 'MET_FLOAT',
    float64 : 'MET_DOUBLE',
}
met2dtype = { v:k for k,v in dtype2met.iteritems() }
met2dtype['MET_LONG']  = int32  # synonyms
met2dtype['MET_ULONG'] = uint32
# Note: MET_*_ARRAY is equivilient to MET_*, we standardize to ARRAY being when with multi-channel images

req_keys = ('ObjectType','NDims','DimSize','ElementType','ElementDataFile')
synonyms = {
    'ElementByteOrderMSB':'BinaryDataByteOrderMSB',
    'Position':'Offset','Origin':'Offset',
    'Orientation':'TransformMatrix','Rotation':'TransformMatrix',
    }

# need to do it this way to support ? as a name
METDistanceUnits = Enum('METDistanceUnits', {'?':0, 'um':1, 'mm':2, 'cm':3, 'UNKNOWN':0}, module=__name__, type=int)

class METModality(int, Enum):
    MET_MOD_CT      = 0
    MET_MOD_MR      = 1
    MET_MOD_NM      = 2
    MET_MOD_US      = 3
    MET_MOD_OTHER   = 4
    MET_MOD_UNKNOWN = 5

def anatomical_orientation(value, ndims):
    # string ndims chars long, each has to be [R|L] | [A|P] | [S|I] and form a distinct set, can be ? for unknown
    value = ''.join(value).strip().upper()
    if len(value) != ndims: raise ValueError('AnatomicalOrientation tag must have one letter per dimension')
    options = { 'R' : 'L', 'L' : 'R', 'A' : 'P', 'P' : 'A', 'S' : 'I', 'I' : 'S' }
    for x in value:
        if x == '?': continue
        if x not in options: raise ValueError('AnatomicalOrientation tag is not well-formed')
        del options[options[x]]
        del options[x]
    return value

# TODO: the ':' seems to be a comment in the header according to source
header_line  = re.compile(r'^[A-Za-z0-9_-]+\s*=')
file_list    = re.compile(r'^LIST(?:\s+([0-9]+)D?)?$')
file_pattern = re.compile(r'^(\S*%[#0-9 +.-]*[hlL]?[dD]\S*)(?:\s+([0-9]+)(?:\s+([0-9]+)(?:\s+([0-9]+))?)?)?$')

def str_nl_check(x):
    x = Unicode(x)
    if '\n' in x or '\r' in x: raise ValueError()
    return x

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

def read_mha_header(f):
    # Read Header
    h = OrderedDict()
    line = f.readline(256)
    while len(line) > 0 and header_line.search(line) is not None:
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

    # Check/Parse Header
    datafile, headersize = parse_mha_header(h, f)

    # Check/Parse Element Type Header Keys
    endian = _bool(h.get('BinaryDataByteOrderMSB', False))
    etype = h['ElementType']
    echns = int(h.get('ElementNumberOfChannels', 1))
    if etype.endswith('_ARRAY'): etype = etype[:-6]
    h['ElementType'] = etype + ('_ARRAY' if echns > 1 else '')
    if etype not in met2dtype or echns < 1: raise ValueError('MHA/MHD file image type not supported')
    dtype = met2dtype[etype]
    dtype = create_im_dtype(dtype, endian, echns)

    return h, dtype, datafile, headersize

def parse_mha_header(h, f):
    for k in req_keys:
        if k not in h: raise ValueError('MHA/MHD file header does not have required field \''+k+'\'')
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
        datafile = None
        if 'HeaderSize' not in h: headersize = f.tell()
    elif re_search(file_list, datafile): # we have a list of files
        file_ndims = int(re_search.match.group(1)) if re_search.match.lastindex == 1 else ndims-1
        if file_ndims == 0 or file_ndims > ndims: raise ValueError('Invalid ElementDataFile in MHA/MHD header')
        datafile = [f.readline().strip() for _ in xrange(prod(shape[file_ndims:]))]
    elif re_search(file_pattern, datafile): # we have a pattern of files
        datafile = parse_pattern(re_search.match, shape[-1])
    return datafile, headersize

def iminfo_mha(filename):
    """Equivilent to iminfo_mhd, MHD vs MHA files are determined based on the header in the file"""
    return iminfo_mhd(filename)
def iminfo_mhd(filename):
    """
    Read the header from an MHA/MHD image file and return shape (H, W) and dtype.
    """
    with openfile(filename, 'rb') as f:
        h, dtype, _, _ = read_mha_header(f)
    return tuple(reversed(splitstr(h['DimSize'], int))), dtype
_single.iminfo.register('.mha', iminfo_mha)
_single.iminfo.register('.mhd', iminfo_mhd)

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
        Many fields are simply ignored (e.g. TransformationMatrix) but they are returned
    """
    # Read/Check/Parse Header
    with openfile(filename, 'rb') as f:
        h, dtype, datafile, headersize = read_mha_header(f)
    shape = splitstr(h['DimSize'], int)
    shape.reverse()

    # Determine the shape/size of a single data file
    file_ndims = len(shape)
    if isinstance(datafile, tuple):
        # follow a pattern to find all the files to be loaded
        pattern, start, stop, step = datafile #pylint: disable=unpacking-non-sequence
        datafile = [pattern % i for i in xrange(start, stop+1, step)]
        file_ndims -= 1
    elif isinstance(datafile, list):
        # a list of files
        while prod(shape[:-file_ndims]) != len(datafile): file_ndims -= 1
    reader = get_reader(h, shape[-file_ndims:], dtype, headersize)

    # Read the data in
    if isinstance(datafile, list):
        # a list of files
        directory = os.path.dirname(os.path.realpath(filename))
        im = empty(shape, dtype)
        ind = [slice(None)] * len(im.shape)
        inds = product(*[xrange(x) for x in im.shape[:-file_ndims]])
        for f,i in zip(datafile, inds):
            ind[:-file_ndims] = i
            im[ind] = reader(os.path.realpath(os.path.join(directory, f)))
        return h, im
    else:
        # datafile is just a file
        return h, reader(datafile or filename)

def get_reader(h, shape, dtype, headersize):
    imread = imread_raw if _bool(h.get('BinaryData', True)) else imread_ascii_raw
    compressed = _bool(h.get('CompressedData', False))
    compression = 'auto' if compressed else None
    datasize = h.get('CompressedDataSize', None) if compressed else prod(shape)*dtype.itemsize
    def _reader(datafile):
        return read_data_file(datafile, headersize, datasize, imread, compression, dtype, shape)
    return _reader

_single.imread.register('.mha', lambda filename: imread_mha(filename)[1])
_single.imread.register('.mhd', lambda filename: imread_mhd(filename)[1])

####### Saving #################################################################

def imsave_mha(filename, im, CompressedData=False, BinaryData=True, **tags):
    """Save an image as an MHA image (data embeded in the metadata file). See imsave_mhd for more information."""
    imsave_mhd(filename, im, 'LOCAL', CompressedData, BinaryData, **tags)
def imsave_mhd(filename, im, datafile=None, CompressedData=False, BinaryData=True, ElementNumberOfChannels=None, **tags):
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
        Cannot output data listed datafiles that don't contain exactly 1 less dimension than the overall number of dimensions
        Cannot save non-image objects
        Cannot create a datafile with a non-MHA/MHD header (HeaderSize) (partially allowed in the Metafile ImageStack)
    Attempting to force these features through tags will raise errors.
    """

    from sys import byteorder

    # Figure out the data type
    was_complex = im.dtype.kind == 'c'
    im = im_decomplexify(im) # we support complex numbers only by breaking them into a 2-channel image
    if im.dtype.type not in dtype2met: raise ValueError('Format of image is not supported')

    # Figure out the dimensions
    if im.ndims > 2: im = im.squeeze(2)
    echans, shape = 1, list(im.shape)
    if ElementNumberOfChannels is not None:
        echans = im.shape[-1] if ElementNumberOfChannels == '*' else int(ElementNumberOfChannels)
    elif was_complex:
        echans = 2
    elif len(shape) in (3,4) and shape[-1] in (3,4) and im.dtype.type == uint8:
        echans = shape[-1]
    if echans != 1:
        if echans != im.shape[-1]: raise ValueError('If ElementNumberOfChannels is provided it must be None, *, 1, or the number of elements in the last dimension of the image')
        del shape[-1]
    shape.reverse()
    ndims = len(shape)

    # Figure out how and where we are saving
    BinaryData = _bool(BinaryData)
    CompressedData = _bool(CompressedData)
    filename = os.path.realpath(filename.strip())
    elem_data_file, datafile = get_datafile(filename, datafile, CompressedData, BinaryData, im.shape[0])

    # Setup tags that we compute from the image or other given options
    alltags = OrderedDict((
        ('ObjectType', 'Image'),
        ('NDims', str(ndims)),
        ('DimSize', itr2str(shape)),
        ('BinaryData', str(BinaryData)),
        ('BinaryDataByteOrderMSB', str(im.dtype.byteorder == '>' or im.dtype.byteorder == '=' and byteorder != 'little')),
        ('CompressedData', str(CompressedData)),
        ))
    if echans != 1:
        alltags['ElementType'] = dtype2met[im.dtype]+'_ARRAY'
        alltags['ElementNumberOfChannels'] = str(echans)
    else:
        alltags['ElementType'] = dtype2met[im.dtype]

    # Process all other tags
    alltags['ElementDataFile'] = elem_data_file.split('\n',1)[0]
    process_tags(tags, alltags, echans, ndims, im.dtype)
    alltags.update(tags)
    del alltags['ElementDataFile']

    # Save data
    imsave = imsave_raw if BinaryData else imsave_ascii_raw
    save_data(im, filename, datafile, elem_data_file, alltags, CompressedData, imsave)

def get_datafile(filename, datafile, comp, binary, shape0):
    if not binary and comp: raise ValueError('Cannot compress non-binary data')
    directory = os.path.dirname(filename)
    if datafile is None:
        ext = ('.gz' if comp else '.bin' if binary else '.txt')
        datafile = os.path.splitext(filename)[0] + ext
        if os.path.exists(datafile):
            from tempfile import mktemp
            datafile = mktemp(ext, os.path.basename(filename), directory)
        return os.path.relpath(datafile, directory), datafile
    elif isinstance(datafile, String):
        datafile = datafile.strip()
        if datafile in ('LOCAL', 'Local', 'local'):
            return 'LOCAL', 'LOCAL'
        elif re_search(file_pattern, datafile):
            p = parse_pattern(re_search.match, shape0)
            return itr2str(p), [os.path.realpath(os.path.join(directory,p[0]%i)) for i in xrange(p[1],p[2]+1,p[3])]
        elif file_list.search(datafile) is not None:
            raise ValueError('Datafile "'+datafile+'" is a reserved name, provide a list of files instead')
        else:
            return os.path.relpath(datafile, directory), datafile
    elif isinstance(datafile, Sequence) and re_search(file_pattern, itr2str(datafile)):
        p = parse_pattern(re_search.match, shape0)
        return itr2str(p), [os.path.realpath(os.path.join(directory,p[0]%i)) for i in xrange(p[1],p[2]+1,p[3])]
    elif isinstance(datafile, Iterable):
        datafile = [os.path.realpath(os.path.join(directory, x)) for x in datafile]
        if len(datafile) != shape0: raise ValueError('When using list datafiles there must be one file for each entry in the highest dimension')
        return 'LIST\n'+'\n'.join(os.path.relpath(x, directory) for x in datafile), datafile

def process_tags(tags, alltags, echans, ndims, dtype): #pylint: disable=too-many-branches
    # These tags are not allowed in the extra tags as they are either implied from the image data or have required values
    not_allowed = req_keys + ('ObjectSubType', 'HeaderSize',  # used to specify how data is stored, int >= -1 with -1 as special
                   'CompressedData', 'CompressedDataSize',
                   'BinaryData', 'BinaryDataByteOrderMSB', 'ElementNumberOfChannels', 'ElementByteOrderMSB')
    strings   = ('Name', 'Comment', 'TransformType', 'AcquisitionDate')
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
        if v is None: raise ValueError('Tags cannot have a value of None')
        if k in not_allowed:
            if alltags.get(k, None) == v: continue
            raise ValueError('Setting the tag "' + k + '" is not allowed')
        elif k in strings:      v = str_nl_check(v)
        elif k in enums:        v = enums[k](v).name
        elif k in ints:         v = str(int(v))
        elif k in floats:       v = str(float(v))
        elif k in sames:        v = itr2str(get_list(v, echans, dtype.type)) if echans != 1 else str(dtype.type(v))
        elif k in n_floats:     v = itr2str(get_list(v, ndims, float))
        elif k in n2_floats:    v = itr2str(get_list(v, (ndims, ndims), float))
        elif k == 'Color':      v = itr2str(get_list(v, 4, float))
        elif k == 'SequenceID': v = itr2str(get_list(v, 4, int))
        elif k == 'AnatomicalOrientation': v = anatomical_orientation(v, ndims)
        elif re.search('[^a-z0-9_-]', k) is not None:
            raise ValueError('Invalid key name "' + k + '"')
        else: continue
        tags[k] = v
    for name_set in only_one:
        if len([1 for n in name_set if n in tags]) > 1: raise ValueError('There can only be one tag from the set '+(', '.join(only_one)))

def save_data(im, filename, datafile, elem_data_file, alltags, comp, imsave):
    hdr = '\n'.join((name+' = '+value) for name, value in alltags.iteritems())+'\n'
    if datafile == 'LOCAL':
        hdr = hdr.encode('utf8')
        if comp:
            hdr += b'CompressedDataSize = '
            cds_off = len(hdr)
            hdr += b'0               \n' # has room for exobytes
        hdr += b'ElementDataFile = LOCAL\n'
        with openfile(filename, 'wb') as f:
            f.write(hdr)
            with openfile(f, 'wb', 'zlib' if comp else None, comp_level=6) as df:
                imsave(df, im)
            if comp:
                cds = f.tell() - len(hdr)
                f.seek(cds_off)
                f.write(str(cds).encode('utf8')) # TODO: make sure there is enough space
    elif isinstance(datafile, list):
        for i,f in enumerate(datafile):
            with openfile(f, 'wb', 'gzip' if comp else None, comp_level=6) as df:
                imsave(df, im[i,...])
        hdr += 'ElementDataFile = '+elem_data_file
        with openfile(filename, 'wb') as f: f.write(hdr.encode('utf8'))
    else:
        with openfile(datafile, 'wb', 'gzip' if comp else None, comp_level=6) as df:
            imsave(df, im)
        if comp: hdr += 'CompressedDataSize = '+str(get_file_size(datafile))+'\n'
        hdr += 'ElementDataFile = '+elem_data_file
        with openfile(filename, 'wb') as f: f.write(hdr.encode('utf8'))

_single.imsave.register('.mha', imsave_mha)
_single.imsave.register('.mhd', imsave_mhd)
