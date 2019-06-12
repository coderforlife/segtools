# See http://www.itk.org/Wiki/MetaIO/Documentation for details
# Offical source:
#   https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/ThirdParty/MetaIO/src/MetaIO/src
#   see metaImage.cxx, metaObject.cxx, and metaUtils.cxx


# TODO: the 3D version is severely limited and was quickly hacked together
# It isn't very memory efficient and is designed to make sure it works with imstack but not the
# more general interface. The header is completely lacking and only supports read-only or
# write-only, but not but. 


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from enum import IntEnum
from collections import OrderedDict, defaultdict
from itertools import product, izip
import os.path

from numpy import empty

from .._single import FileImageSource
from .._stack import HomogeneousFileImageStack, FileImageSlice, FileImageStackHeader
from ...types import create_im_dtype, get_dtype_endian, get_im_dtype_and_nchan, im_decomplexify, im_decomplexify_dtype
from ....general import String, Unicode, Byte, sys_endian, prod, delayed
from ....general.io import openfile, get_file_size, array_read, array_save, array_read_ascii, array_save_ascii

__sys_is_big_endian = sys_endian == '>'

__all__ = ['imread_mha', 'imread_mhd', 'imsave_mha', 'imsave_mhd', 'MetaImage']

# need to do it this way to support ? as a name
METDistanceUnits = Enum('METDistanceUnits',
                        {'?':0, 'um':1, 'mm':2, 'cm':3, 'UNKNOWN':0},
                        module=__name__, typ=int)

class METModality(IntEnum):
    MET_MOD_CT      = 0
    MET_MOD_MR      = 1
    MET_MOD_NM      = 2
    MET_MOD_US      = 3
    MET_MOD_OTHER   = 4
    MET_MOD_UNKNOWN = 5

def _skip_to_val(ch): return ch in ' \t\v\f=:'

def __read_ascii_char(f): return f.read(1).decode('ascii')
def __read_utf8_char(f):
    s = f.read(1)
    if len(s) == 0: return ''
    b = Byte(s)
    if b & 0x10000000:
        if   b & 0x11100000 == 0x11000000: s += f.read(1)
        elif b & 0x11110000 == 0x11100000: s += f.read(2)
        elif b & 0x11111000 == 0x11110000: s += f.read(3)
    return s.decode('utf-8')

def _read_until(f, terminate, maxlen=None, skip_leading=None, read=__read_utf8_char):
    """
    Reads characters from the file-like object until terminate returns True.
    If maxlen is given, it stops once that number of characters has been read as well.
    
    If skip_leading is given then initial characters for which that returns True are skipped and
    not checked against terminate, do not contribute towards reaching maxlen, and are not
    returned.

    Returns the text read and last character read.
    The last character found maybe the character for which terminate returned True or None if
    maxlen was reached or an empty string if the EOF was reached.

    The file-like object should be a binary reader.
    """
    # alternatively, use data = io.StringIO(), len(data) -> data.tell(), and data -> data.getvalue()
    data = ''
    ch = read(f)
    if skip_leading:
        while ch != '' and skip_leading(ch): ch = read(f)
    while ch != '' and not terminate(ch):
        if len(data) == maxlen: return data, None
        data += ch
        ch = read(f)
    return data, ch

def __get_header_fields(): #pylint: disable=too-many-locals
    from abc import ABCMeta, abstractmethod
    from collections import Iterable
    from numbers import Integral
    
    from numpy import dtype, array, eye, fromiter
    from numpy import uint8, int8, uint16, int16, int32, uint32, uint64, int64, float32, float64
    
    from ....general.utils import itr2str, get_list, _bool
    
    class MHDField(object):
        __metaclass__ = ABCMeta
        def __init__(self, name): self.name = name
        def _read_val(self, f, cast, read_to_newline=True):
            text, ch = _read_until(f, Unicode.isspace, 500, _skip_to_val)
            if ch is None: raise ValueError('Cannot read %s from MHA/MHD file' % self.name)
            if read_to_newline and ch not in ('', '\n', '\r'): _read_until(f, lambda ch: ch in '\r\n')
            try: return cast(text)
            except (TypeError, ValueError): raise ValueError('Cannot convert value for %s' % self.name)
        def _get_ndims(self, fields):
            ndims = fields.get('NDims')
            if ndims is None: raise ValueError('Must have NDims field before %s in the MHA/MHD header' % self.name)
            return ndims
        @abstractmethod
        def read(self, f, fields):
            """Reads the value for this field from the given file-like object."""
            pass
        @abstractmethod
        def check(self, x, fields):
            """Checks (and possibly converts) a value to the appropiate type for this field"""
            pass
        def write(self, x, fields): #pylint: disable=no-self-use,unused-argument
            """
            Gets the string representation of the value for this field, which has already been passed
            through check, to be written to a file. This can return None if the entire field should not
            be written.
            """
            return str(x)

    class StringField(MHDField):
        def read(self, f, fields): return _read_until(f, lambda ch: ch in '\r\n', None, _skip_to_val)[0].rstrip()
        def check(self, x, fields):
            x = Unicode(x)
            if '\n' in x or '\r' in x: raise ValueError('Strings field values cannot contain new lines')
            return x
        def write(self, x, fields): return x if len(x) > 0 else None
        
    class BoolField(StringField):
        def read(self, f, fields): return _bool(super(BoolField, self).read(f, fields))
        def check(self, x, fields): return _bool(x, True)
        def write(self, x, fields): return str(bool(x))

    class CompressedDataField(BoolField):
        def __init__(self): super(CompressedDataField, self).__init__('CompressedData')
        def check(self, x, fields):
            if isinstance(x, Integral) and not isinstance(x, bool) or isinstance(x, String) and x.isdigit():
                x = int(x)
                if x < 0 or x > 9: raise ValueError('CompressedData must be True, False, or a number from 0-9')
                return False if x == 0 else x
            return super(CompressedDataField, self).check(x, fields)
        
    class EnumField(StringField):
        def __init__(self, name, enum, unknown=None):
            super(EnumField, self).__init__(name)
            self.enum = enum
            self.unknown = unknown
        def read(self, f, fields): return self.enum(super(EnumField, self).read(f, fields))
        def check(self, x, fields): return self.enum(x)
        def write(self, x, fields): return x.name if x != self.unknown else None

    class StringCheckField(StringField):
        def read(self, f, fields): return self.check(super(StringCheckField, self).read(f, fields), fields)
        
    class ObjectTypeField(StringCheckField):
        def __init__(self): super(ObjectTypeField, self).__init__('ObjectType')
        def check(self, x, fields):
            if x != 'Image': raise ValueError('Non-image MHA/MHD files are not supported')
            return x
        
    class ObjectSubTypeField(StringCheckField):
        def __init__(self): super(ObjectSubTypeField, self).__init__('ObjectSubType')
        def check(self, x, fields): raise ValueError('ObjectSubType field not allowed in MHA/MHD Image files')

    class AnatomicalOrientationField(StringCheckField):
        def __init__(self): super(AnatomicalOrientationField, self).__init__('AnatomicalOrientation')
        def check(self, x, fields):
            ndims = self._get_ndims(fields)
            # string ndims chars long, each has to be [R|L] | [A|P] | [S|I] and form a distinct set, can be ? for unknown
            x = ''.join(x).strip().upper()
            if len(x) != ndims: raise ValueError('AnatomicalOrientation tag must have one letter per dimension')
            options = { 'R' : 'L', 'L' : 'R', 'A' : 'P', 'P' : 'A', 'S' : 'I', 'I' : 'S' }
            for y in x:
                if y == '?': continue
                if y not in options: raise ValueError('AnatomicalOrientation tag is not well-formed')
                del options[options[y]]
                del options[y]
            return x
        def write(self, x, fields): return x if any(c != '?' for c in x) else None # technically: return x[0] != '?'

    class ElementTypeField(StringCheckField):
        # Note: MET_*_ARRAY is equivilient to MET_*
        __dtype2met = {
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
            # Non-image types that are defined in MetaIO (and some are used for headers)
            #??? : 'MET_NONE',
            #??? : 'MET_ASCII_CHAR',
            #??? : 'MET_STRING',
            #??? : 'MET_OTHER',
            #??? : 'MET_FLOAT_MATRIX',
        }
        __met2dtype = { v:k for k,v in __dtype2met.iteritems() }
        __met2dtype['MET_LONG']  = int32  # synonyms
        __met2dtype['MET_ULONG'] = uint32
        def __init__(self): super(ElementTypeField, self).__init__('ElementType')
        def check(self, x, fields):
            if isinstance(x, String):
                if x.endswith('_ARRAY'): x = x[:-6]
                if x not in ElementTypeField.__met2dtype: raise ValueError('MHA/MHD file image type not supported')
                return ElementTypeField.__met2dtype[x]
            if isinstance(x, dtype): x = x.base.type
            if x not in ElementTypeField.__dtype2met: raise ValueError('MHA/MHD file image type not supported')
            return x
        def write(self, x, fields): return ElementTypeField.__dtype2met[x]

    class ElementDataFileField(StringField):
        @staticmethod
        def __parse_pattern(pattern, n):
            parts = pattern.split()
            num = []
            while len(num) < 3 and len(parts) > 1 and parts[-1].isdigit():
                part = parts.pop()
                num = int(part) + num
                pattern = pattern[:-len(part)].strip()
            return ElementDataFileField.__check_pattern([pattern] + num, n)
        @staticmethod
        def __check_pattern(pattern, n):
            pattern, num = pattern[0], pattern[1:]
            if '%' not in pattern or len(num) > 3 or \
               any(not isinstance(i, Integral) for i in num): raise ValueError()
            if   len(num) == 0: start, stop, step = 1, n, 1
            elif len(num) == 1: start, stop, step = num[0], num[0]+n-1, 1
            elif len(num) == 2: start, stop, step = num[0], num[1], (num[1]-num[0])//n
            elif len(num) == 3: start, stop, step = num
            if (stop-start)//step+1 != n: stop = start + (n-1)*step # TODO: > n?
            return (pattern, start, stop, step)
        @staticmethod
        def __get_file_ndims(num_files, shape):
            for file_ndims in xrange(len(shape)-1, -1, -1):
                n = prod(shape[:-file_ndims])
                if n == num_files: return file_ndims
                if n >  num_files: break
            raise ValueError('Invalid number of files given for saving data to')
        def __init__(self): super(ElementDataFileField, self).__init__('ElementDataFile')
        def read(self, f, fields):
            ndims = self._get_ndims(fields)
            shape = fields.get('DimSize')
            if shape is None: raise ValueError('Must have DimSize field before ElementDataFile in the MHA/MHD header')
            datafile = super(ElementDataFileField, self).read(f, ndims)
            if datafile in ('LOCAL', 'Local', 'local'): return None
            elif datafile.startswith('LIST'): # we have a list of files
                parts = datafile.split()
                d, i = parts[1] if len(parts) > 1 else '', 0
                while i < len(d) and d[:i+1].isdigit(): i += 1
                file_ndims = int(d[:i]) if i > 0 else ndims-1
                if file_ndims <= 0 or file_ndims > ndims: file_ndims = ndims-1
                return list(f.readline().strip() for _ in xrange(prod(shape[:-file_ndims])))
            elif '%' in datafile: # we have a pattern of files
                return ElementDataFileField.__parse_pattern(datafile, shape[0])
            return datafile
        def check(self, x, fields):
            shape = fields['DimSize']
            if x in (None, 'LOCAL', 'Local', 'local'):
                if 'HeaderSize' in fields: raise ValueError('HeaderSize field cannot be used with MHA files')
                return None
            if not isinstance(x, Iterable): raise ValueError('Invalid value for ElementDataFile field in MHA/MHD header')
            if isinstance(x, String):
                if x.startswith('LIST'): raise ValueError('Invalid value for ElementDataFile field in MHA/MHD header (cannot start with LIST)')
                if '%' in x: return ElementDataFileField.__parse_pattern(x, shape[0])
                return x
            x = list(x)
            if len(x) > 1 and len(x) <= 4 and \
               '%' in x[0] and all(isinstance(i, Integral) for i in x[1:]):
                return ElementDataFileField.__check_pattern(x, shape[0]) 
            ElementDataFileField.__get_file_ndims(len(x), shape)
            return [str(fn) for fn in x]
        def write(self, x, fields):
            if x is None: return 'LOCAL'
            if isinstance(x, String): return x
            if isinstance(x, tuple): return ('%s %d' % x[:2]) if x[3] == 1 else '%s %d %d %d' % x
            if isinstance(x, list):
                return ('LIST %dD\n'%ElementDataFileField.__get_file_ndims(len(x), fields['DimSize'])) + '\n'.join(x)
            raise ValueError()
        
    class ScalarField(MHDField):
        def __init__(self, name, cast, at_least=None, dont_write=None):
            super(ScalarField, self).__init__(name)
            self.cast = cast
            self.at_least = at_least
            self.dont_write = dont_write
        def read(self, f, fields):
            x = self._read_val(f, self.cast)
            if self.at_least is not None and x < self.at_least:
                raise ValueError('The value for %s must be at least %s' % (self.name, self.at_least))
            return x
        def check(self, x, fields):
            try: x = self.cast(x)
            except (TypeError, ValueError): raise ValueError('Cannot convert the value for %s' % self.name)
            if self.at_least is not None and x < self.at_least:
                raise ValueError('The value for %s must be at least %s' % (self.name, self.at_least))
            return x
        def write(self, x, fields):
            return None if x == self.dont_write or callable(self.dont_write) and self.dont_write(x, fields) else str(x)
        
    class ArrayField(MHDField):
        def __init__(self, name, cast, length=None, dont_write=None):
            super(ArrayField, self).__init__(name)
            self.cast = cast
            self.length = length
            self.dont_write = dont_write
        def _get_n(self, fields):
            return self.length or self._get_ndims(fields)
        def _read(self, f, n): return (self._read_val(f, self.cast, i == n-1) for i in xrange(n))
        def read(self, f, fields): return tuple(self._read(f, self._get_n(fields)))
        def check(self, x, fields): return get_list(x, self._get_n(fields), self.cast, None, tuple)
        def write(self, x, fields): return itr2str(x) if any(i != self.dont_write for i in x) else None
        
    class TransformMatrixField(ArrayField):
        def __init__(self):
            super(TransformMatrixField, self).__init__('TransformMatrix', float32)
        def read(self, f, fields):
            n = self._get_n(fields)
            return fromiter(self._read(f, n*n), self.cast, n*n).reshape((n, n))
        def check(self, x, fields):
            n = self._get_n(fields)
            return array(get_list(x, (n,n), self.cast))
        def write(self, x, fields):
            n = self._get_n(fields)
            return itr2str((eye(n, float32) if (x == 0).all() else x).ravel())

    class HeaderSizeField(ScalarField):
        def __init__(self):
            super(HeaderSizeField, self).__init__('HeaderSize', int32, at_least=-1, dont_write=0)
        def check(self, x, fields):
            if isinstance(x, Integral) or isinstance(x, String) and (x.isdigit() or x == '-1'):
                return super(HeaderSizeField, self).check(x, fields)
            return bytes(x)
        def write(self, x, fields):
            return super(HeaderSizeField, self).write(len(x) if isinstance(x, bytes) else x, fields)

    class MHAHeaderFields(defaultdict):
        def __missing__(self, key): return StringField(key)
        def __init__(self, d): super(MHAHeaderFields, self).__init__(None, d)

    def id_filter(x, _): return x<0
    def slope_filter(x, fields): return x == 1 and fields.get('ElementToIntensityFunctionOffset', 0) == 0
    def offset_filter(x, fields): return x == 0 and fields.get('ElementToIntensityFunctionSlope', 1) == 1
    
    return MHAHeaderFields({
        # does not include Comment, AcquisitionDate, or Name which are string values which is the default
        'ObjectType':       ObjectTypeField(),
        'ObjectSubType':    ObjectSubTypeField(),
        'NDims':            ScalarField('NDims', int32, at_least=1),
        'ID':               ScalarField('ID',       int32, dont_write=id_filter),
        'ParentID':         ScalarField('ParentID', int32, dont_write=id_filter),
        'BinaryData':       BoolField('BinaryData'),
        'CompressedData':   CompressedDataField(),
        'CompressedDataSize':       ScalarField('CompressedDataSize', uint32),
        'BinaryDataByteOrderMSB':   BoolField('BinaryDataByteOrderMSB'),
        'Color':            ArrayField('Color', float32, 4, dont_write=1),
        'Offset':           ArrayField('Offset', float32),
        'TransformMatrix':  TransformMatrixField(),
        'CenterOfRotation': ArrayField('CenterOfRotation', float32),
        'DistanceUnits':    EnumField('DistanceUnits', METDistanceUnits, METDistanceUnits.UNKNOWN), # pylint: disable=no-member
        'AnatomicalOrientation':    AnatomicalOrientationField(),
        'ElementSpacing':   ArrayField('ElementSpacing', float32),
        # Image specific
        'DimSize':          ArrayField('DimSize', int32),
        'HeaderSize':       HeaderSizeField(),
        'Modality':         EnumField('Modality', METModality, METModality.MET_MOD_UNKNOWN),
        'ImagePosition':    ArrayField('ImagePosition', float32),
        'SequenceID':       ArrayField('SequenceID', int32, dont_write=0), # sometimes fixed at length 4, usually length ndims though
        'ElementMin':       ScalarField('ElementMin', float32), # these should really be based on ElementType, but this is how the actual MetaIO reads them
        'ElementMax':       ScalarField('ElementMax', float32),
        'ElementNumberOfChannels':  ScalarField('ElementNumberOfChannels', int32, at_least=1, dont_write=1),
        'ElementSize':      ArrayField('ElementSize', float32),
        'ElementNBits':     ScalarField('ElementNBits', int32, dont_write=lambda x,fields:True), # not used, never written
        'ElementToIntensityFunctionSlope':  ScalarField('ElementToIntensityFunctionSlope',  float32, dont_write=slope_filter),
        'ElementToIntensityFunctionOffset': ScalarField('ElementToIntensityFunctionOffset', float32, dont_write=offset_filter),
        'ElementType':      ElementTypeField(),
        'ElementDataFile':  ElementDataFileField(),
    })
__header_fields = delayed(__get_header_fields, defaultdict)
__req_keys = delayed(lambda:{'ObjectType','NDims','DimSize','ElementType','ElementDataFile'}, set)
__synonyms = delayed(lambda:{
    'ElementByteOrderMSB':'BinaryDataByteOrderMSB',
    'Position':'Offset','Origin':'Offset',
    'Orientation':'TransformMatrix','Rotation':'TransformMatrix',
    },dict)
def __get_defaults():
    from numpy import eye, int32, float32
    return {
        'ObjectType':       lambda ndims:'Image',
        'Offset':           lambda ndims:(int32(0),)*ndims,
        'TransformMatrix':  lambda ndims:eye(ndims, dtype=float32),
        'CenterOfRotation': lambda ndims:(int32(0),)*ndims,
        'ElementSpacing':   lambda ndims:(int32(1),)*ndims,
    }
__defaults = delayed(__get_defaults, dict)
__order_first = delayed(lambda:(
    'Comment', 'ObjectType', 'NDims', 'Name', 'ID', 'ParentID', 'AcquisitionDate', 'Color',
    'BinaryData', 'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
    'TransformationMatrix', 'Offset', 'CenterOfRotation', 'DistanceUnits', 'AnatomicalOrientation', 'ElementSpacing',
    ),tuple)
__order_last = delayed(lambda:(
    'DimSize', 'HeaderSize', 'Modality', 'SequenceID', 'ElementMin', 'ElementMax',
    'ElementNumberOfChannels', 'ElementSize', 'ElementToIntensityFunctionSlope', 'ElementToIntensityFunctionOffset',
    'ElementType', 'ElementDataFile',
    ),tuple)
__order = delayed(lambda:frozenset(__order_first+__order_last), frozenset)

def __check_binary(fields, default_endian=__sys_is_big_endian):
    if fields.setdefault('BinaryData', False):
        fields.setdefault('BinaryDataByteOrderMSB', default_endian)
        if fields.setdefault('CompressedData', False):
            if fields.get('HeaderSize', None) == -1: # TODO: ? and fields.get('CompressedDataSize', 0) == 0:
                #raise ValueError('MHA/MHD HeaderSize cannot be -1 when CompressedData is True and CompressedDataSize is 0 or not given')
                raise ValueError('MHA/MHD HeaderSize cannot be -1 when CompressedData is True')
            fields.setdefault('CompressedDataSize', 0)
        elif 'CompressedDataSize' in fields:
            raise ValueError('MHA/MHD CompressedDataSize cannot be given when CompressedData is False')
    elif fields.get('HeaderSize', None) == -1:
        raise ValueError('MHA/MHD HeaderSize cannot be -1 when BinaryData is False')
    elif 'BinaryDataByteOrderMSB' in fields:
        raise ValueError('MHA/MHD BinaryDataByteOrderMSB cannot be given when BinaryData is False')
    elif 'CompressedDataSize' in fields:
        raise ValueError('MHA/MHD CompressedDataSize cannot be given when CompressedData is False')
    elif fields.pop('CompressedData', False):
        raise ValueError('MHA/MHD CompressedData cannot be True when BinaryData is False')
def _get_dtype(fields):
    return create_im_dtype(fields['ElementType'],
                           fields.get('BinaryDataByteOrderMSB', __sys_is_big_endian),
                           fields.get('ElementNumberOfChannels', 1))

# TODO: use ElementToIntensityFunctionSlope and ElementToIntensityFunctionOffset?


####### Reading ################################################################

def read_mha_header(f):
    # Read Header
    fields = OrderedDict()
    k, sym = _read_until(f, (lambda ch: ch in '=:'), 500, Unicode.isspace)
    while sym not in (None, ''):
        k = k.split('\n')[0].rstrip()
        k = __synonyms.get(k, k)
        fields[k] = __header_fields[k].read(f, fields)
        if k == 'ElementDataFile': break
        k, sym = _read_until(f, (lambda ch: ch in '=:'), 500, Unicode.isspace)
    else: raise ValueError('MHA/MHD file header does not have required field \'ElementDataFile\'')

    # Check/Parse Header
    for k in __req_keys:
        if k not in fields: raise ValueError('MHA/MHD file header does not have required field \''+k+'\'')

    __check_binary(fields)
    headersize = fields.get('HeaderSize', None)
    if fields['ElementDataFile'] is None and headersize is None: headersize = f.tell()
    return fields, headersize

def imread_mha(filename):
    """Equivilent to imread_mhd, MHD vs MHA files are determined based on the header in the file"""
    return imread_mhd(filename)
def imread_mhd(filename):
    """
    Read an MHA/MHD image file. Returns both the header dictionary and the image data.

    Supported image data formats:
      * Unsigned and signed integral data types up to 8 bytes each
      * float and double
      * single and multi-channel
        
    Unsupported features:
      * Non-image files
      * HeaderSize of -1 when data is ASCII or compressed (note: in these situations, the real MetaIO does weird things)
      * Many fields are simply ignored (e.g. TransformationMatrix) but they are parsed and returned
    """
    with openfile(filename, 'rb') as f:
        fields, headersize = read_mha_header(f)
    return fields, read_mha_data(filename, fields, headersize)

def read_mha_data(filename, fields, headersize):
    directory = os.path.dirname(os.path.abspath(filename))

    # Determine the shape/size of a single data file
    shape = list(reversed(fields['DimSize']))
    datafile = fields['ElementDataFile']
    file_ndims = len(shape)
    if isinstance(datafile, tuple):
        # follow a pattern to find all the files to be loaded
        pattern, start, stop, step = datafile #pylint: disable=unpacking-non-sequence
        datafile = [pattern % i for i in xrange(start, stop+1, step)]
        file_ndims -= 1
    elif isinstance(datafile, list):
        # a list of files
        while prod(shape[:-file_ndims]) != len(datafile): file_ndims -= 1
    else:
        datafile = filename if datafile is None else os.path.join(directory, datafile)
    dt = _get_dtype(fields)
    read = __get_mha_reader(fields, shape[-file_ndims:], dt, headersize)

    # Read the data in
    if not isinstance(datafile, list): return read(datafile) # datafile is just a file
    
    # a list of files
    im = empty(shape, dt)
    ind = [slice(None)] * len(im.shape)
    inds = product(*[xrange(x) for x in im.shape[:-file_ndims]])
    for f,i in izip(datafile, inds):
        ind[:-file_ndims] = i
        im[ind] = read(os.path.abspath(os.path.join(directory, f)))
    return im

def __get_mha_reader(fields, shape, dt, headersize):
    binary = fields['BinaryData']
    imread = array_read if binary else array_read_ascii
    comp = 'auto' if binary and fields['CompressedData'] else None
    if headersize == -1: headersize = -prod(shape)*dt.itemsize
    def read(datafile):
        off = headersize
        if off is not None and off < 0:
            off += get_file_size(datafile)
            if off < 0: raise ValueError('MHA/MHD file data file is not large enough for the pixel data')
        with openfile(datafile, 'rb', comp, off=off) as f:
            return imread(f, shape, dt, 'C')
    return read


####### Saving #################################################################

def imsave_mha(filename, im, **fields):
    """
    Save an image as an MHA image (data embeded in the metadata file).

    This changes the fields, setting ElementDataFile to None and removing HeaderSize.

    See imsave_mhd for more information.
    """
    fields['ElementDataFile'] = None
    fields.pop('HeaderSize', None)
    imsave_mhd(filename, im, **fields)
def imsave_mhd(filename, im, **fields):
    """
    Save an image as an MHD image.

    Setting some header fields causes special behaviors during saving:
      * BinaryData - if set to True will cause data to be saved as binary instead of ASCII; this
        will usually save faster and be smaller files but be not directly human-readable
      * BinaryDataByteOrderMSB - can only be given if BinaryData is True and setting this will
        override the image's data type endianness while saving
      * CompressedData - if set to True, BinaryData must also be set to True, and data will be
        compressed, resulting in smaller files but possibly take longer to save; besides setting
        this to True or False it can be set to a number from 0 to 9 to specify a compression level
        (the default is 6)
      * HeaderSize - can only be given if ElementDataFile is non-local; it specifies the offset into
        external files where the image data will be placed such that the first HeaderSize bytes are
        skipped before writing the image data; if set to -1 then BinaryData must be True and
        CompressedData must be False and the data is saved so it is at the very end of the file; if
        not given the destination file will be erased before the data is written; another option for
        HeaderSize is it can be a bytes/bytearray in which case that data is used as the header data
        in any saved file
      * ElementMin/ElementMax - normally these are just sent through as-is, but if given as "True"
        instead of an actual value then the actual min and max will be calculated and saved
      * ElementNumberOfChannels - determines the number of channels in the image; if not given it is
        assumed to be 1 unless the data is complex, then it is 2; if it is given it must be the
        default value or the size of the last dimension in the image, essentially saying the last
        dimension should be treated as channels
      * DimSize - the dimensions of the image, in reverse order from im.shape; if not given it
        defaults to reversed(im.shape) or reversed(im.shape[:-1]) depending on the value of
        ElementNumberOfChannels; if given, then im must be reshape-able to it (not including the
        channels)
      * ElementDataFile - changes where the data is saved to, see below

    If ElementDataFile is not given it will be automatically generated. If it is None, 'LOCAL',
    'Local', or 'local' then the data will be embedded in the main file (MHA-mode). If it is a
    string that contains % it will be treated as a pattern (% must be part of a printf-style
    statement) that is followed by optional start, stop, and step numbers. any other string is
    assumed to be a filename (relative to the saved file). The last option is to use an iterable.
    If the iterable has 2-4 values with the first one containing % and the others being numbers, it
    is parsed as a pattern. Other it is taken as a list of file names. In this case the list must
    be as long as the product of the first dimensions of the image (and never including channels).
    The datafile cannot start with 'LIST'.

    Some header fields are allowed only if they match the image data (allowed so that one can read
    and save the data directly keeping all fields), this includes:
      * ObjectType - must be 'Image'
      * NDims - must be len(DimSize)
      * ElementType - must match the image data type
      * CompressedDataSize - can only be given if CompressedData is True, its value is ignored

    Some header fields are forbidden because they are incompatible with saving images:
      * ObjectSubType

    All other header fields that are known are converted to the appropiate type before saving. All
    unknown header fields are converted to strings and saved. Fields names cannot contain : or =.
    Field values cannot contain new lines.

    Currently many features of MHA/MHD files are not supported, including:
      * Cannot save non-image objects
    """
    fields = parse_mha_fields(im.dtype, im.shape, fields)
    save_mha_data(im, filename, fields)

def parse_mha_fields(dt, shape, fields):
    def get(n): return __header_fields[n].check(fields[n], fields)
    
    # Process field name (synonyms and illegal names)
    fields = { __synonyms.get(k, k.strip()):v for k,v in fields.iteritems() }
    if any(('\n:=' in k) for k in fields): raise ValueError('MHA/MHD headers do not support field names with : or = in them')

    # Figure out the data type
    was_complex = dt.kind == 'c'
    dt = im_decomplexify_dtype(dt) # we support complex numbers only by breaking them into a 2-channel image
    etype = __header_fields['ElementType'].check(dt, fields)
    if 'ElementType' in fields and get('ElementType') != etype: raise ValueError('ElementType field must match image type')
    fields['ElementType'] = etype
    
    # Figure out the dimensions
    echans, shape = get_im_dtype_and_nchan(dt)[1], list(shape)
    if 'ElementNumberOfChannels' in fields:
        ElemNumOfChans = get('ElementNumberOfChannels')
        if ElemNumOfChans != echans:
            echans = shape[-1]
            if ElemNumOfChans != echans:
                raise ValueError('If ElementNumberOfChannels is provided it must be the number of elements in the last dimension of the image or the default value (1 or 2 depending on complex data)')
    else: fields['ElementNumberOfChannels'] = echans
    if echans == shape[-1]: del shape[-1]
    if 'DimSize' in fields:
        DimSize = get('DimSize')
        if prod(DimSize) != prod(shape):
            raise ValueError('If DimSize is provided the image must be reshape-able to it after removing channels')
        shape = tuple(DimSize)
    else:
        shape.reverse()
        fields['DimSize'] = tuple(shape)
    ndims = len(shape)
    if 'NDims' in fields and get('NDims') != ndims: raise ValueError('If NDims is provided it must equal the number of dimensions')
    fields['NDims'] = ndims

    # At this point ElementType, ElementNumberOfChannels, DimSize, NDims have been checked/corrected/set
    # We can now use the automated checkers for the rest
    fields = { k:__header_fields[k].check(v, fields) for k,v in fields.iteritems() }
    for k,v in __defaults.iteritems(): fields.setdefault(k, v(ndims))
    __check_binary(fields, get_dtype_endian(dt) == '>')

    # Ready to save data
    return fields

def save_mha_data(im, filename, fields):
    im = im_decomplexify(im)
    if fields['BinaryData'] and (get_dtype_endian(im.dtype) == '>') != fields['BinaryDataByteOrderMSB']:
        im = im.byteswap() # swaps actual bytes, does not update dtype
        # im = im.newbyteorder() # only updates dtype, not actual bytes
    im = im.reshape(tuple(reversed(fields['DimSize'])) + (fields.get('ElementNumberOfChannels', 1),))
    # im now always ends with the number of channels, even if it is 1

    # Calculate the data file name(s)
    filename = os.path.abspath(filename)
    datafile = __get_datafile(filename, fields)

    # Save data externally and update the CompressedDataSize field
    comp = bool(fields.get('CompressedData', False))
    write = __get_mha_writer(fields)
    if datafile is not None:
        if isinstance(datafile, list): # many external data files
            ndim_files = 1
            for ndim_files in xrange(1, len(im.shape)):
                if prod(im.shape[:ndim_files]) == len(datafile): break
            ind = [slice(None)] * len(im.shape)
            inds = product(*[xrange(x) for x in im.shape[:ndim_files]])
            for df,i in izip(datafile, inds):
                ind[:ndim_files] = i
                write(im[ind], df)
            fields.pop('CompressedDataSize', None)
        else:
            ds = write(im, datafile) # single external data file
            if comp: fields['CompressedDataSize'] = ds
            if not isinstance(datafile, String): datafile.close()
    elif comp: fields['CompressedDataSize'] = b'0               ' # has space for exobytes

    # Order all fields
    #if fields.get('ElementMin') is True: fields['ElementMin'] = im.min()
    #if fields.get('ElementMax') is True: fields['ElementMax'] = im.max()
    ofields = tuple((name,fields[name]) for name in __order_first if name in fields)
    ofields += tuple((name,value) for name,value in fields.iteritems() if name not in __order)
    ofields += tuple((name,fields[name]) for name in __order_last if name in fields)
    # 'Write' the values for the fields
    ofields = tuple((name,__header_fields[name].write(value, fields)) for name,value in ofields)
    # Create the header text
    hdr = ('\n'.join((name+' = '+value) for name,value in ofields if value is not None)+'\n').encode('utf8')

    # Save header and data locally
    with openfile(filename, 'wb') as f:
        f.write(hdr)
        if datafile is None:
            f.flush()
            ds = write(im, f) - len(hdr)
            if comp: # need to go back into header and update compressed data size
                ds = str(ds).encode('utf8')
                if len(ds) <= len(fields['CompressedDataSize']):
                    f.seek(hdr.index(b'0', hdr.index(b'\nCompressedDataSize = ')))
                    f.write(ds)
            return len(hdr)
    headersize = fields.get('HeaderSize', 0)
    return len(headersize) if isinstance(headersize, bytes) else headersize

def __get_datafile(filename, fields):
    directory = os.path.dirname(filename)
    if 'ElementDataFile' not in fields:
        ext = ('.zlib' if fields['CompressedData'] else '.bin') if fields['BinaryData'] else '.txt'
        filename = os.path.splitext(filename)[0]
        datafile = filename + ext
        if os.path.exists(datafile):
            from tempfile import NamedTemporaryFile
            datafile = NamedTemporaryFile('wb', suffix=ext, prefix=filename+'_', dir=directory, delete=False)
            fields['ElementDataFile'] = os.path.relpath(datafile.name, directory)
        else:
            fields['ElementDataFile'] = os.path.relpath(datafile, directory)
        return datafile
    
    datafile = fields['ElementDataFile']
    if isinstance(datafile, tuple):
        pattern, start, stop, step = fields['ElementDataFile']
        return [os.path.abspath(os.path.join(directory, pattern % i)) for i in xrange(start, stop+1, step)]
    if isinstance(datafile, list):
        return [os.path.abspath(os.path.join(directory, fn)) for fn in datafile]
    if isinstance(datafile, String): return os.path.abspath(os.path.join(directory, datafile))
    return datafile

def __get_mha_writer(fields):
    imsave = array_save if fields['BinaryData'] else array_save_ascii
    
    level = fields.get('CompressedData', False)
    if level is True: level = 6
    comp = 'zlib' if level else None

    headersize = fields.get('HeaderSize')
    header = None
    if isinstance(headersize, bytes):
        header = headersize
        headersize = len(header)

    def write(im, datafile):
        file_handle = not isinstance(datafile, String)
        exists = not file_handle and os.path.exists(datafile)
        hs = -im.nbytes if headersize is not None and headersize < 0 else headersize 
        if hs is None or (hs <= 0 and (not exists or (exists and -hs >= get_file_size(datafile)))):
            # Set file to just our data
            with openfile(datafile, 'wb', comp, level) as df: imsave(df, im)
            return get_file_size(datafile)
        else:
            # Set file to just our data preceded by the generic header
            with openfile(datafile, 'r+b' if exists else 'wb') as df_raw:
                if header is not None: df_raw.write(header)
                with openfile(df_raw, 'wb', comp, level, hs) as df: imsave(df, im)
                return df_raw.tell() - hs
    
    return write


####### Image Source ###########################################################

class MetaImage(FileImageSource):
    @classmethod
    def open(cls, filename, readonly=False, **options): #pylint: disable=arguments-differ
        if len(options) > 0: raise ValueError('Invalid option given')
        with openfile(filename, 'rb') as f:
            fields, headersize = read_mha_header(f)
        if fields['NDims'] != 2 or fields.get('ElementNumberOfChannels', 1) > 5:
            raise ValueError('MHA/MHD file not a 2D image')
        return MetaImage(filename, readonly, fields, headersize)

    @classmethod
    def _openable(cls, filename, f, readonly=False, **options):
        #pylint: disable=unused-argument
        if len(options) > 0: return False
        try:
            fields, _ = read_mha_header(f)
            return fields['NDims'] == 2 and fields.get('ElementNumberOfChannels', 1) <= 5
        except StandardError:
            return False

    __forbidden = {'ObjectType', 'ObjectSubType', 'NDims', 'DimSize', 'CompressedDataSize',
                   'ElementNumberOfChannels', 'ElementNBits', 'ElementType'}

    @classmethod
    def create(cls, filename, im, writeonly=False, **fields): #pylint: disable=arguments-differ
        if len(fields.viewkeys() & MetaImage.__forbidden) != 0: raise ValueError("Forbidden fields given")
        fields.setdefault('BinaryData', True)
        mha = os.path.splitext(filename)[1].lower() == '.mha'
        if mha:
            if 'ElementDataFile' in fields: raise ValueError('Cannot specify ElementDataFile with MHA file extension')
            fields['ElementDataFile'] = None
        elif 'ElementDataFile' not in fields:
            ext = ('.zlib' if fields.get('CompressedData',False) else '.bin') if fields['BinaryData'] else '.txt'
            fields['ElementDataFile'] = os.path.splitext(os.path.basename(filename))[0] + ext
        elif fields['ElementDataFile'] in ('LOCAL','Local','local'): raise ValueError('Forbidden ElementDataFile sepcified')
        fields = parse_mha_fields(im.dtype, im.shape, fields)
        imsrc = MetaImage(filename, False, fields, None)
        imsrc.data = im
        return imsrc

    @classmethod
    def _creatable(cls, filename, ext, writeonly=False, **fields):
        return (ext in ('.mha', '.mhd') and
                ('ElementDataFile' not in fields or not (ext == '.mhd' or fields['ElementDataFile'].startswith('LIST') or fields['ElementDataFile'] in ('LOCAL','Local','local')))
                and len(fields.viewkeys() & MetaImage.__forbidden) == 0)

    @classmethod
    def name(cls): return "MHA/MHD"

    @classmethod
    def exts(cls): return ('.mha', '.mhd')

    @classmethod
    def print_help(cls, width):
        from ....imstack import Help
        p = Help(width)
        p.title("MetaImage Image Handler (MHA/MHD)")
        p.text("""
Reads and writes MetaImage files (.mha or .mhd). This supports 2D images with up to 5 channels and
all basic numerical types (unsigned/signed ints and floating-point) except logical and complex
(however these can be accomplished with conversion or multi-channel).

This supports the following MetaImage features:
""")
        p.list("header and image data in single file or seperate",
               "single or multiple data files",
               "compression",
               "user-defined header names/values")
        p.newline()
        p.text("This does not support the following MetaImage features:")
        p.list("non-Image files (specifically not SpatialObjects)",
               "ignores, but parses, many header fields such as TransformMatrix, ElementToIntensityFunctionSlope, and ElementToIntensityFunctionOffset",
               "some header fields while writing (see below)")
        p.newline()
        p.text("""
When reading there are no additional options. When writing there all MetaIO defined header fields
are supported to some extent and user-defined fields can be given. See the MetaIO documentation
for a complete list. For the MetaIO defined header fields that are arrays, specify values as a
space-seperated list of values. The following header fields are treated specially:
""")
        p.list("BinaryData - default is true, if given as false then data will be written as text",
               "CompressedData - default is false, if given as true or a number from 1-9 then data will be compressed (to the given compression level, default 6) [only allowed if BinaryData is true]",
               "BinaryDataByteOrderMSB - forces the saved data byte order [only allowed if BinaryData is true]",
               "HeaderSize - can only be used if extension is not MHA, causes the data to be written this far into the file, otherwise external files are replaced; can also be -1 if binary and not compressed to be placed at the end of an existing file; can also be a string which is used as the header of external files",
               "ElementDataFile - can only be used if extension is not MHA, specifies the filename for the data; file names that start with LIST or are LOCAL/Local/local are forbidden; defaults to this filename with an extension of .bin, .zlib, or .txt depending on the format type")
        p.text("""And the following are forbidden:""")
        p.list('ObjectType', 'ObjectSubType', 'NDims', 'DimSize', 'CompressedDataSize',
               'ElementNumberOfChannels', 'ElementNBits', 'ElementType')

    def __init__(self, filename, readonly, fields, headersize):
        super(MetaImage, self).__init__(filename, readonly)
        self.__fields = fields
        self._set_props(_get_dtype(fields), tuple(reversed(fields['DimSize'])))
        self.__headersize = headersize
    @property
    def header(self): return self.__fields.copy()
    def _get_props(self): pass
    def _get_data(self):
        return read_mha_data(self._filename, self.__fields, self.__headersize)
    def _set_data(self, im):
        if im.dtype != self.dtype or im.shape != self.shape: raise ValueError()
        self.__headersize = save_mha_data(im.data, self._filename, self.__fields)
    def _set_filename(self, filename):
        # TODO: if the data file name was derived from the file name, change it here too
        self._rename(filename)
    @property
    def filenames(self):
        edf = self.__fields['ElementDataFile']
        return (self._filename,) if edf is None else (self._filename,edf)

class MetaImageStack(HomogeneousFileImageStack):
    @classmethod
    def open(cls, filename, readonly=False, **options): #pylint: disable=arguments-differ
        """
        Opens an MHA file. Provide a filename or a file-like object. You can specify if it should be
        opened readonly or not. No extra options are supported.
        """
        if len(options) > 0: raise ValueError('The MHA/MHD ImageStack does not support any additional options')
        with openfile(filename, 'rb') as f: fields,_ = read_mha_header(f)    
        if fields['NDims'] != 3 or fields.get('ElementNumberOfChannels', 1) > 5:
            raise ValueError('MHA/MHD file not a 3D image')
        _,data = imread_mha(filename)
        return MetaImageStack(filename, readonly, fields, data)
    
    @classmethod
    def _openable(cls, filename, f, readonly=False, **opts):
        #pylint: disable=unused-argument
        if len(opts) > 0: return False
        try:
            fields, _ = read_mha_header(f)
            return fields['NDims'] == 3 and fields.get('ElementNumberOfChannels', 1) <= 5
        except StandardError:
            return False

    @classmethod
    def create(cls, filename, ims, writeonly=False, **fields): #pylint: disable=arguments-differ
        """Creates a new MHA file. Provide a filename or a file-like object."""
        if len(fields.viewkeys() & MetaImageStack.__forbidden) != 0: raise ValueError("Forbidden fields given")
        fields.setdefault('BinaryData', True)
        shape, dtype = (len(ims),) + ims.shape, ims.dtype
        mha = os.path.splitext(filename)[1].lower() == '.mha'
        if mha:
            if 'ElementDataFile' in fields: raise ValueError('Cannot specify ElementDataFile with MHA file extension')
            fields['ElementDataFile'] = None
        elif 'ElementDataFile' not in fields:
            ext = ('.zlib' if fields.get('CompressedData',False) else '.bin') if fields['BinaryData'] else '.txt'
            fields['ElementDataFile'] = os.path.splitext(os.path.basename(filename))[0] + ext
        elif fields['ElementDataFile'] in ('LOCAL','Local','local'): raise ValueError('Forbidden ElementDataFile sepcified')
        ims = ims.stack
        ims.flags.writeable = False
        imsave_mha(filename, ims, **fields)
        fields = parse_mha_fields(dtype, shape, fields)
        return MetaImageStack(filename, False, fields, ims)
        
    __forbidden = {'ObjectType', 'ObjectSubType', 'NDims', 'DimSize', 'CompressedDataSize',
                   'ElementNumberOfChannels', 'ElementNBits', 'ElementType'}

    @classmethod
    def _creatable(cls, filename, ext, writeonly=False, **fields):
        return (ext in ('.mha', '.mhd') and
                ('ElementDataFile' not in fields or not (ext == '.mhd' or fields['ElementDataFile'].startswith('LIST') or fields['ElementDataFile'] in ('LOCAL','Local','local')))
                and len(fields.viewkeys() & MetaImageStack.__forbidden) == 0)

    def _delete(self, idxs): raise NotImplemented()
    def _insert(self, idx, ims): raise NotImplemented()

    @classmethod
    def name(cls): return "MHA/MHD"

    @classmethod
    def exts(cls): return ('.mha', '.mhd')
    
    @classmethod
    def print_help(cls, width):
        from ....imstack import Help
        p = Help(width)
        p.title("MetaImage Image Handler (MHA/MHD)")
        p.text("""
Reads and writes MetaImage files (.mha or .mhd). This supports 3D images with up to 5 channels and
all basic numerical types (unsigned/signed ints and floating-point) except logical and complex
(however these can be accomplished with conversion or multi-channel). See the 2D handler information
for more information.""")
        
    def __init__(self, filename, readonly, fields, data):
        self.__filename = filename
        self.__fields = fields
        self._data = data
        w,h,d = fields['DimSize']
        super(MetaImageStack, self).__init__(MHAHeader(fields), [MHASlice(self, z) for z in xrange(d)],
                                             w, h, _get_dtype(fields), readonly)
    @property
    def filenames(self):
        edf = self.__fields['ElementDataFile']
        return (self._filename,) if edf is None else (self._filename,edf)

    @property
    def stack(self): return self.__data
    
class MHASlice(FileImageSlice):
    def __init__(self, stack, z):
        super(MHASlice, self).__init__(stack, z)
        self._im = im = stack._data[z]
        self._set_props(im.dtype, im.shape)
    def _get_props(self): pass
    def _get_data(self):
        return self._im
    def _set_data(self, im): raise NotImplemented()
    def _update(self, z): raise NotImplemented()

class MHAHeader(FileImageStackHeader):
    _fields = None
    _data = None
    def __init__(self, fields):
        self._fields = {}
        self._data = fields
        super(MHAHeader, self).__init__(check=False)
