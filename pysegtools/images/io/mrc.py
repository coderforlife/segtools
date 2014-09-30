from collections import OrderedDict
from numpy import empty, int16, int32, float32

from ...general.datawrapper import ListWrapper, ReadOnlyListWrapper
from ...general.enum import Enum, Flags
from ..types import *
from stack import ImageStack, Header, Field, FixedField, MatchQuality
from _util import file_shift_contents, openfile, imread_raw, imsave_raw

__all__ = ['MRC']

IMOD = 0x444F4D49 # "IMOD"
MAP_ = 0x2050414D # "MAP "
HDR_LEN = 224
LBL_LEN = 80
LBL_COUNT = 10

class MRCMode(int32, Enum):
    Byte    =  0 # 8 bit
    Short   =  1 # 16 bit, signed
    Float   =  2 # 32 bit
    Short2  =  3 # 32 bit, complex, signed
    Float2  =  4 # 64 bit, complex
    _Byte   =  5 # alternate for "Byte"
    UShort  =  6 # 16 bit, non-standard
    Byte3   = 16 # 24 bit, rgb, non-standard

class MRCFlags(int32, Flags):
    SignedByte = 1
    
class MRCEndian(int32, Enum):
    Little    = 0x00004144
    LittleAlt = 0x00000044
    Big       = 0x00001717
    BigAlt    = 0x00000017

# TODO: support mapc/mapr/maps = 2,1,3 using Fortran-ordered arrays?

class MRC(ImageStack):
    """  
    Represents an MRC image. Supports all features of the "IMOD" variant of MRC files.
    """

    @classmethod
    def open(cls, f, readonly=False, **options):
        """
        Opens an MRC file. Provide a filename or a file-like object. You can specify if it should be
        opened readonly or not. No extra options are supported.
        """
        if len(options) > 0: raise ValueError('The MRC ImageStack does not support any additional options')
        readonly = readonly or (hasattr(f, 'mode') and f.mode[0] == 'r' and (len(f.mode) == 1 or f.mode[1] != '+'))
        h = MRCHeader()
        return MRC(h, h._open(f, readonly), readonly)

    @classmethod
    def _openable(cls, f, **opts):
        from struct import unpack
        if len(opts) != 0: return MatchQuality.NotAtAll
        f.seek(0)
        raw = memoryview(f.read(HDR_LEN))
        if len(raw) != HDR_LEN: return MatchQuality.NotAtAll
        map, en = unpack('<ii', raw[208:216])
        endian = '<'
        if map == MAP_:
            if en == MRCEndian.Big or en == MRCEndian.BigAlt: endian = '>'
            elif en != MRCEndian.Little and en != MRCEndian.LittleAlt: return MatchQuality.NotAtAll
        nx, ny, nz, mode = unpack(endian+'4i', raw[:16])
        if nx <= 0 or ny <= 0 or nz < 0 or mode not in MRCMode: return MatchQuality.NotAtAll
        mapc, mapr, maps = unpack(endian+'3i', raw[64:76])
        if mapc != 1 or mapr != 2 or maps != 3: return MatchQuality.NotAtAll
        next = unpack(endian+'i', raw[92:96])[0]
        stamp, flags = unpack(endian+'2i', raw[152:160])
        nlbl = unpack(endian+'i', raw[220:])[0]
        if next < 0 or nlbl < 0 or nlbl > LBL_COUNT or (stamp == IMOD and flags not in MRCFlags): return MatchQuality.NotAtAll
        return MatchQuality.Definitely if map == MAP_ else MatchQuality.Likely

    @classmethod
    def create(cls, f, shape, dtype, **options):
        """
        Creates a new MRC file. Provide a filename or a file-like object. You must provide a shape as
        a height and width tuple and a dtype image type. No extra options are supported.
        """
        if len(options) > 0: raise ValueError('The MRC ImageStack does not support any additional configuration')

        # TODO
        #if hasattr(f, 'mode') and 
        #readonly = (hasattr(f, 'mode') and f.mode[0] == 'r' and (len(f.mode) == 1 or f.mode[1] != '+'))

        h = MRCHeader()
        return MRC(h, h._create(f, readonly), readonly)

    @classmethod
    def _creatable(cls, filename, ext, **opts):
        if len(opts) != 0: return MatchQuality.NotAtAll
        if ext in ('mrc',): return MatchQuality.Definitely
        elif ext in ('st','ali','preali','rec'): return MatchQuality.Likely
        else: return MatchQuality.Unlikely

    @classmethod
    def _description(cls): return "MRC"
        
    def __init__(self, h, f, readonly=False):
        super(MRC, self).__init__(h.nx, h.ny, h.nz, h._dtype, h, readonly)
        h._imstack = self
        self._off = HDR_LEN + LBL_LEN * LBL_COUNT + h.next
        #self._stride = h.nx
        #self._sec_gap = 0
        self._file = f

    # General
    def close(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()
            self._file = None

    # Internal section reading and writing
    def _get_section(self, i, seq):
        if not seq: self._file.seek(self._off + i * self._sec_bytes)
        return imread_raw(self._file, self._shape, self._dtype, 'C')

    def _set_section(self, i, im, seq):
        if not seq: self._file.seek(self._off + i * self._sec_bytes)
        imsave_raw(self._file, im)

    def _del_sections(self, start, stop):
        file_shift_contents(self._file, self._off+stop*self._sec_bytes, self._off+start*self._sec_bytes)

    # Getting Slices
    @property
    def stack(self):
        self._file.seek(self._off)
        return imread_raw(self._file, (self._d,)+self._shape, self._dtype, 'C')

def _f(cast): return Field(cast, False, False)
def _f_ro(cast): return Field(cast, True, False)
def _f_fix(cast, value): return FixedField(cast, value, False)

class MRCHeader(Header):
    __format_new = '10i6f3i3fiih30xhh20xii6h6f3f2ifi' # needs endian byte before using
    __format_old = '<10i6f3i3fiih30xhh20xii6h6f6h3fi' # always little endian

    # These cannot be changed directly: (either they are implied from the data type/size or have a utility to change them properly)
    #  nx, ny, nz, mode, nlabl, next, cmap/stamp, imodStamp/imodFlags, mapc, mapr, maps
    # The following have utility methods to change:
    #  nz, amin, amax, amean, nlabl, next, cmap/stamp, mx, my, mz, xlen, ylen, zlen
    __fields_base = OrderedDict([
            ('nx',_f_ro(int32)),     ('ny',_f_ro(int32)),     ('nz',_f_ro(int32)),     # number of columns, rows, and sections
            ('mode',_f_ro(MRCMode)),                                                   # pixel type (0-4, 6, 16)
            ('nxstart',_f(int32)),   ('nystart',_f(int32)),   ('nzstart',_f(int32)),   # starting point of sub-image (not used in IMOD)
            ('mx',_f_ro(int32)),     ('my',_f_ro(int32)),     ('mz',_f_ro(int32)),     # grid size in X, Y, and Z
            ('xlen',_f_ro(float32)), ('ylen',_f_ro(float32)), ('zlen',_f_ro(float32)), # cell size, pixel spacing = xlen/mx, ...
            ('alpha',_f(float32)),   ('beta',_f(float32)),    ('gamma',_f(float32)),   # cell angles (not used in IMOD)
            ('mapc',_f_fix(int32,1)),('mapr',_f_fix(int32,2)),('maps',_f_fix(int32,3)),# map columns/rows/section in x/y/z (should always be 1,2,3)
            ('amin',_f(float32)),('amax',_f(float32)),('amean',_f(float32)),           # min/max/mean pixel value
            ('ispf',_f(int32)),                                                        # space group number (not used in IMOD)
            ('next',_f_ro(int32)),                                                     # number of bytes in the extended header (called nsymbt in MRC standard)
            ('creatid',_f(int16)),                                                     # used to be an ID, now always 0
            ('nint',_f(int16)),('nreal',_f(int16)),                                    # meaning is dependent on extended header format
            ('imodStamp',_f_ro(int32)),('imodFlags',_f_ro(MRCFlags)),                  # if imodStamp == 0x444F4D49 (IMOD) and imodFlags == 1 then bytes are signed
            ('idtype',_f(int16)),('lens',_f(int16)),('nd1',_f(int16)),('nd2',_f(int16)),('vd1',_f(int16)),('vd2',_f(int16)), # Imaging attributes
            ('tiltangles0',_f(float32)),('tiltangles1',_f(float32)),('tiltangles2',_f(float32)),('tiltangles3',_f(float32)),('tiltangles4',_f(float32)),('tiltangles5',_f(float32)), # Imaging axis
            ])
    __fields_new = OrderedDict([
            ('xorg',_f(float32)),('yorg',_f(float32)),('zorg',_f(float32)),            # origin of image
            ('cmap',_f_fix(int32,MAP_)),('stamp',_f_ro(MRCEndian)),                    # for detecting file type, cmap == 0x2050414D (MAP ) and stamp == 0x00004441 or 0x00001717 for little/big endian
            ('rms',_f(float32)),                                                       # the RMS deviation of densities from mean density
            ('nlabl',_f_ro(int32)),                                                    # number of meaningful labels
            ])
    __fields_old = OrderedDict([
            ('nwave',_f(int16)),('wave1',_f(int16)),('wave2',_f(int16)),('wave3',_f(int16)),('wave4',_f(int16)),('wave5',_f(int16)), # wavelengths
            ('xorg',_f(float32)),('yorg',_f(float32)),('zorg',_f(float32)),            # origin of image
            ('nlabl',_f_ro(int32)),                                                    # number of meaningful labels
            ])

    # Setup all instance variables to make sure they are in __dict__
    
    # Required for headers
    _imstack = None
    _fields = None
    _data = None

    # Specific to MRC
    _is_new = True
    _format = None
    _labels = None
    _extra = None
    _old_next = 0
    _dtype = None

    def __init__(self):
        self._fields = MRCHeader.__fields_base.copy()

    def _open(self, f, readonly):
        from struct import unpack

        ### Opening an existing file ###
        f = openfile(f, 'rb' if readonly else 'r+b')

        # Parse Header
        try:
            raw = f.read(HDR_LEN)
            if len(raw) != HDR_LEN: raise ValueError('MRC file does not have enough bytes for header')
            map, en = unpack('<ii', raw[208:216])
            endian = '<'
            if map == MAP_:
                if en == MRCEndian.Big or en == MRCEndian.BigAlt:
                    endian = '>'
                elif en != MRCEndian.Little and en != MRCEndian.LittleAlt:
                    raise ValueError('MRC file is invalid (stamp is 0x%08x)' % en)
                self._fields.update(MRCHeader.__fields_new)
                self._format = endian + MRCHeader.__format_new
            else:
                self._is_new = False
                self._fields.update(MRCHeader.__fields_old)
                self._format = MRCHeader.__format_old
            self._data = h = OrderedDict(zip(self._fields, unpack(self._format, raw)))
            if self._data['mode'] == 5: self._data['mode'] = 0
            self._check()
            #h['mode'] = MRCMode(h['mode'])
            #h['imodFlags'] = MRCFlags(h['imodFlags']) if h['imodStamp'] == IMOD else MRCFlags._None
            #if self._is_new: h['stamp'] = MRCEndian(h['stamp'])

            mode, next, nlabl = h['mode'], h['next'], h['nlabl']

            if h['nx'] <= 0 or h['ny'] <= 0 or h['nz'] < 0:  raise ValueError('MRC file is invalid (dims are %dx%dx%d)' % (h['nx'], h['ny'], h['nz']))
            if next < 0:                                     raise ValueError('MRC file is invalid (extended header size is %d)' % h['next'])
            if not (0 <= nlabl <= LBL_COUNT):                raise ValueError('MRC file is invalid (the number of labels is %d)' % h['nlabl'])
            if h['mapc'] != 1 or h['mapr'] != 2 or h['maps'] != 3: raise ValueError('MRC file has an unsupported data ordering (%d, %d, %d)' % (h['mapc'], h['mapr'], h['maps']))

            # Read labels and extra header data
            lbls = f.read(LBL_LEN*nlabl)
            if len(lbls) != LBL_LEN*nlabl: raise ValueError('MRC file does not have enough bytes for header')
            self._labels = [lbls[i:i+LBL_LEN].lstrip() for i in xrange(0,len(lbls),LBL_LEN)]
            if next:
                f.seek(HDR_LEN + LBL_LEN * LBL_COUNT)
                self._extra = memoryview(f.read(next))
                if len(self._extra) != next: raise ValueError('MRC file does not have enough bytes for header')
                self._old_next = next

            # Determine data type
            if   mode == MRCMode.Byte:   self._dtype = IM_INT8 if h['imodStamp'] == IMOD and MRCFlags.SignedByte in h['imodFlags'] else IM_UINT8
            elif mode == MRCMode.Short:  self._dtype = IM_INT16.newbyteorder(endian)
            elif mode == MRCMode.Float:  self._dtype = IM_FLOAT32.newbyteorder(endian)
            elif mode == MRCMode.Short2: self._dtype = IM_INT16_2.newbyteorder(endian)
            elif mode == MRCMode.Float2: self._dtype = IM_COMPLEX64.newbyteorder(endian)
            elif mode == MRCMode.UShort: self._dtype = IM_UINT16.newbyteorder(endian)
            elif mode == MRCMode.Byte3:  self._dtype = IM_RGB24
            else:                        raise ValueError('MRC file is invalid (mode is %d)' % mode)
            return f

        except: f.close(); raise

    def _create(self, f, shape, dtype):
        ### Creating a new file ###
        # Get the mode
        if   dtype == IM_RGB24_RAW:      dtype = IM_RGB24
        elif dtype == IM_INT16_2_RAW:    dtype = IM_INT16_2
        elif dtype == IM_INT16_2_RAW_BE: dtype = IM_INT16_2_BE
        if   dtype == IM_UINT8     or dtype == IM_INT8:         mode = MRCMode.Byte
        elif dtype == IM_INT16     or dtype == IM_INT16_BE:     mode = MRCMode.Short
        elif dtype == IM_FLOAT32   or dtype == IM_FLOAT32_BE:   mode = MRCMode.Float
        elif dtype == IM_INT16_2   or dtype == IM_INT16_2_BE:   mode = MRCMode.Short2
        elif dtype == IM_COMPLEX64 or dtype == IM_COMPLEX64_BE: mode = MRCMode.Float2
        elif dtype == IM_UINT16    or dtype == IM_UINT16_BE:    mode = MRCMode.UShort
        elif dtype == IM_RGB24     or dtype == IM_RGB24_RAW:    mode = MRCMode.Byte3
        else: raise ValueError('dtype not supported')
        self._dtype = dtype
        endian = get_dtype_endian(dtype)

        # Create the header and write it
        ny, nx = shape
        self._fields.update(MRCHeader.__fields_new)
        self._format = endian + MRCHeader.__format_new
        self._data = OrderedDict([
##            ('nx',nx, 'ny': ny, 'nz': 0,
##            ('mode',mode,
##            ('nxstart',0, 'nystart': 0,'nzstart': 0,
##            ('mx',nx, 'my': ny, 'mz': 1,
##            ('xlen',float(nx), 'ylen': float(ny), 'zlen': 1.0,
##            ('alpha',90.0, 'beta': 90.0, 'gamma': 90.0, 'mapc': 1, 'mapr': 2, 'maps': 3,
##            ('amin',0.0, 'amax': 0.0, 'amean': 0.0,
##            ('ispf',0, 'next': 0, 'creatid': 0, 'nint': 0, 'nreal': 0,
##            ('imodStamp': IMOD, 'imodFlags': MRCFlags.SignedByte if dtype == IM_INT8 else MRCFlags(0),
##            ('idtype',0, 'lens': 0, 'nd1': 0, 'nd2': 0, 'vd1': 0, 'vd2': 0,
##            ('tiltangles0': 0.0, 'tiltangles1': 0.0, 'tiltangles2': 0.0, 'tiltangles3': 0.0, 'tiltangles4': 0.0, 'tiltangles5': 0.0,
##            ('xorg',0.0), 'yorg': 0.0, 'zorg': 0.0,
##            ('cmap',MAP_), 'stamp': MRCEndian.Little if endian == '<' else MRCEndian.Big,
##            ('rms',0.0),
##            ('nlabl',1),
        ])
        self._labels = ['Python MRC Creation']
        self._check()

        # Open file (truncates if existing) and write new header
        f = openfile(f, 'w+b')
        try:
            self._save(f)
            return f
        except: f.close(); raise

    def _get_field_name(self, f): return f if f in self._fields else None
    
    def convert_to_new_format(self):
        """
        Converts MRC header from old format to new format. Does not write to disk.
        """
        if self._imstack._readonly: raise AttributeError('header is readonly')
        if self._is_new: return
        del self._data['nwave']
        del self._data['wave1']
        del self._data['wave2']
        del self._data['wave3']
        del self._data['wave4']
        del self._data['wave5']
        self._data['cmap'] = MAP_
        self._data['stamp'] = MRCEndian.Little
        self._data['rms'] = float32(0.0)
        self._fields = MRCHeader.__fields_base.copy()
        self._fields.update(MRCHeader.__fields_new)
        self._format = '<' + MRCHeader.__format_new
        self._is_new = True
        
    @property
    def pixel_spacing(self):
        """Gets the pixel spacing of the data"""
        return (self._data['xlen']/self._data['mx'], self._data['ylen']/self._data['my'], self._data['zlen']/self._data['mz'])
    @pixel_spacing.setter
    def pixel_spacing(self, value):
        """Sets the pixel spacing in the header but does not write the header to disk."""
        if self._imstack._readonly: raise AttributeError('header is readonly')
        if len(spacing) != 3: raise ValueError()
        self._data['xlen'] = float32(spacing[0])/self._data['mx']
        self._data['ylen'] = float32(spacing[1])/self._data['my']
        self._data['zlen'] = float32(spacing[2])/self._data['mz']

    def update_pixel_values(self):
        """
        Updates the header properties 'amin', 'amax', and 'amean' to the current image data. Does
        not write the header to disk.
        """
        if self._imstack._readonly: raise Exception('readonly')
        if self._data['nz'] == 0:
            self._data['amin'] = float32(0.0)
            self._data['amax'] = float32(0.0)
            self._data['amean'] = float32(0.0)
        else:
            itr = iter(self._imstack)
            im = itr.next()
            amin = im.min()
            amax = im.max()
            amean = im.mean()
            for im in itr:
                amin = min(amin, im.min())
                amax = max(amax, im.max())
                amean += im.mean()
            self._data['amin'] = float32(amin)
            self._data['amax'] = float32(amax)
            self._data['amean'] = float32(amean) / self._data['nz']
    def _update_depth(self, d):
        if d == 0:
            self._data['amin'] = float32(0.0)
            self._data['amax'] = float32(0.0)
            self._data['amean'] = float32(0.0)
        self._data['zlen'] = self._data['zlen'] / self._data['mz'] * d if d != 0 else float32(1.0)
        self._data['nz'] = int32(d)
        self._data['mz'] = int32(max(d, 1))

    @property
    def labels(self): return ReadOnlyListWrapper(self._labels) if self._imstack._readonly else LabelList(self)
    @labels.setter
    def labels(self, lbls):
        if self._imstack._readonly: raise Exception('readonly')
        lbls = [str(l) for l in lbls]
        if len(lbls) > LBL_COUNT: raise ValueError('lbls is too long (max label count is %d)' % LBL_COUNT)
        if any(len(l) > LBL_LEN for l in lbls): raise ValueError('lbls contains label that is too long (max label length is %d)' % LBL_LEN)
        self._labels[:] = lbls # copy it this way so that any references to the list are updated as well
        self._data['nlabl'] = int32(len(lbls))
        
    @property
    def extra(self): return self._extra # will be readonly in all cases where where header is readonly, and can never change size (I believe)
    @extra.setter
    def extra(self, value):
        if self._imstack._readonly: raise AttributeError('header is readonly')
        if value == None:
            self._extra = None 
            self._data['next'] = int32(0)
        else:
            self._extra = memoryview(value)
            self._data['next'] = int32(len(self._extra) * self._extra.itemsize)
   
    def save(self, update_pixel_values=True):
        """
        Write the header to disk. Updates fields as necessary. The fields amin, amax, and amean are
        only updated when update_pixel_values is True (which is default).
        """
        if self._imstack._readonly: raise AttributeError('header is readonly')
        if update_pixel_values: self.update_header_pixel_values()
        self._check()
        if self._old_next != self._data['next']:
            # Header changed size, need to shift image data
            next = self._data['next']
            new_off = HDR_LEN + LBL_LEN * LBL_COUNT + next
            file_shift_contents(f, self._imstack._off, new_off)
            self._old_next = next
            self._imstack._off = new_off
        self._save(self._imstack._file)
        
    def _save(self, f):
        """Internal saving function"""
        from struct import pack
        f.seek(0)
        values = [self._data[field] for field in self._fields]
        f.write(pack(self._format, *values))
        for lbl in self._labels: f.write(lbl.ljust(LBL_LEN))
        f.write(' ' * LBL_LEN * (LBL_COUNT - len(self._labels)))
        if self._extra: f.write(self._extra)
        f.flush()

class LabelList(ListWrapper):
    """
    A wrapper around the labels list that checks for valid values and keeps the nlabl header value
    up-to-date.
    """
    def __init__(self, h):
        self._hdr = h
        super(LabelList, self).__init__(h._labels)
    def __delitem__(self, i):
        del self._data[i]
        self._hdr['nlabl'] = int32(len(self._data))
    def __setitem__(self, i, value):
        value = str(value)
        if len(value) > LBL_LEN: raise ValueError('label too long')
        self._data[i] = value
        #self._hdr['nlabl'] = len(self._data)
    def insert(self, i, value):    
        value = str(value)
        if len(value) > LBL_LEN: raise ValueError('label too long')
        if len(self._data) >= LBL_COUNT:
            if i == len(self): del self._data[0] # appending
            else: raise ValueError('too many labels') # inserting in middle
        self._data.insert(i, value)
        self._hdr['nlabl'] = int32(len(self._data))
    def extend(self, values):
        values = [str(value) for value in values]
        if len(values) > LBL_COUNT: raise ValueError('adding too many labels')
        if any(len(value) > LBL_LEN for value in values): raise ValueError('label too long')
        if len(self._data) + len(values) > LBL_COUNT:
            del self._data[:len(values)-LBL_LEN]
        self._data.extend(values)
        self._hdr['nlabl'] = int32(len(self._data))
