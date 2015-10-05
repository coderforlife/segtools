"""
Functions that read and write gzipped files or data.

The user of the file doesn't have to worry about the compression, but random access is not allowed.

The Python gzip module was used for some inspiration, particularly with reading files (read and
readline are nearly verbatim from it).

Additions over the default Python gzip module:
    * Supports pure deflate and zlib data in addition to gzip files
    * Supports modifying and retrieving all the gzip header properties
    * Adds and checks header checksums for increased file integrity
    * Allows you to read embedded compressed data with rewind and negative seek support when you
      specify the starting offset of the data
    * Does not use seek except for rewind, negative seeks, and starting offsets (means you can use
      it on unbuffered socket connections when not using those features)
    * Adds readinto() function
    * Adds utility functions for compressing and decompressing buffers and files in the different
      formats and can guess which of the three formats a file may be in

Breaking changes:
    * Does not add an 'open()' function (but would be trivial to add)
    * The constructor has changed:
        * 'filename' and 'fileobj' have been combined into a single argument 'output'
        * 'compresslevel' is now just 'level'
        * fourth argument is now 'method' for the compression method ('deflate', 'zlib', 'gzip')
        * 'mtime' is now only supported as a keyword argument when writing gzip files
        * to include a filename in the gzip header, provide it as a keyword argument
        * Overall: if using 3 or less non-keyword arguments it will work as before otherwise not
    * Undocumented properties have essentially all been removed or renamed, most notably:
        * 'fileobj' and 'myfileobj' are now 'base' ('owns_handle' determines if it is 'my' or not)
        * 'mode' is the actual file mode instead of a 1 or 2 indicating READ or WRITE
        * 'mtime' is now 'gzip_options['mtime']' when method is 'gzip' (otherwise not available)
        * deprecated 'filename' is now 'gzip_options['filename']' when method is 'gzip' (otherwise not available)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io, os, re, sys, struct
from time import time
from collections import OrderedDict
from datetime import date, datetime
import zlib

__all__ = ['gzip_oses', 'default_gzip_os',
           'compress_file', 'decompress_file', 'compress', 'decompress',
           'guess_file_compression_method', 'guess_compression_method',
           'GzipFile']

_FTEXT, _FHCRC, _FEXTRA, _FNAME, _FCOMMENT = 0x01, 0x02, 0x04, 0x08, 0x10

_checksums = {
    'gzip' : zlib.crc32,
    'zlib' : zlib.adler32,
    'deflate' : lambda data, value=None: 0,
    }
gzip_oses = {
    'FAT' : 0,
    'Amiga' : 1,
    'VMS' : 2,
    'Unix' : 3,
    'VM/CMS' : 4,
    'Atari TOS' : 5,
    'HPFS' : 6,
    'Macintosh' : 7,
    'Z-System' : 8,
    'CP/M' : 9,
    'TOPS-20' : 10,
    'NTFS' : 11,
    'QDOS' : 12,
    'Acorn RISCOS' : 13,
    'Unknown' : 255,
    }
__default_gzip_oses = {
    'nt' : 0, 'os2' : 0, 'ce' : 0, # FAT (NT could also be NTFS or HPFS and OS/2 could be HPFS)
    'posix' : 3, # UNIX
    'riscos' : 13, # Acorn RISCOS
    }
default_gzip_os = __default_gzip_oses.get(os.name, 255) # default is unknown, including for 'java'
_exts = {
    'gzip' : '.gz',
    'zlib' : '.zlib',
    'deflate' : '.deflate',
    }

String,Unicode,Byte = (str,str,int) if (sys.version_info[0] == 3) else (basestring,unicode,ord)
_re_not_zero = re.compile(br'[^\0]')
_uint16 = struct.Struct(str('<H'))
_uint32 = struct.Struct(str('<L'))
_uint16_be = struct.Struct(str('>H'))
_uint32_be = struct.Struct(str('>L'))
def _get_filename(f, default=None):
    if isinstance(f, String): return f
    elif hasattr(f, 'name') and isinstance(f.name, String) and (len(f.name) < 2 or f.name[0] != '<' and f.name[-1] != '>'):
        return f.name
    return default
def _gzip_header_str(s):
    if s is None: return None
    s = Unicode(s)
    i = s.find('\x00')
    if i >= 0: s = s[:i]
    s.encode('iso-8859-1') # just to make sure it will be encodable
    return s
def _write_gzip_header_str(f, s, chk16):
    s = s.encode('iso-8859-1') + b'\x00'
    f.write(s)
    return zlib.crc32(s, chk16) & 0xffffffff
def _read_gzip_header_str(read, crc32):
    s = b''
    while True:
        c = read(1)
        if not c or c == b'\x00': break
        s += c
    return s.decode('iso-8859-1'), (zlib.crc32(s+b'\x00', crc32) & 0xffffffff)


# GZIP Format {Little-Endian}
# 1F 8B CM FG [MTIME (4)] XF OS
#   CM = 08 - deflate compression method
#   FG = 01 - file is an ASCII text file
#        02 - CRC16 for header is present
#        04 - extra fields are present
#        08 - original file name is present
#        10 - comment is present
#   MTIME = mod. time as secs since 00:00:00 GMT 01/01/70 of the orig file, when compression started, or 0
#   XF = 2 for max compression, 4 for fastest compression
#   OS = the filesystem where the file came from
# [extra data]
# [filename]
# [comment]
# [CRC16 checksum of header]
# <compressed data>
# CRC32 checksum
# Size of uncompressed data
#
# ZLIB Format {Big-Endian}
# CM-CF FL [DICT (4)]
#   CM = bits 0 to 3 => 8 for deflate compression method
#   CF = bits 4 to 7 => base-2 logarithm of the LZ77 window size minus 8 (0 is 256, 7 is 32K, 8 and above are not allowed)
#   FL = bits 0 to 4 => check bits for CM-CF, when viewed as a 16-bit int is a multiple of 31 (CMCF*256 + FL)
#        bit  5      => 0 almost exclusively, 1 if there is a dictionary to be used (which won't be supported)
#        bits 6 to 7 => compression level, 0 is fastest => 3 is slowest/max
#   DICT = not supported
# <compressed data>
# Adler32 checksum

def compress_file(inpt, output=None, level=9, method=None):
    if method is None: method = 'gzip'

    # Get output filename
    in_filename = _get_filename(inpt)
    if output is None:
        if in_filename is None: raise ValueError('Unable to determine output filename')
        output = in_filename + _exts[method]

    # Get gzip options
    opts = __compress_file_gzip_opts(in_filename, inpt) if method == 'gzip' else {}

    # Copy data
    with GzipFile(output, 'wb', level, method, **opts) as output:
        owns_handle = isinstance(inpt, String)
        if owns_handle: inpt = io.open(inpt, 'rb')
        try:
            while True:
                buf = inpt.read(10*1024*1024)
                if len(buf) == 0: break
                output.write(buf)
        finally:
            if owns_handle: inpt.close()
def __compress_file_gzip_opts(in_filename, inpt):
    opts = {}
    try:
        if in_filename:
            opts['filename'] = os.path.basename(in_filename)
            opts['mtime'] = os.path.getmtime(in_filename)
        else:
            opts['mtime'] = os.fstat(inpt.fileno()).st_mtime
    except (OSError, IOError): pass
    return opts

def decompress_file(inpt, output=None, method=None):
    with GzipFile(inpt, 'rb', method=method) as inpt:
        # Get the output filename if not provided
        in_filename = _get_filename(inpt)
        if not output and inpt.method == 'gzip':
            output = inpt.gzip_options.get('filename')
        if not output and in_filename and in_filename.endswith(_exts[inpt.method]):
            output = in_filename[:-len(_exts[inpt.method])]
        if not output: raise ValueError('Unable to determine output filename')

        # Copy data
        owns_handle = isinstance(output, String)
        if owns_handle: output = io.open(output, 'wb')
        try:
            while True:
                buf = inpt.read(10*1024*1024)
                if len(buf) == 0: break
                output.write(buf)
        finally:
            if owns_handle: output.close()

    # Set mtime on output file if it is available
    if owns_handle and inpt.method == 'gzip' and 'mtime' in inpt.gzip_options:
        try:
            os.utime(output, (time(), inpt.gzip_options['mtime']))
        except OSError: pass

def compress(inpt, level=9, method=None):
    level = int(level)
    if method not in (None, 'gzip', 'zlib', 'deflate'): raise ValueError('Compression method must be one of deflate, zlib, or gzip')
    # zlib.compress doesn't support wbits argument, which we need to set as negative to prevent headers being added
    c = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    data1 = c.compress(inpt)
    data2 = c.flush(zlib.Z_FINISH)
    del c
    if method == 'gzip' or method is None:
        out = bytearray(len(data1)+len(data2)+18)
        out[12:12+len(data1)] = data1
        out[12+len(data1):-8] = data2
        del data1, data2
        # Header
        out[:4] = b'\x1F\x8B\x08\x02'
        _uint32.pack_into(out, 4, int(time()))
        out[8] = 2 if level >= 7 else (4 if level <= 2 else 0)
        out[9] = 0xFF
        _uint16.pack_into(out, 10, 0xffff & zlib.crc32(buffer(out)[:10]))
        # Footer
        _uint32.pack_into(out, -8, 0xffffffff & zlib.crc32(inpt))
        _uint32.pack_into(out, -4, 0xffffffff & len(inpt))
        return bytes(out)
    elif method == 'zlib':
        out = bytearray(len(data1)+len(data2)+6)
        out[2:2+len(data1)] = data1
        out[2+len(data1):-4] = data2
        del data1, data2
        # Header
        header = 0x7800 | (((level+1)//3) << 6)
        mod31 = header % 31
        if mod31 != 0: header += (31 - mod31)
        _uint16_be.pack_into(out, 0, header)
        # Footer
        _uint32_be.pack_into(out, -4, 0xffffffff & zlib.adler32(inpt))
        return bytes(out)
    else: # method == 'deflate'
        return data1 + data2
        

def decompress(inpt, method=None):
    if method is None: method = guess_compression_method(inpt)
    if method == 'gzip': return __decompress_gzip(inpt)
    elif method == 'zlib': return __decompress_zlib(inpt)
    elif method == 'deflate': return zlib.decompress(inpt)
    else: raise ValueError('Compression method must be one of deflate, zlib, gzip, or None')
def __decompress_gzip(inpt):
    if len(inpt) < 18 or inpt[:3] != b'\x1F\x8B\x08': raise IOError('Not a gzipped file')
    flags = Byte(inpt[3])
    if flags & 0xE0: raise IOError('Unknown flags')
    off = (_uint16.unpack_from(inpt, 10)[0] + 12) if flags & _FEXTRA else 10
    if flags & _FNAME:    off = inpt.index(b'\x00', off) + 1
    if flags & _FCOMMENT: off = inpt.index(b'\x00', off) + 1
    if flags & _FHCRC:
        if _uint16.unpack_from(inpt, off)[0] != (zlib.crc32(buffer(inpt)[:off]) & 0xffff): raise IOError('Header corrupted')
        off += 2
    isize = _uint32.unpack_from(inpt, -4)[0]
    out = zlib.decompress(inpt[off:-8], -zlib.MAX_WBITS, isize)
    if _uint32.unpack_from(inpt, -8)[0] != (zlib.crc32(out) & 0xffff): raise IOError("CRC32 check failed")
    if isize != (len(out) & 0xffffffff): raise IOError("Incorrect length of data produced")
    return out
def __decompress_zlib(inpt):
    header = _uint16_be.unpack_from(inpt)[0]
    method = (header >>  8) & 0xF
    wsize  = ((header >> 12) & 0xF) + 8
    fdict  = (header & 0x20) != 0
    if method != 8 or wsize > zlib.MAX_WBITS or fdict: raise IOError('Unknown compression method')
    if header % 31 != 0: raise IOError('Header corrupted')
    out = zlib.decompress(inpt[2:-4], -wsize)
    if _uint32_be.unpack_from(inpt, -4)[0] != (zlib.adler32(out) & 0xffffffff): raise IOError("Adler32 check failed")
    return out

def guess_file_compression_method(f):
    if isinstance(f, String):
        with io.open(f, 'rb') as f: return guess_compression_method(f.read(3))
    else: return guess_compression_method(f.read(3))

def guess_compression_method(buf):
    # Pure deflate streams could randomly look like GZIP or ZLIB
    # Assuming the first 2 or 3 bytes of a deflate stream are independent and uniformly random:
    #  GZIP has a ~1/16.8 million chance of a false positive (could be lowered with >3 bytes of data)
    #  ZLIB has a ~1/1000 chance of a false positive (TODO: wish we could lower this - maybe assume FDCIT is always 0 (1/2000 then, still not great))
    #  GZIP and ZLIB can never be confused we each other
    if len(buf) > 2 and buf[:3] == b'\x1F\x8B\x08': return 'gzip' # could also check flags and checksum
    if len(buf) > 1:
        h = _uint16_be.unpack_from(buf)[0]
        if (h&0x8F00) == 0x0800 and h%31 == 0: return 'zlib'
    return 'deflate'

class GzipFile(io.BufferedIOBase):
    #pylint: disable=too-many-instance-attributes
    __offset = 0
    max_read_chunk = 10 * 1024 * 1024 # 10Mb

    def __init__(self, file, mode=None, level=9, method=None, start_off=None, **kwargs): #pylint: disable=redefined-builtin
        """
        Creates a file-like object that wraps another file-like object and either compresses all
        data written to it or decompresses all data read from it. Cannot be used for both read
        and write operations.

        If you are using a file you can provide the filename as 'file'. You can also provide
        file-like objects for 'file' that follow these rules:
            If writing compressed data it must have:
                mode property if not provided to the constructor
                write amd flush methods
            If reading compressed data it must have:
                read method
            If you want to rewind or negative seek (only available while reading):
                seek method
            Optionally used if available:
                name property
                isatty method
        If you provide a file-like object it will not be closed when this object is closed. This can
        be changed by setting the property owns_handle to True.

        Mode needs to be 'rb' (default) for reading or 'wb' or 'ab' for writing. The 'b' is required
        and the letters 't', 'U' and '+' are all invalid. If using a file-like object that has a
        mode property you do not need to give the mode here. In this case read/write files are
        allowed but are treated as though the mode does not have +. Note that 'wb' truncates a file
        at start_off instead of the entire file.

        The method must be 'deflate' (RFC 1951), 'zlib' (RFC 1950), or 'gzip' (RFC 1952). Default is
        None which means 'gzip' for writing and guess the format when reading. However, it is always
        best if you know the format to supply it.

        The level is the compression level from 0 (none) to 9 (maximum compression - default). It is
        ignored when reading.

        The start_off parameter is used when a compressed entry is embedded within another file. The
        value is the start of the compressed region within the file. When given, the base file is
        seeked upon opening and when rewinding/negative seeking. If it is None and a file-like
        object is given, the current offset is safely queried using tell() but assumed to be 0.

        When writing a gzip file you can include extra information with the following keyword
        arguments:
            os         an integer from 0 to 255 or string that describes the filesystem where the
                       file orginated (default depends on system, only strings in gzip_oses are
                       supported)
            mtime      a datetime object or an integer representing the modification time of the
                       original file as a UNIX timestamp (default is now)
            text       True if the data being written is text (default is binary)
            filename   the original filename that is being compressed (default is filename without
                       .gz if obtainable)
            comment    to a user-readable comment
            extras     an OrderedDict, dictionary, or iterable of tuples of subfields (id:data) with
                       the ids being 2-byte strings and data being convertible to byte.

        When reading gzip data the extra information is available from the gzip_options property.
        The properties when reading or writing are "standardized" so that 'os' is an integer,
        'mtime' is a datetime, 'text' is a boolean, 'filename' and 'comment' or unicode strings, and
        'extras' is an OrderedDict of bytes. Fields not in the header are simply not included. Also,
        a key 'xf' is available in the dictionary corresponding to the XF flag of the header (2 for
        max compression, 4 for fastest compression, and 0 otherwise). This value is ignored if given
        while writing, and a calculated value is used.
        """
        super(GzipFile, self).__init__()

        # Basic checks
        mode, writing = GzipFile.__check_mode(file, mode)
        level = GzipFile.__check_level(level)
        method = GzipFile.__check_method(method, writing)
        start_off = GzipFile.__check_start_off(start_off, mode)

        # Setup properties
        self.owns_handle = isinstance(file, String)
        self.name = _get_filename(file, '')
        if writing and method == 'gzip':
            self.__gzip_options = GzipFile.__check_gzip_opts(kwargs, self.name)
        elif kwargs: raise ValueError('Extra keyword arguments can only be provided when writing gzip data')
        self.__mode = mode
        self.__writing = writing
        self.__start_off = start_off #0 if start_off is None else start_off
        self.__size = 0
        self.__base = self.__open(file)
        if method is None:
            self.__base_buf = self.__base.read(3)
            method = guess_compression_method(self.__base_buf)
        elif 'r' in mode:
            self.__base_buf = b''
        self.__method = method
        self.__calc_checksum = _checksums[method]
        self.__checksum = self.__calc_checksum(b'') & 0xffffffff

        # Initialize based on reading or writing
        if writing:
            self.__zlib = self.__init_writing(level)
        else:
            self.__zlib = self.__read_header()
            self.__new_member = False
            self.__decomp_buf = b'' # data that has been decompressed but not officially read
            self.__min_readsize = 100 # Starts small, scales exponentially


    @staticmethod
    def __check_mode(f, mode):
        is_filename = isinstance(f, String)
        if not mode: mode = f.mode if not is_filename and hasattr(f, 'mode') else 'rb'
        if any(c not in 'rwa+btU' for c in mode) or sum(mode.count(c) for c in 'rwa') != 1 or sum(mode.count(c) for c in 'bt') > 1: raise ValueError('Mode contains invalid characters')
        if 'b' not in mode: raise ValueError('Text mode not supported')
        if is_filename and '+' in mode: raise ValueError('Read/write mode not supported')
        mode = ('r' if 'r' in mode else 'w' if 'w' in mode else 'a') + 'b' # normalized, missing +
        return mode, 'r' not in mode

    @staticmethod
    def __check_level(level):
        level = int(level)
        if level < 0 or level > 9: raise ValueError('Compression level must be between 0 and 9 (inclusive)')
        return level

    @staticmethod
    def __check_method(method, writing):
        if method not in ('deflate', 'zlib', 'gzip', None): raise ValueError('Compression method must be one of deflate, zlib, or gzip (or None if reading)')
        return 'gzip' if method is None and writing else method

    @staticmethod
    def __check_start_off(start_off, mode):
        if start_off is not None:
            start_off = int(start_off)
            if 'a' in mode: raise ValueError('Starting offset not supported in append mode')
            if start_off < 0: raise ValueError('Starting offset cannot be negative')
        return start_off

    @staticmethod
    def __check_gzip_opts(kwargs, filename):
        if len(kwargs.viewkeys()-{'text','os','mtime','filename','comment','extras','xf'}):
            raise ValueError('Gzip options must only include text, os, mtime, filename, comment, and extras')

        gzip_os = kwargs.get('os', default_gzip_os)
        gzip_os = gzip_oses[gzip_os] if isinstance(gzip_os, String) else int(gzip_os)
        if gzip_os > 255 or gzip_os < 0: raise ValueError('Gzip OS is an invalid value')

        mtime = kwargs.get('mtime', datetime.now())
        mtime = ((mtime if isinstance(mtime, datetime) else datetime(mtime.year, mtime.month, mtime.day))
                 if isinstance(mtime, date) else datetime.fromtimestamp(int(mtime)))
        opts = {'os':gzip_os, 'mtime':mtime, 'text':('text' in kwargs and kwargs['text'])}
        
        filename = filename[:-3] if filename and filename.endswith('.gz') else None
        filename = kwargs.get('filename', filename)
        if filename is not None: opts['filename'] = _gzip_header_str(filename)
        
        if 'comment' in kwargs:  opts['comment']  = _gzip_header_str(kwargs['comment'])
        
        extras = kwargs.get('extras')
        if extras is not None:
            from collections import Mapping
            itr = (extras.iteritems() if hasattr(extras, 'iteritems') else extras.items()) \
                  if isinstance(extras, Mapping) else extras
            extras = opts['extras'] = OrderedDict()
            xlen = 2
            for ex_id, data in itr:
                ex_id, data = bytes(ex_id), bytes(data)
                if len(ex_id) != 2: raise ValueError('Gzip extras subfield id must be 2 bytes')
                extras[ex_id] = data
                xlen += 4 + len(data)
            if xlen > 0xFFFF: raise ValueError('Gzip extra data has too much data')
            extras.xlen = xlen
            
        return opts

    def __open(self, f):
        if self.owns_handle:
            if self.__start_off is None:
                f = io.open(f, self.__mode)
            elif self.__writing:
                io.open(f, 'ab').close() # forces file to exist (but does not truncate existing file)
                f = io.open(f, 'r+b') # open file in read+write mode so we don't truncate file at 0
                f.truncate(self.__start_off)
                f.seek(self.__start_off)
            else:
                f = io.open(f, 'rb')
                f.seek(self.__start_off)
        elif self.__start_off is None:
            if 'a' not in self.__mode:
                try: self.__start_off = f.tell()
                except IOError: pass
        else:
            f.seek(self.__start_off)
        return f

    def __check(self, writing=None):
        """Raises an IOError if the underlying file object has been closed."""
        if self.closed: raise IOError('I/O operation on closed file.')
        if writing is not None and self.__writing != writing:
            raise IOError('Cannot write to read-only file' if writing else 'Cannot read, seek, or rewind a write-only file')

    # Readonly Properties
    @property
    def raw(self): return self.__base
    @property
    def method(self): return self.__method
    @property
    def mode(self): return self.__mode
    @property
    def checksum(self): return self.__checksum
    @property
    def gzip_options(self):
        if not hasattr(self, '_GzipFile__gzip_options'): raise AttributeError('Not a gzip file')
        return self.__gzip_options.copy() # they aren't allowed to manipulate the internal dictionary

    # Close
    @property
    def closed(self): return self.__base is None
    def close(self):
        """
        If writing, completely flush the compressor and output the checksum if the format has it.
        Always close the file. If called more than once, subsequent calls are no-op.
        """
        if self.closed: return
        if self.__writing:
            self.__base.write(self.__zlib.flush(zlib.Z_FINISH))
            del self.__zlib, self.__calc_checksum
            if self.__method == 'gzip':
                self.__base.write(_uint32.pack(self.__checksum))
                self.__base.write(_uint32.pack(self.__size & 0xffffffff))
            elif self.__method == 'zlib':
                self.__base.write(_uint32_be.pack(self.__checksum))
            self.__base.flush()
        if self.owns_handle: self.__base.close()
        self.__base = None
    def detatch(self):
        # TODO: should this flush like close()?
        base = self.__base
        self.__base = None
        return base

    # Random Properties
    def fileno(self): raise IOError('gzip file does not have a file number, try this.raw.fileno()')
    def isatty(self):
        """Returns True if the underlying file object is interactive."""
        return self.__base.isatty()
    def __repr__(self):
        return '<gzip ' + repr(self.__base) + ' at ' + hex(id(self)) + '>'
    @property
    def internal_state(self):
        state = (self.__base.tell(), self.__checksum, self.__size, self.__offset, self.__zlib.copy())
        if not self.__writing: state += (self.__new_member, self.__base_buf, self.__decomp_buf)
        return state
    @internal_state.setter
    def internal_state(self, state):
        self.__base.seek(state[0])
        self.__checksum, self.__size, self.__offset = state[1:3]
        self.__zlib = state[4].copy()
        if self.__writing:
            self.__base.truncate(state[0])
        else:
            self.__new_member, self.__base_buf, self.__decomp_buf = state[5:7]

    # Position
    def seekable(self): return not self.__writing
    def tell(self):
        """Return the uncompressed stream file position indicator to the beginning of the file"""
        self.__check()
        return self.__offset + self.__start_off
    def rewind(self):
        self.__check(False)
        self.__base.seek(self.__start_off)
        self.__new_member = True
        self.__decomp_buf = b''
        self.__offset = 0
    def seek(self, offset, whence=None):
        self.__check(False)
        if whence:
            if whence == io.SEEK_CUR: offset += self.__offset
            else: raise IOError('Seek from end not supported') # whence == io.SEEK_END
        else:
            offset -= self.__start_off
        if offset < 0: raise IOError('Invalid offset')
        if offset < self.__offset: self.rewind() # for negative seek, rewind and do positive seek
        self.__skip(offset - self.__offset)
        return self.__offset
    def truncate(self, size_=None): raise IOError('Truncate not supported for gzip files')

    # Writing
    def __init_writing(self, level):
        windowsize = zlib.MAX_WBITS
        if self.__method == 'gzip':
            opts = self.__gzip_options
            flags = _FHCRC
            if opts['text']:       flags |= _FTEXT
            if 'extras'   in opts: flags |= _FEXTRA
            if 'filename' in opts: flags |= _FNAME
            if 'comment'  in opts: flags |= _FCOMMENT
            header = bytearray(10)
            header[:3] = b'\x1F\x8B\x08'
            header[3] = flags
            mtime = opts['mtime']
            if mtime.tzinfo is not None and mtime.tzinfo.utcoffset(mtime) is not None:
                mtime = mtime.replace(tzinfo=None) - mtime.utcoffset()
            _uint32.pack_into(header, 4, (mtime - datetime.fromtimestamp(0)).total_seconds())
            header[8] = opts['xf'] = 2 if level >= 7 else (4 if level <= 2 else 0)
            header[9] = opts['os']
            self.__base.write(header)
            chk16 = zlib.crc32(buffer(header)) & 0xffffffff
            if 'extras' in opts:
                xlen = opts['extras'].xlen
                extras = bytearray(xlen)
                _uint16.pack_into(extras, 0, xlen)
                off = 2
                for ex_id, data in opts['extras'].iteritems():
                    l = len(data)
                    extras[off:off+2] = ex_id
                    _uint16(extras, off+2, l)
                    extras[off+4:off+4+l] = data
                    off += 4 + l
                self.__base.write(extras)
                chk16 = zlib.crc32(buffer(extras)) & 0xffffffff
            if 'filename' in opts: chk16 = _write_gzip_header_str(self.__base, opts['filename'], chk16)
            if 'comment' in opts:  chk16 = _write_gzip_header_str(self.__base, opts['comment'],  chk16)
            self.__base.write(_uint16.pack(chk16 & 0xffff))
        elif self.__method == 'zlib':
            header = 0x7800 | (((level+1)//3) << 6)
            # Make header a multiple of 31
            mod31 = header % 31
            if mod31 != 0: header += (31 - mod31)
            self.__base.write(_uint16_be.pack(header))
            windowsize = 15
        self.__base.flush()
        return zlib.compressobj(level, zlib.DEFLATED, -windowsize)
    def writable(self): return self.__writing
    def write(self, data):
        """Compress the data and write to the underlying file object. Update the checksum."""
        self.__check(True)
        if isinstance(data, memoryview): data = data.tobytes() # Convert data method if called by io.BufferedWriter
        if len(data) > 0:
            self.__size += len(data)
            self.__checksum = self.__calc_checksum(data, self.__checksum) & 0xffffffff
            self.__base.write(self.__zlib.compress(data))
            self.__offset += len(data)
        return len(data)
    def writelines(self, lines):
        for l in lines: self.write(l)
    def flush(self, full=False):
        """
        Flush the data from the compression buffer into the underlying file object. This will
        slightly decrease compression efficency. If full is True a more major flush is performed
        that will degrade compression more but does mean if the data is corrupted some
        decompression will be able to restart.
        """
        self.__check()
        if self.__writing:
            self.__base.write(self.__zlib.flush(zlib.Z_FULL_FLUSH if full else zlib.Z_SYNC_FLUSH))
            self.__base.flush()

    # Reading
    def __read_base(self, n, check_eof=True):
        """
        Read from the base file object. There may be some data in the internal buffer which will be
        used first. If check_eof is True (the default) then this will be gauranteed to return n
        bytes or raise an EOFError, otherwise it may return less than n bytes.
        """
        if n < len(self.__base_buf):
            s = self.__base_buf[:n]
            self.__base_buf = self.__base_buf[n:]
        elif len(self.__base_buf) > 0:
            s = self.__base_buf + self.__base.read(n - len(self.__base_buf))
            self.__base_buf = b''
        else:
            s = self.__base.read(n)
        if check_eof and len(s) != n:
            self.__base_buf = s
            raise EOFError
        return s
    def __skip_0s(self):
        """Skips all upcoming 0s in the data."""
        s = self.__base_buf or self.__base.read(1024)
        self.__base_buf = b''
        while len(s) > 0:
            m = _re_not_zero.search(s)
            if m is not None: # found a non-zero
                self.__base_buf = s[m.start():]
                break
            s = self.__base.read(1024) # read 1kb at a time
    def __read_more(self, n, s):
        """
        Make sure the byte string s is exactly n bytes long. If it is longer, return the trailing
        bytes to the internal buffer. If it is shorter, read some bytes using __read_base.
        """
        if len(s) == n: return s
        if len(s) > n:
            self.__base_buf += s[n:]
            return s[:n]
        return s + self.__read_base(n - len(s))
    def __read_header(self):
        windowsize = zlib.MAX_WBITS
        if self.__method == 'gzip':
            self.__read_header_gzip()
        elif self.__method == 'zlib':
            header = _uint16_be.unpack(self.__read_base(2))[0]
            method = (header >>  8) & 0xF
            windowsize = ((header >> 12) & 0xF) + 8
            fdict  = (header & 0x20) != 0
            #flevel = (header >> 6) & 0x3
            #fcheck = (header & 0x1F)
            if method != 8 or windowsize > zlib.MAX_WBITS or fdict: raise IOError('Unknown compression method')
            if header % 31 != 0: raise IOError('Header corrupted')
        return zlib.decompressobj(-windowsize)
    def __read_header_gzip(self):
        if not hasattr(self, '__gzip_options'): self.__gzip_options = {}
        opts = self.__gzip_options
        header = self.__read_base(10)
        chk16 = zlib.crc32(header) & 0xffffffff
        if header[:3] != b'\x1F\x8B\x08': raise IOError('Not a gzipped file')
        flags = Byte(header[3])
        if flags & 0xE0: raise IOError('Unknown flags')
        opts['text'] = (flags & _FTEXT) != 0
        opts['mtime'] = datetime.fromtimestamp(_uint32.unpack_from(header, 4)[0])
        opts['xf'] = Byte(header[8])
        opts['os'] = Byte(header[9])
        if flags & _FEXTRA:
            # Read the extra field
            xlen = self.__read_base(2)
            chk16 = zlib.crc32(xlen, chk16) & 0xffffffff
            extras = self.__read_base(_uint16.unpack(xlen)[0])
            chk16 = zlib.crc32(extras, chk16) & 0xffffffff
            ext = opts['extras'] = OrderedDict()
            off = 0
            while off+4 <= len(extras):
                l = _uint16.unpack_from(extras, off+2)[0]
                if off+l+4 > len(extras): raise IOError('Invalid extra fields in header')
                ext[extras[off:off+2]] = extras[off+4:off+l+4]
                off += l+4
            if off != len(extras): raise IOError('Invalid extra fields in header')
        if flags & _FNAME:    opts['filename'], chk16 = _read_gzip_header_str(self.__read_base, chk16)
        if flags & _FCOMMENT: opts['comment'],  chk16 = _read_gzip_header_str(self.__read_base, chk16)
        # Read and verify the 16-bit header CRC
        if (flags & _FHCRC) and _uint16.unpack(self.__read_base(2))[0] != (chk16 & 0xffff): raise IOError('Header corrupted')

    def __read_footer(self, footer = ''):
        try:
            if self.__method == 'gzip':
                footer = self.__read_more(8, footer)
                if   _uint32.unpack_from(footer, 0)[0] != self.__checksum: raise IOError("CRC32 check failed")
                elif _uint32.unpack_from(footer, 4)[0] != (self.__size & 0xffffffff): raise IOError("Incorrect length of data produced")
            elif self.__method == 'zlib':
                footer = self.__read_more(4, footer)
                if _uint32_be.unpack(footer)[0] != self.__checksum: raise IOError("Adler32 check failed")
            else:
                self.__read_more(0, footer)
        except EOFError:
            raise IOError("Corrupt file: did not end with checksums")
        # Skip any zero-padding
        self.__skip_0s()
        self.__new_member = True

    def __read(self, size=1024):
        if self.__new_member:
            self.__checksum = self.__calc_checksum(b'') & 0xffffffff
            self.__size = 0
            try: self.__read_header()
            except EOFError:
                if self.__base_buf == b'': return (b'', True)
                raise
            self.__new_member = False
        buf = self.__read_base(size, False)
        if len(buf) == 0:
            data = self.__zlib.flush()
            self.__checksum = self.__calc_checksum(data, self.__checksum) & 0xffffffff
            self.__size += len(data)
            self.__read_footer()
            return (data, True)
        data = self.__zlib.decompress(buf)
        self.__checksum = self.__calc_checksum(data, self.__checksum) & 0xffffffff
        self.__size += len(data)
        if len(self.__zlib.unused_data) != 0:
            self.__read_footer(self.__zlib.unused_data)
        return (data, False)

    def readable(self): return not self.__writing
    def __skip(self, size):
        # Basically read/readinto without saving any of the chunks.
        # It also chooses the chunksize differently.
        if len(self.__decomp_buf) >= size:
            self.__decomp_buf = self.__decomp_buf[size:]
            self.__offset += size
            return size
        orig_size = size
        size -= len(self.__decomp_buf)
        self.__decomp_buf = b''
        eof = False
        while size > 0 and not eof:
            c, eof = self.__read(min(1048576, size+100))
            if len(c) > size:
                self.__decomp_buf = c[size:]
                size = 0
                break
            size -= len(c)
        orig_size -= size
        self.__offset += orig_size
        return orig_size
    def read(self, size=-1):
        self.__check(False)
        if size < 0: size = sys.maxsize
        elif len(self.__decomp_buf) >= size:
            if len(self.__decomp_buf) == size:
                s, self.__decomp_buf = self.__decomp_buf, b''
            else:
                s, self.__decomp_buf = self.__decomp_buf[:size], self.__decomp_buf[size:]
            self.__offset += size
            return s
        readsize = 1024
        chunks = [self.__decomp_buf]
        size -= len(self.__decomp_buf)
        self.__decomp_buf = b''
        eof = False
        while size > 0 and not eof:
            c, eof = self.__read(readsize)
            chunks.append(c)
            size -= len(c)
            readsize = min(self.max_read_chunk, readsize * 2)
        if size < 0:
            chunks[-1] = c[:size]
            self.__decomp_buf = c[size:]
        s = b''.join(chunks)
        self.__offset += len(s)
        return s
    def readinto(self, buf):
        self.__check(False)
        len_buf = len(buf)
        len_dbuf = len(self.__decomp_buf)
        if len_dbuf >= len_buf:
            if len_dbuf == len_buf:
                buf[:] = self.__decomp_buf
                self.__decomp_buf = b''
            else:
                buf[:] = self.__decomp_buf[:len_buf]
                self.__decomp_buf = self.__decomp_buf[len_buf:]
            self.__offset += len_buf
            return len_buf
        readsize = 1024
        buf[:len_dbuf] = self.__decomp_buf
        off = len_dbuf - len_buf # remaining = -off
        self.__decomp_buf = b''
        eof = False
        while off < 0 and not eof:
            c, eof = self.__read(readsize)
            if len(c) > -off:
                self.__decomp_buf = c[-off:]
                buf[off:] = c[:-off]
                off = 0
                break
            end = len(buf) if -off == len(c) else off+len(c)
            buf[off:end] = c
            off += len(c)
            readsize = min(self.max_read_chunk, readsize * 2)
        read = len_buf-off
        self.__offset += read
        return read
    # TODO: def read1(self, n=-1):
    def readline(self, size=-1):
        self.__check(False)
        if size < 0:
            size = sys.maxsize
            readsize = self.__min_readsize
        else:
            readsize = size

        # Shortcut common case - full line found in _decomp_buf
        i = self.__decomp_buf.find(b'\n', stop=size) + 1
        if i > 0 or len(self.__decomp_buf) >= size:
            if i > 0: size = i
            if len(self.__decomp_buf) == size:
                s = self.__decomp_buf
                self.__decomp_buf = b''
            else:
                s = self.__decomp_buf[:size]
                self.__decomp_buf = self.__decomp_buf[size:]
            self.__offset += size
            return s

        chunks = [self.__decomp_buf]
        size -= len(self.__decomp_buf)
        self.__decomp_buf = b''
        while size != 0:
            c = self.read(readsize) # c is at most readsize bytes
            i = c.find(b'\n') + 1

            # We set i=size to break out of the loop under two
            # conditions: 1) there's no newline, and the chunk is
            # larger than size, or 2) there is a newline, but the
            # resulting line would be longer than 'size'.
            if (size < i) or (i == 0 and len(c) >= size): i = size

            if i > 0 or c == b'':
                chunks.append(c[:i])     # Add portion of last chunk
                self.__decomp_buf = c[i:] # Push back rest of chunk
                self.__offset -= len(self.__decomp_buf)
                break

            # Append chunk to list and decrease 'size'
            chunks.append(c)
            size -= len(c)
            readsize = min(size, readsize * 2)

        if readsize > self.__min_readsize:
            self.__min_readsize = min(readsize, self.__min_readsize * 2, 512)
        return b''.join(chunks) # Return resulting line
    def readlines(self, hint=-1):
        if hint < -1:
            data = self.read(-1)
            prev = 0
            i = data.find(b'\n')
            v = memoryview(data)
            lines = []
            while i >= 0:
                lines.append(v[prev:i+1])
                prev = i+1
                i = data.find(b'\n', prev)
            last = v[prev:]
            if len(last) > 0: lines.append(last)
        else:
            while hint > 0:
                line = self.readline()
                hint -= len(line)
                lines.append(line)
    def __iter__(self):
        while True:
            line = self.readline()
            if len(line) == 0: break
            yield line


##if __name__ == '__main__':
##    # Act like gzip; with -d, act like gunzip.
##    # The input file is not deleted nor are any other gzip options or features supported.
##    args = sys.argv[1:]
##    func = compress_file
##    if args and (args[0] == '-d' or args[0] == '--decompress'):
##        args = args[1:]
##        func = decompress_file
##    if not args: args = ["-"]
##    for arg in args:
##        if arg == "-": func(sys.stdin, sys.stdout)
##        else:          func(arg)
##    gzdata = (b'\x1f\x8b\x08\x04\xb2\x17cQ\x02\xff'
##                   b'\x09\x00XX\x05\x00Extra'
##                   b'\x0bI-.\x01\x002\xd1Mx\x04\x00\x00\x00')
