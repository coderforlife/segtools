"""
Functions that read and write gzipped files or data.

The user of the file doesn't have to worry about the compression, but random access is not allowed.

The Python gzip module was used for some inspiration, particularly with reading files (read and
readline are nearly verbatim from it).

Additions over the default Python gzip module:
    * Supports pure deflate and zlib data in addition to gzip files
    * Supports modifying and retrieving all the gzip header properties
    * Adds and checks header checksums for increased file integrity
    * Allows you to read embeded compressed data with rewind and negative seek support when you
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

import io, os, sys, struct
from time import time
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
_default_gzip_oses = {
    'nt' : 0, 'os2' : 0, 'ce' : 0, # FAT (NT could also be NTFS or HPFS and OS/2 could be HPFS)
    'posix' : 3, # UNIX
    'riscos' : 13, # Acorn RISCOS
    }
default_gzip_os = _default_gzip_oses.get(os.name, 255) # default is unknown, including for 'java'
_exts = {
    'gzip' : '.gz',
    'zlib' : '.zlib',
    'deflate' : '.deflate',
    }

String = str if sys.version_info[0] == 3 else basestring
def _pack(fmt, *v): return struct.pack(str(fmt), *v)
def _unpack(fmt, b): return struct.unpack(str(fmt), b)
def _unpack1(fmt, b): return struct.unpack(str(fmt), b)[0]
def _get_filename(f, default=None):
    if isinstance(f, String): return f
    elif hasattr(f, 'name') and (len(f.name) < 2 or f.name[0] != '<' and f.name[-1] != '>'):
        return f.name
    return default
def _gzip_header_str(s):
    if not s: return None
    i = s.find(b'\x00')
    if i >= 0: s = s[:i]
    s = s.encode('iso-8859-1')
    return s + b'\x00' if s else None
def _write_gzip_header_str(f, s, chk16):
    f.write(s)
    return zlib.crc32(s, chk16) & 0xffffffff
def _read_gzip_header_str(read, chk16):
    s = b''
    while True:
        c = read(1)
        if not c or c == b'\x00': break
        s += c
    return s.decode('iso-8859-1'), (zlib.crc32(s+b'\x00', chk16) & 0xffffffff)


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
# CRC16 checksum of header
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
    except OSError: pass
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
    if method == 'gzip' or method is None:
        xf = 2 if level >= 7 else (4 if level <= 2 else 0)
        s = b'\x1F\x8B\x08\x02' + _pack('<LB', int(time()), xf) + b'\xFF'
        s += _pack('<H', zlib.crc32(s) & 0xffff)
        s += zlib.compress(inpt, level)
        s += _pack('<LL', zlib.crc32(inpt) & 0xffffffff, len(inpt) & 0xffffffff)
        return s
    elif method == 'zlib':
        header = 0x7800 + (((level+1)//3) << 6)
        mod31 = header % 31
        if mod31 != 0: header += (31 - mod31)
        s += _pack('>H', header)
        s += zlib.compress(inpt, level)
        s += _pack('<L', zlib.adler32(inpt) & 0xffffffff)
        return s
    elif method == 'deflate':
        return zlib.compress(inpt, level)
    else:
        raise ValueError('Compression method must be one of deflate, zlib, or gzip')

def decompress(inpt, method=None):
    if method is None: method = guess_compression_method(inpt)
    if method == 'gzip': return __decompress_gzip(inpt)
    elif method == 'zlib': return __decompress_zlib(inpt)
    elif method == 'deflate': return zlib.decompress(inpt)
    else: raise ValueError('Compression method must be one of deflate, zlib, gzip, or None')
def __decompress_gzip(inpt):
    magic1, magic2, method, flags = _unpack('<BBBB', inpt[:4])
    if magic1 != 0x1F or magic2 != 0x8B: raise IOError('Not a gzipped file')
    if method != 8: raise IOError('Unknown compression method')
    if flags & 0xE0: raise IOError('Unknown flags')
    off = _unpack1('<H', inpt[10:12]) + 12 if flags & _FEXTRA else 10
    if flags & _FNAME:    off = inpt.index(b'\x00', off) + 1
    if flags & _FCOMMENT: off = inpt.index(b'\x00', off) + 1
    if flags & _FHCRC:
        if _unpack1('<H', inpt[off:off+2]) != (zlib.crc32(inpt[:off]) & 0xffff): raise IOError('Header corrupted')
        off += 2
    crc32, isize = _unpack('<II', inpt[-8:])
    s = zlib.decompress(inpt[off:-8], -zlib.MAX_WBITS, isize)
    checksum = zlib.crc32(s)
    if crc32 != checksum: raise IOError("CRC32 check failed %08x != %08x" % (crc32, checksum))
    if isize != (len(s) & 0xffffffff): raise IOError("Incorrect length of data produced")
    return s
def __decompress_zlib(inpt):
    header = _unpack1('>H', inpt[:2])
    method = (header >>  8) & 0xF
    windowsize = ((header >> 12) & 0xF) + 8
    fdict  = (header & 0x20) != 0
    if method != 8 or windowsize > zlib.MAX_WBITS or fdict: raise IOError('Unknown compression method')
    if header % 31 != 0: raise IOError('Header corrupted')
    s = zlib.decompress(inpt[2:-4], -windowsize)
    a32 = _unpack1('>I', inpt[-4:])
    checksum = zlib.adler32(s)
    if a32 != checksum: raise IOError("Adler32 check failed %08x != %08x" % (a32, checksum))
    return s

def guess_file_compression_method(f):
    if isinstance(f, String):
        with io.open(f, 'rb') as f: return guess_compression_method(f.read(3))
    else: return guess_compression_method(f.read(3))

def guess_compression_method(buf):
    if len(buf) > 2 and buf[0:2] == b'\x1F\x8B\x08': return 'gzip' # could also check flags and checksum, but this seems good enough
    if len(buf) > 1:
        h = _unpack1('>H', buf[:2])
        if (h&0x88) == 0x08 and h%31 == 0: return 'zlib' # about a 1/1000 chance of guessing zlib when actually deflate
    return 'deflate'

class GzipFile(io.BufferedIOBase):
    __offset = 0
    max_read_chunk = 10 * 1024 * 1024 # 10Mb

    def __init__(self, file, mode=None, level=9, method=None, start_off=None, **kwargs):
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

        When writing gzip data you can include extra information with the following keyword arguments:
            os= to an integer from 0 to 255 that describes the filesystem where the file orginated (default depends on system)
            mtime= to an integer representing the modification time of the original file as a UNIX timestamp (default is now)
            text=True if the data being written is text (default is binary)
            filename= to the original filename that is being compressed (default is filename without .gz if obtainable)
            comment= to a user-readable comment
            extras= to a list of 2-element tuples, each has a 2 byte string for the subfield id and a byte string for the subfield data

        When reading gzip data the extra information is available from the gzip_options property.
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
        if method == 'gzip' and writing:
            self.gzip_options = GzipFile.__check_gzip_opts(kwargs, self.name)
        elif kwargs: raise ValueError('Extra keyword arguments can only be provided when writing gzip data')
        self.__mode = mode
        self.__writing = writing
        self.__start_off = 0 if start_off is None else start_off
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
            if 'a' in mode: raise ValueError('Starting offset not supported in append mode') # 'a' in mode
            if start_off < 0: raise ValueError('Starting offset cannot be negative')
        return start_off

    @staticmethod
    def __check_gzip_opts(kwargs, filename):
        if len(kwargs.viewkeys()-{'text','os','comment','filename','mtime','extras'}):
            raise ValueError('Gzip options must only include text, comment, filename, mtime, and extras')
        is_text = 'text' in kwargs and kwargs['text']
        gzip_os = int(kwargs.get('os', default_gzip_os))
        if gzip_os > 255 or gzip_os < 0: raise ValueError('Gzip OS is an invalid value')
        filename = filename[:-3] if filename and filename.endswith('.gz') else ''
        filename = _gzip_header_str(kwargs.get('filename', filename))
        comment = _gzip_header_str(kwargs.get('comment',  ''))
        mtime = int(kwargs.get('mtime', time()))
        extras = kwargs.get('extras')
        if extras and any(len(ex_id) != 2 for ex_id, data in extras): raise ValueError('Gzip extras had a subfield id that was not 2 characters long')
        return {
            'os':gzip_os, 'mtime':mtime, 'text':is_text,
            'filename':filename, 'comment':comment, 'extras':extras
            }

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
            del self.__zlib
            if self.__method == 'gzip':   self.__base.write(_pack('<LL', self.__checksum, self.__size & 0xffffffff))
            elif self.__method == 'zlib': self.__base.write(_pack('>L', self.__checksum))
            del self.__calc_checksum
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
        self.__check()
        return self.__offset
    def rewind(self):
        """Return the uncompressed stream file position indicator to the beginning of the file"""
        self.__check(False)
        self.__base.seek(self.__start_off)
        self.__new_member = True
        self.__decomp_buf = b''
        self.__offset = 0
    def seek(self, offset, whence=None):
        self.__check(False)
        if whence:
            if whence == io.SEEK_CUR:
                offset += self.__offset
            else: raise IOError('Seek from end not supported') # whence == io.SEEK_END
        if offset < self.__offset: self.rewind() # for negative seek, rewind and do positive seek
        self.__skip(offset - self.__offset)
        return self.__offset
    def truncate(self, size_=None): raise IOError('Truncate not supported for gzip files')

    # Writing
    def __init_writing(self, level):
        windowsize = zlib.MAX_WBITS
        if self.__method == 'gzip':
            flags = _FHCRC
            if self.gzip_options['text']:     flags |= _FTEXT
            if self.gzip_options['extras']:   flags |= _FEXTRA
            if self.gzip_options['filename']: flags |= _FNAME
            if self.gzip_options['comment']:  flags |= _FCOMMENT
            xf = 2 if level >= 7 else (4 if level <= 2 else 0)
            s = b'\x1F\x8B\x08' + _pack('<BLBB', flags, self.gzip_options['mtime'], xf, self.gzip_options['os'])
            self.__base.write(s)
            chk16 = zlib.crc32(s) & 0xffffffff
            if self.gzip_options['extras']:
                extras = ''
                for ex_id, data in self.gzip_options['extras']:
                    extras += ex_id + _pack('<H', len(data)) + data
                extras = _pack('<H', len(extras)) + extras
                chk16 = _write_gzip_header_str(self.__base, extras, chk16)
            if self.gzip_options['filename']: chk16 = _write_gzip_header_str(self.__base, self.gzip_options['filename'], chk16)
            if self.gzip_options['comment']:  chk16 = _write_gzip_header_str(self.__base, self.gzip_options['comment'],  chk16)
            self.__base.write(_pack('<H', chk16 & 0xffff))
        elif self.__method == 'zlib':
            header = 0x7800 + (((level+1)//3) << 6)
            # Make header a multiple of 31
            mod31 = header % 31
            if mod31 != 0: header += (31 - mod31)
            self.__base.write(_pack('>H', header))
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
        if n < len(self.__base_buf):
            s = self.__base_buf[:n]
            self.__base_buf = self.__base_buf[n:]
        elif len(self.__base_buf) > 0:
            s = self.__base_buf + self.__base.read(n - len(self.__base_buf))
            self.__base_buf = b''
        else:
            s = self.__base.read(n)
        if check_eof and len(s) != n: raise EOFError
        return s
    def __peek_base(self):
        if len(self.__base_buf) == 0:
            self.__base_buf = self.__base.read(1)
        return self.__base_buf[0]
    def __read_more(self, n, s):
        if len(s) > n:
            self.__base_buf += s[n:]
            return s[:n]
        elif len(s) == n:
            return s
        return s + self.__read_base(n - len(s))
    def __read_header(self):
        windowsize = zlib.MAX_WBITS
        if self.__method == 'gzip':
            self.__read_header_gzip()
        elif self.__method == 'zlib':
            header = self.__read_base(2)
            header = _unpack1('>H', header)
            method = (header >>  8) & 0xF
            windowsize = ((header >> 12) & 0xF) + 8
            fdict  = (header & 0x20) != 0
            #flevel = (header >> 6) & 0x3
            #fcheck = (header & 0x1F)
            if method != 8 or windowsize > zlib.MAX_WBITS or fdict: raise IOError('Unknown compression method')
            if header % 31 != 0: raise IOError('Header corrupted')
        return zlib.decompressobj(-windowsize)
    def __read_header_gzip(self):
        if not hasattr(self, 'gzip_options'):
            self.gzip_options = {
                'os' : 255, 'mtime' : 0, 'text' : False,
                'filename' : None, 'comment' : None, 'extras' : None
                }
        header = self.__read_base(10)
        magic1, magic2, method, flags, mtime, _, gzip_os = _unpack('<BBBBIBB', header)
        if magic1 != 0x1F or magic2 != 0x8B: raise IOError('Not a gzipped file')
        if method != 8: raise IOError('Unknown compression method')
        if flags & 0xE0: raise IOError('Unknown flags')
        self.gzip_options['text'] = bool(flags & _FTEXT)
        self.gzip_options['os'] = gzip_os
        self.gzip_options['mtime'] = mtime
        chk16 = zlib.crc32(header) & 0xffffffff
        if flags & _FEXTRA:
            # Read the extra field
            xlen = self.__read_base(2)
            extras = self.__read_base(_unpack1('<H', xlen))
            chk16 = zlib.crc32(extras, zlib.crc32(xlen, chk16)) & 0xffffffff
            ext = []
            while len(extras) >= 4:
                l = _unpack1('<H', extras[2:4])
                if 4+l > len(extras): raise IOError('Invalid extra fields in header')
                ext.append((extras[:2], extras[4:4+l]))
                extras = extras[4+l:]
            if len(extras) > 0: raise IOError('Invalid extra fields in header')
            self.gzip_options['extras'] = ext
        if flags & _FNAME:    self.gzip_options['filename'], chk16 = _read_gzip_header_str(self.__read_base, chk16)
        if flags & _FCOMMENT: self.gzip_options['comment'],  chk16 = _read_gzip_header_str(self.__read_base, chk16)
        # Read and verify the 16-bit header CRC
        chk16_ = _unpack1('<H', self.__read_base(2))
        if (flags & _FHCRC) and chk16_ != (chk16 & 0xffff): raise IOError('Header corrupted')

    def __read_footer(self, footer = ''):
        try:
            if self.__method == 'gzip':
                footer = self.__read_more(8, footer)
                crc32, isize = _unpack('<II', footer)
                if crc32 != self.__checksum:
                    raise IOError("CRC32 check failed %08x != %08x" % (crc32, self.__checksum))
                elif isize != (self.__size & 0xffffffff):
                    raise IOError("Incorrect length of data produced")
            elif self.__method == 'zlib':
                footer = self.__read_more(4, footer)
                a32 = _unpack1('>I', footer)
                if a32 != self.__checksum:
                    raise IOError("Adler32 check failed %08x != %08x" % (a32, self.__checksum))
            else:
                self.__read_more(0, footer)
        except EOFError:
            raise IOError("Corrupt file: did not end with checksums")
        # Skip any zero-padding
        while self.__peek_base() == b'\x00': self.__read_base(1)
        self.__new_member = True

    def __read(self, size=1024):
        if self.__new_member:
            self.__checksum = self.__calc_checksum(b'') & 0xffffffff
            self.__size = 0
            self.__read_header()
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
            buf[off:off+len(c)] = c
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
