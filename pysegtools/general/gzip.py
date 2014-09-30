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
        * fourth argument is now 'type' for the type of output ('deflate', 'zlib', 'gzip')
        * 'mtime' is now only supported as a keyword argument when writing gzip files
        * to include a filename in the gzip header, provide it as a keyword argument
        * Overall: if using 3 or less non-keyword arguments it will work as before otherwise not
    * Undocumented properties have essentially all been removed or renamed, most notably:
        * 'fileobj' and 'myfileobj' are now 'base' ('owns_handle' determines if it is 'my' or not)
        * 'mode' is the actual file mode instead of a 1 or 2 indicating READ or WRITE
        * 'mtime' is now 'gzip_options['mtime']' when type is 'gzip' (otherwise not available)
        * deprecated 'filename' is now 'gzip_options['filename']' when type is 'gzip' (otherwise not available)
"""


from os import name as os_name
from sys import maxint
from struct import pack, unpack
from time import time
from io import open, BufferedIOBase, SEEK_CUR

from zlib import adler32, crc32
from zlib import DEFLATED, MAX_WBITS
from zlib import compress as zcompress, compressobj, Z_FINISH, Z_SYNC_FLUSH, Z_FULL_FLUSH
from zlib import decompress as zdecompress, decompressobj

__all__ = ['gzip_oses', 'default_gzip_os',
           'compress_file', 'decompress_file', 'compress', 'decompress',
           'guessfiletype', 'guesstype',
           'GzipFile']

FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT = 0x01, 0x02, 0x04, 0x08, 0x10

def no_checksum(data, value=None): return 0
checksums = {
        'gzip' : crc32,
        'zlib' : adler32,
        'deflate' : no_checksum,
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
default_gzip_oses = {
        'nt' : 0, 'os2' : 0, 'ce' : 0, # FAT (NT could also be NTFS or HPFS and OS/2 could be HPFS)
        'posix' : 3, # UNIX
        'riscos' : 13, # Acorn RISCOS
    }
default_gzip_os = default_gzip_oses.get(os_name, 255) # default is unknown, including for 'java'

def get_filename(f, default=None):
    if isinstance(f, basestring):
        return f
    elif hasattr(f, 'name') and (len(f.name) < 2 or f.name[0] != '<' and f.name[-1] != '>'):
        return f.name
    return default
def gzip_header_str(s):
    if not s: return None
    i = s.find('\x00')
    if i >= 0: s = s[:i]
    s = s.encode('iso-8859-1')
    return s + '\x00' if s else None
def write_gzip_header_str(file, s, chk16):
    file.write(s)
    return crc32(s, chk16) & 0xffffffffL
def read_gzip_header_str(read, chk16):
    s = ''
    while True:
        c = read(1)
        if not c or c == '\x00': break
        s += c
    return s.decode('iso-8859-1'), (crc32(s+'\x00', chk16) & 0xffffffffL)


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

def compress_file(input, output=None, level=9, type=None):
    # Get output filename
    in_filename = get_filename(input)
    if output == None:
        if in_filename == None: raise ValueError('Unable to determine output filename')
        output = in_filename + ('.gz' if type=='gzip' or type==None else ('.zlib' if type=='zlib' else '.deflate'))

    # Get gzip options
    opts = {}
    if type == 'gzip' or type == None:
        import os
        try_fstat = True
        if in_filename:
            opts['filename'] = os.path.basename(in_filename) 
            try:
                opts['mtime'] = os.path.getmtime(in_filename)
                try_fstat = False
            except: pass
        if try_fstat:
            try:
                opts['mtime'] = os.fstat(input.fileno()).st_mtime
            except: pass

    # Copy data
    with GzipFile(output, 'wb', level, type, **opts) as output:
        owns_handle = isinstance(input, basestring)
        if owns_handle: input = open(input, 'rb')
        try:
            while True:
                buf = input.read(10*1024*1024)
                if len(buf) == 0: break
                output.write(buf)
        finally:
            if owns_handle: input.close()

def decompress_file(input, output=None, type=None):
    with GzipFile(input, 'rb', type=type) as input:
        # Get the output filename if not provided
        in_filename = get_filename(input)
        if not output:
            if input.type == 'gzip':
                output = input.gzip_options.get('filename')
                if not output and in_filename and in_filename.endswith('.gz'):
                    output = in_filename[:-3]
            elif input.type == 'zlib' and in_filename and in_filename.endswith('.zlib'):
                output = in_filename[:-5]
            elif input.type == 'deflate' and in_filename and in_filename.endswith('.deflate'):
                output = in_filename[:-8]
            if not output: raise ValueError('Unable to determine output filename')

        # Copy data
        owns_handle = isinstance(output, basestring)
        if owns_handle: output = open(output, 'wb')
        try:
            while True:
                buf = input.read(10*1024*1024)
                if len(buf) == 0: break
                output.write(buf)
        finally:
            if owns_handle: output.close()

        # Set mtime on output file if it is available
        if in_filename and input.type == 'gzip' and input.gzip_options['mtime']:
            import os

def compress(input, level=9, type=None):
    level = int(level)
    if type == 'gzip' or type == None:
        xf = 2 if level >= 7 else (4 if level <= 2 else 0)
        s = b'\x1F\x8B\x08\x02' + pack('<LB', int(time()), xf) + b'\xFF'
        s += pack('<H', crc32(s) & 0xffff)
        s += zcompress(input, level)
        s += pack('<LL', crc32(input) & 0xffffffffL, len(input) & 0xffffffffL)
        return s
    elif type == 'zlib':
        header = 0x7800 + (((level+1)//3) << 6)
        mod31 = header % 31
        if mod31 != 0: header += (31 - mod31)
        s += pack('>H', header)
        s += zcompress(input, level)
        s += pack('<L', adler32(input) & 0xffffffffL)
        return s
    elif type == 'deflate':
        return zcompress(input, level)
    else:
        raise ValueError('Compression type must be one of deflate, zlib, or gzip')

def decompress(input, type=None):
    if type == None: type = guesstype(input)
    if type == 'gzip':
        magic1, magic2, method, flags, mtime, xf, os = unpack('<BBBBIBB', input[:10])
        if magic1 != 0x1F or magic2 != 0x8B: raise IOError('Not a gzipped file')
        if method != 8: raise IOError('Unknown compression method')
        if flags & 0xE0: raise IOError('Unknown flags')
        off = unpack('<H', input[10:12])[0] + 12 if flags & FEXTRA else 10
        if flag & FNAME:    off = input.index('\x00', off) + 1
        if flag & FCOMMENT: off = input.index('\x00', off) + 1
        if flags & FHCRC:
            if unpack('<H', input[off:off+2])[0] != (crc32(input[:off]) & 0xffff): raise IOError('Header corrupted')
            off += 2
        crc32, isize = unpack('<II', input[-8:])
        s = zdecompress(input[off:-8], -MAX_WBITS, isize)
        checksum = crc32(s)
        if crc32 != checksum: raise IOError("CRC32 check failed %08x != %08x" % (crc32, checksum))
        if isize != (len(s) & 0xffffffffL): raise IOError("Incorrect length of data produced")
        return s
    elif type == 'zlib':
        header = unpack('>H', input[:2])[0]
        method = (header >>  8) & 0xF
        windowsize = ((header >> 12) & 0xF) + 8
        fdict  = (header & 0x20) != 0
        if method != 8 or windowsize > MAX_WBITS or fdict: raise IOError('Unknown compression method')
        if header % 31 != 0: raise IOError('Header corrupted')
        s = zdecompress(input[2:-4], -windowsize)
        a32 = unpack('>I', input[-4:])[0]
        checksum = adler32(s)
        if a32 != checksum: raise IOError("Adler32 check failed %08x != %08x" % (a32, checksum))
        return s
    elif type == 'deflate':
        return zdecompress(input)
    else:
        raise ValueError('Compression type must be one of deflate, zlib, gzip, or None')

def guessfiletype(f):
    if isinstance(f, basestring):
        with open(f, 'rb') as f: return guesstype(f.read(3))
    else: return guesstype(f.read(3))

def guesstype(buf):
    if len(buf) > 2 and ord(buf[0]) == 0x1F and ord(buf[1]) == 0x8B and ord(buf[2]) == 0x08: return 'gzip' # could also check flags and checksum, but this seems good enough
    elif len(buf) > 1 and (ord(buf[0]) & 0xF) == 0x8 and ((ord(buf[0]) >> 4) & 0xF) <= 0x7 and (ord(buf[0]) * 0xFF + ord(buf[1])) % 31 == 0: return 'zlib' # about a 1/1000 chance of guessing zlib when actually deflate
    else: return 'deflate'

class GzipFile(BufferedIOBase):
    offset = 0
    max_read_chunk = 10 * 1024 * 1024 # 10Mb

    def __init__(self, file, mode=None, level=9, type=None, start_off=None, **kwargs):
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

        The type must be 'deflate' (RFC 1951), 'zlib' (RFC 1950), or 'gzip' (RFC 1952). Default is
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

        is_filename = isinstance(file, basestring)

        # Check mode
        if not mode: mode = file.mode if not is_filename and hasattr(file, 'mode') else 'rb'
        if any(c not in 'rwa+btU' for c in mode) or sum(mode.count(c) for c in 'rwa') != 1 or sum(mode.count(c) for c in 'bt') > 1: raise ValueError('Mode contains invalid characters')
        if 'b' not in mode: raise ValueError('Text mode not supported')
        if is_filename and '+' in mode: raise ValueError('Read/write mode not supported')
        mode = ('r' if 'r' in mode else 'w' if 'w' in mode else 'a') + 'b' # normalized, missing +
        writing = 'r' not in mode

        # Check type
        if type not in ('deflate', 'zlib', 'gzip', None): raise ValueError('Compression type must be one of deflate, zlib, or gzip (or None if reading)')
        if type == None and writing: type = 'gzip'

        # Check level
        level = int(level)
        if level < 0 or level > 9: raise ValueError('Compression level must be between 0 and 9 (inclusive)')

        # Check off
        start_off = start_off if start_off == None else int(start_off)
        if start_off != None:
            if 'a' in mode: raise ValueError('Starting offset not supported in append mode') # 'a' in mode
            if start_off < 0: raise ValueError('Starting offset cannot be negative')

        # Check kwargs
        if type == 'gzip' and writing:
            if len(kwargs.viewkeys() - {'text', 'os', 'comment', 'filename', 'mtime', 'extras'}): raise ValueError('Gzip settings must only include text, comment, filename, mtime, and extras')
            is_text = 'text' in kwargs and kwargs['text']
            os = int(kwargs.get('os', default_gzip_os))
            if os > 255 or os < 0: raise ValueError('Gzip OS is an invalid value')
            filename = get_filename(file)
            filename = filename[:-3] if filename and filename.endswith('.gz') else ''
            filename = gzip_header_str(kwargs.get('filename', filename))
            comment  = gzip_header_str(kwargs.get('comment',  ''))
            mtime    = int(kwargs.get('mtime', time()))
            extras = kwargs.get('extras')
            if extras and any(len(id) != 2 for id, data in extras): raise ValueError('Gzip extras had a subfield id that was not 2 characters long')
            self.gzip_options = {
                    'os' : os, 'mtime' : mtime, 'text' : is_text,
                    'filename' : filename, 'comment' : comment, 'extras' : extras
                }
        elif kwargs: raise ValueError('Extra keyword arguments can only be provided when writing gzip data')

        # Setup properties
        self.owns_handle = is_filename
        self.mode = mode
        self._writing = writing
        self._start_off = 0 if start_off == None else start_off
        if is_filename:
            if start_off == None:
                file = open(file, mode)
            elif writing:
                open(file, 'ab').close() # forces file to exist (but does not truncate existing file)
                file = open(file, 'r+b') # open file in read+write mode so we don't truncate file at 0
                file.truncate(start_off)
                file.seek(start_off)
            else:
                file = open(file, 'rb')
                file.seek(start_off)
        elif start_off == None:
            if 'a' not in mode:
                try: self._start_off = file.tell()
                except: pass
        else:
            file.seek(start_off)
        self.base = file
        if type == None:
            self._base_buf = self.base.read(3)
            type = guesstype(self._base_buf)
        elif 'r' in mode:
            self._base_buf = b''
        self.type = type
        self._calc_checksum = checksums[self.type]
        self.name = get_filename(self.base, '')

        # Initialize based on reading or writing
        if self._writing:
            self._init_writing(level)
        else:
            self._init_reading()

    def _check(self, writing=None):
        """Raises an IOError if the underlying file object has been closed."""
        if self.closed: raise IOError('I/O operation on closed file.')
        if writing != None and self._writing != writing:
            raise IOError('Cannot write to read-only file' if writing else 'Cannot read, seek, or rewind a write-only file')

    # Close
    @property
    def closed(self): return self.base is None
    def close(self):
        """
        If writing, completely flush the compressor and output the checksum if the format has it.
        Always close the file. If called more than once, subsequent calls are no-op.
        """
        if self.closed: return
        if self._writing:
            self.base.write(self.compressor.flush(Z_FINISH))
            del self.compressor
            if self.type == 'gzip':   self.base.write(pack('<LL', self.checksum, self.size & 0xffffffffL))
            elif self.type == 'zlib': self.base.write(pack('>L', self.checksum))
            del self._calc_checksum
            self.base.flush()
        if self.owns_handle: self.base.close()
        self.base = None
    def detatch(self):
        # TODO: should this flush like close()?
        base = self.base
        self.base = None
        self.closed = True
        return base

    # Random Properties
    @property
    def raw(self): return self.base
    def fileno(self): raise IOError('gzip file does not have a file number, try this.base.fileno()')
    def isatty(self):
        """Returns True if the underlying file object is interactive."""
        return self.base.isatty()
    def __repr__(self):
        return '<gzip ' + repr(self.base) + ' at ' + hex(id(self)) + '>'
    @property
    def internal_state(self):
        if self._writing: return (self.base.tell(), self.checksum, self.size, self.offset, self.compressor.copy())
        else:             return (self.base.tell(), self.checksum, self.size, self.offset, self.decompressor.copy(), self._new_member, self._base_buf, self._decomp_buf)
    @internal_state.setter
    def internal_state(self, value):
        self.base.seek(value[0])
        self.checksum = value[1]
        self.size     = value[2]
        self.offset   = value[3]
        if self._writing:
            self.compressor   = value[4].copy()
            self.base.truncate(value[0])
        else:
            self.decompressor = value[4].copy()
            self._new_member  = value[5]
            self._base_buf    = value[6]
            self._decomp_buf  = value[7]

    # Position
    def seekable(self): return not self._writing
    def tell(self):
        self._check()
        return self.offset
    def rewind(self):
        """Return the uncompressed stream file position indicator to the beginning of the file"""
        self._check(False)
        self.base.seek(self._start_off)
        self._new_member = True
        self._decomp_buf = b''
        self.offset = 0
    def seek(self, offset, whence=None):
        self._check(False)
        if whence:
            if whence == SEEK_CUR: offset += self.offset
            else: raise IOError('Seek from end not supported') # whence == SEEK_END
        if offset < self.offset: self.rewind() # for negative seek, rewind and do positive seek
        self._skip(offset - self.offset)
        return self.offset
    def truncate(self, size=None): raise IOError('Truncate not supported for gzip files')

    # Writing
    def _init_writing(self, level):
        self.checksum = self._calc_checksum(b'') & 0xffffffffL
        self.size = 0
        windowsize = MAX_WBITS
        if self.type == 'gzip':
            flags = FHCRC
            if self.gzip_options['text']:     flags |= FTEXT
            if self.gzip_options['extras']:   flags |= FEXTRA
            if self.gzip_options['filename']: flags |= FNAME
            if self.gzip_options['comment']:  flags |= FCOMMENT
            xf = 2 if level >= 7 else (4 if level <= 2 else 0)
            s = b'\x1F\x8B\x08' + pack('<BLBB', flags, self.gzip_options['mtime'], xf, self.gzip_options['os'])
            self.base.write(s)
            chk16 = crc32(s) & 0xffffffffL
            if self.gzip_options['extras']:
                extras = ''
                for id, data in self.gzip_options['extras']:
                    extras += id + pack('<H', len(data)) + data
                extras = pack('<H', len(extras)) + extras
                chk16 = write_gzip_header_str(self.base, extras, chk16)
            if self.gzip_options['filename']: chk16 = write_gzip_header_str(self.base, self.gzip_options['filename'], chk16)
            if self.gzip_options['comment']:  chk16 = write_gzip_header_str(self.base, self.gzip_options['comment'],  chk16)
            self.base.write(pack('<H', chk16 & 0xffff))
        elif self.type == 'zlib':
            header = 0x7800 + (((level+1)//3) << 6)
            # Make header a multiple of 31
            mod31 = header % 31
            if mod31 != 0: header += (31 - mod31)
            self.base.write(pack('>H', header))
            windowsize = 15
        self.base.flush()
        self.compressor = compressobj(level, DEFLATED, -windowsize)
    def writable(self): return self._writing
    def write(self, data):
        """Compress the data and write to the underlying file object. Update the checksum."""
        self._check(True)
        if isinstance(data, memoryview): data = data.tobytes() # Convert data type if called by io.BufferedWriter
        if len(data) > 0:
            self.size += len(data)
            self.checksum = self._calc_checksum(data, self.checksum) & 0xffffffffL
            self.base.write(self.compressor.compress(data))
            self.offset += len(data)
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
        self._check()
        if self._writing:
            self.base.write(self.compressor.flush(Z_FULL_FLUSH if full else Z_SYNC_FLUSH))
            self.base.flush()

    # Reading
    def _init_reading(self):
        self._read_header()
        self._decomp_buf = b'' # data that has been decompressed but not officially read
        self._min_readsize = 100 # Starts small, scales exponentially
        
    def _read_base(self, n, check_eof = True):
        if n < len(self._base_buf):
            s = self._base_buf[:n]
            self._base_buf = self._base_buf[n:]
        elif len(self._base_buf) > 0:
            s = self._base_buf + self.base.read(n - len(self._base_buf))
            self._base_buf = b''
        else:
            s = self.base.read(n)
        if check_eof and len(s) != n: raise EOFError
        return s
    def _peek_base(self):
        if len(self._base_buf) == 0: self._base_buf = self.base.read(1)
        return self._base_buf[0]
    def _read_more(self, n, s):
        if len(s) > n:
            self._base_buf += s[n:]
            return s[:n]
        elif len(s) == n: return s
        return s + self._read_base(n - len(s))
    def _read_header(self):
        self.checksum = self._calc_checksum(b'') & 0xffffffffL
        self.size = 0
        windowsize = MAX_WBITS
        if self.type == 'gzip':
            if not hasattr(self, 'gzip_options'):
                self.gzip_options = {
                        'os' : 255, 'mtime' : 0, 'text' : False, 
                        'filename' : None, 'comment' : None, 'extras' : None
                    }
            header = self._read_base(10)
            magic1, magic2, method, flags, mtime, xf, os = unpack('<BBBBIBB', header)
            if magic1 != 0x1F or magic2 != 0x8B: raise IOError('Not a gzipped file')
            if method != 8: raise IOError('Unknown compression method')
            if flags & 0xE0: raise IOError('Unknown flags')
            self.gzip_options['text'] = bool(flags & FTEXT)
            self.gzip_options['os'] = os
            self.gzip_options['mtime'] = mtime
            chk16 = crc32(header) & 0xffffffffL
            if flags & FEXTRA:
                # Read the extra field
                xlen = self._read_base(2)
                extras = self._read_base(unpack('<H', xlen)[0])
                chk16 = crc32(extras, crc32(xlen, chk16)) & 0xffffffffL
                ext = []
                while len(extras) >= 4:
                    l = unpack('<H', extras[2:4])[0]
                    if 4+l > len(extras): raise IOError('Invalid extra fields in header')
                    ext.append((extras[:2], extras[4:4+l]))
                    extras = extras[4+l:]
                if len(extras) > 0: raise IOError('Invalid extra fields in header')
                self.gzip_options['extras'] = ext
            if flags & FNAME:    self.gzip_options['filename'], chk16 = read_gzip_header_str(self._read_base, chk16)
            if flags & FCOMMENT: self.gzip_options['comment'],  chk16 = read_gzip_header_str(self._read_base, chk16)
            # Read and verify the 16-bit header CRC
            chk16_ = unpack('<H', self._read_base(2))[0]
            if (flags & FHCRC) and chk16_ != (chk16 & 0xffff): raise IOError('Header corrupted')
        elif self.type == 'zlib':
            header = self._read_base(2)
            header = unpack('>H', header)[0]
            method = (header >>  8) & 0xF
            windowsize = ((header >> 12) & 0xF) + 8
            fdict  = (header & 0x20) != 0
            #flevel = (header >>  6) & 0x3
            #fcheck = (header & 0x1F)
            if method != 8 or windowsize > MAX_WBITS or fdict: raise IOError('Unknown compression method')
            if header % 31 != 0: raise IOError('Header corrupted')
        self.decompressor = decompressobj(-windowsize)
        self._new_member = False
    def _read_footer(self, footer = ''):
        try:
            if self.type == 'gzip':
                footer = self._read_more(8, footer)
                crc32, isize = unpack('<II', footer)
                if crc32 != self.checksum:
                    raise IOError("CRC32 check failed %08x != %08x" % (crc32, self.checksum))
                elif isize != (self.size & 0xffffffffL):
                    raise IOError("Incorrect length of data produced")
            elif self.type == 'zlib':
                footer = self._read_more(4, footer)
                a32 = unpack('>I', footer)[0]
                if a32 != self.checksum:
                    raise IOError("Adler32 check failed %08x != %08x" % (a32, self.checksum))
            else: self._read_more(0, footer)
        except EOFError: raise IOError("Corrupt file: did not end with checksums")
        # Skip any zero-padding
        while self._peek_base() == b'\x00': self._read_base(1)
        self._new_member = True

    def _read(self, size=1024):
        if self._new_member: self._read_header()
        buf = self._read_base(size, False)
        if len(buf) == 0:
            data = self.decompressor.flush()
            self.checksum = self._calc_checksum(data, self.checksum) & 0xffffffffL
            self.size += len(data)
            self._read_footer()
            return (data, True)
        data = self.decompressor.decompress(buf)
        self.checksum = self._calc_checksum(data, self.checksum) & 0xffffffffL
        self.size += len(data)
        if len(self.decompressor.unused_data) != 0:
            self._read_footer(self.decompressor.unused_data)
        return (data, False)

    def readable(self): return not self._writing
    def _skip(self, size):
        # Basically read/readinto without saving any of the chunks.
        # It also chooses the chunksize differently.
        if len(self._decomp_buf) >= size:
            self._decomp_buf = self._decomp_buf[size:]
            self.offset += size
            return size
        orig_size = size
        size -= len(self._decomp_buf)
        self._decomp_buf = b''
        eof = False
        while size > 0 and not eof:
            c, eof = self._read(min(1048576, size+100))
            if len(c) > size:
                self._decomp_buf = c[size:]
                size = 0
                break
            size -= len(c)
        orig_size -= size
        self.offset += orig_size
        return orig_size
    def read(self, size=-1):
        self._check(False)
        if size < 0: size = maxint
        elif len(self._decomp_buf) >= size:
            if len(self._decomp_buf) == size:
                s = self._decomp_buf
                self._decomp_buf = b''
            else:
                s = self._decomp_buf[:size]
                self._decomp_buf = self._decomp_buf[size:]
            self.offset += size
            return s
        readsize = 1024
        chunks = [self._decomp_buf]
        size -= len(self._decomp_buf)
        self._decomp_buf = b''
        eof = False
        while size > 0 and not eof:
            c, eof = self._read(readsize)
            chunks.append(c)
            size -= len(c)
            readsize = min(self.max_read_chunk, readsize * 2)
        if size < 0:
            chunks[-1] = c[:size]
            self._decomp_buf = c[size:]
        s = b''.join(chunks)
        self.offset += len(s)
        return s
    def readinto(self, buf):
        self._check(False)
        len_buf = len(buf)
        len_dbuf = len(self._decomp_buf)
        if len_dbuf >= len_buf:
            if len_dbuf == len_buf:
                buf[:] = self._decomp_buf
                self._decomp_buf = b''
            else:
                buf[:] = self._decomp_buf[:len_buf]
                self._decomp_buf = self._decomp_buf[len_buf:]
            self.offset += len_buf
            return len_buf
        readsize = 1024
        buf[:len_dbuf] = self._decomp_buf
        off = len_dbuf - len_buf # remaining = -off
        self._decomp_buf = b''
        eof = False
        while off < 0 and not eof:
            c, eof = self._read(readsize)
            if len(c) > -off:
                self._decomp_buf = c[-off:]
                buf[off:] = c[:-off]
                off = 0
                break
            buf[off:off+len(c)] = c
            off += len(c)
            readsize = min(self.max_read_chunk, readsize * 2)
        read = len_buf-off
        self.offset += read
        return read
    # TODO: def read1(self, n=-1):
    def readline(self, size=-1):
        self._check(False)
        if size < 0:
            size = maxint
            readsize = self._min_readsize
        else:
            readsize = size
        
        # Shortcut common case - full line found in _decomp_buf
        i = self._decomp_buf.find(b'\n', stop=size) + 1
        if i > 0 or len(self._decomp_buf) >= size:
            if i > 0: size = i
            if len(self._decomp_buf) == size:
                s = self._decomp_buf
                self._decomp_buf = b''
            else:
                s = self._decomp_buf[:size]
                self._decomp_buf = self._decomp_buf[size:]
            self.offset += size
            return s

        chunks = [self._decomp_buf]
        size -= len(self._decomp_buf)
        self._decomp_buf = b''
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
                self._decomp_buf = c[i:] # Push back rest of chunk
                self.offset -= len(self._decomp_buf)
                break

            # Append chunk to list and decrease 'size'
            chunks.append(c)
            size -= len(c)
            readsize = min(size, readsize * 2)
        
        if readsize > self._min_readsize:
            self._min_readsize = min(readsize, self._min_readsize * 2, 512)
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
                l = self.readline()
                hint -= len(l)
                lines.append(l)
    def __iter__(self):
        while True:
            l = self.readline()
            if len(l) == 0: break
            yield l


if __name__ == '__main__':
##    import sys
##    
##    # Act like gzip; with -d, act like gunzip.
##    # The input file is not deleted, however, nor are any other gzip options or features supported.
##    args = sys.argv[1:]
##    func = compress_file
##    if args and (args[0] == '-d' or args[0] == '--decompress'):
##        args = args[1:]
##        func = decompress_file
##    if not args: args = ["-"]
##    for arg in args:
##        if arg == "-": func(sys.stdin, sys.stdout)
##        else:          func(arg)
    gzdata = (b'\x1f\x8b\x08\x04\xb2\x17cQ\x02\xff'
                   b'\x09\x00XX\x05\x00Extra'
                   b'\x0bI-.\x01\x002\xd1Mx\x04\x00\x00\x00')
