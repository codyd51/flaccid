from typing import List, Text, Optional
from ctypes import sizeof, LittleEndianStructure, BigEndianStructure, c_uint8, c_uint16, c_uint32, c_uint64, c_uint
from crcmod.predefined import mkPredefinedCrcFun
from enum import Enum


class MetadataBlockType(Enum):
    STREAMINFO = 0
    PADDING = 1
    APPLICATION = 2
    SEEKTABLE = 3
    VORBIS_COMMENT = 4
    CUESHEET = 5
    PICTURE = 6
    # types 7 through 126 are reserved
    INVALID = 127


class MetadataBlockHeader(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('is_last_block', c_uint32, 1),
        ('block_type', c_uint32, 7),
        ('length', c_uint32, 24),
    ]


class MetadataBlockData(BigEndianStructure):
    _pack_ = 1


# some blocks are big endian, like vorbis comment blocks
class MetadataBlockDataLittleEndian(LittleEndianStructure):
    _pack_ = 1


class MetadataBlockStreamInfo(MetadataBlockData):
    _fields_ = [
        ('min_block_size', c_uint16, 16),
        ('max_block_size', c_uint16, 16),
        ('min_frame_size', c_uint32, 24),
        ('max_frame_size', c_uint32, 24),
        ('sample_rate_hz', c_uint32, 20),
        ('num_channels', c_uint8, 3),
        ('bits_per_sample', c_uint8, 5),
        ('total_sample_count', c_uint64, 36),
        ('md5_low', c_uint64, 64),
        ('md5_high', c_uint64, 64),
    ]


class MetadataBlock(object):
    def __init__(self, header, data):
        # type: (MetadataBlockHeader, MetadataBlockData) -> None
        self.header = header
        self.data = data


class MetadataSeekPoint(MetadataBlockData):
    _fields_ = [
        ('first_sample_number', c_uint64, 64),
        ('target_offset', c_uint64, 64),
        ('target_num_samples', c_uint16, 16),
    ]


class MetadataBlockSeekTable(MetadataBlockData):
    def __init__(self, seek_points):
        # type: (List[MetadataSeekPoint]) -> None
        self.seek_points = seek_points


class MetadataBlockVorbisComment(MetadataBlockDataLittleEndian):
    def __init__(self, vendor_string, user_comments):
        # type: (Text, List[Text]) -> None
        self.vendor_string = vendor_string
        self.user_comments = user_comments


class MetadataBlockPadding(MetadataBlockData):
    def __init__(self, length):
        self.length = length


class FrameHeaderRaw(BigEndianStructure):
    SYNC_CODE = '0b11111111111110'

    _pack_ = 1
    _fields_ = [
        ('sync_code', c_uint16, 14),
        ('reserved1', c_uint8, 1),
        ('blocking_strategy', c_uint8, 1),
        ('block_size', c_uint8, 4),
        ('sample_rate', c_uint8, 4),
        ('channel', c_uint8, 4),
        ('sample_size', c_uint8, 3),
        ('reserved2', c_uint8, 1),
        ('frame_number', c_uint8, 8),
        ('crc', c_uint8, 8),
    ]


class BlockingStrategy(Enum):
    FIXED_BLOCK_SIZE = 0
    VARIABLE_BLOCK_SIZE = 1


class ChannelAssignment(Enum):
    MONO = 0b0000
    LEFT_RIGHT = 0b0001
    LEFT_RIGHT_CENTER = 0b0010

    def channel_count(self):
        channel_counts = [1, 2, 3]
        return channel_counts[self.value]


class SubframeType(Enum):
    SUBFRAME_CONSTANT = 0
    SUBFRAME_VERBATIM = 1
    SUBFRAME_LPC = 2


def access_bit(data, num):
    base = int(num/8)
    shift = num % 8
    return (data[base] & (1<<shift)) >> shift


class SubframeHeader(object):
    def __init__(self, flac_file):
        self.flac_file = flac_file
        self.fileoff = self.flac_file.tell()

        self.predictor_order = 0
        self.subframe_type = self.read_subframe_type()

        self.dump()

    def read_wasted_bits_per_sample(self):
        byte = bytearray(self.flac_file.read(1))
        flag = access_bit(byte, 0)
        if flag == 0:
            return 0
        raise NotImplementedError()
        if flag != 1:
            raise RuntimeError('wasted bps flag was something other than 0/1: {}'.format(flag))
        wasted_val = []
        while True:
            flag = c_uint8.from_buffer(bytearray(self.flac_file.read(1))).value
            wasted_val.append(flag)
            if flag == 1:
                break
        wasted_val = reversed(wasted_val)
        wasted_val_str = '0b{}'.format(''.join(wasted_val))
        wasted_val_int = int(wasted_val_str)
        print('wasted_val {} str {} int {}'.format(wasted_val, wasted_val_str, wasted_val_int))
        return wasted_val_int

    def read_subframe_type(self):
        type = c_uint8.from_buffer(bytearray(self.flac_file.read(1))).value
        print('subframe type: {}'.format(hex(type)))
        if type == 0x000000:
            return SubframeType.SUBFRAME_CONSTANT
        elif type == 0x000001:
            return SubframeType.SUBFRAME_VERBATIM
        elif 0x001000 <= type <= 0x001111:
            masked = type & 0x000111
            if masked > 4:
                raise RuntimeError('Reserved subframe type {}'.format(hex(type)))
            self.predictor_order = masked
        elif 0x010000 <= type <= 0x011111:
            raise RuntimeError('Reserved subframe type {}'.format(hex(type)))
        elif 0x100000 <= type <= 0x111111:
            masked = type & 0x011111
            self.predictor_order = masked + 1
            return SubframeType.SUBFRAME_LPC
        raise RuntimeError('Unknown subframe type {}'.format(hex(type)))

    def dump(self):
        print('Subframe header:')
        print('\tSubframe type: {}'.format(self.subframe_type.name))


class FrameHeader(object):
    def __init__(self, raw_header):
        # type: (FrameHeaderRaw) -> None
        self.raw_header = raw_header
        self.validate_header()

        self.blocking_strategy = BlockingStrategy(self.raw_header.blocking_strategy)
        self.block_size = self.read_block_size()
        self.sample_rate = self.read_sample_rate()
        self.channels = self.read_channel_assignment()
        self.sample_bit_count = self.read_sample_size()
        self.frame_number = self.raw_header.frame_number
        self.crc = self.raw_header.crc
        self.verify_crc()
        self.dump()

        # needs extra data at end of frame
        if self.blocking_strategy == BlockingStrategy.VARIABLE_BLOCK_SIZE:
            raise NotImplementedError()
        if not self.raw_header.block_size & ~0b0110:
            raise NotImplementedError()
        if not self.raw_header.sample_rate & ~0b1100:
            raise NotImplementedError()

    def verify_crc(self):
        frame_header_bytes = bytearray(self.raw_header)
        # we want to check everything up to (but not including) the CRC, which is the last byte
        bytes_to_verify = frame_header_bytes[:-1]
        correct_crc = frame_header_bytes[-1]

        crcModFunc = mkPredefinedCrcFun('crc-8')
        actual_crc = crcModFunc(bytes_to_verify)

        if correct_crc != actual_crc:
            raise RuntimeError('Mismatched CRC! Expected {}, but computed value is {}'.format(
                hex(correct_crc),
                hex(actual_crc))
            )
        print('Frame header CRC validated')

    def read_block_size(self):
        raw_block_size = self.raw_header.block_size
        if raw_block_size == int(0b0000):
            raise RuntimeError('frame_header->block_size uses reserved value')
        elif raw_block_size == int(0b0001):
            return 192
        elif int(0b0010) <= raw_block_size <= int(0b0101):
            return 576 * pow(2, raw_block_size - 2)
        elif raw_block_size == int(0b0110):
            # get 8 bit (blocksize - 1) from end of header
            raise NotImplementedError()
        elif raw_block_size == int(0b0111):
            # get 16 bit (blocksize - 1) from end of header
            raise NotImplementedError()
        else:
            return 256 * pow(2, raw_block_size - 8)

    def read_sample_rate(self):
        sample_val = self.raw_header.sample_rate
        value_map = {
            0b0001: 88200, # 88.2kHz
            0b0010: 176400, # 176.4kHz
            0b0011: 192000, # 192kHz
            0b0100: 8000, # 8kHz
            0b0101: 16000, # 16kHz
            0b0110: 22050, # 22.05kHz
            0b0111: 24000, # 24kHz
            0b1000: 32000, # 32kHz
            0b1001: 44100, # 44.1kHz
            0b1010: 48000, # 48kHz
            0b1011: 96000, # 96kHz
        }
        if sample_val in [0b0000, 0b1100, 0b1101, 0b1110]:
            # respectively:
            # get from STREAMINFO metadata block
            # get 8 bit sample rate (in kHz) from end of header
            # get 16 bit sample rate (in kHz) from end of header
            # get 16 bit sample rate (in tens of kHz) from end of header
            raise NotImplementedError()
        elif sample_val in value_map:
            return value_map[sample_val]
        raise RuntimeError('invalid sample rate')

    def read_channel_assignment(self):
        # 0000-0111 : (number of independent channels)-1
        # 1 channel: mono
        # 2 channels: left, right
        # 3 channels: left, right, center
        # 4 channels: front left, front right, back left, back right
        # 5 channels: front left, front right, front center, back/surround left, back/surround right
        # 6 channels: front left, front right, front center, LFE, back/surround left, back/surround right
        # 7 channels: front left, front right, front center, LFE, back center, side left, side right
        # 8 channels: front left, front right, front center, LFE, back left, back right, side left, side right
        # 1000 : left/side stereo: channel 0 is the left channel, channel 1 is the side(difference) channel
        # 1001 : right/side stereo: channel 0 is the side(difference) channel, channel 1 is the right channel
        # 1010 : mid/side stereo: channel 0 is the mid(average) channel, channel 1 is the side(difference) channel
        # 1011-1111 : reserved
        channel_val = self.raw_header.channel
        if channel_val < 0b1000:
            try:
                return ChannelAssignment(channel_val)
            except Exception as e:
                raise NotImplementedError()
        elif 0b1011 <= channel_val <= 0b1111:
            raise RuntimeError('reserved channel assignment')
        raise NotImplementedError()

    def read_sample_size(self):
        sample_size = self.raw_header.sample_size
        if sample_size  == 0b000:
            # get from STREAMINFO metadata block
            raise NotImplementedError()
        elif sample_size in [0b011, 0b111]:
            raise RuntimeError('reserved sample_size requested')
        sample_size_map = {
            0b001: 8,  # 8 bits per sample
            0b010: 12, # 12 bits per sample
            0b100: 16, # 16 bits per sample
            0b101: 20, # 20 bits per sample
            0b110: 24, # 24 bits per sample
        }
        return sample_size_map[sample_size]

    def validate_header(self):
        if self.raw_header.reserved1 != 0:
            raise RuntimeError('frame_header->reserved1 was not 0!')
        if self.raw_header.reserved2 != 0:
            raise RuntimeError('frame_header->reserved2 was not 0!')

    def dump(self):
        # type: () -> None
        print('Frame header:')
        print('\tBlocking strategy: {}'.format(self.blocking_strategy))
        print('\tBlock size: {}'.format(self.block_size))
        print('\tSample rate: {}kHz'.format(self.sample_rate / 1000))
        print('\tChannel info: {}'.format(self.channels))
        print('\tBits per sample: {}'.format(self.sample_bit_count))
        print('\tFrame number: {}'.format(self.frame_number))
        print('\tCRC: {}'.format(hex(self.crc)))


class FlacSubframe(object):
    def __init__(self, parser, frame):
        self.parser = parser
        self.frame = frame
        self.frame_header = frame.header
        self.flac_file = parser.flac
        self.audio_data = []

        self.header = SubframeHeader(self.flac_file)
        self.parse_audio_frame()
        print('FlacSubframe header {}'.format(self.header))

    def parse_audio_frame(self):
        if self.header.subframe_type == SubframeType.SUBFRAME_CONSTANT:
            self.read_audio_sample()
        else:
            raise NotImplementedError()

    def read_audio_sample(self):
        bits_per_sample = self.frame_header.sample_bit_count
        bytes_per_sample = round((bits_per_sample / 8) + 0.5)
        print('bits in sample: {} bytes {}'.format(hex(bits_per_sample), hex(bytes_per_sample)))
        constant_val = c_uint16.from_buffer(bytearray(self.flac_file.read(bytes_per_sample))).value
        print('bin {}'.format(bin(constant_val)))
        # trim to the actual sample bits
        # add 2 to account for '0b' prefix
        constant_val = str(bin(constant_val))[:bits_per_sample+2]
        # back to int
        constant_val = int(constant_val, 2)
        print('frame value: {}'.format(hex(constant_val)))
        self.audio_data.append(constant_val)


class FlacParser(object):
    def __init__(self, flac_file):
        # type: (file) -> None
        self.flac = flac_file
        self.parse_magic()

        # FLAC guarantees at least one metadata block; the stream info block
        self.metadata_blocks = []
        while True:
            try:
                metadata_block = self.parse_metadata_block()
                self.metadata_blocks.append(metadata_block)
            except NotImplementedError:
                print('Parsed up to unknown metadata block type, stopping here')
                break

            if metadata_block.header.is_last_block:
                break
        print('Finished parsing {} metadata blocks'.format(len(self.metadata_blocks)))

        self.stream_info = self.get_metadata_block_with_type(MetadataBlockType.STREAMINFO)
        self.seek_table = self.get_metadata_block_with_type(MetadataBlockType.SEEKTABLE)
        self.vorbis_comments = self.get_metadata_block_with_type(MetadataBlockType.VORBIS_COMMENT)

        self.dump_stream_info()
        self.dump_seek_table()
        self.dump_vorbis_comments()

        frame_header = self.parse_frame_header()
        self.dump_frame_header(frame_header)

    def parse_frame_header(self):
        # type: () -> FrameHeader
        print('frame header at {}'.format(hex(self.flac.tell())))
        raw_header_bytes = bytearray(bytes(self.flac.read(sizeof(FrameHeaderRaw))))
        raw_header = FrameHeaderRaw.from_buffer(raw_header_bytes)
        if bin(raw_header.sync_code) != FrameHeaderRaw.SYNC_CODE:
            raise RuntimeError('FrameHeader sync code was incorrect (got {})'.format(bin(raw_header.sync_code)))
        header = FrameHeader(raw_header)
        return header

    @staticmethod
    def dump_frame_header(frame_header):
        # type: (FrameHeader) -> None
        print('Frame header:')
        print('\tBlocking strategy: {}'.format(frame_header.blocking_strategy))
        print('\tBlock size: {}'.format(frame_header.block_size))
        print('\tSample rate: {}kHz'.format(frame_header.sample_rate / 1000))
        print('\tChannel info: {}'.format(frame_header.channels))
        print('\tBits per sample: {}'.format(frame_header.sample_bit_count))
        print('\tFrame number: {}'.format(frame_header.frame_number))
        print('\tCRC: {}'.format(hex(frame_header.crc)))

    def get_metadata_block_with_type(self, block_type):
        # type: (MetadataBlockType) -> Optional[MetadataBlock]
        for x in self.metadata_blocks:
            if x.header.block_type == block_type.value:
                return x
        return None

    def dump_stream_info(self):
        print('FLAC ({}) audio stream info:'.format(self.flac.name))

        print('\tSmallest block size: {}'.format(self.stream_info.data.min_block_size))
        print('\tLargest block size: {}'.format(self.stream_info.data.max_block_size))
        print('\tSmallest frame size: {}'.format(self.stream_info.data.min_frame_size))
        print('\tLargest frame size: {}'.format(self.stream_info.data.max_frame_size))
        print('\tSample rate (in Hz): {}'.format(self.stream_info.data.sample_rate_hz))
        print('\tChannel count: {}'.format(self.stream_info.data.num_channels))
        print('\tBits per sample: {}'.format(self.stream_info.data.bits_per_sample))
        print('\tTotal sample count: {}'.format(self.stream_info.data.total_sample_count))

        md5_sig = str(self.stream_info.data.md5_low) + str(self.stream_info.data.md5_high)
        print('\tMD5 signature of audio stream: {}'.format(md5_sig))

    def dump_seek_table(self):
        print('Seek table ({} entries):'.format(len(self.seek_table.data.seek_points)))
        for i, seek_point in enumerate(self.seek_table.data.seek_points):
            print('\tseek point {}:'.format(i))
            print('\t\tFirst sample number: {}'.format(seek_point.first_sample_number))
            print('\t\tTarget sample offset: {}'.format(seek_point.target_offset))
            print('\t\tTarget sample count: {}'.format(seek_point.target_num_samples))

    def dump_vorbis_comments(self):
        comments = self.vorbis_comments.data
        print('{} Vorbis comments. Vendor: {}'.format(len(comments.user_comments), comments.vendor_string))
        for comment in comments.user_comments:
            print('\t{}'.format(comment))

    def parse_magic(self):
        correct_magic = b'fLaC'
        magic = self.flac.read(4)
        for i, b in enumerate(magic):
            if b is not correct_magic[i]:
                print('incorrect magic byte {}, expected {}'.format(b, correct_magic[i]))
        print('FLAC magic verified')

    def parse_metadata_header(self):
        header_bytes = bytearray(bytes(self.flac.read(sizeof(MetadataBlockHeader))))
        header = MetadataBlockHeader.from_buffer(header_bytes)
        return header

    def read_ctype_from_file(self, ctype_type):
        raw_bytes = bytearray(bytes(self.flac.read(sizeof(ctype_type))))
        return ctype_type.from_buffer(raw_bytes)

    def parse_seektable(self, header):
        # type: (MetadataBlockHeader) -> MetadataBlockSeekTable
        if header.block_type != MetadataBlockType.SEEKTABLE.value:
            raise RuntimeError('wrong header passed to parse_seektable()')
        # each SeekPoint is 18 bytes as defined by the standard
        if sizeof(MetadataSeekPoint) != 18:
            raise RuntimeError('bad seekpoint def? not 18 bytes, {}'.format(sizeof(MetadataSeekPoint)))
        seekpoint_count = int(header.length / 18)
        seekpoints = []
        for i in range(seekpoint_count):
            seekpoint = self.read_ctype_from_file(MetadataSeekPoint)
            seekpoints.append(seekpoint)
        return MetadataBlockSeekTable(seekpoints)

    def parse_vorbis_comment(self, header):
        # type: (MetadataBlockHeader) -> MetadataBlockVorbisComment
        if header.block_type != MetadataBlockType.VORBIS_COMMENT.value:
            raise RuntimeError('wrong header passed to parse_vorbis_comment()')
        vendor_length = self.read_ctype_from_file(c_uint32).value
        vendor_string = self.flac.read(vendor_length).decode('utf-8')
        user_comment_list_len = self.read_ctype_from_file(c_uint32).value
        user_comments = []
        for i in range(user_comment_list_len):
            user_comment_len = self.read_ctype_from_file(c_uint32).value
            user_comment = self.flac.read(user_comment_len).decode('utf-8')
            user_comments.append(user_comment)
        return MetadataBlockVorbisComment(vendor_string, user_comments)

    def parse_padding(self, header):
        # type: (MetadataBlockHeader) -> MetadataBlockPadding
        if header.block_type != MetadataBlockType.PADDING.value:
            raise RuntimeError('wrong header passed to parse_padding()')
        return MetadataBlockPadding(header.length)

    def parse_data_for_metadata_header(self, header):
        # type: (MetadataBlockHeader) -> MetadataBlockData
        if header.block_type == MetadataBlockType.STREAMINFO.value:
            data_type = MetadataBlockStreamInfo
        elif header.block_type == MetadataBlockType.SEEKTABLE.value:
            return self.parse_seektable(header)
        elif header.block_type == MetadataBlockType.VORBIS_COMMENT.value:
            return self.parse_vorbis_comment(header)
        elif header.block_type == MetadataBlockType.PADDING.value:
            return self.parse_padding(header)
        else:
            raise NotImplementedError('block type {}'.format(header.block_type))
        return self.read_ctype_from_file(data_type)

    def parse_metadata_block(self):
        header = self.parse_metadata_header()
        block_end = self.flac.tell() + header.length
        print('Parsing {} metadata block ({} bytes)'.format(MetadataBlockType(header.block_type).name, header.length))
        if header.is_last_block:
            print('This is the last metadata block before audio blocks')
        data = self.parse_data_for_metadata_header(header)

        # seek to beginning of next block
        self.flac.seek(block_end)

        return MetadataBlock(header, data)

file = 'Bombtrack.flac'
with open(file, 'rb') as flac_file:
    parser = FlacParser(flac_file)


