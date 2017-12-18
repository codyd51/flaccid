from typing import List
from ctypes import sizeof, Structure, BigEndianStructure, c_uint8, c_uint16, c_uint32, c_uint64
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


class FlacParser(object):
    def __init__(self, flac_file):
        # type: (file) -> None
        self.flac = flac_file
        self.parse_magic()

        self.stream_info = None

        # FLAC guarantees at least one metadata block; the stream info block
        self.metadata_blocks = []
        while True:
            metadata_block = self.parse_metadata_block()
            self.metadata_blocks.append(metadata_block)

            if metadata_block.header.is_last_block:
                break
        print('Finished parsing {} metadata blocks'.format(len(self.metadata_blocks)))

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

    def parse_data_for_metadata_header(self, header):
        # type: (MetadataBlockHeader) -> MetadataBlockData
        if header.block_type == MetadataBlockType.STREAMINFO.value:
            data_type = MetadataBlockStreamInfo
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


