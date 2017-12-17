from ctypes import sizeof
from bitfield import make_bf, c_uint, c_ulong


StreamInfoStruct = make_bf(
    'StreamInfoStruct',
    [
        ('min_block_size', c_uint, 16),
        ('max_block_size', c_uint, 16),
        ('min_frame_size', c_uint, 24),
        ('max_frame_size', c_uint, 24),
        ('sample_rate_hz', c_uint, 20),
        ('num_channels', c_uint, 3),
        ('bits_per_sample', c_uint, 5),
        #('total_sample_count', c_ulong, 36),
        #('md5_sig_low', c_uint, 32),
        #('md5_sig_high', c_uint, 32),
        #('md5_sig_high1', c_uint, 32),
        #('md5_sig_high2', c_uint, 32),
    ]
)


class FlacParser(object):
    def __init__(self, flac_file):
        # type: (file) -> None
        self.flac = flac_file
        self.parse_magic()

        self.stream_info = None
        self.parse_stream_info()

    def parse_magic(self):
        correct_magic = b'fLaC'
        magic = self.flac.read(4)
        for i, b in enumerate(magic):
            if b is not correct_magic[i]:
                print('incorrect magic byte {}, expected {}'.format(b, correct_magic[i]))
        print('FLAC magic verified')

    def parse_stream_info(self):
        stream_info_bytes = bytearray(bytes(self.flac.read(sizeof(StreamInfoStruct))))
        print(stream_info_bytes)
        self.stream_info = StreamInfoStruct.from_buffer(stream_info_bytes)

        print('FLAC smallest block size: {}'.format(self.stream_info.min_block_size))
        print('FLAC biggest block size: {}'.format(self.stream_info.max_block_size))
        print('FLAC smallest frame size: {}'.format(self.stream_info.min_frame_size))
        print('FLAC biggest frame size: {}'.format(self.stream_info.max_frame_size))
        print('FLAC sample rate (in Hz): {}'.format(self.stream_info.sample_rate_hz))
        print('FLAC channel count: {}'.format(self.stream_info.num_channels))
        print('FLAC bits per sample: {}'.format(self.stream_info.bits_per_sample))

file = 'Bombtrack.flac'
with open(file, 'rb') as flac_file:
    parser = FlacParser(flac_file)
    print('got parser {}'.format(parser))


