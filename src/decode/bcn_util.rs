//! Shared data and function for the BCn format.

/// Stores the subset indexes for BC6/7 modes with 2 subsets.
///
/// Since each subset index is either 0 or 1, they are stored as the bits of
/// u16.
///
/// `fixup_index_2` is the second fixup index. The first fixup index is always 0.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Subset2Map {
    pub subset_indexes: u16,
    pub fixup_index_2: u8,
}
impl Subset2Map {
    pub const fn get_subset_index(self, pixel_index: u8) -> u8 {
        (self.subset_indexes.wrapping_shr(pixel_index as u32) & 0b1) as u8
    }
}
/// Stores the subset indexes for BC7 modes with 3 subsets.
///
/// Since each subset index is either 0, 1 or 2, they are stored as 2 bits in
/// a u32.
///
/// `fixup_index_2` and `fixup_index_3` are the second and third fixup index
/// respectively. The first fixup index is always 0.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Subset3Map {
    subset_indexes: u32,
    pub fixup_index_2: u8,
    pub fixup_index_3: u8,
}
impl Subset3Map {
    pub const fn get_subset_index(self, pixel_index: u8) -> u8 {
        (self.subset_indexes.wrapping_shr(pixel_index as u32 * 2) & 0b11) as u8
    }
}

pub(crate) const fn subset2(data: [u8; 17]) -> Subset2Map {
    let mut output_p2: u16 = 0;
    let mut fixup_index_2 = 0;

    let mut pixel_index = 0;
    let mut data_index = 0;
    while data_index < data.len() {
        let d = data[data_index];
        data_index += 1;

        if d == b'-' {
            fixup_index_2 = pixel_index;
        } else {
            let d = (d - b'0') as u32;
            assert!(d <= 1);
            output_p2 |= (d as u16) << pixel_index;
            pixel_index += 1;
        }
    }
    assert!(pixel_index == 16);
    assert!(fixup_index_2 != 0);

    let result = Subset2Map {
        subset_indexes: output_p2,
        fixup_index_2,
    };

    // the first subset index is always 0
    assert!(result.get_subset_index(0) == 0);

    result
}
pub(crate) const fn subset3(data: [u8; 18]) -> Subset3Map {
    let mut output: u32 = 0;
    let mut fixup_index_2 = 0;
    let mut fixup_index_3 = 0;

    let mut pixel_index = 0;
    let mut data_index = 0;
    while data_index < data.len() {
        let d = data[data_index];
        data_index += 1;

        if d == b'-' {
            if fixup_index_2 == 0 {
                fixup_index_2 = pixel_index;
            } else {
                fixup_index_3 = pixel_index;
            }
        } else {
            let d = (d - b'0') as u32;
            assert!(d <= 2);
            output |= d << (pixel_index * 2);
            pixel_index += 1;
        }
    }
    assert!(pixel_index == 16);
    assert!(fixup_index_2 != 0);
    assert!(fixup_index_3 != 0);

    let result = Subset3Map {
        subset_indexes: output,
        fixup_index_2,
        fixup_index_3,
    };

    // the first subset index is always 0
    assert!(result.get_subset_index(0) == 0);

    result
}

pub(crate) const PARTITION_SET_2: [Subset2Map; 64] = [
    // 0
    subset2(*b"001100110011001-1"),
    subset2(*b"000100010001000-1"),
    subset2(*b"011101110111011-1"),
    subset2(*b"000100110011011-1"),
    subset2(*b"000000010001001-1"),
    subset2(*b"001101110111111-1"),
    subset2(*b"000100110111111-1"),
    subset2(*b"000000010011011-1"),
    subset2(*b"000000000001001-1"),
    subset2(*b"001101111111111-1"),
    subset2(*b"000000010111111-1"),
    subset2(*b"000000000001011-1"),
    subset2(*b"000101111111111-1"),
    subset2(*b"000000001111111-1"),
    subset2(*b"000011111111111-1"),
    subset2(*b"000000000000111-1"),
    // 16
    subset2(*b"000010001110111-1"),
    subset2(*b"01-11000100000000"),
    subset2(*b"00000000-10001110"),
    subset2(*b"01-11001100010000"),
    subset2(*b"00-11000100000000"),
    subset2(*b"00001000-11001110"),
    subset2(*b"00000000-10001100"),
    subset2(*b"011100110011000-1"),
    subset2(*b"00-11000100010000"),
    subset2(*b"00001000-10001100"),
    subset2(*b"01-10011001100110"),
    subset2(*b"00-11011001101100"),
    subset2(*b"00010111-11101000"),
    subset2(*b"00001111-11110000"),
    subset2(*b"01-11000110001110"),
    subset2(*b"00-11100110011100"),
    // 32
    subset2(*b"010101010101010-1"),
    subset2(*b"000011110000111-1"),
    subset2(*b"010110-1001011010"),
    subset2(*b"00110011-11001100"),
    subset2(*b"00-11110000111100"),
    subset2(*b"01010101-10101010"),
    subset2(*b"011010010110100-1"),
    subset2(*b"010110101010010-1"),
    subset2(*b"01-11001111001110"),
    subset2(*b"00010011-11001000"),
    subset2(*b"00-11001001001100"),
    subset2(*b"00-11101111011100"),
    subset2(*b"01-10100110010110"),
    subset2(*b"001111001100001-1"),
    subset2(*b"011001101001100-1"),
    subset2(*b"000001-1001100000"),
    // 48
    subset2(*b"010011-1001000000"),
    subset2(*b"00-10011100100000"),
    subset2(*b"000000-1001110010"),
    subset2(*b"00000100-11100100"),
    subset2(*b"011011001001001-1"),
    subset2(*b"001101101100100-1"),
    subset2(*b"01-10001110011100"),
    subset2(*b"00-11100111000110"),
    subset2(*b"011011001100100-1"),
    subset2(*b"011000110011100-1"),
    subset2(*b"011111101000000-1"),
    subset2(*b"000110001110011-1"),
    subset2(*b"000011110011001-1"),
    subset2(*b"00-11001111110000"),
    subset2(*b"00-10001011101110"),
    subset2(*b"010001000111011-1"),
];
pub(crate) const PARTITION_SET_3: [Subset3Map; 64] = [
    // 0
    subset3(*b"001-100110221222-2"),
    subset3(*b"000-10011-22112221"),
    subset3(*b"00002001-2211221-1"),
    subset3(*b"022-200220011011-1"),
    subset3(*b"00000000-1122112-2"),
    subset3(*b"001-100110022002-2"),
    subset3(*b"002-200221111111-1"),
    subset3(*b"00110011-2211221-1"),
    subset3(*b"00000000-1111222-2"),
    subset3(*b"00001111-1111222-2"),
    subset3(*b"000011-112222222-2"),
    subset3(*b"001200-120012001-2"),
    subset3(*b"011201-120112011-2"),
    subset3(*b"01220-1220122012-2"),
    subset3(*b"001-101121122122-2"),
    subset3(*b"001-12001-22002220"),
    // 16
    subset3(*b"000-100110112112-2"),
    subset3(*b"011-10011-20012200"),
    subset3(*b"00001122-1122112-2"),
    subset3(*b"002-200220022111-1"),
    subset3(*b"011-101110222022-2"),
    subset3(*b"000-10001-22212221"),
    subset3(*b"000000-110122012-2"),
    subset3(*b"00001100-22-102210"),
    subset3(*b"012-20-12200110000"),
    subset3(*b"00120012-1122222-2"),
    subset3(*b"011012-21-12210110"),
    subset3(*b"000001-1012-211221"),
    subset3(*b"00221102-1102002-2"),
    subset3(*b"01100-1102002222-2"),
    subset3(*b"0011012201-22001-1"),
    subset3(*b"00002000-2211222-1"),
    // 32
    subset3(*b"00000002-1122122-2"),
    subset3(*b"022-200220012001-1"),
    subset3(*b"001-100120022022-2"),
    subset3(*b"01200-12001-200120"),
    subset3(*b"000011-1122-220000"),
    subset3(*b"01201201-20-120120"),
    subset3(*b"01202012-1-2010120"),
    subset3(*b"0011220011-22001-1"),
    subset3(*b"001111-222200001-1"),
    subset3(*b"010-101012222222-2"),
    subset3(*b"00000000-2121212-1"),
    subset3(*b"00221-1220022112-2"),
    subset3(*b"002-200110022001-1"),
    subset3(*b"022012-210220122-1"),
    subset3(*b"010122-222222010-1"),
    subset3(*b"00002121-2121212-1"),
    // 48
    subset3(*b"010-101010101222-2"),
    subset3(*b"022-201110222011-1"),
    subset3(*b"00021-1120002111-2"),
    subset3(*b"00002-1122112211-2"),
    subset3(*b"02220-1110111022-2"),
    subset3(*b"00021112-1112000-2"),
    subset3(*b"01100-1100110222-2"),
    subset3(*b"0000000021-12211-2"),
    subset3(*b"01100-1102222222-2"),
    subset3(*b"0022001100-11002-2"),
    subset3(*b"00221122-1122002-2"),
    subset3(*b"0000000000002-11-2"),
    subset3(*b"000-200010002000-1"),
    subset3(*b"022212220222-122-2"),
    subset3(*b"010-122222222222-2"),
    subset3(*b"011-12011-22012220"),
];

pub(crate) struct BitStream {
    state: u128,
}
impl BitStream {
    pub fn new(block: [u8; 16]) -> Self {
        Self {
            state: u128::from_le_bytes(block),
        }
    }

    pub fn low_u8(&self) -> u8 {
        self.state as u8
    }

    #[inline(always)]
    pub fn skip(&mut self, n: u8) {
        self.state >>= n;
    }

    #[inline]
    pub fn consume_bit(&mut self) -> bool {
        let bit = self.state as u8 & 1 != 0;
        self.skip(1);
        bit
    }

    #[inline]
    pub fn consume_bits(&mut self, count: u8) -> u8 {
        debug_assert!(0 < count && count <= 8);
        let mask = (1_u16 << count).wrapping_sub(1) as u8;
        let bits = self.state as u8 & mask;
        self.skip(count);
        bits
    }
    #[inline]
    pub fn consume_bits_64(&mut self, count: u8) -> u64 {
        debug_assert!(0 < count && count <= 64);
        let mut bits = self.state as u64;
        if count < 64 {
            bits &= (1_u64.wrapping_shl(count as u32)).wrapping_sub(1);
        }
        self.skip(count);
        bits
    }

    #[inline]
    pub fn consume_bits_32(&mut self, count: u8) -> i32 {
        debug_assert!(0 < count && count <= 31);
        let mask = (1_u32 << count).wrapping_sub(1);
        let bits = self.state as u32 & mask;
        self.skip(count);
        bits as i32
    }
    // Consumes the bits in reverse order.
    #[inline]
    pub fn consume_bits_rev(&mut self, count: u8) -> u8 {
        debug_assert!(count <= 8);
        let mask = (1_u16 << count).wrapping_sub(1) as u8;
        let bits = self.state as u8 & mask;
        self.skip(count);
        if count >= 2 {
            bits.reverse_bits() >> (8 - count)
        } else {
            bits
        }
    }
}

/// A list of uncompressed indexes.
///
/// BC7 uses compressed indexes. Instead of using a bit stream and uncompressed
/// the fix-up indexes an the fly, it's faster to decompress all indexes before
/// using them. This allows the compiler to more easily unroll the pixel
/// interpolation loops.
pub(crate) struct Indexes {
    uncompressed: u64,
    bits: u8,
    mask: u64,
}
impl Indexes {
    fn get_mask(bits: u8) -> u64 {
        (1 << bits) - 1
    }

    pub fn new_p1(bits: u8, stream: &mut BitStream) -> Self {
        Self::from_compressed_p1(bits, stream.consume_bits_64(16 * bits - 1))
    }
    pub fn new_p2(bits: u8, stream: &mut BitStream, p2_fixup: u8) -> Self {
        Self::from_compressed_p2(bits, stream.consume_bits_64(16 * bits - 2), p2_fixup)
    }
    pub fn new_p3(bits: u8, stream: &mut BitStream, p2_fixup: u8, p3_fixup: u8) -> Self {
        Self::from_compressed_p3(
            bits,
            stream.consume_bits_64(16 * bits - 3),
            p2_fixup,
            p3_fixup,
        )
    }
    pub fn from_compressed_p1(bits: u8, mut compressed: u64) -> Self {
        debug_assert!(bits <= 4);
        compressed = Self::decompress_single_index(bits, compressed, 0);
        Self {
            uncompressed: compressed,
            bits,
            mask: Self::get_mask(bits),
        }
    }
    pub fn from_compressed_p2(bits: u8, mut compressed: u64, p2_fixup: u8) -> Self {
        debug_assert!(bits <= 4);
        debug_assert!(0 < p2_fixup);
        compressed = Self::decompress_single_index(bits, compressed, 0);
        compressed = Self::decompress_single_index(bits, compressed, p2_fixup);
        Self {
            uncompressed: compressed,
            bits,
            mask: Self::get_mask(bits),
        }
    }
    pub fn from_compressed_p3(bits: u8, mut compressed: u64, p2_fixup: u8, p3_fixup: u8) -> Self {
        debug_assert!(bits <= 4);
        debug_assert!(0 < p2_fixup && p2_fixup < p3_fixup);
        compressed = Self::decompress_single_index(bits, compressed, 0);
        compressed = Self::decompress_single_index(bits, compressed, p2_fixup);
        compressed = Self::decompress_single_index(bits, compressed, p3_fixup);
        Self {
            uncompressed: compressed,
            bits,
            mask: Self::get_mask(bits),
        }
    }
    fn decompress_single_index(bits: u8, mut compressed: u64, index: u8) -> u64 {
        let mask = Self::get_mask(bits);

        let keep_count = index * bits;
        let keep = compressed & ((1 << keep_count) - 1);
        compressed >>= keep_count;
        compressed <<= 1;
        let first = compressed & mask;
        compressed = (compressed & !mask) | (first >> 1);
        compressed <<= keep_count;
        compressed |= keep;
        compressed
    }

    pub fn get_index(&self, pixel_index: u8) -> u8 {
        debug_assert!(pixel_index < 16);

        ((self.uncompressed >> (pixel_index * self.bits)) & self.mask) as u8
    }
}
