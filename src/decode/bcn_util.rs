//! Shared data and function for the BCn format.

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
