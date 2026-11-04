//! Shared data for the BCn encoders/decoders.

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
        debug_assert!(pixel_index < 16);
        (self.subset_indexes.wrapping_shr(pixel_index as u32) & 0b1) as u8
    }

    /// Returns the number of pixels assigned to subset index 0.
    pub const fn count_zeros(self) -> u8 {
        self.subset_indexes.count_zeros() as u8
    }
    /// Returns the number of pixels assigned to subset index 1.
    #[allow(dead_code)]
    pub const fn count_ones(self) -> u8 {
        self.subset_indexes.count_ones() as u8
    }

    /// Reorders the elements in a block according to the subset indexes.
    ///
    /// The relative order of pixels within each subset is preserved. In that
    /// sense, this is a stable partition.
    pub fn sort_block<T: Copy>(self, block: &mut [T; 16]) {
        // This implements counting sort.
        // The idea is that we want to sort the numbers:
        //   for i in 0..16:
        //     i | (subset_index(i) << 4)
        // These 16 numbers are (1) unique and (2) in the range 0..32. So we can
        // use a 32-bit bitset to count them.
        let mut bitset: u32 = 0;
        for i in 0..16 {
            let index = i | (self.get_subset_index(i) << 4);
            bitset |= 1 << index;
        }

        let original = *block;
        let mut count = 0;
        for i in 0..32 {
            // The count < 16 check is just to allow the compiler to optimize
            // away bounds checks on block[count].
            if (bitset & (1 << i)) != 0 && count < 16 {
                block[count] = original[i & 0x0F];
                count += 1;
            }
        }
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
        debug_assert!(pixel_index < 16);
        (self.subset_indexes.wrapping_shr(pixel_index as u32 * 2) & 0b11) as u8
    }

    const ONE_MASK: u32 = 0x5555_5555;
    const TWO_MASK: u32 = 0xAAAA_AAAA;

    /// Returns the number of pixels assigned to subset index 0.
    pub const fn count_zeros(self) -> u8 {
        16 - self.subset_indexes.count_ones() as u8
    }
    /// Returns the number of pixels assigned to subset index 1.
    pub const fn count_ones(self) -> u8 {
        (self.subset_indexes & Self::ONE_MASK).count_ones() as u8
    }
    /// Returns the number of pixels assigned to subset index 2.
    #[allow(dead_code)]
    pub const fn count_twos(self) -> u8 {
        (self.subset_indexes & Self::TWO_MASK).count_ones() as u8
    }

    #[allow(dead_code)]
    pub const fn get_zero_mask(self) -> u32 {
        !(self.get_one_mask() | self.get_two_mask())
    }
    #[allow(dead_code)]
    pub const fn get_one_mask(self) -> u32 {
        let mask = self.subset_indexes & Self::ONE_MASK;
        mask | (mask << 1)
    }
    #[allow(dead_code)]
    pub const fn get_two_mask(self) -> u32 {
        let mask = self.subset_indexes & Self::TWO_MASK;
        mask | (mask >> 1)
    }

    /// Reorders the elements in a block according to the subset indexes.
    ///
    /// The relative order of pixels within each subset is preserved. In that
    /// sense, this is a stable partition.
    pub fn sort_block<T: Copy>(self, block: &mut [T; 16]) {
        // This implements counting sort.
        // The idea is the same as Subset2Map::sort_block, but now we have 3 subsets.
        let mut bitset: u64 = 0;
        for i in 0..16 {
            let index = i | (self.get_subset_index(i) << 4);
            bitset |= 1 << index;
        }

        let original = *block;
        let mut count = 0;
        for i in 0..48 {
            // The count < 16 check is just to allow the compiler to optimize
            // away bounds checks on block[count].
            if (bitset & (1 << i)) != 0 && count < 16 {
                block[count] = original[i & 0x0F];
                count += 1;
            }
        }
    }
}

const fn subset2(data: [u8; 17]) -> Subset2Map {
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
const fn subset3(data: [u8; 18]) -> Subset3Map {
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

/// The number of subsets that appear more than once in `PARTITION_SET_3`.
pub(crate) const PARTITION_SET_3_DUPLICATE_COUNT: usize = 25;
pub(crate) const PARTITION_SET_3_DUPLICATES: [[u8; 3]; 64] = [
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [0, 1, 2],
    [4, 3, 2],
    [5, 6, 3],
    [5, 7, 1],
    [0, 8, 9],
    [10, 11, 9],
    [10, 12, 6],
    [4, 13, 14],
    [15, 16, 14],
    [15, 17, 7],
    [255, 255, 18],
    [255, 255, 19],
    [255, 255, 255],
    [255, 20, 255],
    [10, 255, 255],
    [255, 9, 255],
    [15, 255, 255],
    [255, 14, 255],
    [255, 255, 2],
    [255, 255, 1],
    [255, 255, 3],
    [5, 255, 255],
    [255, 255, 255],
    [255, 255, 21],
    [255, 22, 255],
    [255, 23, 255],
    [255, 255, 24],
    [255, 255, 19],
    [255, 255, 18],
    [255, 255, 20],
    [255, 255, 18],
    [255, 17, 13],
    [255, 12, 8],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 6],
    [0, 255, 255],
    [255, 255, 7],
    [4, 255, 255],
    [255, 255, 16],
    [255, 255, 11],
    [10, 255, 255],
    [255, 255, 9],
    [15, 255, 255],
    [255, 255, 14],
    [10, 255, 255],
    [15, 255, 255],
    [255, 255, 14],
    [255, 255, 9],
    [0, 21, 255],
    [255, 23, 6],
    [4, 24, 255],
    [255, 22, 7],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 20, 19],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BC6HFormat {
    UnsignedF16,
    SignedF16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct EndPointPair {
    pub a: IntColor<i32>,
    pub b: IntColor<i32>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct IntColor<T> {
    pub r: T,
    pub g: T,
    pub b: T,
    _pad: T,
}
impl<T> IntColor<T> {
    pub fn new(r: T, g: T, b: T) -> Self
    where
        T: Default,
    {
        Self {
            r,
            g,
            b,
            _pad: T::default(),
        }
    }
    pub fn rgb(&self) -> [T; 3]
    where
        T: Copy,
    {
        [self.r, self.g, self.b]
    }
    #[inline]
    pub fn get_channel(&self, channel: usize) -> T
    where
        T: Copy,
    {
        match channel {
            0 => self.r,
            1 => self.g,
            2 => self.b,
            _ => panic!("Invalid channel index"),
        }
    }
    #[inline]
    pub fn set_channel(&mut self, channel: usize, value: T) {
        match channel {
            0 => self.r = value,
            1 => self.g = value,
            2 => self.b = value,
            _ => panic!("Invalid channel index"),
        }
    }
}
impl IntColor<i32> {
    pub fn sign_extend_all(&mut self, bit_count: u8) {
        self.r = sign_extend(self.r, bit_count);
        self.g = sign_extend(self.g, bit_count);
        self.b = sign_extend(self.b, bit_count);
    }
    pub fn sign_extend(&mut self, r_bit_count: u8, g_bit_count: u8, b_bit_count: u8) {
        self.r = sign_extend(self.r, r_bit_count);
        self.g = sign_extend(self.g, g_bit_count);
        self.b = sign_extend(self.b, b_bit_count);
    }

    pub fn bit_and(&self, mask: i32) -> Self {
        Self {
            r: self.r & mask,
            g: self.g & mask,
            b: self.b & mask,
            _pad: self._pad & mask,
        }
    }
}
impl core::ops::Add for IntColor<i32> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        IntColor {
            r: self.r.wrapping_add(rhs.r),
            g: self.g.wrapping_add(rhs.g),
            b: self.b.wrapping_add(rhs.b),
            _pad: self._pad.wrapping_add(rhs._pad),
        }
    }
}
fn sign_extend(x: i32, bit_count: u8) -> i32 {
    debug_assert!(bit_count > 0);
    debug_assert!(bit_count < 32);

    // check that all bits outsize bit_count are zero
    debug_assert_eq!(x & !((1 << bit_count) - 1), 0);

    let shift = 32 - bit_count;
    (x << shift) >> shift
}

#[derive(Clone, Copy)]
pub(crate) enum ModeTwo {
    M10_555 = 0b00,
    M7_666 = 0b01,

    M11_544 = 0b00010,
    M11_454 = 0b00110,
    M11_445 = 0b01010,
    M9_555 = 0b01110,
    M8_655 = 0b10010,
    M8_565 = 0b10110,
    M8_556 = 0b11010,
    M6_666 = 0b11110,
}
impl ModeTwo {
    pub fn a0_bit_count(&self) -> u8 {
        match self {
            ModeTwo::M10_555 => 10,
            ModeTwo::M7_666 => 7,

            ModeTwo::M11_544 | ModeTwo::M11_454 | ModeTwo::M11_445 => 11,
            ModeTwo::M9_555 => 9,
            ModeTwo::M8_655 | ModeTwo::M8_565 | ModeTwo::M8_556 => 8,
            ModeTwo::M6_666 => 6,
        }
    }
    pub fn delta_bit_count(&self) -> [u8; 3] {
        match self {
            ModeTwo::M10_555 => [5, 5, 5],
            ModeTwo::M7_666 => [6, 6, 6],

            ModeTwo::M11_544 => [5, 4, 4],
            ModeTwo::M11_454 => [4, 5, 4],
            ModeTwo::M11_445 => [4, 4, 5],
            ModeTwo::M9_555 => [5, 5, 5],
            ModeTwo::M8_655 => [6, 5, 5],
            ModeTwo::M8_565 => [5, 6, 5],
            ModeTwo::M8_556 => [5, 5, 6],
            ModeTwo::M6_666 => [6, 6, 6],
        }
    }

    pub fn transformed(&self) -> bool {
        !matches!(self, ModeTwo::M6_666)
    }
}
#[derive(Clone, Copy)]
pub(crate) enum ModeOne {
    M10_10 = 0b00011,
    M11_9 = 0b00111,
    M12_8 = 0b01011,
    M16_4 = 0b01111,
}
impl ModeOne {
    pub fn a0_bit_count(&self) -> u8 {
        match self {
            ModeOne::M10_10 => 10,
            ModeOne::M11_9 => 11,
            ModeOne::M12_8 => 12,
            ModeOne::M16_4 => 16,
        }
    }
    pub fn b0_bit_count(&self) -> u8 {
        20 - self.a0_bit_count()
    }

    pub fn transformed(&self) -> bool {
        !matches!(self, ModeOne::M10_10)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn start_with_zero() {
        for subset in PARTITION_SET_2.iter() {
            assert_eq!(subset.get_subset_index(0), 0);
        }
        for subset in PARTITION_SET_3.iter() {
            assert_eq!(subset.get_subset_index(0), 0);
        }
    }

    #[test]
    fn count_members() {
        for subset in PARTITION_SET_2.iter() {
            let mut count = [0, 0];
            for i in 0..16 {
                let index = subset.get_subset_index(i);
                count[index as usize] += 1;
            }

            assert_eq!(count[0], subset.count_zeros());
            assert_eq!(count[1], subset.count_ones());
        }

        for subset in PARTITION_SET_3.iter() {
            let mut count = [0, 0, 0];
            for i in 0..16 {
                let index = subset.get_subset_index(i);
                count[index as usize] += 1;
            }

            assert_eq!(count[0], subset.count_zeros());
            assert_eq!(count[1], subset.count_ones());
            assert_eq!(count[2], subset.count_twos());
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn partition_block() {
        for subset in PARTITION_SET_2.iter() {
            let mut block = [0u8; 16];
            for i in 0..16 {
                block[i as usize] = i | (subset.get_subset_index(i) << 4);
            }
            subset.sort_block(&mut block);
            let zeros = subset.count_zeros() as usize;
            assert!(block[..zeros].iter().all(|i| *i < 16));
            assert!(block[zeros..].iter().all(|i| *i >= 16));

            // no duplicates and stable
            block.sort();
            for i in 1..16 {
                assert!(block[i - 1] < block[i]);
            }
        }

        for subset in PARTITION_SET_3.iter() {
            let mut block = [0u8; 16];
            for i in 0..16 {
                block[i as usize] = i | (subset.get_subset_index(i) << 4);
            }
            subset.sort_block(&mut block);
            let zeros = subset.count_zeros() as usize;
            let ones = subset.count_ones() as usize;
            assert!(block[..zeros].iter().all(|i| *i < 16));
            assert!(block[zeros..zeros + ones]
                .iter()
                .all(|i| *i >= 16 && *i < 32));
            assert!(block[zeros + ones..].iter().all(|i| *i >= 32));

            // no duplicates and stable
            block.sort();
            for i in 1..16 {
                assert!(block[i - 1] < block[i]);
            }
        }
    }

    #[test]
    fn subset3_overlap() {
        let mut mask_counter: HashMap<u32, u32> = HashMap::new();

        for subset in PARTITION_SET_3 {
            let masks = [
                subset.get_zero_mask(),
                subset.get_one_mask(),
                subset.get_two_mask(),
            ];

            for mask in masks {
                *mask_counter.entry(mask).or_insert(0) += 1;
            }
        }

        mask_counter.retain(|_, v| *v > 1);
        assert_eq!(mask_counter.len(), PARTITION_SET_3_DUPLICATE_COUNT);

        mask_counter.iter_mut().for_each(|(_, v)| *v = 255);
        for (i, &subset) in PARTITION_SET_3.iter().enumerate() {
            let masks = [
                subset.get_zero_mask(),
                subset.get_one_mask(),
                subset.get_two_mask(),
            ];

            for mask in masks {
                if let Some(counter) = mask_counter.get_mut(&mask) {
                    *counter = u32::min(*counter, i as u32);
                }
            }
        }

        let mut sorted = Vec::from_iter(mask_counter);
        sorted.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        let sorted = Vec::from_iter(sorted.into_iter().map(|(k, _)| k));

        let cache_array: [[u8; 3]; 64] = std::array::from_fn(|i| {
            let subset = PARTITION_SET_3[i];
            [
                subset.get_zero_mask(),
                subset.get_one_mask(),
                subset.get_two_mask(),
            ]
            .map(|mask| sorted.iter().position(|m| *m == mask).unwrap_or(255) as u8)
        });

        assert_eq!(cache_array, PARTITION_SET_3_DUPLICATES);
    }
}
