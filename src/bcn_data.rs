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
    pub const fn count_twos(self) -> u8 {
        (self.subset_indexes & Self::TWO_MASK).count_ones() as u8
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

#[cfg(test)]
mod tests {
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
}
