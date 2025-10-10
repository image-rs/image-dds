#![allow(clippy::needless_range_loop)]

use glam::Vec4;

use crate::{encode::bcn_util, n8};

pub(crate) fn compress_bc7_block(block: [Rgba<8>; 16]) -> [u8; 16] {
    let stats = BlockStats::new(&block);

    // a block of a single color can be compressed exactly
    if let Some(color) = stats.single_color() {
        return compress_single_color(color);
    }

    compress_mode5(block, stats)
}

/// Solid-color blocks can be encoded exactly.
///
/// https://fgiesen.wordpress.com/2024/11/03/bc7-optimal-solid-color-blocks/
fn compress_single_color(color: Rgba<8>) -> [u8; 16] {
    fn optimize(c: u8) -> (u8, u8) {
        (c >> 1, if c < 128 { c + 1 } else { c - 1 } >> 1)
    }

    let (c0_r, c1_r) = optimize(color.r);
    let (c0_g, c1_g) = optimize(color.g);
    let (c0_b, c1_b) = optimize(color.b);

    let c0 = Rgb::new(c0_r, c0_g, c0_b);
    let c1 = Rgb::new(c1_r, c1_g, c1_b);

    mode5(
        Rotation::None,
        [c0, c1],
        IndexList::<2>::constant(1),
        [Alpha::new(color.a); 2],
        // the index for alpha doesn't matter since both endpoints are the same
        IndexList::<2>::constant(1),
    )
}

fn compress_mode5(block: [Rgba<8>; 16], stats: BlockStats) -> [u8; 16] {
    let mut r = block[5];

    let alpha_pixels = block.map(|p| p.alpha());
    fn quantize(min: f32, max: f32) -> (Alpha<8>, Alpha<8>) {
        // floor for min and ceil for max
        (
            Alpha::new((min * 255.0) as u8),
            Alpha::new((max * 255.0 + 0.999) as u8),
        )
    }
    let initial = (n8::f32(stats.min.a), n8::f32(stats.max.a));
    let (a_min, a_max) = bcn_util::refine_endpoints(
        initial.0,
        initial.1,
        bcn_util::RefinementOptions {
            step_initial: (initial.1 - initial.0) * 0.2,
            step_decay: 0.5,
            step_min: 1.0 / 255.0,
            max_iter: 4,
        },
        |(min, max)| {
            let (min, max) = quantize(min, max);
            closest_error_alpha::<2>(min, max, &alpha_pixels)
        },
    );
    let (a_min, a_max) = quantize(a_min, a_max);
    let alpha_list = closest_alpha::<2>(a_min, a_max, &alpha_pixels);

    mode5(
        Rotation::None,
        [Rgb::new(r.r >> 1, r.g >> 1, r.b >> 1); 2],
        IndexList::constant(0),
        [a_min, a_max],
        alpha_list,
    )
}

fn closest_alpha<const B: u8>(
    min: Alpha<8>,
    max: Alpha<8>,
    pixels: &[Alpha<8>; 16],
) -> IndexList<B> {
    debug_assert!(min.a <= max.a);
    if min.a >= max.a {
        // alpha endpoints are constant
        return IndexList::constant(0);
    }
    debug_assert!(min.a < max.a);

    let mut indexes = IndexList::new();
    let a_diff = (max.a - min.a) as u16;
    debug_assert!(a_diff != 0);
    let round = a_diff / 2;
    let max = IndexList::<B>::MAX_INDEX as u16;
    for i in 0..16 {
        let value = (pixels[i].a.saturating_sub(min.a) as u16 * max + round) / a_diff;
        indexes.set(i, value.min(max) as u8);
    }
    indexes
}
/// The square error of the closest alpha values.
fn closest_error_alpha<const B: u8>(min: Alpha<8>, max: Alpha<8>, pixels: &[Alpha<8>; 16]) -> u32 {
    debug_assert!(B == 2 || B == 3);

    debug_assert!(min.a <= max.a);
    if min.a >= max.a {
        debug_assert!(min.a == max.a);
        let a = min.a;
        pixels
            .iter()
            .map(|p| {
                let d = p.a.abs_diff(a) as u32;
                d * d
            })
            .sum()
    } else {
        debug_assert!(min.a < max.a);

        fn error<const N: usize>(pixels: &[Alpha<8>; 16], interpolated: [u8; N]) -> u32 {
            pixels
                .iter()
                .map(|p| {
                    let mut best = 255_u8;
                    for &a in &interpolated {
                        let d = p.a.abs_diff(a);
                        best = best.min(d);
                    }
                    let d = best as u32;
                    d * d
                })
                .sum()
        }

        match B {
            2 => {
                let interpolated = [
                    min.a,
                    interpolate::<2>(min.a, max.a, 1),
                    interpolate::<2>(min.a, max.a, 2),
                    max.a,
                ];
                error(pixels, interpolated)
            }
            3 => {
                let interpolated = [
                    min.a,
                    interpolate::<3>(min.a, max.a, 1),
                    interpolate::<3>(min.a, max.a, 2),
                    interpolate::<3>(min.a, max.a, 3),
                    interpolate::<3>(min.a, max.a, 4),
                    interpolate::<3>(min.a, max.a, 5),
                    interpolate::<3>(min.a, max.a, 6),
                    max.a,
                ];
                error(pixels, interpolated)
            }
            _ => unreachable!(),
        }
    }
}

fn mode5(
    rotation: Rotation,
    mut color: [Rgb<7>; 2],
    color_indexes: IndexList<2>,
    mut alpha: [Alpha<8>; 2],
    alpha_indexes: IndexList<2>,
) -> [u8; 16] {
    let (color_indexes, color_swap) = color_indexes.compress_p1();
    let (alpha_indexes, alpha_swap) = alpha_indexes.compress_p1();

    if color_swap {
        color.swap(0, 1);
    }
    if alpha_swap {
        alpha.swap(0, 1);
    }

    let mut stream = BitStream::new();
    stream.write_mode(5);
    stream.write_rotation(rotation);
    stream.write_endpoints_rgb(color);
    stream.write_endpoints_alpha(alpha);
    stream.write_indexes(color_indexes);
    stream.write_indexes(alpha_indexes);
    stream.finish()
}
fn mode6(mut rgba: [Rgba<7>; 2], mut p: [bool; 2], indexes: IndexList<4>) -> [u8; 16] {
    let (indexes, swap) = indexes.compress_p1();

    if swap {
        rgba.swap(0, 1);
        p.swap(0, 1);
    }

    let mut stream = BitStream::new();
    stream.write_mode(6);
    stream.write_endpoints_rgba(rgba);
    stream.write_endpoints_p(p);
    stream.write_indexes(indexes);
    stream.finish()
}

enum Rotation {
    None = 0,
    AR = 1,
    AG = 2,
    AB = 3,
}

/// A list of 16 indexes each using B bits.
///
/// B must be 2, 3, or 4.
struct IndexList<const B: u8> {
    indexes: u64,
}
impl<const B: u8> IndexList<B> {
    const MAX_INDEX: u8 = (1 << B) - 1;
    const INDEXES_MASK: u64 = if B == 4 {
        u64::MAX
    } else {
        (1 << (B * 16)) - 1
    };

    const fn new() -> Self {
        debug_assert!(B == 2 || B == 3 || B == 4);
        Self { indexes: 0 }
    }
    const CONSTANT_MULTIPLE: u64 = {
        let mut m = 0;
        let mut i = 0;
        while i < 16 {
            m |= 1 << (i * B);
            i += 1;
        }
        m
    };
    fn constant(value: u8) -> Self {
        debug_assert!(B == 2 || B == 3 || B == 4);
        debug_assert!(value <= Self::MAX_INDEX);

        Self {
            indexes: value as u64 * Self::CONSTANT_MULTIPLE,
        }
    }

    fn get(&self, index: usize) -> u8 {
        debug_assert!(index < 16);
        ((self.indexes >> (index * B as usize)) & Self::MAX_INDEX as u64) as u8
    }
    fn set(&mut self, index: usize, value: u8) {
        debug_assert!(index < 16);
        debug_assert!(value <= Self::MAX_INDEX);
        debug_assert!(self.get(index) == 0, "Cannot set an index twice.");
        self.indexes |= (value as u64) << (index * B as usize);
    }

    /// Compresses the index list and returns whether the endpoints need to be swapped.
    fn compress_p1(self) -> (CompressedIndexList, bool) {
        let (compressed, swap) = Self::compress_single_index(self.indexes, 0);
        (
            CompressedIndexList {
                compressed_indexes: compressed,
                bits: 16 * B - 1,
            },
            swap,
        )
    }

    fn compress_single_index(mut indexes: u64, index: u8) -> (u64, bool) {
        debug_assert!(B == 2 || B == 3 || B == 4);
        debug_assert!(index < 16);

        // the MSB of the index-th value has to be 0
        let msb_mask = 1 << (B - 1 + index * B);
        let swap = (indexes & msb_mask) != 0;

        // if the MSB is 1, flip all bits
        if swap {
            indexes ^= Self::INDEXES_MASK;
        }

        // now the MSB of the given index value is 0, so we can drop it
        debug_assert!((indexes & msb_mask) == 0);
        let before_mask = msb_mask - 1;
        let after_mask = !before_mask << 1;

        let compressed = (indexes & before_mask) | ((indexes & after_mask) >> 1);
        (compressed, swap)
    }
}

struct CompressedIndexList {
    compressed_indexes: u64,
    bits: u8,
}

#[repr(C, align(4))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Rgba<const B: u8> {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}
const _: () = {
    assert!(std::mem::size_of::<Rgba<8>>() == std::mem::size_of::<u32>());
    assert!(std::mem::align_of::<Rgba<8>>() == std::mem::align_of::<u32>());
};
impl<const B: u8> Rgba<B> {
    const MAX: u8 = ((1_u32 << B) - 1) as u8;

    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        debug_assert!(0 < B && B <= 8);
        debug_assert!(r <= Self::MAX);
        debug_assert!(g <= Self::MAX);
        debug_assert!(b <= Self::MAX);
        debug_assert!(a <= Self::MAX);
        Self { r, g, b, a }
    }

    pub fn round(v: Vec4) -> Self {
        debug_assert!(0 < B && B <= 8);
        let x = v.min(Vec4::ONE) * (Self::MAX as f32) + 0.5;
        Self::new(x.x as u8, x.y as u8, x.z as u8, x.w as u8)
    }

    pub fn to_u32(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, self.a])
    }
    pub fn from_u32(x: u32) -> Self {
        let [r, g, b, a] = x.to_le_bytes();
        Self::new(r, g, b, a)
    }

    pub fn color(self) -> Rgb<B> {
        Rgb::new(self.r, self.g, self.b)
    }
    pub fn alpha(self) -> Alpha<B> {
        Alpha::new(self.a)
    }

    pub fn promote(self) -> Rgba<8> {
        if B == 8 {
            return Rgba::new(self.r, self.g, self.b, self.a);
        }
        Rgba::new(
            promote(self.r, B),
            promote(self.g, B),
            promote(self.b, B),
            promote(self.a, B),
        )
    }
}
impl<const B: u8> PartialEq for Rgba<B> {
    fn eq(&self, other: &Self) -> bool {
        self.to_u32() == other.to_u32()
    }
}
impl<const B: u8> Eq for Rgba<B> {}

#[repr(C, align(4))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Rgb<const B: u8> {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}
const _: () = {
    assert!(std::mem::size_of::<Rgb<8>>() == std::mem::size_of::<u32>());
    assert!(std::mem::align_of::<Rgb<8>>() == std::mem::align_of::<u32>());
};
impl<const B: u8> Rgb<B> {
    const MAX: u8 = ((1_u32 << B) - 1) as u8;

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        debug_assert!(0 < B && B <= 8);
        debug_assert!(r <= Self::MAX);
        debug_assert!(g <= Self::MAX);
        debug_assert!(b <= Self::MAX);
        Self { r, g, b }
    }

    pub fn to_u32(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, 0])
    }
    pub fn from_u32(x: u32) -> Self {
        let [r, g, b, _] = x.to_le_bytes();
        Self::new(r, g, b)
    }

    pub fn promote(self) -> Rgb<8> {
        if B == 8 {
            return Rgb::new(self.r, self.g, self.b);
        }
        Rgb::new(promote(self.r, B), promote(self.g, B), promote(self.b, B))
    }
}
impl<const B: u8> PartialEq for Rgb<B> {
    fn eq(&self, other: &Self) -> bool {
        self.to_u32() == other.to_u32()
    }
}
impl<const B: u8> Eq for Rgb<B> {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Alpha<const B: u8> {
    pub a: u8,
}
impl<const B: u8> Alpha<B> {
    const MAX: u8 = ((1_u32 << B) - 1) as u8;

    pub const fn new(a: u8) -> Self {
        debug_assert!(0 < B && B <= 8);
        debug_assert!(a <= Self::MAX);
        Self { a }
    }

    pub fn promote(self) -> Alpha<8> {
        if B == 8 {
            return Alpha::new(self.a);
        }
        Alpha::new(promote(self.a, B))
    }
}

#[inline]
fn promote(mut number: u8, number_bits: u8) -> u8 {
    debug_assert!((4..8).contains(&number_bits));
    number <<= 8 - number_bits;
    number |= number >> number_bits;
    number
}

trait AddPBit {
    type Output;
    fn add_p(self, p: bool) -> Self::Output;
}
impl AddPBit for Rgba<5> {
    type Output = Rgba<6>;
    fn add_p(self, p: bool) -> Self::Output {
        let p = p as u8;
        let Self { r, g, b, a } = self;
        Rgba::new((r << 1) | p, (g << 1) | p, (b << 1) | p, (a << 1) | p)
    }
}
impl AddPBit for Rgba<7> {
    type Output = Rgba<8>;
    fn add_p(self, p: bool) -> Self::Output {
        let p = p as u8;
        let Self { r, g, b, a } = self;
        Rgba::new((r << 1) | p, (g << 1) | p, (b << 1) | p, (a << 1) | p)
    }
}
impl AddPBit for Rgb<4> {
    type Output = Rgb<5>;
    fn add_p(self, p: bool) -> Self::Output {
        let p = p as u8;
        Rgb::new((self.r << 1) | p, (self.g << 1) | p, (self.b << 1) | p)
    }
}
impl AddPBit for Rgb<6> {
    type Output = Rgb<7>;
    fn add_p(self, p: bool) -> Self::Output {
        let p = p as u8;
        Rgb::new((self.r << 1) | p, (self.g << 1) | p, (self.b << 1) | p)
    }
}
impl AddPBit for Rgb<7> {
    type Output = Rgb<8>;
    fn add_p(self, p: bool) -> Self::Output {
        let p = p as u8;
        Rgb::new((self.r << 1) | p, (self.g << 1) | p, (self.b << 1) | p)
    }
}

// Weights are all multiplied by 4 compared to the original ones. This changes
// the interpolation formula from
//   ((64-w)*e0 + w*e1 + 32) >> 6
// to
//   ((256-w)*e0 + w*e1 + 128) >> 8
// The nice thing about this is that intermediate results still fit into u16,
// but the compiler can optimize away the `>> 8`.
const WEIGHTS_2: [u16; 4] = [0, 84, 172, 256];
const WEIGHTS_3: [u16; 8] = [0, 36, 72, 108, 148, 184, 220, 256];
const WEIGHTS_4: [u16; 16] = [
    0, 16, 36, 52, 68, 84, 104, 120, 136, 152, 172, 188, 204, 220, 240, 256,
];

fn interpolate<const B: usize>(e0: u8, e1: u8, index: u8) -> u8 {
    let weight = match B {
        2 => WEIGHTS_2[index as usize],
        3 => WEIGHTS_3[index as usize],
        4 => WEIGHTS_4[index as usize],
        _ => unreachable!(),
    };
    let w0 = 256 - weight;
    let w1 = weight;
    ((w0 * e0 as u16 + w1 * e1 as u16 + 128) >> 8) as u8
}

#[derive(Clone, Copy)]
struct BlockStats {
    min: Rgba<8>,
    max: Rgba<8>,
}
impl BlockStats {
    fn new(block: &[Rgba<8>; 16]) -> Self {
        let mut min = Rgba::new(255, 255, 255, 255);
        let mut max = Rgba::new(0, 0, 0, 0);
        for &pixel in block {
            min.r = min.r.min(pixel.r);
            min.g = min.g.min(pixel.g);
            min.b = min.b.min(pixel.b);
            min.a = min.a.min(pixel.a);

            max.r = max.r.max(pixel.r);
            max.g = max.g.max(pixel.g);
            max.b = max.b.max(pixel.b);
            max.a = max.a.max(pixel.a);
        }
        Self { min, max }
    }

    fn single_color(&self) -> Option<Rgba<8>> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }

    fn constant_alpha(&self) -> Option<u8> {
        if self.min.a == self.max.a {
            Some(self.max.a)
        } else {
            None
        }
    }
    /// Returns whether Alpha is 255 everywhere.
    fn opaque(&self) -> bool {
        self.min.a == 255
    }
}

struct BitStream {
    data: u128,
    bits: u8,
}
impl BitStream {
    fn new() -> Self {
        Self { data: 0, bits: 0 }
    }

    #[inline(always)]
    fn write_u64(&mut self, value: u64, bits: u8) {
        debug_assert!(bits < 64);
        debug_assert!(value < (1 << bits));

        self.data |= (value as u128) << self.bits;
        self.bits += bits;
    }

    fn write_mode(&mut self, mode: u8) {
        debug_assert!(mode < 8);
        self.write_u64(1 << mode, mode + 1);
    }
    fn write_rotation(&mut self, rotation: Rotation) {
        self.write_u64(rotation as u64, 2);
    }
    fn write_indexes(&mut self, indexes: CompressedIndexList) {
        self.write_u64(indexes.compressed_indexes, indexes.bits);
    }
    fn write_endpoints_rgba<const B: u8>(&mut self, endpoints: [Rgba<B>; 2]) {
        self.write_u64(endpoints[0].r as u64, B);
        self.write_u64(endpoints[1].r as u64, B);
        self.write_u64(endpoints[0].g as u64, B);
        self.write_u64(endpoints[1].g as u64, B);
        self.write_u64(endpoints[0].b as u64, B);
        self.write_u64(endpoints[1].b as u64, B);
        self.write_u64(endpoints[0].a as u64, B);
        self.write_u64(endpoints[1].a as u64, B);
    }
    fn write_endpoints_rgb<const B: u8>(&mut self, endpoints: [Rgb<B>; 2]) {
        self.write_u64(endpoints[0].r as u64, B);
        self.write_u64(endpoints[1].r as u64, B);
        self.write_u64(endpoints[0].g as u64, B);
        self.write_u64(endpoints[1].g as u64, B);
        self.write_u64(endpoints[0].b as u64, B);
        self.write_u64(endpoints[1].b as u64, B);
    }
    fn write_endpoints_alpha<const B: u8>(&mut self, endpoints: [Alpha<B>; 2]) {
        self.write_u64(endpoints[0].a as u64, B);
        self.write_u64(endpoints[1].a as u64, B);
    }
    fn write_endpoints_p<const N: usize>(&mut self, p: [bool; N]) {
        debug_assert!(N % 2 == 0);
        for p in p {
            self.write_u64(p as u64, 1);
        }
    }

    fn finish(self) -> [u8; 16] {
        debug_assert!(self.bits == 128);
        self.data.to_le_bytes()
    }
}
