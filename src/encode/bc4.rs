use glam::{UVec4, Vec4};

use crate::{n8, s8};

use super::bcn_util::{self, Block4x4};

#[derive(Debug, Clone, Copy)]
pub struct Bc4Options {
    pub dither: bool,
    pub snorm: bool,
    pub brute_force: bool,
    pub use_inter4: bool,
    pub use_inter4_heuristic: bool,
    pub high_quality_quantize: bool,
    pub fast_iter: bool,
}

struct Block {
    b: [Vec4; 4],
}
impl Block {
    fn from_raw(block: [f32; 16]) -> Self {
        let b0 = Vec4::new(block[0], block[1], block[2], block[3]);
        let b1 = Vec4::new(block[4], block[5], block[6], block[7]);
        let b2 = Vec4::new(block[8], block[9], block[10], block[11]);
        let b3 = Vec4::new(block[12], block[13], block[14], block[15]);

        Self {
            b: [
                b0.clamp(Vec4::ZERO, Vec4::ONE),
                b1.clamp(Vec4::ZERO, Vec4::ONE),
                b2.clamp(Vec4::ZERO, Vec4::ONE),
                b3.clamp(Vec4::ZERO, Vec4::ONE),
            ],
        }
    }
    /// Returns the minimum and maximum value in the block.
    fn min_max(&self) -> (f32, f32) {
        let [b0, b1, b2, b3] = self.b;
        let min = b0.min(b1).min(b2).min(b3);
        let max = b0.max(b1).max(b2).max(b3);
        (min.min_element(), max.max_element())
    }
    /// Of the values in the block that are in the range [threshold, 1-threshold],
    /// returns the minimum and maximum.
    fn min_max_with_threshold(&self, threshold: f32) -> (f32, f32) {
        let low = Vec4::splat(threshold);
        let high = Vec4::splat(1.0 - threshold);

        let mut min = Vec4::ONE;
        let mut max = Vec4::ZERO;
        for b in self.b {
            min = min.min(Vec4::select(b.cmpge(low), b, Vec4::ONE));
            max = max.max(Vec4::select(b.cmple(high), b, Vec4::ZERO));
        }
        (min.min_element(), max.max_element())
    }
}
impl Block4x4<f32> for &Block {
    #[inline(always)]
    fn get_pixel_at(&self, index: usize) -> f32 {
        debug_assert!(index < 16);
        let vec_index = (index / 4) % 4;
        let component_index = index % 4;
        self.b[vec_index][component_index]
    }
}

/// The smallest non-zero value that can be represented in a BC4 block.
///
/// This is also the smallest distance of 2 adjacent representable values.
const BC4_MIN_VALUE: f32 = 1. / (255. * 7.);
/// 2 values that are this close will be considered equal.
const BC4_EPSILON: f32 = 1. / (65536.);

pub(crate) fn compress_bc4_block(block: [f32; 16], options: Bc4Options) -> [u8; 8] {
    let block = Block::from_raw(block);

    let (min, max) = block.min_max();
    let diff = max - min;

    // reference for testing
    if options.brute_force && !options.dither && !options.snorm {
        return reference_brute_force(block);
    }

    // single color
    if diff < BC4_EPSILON {
        let value = (min + max) * 0.5;
        return single_color(value, options);
    }

    // If the colors are far away from 0 and 1, then inter6 should always be better
    // than inter4
    const INTER6_THRESHOLD: f32 = 1. / 7.;
    let heuristic = 0. < min - diff * INTER6_THRESHOLD && max + diff * INTER6_THRESHOLD < 1.;
    if !options.use_inter4 || options.use_inter4_heuristic && heuristic {
        return compress_inter6(&block, min, max, options).0;
    }

    // Encode with both inter6 and inter4 and pick the best
    let (inter6, error6) = compress_inter6(&block, min, max, options);
    let (inter4, error4) = compress_inter4(&block, options);

    if error6 < error4 {
        inter6
    } else {
        inter4
    }
}

/// Brute-forces the best BC4 encoding (lowest MSE/highest) for a block.
/// Only UNORM without dithering is supported.
///
/// This is intended for testing purposes only. It's EXTREMELY slow. If you do
/// use it, make sure to enable optimizations.
fn reference_brute_force(block: Block) -> [u8; 8] {
    let (block_min, block_max) = block.min_max();
    let min_max = (block_max * 255. + 2.) as u8;
    let max_min = (block_min * 255. - 1.) as u8;

    let mut best = [0_u8; 8];
    let mut best_error = f32::INFINITY;
    for min in 0..min_max {
        for max in max_min.max(min + 1)..=255 {
            let endpoints6 = EndPoints::new_inter6_unorm(max, min);
            let palette6 = Inter6Palette::from_endpoints(&endpoints6);
            let error6 = palette6.block_closest_error_sq(&block);
            if error6 < best_error {
                best = endpoints6.with_indexes(palette6.block_closest(&block).0);
                best_error = error6;
            }

            let endpoints4 = endpoints6.inter6_to_inter4();
            let palette4 = Inter4Palette::from_endpoints(&endpoints4);
            let error4 = palette4.block_closest_error_sq(&block);
            if error4 < best_error {
                best = endpoints4.with_indexes(palette4.block_closest(&block).0);
                best_error = error4;
            }
        }
    }

    best
}

fn single_color(value: f32, options: Bc4Options) -> [u8; 8] {
    // See if the closest encoded value is good enough. This doesn't improve
    // quality, but it does make the output more compressable for gzip.
    let closest = EndPoints::new_closest(value, options.snorm);
    if (closest.c0_f - value).abs() < BC4_EPSILON {
        return closest.with_indexes(IndexList::new_all(0));
    }

    // The inter6 palette is typically the better palette for single colors.
    // It certainly is for dithering. However, the inter4 palette might just
    // happen to contain a color closer to the input value, so we check both
    // if dithering is disabled.
    //
    // This approach is optimal for the non-dithering case. I verified this
    // with the reference brute-force implementation.

    let endpoints6 = EndPoints::new_inter6(value, value, options.snorm);
    let palette6 = Inter6Palette::from_endpoints(&endpoints6);

    if options.dither {
        let (indexes, _) = palette6.block_dither(value);
        endpoints6.with_indexes(indexes)
    } else {
        // pick the best palette
        let endpoints4 = EndPoints::new_inter4(value, value, options.snorm);
        let palette4 = Inter4Palette::from_endpoints(&endpoints4);

        let (index_value4, _, error4) = palette4.closest(value);
        let (index_value6, _, error6) = palette6.closest(value);

        if error4 < error6 {
            endpoints4.with_indexes(IndexList::new_all(index_value4))
        } else {
            endpoints6.with_indexes(IndexList::new_all(index_value6))
        }
    }
}

fn refine_endpoints(
    mut min: f32,
    mut max: f32,
    mut compute_error: impl Copy + FnMut((f32, f32)) -> f32,
    mut quantize: impl FnMut((f32, f32)) -> (f32, f32),
    options: Bc4Options,
) -> (f32, f32) {
    // Step 1: Improve the endpoints with a local search
    (min, max) = bcn_util::refine_endpoints(
        min,
        max,
        if options.fast_iter {
            bcn_util::RefinementOptions::new_bc4_fast(min, max)
        } else {
            bcn_util::RefinementOptions::new_bc4(min, max)
        },
        compute_error,
    );

    // Step 2: Quantize the endpoints and select the best
    if !options.high_quality_quantize {
        return (min, max);
    }

    const QUANT_STEP: f32 = 1. / 254. + 0.0001;
    let mut error = compute_error(quantize((min, max)));
    for pair in [
        (min + QUANT_STEP, max),
        (min, max - QUANT_STEP),
        (min + QUANT_STEP, max - QUANT_STEP),
    ] {
        let new_error = compute_error(quantize(pair));
        if new_error < error {
            error = new_error;
            min = pair.0;
            max = pair.1;
        }
    }

    (min, max)
}

fn refinement_error_metric<P: Palette>(
    block: &Block,
    _options: Bc4Options,
) -> impl Copy + Fn((f32, f32)) -> f32 + '_ {
    move |(min, max)| {
        let palette = P::new(min, max);
        // TODO: find a better error metric for dithered blocks
        palette.block_closest_error_sq(block)
    }
}

fn compress_inter6(
    block: &Block,
    mut min: f32,
    mut max: f32,
    options: Bc4Options,
) -> ([u8; 8], f32) {
    (min, max) = refine_endpoints(
        min,
        max,
        refinement_error_metric::<Inter6Palette>(block, options),
        move |(min, max)| {
            let endpoints = EndPoints::new_inter6(min, max, options.snorm);
            (endpoints.c0_f, endpoints.c1_f)
        },
        options,
    );

    let endpoints = EndPoints::new_inter6(min, max, options.snorm);
    let palette = Inter6Palette::from_endpoints(&endpoints);

    let (indexes, error) = if options.dither {
        palette.block_dither(block)
    } else {
        palette.block_closest(block)
    };

    (endpoints.with_indexes(indexes), error)
}

fn compress_inter4(block: &Block, options: Bc4Options) -> ([u8; 8], f32) {
    let (mut min, mut max) = block.min_max_with_threshold(BC4_MIN_VALUE);

    (min, max) = refine_endpoints(
        min,
        max,
        refinement_error_metric::<Inter4Palette>(block, options),
        move |(min, max)| {
            let endpoints = EndPoints::new_inter4(min, max, options.snorm);
            (endpoints.c0_f, endpoints.c1_f)
        },
        options,
    );

    let endpoints = EndPoints::new_inter4(min, max, options.snorm);
    let palette = Inter4Palette::from_endpoints(&endpoints);

    let (indexes, error) = if options.dither {
        palette.block_dither(block)
    } else {
        palette.block_closest(block)
    };

    (endpoints.with_indexes(indexes), error)
}

struct EndPoints {
    c0: u8,
    c1: u8,
    c0_f: f32,
    c1_f: f32,
}
impl EndPoints {
    /// Creates a new endpoint pair for a BC4 block.
    /// C0 will be the value closest to the given value and C1_f will be 0.
    ///
    /// The endpoints are **NOT** guaranteed to be in Inter6 mode.
    fn new_closest(value: f32, snorm: bool) -> Self {
        if snorm {
            let closest_s8_norm = (254.0 * value + 0.5) as u8;

            let c0 = s8::from_norm(closest_s8_norm);
            let c1 = s8::from_norm(0);
            let c0_f = s8::uf32(c0);
            let c1_f = s8::uf32(c1);
            debug_assert!(c1_f == 0.0);

            Self { c0, c1, c0_f, c1_f }
        } else {
            // round down min and round up max
            let closest = (255.0 * value + 0.5) as u8;

            let c0 = closest;
            let c1 = 0;
            let c0_f = n8::f32(c0);
            let c1_f = 0.0;

            Self { c0, c1, c0_f, c1_f }
        }
    }
    fn new_inter6(e0: f32, e1: f32, snorm: bool) -> Self {
        let min = e0.min(e1);
        let max = e0.max(e1);

        // For the 6 interpolation mode, we need c0 > c1
        if snorm {
            // round down min and round up max
            let mut min_s8_norm = (254.0 * min) as u8;
            let mut max_s8_norm = 254 - (254.0 * (1.0 - max)) as u8;

            // make sure they are different
            if min_s8_norm == max_s8_norm {
                if min_s8_norm == 0 {
                    max_s8_norm = 1;
                } else {
                    min_s8_norm -= 1;
                }
            }
            debug_assert!(min_s8_norm < max_s8_norm);

            let mut c0 = s8::from_norm(max_s8_norm);
            let mut c1 = s8::from_norm(min_s8_norm);
            debug_assert!(c0 != c1);
            if c0 as i8 <= c1 as i8 {
                // swap
                std::mem::swap(&mut c0, &mut c1);
            }

            let c0_f = s8::uf32(c0);
            let c1_f = s8::uf32(c1);

            Self { c0, c1, c0_f, c1_f }
        } else {
            // round down min and round up max
            let mut min_u8 = (255.0 * min) as u8;
            let mut max_u8 = 255 - (255.0 * (1.0 - max)) as u8;

            // make sure they are different
            if min_u8 == max_u8 {
                if min_u8 == 0 {
                    max_u8 = 1;
                } else {
                    min_u8 -= 1;
                }
            }
            debug_assert!(min_u8 < max_u8);

            let c0 = max_u8;
            let c1 = min_u8;
            let c0_f = n8::f32(c0);
            let c1_f = n8::f32(c1);

            Self { c0, c1, c0_f, c1_f }
        }
    }
    fn new_inter4(e0: f32, e1: f32, snorm: bool) -> Self {
        Self::new_inter6(e0, e1, snorm).inter6_to_inter4()
    }

    fn new_inter6_unorm(c0: u8, c1: u8) -> Self {
        debug_assert!(c0 > c1);
        let c0_f = n8::f32(c0);
        let c1_f = n8::f32(c1);
        Self { c0, c1, c0_f, c1_f }
    }

    fn inter6_to_inter4(&self) -> Self {
        Self {
            c0: self.c1,
            c1: self.c0,
            c0_f: self.c1_f,
            c1_f: self.c0_f,
        }
    }

    fn with_indexes(&self, indexes: IndexList) -> [u8; 8] {
        let index_bytes = indexes.data.to_le_bytes();

        [
            self.c0,
            self.c1,
            index_bytes[0],
            index_bytes[1],
            index_bytes[2],
            index_bytes[3],
            index_bytes[4],
            index_bytes[5],
        ]
    }
}

struct IndexList {
    data: u64,
}
impl IndexList {
    fn new_empty() -> Self {
        Self { data: 0 }
    }
    fn new_all(value: u8) -> Self {
        debug_assert!(value < 8);
        const MASK: u64 = {
            let mut mask: u64 = 0;
            let mut i = 0;
            while i < 16 {
                mask |= 1 << (i * 3);
                i += 1;
            }
            mask
        };

        Self {
            data: (value as u64) * MASK,
        }
    }

    fn get(&self, index: usize) -> u8 {
        debug_assert!(index < 16);
        ((self.data >> (index * 3)) & 0b111) as u8
    }
    fn set(&mut self, index: usize, value: u8) {
        debug_assert!(index < 16);
        debug_assert!(value < 8);
        debug_assert!(self.get(index) == 0, "Cannot set an index twice.");
        self.data |= (value as u64) << (index * 3);
    }
}

trait Palette {
    fn new(c0: f32, c1: f32) -> Self
    where
        Self: Sized;

    fn from_endpoints(endpoints: &EndPoints) -> Self
    where
        Self: Sized,
    {
        Self::new(endpoints.c0_f, endpoints.c1_f)
    }

    /// Returns:
    /// 0: The index value of the closest color in the palette
    /// 1: The closest color in the palette
    /// 2: `abs(pixel - closest)`, aka the absolute error
    fn closest(&self, pixel: f32) -> (u8, f32, f32);

    /// Same as calling `self.closest` for each component of the vector.
    /// Returns:
    /// 0: The index values of the closest colors in the palette
    /// 1: `abs(pixel - closest)`, aka the absolute error
    ///
    /// NOTE: This default implementation doesn't need to be replaced, because
    /// the compiler is good enough at auto-vectorizing. In fact, a custom
    /// implementation is likely slower, since it prevents the compiler for
    /// using AVX instructions.
    fn closest_4(&self, pixels: Vec4) -> (UVec4, Vec4) {
        let (i0, _, e0) = self.closest(pixels.x);
        let (i1, _, e1) = self.closest(pixels.y);
        let (i2, _, e2) = self.closest(pixels.z);
        let (i3, _, e3) = self.closest(pixels.w);

        (
            UVec4::new(i0 as u32, i1 as u32, i2 as u32, i3 as u32),
            Vec4::new(e0, e1, e2, e3),
        )
    }

    /// Returns the index list of the colors in the palette that together
    /// minimize the MSE.
    ///
    /// Returns:
    /// 0: The index list
    /// 1: The total MSE of the block
    ///
    /// Note that the MSE is **NOT** normalized. In other words, the result is
    /// 16x the actual MSE.
    fn block_closest(&self, block: &Block) -> (IndexList, f32) {
        let mut total_error = 0.0;
        let mut index_list = IndexList::new_empty();
        for (pixel_index, pixels) in block.b.iter().enumerate() {
            let (index_value, error) = self.closest_4(*pixels);

            for i in 0..4 {
                index_list.set(pixel_index * 4 + i, index_value[i] as u8);
            }
            total_error += error.dot(error);
        }

        (index_list, total_error)
    }

    /// Returns the mean squared error of the block with the closest colors in
    /// the palette.
    ///
    /// Same as `self.block_closest(block).1`.
    fn block_closest_error_sq(&self, block: &Block) -> f32;

    fn block_dither(&self, block: impl Block4x4<f32>) -> (IndexList, f32) {
        let mut index_list = IndexList::new_empty();
        let mut total_error = 0.0;

        bcn_util::block_dither(block, |pixel_index, pixel| {
            let (index_value, closest, error) = self.closest(pixel);
            index_list.set(pixel_index, index_value);
            total_error += error * error;

            closest
        });

        (index_list, total_error)
    }
}

struct Inter6Palette {
    c0: f32,
    c1: f32,
    factor1: f32,
    factor2: f32,
    add1: f32,
}
impl Inter6Palette {
    const INDEX_MAP: [u8; 8] = [1, 7, 6, 5, 4, 3, 2, 0];
}
impl Palette for Inter6Palette {
    fn new(c0: f32, c1: f32) -> Self {
        debug_assert!(c0 != c1);

        let factor1 = 7.0 / (c0 - c1);
        let factor2 = (1.0 / 7.0) * (c0 - c1);
        let add1 = 0.5 - c1 * factor1;

        Self {
            c0,
            c1,
            factor1,
            factor2,
            add1,
        }
    }

    fn closest(&self, pixel: f32) -> (u8, f32, f32) {
        let blend = pixel * self.factor1 + self.add1;
        let blend7 = ((blend) as u8).min(7);

        let index_value = Self::INDEX_MAP[blend7 as usize];
        let closest = blend7 as f32 * self.factor2 + self.c1;
        let error = pixel - closest;

        (index_value, closest, error.abs())
    }

    fn block_closest_error_sq(&self, block: &Block) -> f32 {
        // compute all min errors in parallel
        let [b0, b1, b2, b3] = block.b;

        // prepare endpoints for interpolation
        let c0 = Vec4::splat(self.c0);
        let cd = Vec4::splat(self.c1 - self.c0);

        // start with c0
        let mut e0 = (c0 - b0).abs();
        let mut e1 = (c0 - b1).abs();
        let mut e2 = (c0 - b2).abs();
        let mut e3 = (c0 - b3).abs();

        // and now the other 7 colors
        const FACTORS: [f32; 7] = [1. / 7., 2. / 7., 3. / 7., 4. / 7., 5. / 7., 6. / 7., 1.];
        for f in FACTORS {
            let c = c0 + cd * f;

            e0 = e0.min((c - b0).abs());
            e1 = e1.min((c - b1).abs());
            e2 = e2.min((c - b2).abs());
            e3 = e3.min((c - b3).abs());
        }

        // e0-3 now contain the min error for all pixels
        // so now just square and add
        let e = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
        e.x + e.y + e.z + e.w
    }
}

struct Inter4Palette {
    colors: [f32; 8],
}
impl Palette for Inter4Palette {
    fn new(c0: f32, c1: f32) -> Self {
        Self {
            colors: [
                c0,
                c1,
                c0 * 0.8 + c1 * 0.2,
                c0 * 0.6 + c1 * 0.4,
                c0 * 0.4 + c1 * 0.6,
                c0 * 0.2 + c1 * 0.8,
                0.0,
                1.0,
            ],
        }
    }

    fn closest(&self, pixel: f32) -> (u8, f32, f32) {
        // this handles the case where pixel is 0 or 1
        let (mut index_value, mut min_error) = if pixel >= 0.5 {
            (7_u8, 1.0 - pixel)
        } else {
            (6_u8, pixel)
        };

        // for the rest, check the palette
        #[allow(clippy::needless_range_loop)]
        for i in 0..6 {
            let error = (pixel - self.colors[i]).abs();
            if error < min_error {
                min_error = error;
                index_value = i as u8;
            }
        }

        (index_value, self.colors[index_value as usize], min_error)
    }

    fn block_closest_error_sq(&self, block: &Block) -> f32 {
        // compute all min errors in parallel
        let [b0, b1, b2, b3] = block.b;

        // since all pixels are in the range 0-1, we can initialize the min
        // error with pixel.min(1.0 - pixel) and then do the other 6 colors
        let mut e0 = b0.min(1.0 - b0);
        let mut e1 = b1.min(1.0 - b1);
        let mut e2 = b2.min(1.0 - b2);
        let mut e3 = b3.min(1.0 - b3);

        // and now the other 6 colors
        for i in 0..6 {
            let c = Vec4::splat(self.colors[i]);

            e0 = e0.min((c - b0).abs());
            e1 = e1.min((c - b1).abs());
            e2 = e2.min((c - b2).abs());
            e3 = e3.min((c - b3).abs());
        }

        // e0-3 now contain the min error for all pixels
        // so now just square and add
        let e = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
        e.x + e.y + e.z + e.w
    }
}
