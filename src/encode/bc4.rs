use crate::{n8, s8};

use super::bcn_util;

#[derive(Debug, Clone, Copy)]
pub struct Bc4Options {
    pub dither: bool,
    pub snorm: bool,
}

/// The smallest non-zero value that can be represented in a BC4 block.
///
/// This is also the smallest distance of 2 adjacent representable values.
const BC4_MIN_VALUE: f32 = 1. / (255. * 7.);
/// 2 values that are this close will be considered equal.
const BC4_EPSILON: f32 = 1. / (65536.);

pub(crate) fn compress_bc4_block(mut block: [f32; 16], options: Bc4Options) -> [u8; 8] {
    // clamp to 0-1
    block.iter_mut().for_each(|x| *x = x.clamp(0.0, 1.0));

    let mut min = block[0];
    let mut max = block[0];
    for value in block {
        min = min.min(value);
        max = max.max(value);
    }
    let diff = max - min;

    // reference for testing
    // if !options.dither && !options.snorm {
    //     return reference_brute_force(block);
    // }

    // single color
    if diff < BC4_EPSILON {
        let value = (min + max) * 0.5;
        return single_color(value, options);
    }

    // If the colors are far away from 0 and 1, then inter6 should always be better
    // than inter4
    const INTER6_THRESHOLD: f32 = 1. / 7.;
    if 0. < min - diff * INTER6_THRESHOLD && max + diff * INTER6_THRESHOLD < 1. {
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
#[allow(unused)]
fn reference_brute_force(block: [f32; 16]) -> [u8; 8] {
    let mut best = [0_u8; 8];
    let mut best_error = f32::INFINITY;

    for min in 0..255 {
        for max in (min + 1)..=255 {
            let endpoints6 = EndPoints::new_inter6_unorm(max, min);
            let palette6 = Inter6Palette::from_endpoints(&endpoints6);
            let (indexes6, error6) = palette6.block_closest(&block);
            if error6 * error6 < best_error {
                best = endpoints6.with_indexes(indexes6);
                best_error = error6 * error6;
            }

            let endpoints4 = endpoints6.inter6_to_inter4();
            let palette4 = Inter4Palette::from_endpoints(&endpoints4);
            let (indexes4, error4) = palette4.block_closest(&block);
            if error4 * error4 < best_error {
                best = endpoints4.with_indexes(indexes4);
                best_error = error4 * error4;
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

    let endpoints6 = EndPoints::new_inter6(value, value, options.snorm);
    let palette6 = Inter6Palette::from_endpoints(&endpoints6);

    if options.dither {
        let (indexes, _) = palette6.block_dither(&[value; 16]);
        endpoints6.with_indexes(indexes)
    } else {
        // pick the best palette
        let endpoints4 = EndPoints::new_inter4(value, value, options.snorm);
        let palette4 = Inter4Palette::from_endpoints(&endpoints4);

        let (index_value4, _, error4) = palette4.closest(value);
        let (index_value6, _, error6) = palette6.closest(value);

        if error4.abs() < error6.abs() {
            endpoints4.with_indexes(IndexList::new_all(index_value4))
        } else {
            endpoints6.with_indexes(IndexList::new_all(index_value6))
        }
    }
}

fn refine_endpoints(
    mut min: f32,
    mut max: f32,
    mut compute_error: impl FnMut((f32, f32)) -> f32,
    mut quantize: impl FnMut((f32, f32)) -> (f32, f32),
) -> (f32, f32) {
    // Step 1: Improve the endpoints with a local search
    const STEP_DECAY: f32 = 0.5;
    const MIN_STEP: f32 = 1. / 255. / 2.;
    const INITIAL_PORTION: f32 = 0.15;
    const _ITERATIONS: u32 = {
        let mut count: u32 = 0;
        let mut step = INITIAL_PORTION;
        while step > MIN_STEP {
            count += 1;
            step *= STEP_DECAY;
        }
        count
    };

    let mut step = INITIAL_PORTION * (max - min);
    let mut error = compute_error((min, max));
    while step > MIN_STEP {
        for (delta_min, delta_max) in [(step, 0.0), (0.0, step), (-step, 0.0), (0.0, -step)] {
            let new_min = (min + delta_min).clamp(0.0, 1.0);
            let new_max = (max + delta_max).clamp(0.0, 1.0);
            if new_min < new_max {
                let new_error = compute_error((new_min, new_max));
                if new_error < error {
                    error = new_error;
                    min = new_min;
                    max = new_max;
                }
            }
        }
        step *= STEP_DECAY;
    }

    // Step 2: Quantize the endpoints and select the best
    const QUANT_STEP: f32 = 1. / 254. + 0.0001;
    error = compute_error(quantize((min, max)));
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
    block: &[f32; 16],
    _options: Bc4Options,
) -> impl Fn((f32, f32)) -> f32 + '_ {
    move |(min, max)| {
        let palette = P::new(min, max);
        // TODO: find a better error metric for dithered blocks
        palette.block_closest_mse(block)
    }
}

fn compress_inter6(
    block: &[f32; 16],
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

fn compress_inter4(block: &[f32; 16], options: Bc4Options) -> ([u8; 8], f32) {
    let mut min: f32 = 1.0;
    let mut max: f32 = 0.0;
    for &value in block {
        if value > BC4_MIN_VALUE {
            min = min.min(value);
        }
        if value < 1.0 - BC4_MIN_VALUE {
            max = max.max(value);
        }
    }

    (min, max) = refine_endpoints(
        min,
        max,
        refinement_error_metric::<Inter4Palette>(block, options),
        move |(min, max)| {
            let endpoints = EndPoints::new_inter4(min, max, options.snorm);
            (endpoints.c0_f, endpoints.c1_f)
        },
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
        assert!(c0 > c1);
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

    /// Returns:
    /// 0: The index value of the closest color in the palette
    /// 1: The closest color in the palette
    /// 2: `pixel - closest`, aka the (signed) error
    fn closest(&self, pixel: f32) -> (u8, f32, f32);

    /// Returns the square of the error between the pixel and the closest color
    /// in the palette.
    ///
    /// Same as `self.closest(pixel).2.powi(2)`.
    fn closest_error_sq(&self, pixel: f32) -> f32;

    fn from_endpoints(endpoints: &EndPoints) -> Self
    where
        Self: Sized,
    {
        Self::new(endpoints.c0_f, endpoints.c1_f)
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
    fn block_closest(&self, block: &[f32; 16]) -> (IndexList, f32) {
        let mut total_error = 0.0;
        let mut index_list = IndexList::new_empty();
        for (pixel_index, pixel) in block.iter().copied().enumerate() {
            let (index_value, _, error) = self.closest(pixel);

            index_list.set(pixel_index, index_value);
            total_error += error * error;
        }

        (index_list, total_error)
    }

    /// Returns the mean squared error of the block with the closest colors in
    /// the palette.
    ///
    /// Same as `self.block_closest(block).1`.
    fn block_closest_mse(&self, block: &[f32; 16]) -> f32 {
        block
            .iter()
            .copied()
            .map(|pixel| self.closest_error_sq(pixel))
            .sum()
    }

    fn block_dither(&self, block: &[f32; 16]) -> (IndexList, f32) {
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
}
impl Inter6Palette {
    const INDEX_MAP: [u8; 8] = [1, 7, 6, 5, 4, 3, 2, 0];
}
impl Palette for Inter6Palette {
    fn new(c0: f32, c1: f32) -> Self {
        Self { c0, c1 }
    }

    fn closest(&self, pixel: f32) -> (u8, f32, f32) {
        let blend = (pixel - self.c1) / (self.c0 - self.c1);
        let blend7 = ((blend * 7.0 + 0.5) as u8).min(7);

        let index_value = Self::INDEX_MAP[blend7 as usize];
        let closest = blend7 as f32 * (1.0 / 7.0) * (self.c0 - self.c1) + self.c1;
        let error = pixel - closest;

        (index_value, closest, error)
    }
    fn closest_error_sq(&self, pixel: f32) -> f32 {
        let (_, _, error) = self.closest(pixel);
        error * error
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
            let error = pixel - self.colors[i];
            if error.abs() < min_error.abs() {
                min_error = error;
                index_value = i as u8;
            }
        }

        (index_value, self.colors[index_value as usize], min_error)
    }

    fn closest_error_sq(&self, pixel: f32) -> f32 {
        // this handles the case where pixel is 0 or 1
        let zero_one_error = pixel.min(1.0 - pixel);

        let mut min_sq_error = zero_one_error * zero_one_error;

        // for the rest, check the palette
        #[allow(clippy::needless_range_loop)]
        for i in 0..6 {
            let mut error = pixel - self.colors[i];
            error *= error;
            if error < min_sq_error {
                min_sq_error = error;
            }
        }

        min_sq_error
    }
}
