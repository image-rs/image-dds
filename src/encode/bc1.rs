#![allow(clippy::needless_range_loop)]

use glam::Vec3A;

use crate::{n5, n6};

use super::bcn_util::{self, Block4x4};

#[derive(Debug, Clone, Copy)]
pub struct Bc1Options {
    pub dither: bool,
    /// Setting this to `true` will disable the use of the mode with the default color. This is useful for BC2 and BC3 encoding.
    pub no_default: bool,
    pub alpha_threshold: f32,
}
impl Default for Bc1Options {
    fn default() -> Self {
        Self {
            dither: false,
            no_default: false,
            alpha_threshold: 0.5,
        }
    }
}

/// This is a completely transparent BC1 block in P3 default mode.
///
/// While the last 4 bytes have to be 0xFF, we can chose any u16 values for the
/// endpoints such that c0 < c1. While choosing c0 == c1 is also valid, there
/// are BC1 decoders that do NOT handle c0 == c1 correctly, so this must be
/// avoided.
///
/// I selected c0 = 0 and c1 = 0xFFFF to hopefully make those block easier to
/// compress for gzip and co.
const TRANSPARENT_BLOCK: [u8; 8] = [0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

pub(crate) fn compress_bc1_block(mut block: [[f32; 4]; 16], options: Bc1Options) -> [u8; 8] {
    // clamp 0 to 1
    for pixel in block.iter_mut() {
        for value in pixel.iter_mut() {
            *value = value.clamp(0., 1.);
        }
    }

    compress(block, Uniform, options)
}
fn compress(block: [[f32; 4]; 16], error_metric: impl ErrorMetric, options: Bc1Options) -> [u8; 8] {
    // separate color and alpha
    let colors: [Vec3A; 16] = block.map(|[r, g, b, _]| Vec3A::new(r, g, b));
    let alpha_map = get_alpha_map(&block, options.alpha_threshold);

    // Don't use the default color mode in BC2 and BC3
    if options.no_default {
        return compress_p4(colors, error_metric, options).0;
    }

    if alpha_map == AlphaMap::ALL_TRANSPARENT {
        return TRANSPARENT_BLOCK;
    }

    if alpha_map != AlphaMap::ALL_OPAQUE {
        return compress_p3_default(colors, alpha_map, error_metric, options).0;
    }

    // We have a choice to make. So just try both and pick whichever is better
    let (p4, p4_error) = compress_p4(colors, error_metric, options);
    let (p3, p3_error) = compress_p3_default(colors, alpha_map, error_metric, options);
    if p4_error < p3_error {
        p4
    } else {
        p3
    }
}
fn compress_p4(
    block: [Vec3A; 16],
    error_metric: impl ErrorMetric,
    options: Bc1Options,
) -> ([u8; 8], f32) {
    // single-color optimization
    if let Some(color) = get_single_color(&block) {
        return compress_single_color_p4(color, error_metric, options);
    }

    let (min, max) = get_initial_endpoints(&block);

    let endpoints = EndPoints::new_p4(
        R5G6B5Color::from_color_round(min),
        R5G6B5Color::from_color_round(max),
    );
    let palette = P4Palette::from(&endpoints, error_metric);

    let (indexes, error) = if options.dither {
        palette.block_dither(&block)
    } else {
        palette.block_closest(&block)
    };

    (endpoints.with_indexes(indexes), error)
}
fn compress_p3_default(
    block: [Vec3A; 16],
    alpha_map: AlphaMap,
    error_metric: impl ErrorMetric,
    options: Bc1Options,
) -> ([u8; 8], f32) {
    let mut color_buffer = [Vec3A::ZERO; 16];
    let colors = get_opaque_colors(&block, alpha_map, &mut color_buffer);
    if colors.is_empty() {
        return (TRANSPARENT_BLOCK, 0.0);
    }

    // single-color optimization
    if let Some(color) = get_single_color(colors) {
        return compress_single_color_p3(color, alpha_map, error_metric, options);
    }

    // general case
    let (min, max) = get_initial_endpoints(colors);

    let endpoints = EndPoints::new_p3_default(
        R5G6B5Color::from_color_round(min),
        R5G6B5Color::from_color_round(max),
    );
    let palette = P3Palette::from(&endpoints, error_metric);

    let (indexes, error) = if options.dither {
        palette.block_dither(&block, alpha_map)
    } else {
        palette.block_closest(&block, alpha_map)
    };

    (endpoints.with_indexes(indexes), error)
}

fn get_single_color(colors: &[Vec3A]) -> Option<Vec3A> {
    if colors.is_empty() {
        return None;
    }

    let mut min = colors[0];
    let mut max = colors[0];
    for c in colors {
        min = min.min(*c);
        max = max.max(*c);
    }

    const BC1_EPSILON: f32 = 1.0 / 255.0 / 2.0;
    let diff = (max - min).abs();
    if diff.max_element() < BC1_EPSILON {
        Some((min + max) * 0.5)
    } else {
        None
    }
}
fn compress_single_color_p4(
    color: Vec3A,
    error_metric: impl ErrorMetric,
    options: Bc1Options,
) -> ([u8; 8], f32) {
    let min = R5G6B5Color::from_color_floor(color);
    let max = R5G6B5Color::from_color_ceil(color);

    if min == max {
        // Lucky. The color can be presented exactly by a RGB565 color.
        let endpoints = EndPoints::new_p4(R5G6B5Color::BLACK, max);
        let palette = P4Palette::from(&endpoints, error_metric);
        let (indexes, error) = palette.block_closest(color);
        return (endpoints.with_indexes(indexes), error);
    }

    let mut candidates = CandidateList::new(error_metric, options.dither);

    // add baseline
    candidates.add_p4(color, min, max);

    // Without dithering, we might be able to find a single interpolation that
    // approximates the target color more closely.
    let (min, max) = find_optimal_single_color_endpoints(color, 1. / 3.);
    candidates.add_p4(color, min, max);

    let (min, max) = find_optimal_single_color_endpoints(color, 2. / 3.);
    candidates.add_p4(color, min, max);

    candidates.get_best()
}
fn compress_single_color_p3(
    color: Vec3A,
    alpha_map: AlphaMap,
    error_metric: impl ErrorMetric,
    options: Bc1Options,
) -> ([u8; 8], f32) {
    let min = R5G6B5Color::from_color_floor(color);
    let max = R5G6B5Color::from_color_ceil(color);

    if min == max {
        // Lucky. The color can be presented exactly by a RGB565 color.
        let endpoints = EndPoints::new_p3_default(R5G6B5Color::BLACK, max);
        let palette = P3Palette::from(&endpoints, error_metric);
        let (indexes, error) = palette.block_closest(color, alpha_map);
        return (endpoints.with_indexes(indexes), error);
    }

    let mut candidates = CandidateList::new(error_metric, options.dither);

    // add baseline
    candidates.add_p3(color, alpha_map, min, max);

    // Without dithering, we might be able to find a single interpolation that
    // approximates the target color more closely.
    let (min, max) = find_optimal_single_color_endpoints(color, 0.5);
    candidates.add_p3(color, alpha_map, min, max);

    candidates.get_best()
}
fn find_optimal_single_color_endpoints(color: Vec3A, weight: f32) -> (R5G6B5Color, R5G6B5Color) {
    /// This finds the optimal endpoints `(c0, c1)` such that:
    ///     |color - c0/max * weight - c1/max * (1 - weight)|
    /// is minimized.
    fn optimal_channel(color: f32, weight: f32, max: u8) -> (u8, u8) {
        let c0_max = max.min((color * max as f32) as u8);

        let w0 = weight / max as f32;
        let w1 = (1.0 - weight) / max as f32;
        let error_weight = 0.03 / max as f32;
        let get_error = |c0: u8, c1: u8| {
            // The spec affords BC1 a +-3% * |c0 - c1| error margin
            let error = (c1 as f32 - c0 as f32) * error_weight;
            let reference = c0 as f32 * w0 + c1 as f32 * w1;
            let i0 = reference - error;
            let i1 = reference + error;
            f32::max((i0 - color).abs(), (i1 - color).abs())
        };

        let mut best_c0: u8 = 0;
        let mut best_c1: u8 = 0;
        let mut best_error = f32::INFINITY;

        for c0 in 0..=c0_max {
            let c1_ideal = (color - c0 as f32 * w0) / w1;
            let c1_floor = max.min(c1_ideal as u8);
            let c1_ceil = max.min(c1_floor + 1);

            let error_floor = get_error(c0, c1_floor);
            if error_floor < best_error {
                best_c0 = c0;
                best_c1 = c1_floor;
                best_error = error_floor;
            }

            let error_ceil = get_error(c0, c1_ceil);
            if error_ceil < best_error {
                best_c0 = c0;
                best_c1 = c1_ceil;
                best_error = error_ceil;
            }
        }

        debug_assert!(best_error.is_finite());
        debug_assert!(best_c0 <= best_c1 && best_c1 <= max);

        (best_c0, best_c1)
    }

    let r = optimal_channel(color.x, weight, 31);
    let g = optimal_channel(color.y, weight, 63);
    let b = optimal_channel(color.z, weight, 31);

    let min = R5G6B5Color::new(r.0, g.0, b.0);
    let max = R5G6B5Color::new(r.1, g.1, b.1);

    (min, max)
}

fn get_opaque_colors<'a>(
    block: &'a [Vec3A; 16],
    alpha_map: AlphaMap,
    buffer: &'a mut [Vec3A; 16],
) -> &'a [Vec3A] {
    if alpha_map == AlphaMap::ALL_OPAQUE {
        return block;
    }
    let mut count = 0;
    for (i, &pixel) in block.iter().enumerate() {
        if alpha_map.is_opaque(i) {
            buffer[count] = pixel;
            count += 1;
        }
    }
    &buffer[..count]
}

fn get_initial_endpoints(colors: &[Vec3A]) -> (Vec3A, Vec3A) {
    debug_assert!(colors.len() <= 16);

    let mut min = colors[0];
    let mut max = colors[0];
    for pixel in colors.iter().skip(1) {
        min = min.min(*pixel);
        max = max.max(*pixel);
    }
    (min, max)
}

fn get_alpha_map(block: &[[f32; 4]], alpha_threshold: f32) -> AlphaMap {
    let mut alpha_map = AlphaMap::ALL_TRANSPARENT;
    for (i, pixel) in block.iter().enumerate() {
        if pixel[3] >= alpha_threshold {
            alpha_map.set_opaque(i);
        }
    }
    alpha_map
}

struct CandidateList<E> {
    data: [u8; 8],
    error: f32,
    error_metric: E,
    dithering: bool,
}
impl<E> CandidateList<E> {
    fn new(error_metric: E, dithering: bool) -> Self {
        Self {
            data: [0; 8],
            error: f32::INFINITY,
            error_metric,
            dithering,
        }
    }
    fn get_best(self) -> ([u8; 8], f32) {
        debug_assert!(self.error.is_finite());
        (self.data, self.error)
    }

    fn add_p4(&mut self, block: impl Block4x4<Vec3A> + Copy, e0: R5G6B5Color, e1: R5G6B5Color)
    where
        E: ErrorMetric,
    {
        let endpoints = EndPoints::new_p4(e0, e1);
        let palette = P4Palette::from(&endpoints, self.error_metric);

        let (indexes, error) = if self.dithering {
            palette.block_dither(block)
        } else {
            palette.block_closest(block)
        };

        if error < self.error {
            self.data = endpoints.with_indexes(indexes);
            self.error = error;
        }
    }
    fn add_p3(
        &mut self,
        block: impl Block4x4<Vec3A> + Copy,
        alpha_map: AlphaMap,
        e0: R5G6B5Color,
        e1: R5G6B5Color,
    ) where
        E: ErrorMetric,
    {
        let endpoints = EndPoints::new_p3_default(e0, e1);
        let palette = P3Palette::from(&endpoints, self.error_metric);

        let (indexes, error) = if self.dithering {
            palette.block_dither(block, alpha_map)
        } else {
            palette.block_closest(block, alpha_map)
        };

        if error < self.error {
            self.data = endpoints.with_indexes(indexes);
            self.error = error;
        }
    }
}

#[derive(Debug, Clone)]
struct EndPoints {
    c0: R5G6B5Color,
    c1: R5G6B5Color,
    c0_f: Vec3A,
    c1_f: Vec3A,
}
impl EndPoints {
    fn new_p4(mut c0: R5G6B5Color, mut c1: R5G6B5Color) -> Self {
        let c0_u = c0.to_u16();
        let c1_u = c1.to_u16();
        #[allow(clippy::comparison_chain)]
        if c0_u < c1_u {
            std::mem::swap(&mut c0, &mut c1);
        } else if c0_u == c1_u {
            // change the b channel
            if c1.b == 0 {
                c0.b = 1;
            } else {
                c1.b -= 1;
            }
        }

        debug_assert!(c0.to_u16() > c1.to_u16());

        let c0_f = c0.to_color();
        let c1_f = c1.to_color();
        Self { c0, c1, c0_f, c1_f }
    }
    fn new_p3_default(mut c0: R5G6B5Color, mut c1: R5G6B5Color) -> Self {
        let c0_u = c0.to_u16();
        let c1_u = c1.to_u16();
        if c0_u > c1_u {
            std::mem::swap(&mut c0, &mut c1);
        }

        debug_assert!(c0.to_u16() <= c1.to_u16());

        let c0_f = c0.to_color();
        let c1_f = c1.to_color();
        Self { c0, c1, c0_f, c1_f }
    }

    fn with_indexes(&self, indexes: IndexList) -> [u8; 8] {
        let c0 = self.c0.to_u16().to_le_bytes();
        let c1 = self.c1.to_u16().to_le_bytes();
        let [i0, i1, i2, i3] = indexes.data.to_le_bytes();

        [c0[0], c0[1], c1[0], c1[1], i0, i1, i2, i3]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct R5G6B5Color {
    r: u8,
    g: u8,
    b: u8,
}
impl R5G6B5Color {
    const BLACK: Self = Self::new(0, 0, 0);
    const WHITE: Self = Self::new(31, 63, 31);

    const fn new(r: u8, g: u8, b: u8) -> Self {
        debug_assert!(r < 32);
        debug_assert!(g < 64);
        debug_assert!(b < 32);

        Self { r, g, b }
    }

    fn from_color_round(color: Vec3A) -> Self {
        let r = n5::from_f32(color.x);
        let g = n6::from_f32(color.y);
        let b = n5::from_f32(color.z);
        Self::new(r, g, b)
    }
    fn from_color_floor(color: Vec3A) -> Self {
        let r = (color.x.min(1.0) * 31.0) as u8;
        let g = (color.y.min(1.0) * 63.0) as u8;
        let b = (color.z.min(1.0) * 31.0) as u8;
        Self::new(r, g, b)
    }
    fn from_color_ceil(color: Vec3A) -> Self {
        let r = 31 - ((1.0 - color.x).min(1.0) * 31.0) as u8;
        let g = 63 - ((1.0 - color.y).min(1.0) * 63.0) as u8;
        let b = 31 - ((1.0 - color.z).min(1.0) * 31.0) as u8;
        Self::new(r, g, b)
    }
    fn to_color(self) -> Vec3A {
        self.debug_check();
        Vec3A::new(n5::f32(self.r), n6::f32(self.g), n5::f32(self.b))
    }

    fn from_u16(q: u16) -> Self {
        Self {
            r: ((q >> 11) & 0b11111) as u8,
            g: ((q >> 5) & 0b111111) as u8,
            b: (q & 0b11111) as u8,
        }
    }
    fn to_u16(self) -> u16 {
        self.debug_check();
        (self.r as u16) << 11 | (self.g as u16) << 5 | self.b as u16
    }

    fn debug_check(&self) {
        debug_assert!(self.r < 32);
        debug_assert!(self.g < 64);
        debug_assert!(self.b < 32);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AlphaMap {
    data: u16,
}
impl AlphaMap {
    const ALL_TRANSPARENT: Self = Self { data: 0 };
    const ALL_OPAQUE: Self = Self { data: u16::MAX };

    fn set_transparent(&mut self, index: usize) {
        self.data &= !(1 << index);
    }
    fn set_opaque(&mut self, index: usize) {
        self.data |= 1 << index;
    }

    fn is_transparent(&self, index: usize) -> bool {
        (self.data & (1 << index)) == 0
    }
    fn is_opaque(&self, index: usize) -> bool {
        !self.is_transparent(index)
    }

    fn transparent_count(&self) -> u32 {
        self.data.count_zeros()
    }
}

struct IndexList {
    data: u32,
}
impl IndexList {
    fn new_empty() -> Self {
        Self { data: 0 }
    }

    fn get(&self, index: usize) -> u8 {
        debug_assert!(index < 16);
        ((self.data >> (index * 2)) & 0b11) as u8
    }
    fn set(&mut self, index: usize, value: u8) {
        debug_assert!(index < 16);
        debug_assert!(value < 4);
        debug_assert!(self.get(index) == 0, "Cannot set an index twice.");
        self.data |= (value as u32) << (index * 2);
    }
}

struct P4Palette<E> {
    colors: [Vec3A; 4],
    error_metric: E,
}
impl<E: ErrorMetric> P4Palette<E> {
    fn new(c0: Vec3A, c1: Vec3A, error_metric: E) -> Self {
        Self {
            colors: [
                c0,
                c1,
                c0 * (2. / 3.) + c1 * (1. / 3.),
                c0 * (1. / 3.) + c1 * (2. / 3.),
            ],
            error_metric,
        }
    }
    fn from(endpoints: &EndPoints, error_metric: E) -> Self {
        Self::new(endpoints.c0_f, endpoints.c1_f, error_metric)
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
    fn block_closest(&self, block: impl Block4x4<Vec3A>) -> (IndexList, f32) {
        let mut total_error = 0.0;
        let mut index_list = IndexList::new_empty();
        for pixel_index in 0..16 {
            let pixel = block.get_pixel_at(pixel_index);
            let (index_value, _, error) = self.closest(pixel);
            index_list.set(pixel_index, index_value);
            total_error += error * error;
        }

        (index_list, total_error)
    }

    fn block_dither(&self, block: impl Block4x4<Vec3A> + Copy) -> (IndexList, f32) {
        let mut index_list = IndexList::new_empty();
        let mut total_error = 0.0;

        // This implements a modified version of the Floyd-Steinberg dithering
        bcn_util::block_dither(block, |pixel_index, pixel| {
            let (index_value, closest, error) = self.closest(pixel);
            index_list.set(pixel_index, index_value);
            total_error += error * error;
            closest
        });

        (index_list, total_error)
    }
}
impl<E: ErrorMetric> Palette<4> for P4Palette<E> {
    type E = E;

    fn get_colors(&self) -> &[Vec3A; 4] {
        &self.colors
    }
    fn get_error_metric(&self) -> Self::E {
        self.error_metric
    }
}

struct P3Palette<E> {
    colors: [Vec3A; 3],
    error_metric: E,
}
impl<E: ErrorMetric> P3Palette<E> {
    const DEFAULT: u8 = 3;

    fn new(c0: Vec3A, c1: Vec3A, error_metric: E) -> Self {
        Self {
            colors: [c0, c1, (c0 + c1) * 0.5],
            error_metric,
        }
    }
    fn from(endpoints: &EndPoints, error_metric: E) -> Self {
        Self::new(endpoints.c0_f, endpoints.c1_f, error_metric)
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
    fn block_closest(&self, block: impl Block4x4<Vec3A>, alpha_map: AlphaMap) -> (IndexList, f32) {
        let mut total_error = 0.0;
        let mut index_list = IndexList::new_empty();
        for pixel_index in 0..16 {
            let pixel = block.get_pixel_at(pixel_index);
            if alpha_map.is_opaque(pixel_index) {
                let (index_value, _, error) = self.closest(pixel);
                index_list.set(pixel_index, index_value);
                total_error += error * error;
            } else {
                index_list.set(pixel_index, Self::DEFAULT);
            }
        }

        (index_list, total_error)
    }

    fn block_dither(
        &self,
        block: impl Block4x4<Vec3A> + Copy,
        alpha_map: AlphaMap,
    ) -> (IndexList, f32) {
        let mut index_list = IndexList::new_empty();
        let mut total_error = 0.0;

        // This implements a modified version of the Floyd-Steinberg dithering
        bcn_util::block_dither(block, |pixel_index, pixel| {
            if alpha_map.is_opaque(pixel_index) {
                let (index_value, closest, error) = self.closest(pixel);
                index_list.set(pixel_index, index_value);
                total_error += error * error;
                closest
            } else {
                index_list.set(pixel_index, Self::DEFAULT);
                block.get_pixel_at(pixel_index)
            }
        });

        (index_list, total_error)
    }
}
impl<E: ErrorMetric> Palette<3> for P3Palette<E> {
    type E = E;

    fn get_colors(&self) -> &[Vec3A; 3] {
        &self.colors
    }
    fn get_error_metric(&self) -> Self::E {
        self.error_metric
    }
}

trait Palette<const N: usize> {
    type E: ErrorMetric;

    fn get_colors(&self) -> &[Vec3A; N];
    fn get_error_metric(&self) -> Self::E;

    /// Returns:
    /// 0: The index value of the closest color in the palette
    /// 1: The closest color in the palette
    /// 2: `(pixel - closest) ** 2`, aka the squared error
    fn closest(&self, color: Vec3A) -> (u8, Vec3A, f32) {
        let colors = self.get_colors();
        let error_metric = self.get_error_metric();

        let mut best_index = 0;
        let mut min_error = error_metric.error_sq(color, colors[0]);
        for i in 1..N {
            let error = error_metric.error_sq(color, colors[i]);
            if error < min_error {
                best_index = i as u8;
                min_error = error;
            }
        }

        (best_index, colors[best_index as usize], min_error)
    }

    /// Returns the square of the error between the pixel and the closest color
    /// in the palette.
    ///
    /// Same as `self.closest(pixel).2`.
    fn closest_error_sq(&self, color: Vec3A) -> f32 {
        let colors = self.get_colors();
        let error_metric = self.get_error_metric();

        let mut min_error = error_metric.error_sq(color, colors[0]);
        for i in 1..N {
            let error = error_metric.error_sq(color, colors[i]);
            min_error = min_error.min(error);
        }

        min_error
    }
}

trait ErrorMetric: Copy {
    /// Returns the square of the error between the two colors.
    fn error_sq(&self, a: Vec3A, b: Vec3A) -> f32;
}
#[derive(Debug, Clone, Copy)]
struct Uniform;
impl ErrorMetric for Uniform {
    fn error_sq(&self, a: Vec3A, b: Vec3A) -> f32 {
        let diff = a - b;
        diff.dot(diff) * (1. / 3.)
    }
}
#[derive(Debug, Clone, Copy)]
struct Perceptual;
impl ErrorMetric for Perceptual {
    fn error_sq(&self, a: Vec3A, b: Vec3A) -> f32 {
        let diff = a - b;
        diff.x * diff.x * 0.299 + diff.y * diff.y * 0.587 + diff.z * diff.z * 0.114
    }
}
