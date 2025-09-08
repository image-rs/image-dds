#![allow(clippy::needless_range_loop)]

use glam::Vec3A;

use crate::{fast_oklab_to_srgb, fast_srgb_to_oklab, n5, n6, util::unlikely_branch};

use super::bcn_util::{self, Block4x4, ColorSpace};

const ALPHA_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub(crate) struct Bc1Options {
    pub dither: bool,
    /// Setting this to `true` will disable the use of the mode with the default color. This is useful for BC2 and BC3 encoding.
    pub no_p3_default: bool,
    pub perceptual: bool,
    pub opaque_always_p4: bool,
    pub refine_max_iter: u8,
    pub quantization: Quantization,
}
impl Default for Bc1Options {
    fn default() -> Self {
        Self {
            dither: false,
            no_p3_default: false,
            perceptual: false,
            opaque_always_p4: false,
            refine_max_iter: 10,
            quantization: Quantization::ChannelWise,
        }
    }
}
impl Bc1Options {
    /// If `true`, then the encoder will assume that all alpha values are 1.0
    fn assume_opaque_alpha(&self) -> bool {
        // If no_p3_default is set (BC2, BC3), then we must ignore the alpha channel.
        // This is done most easily by just assuming that all alpha values are 1.0.
        self.no_p3_default
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

pub(crate) fn compress_bc1_block(block: [[f32; 4]; 16], options: Bc1Options) -> [u8; 8] {
    // determine the binary alpha of each pixel
    let alpha_map = if options.assume_opaque_alpha() {
        AlphaMap::ALL_OPAQUE
    } else {
        get_alpha_map(&block)
    };

    // If the alpha if all pixels is transparent, then there is no point in
    // looking at their colors. Just return the prepared transparent block.
    if alpha_map == AlphaMap::ALL_TRANSPARENT {
        debug_assert!(!options.no_p3_default);
        return TRANSPARENT_BLOCK;
    }

    // map the block to the right format and clamp to [0, 1]
    let colors: [Vec3A; 16] =
        block.map(|[r, g, b, _]| Vec3A::new(r, g, b).clamp(Vec3A::ZERO, Vec3A::ONE));

    // compress based on error metric
    if options.perceptual {
        compress(&colors, alpha_map, Perceptual, options)
    } else {
        compress(&colors, alpha_map, Uniform, options)
    }
}
fn compress(
    block: &[Vec3A; 16],
    alpha_map: AlphaMap,
    error_metric: impl ErrorMetric,
    options: Bc1Options,
) -> [u8; 8] {
    debug_assert!(alpha_map != AlphaMap::ALL_TRANSPARENT);

    // If there are transparent pixels, then we have no choice but to use P3.
    if alpha_map != AlphaMap::ALL_OPAQUE {
        debug_assert!(!options.no_p3_default);
        return compress_p3_default(block, alpha_map, error_metric, options).0;
    }

    // If the options demand P4, then use P4.
    if options.no_p3_default || options.opaque_always_p4 {
        return compress_p4(block, error_metric, options).0;
    }

    // Otherwise, use both and pick whichever is better.
    let (p4, p4_error) = compress_p4(block, error_metric, options);
    let (p3, p3_error) = compress_p3_default(block, alpha_map, error_metric, options);
    if p4_error < p3_error {
        p4
    } else {
        p3
    }
}
fn compress_p4(
    block: &[Vec3A; 16],
    error_metric: impl ErrorMetric,
    options: Bc1Options,
) -> ([u8; 8], f32) {
    let palette_info = PaletteInfo::new_p4(error_metric, options);
    compress_with_palette(block, AlphaMap::ALL_OPAQUE, options, palette_info)
}
fn compress_p3_default(
    block: &[Vec3A; 16],
    alpha_map: AlphaMap,
    error_metric: impl ErrorMetric,
    options: Bc1Options,
) -> ([u8; 8], f32) {
    let palette_info = PaletteInfo::new_p3(error_metric, options);
    compress_with_palette(block, alpha_map, options, palette_info)
}
fn compress_with_palette(
    block: &[Vec3A; 16],
    alpha_map: AlphaMap,
    options: Bc1Options,
    palette_info: PaletteInfo<impl ErrorMetric>,
) -> ([u8; 8], f32) {
    // single-color optimization
    if let Some(color) = get_single_color(block, alpha_map) {
        return compress_single_color(color, alpha_map, palette_info);
    }

    // From now on, we work in the color space of the error metric
    let block = block.map(|p| palette_info.error_metric.srgb_to_color_space(p));

    // The general approach is as follows:
    // 1. Find decent initial endpoints. This is currently done by fitting a line
    //    through the colors and then projecting the colors onto that line to find
    //    the reasonable min and max.
    // 2. Refine the endpoints. The basic idea is to try small variations of the
    //    endpoints and see if that improves the error. This is repeated with
    //    smaller and smaller variations until a maximum number of iterations is
    //    reached or the variations are too small.
    // 3. Quantize the endpoints to R5G6B5.
    // 4. Given the quantized endpoints, find the best matching indices to
    //    finalize the BC1 encoded block.

    let (mut min, mut max) = get_initial_endpoints_from(&block, alpha_map);
    (min, max) = refine(min, max, &block, alpha_map, options, palette_info);
    let quantization = options.quantization;
    let endpoints = pick_best_quantization(min, max, &block, alpha_map, quantization, palette_info);

    let (indexes, error) = palette_info.block(&endpoints, &block, alpha_map);

    (endpoints.with_indexes(indexes), error)
}

fn refine(
    min: ColorSpace,
    max: ColorSpace,
    block: impl Block4x4<ColorSpace> + Copy,
    alpha_map: AlphaMap,
    options: Bc1Options,
    palette_info: PaletteInfo<impl ErrorMetric>,
) -> (ColorSpace, ColorSpace) {
    let min_max_dist = min.0.distance(max.0);
    let max_iter = options.refine_max_iter as u32;
    let refine_options = bcn_util::RefinementOptions::new_bc1(min_max_dist, max_iter);

    bcn_util::refine_endpoints(min, max, refine_options, move |(min, max)| {
        let error_metric = palette_info.error_metric;

        let min = error_metric.color_space_to_srgb(min);
        let max = error_metric.color_space_to_srgb(max);
        let c0 = R5G6B5Color::from_color_round(min);
        let c1 = R5G6B5Color::from_color_round(max);

        if palette_info.mode == PaletteMode::P4 {
            let endpoints = EndPoints::new_p4(c0, c1);
            let palette = Palette::new_p4(&endpoints, error_metric);
            palette.block_closest_error_p4(block)
        } else {
            let endpoints = EndPoints::new_p3_default(c0, c1);
            let palette = Palette::new_p3(&endpoints, error_metric);
            palette.block_closest_error_p3(block, alpha_map)
        }
    })
}

fn get_single_color(block: &[Vec3A; 16], alpha_map: AlphaMap) -> Option<Vec3A> {
    if block.is_empty() {
        return None;
    }

    let mut min = block[0];
    let mut max = block[0];
    for i in 0..16 {
        let c = block[i];
        if alpha_map.is_opaque(i) {
            min = min.min(c);
            max = max.max(c);
        }
    }

    const BC1_EPSILON: f32 = 1.0 / 255.0 / 2.0;
    let diff = (max - min).abs();
    if diff.max_element() < BC1_EPSILON {
        Some((min + max) * 0.5)
    } else {
        None
    }
}
fn compress_single_color(
    color: Vec3A,
    alpha_map: AlphaMap,
    palette_info: PaletteInfo<impl ErrorMetric>,
) -> ([u8; 8], f32) {
    let min = R5G6B5Color::from_color_floor(color);
    let max = R5G6B5Color::from_color_ceil(color);

    let in_color_space = palette_info.error_metric.srgb_to_color_space(color);

    if min == max {
        // Lucky. The color can be presented exactly by a RGB565 color.
        let endpoints = palette_info.create_endpoints(R5G6B5Color::BLACK, max);
        let (indexes, error) = palette_info.block(&endpoints, in_color_space, alpha_map);
        return (endpoints.with_indexes(indexes), error);
    }

    let mut candidates = CandidateList::new();

    // add baseline
    candidates.add(in_color_space, alpha_map, min, max, palette_info);

    // Without dithering, we might be able to find a single interpolation that
    // approximates the target color more closely.
    if palette_info.mode == PaletteMode::P4 {
        let (min, max) = find_optimal_single_color_endpoints(color, 1. / 3.);
        candidates.add(in_color_space, alpha_map, min, max, palette_info);
        let (min, max) = find_optimal_single_color_endpoints(color, 2. / 3.);
        candidates.add(in_color_space, alpha_map, min, max, palette_info);
    } else {
        let (min, max) = find_optimal_single_color_endpoints(color, 0.5);
        candidates.add(in_color_space, alpha_map, min, max, palette_info);
    }

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
    block: &'a [ColorSpace; 16],
    alpha_map: AlphaMap,
    buffer: &'a mut [ColorSpace; 16],
) -> &'a [ColorSpace] {
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

fn pick_best_quantization(
    c0: ColorSpace,
    c1: ColorSpace,
    block: impl Block4x4<ColorSpace> + Copy,
    alpha_map: AlphaMap,
    quantization: Quantization,
    palette_info: PaletteInfo<impl ErrorMetric>,
) -> EndPoints {
    let error_metric = palette_info.error_metric;

    let c0_f = error_metric.color_space_to_srgb(c0);
    let c1_f = error_metric.color_space_to_srgb(c1);

    let (c0, c1) = quantization.pick_best(c0_f, c1_f, move |c0, c1| {
        if palette_info.dither {
            unlikely_branch();
            let endpoints = palette_info.create_endpoints(c0, c1);
            let palette = palette_info.create_palette(&endpoints);
            return palette.block_dither(block, alpha_map).1;
        }

        if palette_info.mode == PaletteMode::P4 {
            let endpoints = EndPoints::new_p4(c0, c1);
            let palette = Palette::new_p4(&endpoints, error_metric);
            palette.block_closest_error_p4(block)
        } else {
            let endpoints = EndPoints::new_p3_default(c0, c1);
            let palette = Palette::new_p3(&endpoints, error_metric);
            palette.block_closest_error_p3(block, alpha_map)
        }
    });

    palette_info.create_endpoints(c0, c1)
}

/// BC1 encoding is a discrete optimization problem. However, we treat it as a
/// continuous problem and then quantize the results to get the final R5G6B5
/// endpoints.
///
/// This enum determines how the quantization is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quantization {
    /// Continuous endpoints are rounded to the nearest R5G6B5 color.
    ///
    /// This is the fastest possible quantization, but also the one with the
    /// worst quality.
    Round,
    /// Each channel is optimized independently. Optimization is performed
    /// using all 4 possible combinations of floor/ceil for each endpoint (per
    /// channel).
    ///
    /// This option is rather slow, but provides good quality.
    ChannelWise,
    /// This is a mix between `Round` and `ChannelWise`.
    ///
    /// This is based on 3 ideas:
    ///
    /// 1. The main performance cost comes from calling the error metric, so
    ///    we want to minimize the number of calls to it.
    /// 2. If the min and max (floor and ceil) of one endpoint is the same,
    ///    then the number of calls to the error metric shrinks from 3 to 1.
    /// 3. If the non-quantized endpoint is very close to the quantized min or
    ///    max, then it is likely that the optimal quantized value is the min or
    ///    max. So we can skip the other value.
    ///    E.g. if the non-quantized value of a channel is 0.99, then the
    ///    optimal quantized value for that channel is likely 1 and not 0.
    ///
    /// Using these ideas, a threshold can be defined for when a non-quantized
    /// value is considered "close enough" to the min or max to skip checking
    /// the other. This threshold can also be thought of a radius around the min
    /// and max. As such, a threshold of 0 will have the same behavior as
    /// `ChannelWise`, while a threshold of 0.5 will have the same behavior as
    /// `Round`.
    ///
    /// Using this threshold, we can smoothly trade-off between quality and
    /// performance.
    ///
    /// Right now, the threshold is hardcoded to 0.25.
    ChannelWiseOptimized,
}
impl Quantization {
    fn pick_best(
        self,
        c0: Vec3A,
        c1: Vec3A,
        mut error_metric: impl FnMut(R5G6B5Color, R5G6B5Color) -> f32,
    ) -> (R5G6B5Color, R5G6B5Color) {
        let mut best: (R5G6B5Color, R5G6B5Color) = (
            R5G6B5Color::from_color_round(c0),
            R5G6B5Color::from_color_round(c1),
        );

        if self == Quantization::Round {
            // For simple rounding, we don't need to optimize at all
            return best;
        }

        let mut best_error = error_metric(best.0, best.1);

        let get_range = match self {
            Quantization::ChannelWiseOptimized => Self::optimized_range,
            _ => Self::full_range,
        };

        let (c0_min, c0_max) = get_range(c0);
        let (c1_min, c1_max) = get_range(c1);

        // Channel-wise optimization

        macro_rules! optimize_channel {
            ($c:ident) => {
                for channel0 in c0_min.$c..=c0_max.$c {
                    for channel1 in c1_min.$c..=c1_max.$c {
                        if channel0 == best.0.$c && channel1 == best.1.$c {
                            continue;
                        }
                        let (mut c0, mut c1) = best;
                        c0.$c = channel0;
                        c1.$c = channel1;
                        let error = error_metric(c0, c1);
                        if error < best_error {
                            best = (c0, c1);
                            best_error = error;
                        }
                    }
                }
            };
        }

        optimize_channel!(r);
        optimize_channel!(g);
        optimize_channel!(b);

        best
    }

    /// Returns the floor and ceil of the given color.
    fn full_range(c: Vec3A) -> (R5G6B5Color, R5G6B5Color) {
        (
            R5G6B5Color::from_color_floor(c),
            R5G6B5Color::from_color_ceil(c),
        )
    }

    const CULL_THRESHOLD: f32 = 0.25;
    /// Returns the floor and ceil of the given color. But if the color value
    /// is very close to the floor or ceil, then it will only return one of the
    /// two (returning the same value floor and ceil).
    ///
    /// The threshold for "very close" is defined by `CULL_THRESHOLD`. A
    /// threshold of 0 will behave the same as `full_range`, while a threshold
    /// of 0.5 will behave the same as `Quantization::Round`.
    fn optimized_range(c: Vec3A) -> (R5G6B5Color, R5G6B5Color) {
        let mut floor = R5G6B5Color::from_color_floor(c);
        let mut ceil = R5G6B5Color::from_color_ceil(c);

        let i = c.min(Vec3A::ONE) * Vec3A::new(31.0, 63.0, 31.0);
        let floor_dist = i - Vec3A::new(floor.r as f32, floor.g as f32, floor.b as f32);

        const FLOOR_THRESHOLD: f32 = Quantization::CULL_THRESHOLD;
        const CEIL_THRESHOLD: f32 = 1.0 - Quantization::CULL_THRESHOLD;

        if floor_dist.x < FLOOR_THRESHOLD {
            ceil.r = floor.r;
        }
        if floor_dist.x > CEIL_THRESHOLD {
            floor.r = ceil.r;
        }

        if floor_dist.y < FLOOR_THRESHOLD {
            ceil.g = floor.g;
        }
        if floor_dist.y > CEIL_THRESHOLD {
            floor.g = ceil.g;
        }

        if floor_dist.z < FLOOR_THRESHOLD {
            ceil.b = floor.b;
        }
        if floor_dist.z > CEIL_THRESHOLD {
            floor.b = ceil.b;
        }

        (floor, ceil)
    }
}

fn get_initial_endpoints_from(
    block: &[ColorSpace; 16],
    alpha_map: AlphaMap,
) -> (ColorSpace, ColorSpace) {
    if alpha_map == AlphaMap::ALL_OPAQUE {
        get_initial_endpoints(block)
    } else {
        debug_assert!(alpha_map != AlphaMap::ALL_TRANSPARENT);
        let mut color_buffer = [ColorSpace::default(); 16];
        let colors = get_opaque_colors(block, alpha_map, &mut color_buffer);
        debug_assert!(!colors.is_empty());
        get_initial_endpoints(colors)
    }
}
fn get_initial_endpoints(colors: &[ColorSpace]) -> (ColorSpace, ColorSpace) {
    debug_assert!(colors.len() <= 16);

    // find the best line through the colors
    let line = ColorLine::new(colors);

    // sort all colors along the line and find the min/max projection
    let mut min_t = f32::INFINITY;
    let mut max_t = f32::NEG_INFINITY;
    for &color in colors.iter() {
        let t = line.project(color);
        min_t = min_t.min(t);
        max_t = max_t.max(t);
    }

    // Instead of using min_t and max_t directly, it's better to slightly nudge
    // them towards the midpoint. This prevent extreme endpoints and makes the
    // refinement converge faster.
    let nudge_factor = 0.90;
    let mid_t = (min_t + max_t) * 0.5;
    min_t = mid_t + (min_t - mid_t) * nudge_factor;
    max_t = mid_t + (max_t - mid_t) * nudge_factor;

    // select initial points along the line
    (line.at(min_t), line.at(max_t))
}

fn get_alpha_map(block: &[[f32; 4]; 16]) -> AlphaMap {
    let mut alpha_map = AlphaMap::ALL_TRANSPARENT;
    for (i, pixel) in block.iter().enumerate() {
        alpha_map.set_opaque_if(i, pixel[3] >= ALPHA_THRESHOLD);
    }
    alpha_map
}

fn mean(colors: &[ColorSpace]) -> ColorSpace {
    let mut mean = Vec3A::ZERO;
    for color in colors {
        mean += color.0;
    }
    ColorSpace(mean * (1. / colors.len() as f32))
}
fn covariance_matrix(colors: &[ColorSpace], centroid: Vec3A) -> [Vec3A; 3] {
    let mut cov = [Vec3A::ZERO; 3];

    for p in colors {
        let d = p.0 - centroid;
        cov[0] += d * d.x;
        cov[1] += d * d.y;
        cov[2] += d * d.z;
    }

    let n = colors.len() as f32;
    let n_r = 1.0 / n;
    for i in 0..3 {
        cov[i] *= n_r;
    }

    cov
}
fn largest_eigenvector(matrix: [Vec3A; 3]) -> Vec3A {
    // A simple power iteration method to approximate the dominant eigenvector
    let mut v = Vec3A::ONE;
    for _ in 0..2 {
        let r = matrix[0].dot(v);
        let g = matrix[1].dot(v);
        let b = matrix[2].dot(v);
        v = Vec3A::new(r, g, b).normalize_or_zero();
    }
    v
}

struct ColorLine {
    /// The centroid of the colors
    centroid: ColorSpace,
    /// The normalized direction of the line
    d: Vec3A,
}
impl ColorLine {
    fn new(colors: &[ColorSpace]) -> Self {
        debug_assert!(!colors.is_empty());

        let centroid = mean(colors);
        let covariance = covariance_matrix(colors, centroid.0);
        let eigenvector = largest_eigenvector(covariance);

        Self {
            centroid,
            d: eigenvector,
        }
    }

    fn at(&self, t: f32) -> ColorSpace {
        ColorSpace(self.centroid.0 + self.d * t)
    }

    /// Projects the color onto the line and returns the distance from the
    /// centroid.
    fn project(&self, color: ColorSpace) -> f32 {
        let diff = color.0 - self.centroid.0;
        diff.dot(self.d)
    }
}

struct CandidateList {
    data: [u8; 8],
    error: f32,
}
impl CandidateList {
    fn new() -> Self {
        Self {
            data: [0; 8],
            error: f32::INFINITY,
        }
    }
    fn get_best(self) -> ([u8; 8], f32) {
        debug_assert!(self.error.is_finite());
        (self.data, self.error)
    }

    fn add(
        &mut self,
        block: impl Block4x4<ColorSpace> + Copy,
        alpha_map: AlphaMap,
        e0: R5G6B5Color,
        e1: R5G6B5Color,
        palette_info: PaletteInfo<impl ErrorMetric>,
    ) {
        let endpoints = palette_info.create_endpoints(e0, e1);
        let (indexes, error) = palette_info.block(&endpoints, block, alpha_map);

        if error < self.error {
            self.data = endpoints.with_indexes(indexes);
            self.error = error;
        }
    }
}

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

#[derive(Clone, Copy, PartialEq, Eq)]
struct R5G6B5Color {
    r: u8,
    g: u8,
    b: u8,
}
impl R5G6B5Color {
    const BLACK: Self = Self::new(0, 0, 0);

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

    fn to_u16(self) -> u16 {
        self.debug_check();
        ((self.r as u16) << 11) | ((self.g as u16) << 5) | self.b as u16
    }

    fn debug_check(&self) {
        debug_assert!(self.r < 32);
        debug_assert!(self.g < 64);
        debug_assert!(self.b < 32);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct AlphaMap {
    data: u16,
}
impl AlphaMap {
    const ALL_TRANSPARENT: Self = Self { data: 0 };
    const ALL_OPAQUE: Self = Self { data: u16::MAX };

    fn set_opaque_if(&mut self, index: usize, cond: bool) {
        self.data |= (cond as u16) << index;
    }

    fn is_transparent(&self, index: usize) -> bool {
        (self.data & (1 << index)) == 0
    }
    fn is_opaque(&self, index: usize) -> bool {
        !self.is_transparent(index)
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum PaletteMode {
    P4,
    P3,
}
struct Palette<E> {
    colors: [ColorSpace; 4],
    mode: PaletteMode,
    error_metric: E,
}
impl<E: ErrorMetric> Palette<E> {
    fn new_p3(endpoints: &EndPoints, error_metric: E) -> Self {
        let c0 = endpoints.c0_f;
        let c1 = endpoints.c1_f;

        Self {
            // Fill the last color with c0. This gets us to 4 colors, but since
            // it's the same as c0, it won't affect the closest color search,
            // since its error will be the same as c0 and therefore *not* less
            // than the current smallest error. See `closest()` and `closest_error_sq`.
            colors: [c0, c1, (c0 + c1) * 0.5, c0].map(|c| error_metric.srgb_to_color_space(c)),
            mode: PaletteMode::P3,
            error_metric,
        }
    }

    fn new_p4(endpoints: &EndPoints, error_metric: E) -> Self {
        let c0 = endpoints.c0_f;
        let c1 = endpoints.c1_f;

        Self {
            colors: [
                c0,
                c1,
                c0 * (2. / 3.) + c1 * (1. / 3.),
                c0 * (1. / 3.) + c1 * (2. / 3.),
            ]
            .map(|c| error_metric.srgb_to_color_space(c)),
            mode: PaletteMode::P4,
            error_metric,
        }
    }

    fn transparent_index(&self) -> u8 {
        debug_assert!(
            self.mode == PaletteMode::P3,
            "P4 does not support transparency"
        );

        3
    }

    /// Returns:
    /// 0: The index value of the closest color in the palette
    /// 1: The closest color in the palette
    /// 2: `(pixel - closest) ** 2`, aka the squared error
    fn closest(&self, color: ColorSpace) -> (u8, ColorSpace, f32) {
        let error_metric = self.error_metric;

        let mut best_index = 0;
        let mut min_error = error_metric.error_sq(color, self.colors[0]);
        for i in 1..4 {
            if i == 3 && self.mode == PaletteMode::P3 {
                // In P3 mode, the last color doesn't affect the result.
                // This branch is unnecessary for correctness, but it does slightly
                // improve performance.
                break;
            }
            let error = error_metric.error_sq(color, self.colors[i]);
            if error < min_error {
                best_index = i as u8;
                min_error = error;
            }
        }

        (best_index, self.colors[best_index as usize], min_error)
    }

    /// Returns the square of the error between the pixel and the closest color
    /// in the palette.
    ///
    /// Same as `self.closest(pixel).2`.
    fn closest_error_sq(&self, color: ColorSpace) -> f32 {
        let error_metric = self.error_metric;

        let e0 = error_metric.error_sq(color, self.colors[0]);
        let e1 = error_metric.error_sq(color, self.colors[1]);
        let e2 = error_metric.error_sq(color, self.colors[2]);
        let e3 = error_metric.error_sq(color, self.colors[3]);

        e0.min(e1).min(e2).min(e3)
    }
    /// Same as `closest_error_sq` but optimized for P3 palettes.
    ///
    /// Calling for this on P4 palettes is not allowed.
    fn closest_error_sq_p3(&self, color: ColorSpace) -> f32 {
        debug_assert!(self.mode == PaletteMode::P3);

        let error_metric = self.error_metric;

        let e0 = error_metric.error_sq(color, self.colors[0]);
        let e1 = error_metric.error_sq(color, self.colors[1]);
        let e2 = error_metric.error_sq(color, self.colors[2]);

        e0.min(e1).min(e2)
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
    fn block_closest(
        &self,
        block: impl Block4x4<ColorSpace>,
        alpha_map: AlphaMap,
    ) -> (IndexList, f32) {
        let mut total_error = 0.0;
        let mut index_list = IndexList::new_empty();

        for pixel_index in 0..16 {
            if alpha_map.is_opaque(pixel_index) {
                let pixel = block.get_pixel_at(pixel_index);
                let (index_value, _, error_sq) = self.closest(pixel);
                index_list.set(pixel_index, index_value);
                total_error += error_sq;
            } else {
                index_list.set(pixel_index, self.transparent_index());
            }
        }

        (index_list, total_error)
    }
    /// Same as `block_closest(block).1` but faster and only for P4.
    fn block_closest_error_p4(&self, block: impl Block4x4<ColorSpace>) -> f32 {
        debug_assert!(self.mode == PaletteMode::P4);
        let mut total_error = 0.0;
        for pixel_index in 0..16 {
            let pixel = block.get_pixel_at(pixel_index);
            total_error += self.closest_error_sq(pixel);
        }
        total_error
    }
    /// Same as `block_closest(block).1` but faster and only for P3.
    fn block_closest_error_p3(&self, block: impl Block4x4<ColorSpace>, alpha_map: AlphaMap) -> f32 {
        debug_assert!(self.mode == PaletteMode::P3);
        let mut total_error = 0.0;
        if alpha_map == AlphaMap::ALL_OPAQUE {
            for pixel_index in 0..16 {
                let pixel = block.get_pixel_at(pixel_index);
                total_error += self.closest_error_sq_p3(pixel);
            }
        } else {
            for pixel_index in 0..16 {
                if alpha_map.is_opaque(pixel_index) {
                    let pixel = block.get_pixel_at(pixel_index);
                    total_error += self.closest_error_sq_p3(pixel);
                }
            }
        }
        total_error
    }

    fn block_dither(
        &self,
        block: impl Block4x4<ColorSpace> + Copy,
        alpha_map: AlphaMap,
    ) -> (IndexList, f32) {
        let mut index_list = IndexList::new_empty();
        let mut total_error = 0.0;

        // This implements a modified version of the Floyd-Steinberg dithering
        bcn_util::block_dither(block, |pixel_index, pixel| {
            if alpha_map.is_opaque(pixel_index) {
                let (index_value, closest, error_sq) = self.closest(pixel);
                index_list.set(pixel_index, index_value);
                total_error += error_sq;
                closest
            } else {
                index_list.set(pixel_index, self.transparent_index());
                block.get_pixel_at(pixel_index)
            }
        });

        (index_list, total_error)
    }
}

#[derive(Clone, Copy)]
struct PaletteInfo<E> {
    mode: PaletteMode,
    dither: bool,
    error_metric: E,
}
impl<E: ErrorMetric> PaletteInfo<E> {
    fn new_p3(error_metric: E, options: Bc1Options) -> Self {
        Self {
            mode: PaletteMode::P3,
            dither: options.dither,
            error_metric,
        }
    }
    fn new_p4(error_metric: E, options: Bc1Options) -> Self {
        Self {
            mode: PaletteMode::P4,
            dither: options.dither,
            error_metric,
        }
    }

    fn create_endpoints(&self, e0: R5G6B5Color, e1: R5G6B5Color) -> EndPoints {
        match self.mode {
            PaletteMode::P4 => EndPoints::new_p4(e0, e1),
            PaletteMode::P3 => EndPoints::new_p3_default(e0, e1),
        }
    }

    fn create_palette(&self, endpoints: &EndPoints) -> Palette<E> {
        match self.mode {
            PaletteMode::P4 => Palette::new_p4(endpoints, self.error_metric),
            PaletteMode::P3 => Palette::new_p3(endpoints, self.error_metric),
        }
    }

    fn block(
        &self,
        endpoints: &EndPoints,
        block: impl Block4x4<ColorSpace> + Copy,
        alpha_map: AlphaMap,
    ) -> (IndexList, f32) {
        let p = self.create_palette(endpoints);
        if self.dither {
            p.block_dither(block, alpha_map)
        } else {
            p.block_closest(block, alpha_map)
        }
    }
}

trait ErrorMetric: Copy {
    fn srgb_to_color_space(&self, color: Vec3A) -> ColorSpace;
    fn color_space_to_srgb(&self, color: ColorSpace) -> Vec3A;

    /// Returns the square of the error between the two colors.
    fn error_sq(&self, a: ColorSpace, b: ColorSpace) -> f32 {
        a.0.distance_squared(b.0)
    }
}
#[derive(Clone, Copy)]
struct Uniform;
impl ErrorMetric for Uniform {
    #[inline]
    fn srgb_to_color_space(&self, color: Vec3A) -> ColorSpace {
        ColorSpace(color)
    }
    #[inline]
    fn color_space_to_srgb(&self, color: ColorSpace) -> Vec3A {
        color.0
    }
}
#[derive(Clone, Copy)]
struct Perceptual;
impl ErrorMetric for Perceptual {
    #[inline]
    fn srgb_to_color_space(&self, color: Vec3A) -> ColorSpace {
        ColorSpace(fast_srgb_to_oklab(color))
    }
    #[inline]
    fn color_space_to_srgb(&self, color: ColorSpace) -> Vec3A {
        fast_oklab_to_srgb(color.0)
    }
}
