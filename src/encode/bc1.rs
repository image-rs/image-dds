use crate::{n5, n6};

use super::bcn_util;

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

    // separate color and alpha
    let colors: [Color3; 16] = block.map(|p| p.into());
    let alpha_map = get_alpha_map(&block, options.alpha_threshold);

    // Don't use the default color mode in BC2 and BC3
    if options.no_default {
        return compress_p4(colors, options).0;
    }

    if alpha_map == AlphaMap::ALL_TRANSPARENT {
        return TRANSPARENT_BLOCK;
    }

    if alpha_map != AlphaMap::ALL_OPAQUE {
        return compress_p3_default(colors, alpha_map, options).0;
    }

    // We have a choice to make. So just try both and pick whichever is better
    let (p4, p4_error) = compress_p4(colors, options);
    let (p3, p3_error) = compress_p3_default(colors, alpha_map, options);
    if p4_error < p3_error {
        p4
    } else {
        p3
    }
}
fn compress_p4(block: [Color3; 16], options: Bc1Options) -> ([u8; 8], f32) {
    let (min, max) = get_initial_endpoints(&block);

    let endpoints = EndPoints::new_p4(min.into(), max.into());
    let palette = P4Palette::new(endpoints.c0_f, endpoints.c1_f, Uniform);

    let (indexes, error) = if options.dither {
        palette.block_dither(&block)
    } else {
        palette.block_closest(&block)
    };

    (endpoints.with_indexes(indexes), error)
}
fn compress_p3_default(
    block: [Color3; 16],
    alpha_map: AlphaMap,
    options: Bc1Options,
) -> ([u8; 8], f32) {
    let (mut color_buffer, color_count) = get_opaque_colors(&block, alpha_map);
    if color_count == 0 {
        return (TRANSPARENT_BLOCK, 0.0);
    }
    let colors = &mut color_buffer[..color_count];

    let (min, max) = get_initial_endpoints(colors);

    let endpoints = EndPoints::new_p3_default(min.into(), max.into());
    let palette = P3Palette::new(endpoints.c0_f, endpoints.c1_f, Uniform);

    let (indexes, error) = if options.dither {
        palette.block_dither(&block, alpha_map)
    } else {
        palette.block_closest(&block, alpha_map)
    };

    (endpoints.with_indexes(indexes), error)
}

fn get_opaque_colors(block: &[Color3; 16], alpha_map: AlphaMap) -> ([Color3; 16], usize) {
    let mut opaque_colors = [Color3::ZERO; 16];
    let mut count = 0;
    for (i, &pixel) in block.iter().enumerate() {
        if alpha_map.is_opaque(i) {
            opaque_colors[count] = pixel;
            count += 1;
        }
    }
    (opaque_colors, count)
}

fn get_initial_endpoints(colors: &[Color3]) -> (Color3, Color3) {
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

#[derive(Debug, Clone)]
struct EndPoints {
    c0: R5G6B5Color,
    c1: R5G6B5Color,
    c0_f: Color3,
    c1_f: Color3,
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

#[derive(Debug, Clone, Copy)]
struct R5G6B5Color {
    r: u8,
    g: u8,
    b: u8,
}
impl R5G6B5Color {
    fn from_color(color: Color3) -> Self {
        let r = n5::from_f32(color.r);
        let g = n6::from_f32(color.g);
        let b = n5::from_f32(color.b);
        Self { r, g, b }
    }
    fn to_color(self) -> Color3 {
        self.debug_check();
        Color3::new(n5::f32(self.r), n6::f32(self.g), n5::f32(self.b))
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
impl From<Color3> for R5G6B5Color {
    fn from(color: Color3) -> Self {
        Self::from_color(color)
    }
}

#[derive(Debug, Clone, Copy)]
struct Color3 {
    r: f32,
    g: f32,
    b: f32,
}
impl Color3 {
    const ZERO: Self = Self::new(0., 0., 0.);

    const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    fn clamp(&self) -> Self {
        Self::new(
            self.r.clamp(0., 1.),
            self.g.clamp(0., 1.),
            self.b.clamp(0., 1.),
        )
    }

    fn min(&self, other: Self) -> Self {
        Self::new(
            self.r.min(other.r),
            self.g.min(other.g),
            self.b.min(other.b),
        )
    }
    fn max(&self, other: Self) -> Self {
        Self::new(
            self.r.max(other.r),
            self.g.max(other.g),
            self.b.max(other.b),
        )
    }
}
impl From<[f32; 4]> for Color3 {
    fn from([r, g, b, _]: [f32; 4]) -> Self {
        Self::new(r, g, b)
    }
}
impl Default for Color3 {
    fn default() -> Self {
        Self::ZERO
    }
}
impl std::ops::Add<Color3> for Color3 {
    type Output = Color3;

    fn add(self, rhs: Color3) -> Self::Output {
        Color3::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}
impl std::ops::AddAssign<Color3> for Color3 {
    fn add_assign(&mut self, rhs: Color3) {
        *self = *self + rhs;
    }
}
impl std::ops::Sub<Color3> for Color3 {
    type Output = Color3;

    fn sub(self, rhs: Color3) -> Self::Output {
        Color3::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}
impl std::ops::Mul<f32> for Color3 {
    type Output = Color3;

    fn mul(self, rhs: f32) -> Self::Output {
        Color3::new(self.r * rhs, self.g * rhs, self.b * rhs)
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
    colors: [Color3; 4],
    error_metric: E,
}
impl<E> P4Palette<E> {
    fn new(c0: Color3, c1: Color3, error_metric: E) -> Self {
        let colors = [
            c0,
            c1,
            c0 * (2. / 3.) + c1 * (1. / 3.),
            c0 * (1. / 3.) + c1 * (2. / 3.),
        ];

        Self {
            colors,
            error_metric,
        }
    }
}
impl<E: ErrorMetric> Palette for P4Palette<E> {
    fn closest(&self, color: Color3) -> (u8, Color3, f32) {
        let mut best_index = 0;
        let mut min_error = self.error_metric.error_sq(color, self.colors[0]);
        for i in 1..4 {
            let error = self.error_metric.error_sq(color, self.colors[i]);
            if error < min_error {
                best_index = i as u8;
                min_error = error;
            }
        }

        (best_index, self.colors[best_index as usize], min_error)
    }

    fn closest_error_sq(&self, pixel: Color3) -> f32 {
        let error = self.colors.map(|c| self.error_metric.error_sq(pixel, c));
        let mut min = error[0];
        for e in error {
            min = min.min(e);
        }
        min
    }
}

struct P3Palette<E> {
    colors: [Color3; 3],
    error_metric: E,
}
impl<E: ErrorMetric> P3Palette<E> {
    const DEFAULT: u8 = 3;

    fn new(c0: Color3, c1: Color3, error_metric: E) -> Self {
        let colors = [c0, c1, (c0 + c1) * 0.5];
        Self {
            colors,
            error_metric,
        }
    }

    fn closest(&self, color: Color3) -> (u8, Color3, f32) {
        let mut best_index = 0;
        let mut min_error = self.error_metric.error_sq(color, self.colors[0]);
        for i in 1..3 {
            let error = self.error_metric.error_sq(color, self.colors[i]);
            if error < min_error {
                best_index = i as u8;
                min_error = error;
            }
        }

        (best_index, self.colors[best_index as usize], min_error)
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
    fn block_closest(&self, block: &[Color3; 16], alpha_map: AlphaMap) -> (IndexList, f32) {
        let mut total_error = 0.0;
        let mut index_list = IndexList::new_empty();
        for (pixel_index, pixel) in block.iter().copied().enumerate() {
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

    fn block_dither(&self, block: &[Color3; 16], alpha_map: AlphaMap) -> (IndexList, f32) {
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
                block[pixel_index]
            }
        });

        (index_list, total_error)
    }
}

trait ErrorMetric {
    /// Returns the square of the error between the two colors.
    fn error_sq(&self, a: Color3, b: Color3) -> f32;
}
struct Uniform;
impl ErrorMetric for Uniform {
    fn error_sq(&self, a: Color3, b: Color3) -> f32 {
        let diff = a - b;
        (diff.r * diff.r + diff.g * diff.g + diff.b * diff.b) * (1. / 3.)
    }
}
struct Perceptual;
impl ErrorMetric for Perceptual {
    fn error_sq(&self, a: Color3, b: Color3) -> f32 {
        let diff = a - b;
        diff.r * diff.r * 0.299 + diff.g * diff.g * 0.587 + diff.b * diff.b * 0.114
    }
}

trait Palette {
    /// Returns:
    /// 0: The index value of the closest color in the palette
    /// 1: The closest color in the palette
    /// 2: `(pixel - closest) ** 2`, aka the squared error
    fn closest(&self, color: Color3) -> (u8, Color3, f32);

    /// Returns the square of the error between the pixel and the closest color
    /// in the palette.
    ///
    /// Same as `self.closest(pixel).2`.
    fn closest_error_sq(&self, pixel: Color3) -> f32 {
        self.closest(pixel).2
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
    fn block_closest(&self, block: &[Color3; 16]) -> (IndexList, f32) {
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
    fn block_closest_mse(&self, block: &[Color3; 16]) -> f32 {
        block
            .iter()
            .copied()
            .map(|pixel| self.closest_error_sq(pixel))
            .sum()
    }

    fn block_dither(&self, block: &[Color3; 16]) -> (IndexList, f32) {
        let mut index_list = IndexList::new_empty();
        let mut total_error = 0.0;

        // This implements a modified version of the Floyd-Steinberg dithering
        let mut error_map = [Color3::ZERO; 16];
        for y in 0..4 {
            for x in 0..4 {
                let pixel_index = y * 4 + x;
                let pixel = (block[pixel_index] + error_map[pixel_index]).clamp();
                let (index_value, closest, sq_error) = self.closest(pixel);
                index_list.set(pixel_index, index_value);
                total_error += sq_error;
                let error = pixel - closest;

                // diffuse the error
                let mut weight_right = 7. / 16.;
                let mut weight_next_left = 3. / 16.;
                let mut weight_next_middle = 5. / 16.;
                let mut weight_next_right = 1. / 16.;
                // adjust the weights, so we lose as little of the error as possible
                if x == 0 {
                    weight_next_left = 0.0;
                    weight_next_middle = 6. / 16.;
                    weight_next_right = 2. / 16.;
                }
                if x == 3 {
                    // we lose 25% of the error, per pixel in the last column
                    weight_right = 0.0;
                    weight_next_left = 5. / 16.;
                    weight_next_middle = 7. / 16.;
                    weight_next_right = 0.0;
                }
                if y == 3 {
                    // we lose 50% of the error, per pixel in the last row
                    weight_right = 8. / 16.;
                    weight_next_left = 0.0;
                    weight_next_middle = 0.0;
                    weight_next_right = 0.0;
                }

                if x < 3 {
                    error_map[pixel_index + 1] += error * weight_right;
                }
                if y < 3 {
                    if x > 0 {
                        error_map[pixel_index + 4 - 1] += error * weight_next_left;
                    }
                    error_map[pixel_index + 4] += error * weight_next_middle;
                    if x < 3 {
                        error_map[pixel_index + 4 + 1] += error * weight_next_right;
                    }
                }
            }
        }

        (index_list, total_error)
    }
}
