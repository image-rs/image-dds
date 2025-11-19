use super::read_write::{
    for_each_block_rect_untyped, for_each_block_untyped, process_4x4_blocks_helper, PixelRange,
};
use super::{Args, Decoder, DecoderSet, RArgs};
use crate::{NormConvert, WithPrecision};

use crate::util::closure_types;
use crate::{Channels::*, ColorFormat};

// helpers

macro_rules! underlying {
    ($channels:expr, $out:ty, $bytes_per_block:literal, $f:expr) => {{
        const BYTES_PER_BLOCK: usize = $bytes_per_block;
        const CHANNELS: usize = $channels.count() as usize;
        type OutPixel = [$out; CHANNELS];

        fn process_blocks(
            encoded_blocks: &[u8],
            decoded: &mut [u8],
            stride: usize,
            range: PixelRange,
        ) {
            let f = closure_types::<[u8; BYTES_PER_BLOCK], [OutPixel; 16], _>($f);
            process_4x4_blocks_helper(encoded_blocks, decoded, stride, range, f)
        }

        const NATIVE_COLOR: ColorFormat =
            ColorFormat::new($channels, <$out as WithPrecision>::PRECISION);

        Decoder::new_with_all_channels(
            NATIVE_COLOR,
            |Args(r, out, context)| {
                for_each_block_untyped::<4, 4, BYTES_PER_BLOCK, OutPixel>(
                    r,
                    out,
                    context,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
            |RArgs(r, out, offset, context)| {
                for_each_block_rect_untyped::<4, 4, BYTES_PER_BLOCK>(
                    r,
                    out,
                    offset,
                    context,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
        )
    }};
}

macro_rules! gray {
    ($out:ty, $bytes_per_block:literal, $f:expr) => {
        underlying!(Grayscale, $out, $bytes_per_block, $f)
    };
}
macro_rules! rgb {
    ($out:ty, $bytes_per_block:literal, $f:expr) => {
        underlying!(Rgb, $out, $bytes_per_block, $f)
    };
}
macro_rules! rgba {
    ($out:ty, $bytes_per_block:literal, $f:expr) => {
        underlying!(Rgba, $out, $bytes_per_block, $f)
    };
}

fn with_precision<const N: usize, const C: usize, I, O>(
    f: impl Copy + Fn([u8; N]) -> [[I; C]; 16],
) -> impl Copy + Fn([u8; N]) -> [[O; C]; 16]
where
    I: NormConvert<O>,
{
    move |block_bytes| f(block_bytes).map(|p| p.map(NormConvert::to))
}

// decoders

pub(crate) const BC1_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, 8, blocks::bc1_u8_rgba),
    rgba!(u16, 8, with_precision(blocks::bc1_u8_rgba)),
    rgba!(f32, 8, with_precision(blocks::bc1_u8_rgba)),
]);

pub(crate) const BC2_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, 16, blocks::bc2_u8_rgba),
    rgba!(u16, 16, with_precision(blocks::bc2_u8_rgba)),
    rgba!(f32, 16, with_precision(blocks::bc2_u8_rgba)),
    rgb!(u8, 16, blocks::bc2_u8_rgb),
    rgb!(u16, 16, with_precision(blocks::bc2_u8_rgb)),
    rgb!(f32, 16, with_precision(blocks::bc2_u8_rgb)),
]);

pub(crate) const BC2_UNORM_PREMULTIPLIED_ALPHA: DecoderSet = DecoderSet::new(&[
    rgba!(u8, 16, blocks::bc2_premultiplied_alpha_u8_rgba),
    rgba!(
        u16,
        16,
        with_precision(blocks::bc2_premultiplied_alpha_u8_rgba)
    ),
    rgba!(
        f32,
        16,
        with_precision(blocks::bc2_premultiplied_alpha_u8_rgba)
    ),
]);

pub(crate) const BC3_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, 16, blocks::bc3_u8_rgba),
    rgba!(u16, 16, with_precision(blocks::bc3_u8_rgba)),
    rgba!(f32, 16, with_precision(blocks::bc3_u8_rgba)),
    rgb!(u8, 16, blocks::bc3_u8_rgb),
    rgb!(u16, 16, with_precision(blocks::bc3_u8_rgb)),
    rgb!(f32, 16, with_precision(blocks::bc3_u8_rgb)),
]);

pub(crate) const BC3_UNORM_PREMULTIPLIED_ALPHA: DecoderSet = DecoderSet::new(&[
    rgba!(u8, 16, blocks::bc3_premultiplied_alpha_u8_rgba),
    rgba!(
        u16,
        16,
        with_precision(blocks::bc3_premultiplied_alpha_u8_rgba)
    ),
    rgba!(
        f32,
        16,
        with_precision(blocks::bc3_premultiplied_alpha_u8_rgba)
    ),
]);

pub(crate) const BC3_UNORM_RXGB: DecoderSet = DecoderSet::new(&[
    rgb!(u8, 16, blocks::bc3_rxgb_u8_rgb),
    rgb!(u16, 16, with_precision(blocks::bc3_rxgb_u8_rgb)),
    rgb!(f32, 16, with_precision(blocks::bc3_rxgb_u8_rgb)),
]);

pub(crate) const BC3_UNORM_NORMAL: DecoderSet = DecoderSet::new(&[
    rgb!(u8, 16, blocks::bc3n_u8_rgb),
    rgb!(u16, 16, with_precision(blocks::bc3n_u8_rgb)),
    rgb!(f32, 16, with_precision(blocks::bc3n_u8_rgb)),
]);

pub(crate) const BC4_UNORM: DecoderSet = DecoderSet::new(&[
    gray!(u8, 8, blocks::bc4u_gray),
    gray!(u16, 8, blocks::bc4u_gray),
    gray!(f32, 8, blocks::bc4u_gray),
]);

pub(crate) const BC4_SNORM: DecoderSet = DecoderSet::new(&[
    gray!(u8, 8, blocks::bc4s_gray),
    gray!(u16, 8, blocks::bc4s_gray),
    gray!(f32, 8, blocks::bc4s_gray),
]);

pub(crate) const BC5_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, 16, blocks::bc5u_rgb),
    rgb!(u16, 16, blocks::bc5u_rgb),
    rgb!(f32, 16, blocks::bc5u_rgb),
]);

pub(crate) const BC5_SNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, 16, blocks::bc5s_rgb),
    rgb!(u16, 16, blocks::bc5s_rgb),
    rgb!(f32, 16, blocks::bc5s_rgb),
]);

pub(crate) const BC6H_UF16: DecoderSet = DecoderSet::new(&[
    rgb!(f32, 16, blocks::bc6_u_f32),
    rgb!(u16, 16, blocks::bc6_u_u16),
    rgb!(u8, 16, blocks::bc6_u_u8),
]);
pub(crate) const BC6H_SF16: DecoderSet = DecoderSet::new(&[
    rgb!(f32, 16, blocks::bc6_s_f32),
    rgb!(u16, 16, blocks::bc6_s_u16),
    rgb!(u8, 16, blocks::bc6_s_u8),
]);

pub(crate) const BC7_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, 16, blocks::bc7_u8_rgba),
    rgba!(u16, 16, blocks::bc7_u16_rgba),
    rgba!(f32, 16, blocks::bc7_f32_rgba),
]);

/// Internal module for the underlying logic of decoding BC1-7 blocks.
mod blocks {
    // use crate::decode::convert::{bc6h_uf16, fp16, n4, n8, s8, Norm, ToRgba, B5G6R5};
    use crate::{bc6h_uf16, fp16, n4, n8, s8, Norm, ToRgba, B5G6R5};

    /// Decodes a BC1 block into 16 RGBA pixels.
    pub(crate) fn bc1_u8_rgba(block_bytes: [u8; 8]) -> [[u8; 4]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc1
        let color0_u16 = u16::from_le_bytes([block_bytes[0], block_bytes[1]]);
        let color1_u16 = u16::from_le_bytes([block_bytes[2], block_bytes[3]]);

        let c0_bgr = B5G6R5::from_u16(color0_u16);
        let c1_bgr = B5G6R5::from_u16(color1_u16);

        let c0 = c0_bgr.to_n8().to_rgba();
        let c1 = c1_bgr.to_n8().to_rgba();

        let mut pixels: [[u8; 4]; 16] = Default::default();

        let (c2, c3) = if color0_u16 > color1_u16 {
            (
                c0_bgr.one_third_color_rgb8(c1_bgr).to_rgba(),
                c1_bgr.one_third_color_rgb8(c0_bgr).to_rgba(),
            )
        } else {
            (
                c0_bgr.mid_color_rgb8(c1_bgr).to_rgba(),
                [0, 0, 0, 0], // transparent
            )
        };

        let lut = [c0, c1, c2, c3];
        let indexes = u32::from_le_bytes([
            block_bytes[4],
            block_bytes[5],
            block_bytes[6],
            block_bytes[7],
        ]);
        for (i, pixel) in pixels.iter_mut().enumerate() {
            let index = (indexes >> (i * 2)) & 0b11;
            *pixel = lut[index as usize];
        }

        pixels
    }

    fn bc1_no_default_u8_rgba(block_bytes: [u8; 8]) -> [[u8; 4]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc1
        let color0_u16 = u16::from_le_bytes([block_bytes[0], block_bytes[1]]);
        let color1_u16 = u16::from_le_bytes([block_bytes[2], block_bytes[3]]);

        let c0_bgr = B5G6R5::from_u16(color0_u16);
        let c1_bgr = B5G6R5::from_u16(color1_u16);

        let c0 = c0_bgr.to_n8().to_rgba();
        let c1 = c1_bgr.to_n8().to_rgba();
        let c2 = c0_bgr.one_third_color_rgb8(c1_bgr).to_rgba();
        let c3 = c1_bgr.one_third_color_rgb8(c0_bgr).to_rgba();

        let mut pixels: [[u8; 4]; 16] = Default::default();

        let lut = [c0, c1, c2, c3];
        let indexes = u32::from_le_bytes([
            block_bytes[4],
            block_bytes[5],
            block_bytes[6],
            block_bytes[7],
        ]);
        for (i, pixel) in pixels.iter_mut().enumerate() {
            let index = (indexes >> (i * 2)) & 0b11;
            *pixel = lut[index as usize];
        }

        pixels
    }

    fn split_16(x: [u8; 16]) -> ([u8; 8], [u8; 8]) {
        let lower = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]];
        let upper = [x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]];
        (lower, upper)
    }

    /// Decodes a BC2 block into 16 RGBA pixels.
    pub(crate) fn bc2_u8_rgba(block_bytes: [u8; 16]) -> [[u8; 4]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc2
        let (alpha_bytes, bc1_bytes) = split_16(block_bytes);
        let mut pixels = bc1_no_default_u8_rgba(bc1_bytes);

        for i in 0..4 {
            let alpha_byte_high = alpha_bytes[i * 2];
            let alpha_byte_low = alpha_bytes[i * 2 + 1];
            let alpha = [
                alpha_byte_high & 0xF,
                alpha_byte_high >> 4,
                alpha_byte_low & 0xF,
                alpha_byte_low >> 4,
            ]
            .map(n4::n8);

            for (j, &alpha) in alpha.iter().enumerate() {
                pixels[i * 4 + j][3] = alpha;
            }
        }

        pixels
    }
    pub(crate) fn bc2_u8_rgb(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc2
        let (_, bc1_bytes) = split_16(block_bytes);
        let pixels = bc1_no_default_u8_rgba(bc1_bytes);
        pixels.map(|[r, g, b, _]| [r, g, b])
    }
    pub(crate) fn bc2_premultiplied_alpha_u8_rgba(block_bytes: [u8; 16]) -> [[u8; 4]; 16] {
        let mut pixels = bc2_u8_rgba(block_bytes);
        to_straight_alpha(&mut pixels);
        pixels
    }

    fn to_straight_alpha(pixels: &mut [[u8; 4]; 16]) {
        for pixel in pixels.iter_mut() {
            let mut alpha = pixel[3];
            if alpha == 0 {
                alpha = 255;
            }
            for channel in &mut pixel[..3] {
                *channel = (*channel as u16 * 255 / alpha as u16).min(255) as u8;
            }
        }
    }

    /// Decodes a BC3 block into 16 RGBA pixels.
    pub(crate) fn bc3_u8_rgba(block_bytes: [u8; 16]) -> [[u8; 4]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc3
        let (alpha_bytes, bc1_bytes) = split_16(block_bytes);

        let mut pixels = bc1_u8_rgba(bc1_bytes);
        let alpha = bc4u_gray(alpha_bytes);

        for i in 0..4 {
            for j in 0..4 {
                pixels[i * 4 + j][3] = alpha[i * 4 + j][0];
            }
        }

        pixels
    }
    pub(crate) fn bc3_u8_rgb(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc3
        let (_, bc1_bytes) = split_16(block_bytes);
        let pixels = bc1_u8_rgba(bc1_bytes);
        pixels.map(|[r, g, b, _]| [r, g, b])
    }
    pub(crate) fn bc3_rxgb_u8_rgb(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        bc3_u8_rgba(block_bytes).map(|[_, g, b, r]| [r, g, b])
    }
    pub(crate) fn bc3n_u8_rgb(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        fn calc_b(r: u8, g: u8) -> u8 {
            let x = r as f32 * (2.0 / 255.0) - 1.0;
            let y = g as f32 * (2.0 / 255.0) - 1.0;
            let z = (1.0 - x * x - y * y).max(0.0).sqrt();
            (z * (0.5 * 255.0) + (0.5 * 255.0 + 0.5)) as u8
        }
        bc3_u8_rgba(block_bytes).map(|[_, g, _, r]| [r, g, calc_b(r, g)])
    }
    pub(crate) fn bc3_premultiplied_alpha_u8_rgba(block_bytes: [u8; 16]) -> [[u8; 4]; 16] {
        let mut pixels = bc3_u8_rgba(block_bytes);
        to_straight_alpha(&mut pixels);
        pixels
    }

    pub(crate) trait BC4uOperations: Norm {
        /// Given a UNORM 8 endpoint, convert to Self.
        fn from_byte(byte: u8) -> Self;
        /// Given a UNORM in the range `0..=255*7`, convert to Self.
        fn from_interpolation_6(interpolation: u16) -> Self;
        /// Given a UNORM in the range `0..=255*5`, convert to Self.
        fn from_interpolation_4(interpolation: u16) -> Self;
    }
    impl BC4uOperations for u8 {
        fn from_byte(byte: u8) -> Self {
            byte
        }
        fn from_interpolation_6(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1785);
            ((interpolation as u32 * 9360 + 32160) >> 16) as u8
        }
        fn from_interpolation_4(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1275);
            ((interpolation as u32 * 13104 + 30288) >> 16) as u8
        }
    }
    impl BC4uOperations for u16 {
        fn from_byte(byte: u8) -> Self {
            n8::n16(byte)
        }
        fn from_interpolation_6(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1785);
            ((interpolation as u32 * 2406112 + 28064) >> 16) as u16
        }
        fn from_interpolation_4(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1275);
            ((interpolation as u32 * 3368544 + 34368) >> 16) as u16
        }
    }
    impl BC4uOperations for f32 {
        fn from_byte(byte: u8) -> Self {
            n8::f32(byte)
        }
        fn from_interpolation_6(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1785);
            const F: f32 = 1.0 / 1785.0;
            interpolation as f32 * F
        }
        fn from_interpolation_4(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1275);
            const F: f32 = 1.0 / 1275.0;
            interpolation as f32 * F
        }
    }
    pub(crate) fn bc4u_gray<T: BC4uOperations>(block_bytes: [u8; 8]) -> [[T; 1]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc4
        let c0_u8 = block_bytes[0];
        let c1_u8 = block_bytes[1];
        let c0_u16 = c0_u8 as u16;
        let c1_u16 = c1_u8 as u16;

        let c0 = T::from_byte(c0_u8);
        let c1 = T::from_byte(c1_u8);

        let (c2, c3, c4, c5, c6, c7) = if c0_u8 > c1_u8 {
            // 6 interpolated colors
            (
                T::from_interpolation_6(c0_u16 * 6 + c1_u16),
                T::from_interpolation_6(c0_u16 * 5 + c1_u16 * 2),
                T::from_interpolation_6(c0_u16 * 4 + c1_u16 * 3),
                T::from_interpolation_6(c0_u16 * 3 + c1_u16 * 4),
                T::from_interpolation_6(c0_u16 * 2 + c1_u16 * 5),
                T::from_interpolation_6(c0_u16 + c1_u16 * 6),
            )
        } else {
            // 4 interpolated colors
            (
                T::from_interpolation_4(c0_u16 * 4 + c1_u16),
                T::from_interpolation_4(c0_u16 * 3 + c1_u16 * 2),
                T::from_interpolation_4(c0_u16 * 2 + c1_u16 * 3),
                T::from_interpolation_4(c0_u16 + c1_u16 * 4),
                T::ZERO,
                T::ONE,
            )
        };

        let mut pixels: [[T; 1]; 16] = Default::default();

        let lut = [c0, c1, c2, c3, c4, c5, c6, c7];
        let indexes0 = u32::from_le_bytes([block_bytes[2], block_bytes[3], block_bytes[4], 0]);
        let indexes1 = u32::from_le_bytes([block_bytes[5], block_bytes[6], block_bytes[7], 0]);
        for (i, indexes) in [indexes0, indexes1].into_iter().enumerate() {
            for j in 0..8 {
                let index = (indexes >> (j * 3)) & 0b111;
                pixels[i * 8 + j][0] = lut[index as usize];
            }
        }

        pixels
    }

    pub(crate) trait BC4sOperations: Norm {
        /// Given a UNORM 8 endpoint, convert to Self.
        fn from_byte(byte: u8) -> Self;
        /// Given a UNORM in the range `0..=254*7`, convert to Self.
        fn from_interpolation_6(interpolation: u16) -> Self;
        /// Given a UNORM in the range `0..=254*5`, convert to Self.
        fn from_interpolation_4(interpolation: u16) -> Self;
    }
    impl BC4sOperations for u8 {
        fn from_byte(byte: u8) -> Self {
            s8::n8(byte)
        }
        fn from_interpolation_6(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1778);
            ((interpolation as u32 * 255 + (7 * 254) / 2) / (7 * 254)) as u8
        }
        fn from_interpolation_4(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1270);
            ((interpolation as u32 * 255 + (5 * 254) / 2) / (5 * 254)) as u8
        }
    }
    impl BC4sOperations for u16 {
        fn from_byte(byte: u8) -> Self {
            s8::n16(byte)
        }
        fn from_interpolation_6(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1778);
            ((interpolation as u32 * 65535 + (7 * 254) / 2) / (7 * 254)) as u16
        }
        fn from_interpolation_4(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1270);
            ((interpolation as u32 * 65535 + (5 * 254) / 2) / (5 * 254)) as u16
        }
    }
    impl BC4sOperations for f32 {
        fn from_byte(byte: u8) -> Self {
            s8::uf32(byte)
        }
        fn from_interpolation_6(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1778);
            const C: f32 = 1.0 / 1778.0;
            interpolation as f32 * C
        }
        fn from_interpolation_4(interpolation: u16) -> Self {
            debug_assert!(interpolation <= 1270);
            const C: f32 = 1.0 / 1270.0;
            interpolation as f32 * C
        }
    }
    pub(crate) fn bc4s_gray<T: BC4sOperations>(block_bytes: [u8; 8]) -> [[T; 1]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc4
        let red0 = block_bytes[0];
        let red1 = block_bytes[1];

        let r0_254 = s8::norm(red0) as u16;
        let r1_254 = s8::norm(red1) as u16;

        let c0 = T::from_byte(red0);
        let c1 = T::from_byte(red1);

        let (c2, c3, c4, c5, c6, c7) = if red0 as i8 > red1 as i8 {
            // 6 interpolated colors
            (
                T::from_interpolation_6(r0_254 * 6 + r1_254),
                T::from_interpolation_6(r0_254 * 5 + r1_254 * 2),
                T::from_interpolation_6(r0_254 * 4 + r1_254 * 3),
                T::from_interpolation_6(r0_254 * 3 + r1_254 * 4),
                T::from_interpolation_6(r0_254 * 2 + r1_254 * 5),
                T::from_interpolation_6(r0_254 + r1_254 * 6),
            )
        } else {
            // 4 interpolated colors
            (
                T::from_interpolation_4(r0_254 * 4 + r1_254),
                T::from_interpolation_4(r0_254 * 3 + r1_254 * 2),
                T::from_interpolation_4(r0_254 * 2 + r1_254 * 3),
                T::from_interpolation_4(r0_254 + r1_254 * 4),
                T::ZERO,
                T::ONE,
            )
        };

        let mut pixels: [[T; 1]; 16] = Default::default();

        let lut = [c0, c1, c2, c3, c4, c5, c6, c7];
        let indexes0 = u32::from_le_bytes([block_bytes[2], block_bytes[3], block_bytes[4], 0]);
        let indexes1 = u32::from_le_bytes([block_bytes[5], block_bytes[6], block_bytes[7], 0]);
        for (i, indexes) in [indexes0, indexes1].into_iter().enumerate() {
            for j in 0..8 {
                let index = (indexes >> (j * 3)) & 0b111;
                pixels[i * 8 + j][0] = lut[index as usize];
            }
        }

        pixels
    }

    /// Decodes a BC5 UNORM block into 16 RGB pixels.
    pub(crate) fn bc5u_rgb<T: BC4uOperations>(block_bytes: [u8; 16]) -> [[T; 3]; 16] {
        let (red_bytes, green_bytes) = split_16(block_bytes);
        let red = bc4u_gray(red_bytes);
        let green = bc4u_gray(green_bytes);

        let mut pixels: [[T; 3]; 16] = Default::default();
        for (i, pixel) in pixels.iter_mut().enumerate() {
            pixel[0] = red[i][0];
            pixel[1] = green[i][0];
            pixel[2] = T::ZERO;
        }

        pixels
    }

    /// Decodes a BC5 SNORM block into 16 RGB pixels.
    pub(crate) fn bc5s_rgb<T: BC4sOperations>(block_bytes: [u8; 16]) -> [[T; 3]; 16] {
        let (red_bytes, green_bytes) = split_16(block_bytes);
        let red = bc4s_gray(red_bytes);
        let green = bc4s_gray(green_bytes);

        let mut pixels: [[T; 3]; 16] = Default::default();
        for (i, pixel) in pixels.iter_mut().enumerate() {
            pixel[0] = red[i][0];
            pixel[1] = green[i][0];
            pixel[2] = T::HALF;
        }

        pixels
    }

    pub(crate) fn bc6_s_f32(block_bytes: [u8; 16]) -> [[f32; 3]; 16] {
        super::super::bc6::decode_bc6_block(block_bytes, super::super::bc6::BC6HFormat::SignedF16)
            .map(|p| p.map(fp16::f32))
    }
    pub(crate) fn bc6_s_u16(block_bytes: [u8; 16]) -> [[u16; 3]; 16] {
        super::super::bc6::decode_bc6_block(block_bytes, super::super::bc6::BC6HFormat::SignedF16)
            .map(|p| p.map(fp16::n16))
    }
    pub(crate) fn bc6_s_u8(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        super::super::bc6::decode_bc6_block(block_bytes, super::super::bc6::BC6HFormat::SignedF16)
            .map(|p| p.map(fp16::n8))
    }
    pub(crate) fn bc6_u_f32(block_bytes: [u8; 16]) -> [[f32; 3]; 16] {
        super::super::bc6::decode_bc6_block(block_bytes, super::super::bc6::BC6HFormat::UnsignedF16)
            .map(|p| p.map(bc6h_uf16::f32))
    }
    pub(crate) fn bc6_u_u16(block_bytes: [u8; 16]) -> [[u16; 3]; 16] {
        super::super::bc6::decode_bc6_block(block_bytes, super::super::bc6::BC6HFormat::UnsignedF16)
            .map(|p| p.map(bc6h_uf16::n16))
    }
    pub(crate) fn bc6_u_u8(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        super::super::bc6::decode_bc6_block(block_bytes, super::super::bc6::BC6HFormat::UnsignedF16)
            .map(|p| p.map(bc6h_uf16::n8))
    }

    /// Decodes a BC7 UNORM block into 16 RGBA pixels.
    pub(crate) fn bc7_u8_rgba(block_bytes: [u8; 16]) -> [[u8; 4]; 16] {
        super::super::bc7::decode_bc7_block(block_bytes)
    }
    pub(crate) fn bc7_u16_rgba(block_bytes: [u8; 16]) -> [[u16; 4]; 16] {
        super::super::bc7::decode_bc7_block(block_bytes).map(|p| p.map(n8::n16))
    }
    pub(crate) fn bc7_f32_rgba(block_bytes: [u8; 16]) -> [[f32; 4]; 16] {
        super::super::bc7::decode_bc7_block(block_bytes).map(|p| p.map(n8::f32))
    }
}
