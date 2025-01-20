use super::read_write::for_each_block_4x4;
use super::{Args, Decoder, DecoderSet, WithPrecision};

use crate::Channels::*;

// helpers

macro_rules! gray {
    ($out:ty, $f:expr) => {
        Decoder::new_without_rect_decode(
            Grayscale,
            <$out as WithPrecision>::PRECISION,
            |Args(r, out, context)| {
                let f = $f;
                for_each_block_4x4(r, out, context.size, |pixel| -> [[$out; 1]; 16] {
                    f(pixel)
                })
            },
        )
    };
}
macro_rules! rgb {
    ($out:ty, $f:expr) => {
        Decoder::new_without_rect_decode(
            Rgb,
            <$out as WithPrecision>::PRECISION,
            |Args(r, out, context)| {
                let f = $f;
                for_each_block_4x4(r, out, context.size, |pixel| -> [[$out; 3]; 16] {
                    f(pixel)
                })
            },
        )
    };
}
macro_rules! rgba {
    ($out:ty, $f:expr) => {
        Decoder::new_without_rect_decode(
            Rgba,
            <$out as WithPrecision>::PRECISION,
            |Args(r, out, context)| {
                let f = $f;
                for_each_block_4x4(r, out, context.size, |pixel| -> [[$out; 4]; 16] {
                    f(pixel)
                })
            },
        )
    };
}

pub(crate) const BC1_UNORM: DecoderSet = DecoderSet::new(&[rgba!(u8, blocks::bc1_u8_rgba)]);

pub(crate) const BC2_UNORM: DecoderSet = DecoderSet::new(&[rgba!(u8, blocks::bc2_u8_rgba)]);

pub(crate) const BC3_UNORM: DecoderSet = DecoderSet::new(&[rgba!(u8, blocks::bc3_u8_rgba)]);

pub(crate) const BC3_UNORM_RXGB: DecoderSet = DecoderSet::new(&[rgb!(u8, blocks::bc3_rxgb_u8_rgb)]);

pub(crate) const BC4_UNORM: DecoderSet = DecoderSet::new(&[gray!(u8, blocks::bc4u_u8_gray)]);

pub(crate) const BC4_SNORM: DecoderSet = DecoderSet::new(&[gray!(u8, blocks::bc4s_u8_gray)]);

pub(crate) const BC5_UNORM: DecoderSet = DecoderSet::new(&[rgb!(u8, blocks::bc5u_u8_rgb)]);

pub(crate) const BC5_SNORM: DecoderSet = DecoderSet::new(&[rgb!(u8, blocks::bc5s_u8_rgb)]);

/// Internal module for the underlying logic of decoding BC1-7 blocks.
mod blocks {
    use crate::{
        decode::convert::{n4, s8, ToRgba, B5G6R5},
        util::div_round_fast,
    };

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

        fn to_rgba(rgb: [u8; 3]) -> [u8; 4] {
            [rgb[0], rgb[1], rgb[2], 255]
        }
        let (c2, c3) = if color0_u16 > color1_u16 {
            (
                to_rgba(c0_bgr.one_third_color_rgb8(c1_bgr)),
                to_rgba(c0_bgr.two_third_color_rgb8(c1_bgr)),
            )
        } else {
            (
                to_rgba(c0_bgr.mid_color_rgb8(c1_bgr)),
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

    /// Decodes a BC2 block into 16 RGBA pixels.
    pub(crate) fn bc2_u8_rgba(block_bytes: [u8; 16]) -> [[u8; 4]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc2
        let alpha_bytes: [u8; 8] = block_bytes[0..8].try_into().unwrap();
        let bc1_bytes: [u8; 8] = block_bytes[8..16].try_into().unwrap();
        let mut pixels = bc1_u8_rgba(bc1_bytes);

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

    /// Decodes a BC3 block into 16 RGBA pixels.
    pub(crate) fn bc3_u8_rgba(block_bytes: [u8; 16]) -> [[u8; 4]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc3
        let alpha_bytes: [u8; 8] = block_bytes[0..8].try_into().unwrap();
        let bc1_bytes: [u8; 8] = block_bytes[8..16].try_into().unwrap();

        let mut pixels = bc1_u8_rgba(bc1_bytes);
        let alpha = bc4u_u8_gray(alpha_bytes);

        for i in 0..4 {
            for j in 0..4 {
                pixels[i * 4 + j][3] = alpha[i * 4 + j][0];
            }
        }

        pixels
    }
    pub(crate) fn bc3_rxgb_u8_rgb(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        bc3_u8_rgba(block_bytes).map(|[_, g, b, r]| [r, g, b])
    }

    /// Decodes a BC4 UNORM block of into 16 grayscale pixels.
    pub(crate) fn bc4u_u8_gray(block_bytes: [u8; 8]) -> [[u8; 1]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc4
        let c0 = block_bytes[0];
        let c1 = block_bytes[1];

        let c0_16 = c0 as u16;
        let c1_16 = c1 as u16;

        let (c2, c3, c4, c5, c6, c7) = if c0 > c1 {
            // 6 interpolated colors
            (
                div_round_fast(c0_16 * 6 + c1_16, 7) as u8,
                div_round_fast(c0_16 * 5 + c1_16 * 2, 7) as u8,
                div_round_fast(c0_16 * 4 + c1_16 * 3, 7) as u8,
                div_round_fast(c0_16 * 3 + c1_16 * 4, 7) as u8,
                div_round_fast(c0_16 * 2 + c1_16 * 5, 7) as u8,
                div_round_fast(c0_16 + c1_16 * 6, 7) as u8,
            )
        } else {
            // 4 interpolated colors
            (
                div_round_fast(c0_16 * 4 + c1_16, 5) as u8,
                div_round_fast(c0_16 * 3 + c1_16 * 2, 5) as u8,
                div_round_fast(c0_16 * 2 + c1_16 * 3, 5) as u8,
                div_round_fast(c0_16 + c1_16 * 4, 5) as u8,
                0,
                255,
            )
        };

        let mut pixels: [[u8; 1]; 16] = Default::default();

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

    /// Decodes a BC4 SNORM block of into 16 grayscale pixels.
    pub(crate) fn bc4s_u8_gray(block_bytes: [u8; 8]) -> [[u8; 1]; 16] {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc4
        let red0 = block_bytes[0];
        let red1 = block_bytes[1];

        let c0 = s8::n8(red0);
        let c1 = s8::n8(red1);

        // exact f32 values of c0 and c1
        const CONVERSION_FACTOR: f32 = 255.0 / 254.0;
        let c0_f = red0.wrapping_add(128).saturating_sub(1) as f32 * CONVERSION_FACTOR;
        let c1_f = red1.wrapping_add(128).saturating_sub(1) as f32 * CONVERSION_FACTOR;

        fn interpolate(red0: f32, red1: f32, blend: f32) -> u8 {
            (red0 * (1.0 - blend) + red1 * blend + 0.5) as u8
        }
        let (c2, c3, c4, c5, c6, c7) = if c0 > c1 {
            // 6 interpolated colors
            (
                interpolate(c0_f, c1_f, 1.0 / 7.0),
                interpolate(c0_f, c1_f, 2.0 / 7.0),
                interpolate(c0_f, c1_f, 3.0 / 7.0),
                interpolate(c0_f, c1_f, 4.0 / 7.0),
                interpolate(c0_f, c1_f, 5.0 / 7.0),
                interpolate(c0_f, c1_f, 6.0 / 7.0),
            )
        } else {
            // 4 interpolated colors
            (
                interpolate(c0_f, c1_f, 1.0 / 5.0),
                interpolate(c0_f, c1_f, 2.0 / 5.0),
                interpolate(c0_f, c1_f, 3.0 / 5.0),
                interpolate(c0_f, c1_f, 4.0 / 5.0),
                0,
                255,
            )
        };

        let mut pixels: [[u8; 1]; 16] = Default::default();

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
    pub(crate) fn bc5u_u8_rgb(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        let red = bc4u_u8_gray(block_bytes[0..8].try_into().unwrap());
        let green = bc4u_u8_gray(block_bytes[8..16].try_into().unwrap());

        let mut pixels: [[u8; 3]; 16] = Default::default();
        for (i, pixel) in pixels.iter_mut().enumerate() {
            pixel[0] = red[i][0];
            pixel[1] = green[i][0];
            pixel[2] = 0;
        }

        pixels
    }

    /// Decodes a BC5 UNORM block into 16 RGB pixels.
    pub(crate) fn bc5s_u8_rgb(block_bytes: [u8; 16]) -> [[u8; 3]; 16] {
        let red = bc4s_u8_gray(block_bytes[0..8].try_into().unwrap());
        let green = bc4s_u8_gray(block_bytes[8..16].try_into().unwrap());

        let mut pixels: [[u8; 3]; 16] = Default::default();
        for (i, pixel) in pixels.iter_mut().enumerate() {
            pixel[0] = red[i][0];
            pixel[1] = green[i][0];
            pixel[2] = 128;
        }

        pixels
    }
}
