use crate::cast::FromLeBytes;
use crate::util::closure_types;
use crate::{n1, n8, yuv10, yuv16, yuv8, WithPrecision};
use crate::{Channels::*, ColorFormat};

use super::read_write::{
    for_each_block_rect_untyped, for_each_block_untyped, process_2x1_blocks_helper,
    process_8x1_blocks_helper, PixelRange,
};
use super::{Args, Decoder, DecoderSet, RArgs};

// helpers

macro_rules! underlying {
    ($channels:expr, $out:ty, $bpb:literal, $f:expr) => {{
        const BYTES_PER_BLOCK: usize = $bpb;
        const CHANNELS: usize = $channels.count() as usize;
        type OutPixel = [$out; CHANNELS];

        fn process_blocks(
            encoded_blocks: &[u8],
            decoded: &mut [u8],
            _stride: usize,
            range: PixelRange,
        ) {
            let f = closure_types::<[u8; BYTES_PER_BLOCK], [OutPixel; 2], _>($f);
            process_2x1_blocks_helper(encoded_blocks, decoded, range, f)
        }

        const NATIVE_COLOR: ColorFormat =
            ColorFormat::new($channels, <$out as WithPrecision>::PRECISION);

        Decoder::new_with_all_channels(
            NATIVE_COLOR,
            |Args(r, out, context)| {
                for_each_block_untyped::<2, 1, BYTES_PER_BLOCK, OutPixel>(
                    r,
                    out,
                    context,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
            |RArgs(r, out, row_pitch, rect, context)| {
                for_each_block_rect_untyped::<2, 1, BYTES_PER_BLOCK>(
                    r,
                    out,
                    row_pitch,
                    context,
                    rect,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
        )
    }};
}

macro_rules! rgb {
    ($out:ty, $f:expr) => {
        underlying!(Rgb, $out, 4, $f)
    };
    ($out:ty, $bpb:literal, $f:expr) => {
        underlying!(Rgb, $out, $bpb, $f)
    };
}

macro_rules! r1 {
    ($channels:expr, $out:ty, $f:expr) => {{
        const CHANNELS: usize = $channels.count() as usize;
        type OutPixel = [$out; CHANNELS];

        fn process_blocks(
            encoded_blocks: &[u8],
            decoded: &mut [u8],
            stride: usize,
            range: PixelRange,
        ) {
            let f = closure_types::<u8, [OutPixel; 8], _>($f);
            process_8x1_blocks_helper(encoded_blocks, decoded, stride, range, f)
        }

        const NATIVE_COLOR: ColorFormat =
            ColorFormat::new($channels, <$out as WithPrecision>::PRECISION);

        Decoder::new_with_all_channels(
            NATIVE_COLOR,
            |Args(r, out, context)| {
                for_each_block_untyped::<8, 1, 1, OutPixel>(
                    r,
                    out,
                    context,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
            |RArgs(r, out, row_pitch, rect, context)| {
                for_each_block_rect_untyped::<8, 1, 1>(
                    r,
                    out,
                    row_pitch,
                    context,
                    rect,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
        )
    }};
}

// decoders

#[inline]
fn decode_rg_bg<T: Copy>([r, g1, b, g2]: [T; 4]) -> [[T; 3]; 2] {
    [[r, g1, b], [r, g2, b]]
}
pub(crate) const R8G8_B8G8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, decode_rg_bg),
    rgb!(u16, |pair| decode_rg_bg(pair.map(n8::n16))),
    rgb!(f32, |pair| decode_rg_bg(pair.map(n8::f32))),
]);

#[inline]
fn decode_gr_bg<T: Copy>([g1, r, g2, b]: [T; 4]) -> [[T; 3]; 2] {
    [[r, g1, b], [r, g2, b]]
}
pub(crate) const G8R8_G8B8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, decode_gr_bg),
    rgb!(u16, |pair| decode_gr_bg(pair.map(n8::n16))),
    rgb!(f32, |pair| decode_gr_bg(pair.map(n8::f32))),
]);

#[inline]
fn decode_yuv2<T>([y0, u0, y1, v0]: [u8; 4], decode: impl Fn([u8; 3]) -> T) -> [T; 2] {
    [decode([y0, u0, v0]), decode([y1, u0, v0])]
}
pub(crate) const YUY2: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |pair| decode_yuv2(pair, yuv8::n8)),
    rgb!(u16, |pair| decode_yuv2(pair, yuv8::n16)),
    rgb!(f32, |pair| decode_yuv2(pair, yuv8::f32)),
]);

#[inline]
fn decode_uyvy<T>([u0, y0, v0, y1]: [u8; 4], decode: impl Fn([u8; 3]) -> T) -> [T; 2] {
    [decode([y0, u0, v0]), decode([y1, u0, v0])]
}
pub(crate) const UYVY: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |pair| decode_uyvy(pair, yuv8::n8)),
    rgb!(u16, |pair| decode_uyvy(pair, yuv8::n16)),
    rgb!(f32, |pair| decode_uyvy(pair, yuv8::f32)),
]);

#[inline]
fn decode_y210<T>(block: [u8; 8], decode: impl Fn([u16; 3]) -> T) -> [T; 2] {
    let yuyv: [u16; 4] = FromLeBytes::from_le_bytes(block);
    let [y0, u0, y1, v0]: [u16; 4] = yuyv.map(|c| c >> 6);
    [decode([y0, u0, v0]), decode([y1, u0, v0])]
}
pub(crate) const Y210: DecoderSet = DecoderSet::new(&[
    rgb!(u16, 8, |pair| decode_y210(pair, yuv10::n16)),
    rgb!(f32, 8, |pair| decode_y210(pair, yuv10::f32)),
    rgb!(u8, 8, |pair| decode_y210(pair, yuv10::n8)),
]);

#[inline]
fn decode_y216<T>(block: [u8; 8], decode: impl Fn([u16; 3]) -> T) -> [T; 2] {
    let [y0, u0, y1, v0]: [u16; 4] = FromLeBytes::from_le_bytes(block);
    [decode([y0, u0, v0]), decode([y1, u0, v0])]
}
pub(crate) const Y216: DecoderSet = DecoderSet::new(&[
    rgb!(u16, 8, |pair| decode_y216(pair, yuv16::n16)),
    rgb!(f32, 8, |pair| decode_y216(pair, yuv16::f32)),
    rgb!(u8, 8, |pair| decode_y216(pair, yuv16::n8)),
]);

#[inline]
fn r1_bits(bits: u8) -> [u8; 8] {
    let mut out = [0; 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        out[i] = (bits >> (7 - i)) & 1;
    }
    out
}
pub(crate) const R1_UNORM: DecoderSet = DecoderSet::new(&[
    r1!(Grayscale, u8, |block| r1_bits(block)
        .map(n1::n8)
        .map(|p| [p])),
    r1!(Grayscale, u16, |block| r1_bits(block)
        .map(n1::n16)
        .map(|p| [p])),
    r1!(Grayscale, f32, |block| r1_bits(block)
        .map(n1::f32)
        .map(|p| [p])),
]);
