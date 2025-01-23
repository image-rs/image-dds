use super::read_write::{
    process_pixels_helper, process_pixels_helper_unroll, PixelArgs, ProcessPixelsFn,
};
use super::{Args, DecodeFn, DecoderSet, UncompressedDecoder};
use crate::{
    fp, fp10, fp11, fp16, n10, n16, n2, n4, n8, rgb9995f, s16, s8, xr10, yuv10, yuv16, yuv8, Norm,
    SwapRB, ToRgba, WithPrecision, B5G5R5A1, B5G6R5,
};

use crate::util::{closure_types, le_to_native_endian_16, le_to_native_endian_32};
use crate::{Channels::*, ColorFormat, Precision::*};

// helpers

macro_rules! underlying {
    ($channels:expr, $out:ty, $in_pixel:ty, process_fn = $f:ident) => {{
        const OUT_COUNT: usize = $channels.count() as usize;
        type InPixel = $in_pixel;
        type OutPixel = [$out; OUT_COUNT];

        UncompressedDecoder::new::<InPixel, OutPixel>(
            ColorFormat::new($channels, <$out as WithPrecision>::PRECISION),
            $f,
        )
    }};
    ($channels:expr, $out:ty, $in_pixel:ty, $f:expr) => {{
        const OUT_COUNT: usize = $channels.count() as usize;
        type InPixel = $in_pixel;
        type OutPixel = [$out; OUT_COUNT];

        fn process_pixels(PixelArgs(encoded, decoded): PixelArgs) {
            let f = closure_types::<InPixel, OutPixel, _>($f);
            process_pixels_helper(encoded, decoded, f);
        }

        UncompressedDecoder::new::<InPixel, OutPixel>(
            ColorFormat::new($channels, <$out as WithPrecision>::PRECISION),
            process_pixels,
        )
    }};
}
macro_rules! gray {
    ($out:ty, $in_pixel:ty, process_fn = $f:ident) => {
        underlying!(Grayscale, $out, $in_pixel, process_fn = $f)
    };
    ($out:ty, $in_pixel:ty, $f:expr) => {
        underlying!(Grayscale, $out, $in_pixel, $f)
    };
}
macro_rules! alpha {
    ($out:ty, $in_pixel:ty, process_fn = $f:ident) => {
        underlying!(Alpha, $out, $in_pixel, process_fn = $f)
    };
    ($out:ty, $in_pixel:ty, $f:expr) => {
        underlying!(Alpha, $out, $in_pixel, $f)
    };
}
macro_rules! rgb {
    ($out:ty, $in_pixel:ty, process_fn = $f:ident) => {
        underlying!(Rgb, $out, $in_pixel, process_fn = $f)
    };
    ($out:ty, $in_pixel:ty, $f:expr) => {
        underlying!(Rgb, $out, $in_pixel, $f)
    };
}
macro_rules! rgba {
    ($out:ty, $in_pixel:ty, process_fn = $f:ident) => {
        underlying!(Rgba, $out, $in_pixel, process_fn = $f)
    };
    ($out:ty, $in_pixel:ty, $f:expr) => {
        underlying!(Rgba, $out, $in_pixel, $f)
    };
}

// Dedicated (whole-image) decoding functions.
//
// Some formats allow us to basically just memcpy the data into the output
// buffer. This allows for some very efficient decoding.
// Note that this is only an optimization, and not required for correctness.

const COPY_U8: DecodeFn = |Args(r, out, _)| {
    r.read_exact(out)?;
    Ok(())
};
const COPY_U16: DecodeFn = |Args(r, out, _)| {
    r.read_exact(out)?;
    le_to_native_endian_16(out);
    Ok(())
};
const COPY_U32: DecodeFn = |Args(r, out, _)| {
    r.read_exact(out)?;
    le_to_native_endian_32(out);
    Ok(())
};
const COPY_S8: DecodeFn = |Args(r, out, _)| {
    r.read_exact(out)?;
    out.iter_mut().for_each(|v| *v = s8::n8(*v));
    Ok(())
};

// Dedicated pixel-processing functions.
//
// Some formats have pixel-processing functions that are virtually identical.
// By defining them once here, we (1) save the compiler from having to compile
// duplicate functions and (2) share optimizations across formats.

// TODO: rename
macro_rules! foo {
    ($f:expr) => {
        |PixelArgs(encoded, decoded)| process_pixels_helper(encoded, decoded, $f)
    };
}

const PROCESS_COPY: ProcessPixelsFn = |PixelArgs(encoded, decoded)| {
    debug_assert!(encoded.len() == decoded.len());
    decoded.copy_from_slice(encoded);
};

const N8_TO_U8: ProcessPixelsFn = PROCESS_COPY;
const N8_TO_U16: ProcessPixelsFn = foo!(n8::n16);
const N8_TO_F32: ProcessPixelsFn = foo!(n8::f32);

const S8_TO_U8: ProcessPixelsFn = foo!(s8::n8);
const S8_TO_U16: ProcessPixelsFn = foo!(s8::n16);
const S8_TO_F32: ProcessPixelsFn = foo!(s8::uf32);

const N16_TO_U8: ProcessPixelsFn = foo!(n16::n8);
const N16_TO_U16: ProcessPixelsFn = foo!(|x: u16| x);
const N16_TO_F32: ProcessPixelsFn = foo!(n16::f32);

const S16_TO_U8: ProcessPixelsFn = foo!(s16::n8);
const S16_TO_U16: ProcessPixelsFn = foo!(s16::n16);
const S16_TO_F32: ProcessPixelsFn = foo!(s16::uf32);

const F16_TO_U8: ProcessPixelsFn = foo!(fp16::n8);
const F16_TO_U16: ProcessPixelsFn = |PixelArgs(encoded, decoded)| {
    process_pixels_helper_unroll::<4, _, _, _>(encoded, decoded, fp16::n16)
};
const F16_TO_F32: ProcessPixelsFn = |PixelArgs(encoded, decoded)| {
    process_pixels_helper_unroll::<4, _, _, _>(encoded, decoded, fp16::f32)
};

const F32_TO_U8: ProcessPixelsFn = foo!(fp::n8);
const F32_TO_U16: ProcessPixelsFn = foo!(fp::n16);
const F32_TO_F32: ProcessPixelsFn = foo!(|x: f32| x);

// decoders

pub(crate) const R8G8B8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u8, [u8; 3], process_fn = N8_TO_U8),
    rgb!(u16, [u8; 3], process_fn = N8_TO_U16),
    rgb!(f32, [u8; 3], process_fn = N8_TO_F32),
])
.add_specialized(Rgb, U8, COPY_U8);

pub(crate) const B8G8R8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u8, [u8; 3], |bgr| bgr.swap_rb()),
    rgb!(u16, [u8; 3], |bgr| bgr.swap_rb().map(n8::n16)),
    rgb!(f32, [u8; 3], |bgr| bgr.swap_rb().map(n8::f32)),
]);

pub(crate) const R8G8B8A8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u8, [u8; 4], process_fn = N8_TO_U8),
    rgba!(u16, [u8; 4], process_fn = N8_TO_U16),
    rgba!(f32, [u8; 4], process_fn = N8_TO_F32),
])
.add_specialized(Rgba, U8, COPY_U8);

pub(crate) const R8G8B8A8_SNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u8, [u8; 4], process_fn = S8_TO_U8),
    rgba!(u16, [u8; 4], process_fn = S8_TO_U16),
    rgba!(f32, [u8; 4], process_fn = S8_TO_F32),
])
.add_specialized(Rgba, U8, COPY_S8);

pub(crate) const B8G8R8A8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u8, [u8; 4], |bgra| bgra.swap_rb()),
    rgba!(u16, [u8; 4], |bgra| bgra.swap_rb().map(n8::n16)),
    rgba!(f32, [u8; 4], |bgra| bgra.swap_rb().map(n8::f32)),
])
.add_specialized(Rgba, U8, |Args(r, out, _)| {
    // read everything in BGRA order
    r.read_exact(out)?;
    // swap R and B
    for i in (0..out.len()).step_by(4) {
        out.swap(i, i + 2);
    }
    Ok(())
});

#[inline(always)]
fn bgrx_to_rgb([b, g, r, _]: [u8; 4]) -> [u8; 3] {
    [r, g, b]
}
pub(crate) const B8G8R8X8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u8, [u8; 4], bgrx_to_rgb),
    rgb!(u16, [u8; 4], |bgrx| bgrx_to_rgb(bgrx).map(n8::n16)),
    rgb!(f32, [u8; 4], |bgrx| bgrx_to_rgb(bgrx).map(n8::f32)),
    // the format is literally optimized for this
    rgba!(u8, [u8; 4], |bgrx| bgrx_to_rgb(bgrx).to_rgba()),
]);

pub(crate) const B5G6R5_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u8, [u16; 1], |[bgr]| B5G6R5::from_u16(bgr).to_n8()),
    rgb!(u16, [u16; 1], |[bgr]| B5G6R5::from_u16(bgr).to_n16()),
    rgb!(f32, [u16; 1], |[bgr]| B5G6R5::from_u16(bgr).to_f32()),
]);

pub(crate) const B5G5R5A1_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u8, [u16; 1], |[bgra]| B5G5R5A1::from_u16(bgra).to_n8()),
    rgba!(u16, [u16; 1], |[bgra]| B5G5R5A1::from_u16(bgra).to_n16()),
    rgba!(f32, [u16; 1], |[bgra]| B5G5R5A1::from_u16(bgra).to_f32()),
]);

#[inline(always)]
fn unpack_bgra4444([low, high]: [u8; 2]) -> [u8; 4] {
    let b4 = low & 0xF;
    let g4 = (low >> 4) & 0xF;
    let r4 = high & 0xF;
    let a4 = (high >> 4) & 0xF;

    [r4, g4, b4, a4]
}
pub(crate) const B4G4R4A4_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u8, [u8; 2], |bgra| unpack_bgra4444(bgra).map(n4::n8)),
    rgba!(u16, [u8; 2], |bgra| unpack_bgra4444(bgra).map(n4::n16)),
    rgba!(f32, [u8; 2], |bgra| unpack_bgra4444(bgra).map(n4::f32)),
]);

#[inline(always)]
fn unpack_abgr4444([low, high]: [u8; 2]) -> [u8; 4] {
    let a4 = low & 0xF;
    let b4 = (low >> 4) & 0xF;
    let g4 = high & 0xF;
    let r4 = (high >> 4) & 0xF;

    [r4, g4, b4, a4]
}
pub(crate) const A4B4G4R4_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u8, [u8; 2], |bgra| unpack_abgr4444(bgra).map(n4::n8)),
    rgba!(u16, [u8; 2], |bgra| unpack_abgr4444(bgra).map(n4::n16)),
    rgba!(f32, [u8; 2], |bgra| unpack_abgr4444(bgra).map(n4::f32)),
]);

pub(crate) const R8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    gray!(u8, [u8; 1], process_fn = N8_TO_U8),
    gray!(u16, [u8; 1], process_fn = N8_TO_U16),
    gray!(f32, [u8; 1], process_fn = N8_TO_F32),
])
.add_specialized(Grayscale, U8, COPY_U8);

pub(crate) const R8_SNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    gray!(u8, [u8; 1], process_fn = S8_TO_U8),
    gray!(u16, [u8; 1], process_fn = S8_TO_U16),
    gray!(f32, [u8; 1], process_fn = S8_TO_F32),
])
.add_specialized(Grayscale, U8, COPY_S8);

pub(crate) const R8G8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u8, [u8; 2], |rg| [rg[0], rg[1], 0]),
    rgb!(u16, [u8; 2], |rg| [rg[0], rg[1], 0].map(n8::n16)),
    rgb!(f32, [u8; 2], |rg| [rg[0], rg[1], 0].map(n8::f32)),
]);

pub(crate) const R8G8_SNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u8, [u8; 2], |[r, g]| [s8::n8(r), s8::n8(g), Norm::HALF]),
    rgb!(u16, [u8; 2], |[r, g]| [s8::n16(r), s8::n16(g), Norm::HALF]),
    rgb!(f32, [u8; 2], |[r, g]| [
        s8::uf32(r),
        s8::uf32(g),
        Norm::HALF
    ]),
]);

pub(crate) const A8_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    alpha!(u8, [u8; 1], process_fn = N8_TO_U8),
    alpha!(u16, [u8; 1], process_fn = N8_TO_U16),
    alpha!(f32, [u8; 1], process_fn = N8_TO_F32),
])
.add_specialized(Alpha, U8, COPY_U8);

pub(crate) const R16_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    gray!(u16, [u16; 1], process_fn = N16_TO_U16),
    gray!(u8, [u16; 1], process_fn = N16_TO_U8),
    gray!(f32, [u16; 1], process_fn = N16_TO_F32),
])
.add_specialized(Grayscale, U16, COPY_U16);

pub(crate) const R16_SNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    gray!(u16, [u16; 1], process_fn = S16_TO_U16),
    gray!(u8, [u16; 1], process_fn = S16_TO_U8),
    gray!(f32, [u16; 1], process_fn = S16_TO_F32),
]);

pub(crate) const R16G16_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u16, [u16; 2], |rg| [rg[0], rg[1], 0]),
    rgb!(u8, [u16; 2], |rg| [rg[0], rg[1], 0].map(n16::n8)),
    rgb!(f32, [u16; 2], |rg| [rg[0], rg[1], 0].map(n16::f32)),
]);

pub(crate) const R16G16_SNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(u16, [u16; 2], |[r, g]| [
        s16::n16(r),
        s16::n16(g),
        Norm::HALF
    ]),
    rgb!(u8, [u16; 2], |[r, g]| [s16::n8(r), s16::n8(g), Norm::HALF]),
    rgb!(f32, [u16; 2], |[r, g]| [
        s16::uf32(r),
        s16::uf32(g),
        Norm::HALF
    ]),
]);

pub(crate) const R16G16B16A16_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u16, [u16; 4], process_fn = N16_TO_U16),
    rgba!(u8, [u16; 4], process_fn = N16_TO_U8),
    rgba!(f32, [u16; 4], process_fn = N16_TO_F32),
])
.add_specialized(Rgba, U16, COPY_U16);

pub(crate) const R16G16B16A16_SNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u16, [u16; 4], process_fn = S16_TO_U16),
    rgba!(u8, [u16; 4], process_fn = S16_TO_U8),
    rgba!(f32, [u16; 4], process_fn = S16_TO_F32),
]);

#[inline(always)]
fn unpack_rgba1010102(rgba: u32) -> (u16, u16, u16, u8) {
    let r10 = rgba & 0x3FF;
    let g10 = (rgba >> 10) & 0x3FF;
    let b10 = (rgba >> 20) & 0x3FF;
    let a2 = (rgba >> 30) & 0x3;
    (r10 as u16, g10 as u16, b10 as u16, a2 as u8)
}
pub(crate) const R10G10B10A2_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u16, [u32; 1], |[rgba]| {
        let (r, g, b, a) = unpack_rgba1010102(rgba);
        [n10::n16(r), n10::n16(g), n10::n16(b), n2::n16(a)]
    }),
    rgba!(u8, [u32; 1], |[rgba]| {
        let (r, g, b, a) = unpack_rgba1010102(rgba);
        [n10::n8(r), n10::n8(g), n10::n8(b), n2::n8(a)]
    }),
    rgba!(f32, [u32; 1], |[rgba]| {
        let (r, g, b, a) = unpack_rgba1010102(rgba);
        [n10::f32(r), n10::f32(g), n10::f32(b), n2::f32(a)]
    }),
]);

#[inline(always)]
fn unpack_rgb111110f(rgb: u32) -> [u16; 3] {
    let r11 = (rgb & 0x7FF) as u16;
    let g11 = ((rgb >> 11) & 0x7FF) as u16;
    let b10 = ((rgb >> 22) & 0x3FF) as u16;
    [r11, g11, b10]
}
pub(crate) const R11G11B10_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(f32, [u32; 1], |[rgb]| {
        let [r11, g11, b10] = unpack_rgb111110f(rgb);
        [fp11::f32(r11), fp11::f32(g11), fp10::f32(b10)]
    }),
    rgb!(u16, [u32; 1], |[rgb]| {
        let [r11, g11, b10] = unpack_rgb111110f(rgb);
        [fp11::n16(r11), fp11::n16(g11), fp10::n16(b10)]
    }),
    rgb!(u8, [u32; 1], |[rgb]| {
        let [r11, g11, b10] = unpack_rgb111110f(rgb);
        [fp11::n8(r11), fp11::n8(g11), fp10::n8(b10)]
    }),
]);

pub(crate) const R9G9B9E5_SHAREDEXP: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(f32, [u32; 1], |[rgb]| rgb9995f::f32(rgb)),
    rgb!(u16, [u32; 1], |[rgb]| rgb9995f::n16(rgb)),
    rgb!(u8, [u32; 1], |[rgb]| rgb9995f::n8(rgb)),
]);

pub(crate) const R16_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    gray!(f32, [u16; 1], process_fn = F16_TO_F32),
    gray!(u8, [u16; 1], process_fn = F16_TO_U8),
    gray!(u16, [u16; 1], process_fn = F16_TO_U16),
]);

pub(crate) const R16G16_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(f32, [u16; 2], |[r, g]| [fp16::f32(r), fp16::f32(g), 0.0]),
    rgb!(u16, [u16; 2], |[r, g]| [fp16::n16(r), fp16::n16(g), 0]),
    rgb!(u8, [u16; 2], |[r, g]| [fp16::n8(r), fp16::n8(g), 0]),
]);

pub(crate) const R16G16B16A16_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(f32, [u16; 4], process_fn = F16_TO_F32),
    rgba!(u8, [u16; 4], process_fn = F16_TO_U8),
    rgba!(u16, [u16; 4], process_fn = F16_TO_U16),
]);

pub(crate) const R32_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    gray!(f32, [f32; 1], process_fn = F32_TO_F32),
    gray!(u8, [f32; 1], process_fn = F32_TO_U8),
    gray!(u16, [f32; 1], process_fn = F32_TO_U16),
])
.add_specialized(Grayscale, F32, COPY_U32);

pub(crate) const R32G32_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(f32, [f32; 2], |[r, g]| [r, g, 0.0]),
    rgb!(u16, [f32; 2], |[r, g]| [fp::n16(r), fp::n16(g), 0]),
    rgb!(u8, [f32; 2], |[r, g]| [fp::n8(r), fp::n8(g), 0]),
]);

pub(crate) const R32G32B32_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    rgb!(f32, [f32; 3], process_fn = F32_TO_F32),
    rgb!(u8, [f32; 3], process_fn = F32_TO_U8),
    rgb!(u16, [f32; 3], process_fn = F32_TO_U16),
])
.add_specialized(Rgb, F32, COPY_U32);

pub(crate) const R32G32B32A32_FLOAT: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(f32, [f32; 4], process_fn = F32_TO_F32),
    rgba!(u8, [f32; 4], process_fn = F32_TO_U8),
    rgba!(u16, [f32; 4], process_fn = F32_TO_U16),
])
.add_specialized(Rgba, F32, COPY_U32);

#[inline(always)]
fn unpack_rgba1010102_xr(rgba: u32) -> ([u16; 3], u8) {
    let r_fixed = rgba & 0x3FF;
    let g_fixed = (rgba >> 10) & 0x3FF;
    let b_fixed = (rgba >> 20) & 0x3FF;
    let a2 = (rgba >> 30) & 0x3;

    ([r_fixed as u16, g_fixed as u16, b_fixed as u16], a2 as u8)
}
pub(crate) const R10G10B10_XR_BIAS_A2_UNORM: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(f32, [u32; 1], |[rgba]| {
        let (rgb, a2) = unpack_rgba1010102_xr(rgba);
        let [r, g, b] = rgb.map(xr10::f32);
        [r, g, b, n2::f32(a2)]
    }),
    rgba!(u16, [u32; 1], |[rgba]| {
        let (rgb, a2) = unpack_rgba1010102_xr(rgba);
        let [r, g, b] = rgb.map(xr10::n16);
        [r, g, b, n2::n16(a2)]
    }),
    rgba!(u8, [u32; 1], |[rgba]| {
        let (rgb, a2) = unpack_rgba1010102_xr(rgba);
        let [r, g, b] = rgb.map(xr10::n8);
        [r, g, b, n2::n8(a2)]
    }),
]);

fn unpack_ayuv<T>(
    ayuv: [u8; 4],
    decode_yuv: impl Fn([u8; 3]) -> [T; 3],
    decode_alpha: impl Fn(u8) -> T,
) -> [T; 4] {
    let [v, u, y, a] = ayuv;
    let [y, u, v] = decode_yuv([y, u, v]);
    [y, u, v, decode_alpha(a)]
}
pub(crate) const AYUV: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u8, [u8; 4], |ayuv| unpack_ayuv(ayuv, yuv8::n8, |x| x)),
    rgba!(u16, [u8; 4], |ayuv| unpack_ayuv(ayuv, yuv8::n16, n8::n16)),
    rgba!(f32, [u8; 4], |ayuv| unpack_ayuv(ayuv, yuv8::f32, n8::f32)),
]);

fn unpack_y410<T>(
    y410: u32,
    decode_yuv: impl Fn([u16; 3]) -> [T; 3],
    decode_alpha: impl Fn(u8) -> T,
) -> [T; 4] {
    let ([u, y, v], a) = unpack_rgba1010102_xr(y410);
    let [y, u, v] = decode_yuv([y, u, v]);
    [y, u, v, decode_alpha(a)]
}
pub(crate) const Y410: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u16, u32, |y410| unpack_y410(y410, yuv10::n16, n2::n16)),
    rgba!(f32, u32, |y410| unpack_y410(y410, yuv10::f32, n2::f32)),
    rgba!(u8, u32, |y410| unpack_y410(y410, yuv10::n8, n2::n8)),
]);

fn unpack_y416<T>(
    y416: [u16; 4],
    decode_yuv: impl Fn([u16; 3]) -> [T; 3],
    decode_alpha: impl Fn(u16) -> T,
) -> [T; 4] {
    let [u, y, v, a] = y416;
    let [y, u, v] = decode_yuv([y, u, v]);
    [y, u, v, decode_alpha(a)]
}
pub(crate) const Y416: DecoderSet = DecoderSet::new_uncompressed(&[
    rgba!(u16, [u16; 4], |y416| unpack_y416(y416, yuv16::n16, |x| x)),
    rgba!(f32, [u16; 4], |y416| unpack_y416(
        y416,
        yuv16::f32,
        n16::f32
    )),
    rgba!(u8, [u16; 4], |y416| unpack_y416(y416, yuv16::n8, n16::n8)),
]);
