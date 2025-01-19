use super::convert::{
    f10_to_f32, f11_to_f32, f16_to_f32, fp, n10, n16, n2, n4, n8, s16, s8, xr10, SwapRB, ToRgb,
    ToRgba, B5G5R5A1, B5G6R5,
};
use super::read_write::{for_each_pixel, for_each_pixel_rect};
use super::{Args, DecodeFn, Decoder, DecoderSet, RArgs, WithPrecision};

use crate::util::{le_to_native_endian_16, le_to_native_endian_32};
use crate::Channels::*;

// helpers

macro_rules! underlying {
    ($channels:expr, $out:ty, $f:expr) => {
        Decoder::new(
            $channels,
            <$out as WithPrecision>::PRECISION,
            |Args(r, out, _)| {
                let f = $f;
                const COUNT: usize = $channels.count() as usize;
                for_each_pixel(r, out, |pixel| -> [$out; COUNT] { f(pixel) })
            },
            |RArgs(r, out, row_pitch, rect, context)| {
                let f = $f;
                const COUNT: usize = $channels.count() as usize;
                for_each_pixel_rect(
                    r,
                    out,
                    row_pitch,
                    context.size,
                    rect,
                    |pixel| -> [$out; COUNT] { f(pixel) },
                )
            },
        )
    };
}
macro_rules! gray {
    ($out:ty, $f:expr) => {
        underlying!(Grayscale, $out, $f)
    };
}
macro_rules! alpha {
    ($out:ty, $f:expr) => {
        underlying!(Alpha, $out, $f)
    };
}
macro_rules! rgb {
    ($out:ty, $f:expr) => {
        underlying!(Rgb, $out, $f)
    };
}
macro_rules! rgba {
    ($out:ty, $f:expr) => {
        underlying!(Rgba, $out, $f)
    };
}

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

// decoders

pub(crate) const R8G8B8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |rgb: [u8; 3]| rgb).with_decode_fn(COPY_U8),
    rgb!(u16, |rgb: [u8; 3]| rgb.map(n8::n16)),
    rgb!(f32, |rgb: [u8; 3]| rgb.map(n8::f32)),
    rgba!(u8, |rgb: [u8; 3]| rgb.to_rgba()),
    rgba!(u16, |rgb: [u8; 3]| rgb.map(n8::n16).to_rgba()),
    rgba!(f32, |rgb: [u8; 3]| rgb.map(n8::f32).to_rgba()),
]);

pub(crate) const B8G8R8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |bgr: [u8; 3]| bgr.swap_rb()),
    rgb!(u16, |bgr: [u8; 3]| bgr.swap_rb().map(n8::n16)),
    rgb!(f32, |bgr: [u8; 3]| bgr.swap_rb().map(n8::f32)),
    rgba!(u8, |bgr: [u8; 3]| bgr.swap_rb().to_rgba()),
    rgba!(u16, |bgr: [u8; 3]| bgr.swap_rb().map(n8::n16).to_rgba()),
    rgba!(f32, |bgr: [u8; 3]| bgr.swap_rb().map(n8::f32).to_rgba()),
]);

pub(crate) const R8G8B8A8_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, |rgba: [u8; 4]| rgba).with_decode_fn(COPY_U8),
    rgba!(u16, |rgba: [u8; 4]| rgba.map(n8::n16)),
    rgba!(f32, |rgba: [u8; 4]| rgba.map(n8::f32)),
    rgb!(u8, |rgba: [u8; 4]| rgba.to_rgb()),
    rgb!(u16, |rgba: [u8; 4]| rgba.to_rgb().map(n8::n16)),
    rgb!(f32, |rgba: [u8; 4]| rgba.to_rgb().map(n8::f32)),
]);

pub(crate) const R8G8B8A8_SNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, |rgba: [u8; 4]| rgba.map(s8::n8)).with_decode_fn(COPY_S8),
    rgba!(u16, |rgba: [u8; 4]| rgba.map(s8::n16)),
    rgba!(f32, |rgba: [u8; 4]| rgba.map(s8::uf32)),
    rgb!(u8, |rgba: [u8; 4]| rgba.to_rgb().map(s8::n8)),
    rgb!(u16, |rgba: [u8; 4]| rgba.to_rgb().map(s8::n16)),
    rgb!(f32, |rgba: [u8; 4]| rgba.to_rgb().map(s8::uf32)),
]);

pub(crate) const B8G8R8A8_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, |bgra: [u8; 4]| bgra.swap_rb()).with_decode_fn(|Args(r, out, _)| {
        // read everything in BGRA order
        r.read_exact(out)?;
        // swap R and B
        for i in (0..out.len()).step_by(4) {
            out.swap(i, i + 2);
        }
        Ok(())
    }),
    rgba!(u16, |bgra: [u8; 4]| bgra.swap_rb().map(n8::n16)),
    rgba!(f32, |bgra: [u8; 4]| bgra.swap_rb().map(n8::f32)),
    rgb!(u8, |bgra: [u8; 4]| bgra.to_rgb().swap_rb()),
    rgb!(u16, |bgra: [u8; 4]| bgra.to_rgb().swap_rb().map(n8::n16)),
    rgb!(f32, |bgra: [u8; 4]| bgra.to_rgb().swap_rb().map(n8::f32)),
]);

#[inline(always)]
fn bgrx_to_rgb([b, g, r, _]: [u8; 4]) -> [u8; 3] {
    [r, g, b]
}
pub(crate) const B8G8R8X8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |bgrx: [u8; 4]| bgrx_to_rgb(bgrx)),
    rgb!(u16, |bgrx: [u8; 4]| bgrx_to_rgb(bgrx).map(n8::n16)),
    rgb!(f32, |bgrx: [u8; 4]| bgrx_to_rgb(bgrx).map(n8::f32)),
    // TODO: Optimize
    rgba!(u8, |bgrx: [u8; 4]| bgrx_to_rgb(bgrx).to_rgba()),
    rgba!(u16, |bgrx: [u8; 4]| bgrx_to_rgb(bgrx)
        .map(n8::n16)
        .to_rgba()),
    rgba!(f32, |bgrx: [u8; 4]| bgrx_to_rgb(bgrx)
        .map(n8::f32)
        .to_rgba()),
]);

pub(crate) const B5G6R5_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |[bgr]: [u16; 1]| B5G6R5::from_u16(bgr).to_n8()),
    rgb!(u16, |[bgr]: [u16; 1]| B5G6R5::from_u16(bgr).to_n16()),
    rgb!(f32, |[bgr]: [u16; 1]| B5G6R5::from_u16(bgr).to_f32()),
    rgba!(u8, |[bgr]: [u16; 1]| B5G6R5::from_u16(bgr)
        .to_n8()
        .to_rgba()),
    rgba!(u16, |[bgr]: [u16; 1]| B5G6R5::from_u16(bgr)
        .to_n16()
        .to_rgba()),
    rgba!(f32, |[bgr]: [u16; 1]| B5G6R5::from_u16(bgr)
        .to_f32()
        .to_rgba()),
]);

pub(crate) const B5G5R5A1_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, |[bgra]: [u16; 1]| B5G5R5A1::from_u16(bgra).to_n8()),
    rgba!(u16, |[bgra]: [u16; 1]| B5G5R5A1::from_u16(bgra).to_n16()),
    rgba!(f32, |[bgra]: [u16; 1]| B5G5R5A1::from_u16(bgra).to_f32()),
    rgb!(u8, |[bgra]: [u16; 1]| B5G5R5A1::from_u16(bgra)
        .to_n8()
        .to_rgb()),
    rgb!(u16, |[bgra]: [u16; 1]| B5G5R5A1::from_u16(bgra)
        .to_n16()
        .to_rgb()),
    rgb!(f32, |[bgra]: [u16; 1]| B5G5R5A1::from_u16(bgra)
        .to_f32()
        .to_rgb()),
]);

#[inline(always)]
fn unpack_bgra4444([low, high]: [u8; 2]) -> [u8; 4] {
    let b4 = low & 0xF;
    let g4 = (low >> 4) & 0xF;
    let r4 = high & 0xF;
    let a4 = (high >> 4) & 0xF;

    [r4, g4, b4, a4]
}
pub(crate) const B4G4R4A4_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u8, |bgra: [u8; 2]| unpack_bgra4444(bgra).map(n4::n8)),
    rgba!(u16, |bgra: [u8; 2]| unpack_bgra4444(bgra).map(n4::n16)),
    rgba!(f32, |bgra: [u8; 2]| unpack_bgra4444(bgra).map(n4::f32)),
    rgb!(u8, |bgra: [u8; 2]| unpack_bgra4444(bgra)
        .to_rgb()
        .map(n4::n8)),
    rgb!(u16, |bgra: [u8; 2]| unpack_bgra4444(bgra)
        .to_rgb()
        .map(n4::n16)),
    rgb!(f32, |bgra: [u8; 2]| unpack_bgra4444(bgra)
        .to_rgb()
        .map(n4::f32)),
]);

pub(crate) const R8_UNORM: DecoderSet = DecoderSet::new(&[
    gray!(u8, |r: [u8; 1]| r).with_decode_fn(COPY_U8),
    gray!(u16, |r: [u8; 1]| r.map(n8::n16)),
    gray!(f32, |r: [u8; 1]| r.map(n8::f32)),
    // TODO: Rgb and Rgba
]);

pub(crate) const R8_SNORM: DecoderSet = DecoderSet::new(&[
    gray!(u8, |r: [u8; 1]| r.map(s8::n8)).with_decode_fn(COPY_S8),
    gray!(u16, |r: [u8; 1]| r.map(s8::n16)),
    gray!(f32, |r: [u8; 1]| r.map(s8::uf32)),
    // TODO: Rgb and Rgba
]);

pub(crate) const R8G8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |rg: [u8; 2]| [rg[0], rg[1], 0]),
    rgb!(u16, |rg: [u8; 2]| [rg[0], rg[1], 0].map(n8::n16)),
    rgb!(f32, |rg: [u8; 2]| [rg[0], rg[1], 0].map(n8::f32)),
    rgba!(u8, |rg: [u8; 2]| [rg[0], rg[1], 0].to_rgba()),
    rgba!(u16, |rg: [u8; 2]| [rg[0], rg[1], 0].map(n8::n16).to_rgba()),
    rgba!(f32, |rg: [u8; 2]| [rg[0], rg[1], 0].map(n8::f32).to_rgba()),
]);

pub(crate) const R8G8_SNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |[r, g]: [u8; 2]| [s8::n8(r), s8::n8(g), 128]),
    rgb!(u16, |[r, g]: [u8; 2]| [s8::n16(r), s8::n16(g), 32768]),
    rgb!(f32, |[r, g]: [u8; 2]| [s8::uf32(r), s8::uf32(g), 0.5]),
    rgba!(u8, |[r, g]: [u8; 2]| [s8::n8(r), s8::n8(g), 128].to_rgba()),
    rgba!(u16, |[r, g]: [u8; 2]| [s8::n16(r), s8::n16(g), 32768]
        .to_rgba()),
    rgba!(f32, |[r, g]: [u8; 2]| [s8::uf32(r), s8::uf32(g), 0.5]
        .to_rgba()),
]);

pub(crate) const A8_UNORM: DecoderSet = DecoderSet::new(&[
    alpha!(u8, |a: [u8; 1]| a).with_decode_fn(COPY_U8),
    alpha!(u16, |a: [u8; 1]| a.map(n8::n16)),
    alpha!(f32, |a: [u8; 1]| a.map(n8::f32)),
    rgba!(u8, |[a]: [u8; 1]| [0, 0, 0, a]),
    rgba!(u16, |[a]: [u8; 1]| [0, 0, 0, n8::n16(a)]),
    rgba!(f32, |[a]: [u8; 1]| [0.0, 0.0, 0.0, n8::f32(a)]),
]);

pub(crate) const R16_UNORM: DecoderSet = DecoderSet::new(&[
    gray!(u16, |r: [u16; 1]| r).with_decode_fn(COPY_U16),
    gray!(u8, |r: [u16; 1]| r.map(n16::n8)),
    gray!(f32, |r: [u16; 1]| r.map(n16::f32)),
    // TODO: Rgb and Rgba
]);

pub(crate) const R16_SNORM: DecoderSet = DecoderSet::new(&[
    gray!(u16, |r: [u16; 1]| r.map(s16::n16)),
    gray!(u8, |r: [u16; 1]| r.map(s16::n8)),
    gray!(f32, |r: [u16; 1]| r.map(s16::uf32)),
    // TODO: Rgb and Rgba
]);

pub(crate) const R16G16_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u16, |rg: [u16; 2]| [rg[0], rg[1], 0]),
    rgb!(u8, |rg: [u16; 2]| [rg[0], rg[1], 0].map(n16::n8)),
    rgb!(f32, |rg: [u16; 2]| [rg[0], rg[1], 0].map(n16::f32)),
    rgba!(u8, |rg: [u16; 2]| [rg[0], rg[1], 0].map(n16::n8).to_rgba()),
    rgba!(u16, |rg: [u16; 2]| [rg[0], rg[1], 0].to_rgba()),
    rgba!(f32, |rg: [u16; 2]| [rg[0], rg[1], 0]
        .map(n16::f32)
        .to_rgba()),
]);

pub(crate) const R16G16_SNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u16, |[r, g]: [u16; 2]| [s16::n16(r), s16::n16(g), 32768]),
    rgb!(u8, |[r, g]: [u16; 2]| [s16::n8(r), s16::n8(g), 128]),
    rgb!(f32, |[r, g]: [u16; 2]| [s16::uf32(r), s16::uf32(g), 0.5]),
    rgba!(u8, |[r, g]: [u16; 2]| [s16::n8(r), s16::n8(g), 128]
        .to_rgba()),
    rgba!(u16, |[r, g]: [u16; 2]| [s16::n16(r), s16::n16(g), 32768]
        .to_rgba()),
    rgba!(f32, |[r, g]: [u16; 2]| [s16::uf32(r), s16::uf32(g), 0.5]
        .to_rgba()),
]);

pub(crate) const R16G16B16A16_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u16, |rgba: [u16; 4]| rgba).with_decode_fn(COPY_U16),
    rgba!(u8, |rgba: [u16; 4]| rgba.map(n16::n8)),
    rgba!(f32, |rgba: [u16; 4]| rgba.map(n16::f32)),
    rgb!(u8, |rgba: [u16; 4]| rgba.to_rgb().map(n16::n8)),
    rgb!(u16, |rgba: [u16; 4]| rgba.to_rgb()),
    rgb!(f32, |rgba: [u16; 4]| rgba.to_rgb().map(n16::f32)),
]);

pub(crate) const R16G16B16A16_SNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u16, |rgba: [u16; 4]| rgba.map(s16::n16)),
    rgba!(u8, |rgba: [u16; 4]| rgba.map(s16::n8)),
    rgba!(f32, |rgba: [u16; 4]| rgba.map(s16::uf32)),
    rgb!(u8, |rgba: [u16; 4]| rgba.to_rgb().map(s16::n8)),
    rgb!(u16, |rgba: [u16; 4]| rgba.to_rgb().map(s16::n16)),
    rgb!(f32, |rgba: [u16; 4]| rgba.to_rgb().map(s16::uf32)),
]);

#[inline(always)]
fn unpack_rgba1010102(rgba: u32) -> (u16, u16, u16, u8) {
    let r10 = rgba & 0x3FF;
    let g10 = (rgba >> 10) & 0x3FF;
    let b10 = (rgba >> 20) & 0x3FF;
    let a2 = (rgba >> 30) & 0x3;
    (r10 as u16, g10 as u16, b10 as u16, a2 as u8)
}
pub(crate) const R10G10B10A2_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(u16, |[rgba]: [u32; 1]| {
        let (r, g, b, a) = unpack_rgba1010102(rgba);
        [n10::n16(r), n10::n16(g), n10::n16(b), n2::n16(a)]
    }),
    rgba!(u8, |[rgba]: [u32; 1]| {
        let (r, g, b, a) = unpack_rgba1010102(rgba);
        [n10::n8(r), n10::n8(g), n10::n8(b), n2::n8(a)]
    }),
    rgba!(f32, |[rgba]: [u32; 1]| {
        let (r, g, b, a) = unpack_rgba1010102(rgba);
        [n10::f32(r), n10::f32(g), n10::f32(b), n2::f32(a)]
    }),
]);

#[inline(always)]
fn unpack_rgb111110f(rgb: u32) -> [f32; 3] {
    let r = f11_to_f32((rgb & 0x7FF) as u16);
    let g = f11_to_f32(((rgb >> 11) & 0x7FF) as u16);
    let b = f10_to_f32(((rgb >> 22) & 0x3FF) as u16);
    [r, g, b]
}
pub(crate) const R11G11B10_FLOAT: DecoderSet = DecoderSet::new(&[
    rgb!(f32, |[rgb]: [u32; 1]| unpack_rgb111110f(rgb)),
    rgb!(u16, |[rgb]: [u32; 1]| unpack_rgb111110f(rgb).map(fp::n16)),
    rgb!(u8, |[rgb]: [u32; 1]| unpack_rgb111110f(rgb).map(fp::n8)),
    rgba!(f32, |[rgb]: [u32; 1]| unpack_rgb111110f(rgb).to_rgba()),
    rgba!(u8, |[rgb]: [u32; 1]| unpack_rgb111110f(rgb)
        .map(fp::n8)
        .to_rgba()),
    rgba!(u16, |[rgb]: [u32; 1]| unpack_rgb111110f(rgb)
        .map(fp::n16)
        .to_rgba()),
]);

#[inline(always)]
fn unpack_rgb9995f(rgb: u32) -> [f32; 3] {
    let r_mant = rgb & 0x1FF;
    let g_mant = (rgb >> 9) & 0x1FF;
    let b_mant = (rgb >> 18) & 0x1FF;
    let exp = (rgb >> 27) & 0x1F;

    fn to_f32(e5: u32, m9: u32) -> f32 {
        // based on f16_to_f32
        if e5 == 0 {
            // denorm
            m9 as f32 * 2.0_f32.powi(-23)
        } else if e5 != 31 {
            m9 as f32 * 2.0_f32.powi(e5 as i32 - 24)
        } else if m9 == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    }

    let r = to_f32(exp, r_mant);
    let g = to_f32(exp, g_mant);
    let b = to_f32(exp, b_mant);

    [r, g, b]
}
pub(crate) const R9G9B9E5_SHAREDEXP: DecoderSet = DecoderSet::new(&[
    rgb!(f32, |[rgb]: [u32; 1]| unpack_rgb9995f(rgb)),
    rgb!(u16, |[rgb]: [u32; 1]| unpack_rgb9995f(rgb).map(fp::n16)),
    rgb!(u8, |[rgb]: [u32; 1]| unpack_rgb9995f(rgb).map(fp::n8)),
    rgba!(f32, |[rgb]: [u32; 1]| unpack_rgb9995f(rgb).to_rgba()),
    rgba!(u16, |[rgb]: [u32; 1]| unpack_rgb9995f(rgb)
        .map(fp::n16)
        .to_rgba()),
    rgba!(u8, |[rgb]: [u32; 1]| unpack_rgb9995f(rgb)
        .map(fp::n8)
        .to_rgba()),
]);

pub(crate) const R16_FLOAT: DecoderSet = DecoderSet::new(&[
    gray!(f32, |r: [u16; 1]| r.map(f16_to_f32)),
    gray!(u16, |r: [u16; 1]| r.map(f16_to_f32).map(fp::n16)),
    gray!(u8, |r: [u16; 1]| r.map(f16_to_f32).map(fp::n8)),
    // TODO: Rgb and Rgba
]);

pub(crate) const R16G16_FLOAT: DecoderSet = DecoderSet::new(&[
    rgb!(f32, |[r, g]: [u16; 2]| [f16_to_f32(r), f16_to_f32(g), 0.0]),
    rgb!(u16, |[r, g]: [u16; 2]| [
        fp::n16(f16_to_f32(r)),
        fp::n16(f16_to_f32(g)),
        0
    ]),
    rgb!(u8, |[r, g]: [u16; 2]| [
        fp::n8(f16_to_f32(r)),
        fp::n8(f16_to_f32(g)),
        0
    ]),
    rgba!(f32, |[r, g]: [u16; 2]| [f16_to_f32(r), f16_to_f32(g), 0.0,]
        .to_rgba()),
    rgba!(u16, |[r, g]: [u16; 2]| [
        fp::n16(f16_to_f32(r)),
        fp::n16(f16_to_f32(g)),
        0,
    ]
    .to_rgba()),
    rgba!(u8, |[r, g]: [u16; 2]| [
        fp::n8(f16_to_f32(r)),
        fp::n8(f16_to_f32(g)),
        0,
    ]
    .to_rgba()),
]);

pub(crate) const R16G16B16A16_FLOAT: DecoderSet = DecoderSet::new(&[
    rgba!(f32, |rgba: [u16; 4]| rgba.map(f16_to_f32)),
    rgba!(u16, |rgba: [u16; 4]| rgba.map(f16_to_f32).map(fp::n16)),
    rgba!(u8, |rgba: [u16; 4]| rgba.map(f16_to_f32).map(fp::n8)),
]);

pub(crate) const R32_FLOAT: DecoderSet = DecoderSet::new(&[
    gray!(f32, |r: [f32; 1]| r).with_decode_fn(COPY_U32),
    gray!(u16, |r: [f32; 1]| r.map(fp::n16)),
    gray!(u8, |r: [f32; 1]| r.map(fp::n8)),
]);

pub(crate) const R32G32_FLOAT: DecoderSet = DecoderSet::new(&[
    rgb!(f32, |[r, g]: [f32; 2]| [r, g, 0.0]),
    rgb!(u16, |[r, g]: [f32; 2]| [fp::n16(r), fp::n16(g), 0]),
    rgb!(u8, |[r, g]: [f32; 2]| [fp::n8(r), fp::n8(g), 0]),
    rgba!(f32, |[r, g]: [f32; 2]| [r, g, 0.0].to_rgba()),
    rgba!(u16, |[r, g]: [f32; 2]| [fp::n16(r), fp::n16(g), 0]
        .to_rgba()),
    rgba!(u8, |[r, g]: [f32; 2]| [fp::n8(r), fp::n8(g), 0].to_rgba()),
]);

pub(crate) const R32G32B32_FLOAT: DecoderSet = DecoderSet::new(&[
    rgb!(f32, |rgb: [f32; 3]| rgb).with_decode_fn(COPY_U32),
    rgb!(u16, |rgb: [f32; 3]| rgb.map(fp::n16)),
    rgb!(u8, |rgb: [f32; 3]| rgb.map(fp::n8)),
    rgba!(f32, |rgb: [f32; 3]| rgb.to_rgba()),
    rgba!(u16, |rgb: [f32; 3]| rgb.map(fp::n16).to_rgba()),
    rgba!(u8, |rgb: [f32; 3]| rgb.map(fp::n8).to_rgba()),
]);

pub(crate) const R32G32B32A32_FLOAT: DecoderSet = DecoderSet::new(&[
    rgba!(f32, |rgba: [f32; 4]| rgba).with_decode_fn(COPY_U32),
    rgba!(u16, |rgb: [f32; 4]| rgb.map(fp::n16)),
    rgba!(u8, |rgb: [f32; 4]| rgb.map(fp::n8)),
]);

#[inline(always)]
fn unpack_rgba1010102_xr(rgba: u32) -> ([u16; 3], u8) {
    let r_fixed = rgba & 0x3FF;
    let g_fixed = (rgba >> 10) & 0x3FF;
    let b_fixed = (rgba >> 20) & 0x3FF;
    let a2 = (rgba >> 30) & 0x3;

    ([r_fixed as u16, g_fixed as u16, b_fixed as u16], a2 as u8)
}
pub(crate) const R10G10B10_XR_BIAS_A2_UNORM: DecoderSet = DecoderSet::new(&[
    rgba!(f32, |[rgba]: [u32; 1]| {
        let (rgb, a2) = unpack_rgba1010102_xr(rgba);
        let [r, g, b] = rgb.map(xr10::f32);
        [r, g, b, n2::f32(a2)]
    }),
    rgba!(u16, |[rgba]: [u32; 1]| {
        let (rgb, a2) = unpack_rgba1010102_xr(rgba);
        let [r, g, b] = rgb.map(xr10::n16);
        [r, g, b, n2::n16(a2)]
    }),
    rgba!(u8, |[rgba]: [u32; 1]| {
        let (rgb, a2) = unpack_rgba1010102_xr(rgba);
        let [r, g, b] = rgb.map(xr10::n8);
        [r, g, b, n2::n8(a2)]
    }),
    rgb!(f32, |[rgba]: [u32; 1]| {
        unpack_rgba1010102_xr(rgba).0.map(xr10::f32)
    }),
    rgb!(u16, |[rgba]: [u32; 1]| {
        unpack_rgba1010102_xr(rgba).0.map(xr10::n16)
    }),
    rgb!(u8, |[rgba]: [u32; 1]| {
        unpack_rgba1010102_xr(rgba).0.map(xr10::n8)
    }),
]);
