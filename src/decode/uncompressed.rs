use super::convert::{n4, n8, s8, SwapRB, ToRgba, B5G5R5A1, B5G6R5};
use super::util::for_each_pixel;
use super::{Decoder, DecoderSet};

use crate::{Channels::*, Precision, Precision::*};

// helpers

trait WithPrecision {
    const PRECISION: Precision;
}
impl WithPrecision for u8 {
    const PRECISION: Precision = U8;
}
impl WithPrecision for u16 {
    const PRECISION: Precision = U16;
}
impl WithPrecision for f32 {
    const PRECISION: Precision = F32;
}

macro_rules! gray {
    ($out:ty, $f:expr) => {
        Decoder::new(
            Grayscale,
            <$out as WithPrecision>::PRECISION,
            |r, out, _| {
                let f = $f;
                for_each_pixel(r, out, |pixel| -> [$out; 1] { f(pixel) })
            },
        )
    };
}
macro_rules! alpha {
    ($out:ty, $f:expr) => {
        Decoder::new(Alpha, <$out as WithPrecision>::PRECISION, |r, out, _| {
            let f = $f;
            for_each_pixel(r, out, |pixel| -> [$out; 1] { f(pixel) })
        })
    };
}
macro_rules! rgb {
    ($out:ty, $f:expr) => {
        Decoder::new(Rgb, <$out as WithPrecision>::PRECISION, |r, out, _| {
            let f = $f;
            for_each_pixel(r, out, |pixel| -> [$out; 3] { f(pixel) })
        })
    };
}
macro_rules! rgba {
    ($out:ty, $f:expr) => {
        Decoder::new(Rgba, <$out as WithPrecision>::PRECISION, |r, out, _| {
            let f = $f;
            for_each_pixel(r, out, |pixel| -> [$out; 4] { f(pixel) })
        })
    };
}

pub(crate) const R8G8B8_UNORM: DecoderSet = DecoderSet::new(&[
    Decoder::new(Rgb, U8, |r, out, _| {
        r.read_exact(out)?;
        Ok(())
    }),
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
    Decoder::new(Rgba, U8, |r, out, _| {
        r.read_exact(out)?;
        Ok(())
    }),
    rgba!(u16, |rgba: [u8; 4]| rgba.map(n8::n16)),
    rgba!(f32, |rgba: [u8; 4]| rgba.map(n8::f32)),
]);

pub(crate) const R8G8B8A8_SNORM: DecoderSet = DecoderSet::new(&[
    Decoder::new(Rgba, U8, |r, out, _| {
        r.read_exact(out)?;
        out.iter_mut().for_each(|v| *v = s8::n8(*v));
        Ok(())
    }),
    rgba!(u16, |rgba: [u8; 4]| rgba.map(s8::n16)),
    rgba!(f32, |rgba: [u8; 4]| rgba.map(s8::uf32)),
]);

pub(crate) const B8G8R8A8_UNORM: DecoderSet = DecoderSet::new(&[
    Decoder::new(Rgba, U8, |r, out, _| {
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
]);

pub(crate) const R8_UNORM: DecoderSet = DecoderSet::new(&[
    Decoder::new(Grayscale, U8, |r, out, _| {
        r.read_exact(out)?;
        Ok(())
    }),
    gray!(u16, |r: [u8; 1]| r.map(n8::n16)),
    gray!(f32, |r: [u8; 1]| r.map(n8::f32)),
    // TODO: Rgb and Rgba
]);

pub(crate) const R8_SNORM: DecoderSet = DecoderSet::new(&[
    Decoder::new(Grayscale, U8, |r, out, _| {
        r.read_exact(out)?;
        out.iter_mut().for_each(|v| *v = s8::n8(*v));
        Ok(())
    }),
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
    Decoder::new(Alpha, U8, |r, out, _| {
        r.read_exact(out)?;
        Ok(())
    }),
    alpha!(u16, |a: [u8; 1]| a.map(n8::n16)),
    alpha!(f32, |a: [u8; 1]| a.map(n8::f32)),
    rgba!(u8, |[a]: [u8; 1]| [0, 0, 0, a]),
    rgba!(u16, |[a]: [u8; 1]| [0, 0, 0, n8::n16(a)]),
    rgba!(f32, |[a]: [u8; 1]| [0.0, 0.0, 0.0, n8::f32(a)]),
]);
