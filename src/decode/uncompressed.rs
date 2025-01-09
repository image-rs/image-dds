use super::convert::{
    snorm8_to_uf32, snorm8_to_unorm16, snorm8_to_unorm8, unorm8_to_f32, unorm8_to_unorm16, SwapRB,
    ToRgba,
};
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
    rgb!(u16, |rgb: [u8; 3]| rgb.map(unorm8_to_unorm16)),
    rgb!(f32, |rgb: [u8; 3]| rgb.map(unorm8_to_f32)),
    rgba!(u8, |rgb: [u8; 3]| rgb.to_rgba()),
    rgba!(u16, |rgb: [u8; 3]| rgb.map(unorm8_to_unorm16).to_rgba()),
    rgba!(f32, |rgb: [u8; 3]| rgb.map(unorm8_to_f32).to_rgba()),
]);

pub(crate) const B8G8R8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(u8, |bgr: [u8; 3]| bgr.swap_rb()),
    rgb!(u16, |bgr: [u8; 3]| bgr.swap_rb().map(unorm8_to_unorm16)),
    rgb!(f32, |bgr: [u8; 3]| bgr.swap_rb().map(unorm8_to_f32)),
    rgba!(u8, |bgr: [u8; 3]| bgr.swap_rb().to_rgba()),
    rgba!(u16, |bgr: [u8; 3]| bgr
        .swap_rb()
        .map(unorm8_to_unorm16)
        .to_rgba()),
    rgba!(f32, |bgr: [u8; 3]| bgr
        .swap_rb()
        .map(unorm8_to_f32)
        .to_rgba()),
]);

pub(crate) const R8G8B8A8_UNORM: DecoderSet = DecoderSet::new(&[
    Decoder::new(Rgba, U8, |r, out, _| {
        r.read_exact(out)?;
        Ok(())
    }),
    rgba!(u16, |rgba: [u8; 4]| rgba.map(unorm8_to_unorm16)),
    rgba!(f32, |rgba: [u8; 4]| rgba.map(unorm8_to_f32)),
]);

pub(crate) const R8G8B8A8_SNORM: DecoderSet = DecoderSet::new(&[
    Decoder::new(Rgba, U8, |r, out, _| {
        r.read_exact(out)?;
        out.iter_mut().for_each(|v| *v = snorm8_to_unorm8(*v));
        Ok(())
    }),
    rgba!(u16, |rgba: [u8; 4]| rgba.map(snorm8_to_unorm16)),
    rgba!(f32, |rgba: [u8; 4]| rgba.map(snorm8_to_uf32)),
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
    rgba!(u16, |bgra: [u8; 4]| bgra.swap_rb().map(unorm8_to_unorm16)),
    rgba!(f32, |bgra: [u8; 4]| bgra.swap_rb().map(unorm8_to_f32)),
]);
