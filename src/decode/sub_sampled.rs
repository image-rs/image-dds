use crate::Channels::*;

use super::convert::{n8, ToRgba};
use super::read_write::for_each_pair;
use super::{Decoder, DecoderSet, Io, WithPrecision};

// helpers

macro_rules! rgb {
    ($out:ty, $f1:expr, $f2:expr) => {
        Decoder::new(
            Rgb,
            <$out as WithPrecision>::PRECISION,
            |Io(r, out), context| {
                let f1 = $f1;
                let f2 = $f2;
                for_each_pair(
                    r,
                    out,
                    context.size,
                    |pixel| -> [$out; 3] { f1(pixel) },
                    |pixel| -> [$out; 3] { f2(pixel) },
                )
            },
        )
    };
}
macro_rules! rgba {
    ($out:ty, $f1:expr, $f2:expr) => {
        Decoder::new(
            Rgba,
            <$out as WithPrecision>::PRECISION,
            |Io(r, out), context| {
                let f1 = $f1;
                let f2 = $f2;
                for_each_pair(
                    r,
                    out,
                    context.size,
                    |pixel| -> [$out; 4] { f1(pixel) },
                    |pixel| -> [$out; 4] { f2(pixel) },
                )
            },
        )
    };
}

// decoders

pub(crate) const R8G8_B8G8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(
        u8,
        |[r, g1, b, _g2]: [u8; 4]| [r, g1, b],
        |[r, _g1, b, g2]: [u8; 4]| [r, g2, b]
    ),
    rgb!(
        u16,
        |[r, g1, b, _g2]: [u8; 4]| [r, g1, b].map(n8::n16),
        |[r, _g1, b, g2]: [u8; 4]| [r, g2, b].map(n8::n16)
    ),
    rgb!(
        f32,
        |[r, g1, b, _g2]: [u8; 4]| [r, g1, b].map(n8::f32),
        |[r, _g1, b, g2]: [u8; 4]| [r, g2, b].map(n8::f32)
    ),
    rgba!(
        u8,
        |[r, g1, b, _g2]: [u8; 4]| [r, g1, b].to_rgba(),
        |[r, _g1, b, g2]: [u8; 4]| [r, g2, b].to_rgba()
    ),
    rgba!(
        u16,
        |[r, g1, b, _g2]: [u8; 4]| [r, g1, b].map(n8::n16).to_rgba(),
        |[r, _g1, b, g2]: [u8; 4]| [r, g2, b].map(n8::n16).to_rgba()
    ),
    rgba!(
        f32,
        |[r, g1, b, _g2]: [u8; 4]| [r, g1, b].map(n8::f32).to_rgba(),
        |[r, _g1, b, g2]: [u8; 4]| [r, g2, b].map(n8::f32).to_rgba()
    ),
]);

pub(crate) const G8R8_G8B8_UNORM: DecoderSet = DecoderSet::new(&[
    rgb!(
        u8,
        |[g1, r, _g2, b]: [u8; 4]| [r, g1, b],
        |[_g1, r, g2, b]: [u8; 4]| [r, g2, b]
    ),
    rgb!(
        u16,
        |[g1, r, _g2, b]: [u8; 4]| [r, g1, b].map(n8::n16),
        |[_g1, r, g2, b]: [u8; 4]| [r, g2, b].map(n8::n16)
    ),
    rgb!(
        f32,
        |[g1, r, _g2, b]: [u8; 4]| [r, g1, b].map(n8::f32),
        |[_g1, r, g2, b]: [u8; 4]| [r, g2, b].map(n8::f32)
    ),
    rgba!(
        u8,
        |[g1, r, _g2, b]: [u8; 4]| [r, g1, b].to_rgba(),
        |[_g1, r, g2, b]: [u8; 4]| [r, g2, b].to_rgba()
    ),
    rgba!(
        u16,
        |[g1, r, _g2, b]: [u8; 4]| [r, g1, b].map(n8::n16).to_rgba(),
        |[_g1, r, g2, b]: [u8; 4]| [r, g2, b].map(n8::n16).to_rgba()
    ),
    rgba!(
        f32,
        |[g1, r, _g2, b]: [u8; 4]| [r, g1, b].map(n8::f32).to_rgba(),
        |[_g1, r, g2, b]: [u8; 4]| [r, g2, b].map(n8::f32).to_rgba()
    ),
]);
