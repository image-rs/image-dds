use super::{convert::snorm8_to_unorm8, util::for_each_pixel_u8, Decoder, DecoderSet};

use crate::Channels::*;
use crate::Precision::*;

pub(crate) const R8G8B8_UNORM: DecoderSet = DecoderSet::new(
    &[Rgb],
    &[U8],
    &[Decoder::new(Rgb, U8, |r, out, _| {
        r.read_exact(out)?;
        Ok(())
    })],
);

pub(crate) const B8G8R8_UNORM: DecoderSet = DecoderSet::new(
    &[Rgb],
    &[U8],
    &[Decoder::new(Rgb, U8, |r, out, _| {
        for_each_pixel_u8(r, out, |[b, g, r]| [r, g, b])
    })],
);

pub(crate) const R8G8B8A8_UNORM: DecoderSet = DecoderSet::new(
    &[Rgba],
    &[U8],
    &[Decoder::new(Rgba, U8, |r, out, _| {
        r.read_exact(out)?;
        Ok(())
    })],
);

pub(crate) const R8G8B8A8_SNORM: DecoderSet = DecoderSet::new(
    &[Rgba],
    &[U8],
    &[Decoder::new(Rgba, U8, |r, out, _| {
        r.read_exact(out)?;
        out.iter_mut().for_each(|v| *v = snorm8_to_unorm8(*v));
        Ok(())
    })],
);

pub(crate) const B8G8R8A8_UNORM: DecoderSet = DecoderSet::new(
    &[Rgba],
    &[U8],
    &[Decoder::new(Rgba, U8, |r, out, _| {
        // read everything in BGRA order
        r.read_exact(out)?;
        // swap R and B
        for i in (0..out.len()).step_by(4) {
            out.swap(i, i + 2);
        }
        Ok(())
    })],
);
