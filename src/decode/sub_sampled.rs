use crate::{Channels::*, Precision, Precision::*};

use super::{Decoder, DecoderSet};

// helpers

// decoders

pub(crate) const R8G8_B8G8_UNORM: DecoderSet =
    DecoderSet::new(&[Decoder::new(Rgb, U8, |r, out, context| todo!())]);

pub(crate) const G8R8_G8B8_UNORM: DecoderSet =
    DecoderSet::new(&[Decoder::new(Rgb, U8, |r, out, context| todo!())]);
