use std::io::Read;

use crate::{Channels, DecodeError, Precision, Size};

mod bc1;
mod convert;
mod uncompressed;
mod util;

pub(crate) use uncompressed::*;

pub(crate) struct DecodeContext {
    size: Size,
}

pub(crate) type DecodeFn =
    fn(reader: &mut dyn Read, output: &mut [u8], context: DecodeContext) -> Result<(), DecodeError>;

pub(crate) struct Decoder {
    pub channels: Channels,
    pub precision: Precision,
    decode_fn: DecodeFn,
}
impl Decoder {
    pub const fn new(channels: Channels, precision: Precision, decode_fn: DecodeFn) -> Self {
        Self {
            channels,
            precision,
            decode_fn,
        }
    }

    pub fn decode(
        &self,
        reader: &mut dyn Read,
        size: Size,
        output: &mut [u8],
    ) -> Result<(), DecodeError> {
        check_buffer_len(size, self.channels, self.precision, output)?;

        // never decode empty images
        if size.is_empty() {
            return Ok(());
        }

        (self.decode_fn)(reader, output, DecodeContext { size })
    }
}

/// Verifies that the buffer is exactly as long as expected.
fn check_buffer_len(
    size: Size,
    channels: Channels,
    precision: Precision,
    buf: &[u8],
) -> Result<(), DecodeError> {
    // overflow isn't possible here
    let bytes_per_pixel = channels.count() as usize * precision.size() as usize;
    // saturate to usize::MAX on overflow
    let required_bytes = usize::saturating_mul(size.width as usize, size.height as usize)
        .saturating_mul(bytes_per_pixel);

    if buf.len() != required_bytes {
        Err(DecodeError::UnexpectedBufferSize {
            expected: required_bytes,
            actual: buf.len(),
        })
    } else {
        Ok(())
    }
}

pub(crate) struct DecoderSet {
    pub decoders: &'static [Decoder],
    pub supported_channels: &'static [Channels],
    pub supported_precisions: &'static [Precision],
}
impl DecoderSet {
    pub const fn new(
        channels: &'static [Channels],
        precisions: &'static [Precision],
        decoders: &'static [Decoder],
    ) -> Self {
        let value = Self {
            decoders,
            supported_channels: channels,
            supported_precisions: precisions,
        };
        value.verify();
        value
    }
    pub const fn main(&self) -> &'static Decoder {
        &self.decoders[0]
    }

    pub const fn verify(&self) {
        // 1. The list must be non-empty.
        assert!(!self.decoders.is_empty());

        // 2. No color channel-precision combination may be repeated.
        {
            let mut bitset: u32 = 0;
            let mut i = 0;
            while i < self.decoders.len() {
                let decoder = &self.decoders[i];

                let key = decoder.channels as u32 * Precision::VARIANTS + decoder.precision as u32;
                assert!(key < 32);

                let bit_mask = 1 << key;
                if bitset & bit_mask != 0 {
                    panic!("Repeated color channel-precision combination");
                }
                bitset |= bit_mask;

                i += 1;
            }
        }

        // 3. Color channel-precision combination must be exhaustive.
        let mut channels_bitset: u32 = 0;
        let mut precision_bitset: u32 = 0;
        {
            let mut i = 0;
            while i < self.decoders.len() {
                let decoder = &self.decoders[i];

                channels_bitset |= 1 << decoder.channels as u32;
                precision_bitset |= 1 << decoder.precision as u32;

                i += 1;
            }

            let channels_count = channels_bitset.count_ones();
            let precision_count = precision_bitset.count_ones();
            // the expected number of decoders IF all combinations are present
            let expected = channels_count * precision_count;
            if self.decoders.len() != expected as usize {
                panic!("Missing color channel-precision combination");
            }
        }

        // Supported channels must match decoder channels
        {
            let mut supported_bitset: u32 = 0;
            let mut i = 0;
            while i < self.supported_channels.len() {
                let color = self.supported_channels[i] as u32;
                supported_bitset |= 1 << color;
                i += 1;
            }

            if supported_bitset != channels_bitset {
                panic!("Supported channels do not match decoder channels");
            }
            if supported_bitset.count_ones() != self.supported_channels.len() as u32 {
                panic!("Supported channels should contain no duplicates");
            }
        }

        // Supported precisions must match decoder precisions
        {
            let mut supported_bitset: u32 = 0;
            let mut i = 0;
            while i < self.supported_precisions.len() {
                let precision = self.supported_precisions[i] as u32;
                supported_bitset |= 1 << precision;
                i += 1;
            }

            if supported_bitset != precision_bitset {
                panic!("Supported precisions do not match decoder precisions");
            }
            if supported_bitset.count_ones() != self.supported_precisions.len() as u32 {
                panic!("Supported precisions should contain no duplicates");
            }
        }
    }
}
