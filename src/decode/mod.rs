use std::io::Read;

use crate::{Channels, DecodeError, Precision, Size, TinyEnum, TinySet};

mod bc;
mod convert;
mod read_write;
mod sub_sampled;
mod uncompressed;

pub(crate) use sub_sampled::*;
pub(crate) use uncompressed::*;

pub(crate) struct DecodeContext {
    size: Size,
}

/// This is a silly hack to make [DecodeFn] `const`-compatible on MSRV.
///
/// The issue is that `const fn`s not not allow mutable references. On older
/// Rust versions, this also included multiple references in function pointers.
/// Of course, functions pointers can't be called in `const`, so them having
/// mutable references doesn't matter, but the compiler wasn't smart enough
/// back then. It only looked at types, saw an `&mut` and rejected the code.
///
/// The "fix" is to wrap all mutable references in a struct so that compiler
/// can't see them in the type signature of the function pointer anymore. Truly
/// silly, and thankfully not necessary on never compiler versions.
pub(crate) struct Args<'a, 'b>(pub &'a mut dyn Read, pub &'b mut [u8], DecodeContext);

pub(crate) type DecodeFn = fn(args: Args) -> Result<(), DecodeError>;

pub(crate) struct Decoder {
    pub channels: Channels,
    pub precision: Precision,
    pub disabled: bool,
    decode_fn: DecodeFn,
}
impl Decoder {
    const DISABLED_FN: DecodeFn = |_| unreachable!();

    pub const fn new(channels: Channels, precision: Precision, decode_fn: DecodeFn) -> Self {
        Self {
            channels,
            precision,
            disabled: false,
            decode_fn,
        }
    }
    pub const fn new_disabled(channels: Channels, precision: Precision) -> Self {
        Self {
            channels,
            precision,
            disabled: true,
            decode_fn: Self::DISABLED_FN,
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

        (self.decode_fn)(Args(reader, output, DecodeContext { size }))
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

pub(crate) trait WithPrecision {
    const PRECISION: Precision;
}
impl WithPrecision for u8 {
    const PRECISION: Precision = Precision::U8;
}
impl WithPrecision for u16 {
    const PRECISION: Precision = Precision::U16;
}
impl WithPrecision for f32 {
    const PRECISION: Precision = Precision::F32;
}

pub(crate) struct DecoderSet {
    pub decoders: &'static [Decoder],
    pub supported_channels: TinySet<Channels>,
    pub supported_precisions: TinySet<Precision>,
}
impl DecoderSet {
    pub const fn new(decoders: &'static [Decoder]) -> Self {
        let channels = TinySet::from_raw_unchecked({
            let mut set: u8 = 0;

            let mut i = 0;
            while i < decoders.len() {
                let decoder = &decoders[i];
                set |= 1 << decoder.channels as u8;
                i += 1;
            }
            set
        });
        let precisions = TinySet::from_raw_unchecked({
            let mut set: u8 = 0;

            let mut i = 0;
            while i < decoders.len() {
                let decoder = &decoders[i];
                set |= 1 << decoder.precision as u8;
                i += 1;
            }
            set
        });

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

                let key = decoder.channels as u32 * Precision::VARIANTS.len() as u32
                    + decoder.precision as u32;
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

        // 4. u8, u16, and f32 must be supported precisions.
        {
            let mut required: u32 = 0;
            required |= 1 << Precision::U8 as u32;
            required |= 1 << Precision::U16 as u32;
            required |= 1 << Precision::F32 as u32;

            // TODO: Enable soon.
            // assert!((precision_bitset & required) == required);
        }
    }
}
