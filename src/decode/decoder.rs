use std::io::{Read, Seek};
use std::mem::size_of;

use crate::{
    decode::read_write::for_each_pixel_rect_untyped, Channels, ColorFormat, DecodeError, Precision,
    Rect, Size,
};

use super::{
    adapt::{adapt, UncompressedAdapter},
    read_write::{for_each_pixel_untyped, PixelSize, ProcessPixelsFn},
};

pub(crate) type DecodeFn = fn(args: Args) -> Result<(), DecodeError>;
pub(crate) type DecodeRectFn = fn(args: RArgs) -> Result<(), DecodeError>;

pub(crate) struct DecodeContext {
    pub color: ColorFormat,
    pub size: Size,
}

pub(crate) trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

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
pub(crate) struct Args<'a, 'b>(pub &'a mut dyn Read, pub &'b mut [u8], pub DecodeContext);
impl<'a, 'b> Args<'a, 'b> {
    pub fn new(
        reader: &'a mut dyn Read,
        output: &'b mut [u8],
        context: DecodeContext,
    ) -> Result<Self, DecodeError> {
        let required_bytes = context
            .size
            .pixels()
            .saturating_mul(context.color.bytes_per_pixel() as u64)
            .try_into()
            .unwrap_or(usize::MAX);

        if output.len() != required_bytes {
            return Err(DecodeError::UnexpectedBufferSize {
                expected: required_bytes,
            });
        }

        Ok(Self(reader, output, context))
    }
}

pub(crate) struct RArgs<'a, 'b>(
    pub &'a mut dyn ReadSeek,
    pub &'b mut [u8],
    pub usize,
    pub Rect,
    pub DecodeContext,
);
impl<'a, 'b> RArgs<'a, 'b> {
    pub fn new(
        reader: &'a mut dyn ReadSeek,
        output: &'b mut [u8],
        row_pitch: usize,
        rect: Rect,
        context: DecodeContext,
    ) -> Result<Self, DecodeError> {
        // Check that the rect is within the bounds of the image.
        if !rect.is_within_bounds(context.size) {
            return Err(DecodeError::RectOutOfBounds);
        }

        // Check row pitch
        let min_row_pitch = usize::saturating_mul(
            rect.width as usize,
            context.color.bytes_per_pixel() as usize,
        );
        if row_pitch < min_row_pitch {
            return Err(DecodeError::RowPitchTooSmall {
                required_minimum: min_row_pitch,
            });
        }

        // Check that the buffer is long enough
        // saturate to usize::MAX on overflow
        let required_bytes = usize::saturating_mul(row_pitch, rect.height as usize);
        if output.len() < required_bytes {
            return Err(DecodeError::RectBufferTooSmall {
                required_minimum: required_bytes,
            });
        }

        Ok(Self(reader, output, row_pitch, rect, context))
    }
}

/// A macro to iterate over an array in a const context.
macro_rules! const_for {
    ($var:ident, $array:expr, $block:expr) => {{
        let mut i = 0;
        while i < $array.len() {
            let $var = &$array[i];
            {
                $block
            }
            i += 1;
        }
    }};
}

/// Contains decode functions directly. These functions can be used as is.
#[derive(Clone)]
pub(crate) struct DirectDecoder {
    color: ColorFormat,
    decode_fn: DecodeFn,
    decode_rect_fn: DecodeRectFn,
}
impl DirectDecoder {
    pub const fn new(
        color: ColorFormat,
        decode_fn: DecodeFn,
        decode_rect_fn: DecodeRectFn,
    ) -> Self {
        Self {
            color,
            decode_fn,
            decode_rect_fn,
        }
    }
}

/// A decoder for uncompressed pixel formats. This contains only a single
/// [`ProcessPixelsFn`] that can be used for both full images and rects.
#[derive(Clone, Copy)]
pub(crate) struct UncompressedDecoder {
    process_fn: ProcessPixelsFn,
    color: ColorFormat,
    /// The size in bytes of an encoded pixel
    encoded_pixel_size: u8,
}
impl UncompressedDecoder {
    pub const fn new<InPixel, OutPixel>(color: ColorFormat, process_fn: ProcessPixelsFn) -> Self {
        assert!(size_of::<OutPixel>() == color.bytes_per_pixel() as usize);

        Self {
            process_fn,
            color,
            encoded_pixel_size: size_of::<InPixel>() as u8,
        }
    }

    fn fn_for(&self, color: ColorFormat) -> (impl Fn(&[u8], &mut [u8]), PixelSize) {
        assert!(self.color.precision == color.precision);

        let from_channels = self.color.channels;
        let to_channels = color.channels;
        let precision = color.precision;
        let process_fn = self.process_fn;

        (
            move |encoded, decoded| {
                adapt(
                    UncompressedAdapter {
                        encoded,
                        decoded,
                        process_fn,
                    },
                    from_channels,
                    to_channels,
                    precision,
                );
            },
            PixelSize {
                encoded_size: self.encoded_pixel_size,
                decoded_size: color.bytes_per_pixel(),
            },
        )
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BitSet(u8);
impl BitSet {
    pub const ALL: Self = Self(0xFF);
    pub const EMPTY: Self = Self(0);

    pub const fn contains(&self, value: u8) -> bool {
        debug_assert!(value < 8);
        (self.0 & (1 << value)) != 0
    }
    pub const fn contains_channels(&self, channels: Channels) -> bool {
        self.contains(channels as u8)
    }
    pub const fn contains_precision(&self, precision: Precision) -> bool {
        self.contains(precision as u8)
    }

    pub const fn with(&self, value: u8) -> BitSet {
        debug_assert!(value < 8);
        BitSet(self.0 | (1 << value))
    }
    pub const fn with_channels(&self, channels: Channels) -> BitSet {
        self.with(channels as u8)
    }
    pub const fn with_precision(&self, precision: Precision) -> BitSet {
        self.with(precision as u8)
    }
}

struct DirectDecoderSet {
    decoders: &'static [DirectDecoder],
    color: ColorFormat,
    supported_channels: BitSet,
    supported_precisions: BitSet,
}
impl DirectDecoderSet {
    const fn new(decoders: &'static [DirectDecoder]) -> Self {
        assert!(!decoders.is_empty());

        let mut channels = BitSet::EMPTY;
        let mut precisions = BitSet::EMPTY;
        const_for!(decoder, decoders, {
            channels = channels.with_channels(decoder.color.channels);
            precisions = precisions.with_precision(decoder.color.precision);
        });

        let value = Self {
            decoders,
            color: decoders[0].color,
            supported_channels: channels,
            supported_precisions: precisions,
        };

        #[cfg(debug_assertions)]
        value.verify();

        value
    }
    #[cfg(debug_assertions)]
    const fn verify(&self) {
        // 1. The list must be non-empty.
        assert!(!self.decoders.is_empty());

        // 2. No color channel-precision combination may be repeated.
        {
            let mut bitset: u32 = 0;
            const_for!(decoder, self.decoders, {
                let bit_mask = 1 << decoder.color.key();
                if bitset & bit_mask != 0 {
                    panic!("Repeated color channel-precision combination");
                }
                bitset |= bit_mask;
            });
        }

        // 3. Color channel-precision combination must be exhaustive.
        {
            let channels_count = self.supported_channels.0.count_ones();
            let precision_count = self.supported_precisions.0.count_ones();
            // the expected number of decoders IF all combinations are present
            let expected = channels_count * precision_count;
            if self.decoders.len() != expected as usize {
                panic!("Missing color channel-precision combination");
            }
        }

        // 4. All precisions must be supported
        {
            let precision_count = self.supported_precisions.0.count_ones();
            if precision_count != Precision::COUNT as u32 {
                panic!("All precisions must be supported");
            }
        }
    }

    fn get_decoder(&self, color: ColorFormat) -> &DirectDecoder {
        if let Some(decoder) = self.decoders.iter().find(|d| d.color == color) {
            return decoder;
        }
        unreachable!("Calling a decoder set with an unsupported color format is invalid and a bug in the implementation");
    }

    fn decode(&self, color: ColorFormat, args: Args) -> Result<(), DecodeError> {
        let decoder = self.get_decoder(color);
        (decoder.decode_fn)(args)
    }
    fn decode_rect(&self, color: ColorFormat, args: RArgs) -> Result<(), DecodeError> {
        let decoder = self.get_decoder(color);
        (decoder.decode_rect_fn)(args)
    }
}

struct UncompressedDecoderSet {
    decoders: &'static [UncompressedDecoder],
}
impl UncompressedDecoderSet {
    const fn new(decoders: &'static [UncompressedDecoder]) -> Self {
        #[cfg(debug_assertions)]
        Self::verify(decoders);

        Self { decoders }
    }
    #[cfg(debug_assertions)]
    const fn verify(decoders: &'static [UncompressedDecoder]) {
        // 1. The list must be non-empty.
        assert!(!decoders.is_empty());

        // 2. No color channel-precision combination may be repeated.
        {
            let mut bitset: u32 = 0;
            const_for!(decoder, decoders, {
                let bit_mask = 1 << decoder.color.key();
                if bitset & bit_mask != 0 {
                    panic!("Repeated color channel-precision combination");
                }
                bitset |= bit_mask;
            });
        }

        // 3. All precisions must be present
        {
            let mut precision_bitset: u32 = 0;
            const_for!(decoder, decoders, {
                precision_bitset |= 1 << decoder.color.precision as u32;
            });

            let precision_count = precision_bitset.count_ones();
            if precision_count != Precision::COUNT as u32 {
                panic!("Missing color channel-precision combination");
            }
        }
    }

    const fn native_color(&self) -> ColorFormat {
        self.decoders[0].color
    }

    fn get_closest_process_fn(&self, color: ColorFormat) -> UncompressedDecoder {
        // try to find a color format that is efficient and retains as
        // much color information as possible
        use Channels::*;
        let channel_preference: &[Channels] = match color.channels {
            Channels::Grayscale => &[Grayscale, Rgb, Rgba],
            Channels::Alpha => &[Alpha, Rgba],
            Channels::Rgb => &[Rgb, Rgba, Grayscale],
            Channels::Rgba => &[Rgba, Rgb, Grayscale, Alpha],
        };
        for &channels in channel_preference {
            let color = ColorFormat::new(channels, color.precision);
            if let Some(process_fn) = self.decoders.iter().find(|d| d.color == color) {
                return *process_fn;
            }
        }

        // we give up. Find any with the same precision
        if let Some(process_fn) = self
            .decoders
            .iter()
            .find(|d| d.color.precision == color.precision)
        {
            return *process_fn;
        }

        unreachable!("This object is invalid, because it should have at least one process function of every precision");
    }

    fn decode(&self, color: ColorFormat, args: Args) -> Result<(), DecodeError> {
        let decoder = self.get_closest_process_fn(color);
        let (process_fn, pixel_size) = decoder.fn_for(color);
        for_each_pixel_untyped(args.0, args.1, pixel_size, process_fn)
    }
    fn decode_rect(&self, color: ColorFormat, args: RArgs) -> Result<(), DecodeError> {
        let decoder = self.get_closest_process_fn(color);
        let (process_fn, pixel_size) = decoder.fn_for(color);
        for_each_pixel_rect_untyped(
            args.0,
            args.1,
            args.2,
            args.4.size,
            args.3,
            pixel_size,
            process_fn,
        )
    }
}

enum Inner {
    List(DirectDecoderSet),
    Uncompressed(UncompressedDecoderSet),
}

struct SpecializedDecodeFn {
    decode_fn: DecodeFn,
    color: ColorFormat,
}

pub(crate) struct DecoderSet {
    decoders: Inner,
    optimized: Option<SpecializedDecodeFn>,
}
impl DecoderSet {
    pub const fn new(decoders: &'static [DirectDecoder]) -> Self {
        Self {
            decoders: Inner::List(DirectDecoderSet::new(decoders)),
            optimized: None,
        }
    }
    pub const fn new_uncompressed(decoders: &'static [UncompressedDecoder]) -> Self {
        Self {
            decoders: Inner::Uncompressed(UncompressedDecoderSet::new(decoders)),
            optimized: None,
        }
    }
    pub const fn add_specialized(
        self,
        channels: Channels,
        precision: Precision,
        decode_fn: DecodeFn,
    ) -> Self {
        assert!(self.optimized.is_none());
        Self {
            decoders: self.decoders,
            optimized: Some(SpecializedDecodeFn {
                decode_fn,
                color: ColorFormat::new(channels, precision),
            }),
        }
    }

    pub const fn native_color(&self) -> ColorFormat {
        match &self.decoders {
            Inner::List(list) => list.color,
            Inner::Uncompressed(list) => list.native_color(),
        }
    }

    pub const fn supported_channels(&self) -> BitSet {
        match &self.decoders {
            Inner::List(list) => list.supported_channels,
            Inner::Uncompressed(_) => BitSet::ALL,
        }
    }
    pub const fn supported_precisions(&self) -> BitSet {
        match &self.decoders {
            Inner::List(list) => list.supported_precisions,
            Inner::Uncompressed(_) => BitSet::ALL,
        }
    }

    pub fn decode(
        &self,
        color: ColorFormat,
        reader: &mut dyn Read,
        size: Size,
        output: &mut [u8],
    ) -> Result<(), DecodeError> {
        let args = Args::new(reader, output, DecodeContext { color, size })?;

        // never decode empty images
        if size.is_empty() {
            return Ok(());
        }

        if let Some(optimized) = &self.optimized {
            if optimized.color == color {
                // some decoder sets have specially optimized full-image decoders
                return (optimized.decode_fn)(args);
            }
        }

        match &self.decoders {
            Inner::List(list) => list.decode(color, args),
            Inner::Uncompressed(list) => list.decode(color, args),
        }
    }

    pub fn decode_rect(
        &self,
        color: ColorFormat,
        reader: &mut dyn ReadSeek,
        size: Size,
        rect: Rect,
        output: &mut [u8],
        row_pitch: usize,
    ) -> Result<(), DecodeError> {
        let args = RArgs::new(
            reader,
            output,
            row_pitch,
            rect,
            DecodeContext { color, size },
        )?;

        // never decode empty rects
        if rect.size().is_empty() {
            return Ok(());
        }

        match &self.decoders {
            Inner::List(list) => list.decode_rect(color, args),
            Inner::Uncompressed(list) => list.decode_rect(color, args),
        }
    }
}
