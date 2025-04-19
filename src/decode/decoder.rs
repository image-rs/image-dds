use std::io::{Read, Seek};
use std::mem::size_of;

use crate::{
    Channels, ColorFormat, ColorFormatSet, DecodingError, ImageViewMut, Precision, Rect, Size,
};

use super::DecodeOptions;

pub(crate) type DecodeFn = fn(args: Args) -> Result<(), DecodingError>;
pub(crate) type DecodeRectFn = fn(args: RArgs) -> Result<(), DecodingError>;

pub(crate) struct DecodeContext {
    pub color: ColorFormat,
    pub size: Size,
    pub memory_limit: usize,
}
impl DecodeContext {
    pub fn reserve_bytes(&mut self, bytes: usize) -> Result<(), DecodingError> {
        if self.memory_limit < bytes {
            return Err(DecodingError::MemoryLimitExceeded);
        }

        self.memory_limit -= bytes;
        Ok(())
    }
    pub fn alloc<T: Default + Copy>(&mut self, len: usize) -> Result<Box<[T]>, DecodingError> {
        self.reserve_bytes(len * size_of::<T>())?;
        Ok(vec![T::default(); len].into_boxed_slice())
    }
    pub fn alloc_capacity<T: Default + Copy>(
        &mut self,
        len: usize,
    ) -> Result<Vec<T>, DecodingError> {
        self.reserve_bytes(len * size_of::<T>())?;
        Ok(Vec::with_capacity(len))
    }
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
    ) -> Result<Self, DecodingError> {
        let bytes_per_pixel = context.color.bytes_per_pixel() as u64;
        assert_eq!(
            output.len() as u64,
            context.size.pixels().saturating_mul(bytes_per_pixel)
        );

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
    ) -> Result<Self, DecodingError> {
        // Check that the rect is within the bounds of the image.
        if !rect.is_within_bounds(context.size) {
            return Err(DecodingError::RectOutOfBounds);
        }

        // Check row pitch
        let min_row_pitch = if rect.size().is_empty() {
            0
        } else {
            usize::saturating_mul(
                rect.width as usize,
                context.color.bytes_per_pixel() as usize,
            )
        };
        if row_pitch < min_row_pitch {
            return Err(DecodingError::RowPitchTooSmall {
                required_minimum: min_row_pitch,
            });
        }

        // Check that the buffer is long enough
        // saturate to usize::MAX on overflow
        let required_bytes = if rect.size().is_empty() {
            0
        } else {
            usize::saturating_mul(row_pitch, (rect.height - 1) as usize)
                .saturating_add(min_row_pitch)
        };
        if output.len() < required_bytes {
            return Err(DecodingError::RectBufferTooSmall {
                required_minimum: required_bytes,
            });
        }

        Ok(Self(reader, output, row_pitch, rect, context))
    }
}

/// Contains decode functions directly. These functions can be used as is.
pub(crate) struct Decoder {
    native_color: ColorFormat,
    supported_colors: ColorFormatSet,
    decode_fn: DecodeFn,
    decode_rect_fn: DecodeRectFn,
}
impl Decoder {
    pub const fn new_with_all_channels(
        color: ColorFormat,
        decode_fn: DecodeFn,
        decode_rect_fn: DecodeRectFn,
    ) -> Self {
        Self {
            native_color: color,
            supported_colors: ColorFormatSet::from_precision(color.precision),
            decode_fn,
            decode_rect_fn,
        }
    }
}

struct SpecializedDecodeFn {
    decode_fn: DecodeFn,
    color: ColorFormat,
}

pub(crate) struct DecoderSet {
    decoders: &'static [Decoder],
    optimized: Option<SpecializedDecodeFn>,
}
impl DecoderSet {
    pub const fn new(decoders: &'static [Decoder]) -> Self {
        #[cfg(debug_assertions)]
        Self::verify(decoders);

        Self {
            decoders,
            optimized: None,
        }
    }
    #[cfg(debug_assertions)]
    const fn verify(decoders: &'static [Decoder]) {
        debug_assert!(!decoders.is_empty());

        let mut supported_colors = ColorFormatSet::EMPTY;
        let mut native_colors = ColorFormatSet::EMPTY;

        let mut i = 0;
        while i < decoders.len() {
            let decoder = &decoders[i];
            supported_colors = supported_colors.union(decoder.supported_colors);
            native_colors = native_colors.union(ColorFormatSet::from_single(decoder.native_color));
            i += 1;
        }

        debug_assert!(supported_colors.is_all(), "All colors must be supported");
        debug_assert!(
            native_colors.len() as usize == decoders.len(),
            "There should only be one decoder per native color."
        );
    }
    pub const fn add_specialized(
        self,
        channels: Channels,
        precision: Precision,
        decode_fn: DecodeFn,
    ) -> Self {
        debug_assert!(self.optimized.is_none());
        Self {
            decoders: self.decoders,
            optimized: Some(SpecializedDecodeFn {
                decode_fn,
                color: ColorFormat::new(channels, precision),
            }),
        }
    }

    pub const fn native_color(&self) -> ColorFormat {
        self.decoders[0].native_color
    }

    fn get_decoder(&self, color: ColorFormat) -> &Decoder {
        // try to find an exact match
        if let Some(decoder) = self.decoders.iter().find(|d| d.native_color == color) {
            return decoder;
        }

        // get any decoders
        self.decoders
            .iter()
            .find(|d| d.supported_colors.contains(color))
            .expect("All color formats should be supported")
    }

    pub fn decode(
        &self,
        reader: &mut dyn Read,
        mut image: ImageViewMut,
        options: &DecodeOptions,
    ) -> Result<(), DecodingError> {
        let color = image.color();
        let size = image.size();

        let args = Args::new(
            reader,
            image.data(),
            DecodeContext {
                color,
                size,
                memory_limit: options.memory_limit,
            },
        )?;

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

        let decoder = self.get_decoder(color);
        (decoder.decode_fn)(args)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_rect(
        &self,
        color: ColorFormat,
        reader: &mut dyn ReadSeek,
        size: Size,
        rect: Rect,
        output: &mut [u8],
        row_pitch: usize,
        options: &DecodeOptions,
    ) -> Result<(), DecodingError> {
        let args = RArgs::new(
            reader,
            output,
            row_pitch,
            rect,
            DecodeContext {
                color,
                size,
                memory_limit: options.memory_limit,
            },
        )?;

        // never decode empty rectangles
        if rect.size().is_empty() {
            return Ok(());
        }

        let decoder = self.get_decoder(color);
        (decoder.decode_rect_fn)(args)
    }
}
