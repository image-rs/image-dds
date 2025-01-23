use std::io::Write;

use bitflags::bitflags;

use crate::{cast, util, ColorFormat, ColorFormatSet, Precision};

use super::{Args, DecodedArgs, DitheredChannels, EncodeError, EncodeOptions, Encoder};

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Flags: u8 {
        /// Whether all U8 values will be encoded exactly, meaning no loss of
        /// precision.
        ///
        /// SNORM8 is considered exact.
        const EXACT_U8 = 0x1;
        /// Whether all U16 values will be encoded exactly, meaning no loss of
        /// precision.
        ///
        /// SNORM16 is considered exact.
        ///
        /// This flag implies `EXACT_U8`.
        const EXACT_U16 = 0x2 | Self::EXACT_U8.bits();
        /// Whether all F32 values will be encoded exactly, meaning no loss of
        /// precision.
        ///
        /// This flag implies `EXACT_U16` and `EXACT_U8`.
        const EXACT_F32 = 0x4 | Self::EXACT_U16.bits();
        /// Whether color dithering is supported.
        const DITHER_COLOR = 0x8;
        /// Whether alpha dithering is supported.
        const DITHER_ALPHA = 0x16;
        /// Whether both alpha and color dithering is supported.
        const DITHER_ALL = Self::DITHER_COLOR.bits() | Self::DITHER_ALPHA.bits();
    }
}

impl Flags {
    pub const fn exact_for(precision: Precision) -> Self {
        match precision {
            Precision::U8 => Flags::EXACT_U8,
            Precision::U16 => Flags::EXACT_U16,
            Precision::F32 => Flags::EXACT_F32,
        }
    }
}

pub(crate) struct BaseEncoder {
    pub color_formats: ColorFormatSet,
    pub flags: Flags,
    pub encode: fn(Args) -> Result<(), EncodeError>,
}
impl BaseEncoder {
    pub const fn copy(color: ColorFormat) -> Self {
        Self {
            color_formats: ColorFormatSet::from_single(color),
            flags: Flags::exact_for(color.precision),
            encode: |args| copy_directly(args),
        }
    }

    pub const fn add_flags(mut self, flags: Flags) -> Self {
        self.flags = self.flags.union(flags);
        self
    }
}
impl Encoder for BaseEncoder {
    fn supported_color_formats(&self) -> ColorFormatSet {
        self.color_formats
    }

    fn supports_dithering(&self) -> DitheredChannels {
        let color = self.flags.contains(Flags::DITHER_COLOR);
        let alpha = self.flags.contains(Flags::DITHER_ALPHA);
        match (color, alpha) {
            (true, true) => DitheredChannels::All,
            (true, false) => DitheredChannels::ColorOnly,
            (false, true) => DitheredChannels::AlphaOnly,
            (false, false) => DitheredChannels::None,
        }
    }

    fn encode(
        &self,
        data: &[u8],
        width: u32,
        color: ColorFormat,
        writer: &mut dyn Write,
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        if !self.color_formats.contains(color) {
            return Err(EncodeError::UnsupportedColorFormat(color));
        }

        (self.encode)(Args(data, width, color, writer, options.clone()))
    }
}
impl Encoder for &[BaseEncoder] {
    fn supported_color_formats(&self) -> ColorFormatSet {
        let mut set = ColorFormatSet::EMPTY;
        for encoder in *self {
            set = set.union(encoder.supported_color_formats());
        }
        set
    }

    fn supports_dithering(&self) -> DitheredChannels {
        self.iter()
            .filter_map(|e| {
                let d = e.supports_dithering();
                if d != DitheredChannels::None {
                    Some(d)
                } else {
                    None
                }
            })
            .next()
            .unwrap_or(DitheredChannels::None)
    }

    fn encode(
        &self,
        data: &[u8],
        width: u32,
        color: ColorFormat,
        writer: &mut dyn Write,
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        // Firstly, if we have an encoder that can encode the current precision
        // exactly, use it.
        let precision_flag = Flags::exact_for(color.precision);
        for encoder in *self {
            if encoder.supported_color_formats().contains(color)
                && encoder.flags.contains(precision_flag)
            {
                return encoder.encode(data, width, color, writer, options);
            }
        }

        // Secondly, search for encoders that perform the requested dithering.
        for encoder in *self {
            if encoder.supported_color_formats().contains(color)
                && encoder.supports_dithering().intersect(options.dither) != DitheredChannels::None
            {
                return encoder.encode(data, width, color, writer, options);
            }
        }

        // Lastly, just pick any encoder that can do the job.
        for encoder in *self {
            if encoder.supported_color_formats().contains(color) {
                return encoder.encode(data, width, color, writer, options);
            }
        }
        Err(EncodeError::UnsupportedColorFormat(color))
    }
}

fn copy_directly(args: Args) -> Result<(), EncodeError> {
    let DecodedArgs {
        data,
        color,
        writer,
        ..
    } = DecodedArgs::from(args)?;

    // We can always just write everything directly on LE systems
    // and when the precision is U8
    if cfg!(target_endian = "little") || color.precision == Precision::U8 {
        writer.write_all(data)?;
        return Ok(());
    }

    // We need to convert to LE, so we need to allocate a buffer
    let mut buffer = [0_u8; 4096];
    let chuck_size = buffer.len();

    for chunk in data.chunks(chuck_size) {
        debug_assert!(chunk.len() % color.precision.size() as usize == 0);
        let chunk_buffer = &mut buffer[..chunk.len()];
        chunk_buffer.copy_from_slice(chunk);
        convert_to_le(color.precision, chunk_buffer);
        writer.write_all(chunk_buffer)?;
    }

    Ok(())
}

pub(crate) fn convert_to_le(precision: Precision, buffer: &mut [u8]) {
    match precision {
        Precision::U8 => {}
        Precision::U16 => util::le_to_native_endian_16(buffer),
        Precision::F32 => util::le_to_native_endian_32(buffer),
    }
}

pub(crate) trait ToLe: Sized {
    fn to_le(buffer: &mut [Self]);
}
impl ToLe for u8 {
    fn to_le(_buffer: &mut [Self]) {}
}
impl ToLe for u16 {
    fn to_le(buffer: &mut [Self]) {
        util::le_to_native_endian_16(cast::as_bytes_mut(buffer));
    }
}
impl ToLe for u32 {
    fn to_le(buffer: &mut [Self]) {
        util::le_to_native_endian_32(cast::as_bytes_mut(buffer));
    }
}
impl ToLe for f32 {
    fn to_le(buffer: &mut [Self]) {
        util::le_to_native_endian_32(cast::as_bytes_mut(buffer));
    }
}
impl<const N: usize, T> ToLe for [T; N]
where
    T: ToLe + cast::Castable,
{
    fn to_le(buffer: &mut [Self]) {
        let flat = cast::as_flattened_mut(buffer);
        T::to_le(flat);
    }
}
