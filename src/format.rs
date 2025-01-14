use std::io::Read;

use crate::{
    cast, detect, util::div_ceil, DecodeError, DxgiFormat, FourCC, Header, TinyEnum, TinySet,
};

/// The number and semantics of the color channels in a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channels {
    /// The image only contains a single (color) channel.
    ///
    /// This (color) channels may be luminosity or one of the RGB channels (typically R).
    Grayscale,
    /// The image contains only alpha values.
    Alpha,
    /// The image contains RGB values.
    Rgb,
    /// The image contains RGBA values.
    Rgba,
}
impl Channels {
    /// Returns the number of channels.
    pub const fn count(&self) -> u8 {
        match self {
            Self::Grayscale | Self::Alpha => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}
impl TinyEnum for Channels {
    const VARIANTS: &'static [Self] = &[Self::Grayscale, Self::Alpha, Self::Rgb, Self::Rgba];

    fn bit_mask(self) -> u8 {
        1 << self as u8
    }
}

/// The precision/bit depth of the values in a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 8-bit unsigned integer.
    ///
    /// This represents normalized values in the range `[0, 255]`.
    U8,
    /// 16-bit unsigned integer.
    ///
    /// This represents normalized values in the range `[0, 65535]`.
    U16,
    /// 32-bit floating point.
    ///
    /// Values **might not** be normalized to the range `[0, 1]`.
    F32,
}
impl Precision {
    /// Returns the size of a single value of this precision in bytes.
    pub const fn size(&self) -> u8 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}
impl TinyEnum for Precision {
    const VARIANTS: &'static [Self] = &[Self::U8, Self::U16, Self::F32];

    fn bit_mask(self) -> u8 {
        1 << self as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum SupportedFormat {
    // uncompressed formats
    R8G8B8_UNORM,
    B8G8R8_UNORM,
    R8G8B8A8_UNORM,
    R8G8B8A8_SNORM,
    B8G8R8A8_UNORM,
    B8G8R8X8_UNORM,
    B5G6R5_UNORM,
    B5G5R5A1_UNORM,
    B4G4R4A4_UNORM,
    R8_SNORM,
    R8_UNORM,
    R8G8_UNORM,
    R8G8_SNORM,
    A8_UNORM,
    R16_UNORM,
    R16_SNORM,
    R16G16_UNORM,
    R16G16_SNORM,
    R16G16B16A16_UNORM,
    R16G16B16A16_SNORM,
    R10G10B10A2_UNORM,
    R11G11B10_FLOAT,
    R9G9B9E5_SHAREDEXP,
    R16_FLOAT,
    R16G16_FLOAT,
    R16G16B16A16_FLOAT,
    R32_FLOAT,
    R32G32_FLOAT,
    R32G32B32_FLOAT,
    R32G32B32A32_FLOAT,
    R10G10B10_XR_BIAS_A2_UNORM,

    // sub-sampled formats
    R8G8_B8G8_UNORM,
    G8R8_G8B8_UNORM,

    // block compression formats
    BC1_UNORM,
    BC2_UNORM,
    BC3_UNORM,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC6H_UF16,
    BC6H_SF16,
    BC7_UNORM,
}
impl SupportedFormat {
    /// Returns the format of the surfaces from a DDS header.
    pub fn from_header(header: &Header) -> Result<SupportedFormat, DecodeError> {
        if let Some(dx10_header) = &header.dxt10 {
            // decide based on DXGI format
            detect::dxgi_format_to_supported(dx10_header.dxgi_format)
                .ok_or(DecodeError::UnsupportedDxgiFormat(dx10_header.dxgi_format))
        } else if let Some(four_cc) = header.pixel_format.four_cc {
            // decide based on FourCC
            detect::four_cc_to_supported(four_cc).ok_or(DecodeError::UnsupportedFourCC(four_cc))
        } else {
            // decide based on PixelFormat
            detect::pixel_format_to_supported(&header.pixel_format)
                .ok_or(DecodeError::UnsupportedPixelFormat)
        }
    }
    /// Returns the format of a surface from a DXGI format.
    ///
    /// `None` if the DXGI format is not supported for decoding.
    pub const fn from_dxgi(dxgi: DxgiFormat) -> Option<SupportedFormat> {
        detect::dxgi_format_to_supported(dxgi)
    }
    /// Returns the format of a surface from a FourCC code.
    ///
    /// `None` if the FourCC code is not supported for decoding.
    pub const fn from_four_cc(four_cc: FourCC) -> Option<SupportedFormat> {
        detect::four_cc_to_supported(four_cc)
    }

    /// The number and type of (color) channels in the surface.
    ///
    /// If the channels of a format cannot be accurately described by
    /// [`Channels`], the next larger type is used. For example, a format with
    /// only R and G channels will be described as [`Channels::Rgb`].
    pub const fn channels(&self) -> Channels {
        decoders::get_decoders(*self).main().channels
    }
    /// The precision/bit depth closest to the values in the surface.
    ///
    /// DDS supports formats with various precisions and ranges, and not all of
    /// them can be represented *exactly* by the `Precision` enum. The closest
    /// precision is chosen based on the format's range and encoded bit depth.
    /// It is typically larger than the encoded bit depth.
    ///
    /// E.g. the format `B5G6R5_UNORM` is a 5/6-bit per channel format and the
    /// closest precision is `U8`. While `U8` can closely approximate all
    /// `B5G6R5_UNORM` values, it is not exact. E.g. a 5-bit UNORM value of 11
    /// is 90.48 as an 8-bit UNORM value exactly but will be rounded to 90.
    pub const fn precision(&self) -> Precision {
        decoders::get_decoders(*self).main().precision
    }

    /// A set of all channels this formats supports for decoding.
    ///
    /// This list is guaranteed to be without duplicates and to contain
    /// `self.channels()`.
    pub const fn supported_channels(&self) -> TinySet<Channels> {
        decoders::get_decoders(*self).supported_channels
    }
    /// A set of all precisions this formats supports for decoding.
    ///
    /// This list is guaranteed to be without duplicates and to contain
    /// `self.precision()`.
    pub const fn supported_precisions(&self) -> TinySet<Precision> {
        decoders::get_decoders(*self).supported_precisions
    }

    pub fn decode(
        &self,
        reader: &mut dyn Read,
        size: Size,
        channels: Channels,
        precision: Precision,
        output: &mut [u8],
    ) -> Result<(), DecodeError> {
        let decoders = decoders::get_decoders(*self).decoders;
        let found = decoders
            .iter()
            .find(|d| d.channels == channels && d.precision == precision);

        if let Some(decoder) = found {
            if !decoder.disabled {
                return decoder.decode(reader, size, output);
            }
        }

        Err(DecodeError::UnsupportedChannelsPrecision {
            format: *self,
            channels,
            precision,
            missing_feature: found.is_some(),
        })
    }

    /// A convenience method to decode with [`Precision::U8`].
    ///
    /// See [`Self::decode`] for more details.
    pub fn decode_u8(
        &self,
        reader: &mut dyn Read,
        size: Size,
        color_type: Channels,
        output: &mut [u8],
    ) -> Result<(), DecodeError> {
        self.decode(reader, size, color_type, Precision::U8, output)
    }
    /// A convenience method to decode with [`Precision::U16`].
    ///
    /// See [`Self::decode`] for more details.
    pub fn decode_u16(
        &self,
        reader: &mut dyn Read,
        size: Size,
        color_type: Channels,
        output: &mut [u16],
    ) -> Result<(), DecodeError> {
        self.decode(
            reader,
            size,
            color_type,
            Precision::U16,
            cast::as_bytes_mut(output),
        )
    }
    /// A convenience method to decode with [`Precision::F32`].
    ///
    /// See [`Self::decode`] for more details.
    pub fn decode_f32(
        &self,
        reader: &mut dyn Read,
        size: Size,
        color_type: Channels,
        output: &mut [f32],
    ) -> Result<(), DecodeError> {
        self.decode(
            reader,
            size,
            color_type,
            Precision::F32,
            cast::as_bytes_mut(output),
        )
    }

    /// Returns the number of bytes required to store a surface of the given dimensions.
    ///
    /// If the number of bytes overflows a `usize`, `None` is returned.
    pub(crate) fn get_surface_bytes(&self, size: Size) -> Option<u64> {
        match self {
            // 1 bytes per pixel
            Self::R8_UNORM | Self::R8_SNORM | Self::A8_UNORM => Some(size.pixels()),
            // 2 bytes per pixel
            Self::B5G6R5_UNORM
            | Self::B5G5R5A1_UNORM
            | Self::B4G4R4A4_UNORM
            | Self::R8G8_UNORM
            | Self::R8G8_SNORM
            | Self::R16_UNORM
            | Self::R16_SNORM
            | Self::R16_FLOAT => size.pixels().checked_mul(2),
            // 3 bytes per pixel
            Self::R8G8B8_UNORM | Self::B8G8R8_UNORM => size.pixels().checked_mul(3),
            // 4 bytes per pixel
            Self::R8G8B8A8_UNORM
            | Self::R8G8B8A8_SNORM
            | Self::B8G8R8A8_UNORM
            | Self::B8G8R8X8_UNORM
            | Self::R16G16_UNORM
            | Self::R16G16_SNORM
            | Self::R10G10B10A2_UNORM
            | Self::R11G11B10_FLOAT
            | Self::R9G9B9E5_SHAREDEXP
            | Self::R16G16_FLOAT
            | Self::R32_FLOAT
            | Self::R10G10B10_XR_BIAS_A2_UNORM => size.pixels().checked_mul(4),
            // 8 bytes per pixel
            Self::R16G16B16A16_UNORM
            | Self::R16G16B16A16_SNORM
            | Self::R16G16B16A16_FLOAT
            | Self::R32G32_FLOAT => size.pixels().checked_mul(8),
            // 12 bytes per pixel
            Self::R32G32B32_FLOAT => size.pixels().checked_mul(12),
            // 16 bytes per pixel
            Self::R32G32B32A32_FLOAT => size.pixels().checked_mul(16),

            // sub-sampled formats
            Self::R8G8_B8G8_UNORM | Self::G8R8_G8B8_UNORM => {
                // 4 bytes per one 2x1 block
                let blocks_x = div_ceil(size.width, 2);
                let blocks_y = size.height;
                let blocks = u64::checked_mul(blocks_x as u64, blocks_y as u64)?;
                blocks.checked_mul(4)
            }

            // block compression formats
            Self::BC1_UNORM | Self::BC4_UNORM | Self::BC4_SNORM => {
                // 8 bytes per one 4x4 block
                let blocks_x = div_ceil(size.width, 4);
                let blocks_y = div_ceil(size.height, 4);
                let blocks = u64::checked_mul(blocks_x as u64, blocks_y as u64)?;
                blocks.checked_mul(8)
            }
            Self::BC2_UNORM
            | Self::BC3_UNORM
            | Self::BC5_UNORM
            | Self::BC5_SNORM
            | Self::BC6H_SF16
            | Self::BC6H_UF16
            | Self::BC7_UNORM => {
                // 16 bytes per one 4x4 block
                let blocks_x = div_ceil(size.width, 4);
                let blocks_y = div_ceil(size.height, 4);
                let blocks = u64::checked_mul(blocks_x as u64, blocks_y as u64)?;
                blocks.checked_mul(16)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}
impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
    pub const fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}
impl From<(u32, u32)> for Size {
    fn from((width, height): (u32, u32)) -> Self {
        Self { width, height }
    }
}

mod decoders {

    use crate::decode::{self, DecodeFn, Decoder, DecoderSet};

    use super::{Channels, Precision, SupportedFormat};

    const noop_decode: DecodeFn = |_| Ok(());

    pub(crate) const fn get_decoders(format: SupportedFormat) -> DecoderSet {
        use Channels::*;
        use Precision::*;

        /// A helper macro to make it easier to define a const array of decoders.
        macro_rules! decoders {
            ($c:ident, $p:ident, $d:expr) => {{
                const DECODER: Decoder = Decoder::new($c, $p, $d);
                const INFO: DecoderSet = DecoderSet::new(&[DECODER]);
                INFO
            }};
        }

        match format {
            // uncompressed formats
            SupportedFormat::R8G8B8_UNORM => decode::R8G8B8_UNORM,
            SupportedFormat::B8G8R8_UNORM => decode::B8G8R8_UNORM,
            SupportedFormat::R8G8B8A8_UNORM => decode::R8G8B8A8_UNORM,
            SupportedFormat::R8G8B8A8_SNORM => decode::R8G8B8A8_SNORM,
            SupportedFormat::B8G8R8A8_UNORM => decode::B8G8R8A8_UNORM,
            SupportedFormat::B8G8R8X8_UNORM => decode::B8G8R8X8_UNORM,
            SupportedFormat::B5G6R5_UNORM => decode::B5G6R5_UNORM,
            SupportedFormat::B5G5R5A1_UNORM => decode::B5G5R5A1_UNORM,
            SupportedFormat::B4G4R4A4_UNORM => decode::B4G4R4A4_UNORM,
            SupportedFormat::R8_SNORM => decode::R8_SNORM,
            SupportedFormat::R8_UNORM => decode::R8_UNORM,
            SupportedFormat::R8G8_UNORM => decode::R8G8_UNORM,
            SupportedFormat::R8G8_SNORM => decode::R8G8_SNORM,
            SupportedFormat::A8_UNORM => decode::A8_UNORM,
            SupportedFormat::R16_UNORM => decode::R16_UNORM,
            SupportedFormat::R16_SNORM => decode::R16_SNORM,
            SupportedFormat::R16G16_UNORM => decode::R16G16_UNORM,
            SupportedFormat::R16G16_SNORM => decode::R16G16_SNORM,
            SupportedFormat::R16G16B16A16_UNORM => decode::R16G16B16A16_UNORM,
            SupportedFormat::R16G16B16A16_SNORM => decode::R16G16B16A16_SNORM,
            SupportedFormat::R10G10B10A2_UNORM => decode::R10G10B10A2_UNORM,
            SupportedFormat::R11G11B10_FLOAT => decode::R11G11B10_FLOAT,
            SupportedFormat::R9G9B9E5_SHAREDEXP => decode::R9G9B9E5_SHAREDEXP,
            SupportedFormat::R16_FLOAT => decode::R16_FLOAT,
            SupportedFormat::R16G16_FLOAT => decode::R16G16_FLOAT,
            SupportedFormat::R16G16B16A16_FLOAT => decode::R16G16B16A16_FLOAT,
            SupportedFormat::R32_FLOAT => decode::R32_FLOAT,
            SupportedFormat::R32G32_FLOAT => decode::R32G32_FLOAT,
            SupportedFormat::R32G32B32_FLOAT => decode::R32G32B32_FLOAT,
            SupportedFormat::R32G32B32A32_FLOAT => decode::R32G32B32A32_FLOAT,
            SupportedFormat::R10G10B10_XR_BIAS_A2_UNORM => decode::R10G10B10_XR_BIAS_A2_UNORM,

            // sub-sampled formats
            SupportedFormat::R8G8_B8G8_UNORM => decode::R8G8_B8G8_UNORM,
            SupportedFormat::G8R8_G8B8_UNORM => decode::G8R8_G8B8_UNORM,

            // block compression formats
            SupportedFormat::BC1_UNORM => decode::BC1_UNORM,
            SupportedFormat::BC2_UNORM => decode::BC2_UNORM,
            SupportedFormat::BC3_UNORM => decode::BC3_UNORM,
            SupportedFormat::BC4_UNORM => decode::BC4_UNORM,
            SupportedFormat::BC4_SNORM => decode::BC4_SNORM,
            SupportedFormat::BC5_UNORM => decode::BC5_UNORM,
            SupportedFormat::BC5_SNORM => decode::BC5_SNORM,
            SupportedFormat::BC6H_UF16 => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::BC6H_SF16 => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::BC7_UNORM => decoders!(Rgb, U8, noop_decode),
        }
    }
}
