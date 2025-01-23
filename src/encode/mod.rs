use std::io::Write;

use crate::{
    Channels, ColorFormat, ColorFormatSet, Dx9PixelFormat, DxgiFormat, FourCC, MaskPixelFormat,
    PixelFormatFlags, Precision, Size,
};

mod bc;
mod bc4;
mod sub_sampled;
mod uncompressed;
mod write;

use bc::*;
use sub_sampled::*;
use uncompressed::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum EncodeFormat {
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
    A4B4G4R4_UNORM,
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
    AYUV,
    Y410,
    Y416,

    // sub-sampled formats
    R1_UNORM,
    R8G8_B8G8_UNORM,
    G8R8_G8B8_UNORM,
    UYVY,
    YUY2,
    Y210,
    Y216,

    // block compression formats
    BC1_UNORM,
    BC2_UNORM,
    BC2_UNORM_PREMULTIPLIED_ALPHA,
    BC3_UNORM,
    BC3_UNORM_PREMULTIPLIED_ALPHA,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC6H_UF16,
    BC6H_SF16,
    BC7_UNORM,
}
impl EncodeFormat {
    pub const fn channels(self) -> Channels {
        match self {
            EncodeFormat::R8_SNORM
            | EncodeFormat::R8_UNORM
            | EncodeFormat::R16_UNORM
            | EncodeFormat::R16_SNORM
            | EncodeFormat::R16_FLOAT
            | EncodeFormat::R32_FLOAT
            | EncodeFormat::R1_UNORM
            | EncodeFormat::BC4_UNORM
            | EncodeFormat::BC4_SNORM => Channels::Grayscale,

            EncodeFormat::A8_UNORM => Channels::Alpha,

            EncodeFormat::R8G8B8_UNORM
            | EncodeFormat::B8G8R8_UNORM
            | EncodeFormat::B8G8R8X8_UNORM
            | EncodeFormat::B5G6R5_UNORM
            | EncodeFormat::R8G8_UNORM
            | EncodeFormat::R8G8_SNORM
            | EncodeFormat::R16G16_UNORM
            | EncodeFormat::R16G16_SNORM
            | EncodeFormat::R11G11B10_FLOAT
            | EncodeFormat::R9G9B9E5_SHAREDEXP
            | EncodeFormat::R16G16_FLOAT
            | EncodeFormat::R32G32_FLOAT
            | EncodeFormat::R32G32B32_FLOAT
            | EncodeFormat::Y410
            | EncodeFormat::Y416
            | EncodeFormat::R8G8_B8G8_UNORM
            | EncodeFormat::G8R8_G8B8_UNORM
            | EncodeFormat::UYVY
            | EncodeFormat::YUY2
            | EncodeFormat::Y210
            | EncodeFormat::Y216
            | EncodeFormat::BC5_UNORM
            | EncodeFormat::BC5_SNORM
            | EncodeFormat::BC6H_UF16
            | EncodeFormat::BC6H_SF16 => Channels::Rgb,

            EncodeFormat::R8G8B8A8_UNORM
            | EncodeFormat::R8G8B8A8_SNORM
            | EncodeFormat::B8G8R8A8_UNORM
            | EncodeFormat::B5G5R5A1_UNORM
            | EncodeFormat::B4G4R4A4_UNORM
            | EncodeFormat::A4B4G4R4_UNORM
            | EncodeFormat::R16G16B16A16_UNORM
            | EncodeFormat::R16G16B16A16_SNORM
            | EncodeFormat::R10G10B10A2_UNORM
            | EncodeFormat::R16G16B16A16_FLOAT
            | EncodeFormat::R32G32B32A32_FLOAT
            | EncodeFormat::R10G10B10_XR_BIAS_A2_UNORM
            | EncodeFormat::AYUV
            | EncodeFormat::BC1_UNORM
            | EncodeFormat::BC2_UNORM
            | EncodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA
            | EncodeFormat::BC3_UNORM
            | EncodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA
            | EncodeFormat::BC7_UNORM => Channels::Rgba,
        }
    }
    pub const fn precision(self) -> Precision {
        match self {
            EncodeFormat::R8G8B8_UNORM
            | EncodeFormat::B8G8R8_UNORM
            | EncodeFormat::R8G8B8A8_UNORM
            | EncodeFormat::R8G8B8A8_SNORM
            | EncodeFormat::B8G8R8A8_UNORM
            | EncodeFormat::B8G8R8X8_UNORM
            | EncodeFormat::B5G6R5_UNORM
            | EncodeFormat::B5G5R5A1_UNORM
            | EncodeFormat::B4G4R4A4_UNORM
            | EncodeFormat::A4B4G4R4_UNORM
            | EncodeFormat::R8_SNORM
            | EncodeFormat::R8_UNORM
            | EncodeFormat::R8G8_UNORM
            | EncodeFormat::R8G8_SNORM
            | EncodeFormat::A8_UNORM
            | EncodeFormat::AYUV
            | EncodeFormat::R1_UNORM
            | EncodeFormat::R8G8_B8G8_UNORM
            | EncodeFormat::G8R8_G8B8_UNORM
            | EncodeFormat::UYVY
            | EncodeFormat::YUY2
            | EncodeFormat::BC1_UNORM
            | EncodeFormat::BC2_UNORM
            | EncodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA
            | EncodeFormat::BC3_UNORM
            | EncodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA
            | EncodeFormat::BC4_UNORM
            | EncodeFormat::BC4_SNORM
            | EncodeFormat::BC5_UNORM
            | EncodeFormat::BC5_SNORM
            | EncodeFormat::BC7_UNORM => Precision::U8,

            EncodeFormat::R16_UNORM
            | EncodeFormat::R16_SNORM
            | EncodeFormat::R16G16_UNORM
            | EncodeFormat::R16G16_SNORM
            | EncodeFormat::R16G16B16A16_UNORM
            | EncodeFormat::R16G16B16A16_SNORM
            | EncodeFormat::R10G10B10A2_UNORM
            | EncodeFormat::Y410
            | EncodeFormat::Y416
            | EncodeFormat::Y210
            | EncodeFormat::Y216 => Precision::U16,

            EncodeFormat::R16_FLOAT
            | EncodeFormat::R16G16_FLOAT
            | EncodeFormat::R16G16B16A16_FLOAT
            | EncodeFormat::BC6H_UF16
            | EncodeFormat::BC6H_SF16
            | EncodeFormat::R11G11B10_FLOAT
            | EncodeFormat::R9G9B9E5_SHAREDEXP
            | EncodeFormat::R10G10B10_XR_BIAS_A2_UNORM
            | EncodeFormat::R32_FLOAT
            | EncodeFormat::R32G32_FLOAT
            | EncodeFormat::R32G32B32_FLOAT
            | EncodeFormat::R32G32B32A32_FLOAT => Precision::F32,
        }
    }

    pub fn encode<W: Write>(
        &self,
        writer: &mut W,
        size: Size,
        color: ColorFormat,
        data: &[u8],
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        if let Some(encoder) = self.get_encoder() {
            encoder.encode(data, size.width, color, writer, options)
        } else {
            // TODO:
            Err(EncodeError::UnsupportedColorFormat(color))
        }
    }

    pub fn supports_dither(self) -> DitheredChannels {
        self.get_encoder()
            .map_or(DitheredChannels::None, |encoder| {
                encoder.supports_dithering()
            })
    }

    const fn get_encoder(self) -> Option<&'static dyn Encoder> {
        Some(match self {
            EncodeFormat::R8G8B8_UNORM => &R8G8B8_UNORM,
            EncodeFormat::B8G8R8_UNORM => &B8G8R8_UNORM,
            EncodeFormat::R8G8B8A8_UNORM => &R8G8B8A8_UNORM,
            EncodeFormat::R8G8B8A8_SNORM => &R8G8B8A8_SNORM,
            EncodeFormat::B8G8R8A8_UNORM => &B8G8R8A8_UNORM,
            EncodeFormat::B8G8R8X8_UNORM => &B8G8R8X8_UNORM,
            EncodeFormat::B5G6R5_UNORM => &B5G6R5_UNORM,
            EncodeFormat::B5G5R5A1_UNORM => &B5G5R5A1_UNORM,
            EncodeFormat::B4G4R4A4_UNORM => &B4G4R4A4_UNORM,
            EncodeFormat::A4B4G4R4_UNORM => &A4B4G4R4_UNORM,
            EncodeFormat::R8_SNORM => &R8_SNORM,
            EncodeFormat::R8_UNORM => &R8_UNORM,
            EncodeFormat::R8G8_UNORM => &R8G8_UNORM,
            EncodeFormat::R8G8_SNORM => &R8G8_SNORM,
            EncodeFormat::A8_UNORM => &A8_UNORM,
            EncodeFormat::R16_UNORM => &R16_UNORM,
            EncodeFormat::R16_SNORM => &R16_SNORM,
            EncodeFormat::R16G16_UNORM => &R16G16_UNORM,
            EncodeFormat::R16G16_SNORM => &R16G16_SNORM,
            EncodeFormat::R16G16B16A16_UNORM => &R16G16B16A16_UNORM,
            EncodeFormat::R16G16B16A16_SNORM => &R16G16B16A16_SNORM,
            EncodeFormat::R10G10B10A2_UNORM => &R10G10B10A2_UNORM,
            EncodeFormat::R11G11B10_FLOAT => &R11G11B10_FLOAT,
            EncodeFormat::R9G9B9E5_SHAREDEXP => &R9G9B9E5_SHAREDEXP,
            EncodeFormat::R16_FLOAT => &R16_FLOAT,
            EncodeFormat::R16G16_FLOAT => &R16G16_FLOAT,
            EncodeFormat::R16G16B16A16_FLOAT => &R16G16B16A16_FLOAT,
            EncodeFormat::R32_FLOAT => &R32_FLOAT,
            EncodeFormat::R32G32_FLOAT => &R32G32_FLOAT,
            EncodeFormat::R32G32B32_FLOAT => &R32G32B32_FLOAT,
            EncodeFormat::R32G32B32A32_FLOAT => &R32G32B32A32_FLOAT,
            EncodeFormat::R10G10B10_XR_BIAS_A2_UNORM => &R10G10B10_XR_BIAS_A2_UNORM,
            EncodeFormat::AYUV => &AYUV,
            EncodeFormat::Y410 => &Y410,
            EncodeFormat::Y416 => &Y416,

            EncodeFormat::R1_UNORM => &R1_UNORM,
            EncodeFormat::R8G8_B8G8_UNORM => &R8G8_B8G8_UNORM,
            EncodeFormat::G8R8_G8B8_UNORM => &G8R8_G8B8_UNORM,
            EncodeFormat::UYVY => &UYVY,
            EncodeFormat::YUY2 => &YUY2,
            EncodeFormat::Y210 => &Y210,
            EncodeFormat::Y216 => &Y216,

            // EncodeFormat::BC1_UNORM => &BC1_UNORM,
            // EncodeFormat::BC2_UNORM => &BC2_UNORM,
            // EncodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA => &BC2_UNORM_PREMULTIPLIED_ALPHA,
            // EncodeFormat::BC3_UNORM => &BC3_UNORM,
            // EncodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA => &BC3_UNORM_PREMULTIPLIED_ALPHA,
            EncodeFormat::BC4_UNORM => &BC4_UNORM,
            EncodeFormat::BC4_SNORM => &BC4_SNORM,
            EncodeFormat::BC5_UNORM => &BC5_UNORM,
            EncodeFormat::BC5_SNORM => &BC5_SNORM,
            // EncodeFormat::BC6H_UF16 => &BC6H_UF16,
            // EncodeFormat::BC6H_SF16 => &BC6H_SF16,
            // EncodeFormat::BC7_UNORM => &BC7_UNORM,
            _ => return None,
        })
    }
}
impl TryFrom<EncodeFormat> for DxgiFormat {
    type Error = ();

    fn try_from(value: EncodeFormat) -> Result<DxgiFormat, Self::Error> {
        Ok(match value {
            EncodeFormat::R8G8B8A8_UNORM => DxgiFormat::R8G8B8A8_UNORM,
            EncodeFormat::R8G8B8A8_SNORM => DxgiFormat::R8G8B8A8_SNORM,
            EncodeFormat::B8G8R8A8_UNORM => DxgiFormat::B8G8R8A8_UNORM,
            EncodeFormat::B8G8R8X8_UNORM => DxgiFormat::B8G8R8X8_UNORM,
            EncodeFormat::B5G6R5_UNORM => DxgiFormat::B5G6R5_UNORM,
            EncodeFormat::B5G5R5A1_UNORM => DxgiFormat::B5G5R5A1_UNORM,
            EncodeFormat::B4G4R4A4_UNORM => DxgiFormat::B4G4R4A4_UNORM,
            EncodeFormat::A4B4G4R4_UNORM => DxgiFormat::A4B4G4R4_UNORM,
            EncodeFormat::R8_SNORM => DxgiFormat::R8_SNORM,
            EncodeFormat::R8_UNORM => DxgiFormat::R8_UNORM,
            EncodeFormat::R8G8_UNORM => DxgiFormat::R8G8_UNORM,
            EncodeFormat::R8G8_SNORM => DxgiFormat::R8G8_SNORM,
            EncodeFormat::A8_UNORM => DxgiFormat::A8_UNORM,
            EncodeFormat::R16_UNORM => DxgiFormat::R16_UNORM,
            EncodeFormat::R16_SNORM => DxgiFormat::R16_SNORM,
            EncodeFormat::R16G16_UNORM => DxgiFormat::R16G16_UNORM,
            EncodeFormat::R16G16_SNORM => DxgiFormat::R16G16_SNORM,
            EncodeFormat::R16G16B16A16_UNORM => DxgiFormat::R16G16B16A16_UNORM,
            EncodeFormat::R16G16B16A16_SNORM => DxgiFormat::R16G16B16A16_SNORM,
            EncodeFormat::R10G10B10A2_UNORM => DxgiFormat::R10G10B10A2_UNORM,
            EncodeFormat::R11G11B10_FLOAT => DxgiFormat::R11G11B10_FLOAT,
            EncodeFormat::R9G9B9E5_SHAREDEXP => DxgiFormat::R9G9B9E5_SHAREDEXP,
            EncodeFormat::R16_FLOAT => DxgiFormat::R16_FLOAT,
            EncodeFormat::R16G16_FLOAT => DxgiFormat::R16G16_FLOAT,
            EncodeFormat::R16G16B16A16_FLOAT => DxgiFormat::R16G16B16A16_FLOAT,
            EncodeFormat::R32_FLOAT => DxgiFormat::R32_FLOAT,
            EncodeFormat::R32G32_FLOAT => DxgiFormat::R32G32_FLOAT,
            EncodeFormat::R32G32B32_FLOAT => DxgiFormat::R32G32B32_FLOAT,
            EncodeFormat::R32G32B32A32_FLOAT => DxgiFormat::R32G32B32A32_FLOAT,
            EncodeFormat::R10G10B10_XR_BIAS_A2_UNORM => DxgiFormat::R10G10B10_XR_BIAS_A2_UNORM,
            EncodeFormat::AYUV => DxgiFormat::AYUV,
            EncodeFormat::Y410 => DxgiFormat::Y410,
            EncodeFormat::Y416 => DxgiFormat::Y416,
            EncodeFormat::R1_UNORM => DxgiFormat::R1_UNORM,
            EncodeFormat::R8G8_B8G8_UNORM => DxgiFormat::R8G8_B8G8_UNORM,
            EncodeFormat::G8R8_G8B8_UNORM => DxgiFormat::G8R8_G8B8_UNORM,
            EncodeFormat::YUY2 => DxgiFormat::YUY2,
            EncodeFormat::Y210 => DxgiFormat::Y210,
            EncodeFormat::Y216 => DxgiFormat::Y216,
            EncodeFormat::BC1_UNORM => DxgiFormat::BC1_UNORM,
            EncodeFormat::BC2_UNORM => DxgiFormat::BC2_UNORM,
            EncodeFormat::BC3_UNORM => DxgiFormat::BC3_UNORM,
            EncodeFormat::BC4_UNORM => DxgiFormat::BC4_UNORM,
            EncodeFormat::BC4_SNORM => DxgiFormat::BC4_SNORM,
            EncodeFormat::BC5_UNORM => DxgiFormat::BC5_UNORM,
            EncodeFormat::BC5_SNORM => DxgiFormat::BC5_SNORM,
            EncodeFormat::BC6H_UF16 => DxgiFormat::BC6H_UF16,
            EncodeFormat::BC6H_SF16 => DxgiFormat::BC6H_SF16,
            EncodeFormat::BC7_UNORM => DxgiFormat::BC7_UNORM,
            EncodeFormat::R8G8B8_UNORM
            | EncodeFormat::B8G8R8_UNORM
            | EncodeFormat::UYVY
            | EncodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA
            | EncodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA => return Err(()),
        })
    }
}
impl TryFrom<EncodeFormat> for FourCC {
    type Error = ();

    fn try_from(value: EncodeFormat) -> Result<Self, Self::Error> {
        match value {
            EncodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA => Ok(FourCC::DXT2),
            EncodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA => Ok(FourCC::DXT4),
            EncodeFormat::BC1_UNORM => Ok(FourCC::DXT1),
            EncodeFormat::BC2_UNORM => Ok(FourCC::DXT3),
            EncodeFormat::BC3_UNORM => Ok(FourCC::DXT5),
            EncodeFormat::BC4_UNORM => Ok(FourCC::BC4U),
            EncodeFormat::BC4_SNORM => Ok(FourCC::BC4S),
            EncodeFormat::BC5_UNORM => Ok(FourCC::BC5U),
            EncodeFormat::BC5_SNORM => Ok(FourCC::BC5S),

            EncodeFormat::R8G8_B8G8_UNORM => Ok(FourCC::RGBG),
            EncodeFormat::G8R8_G8B8_UNORM => Ok(FourCC::GRGB),
            EncodeFormat::UYVY => Ok(FourCC::UYVY),
            EncodeFormat::YUY2 => Ok(FourCC::YUY2),
            _ => Err(()),
        }
    }
}
impl TryFrom<EncodeFormat> for Dx9PixelFormat {
    type Error = ();

    fn try_from(value: EncodeFormat) -> Result<Self, Self::Error> {
        match value {
            EncodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA => Ok(FourCC::DXT2.into()),
            EncodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA => Ok(FourCC::DXT4.into()),
            EncodeFormat::UYVY => Ok(FourCC::UYVY.into()),

            EncodeFormat::B8G8R8_UNORM => Ok(Self::Mask(MaskPixelFormat {
                flags: PixelFormatFlags::RGB,
                rgb_bit_count: 24,
                r_bit_mask: 0x00FF0000,
                g_bit_mask: 0x0000FF00,
                b_bit_mask: 0x000000FF,
                a_bit_mask: 0,
            })),
            EncodeFormat::R8G8B8_UNORM => Ok(Self::Mask(MaskPixelFormat {
                flags: PixelFormatFlags::RGB,
                rgb_bit_count: 24,
                r_bit_mask: 0x000000FF,
                g_bit_mask: 0x0000FF00,
                b_bit_mask: 0x00FF0000,
                a_bit_mask: 0,
            })),

            _ => Err(()),
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum EncodeError {
    UnsupportedColorFormat(ColorFormat),
    InvalidLines,
    Io(std::io::Error),
}
impl From<std::io::Error> for EncodeError {
    fn from(err: std::io::Error) -> Self {
        EncodeError::Io(err)
    }
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EncodeError::UnsupportedColorFormat(color) => {
                write!(f, "Unsupported color format: {:?}", color)
            }
            EncodeError::InvalidLines => write!(f, "Invalid lines"),
            EncodeError::Io(err) => write!(f, "IO error: {}", err),
        }
    }
}
impl std::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EncodeError::Io(err) => Some(err),
            _ => None,
        }
    }
}

/// A line-based encoder.
trait Encoder {
    /// A non-empty list of all supported color formats for
    /// [`Encoder::encode`].
    fn supported_color_formats(&self) -> ColorFormatSet;

    fn supports_dithering(&self) -> DitheredChannels;

    /// Encodes the given image.
    fn encode(
        &self,
        data: &[u8],
        width: u32,
        color: ColorFormat,
        writer: &mut dyn Write,
        options: &EncodeOptions,
    ) -> Result<(), EncodeError>;
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EncodeOptions {
    pub dither: DitheredChannels,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            dither: DitheredChannels::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DitheredChannels {
    /// Dithering is disabled for all channels.
    None,
    /// Dithering is enabled for all channels (RGBA).
    All,
    /// Dithering is enabled only for color channels (RGB).
    ColorOnly,
    /// Dithering is enabled only for the alpha channel.
    AlphaOnly,
}
impl DitheredChannels {
    pub(crate) fn intersect(self, other: Self) -> Self {
        match (self, other) {
            (DitheredChannels::None, _) | (_, DitheredChannels::None) => DitheredChannels::None,
            (DitheredChannels::All, other) | (other, DitheredChannels::All) => other,
            (DitheredChannels::ColorOnly, DitheredChannels::AlphaOnly)
            | (DitheredChannels::AlphaOnly, DitheredChannels::ColorOnly) => DitheredChannels::None,
            (DitheredChannels::ColorOnly, DitheredChannels::ColorOnly) => {
                DitheredChannels::ColorOnly
            }
            (DitheredChannels::AlphaOnly, DitheredChannels::AlphaOnly) => {
                DitheredChannels::AlphaOnly
            }
        }
    }

    pub(crate) fn color(self) -> bool {
        matches!(self, DitheredChannels::All | DitheredChannels::ColorOnly)
    }
    pub(crate) fn alpha(self) -> bool {
        matches!(self, DitheredChannels::All | DitheredChannels::AlphaOnly)
    }
}

pub(crate) struct Args<'a, 'b>(&'a [u8], u32, ColorFormat, &'b mut dyn Write, EncodeOptions);

pub(crate) struct DecodedArgs<'a, 'b> {
    data: &'a [u8],
    width: usize,
    height: usize,
    color: ColorFormat,
    writer: &'b mut dyn Write,
    options: EncodeOptions,
}
impl<'a, 'b> DecodedArgs<'a, 'b> {
    fn from(args: Args<'a, 'b>) -> Result<Self, EncodeError> {
        let color = args.2;
        let writer = args.3;
        let options = args.4;

        let data = args.0;
        if data.is_empty() {
            return Err(EncodeError::InvalidLines);
        }

        let bytes_per_pixel = color.bytes_per_pixel() as usize;
        debug_assert!(bytes_per_pixel > 0);
        if data.len() % bytes_per_pixel != 0 {
            return Err(EncodeError::InvalidLines);
        }

        let width = args.1 as usize;
        let stride = width * bytes_per_pixel;
        if stride == 0 || data.len() % stride != 0 {
            return Err(EncodeError::InvalidLines);
        }

        let height = data.len() / stride;
        if stride * height != data.len() {
            return Err(EncodeError::InvalidLines);
        }

        Ok(Self {
            data,
            width,
            height,
            color,
            writer,
            options,
        })
    }
}
