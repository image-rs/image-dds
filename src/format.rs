use crate::header::{Dx9PixelFormat, DxgiFormat, FourCC, Header, MaskPixelFormat};
use crate::{
    decode::get_decoders, detect, encode::get_encoders, Channels, ColorFormat, EncodingSupport,
    FormatError, Precision,
};

/// The format of the pixel data of a surface.
///
/// This enumeration is modelled after the [DXGI_FORMAT enumeration](https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format)
/// and has the same semantics and naming conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum Format {
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

    // bi-planar formats
    NV12,
    P010,
    P016,

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

    // ASTC
    ASTC_4X4_UNORM,
    ASTC_5X4_UNORM,
    ASTC_5X5_UNORM,
    ASTC_6X5_UNORM,
    ASTC_6X6_UNORM,
    ASTC_8X5_UNORM,
    ASTC_8X6_UNORM,
    ASTC_8X8_UNORM,
    ASTC_10X5_UNORM,
    ASTC_10X6_UNORM,
    ASTC_10X8_UNORM,
    ASTC_10X10_UNORM,
    ASTC_12X10_UNORM,
    ASTC_12X12_UNORM,

    // non-standard formats
    /// This is just [`Format::BC3_UNORM`], but with the R channel stored in alpha.
    ///
    /// BC3 stores the A channel with a much higher precision than the other
    /// (color) channels. RXGB uses this by storing the R channel of the image in
    /// the BC3 A channel, effectively increasing the precision of not just the
    /// R channel, but also the G and B channels.
    ///
    /// Note that this is an RGB format. The BC3-encoded R channel is commonly
    /// set to 0 to improve the quality of G and B.
    BC3_UNORM_RXGB,
}
impl Format {
    /// Returns the format of the surfaces from a DDS header.
    pub fn from_header(header: &Header) -> Result<Format, FormatError> {
        match header {
            Header::Dx9(dx9) => match &dx9.pixel_format {
                Dx9PixelFormat::FourCC(four_cc) => detect::four_cc_to_supported(*four_cc)
                    .ok_or(FormatError::UnsupportedFourCC(*four_cc)),
                Dx9PixelFormat::Mask(pixel_format) => detect::masked_to_supported(pixel_format)
                    .ok_or(FormatError::UnsupportedPixelFormat),
            },
            Header::Dx10(dx10) => {
                if let Some(format) = detect::special_cases(dx10) {
                    return Ok(format);
                }

                detect::dxgi_format_to_supported(dx10.dxgi_format)
                    .ok_or(FormatError::UnsupportedDxgiFormat(dx10.dxgi_format))
            }
        }
    }
    /// Returns the format of a surface from a DXGI format.
    ///
    /// `None` if the DXGI format is not supported for decoding.
    pub const fn from_dxgi(dxgi: DxgiFormat) -> Option<Format> {
        detect::dxgi_format_to_supported(dxgi)
    }
    /// Returns the format of a surface from a FourCC code.
    ///
    /// `None` if the FourCC code is not supported for decoding.
    pub const fn from_four_cc(four_cc: FourCC) -> Option<Format> {
        detect::four_cc_to_supported(four_cc)
    }

    /// The number and type of (color) channels in the surface.
    ///
    /// If the channels of a format cannot be accurately described by
    /// [`Channels`], the next larger type is used. For example, a format with
    /// only R and G channels will be described as [`Channels::Rgb`].
    pub const fn channels(&self) -> Channels {
        self.color().channels
    }
    /// The precision/bit depth closest to the values in the surface.
    ///
    /// DDS supports formats with various precisions and ranges, and not all of
    /// them can be represented *exactly* by the [`Precision`] enum. The closest
    /// precision is chosen based on the format's range and encoded bit depth.
    /// It is typically larger than the encoded bit depth.
    ///
    /// E.g. the format `B5G6R5_UNORM` is a 5/6-bit per channel format and the
    /// closest precision is [`Precision::U8`]. While `U8` can closely
    /// approximate all `B5G6R5_UNORM` values, it is not exact. E.g. a 5-bit
    /// UNORM value of 11 is 90.48 as an 8-bit UNORM value exactly but will be
    /// rounded to 90.
    pub const fn precision(&self) -> Precision {
        self.color().precision
    }
    /// The native color format of the surface.
    ///
    /// This is simply [`Self::channels`] and [`Self::precision`] combined.
    pub const fn color(&self) -> ColorFormat {
        get_decoders(*self).native_color()
    }

    /// Returns information about the encoding support of this format.
    ///
    /// If the format does not support encoding, `None` is returned.
    pub const fn encoding_support(self) -> Option<EncodingSupport> {
        if let Some(encoders) = get_encoders(self) {
            Some(encoders.encoding_support())
        } else {
            None
        }
    }
}

impl TryFrom<Format> for DxgiFormat {
    type Error = ();

    fn try_from(value: Format) -> Result<DxgiFormat, Self::Error> {
        Ok(match value {
            // uncompressed
            Format::R8G8B8A8_UNORM => DxgiFormat::R8G8B8A8_UNORM,
            Format::R8G8B8A8_SNORM => DxgiFormat::R8G8B8A8_SNORM,
            Format::B8G8R8A8_UNORM => DxgiFormat::B8G8R8A8_UNORM,
            Format::B8G8R8X8_UNORM => DxgiFormat::B8G8R8X8_UNORM,
            Format::B5G6R5_UNORM => DxgiFormat::B5G6R5_UNORM,
            Format::B5G5R5A1_UNORM => DxgiFormat::B5G5R5A1_UNORM,
            Format::B4G4R4A4_UNORM => DxgiFormat::B4G4R4A4_UNORM,
            Format::A4B4G4R4_UNORM => DxgiFormat::A4B4G4R4_UNORM,
            Format::R8_SNORM => DxgiFormat::R8_SNORM,
            Format::R8_UNORM => DxgiFormat::R8_UNORM,
            Format::R8G8_UNORM => DxgiFormat::R8G8_UNORM,
            Format::R8G8_SNORM => DxgiFormat::R8G8_SNORM,
            Format::A8_UNORM => DxgiFormat::A8_UNORM,
            Format::R16_UNORM => DxgiFormat::R16_UNORM,
            Format::R16_SNORM => DxgiFormat::R16_SNORM,
            Format::R16G16_UNORM => DxgiFormat::R16G16_UNORM,
            Format::R16G16_SNORM => DxgiFormat::R16G16_SNORM,
            Format::R16G16B16A16_UNORM => DxgiFormat::R16G16B16A16_UNORM,
            Format::R16G16B16A16_SNORM => DxgiFormat::R16G16B16A16_SNORM,
            Format::R10G10B10A2_UNORM => DxgiFormat::R10G10B10A2_UNORM,
            Format::R11G11B10_FLOAT => DxgiFormat::R11G11B10_FLOAT,
            Format::R9G9B9E5_SHAREDEXP => DxgiFormat::R9G9B9E5_SHAREDEXP,
            Format::R16_FLOAT => DxgiFormat::R16_FLOAT,
            Format::R16G16_FLOAT => DxgiFormat::R16G16_FLOAT,
            Format::R16G16B16A16_FLOAT => DxgiFormat::R16G16B16A16_FLOAT,
            Format::R32_FLOAT => DxgiFormat::R32_FLOAT,
            Format::R32G32_FLOAT => DxgiFormat::R32G32_FLOAT,
            Format::R32G32B32_FLOAT => DxgiFormat::R32G32B32_FLOAT,
            Format::R32G32B32A32_FLOAT => DxgiFormat::R32G32B32A32_FLOAT,
            Format::R10G10B10_XR_BIAS_A2_UNORM => DxgiFormat::R10G10B10_XR_BIAS_A2_UNORM,
            Format::AYUV => DxgiFormat::AYUV,
            Format::Y410 => DxgiFormat::Y410,
            Format::Y416 => DxgiFormat::Y416,

            // sub-sampled
            Format::R1_UNORM => DxgiFormat::R1_UNORM,
            Format::R8G8_B8G8_UNORM => DxgiFormat::R8G8_B8G8_UNORM,
            Format::G8R8_G8B8_UNORM => DxgiFormat::G8R8_G8B8_UNORM,
            Format::YUY2 => DxgiFormat::YUY2,
            Format::Y210 => DxgiFormat::Y210,
            Format::Y216 => DxgiFormat::Y216,

            // bi-planar
            Format::NV12 => DxgiFormat::NV12,
            Format::P010 => DxgiFormat::P010,
            Format::P016 => DxgiFormat::P016,

            // block compression
            Format::BC1_UNORM => DxgiFormat::BC1_UNORM,
            Format::BC2_UNORM => DxgiFormat::BC2_UNORM,
            Format::BC3_UNORM => DxgiFormat::BC3_UNORM,
            Format::BC4_UNORM => DxgiFormat::BC4_UNORM,
            Format::BC4_SNORM => DxgiFormat::BC4_SNORM,
            Format::BC5_UNORM => DxgiFormat::BC5_UNORM,
            Format::BC5_SNORM => DxgiFormat::BC5_SNORM,
            Format::BC6H_UF16 => DxgiFormat::BC6H_UF16,
            Format::BC6H_SF16 => DxgiFormat::BC6H_SF16,
            Format::BC7_UNORM => DxgiFormat::BC7_UNORM,

            // ASTC
            Format::ASTC_4X4_UNORM => DxgiFormat::ASTC_4X4_UNORM,
            Format::ASTC_5X4_UNORM => DxgiFormat::ASTC_5X4_UNORM,
            Format::ASTC_5X5_UNORM => DxgiFormat::ASTC_5X5_UNORM,
            Format::ASTC_6X5_UNORM => DxgiFormat::ASTC_6X5_UNORM,
            Format::ASTC_6X6_UNORM => DxgiFormat::ASTC_6X6_UNORM,
            Format::ASTC_8X5_UNORM => DxgiFormat::ASTC_8X5_UNORM,
            Format::ASTC_8X6_UNORM => DxgiFormat::ASTC_8X6_UNORM,
            Format::ASTC_8X8_UNORM => DxgiFormat::ASTC_8X8_UNORM,
            Format::ASTC_10X5_UNORM => DxgiFormat::ASTC_10X5_UNORM,
            Format::ASTC_10X6_UNORM => DxgiFormat::ASTC_10X6_UNORM,
            Format::ASTC_10X8_UNORM => DxgiFormat::ASTC_10X8_UNORM,
            Format::ASTC_10X10_UNORM => DxgiFormat::ASTC_10X10_UNORM,
            Format::ASTC_12X10_UNORM => DxgiFormat::ASTC_12X10_UNORM,
            Format::ASTC_12X12_UNORM => DxgiFormat::ASTC_12X12_UNORM,

            // cannot be represented by DXGI
            Format::R8G8B8_UNORM
            | Format::B8G8R8_UNORM
            | Format::UYVY
            | Format::BC2_UNORM_PREMULTIPLIED_ALPHA
            | Format::BC3_UNORM_PREMULTIPLIED_ALPHA
            | Format::BC3_UNORM_RXGB => return Err(()),
        })
    }
}
impl TryFrom<Format> for FourCC {
    type Error = ();

    fn try_from(value: Format) -> Result<Self, Self::Error> {
        match value {
            Format::BC2_UNORM_PREMULTIPLIED_ALPHA => Ok(FourCC::DXT2),
            Format::BC3_UNORM_PREMULTIPLIED_ALPHA => Ok(FourCC::DXT4),
            Format::BC1_UNORM => Ok(FourCC::DXT1),
            Format::BC2_UNORM => Ok(FourCC::DXT3),
            Format::BC3_UNORM => Ok(FourCC::DXT5),
            Format::BC4_UNORM => Ok(FourCC::BC4U),
            Format::BC4_SNORM => Ok(FourCC::BC4S),
            Format::BC5_UNORM => Ok(FourCC::BC5U),
            Format::BC5_SNORM => Ok(FourCC::BC5S),

            Format::R8G8_B8G8_UNORM => Ok(FourCC::RGBG),
            Format::G8R8_G8B8_UNORM => Ok(FourCC::GRGB),
            Format::UYVY => Ok(FourCC::UYVY),
            Format::YUY2 => Ok(FourCC::YUY2),

            Format::BC3_UNORM_RXGB => Ok(FourCC::RXGB),
            _ => Err(()),
        }
    }
}
impl TryFrom<Format> for MaskPixelFormat {
    type Error = ();

    fn try_from(value: Format) -> Result<Self, Self::Error> {
        detect::supported_to_masked(value).ok_or(())
    }
}
impl TryFrom<Format> for Dx9PixelFormat {
    type Error = ();

    fn try_from(value: Format) -> Result<Self, Self::Error> {
        if let Ok(four_cc) = FourCC::try_from(value) {
            return Ok(four_cc.into());
        }
        if let Ok(masked) = MaskPixelFormat::try_from(value) {
            return Ok(masked.into());
        }
        Err(())
    }
}
