use std::io::{Read, Seek};

use crate::{
    cast,
    decode::{self, DecoderSet, ReadSeek},
    detect, Channels, ColorFormat, DecodeError, Dx9PixelFormat, DxgiFormat, FourCC, Header,
    Precision, Rect, Size,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum DecodeFormat {
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

    // non-standard formats
    /// This is just [`BC3_UNORM`], but with the R channel stored in alpha.
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
impl DecodeFormat {
    /// Returns the format of the surfaces from a DDS header.
    pub fn from_header(header: &Header) -> Result<DecodeFormat, DecodeError> {
        match header {
            Header::Dx9(dx9) => match &dx9.pixel_format {
                Dx9PixelFormat::FourCC(four_cc) => detect::four_cc_to_supported(*four_cc)
                    .ok_or(DecodeError::UnsupportedFourCC(*four_cc)),
                Dx9PixelFormat::Mask(pixel_format) => {
                    detect::pixel_format_to_supported(pixel_format)
                        .ok_or(DecodeError::UnsupportedPixelFormat)
                }
            },
            Header::Dx10(dx10) => {
                if let Some(format) = detect::special_cases(dx10) {
                    return Ok(format);
                }

                detect::dxgi_format_to_supported(dx10.dxgi_format)
                    .ok_or(DecodeError::UnsupportedDxgiFormat(dx10.dxgi_format))
            }
        }
    }
    /// Returns the format of a surface from a DXGI format.
    ///
    /// `None` if the DXGI format is not supported for decoding.
    pub const fn from_dxgi(dxgi: DxgiFormat) -> Option<DecodeFormat> {
        detect::dxgi_format_to_supported(dxgi)
    }
    /// Returns the format of a surface from a FourCC code.
    ///
    /// `None` if the FourCC code is not supported for decoding.
    pub const fn from_four_cc(four_cc: FourCC) -> Option<DecodeFormat> {
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

    /// Returns `true` if this format supports decoding as the given color
    /// format.
    ///
    /// ## Channel and precision combinations
    ///
    /// All color formats that consist of a supported channels type and
    /// supported precision are supported. This means that all combinations
    /// of channel and precisions from [`Self::supports_channels`] and
    /// [`Self::supports_precision`] respectively are supported color formats.
    pub fn supports(&self, color: ColorFormat) -> bool {
        self.supports_channels(color.channels) && self.supports_precision(color.precision)
    }
    /// Whether this format supports decoding with the given channels.
    ///
    /// `self.supports_channels(self.channels())` is always `true`.
    pub fn supports_channels(&self, channels: Channels) -> bool {
        get_decoders(*self)
            .supported_channels()
            .contains_channels(channels)
    }
    /// Whether this format supports decoding with the given precision.
    ///
    /// `self.supports_precision(self.precision())` is always `true`.
    pub fn supports_precision(&self, precision: Precision) -> bool {
        get_decoders(*self)
            .supported_precisions()
            .contains_precision(precision)
    }

    /// Decodes the image data of a surface from the given reader and writes it
    /// to the given output buffer.
    ///
    /// If this format does not support the given channels and precision, an
    /// error is returned. Support can be checked ahead of time with
    /// [`Self::supported_channels`] and [`Self::supported_precisions`].
    ///
    /// ## Output buffer
    ///
    /// The output buffer must be exactly the right size to hold the decoded
    /// image data.
    ///
    /// The size in bytes of the output buffer can be calculated as
    /// `size.pixels() * color.bytes_per_pixel()`. If you are using one of the
    /// `decode_<precision>` methods, the length of the types output buffer is
    /// `size.pixels() * channels.count()`
    ///
    /// It is highly recommended for the output buffer to be aligned to the
    /// given precision to improve performance. E.g. if the precision is `U16`,
    /// the output buffer should be aligned to 2 bytes. As such, using the
    /// `decode_<precision>` methods is recommended.
    ///
    /// ## State of the reader
    ///
    /// The reader is expected to be positioned at the start of the encoded
    /// image data of the current surface.
    ///
    /// If the operation completes successfully, the reader will be positioned
    /// at the end of the encoded image data, meaning that the next byte read
    /// will be the first byte of either the next encoded surface or EOF.
    ///
    /// If the operation fails and returns an error, the position of the reader
    /// remains unchanged.
    ///
    /// ## Panics
    ///
    /// This method will only panic in the given reader panics while reading.
    pub fn decode(
        &self,
        reader: &mut dyn Read,
        size: Size,
        color: ColorFormat,
        output: &mut [u8],
    ) -> Result<(), DecodeError> {
        if !self.supports(color) {
            return Err(DecodeError::UnsupportedColorFormat {
                color,
                format: *self,
            });
        }

        get_decoders(*self).decode(color, reader, size, output)
    }

    /// A convenience method to decode with [`Precision::U8`].
    ///
    /// See [`Self::decode`] for more details.
    pub fn decode_u8(
        &self,
        reader: &mut dyn Read,
        size: Size,
        channels: Channels,
        output: &mut [u8],
    ) -> Result<(), DecodeError> {
        self.decode(
            reader,
            size,
            ColorFormat::new(channels, Precision::U8),
            output,
        )
    }
    /// A convenience method to decode with [`Precision::U16`].
    ///
    /// See [`Self::decode`] for more details.
    pub fn decode_u16(
        &self,
        reader: &mut dyn Read,
        size: Size,
        channels: Channels,
        output: &mut [u16],
    ) -> Result<(), DecodeError> {
        self.decode(
            reader,
            size,
            ColorFormat::new(channels, Precision::U16),
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
        channels: Channels,
        output: &mut [f32],
    ) -> Result<(), DecodeError> {
        self.decode(
            reader,
            size,
            ColorFormat::new(channels, Precision::F32),
            cast::as_bytes_mut(output),
        )
    }

    /// Decodes a rectangle of the image data of a surface from the given reader
    /// and writes it to the given output buffer.
    ///
    /// If this format does not support the given channels and precision, an
    /// error is returned. Support can be checked ahead of time with
    /// [`Self::supported_channels`] and [`Self::supported_precisions`].
    ///
    /// ## Row pitch and the output buffer
    ///
    /// The `row_pitch` parameter specifies the number of bytes between the start
    /// of one row and the start of the next row in the output buffer.
    ///
    /// It is highly recommended for the output buffer to be aligned to the
    /// given precision to improve performance. E.g. if the precision is `U16`,
    /// the output buffer should be aligned to 2 bytes. As such, using the
    /// `decode_*` methods is recommended.
    ///
    /// ## State of the reader
    ///
    /// The reader is expected to be positioned at the start of the encoded
    /// image data of the current surface.
    ///
    /// If the operation completes successfully, the reader will be positioned
    /// at the end of the encoded image data, meaning that the next byte read
    /// will be the first byte of either the next encoded surface or EOF.
    ///
    /// If the operation fails and returns an error, the position of the reader
    /// remains unchanged.
    ///
    /// ## Panics
    ///
    /// This method will only panic in the given reader panics while reading.
    pub fn decode_rect<R: Read + Seek>(
        &self,
        reader: &mut R,
        size: Size,
        rect: Rect,
        color: ColorFormat,
        output: &mut [u8],
        row_pitch: usize,
    ) -> Result<(), DecodeError> {
        if !self.supports(color) {
            return Err(DecodeError::UnsupportedColorFormat {
                color,
                format: *self,
            });
        }

        let reader = reader as &mut dyn ReadSeek;
        let decoders = get_decoders(*self);
        decoders.decode_rect(color, reader, size, rect, output, row_pitch)
    }
}

const fn get_decoders(format: DecodeFormat) -> DecoderSet {
    match format {
        // uncompressed formats
        DecodeFormat::R8G8B8_UNORM => decode::R8G8B8_UNORM,
        DecodeFormat::B8G8R8_UNORM => decode::B8G8R8_UNORM,
        DecodeFormat::R8G8B8A8_UNORM => decode::R8G8B8A8_UNORM,
        DecodeFormat::R8G8B8A8_SNORM => decode::R8G8B8A8_SNORM,
        DecodeFormat::B8G8R8A8_UNORM => decode::B8G8R8A8_UNORM,
        DecodeFormat::B8G8R8X8_UNORM => decode::B8G8R8X8_UNORM,
        DecodeFormat::B5G6R5_UNORM => decode::B5G6R5_UNORM,
        DecodeFormat::B5G5R5A1_UNORM => decode::B5G5R5A1_UNORM,
        DecodeFormat::B4G4R4A4_UNORM => decode::B4G4R4A4_UNORM,
        DecodeFormat::A4B4G4R4_UNORM => decode::A4B4G4R4_UNORM,
        DecodeFormat::R8_SNORM => decode::R8_SNORM,
        DecodeFormat::R8_UNORM => decode::R8_UNORM,
        DecodeFormat::R8G8_UNORM => decode::R8G8_UNORM,
        DecodeFormat::R8G8_SNORM => decode::R8G8_SNORM,
        DecodeFormat::A8_UNORM => decode::A8_UNORM,
        DecodeFormat::R16_UNORM => decode::R16_UNORM,
        DecodeFormat::R16_SNORM => decode::R16_SNORM,
        DecodeFormat::R16G16_UNORM => decode::R16G16_UNORM,
        DecodeFormat::R16G16_SNORM => decode::R16G16_SNORM,
        DecodeFormat::R16G16B16A16_UNORM => decode::R16G16B16A16_UNORM,
        DecodeFormat::R16G16B16A16_SNORM => decode::R16G16B16A16_SNORM,
        DecodeFormat::R10G10B10A2_UNORM => decode::R10G10B10A2_UNORM,
        DecodeFormat::R11G11B10_FLOAT => decode::R11G11B10_FLOAT,
        DecodeFormat::R9G9B9E5_SHAREDEXP => decode::R9G9B9E5_SHAREDEXP,
        DecodeFormat::R16_FLOAT => decode::R16_FLOAT,
        DecodeFormat::R16G16_FLOAT => decode::R16G16_FLOAT,
        DecodeFormat::R16G16B16A16_FLOAT => decode::R16G16B16A16_FLOAT,
        DecodeFormat::R32_FLOAT => decode::R32_FLOAT,
        DecodeFormat::R32G32_FLOAT => decode::R32G32_FLOAT,
        DecodeFormat::R32G32B32_FLOAT => decode::R32G32B32_FLOAT,
        DecodeFormat::R32G32B32A32_FLOAT => decode::R32G32B32A32_FLOAT,
        DecodeFormat::R10G10B10_XR_BIAS_A2_UNORM => decode::R10G10B10_XR_BIAS_A2_UNORM,
        DecodeFormat::AYUV => decode::AYUV,
        DecodeFormat::Y410 => decode::Y410,
        DecodeFormat::Y416 => decode::Y416,

        // sub-sampled formats
        DecodeFormat::R1_UNORM => decode::R1_UNORM,
        DecodeFormat::R8G8_B8G8_UNORM => decode::R8G8_B8G8_UNORM,
        DecodeFormat::G8R8_G8B8_UNORM => decode::G8R8_G8B8_UNORM,
        DecodeFormat::UYVY => decode::UYVY,
        DecodeFormat::YUY2 => decode::YUY2,
        DecodeFormat::Y210 => decode::Y210,
        DecodeFormat::Y216 => decode::Y216,

        // block compression formats
        DecodeFormat::BC1_UNORM => decode::BC1_UNORM,
        DecodeFormat::BC2_UNORM => decode::BC2_UNORM,
        DecodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA => decode::BC2_UNORM_PREMULTIPLIED_ALPHA,
        DecodeFormat::BC3_UNORM => decode::BC3_UNORM,
        DecodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA => decode::BC3_UNORM_PREMULTIPLIED_ALPHA,
        DecodeFormat::BC4_UNORM => decode::BC4_UNORM,
        DecodeFormat::BC4_SNORM => decode::BC4_SNORM,
        DecodeFormat::BC5_UNORM => decode::BC5_UNORM,
        DecodeFormat::BC5_SNORM => decode::BC5_SNORM,
        DecodeFormat::BC6H_UF16 => decode::BC6H_UF16,
        DecodeFormat::BC6H_SF16 => decode::BC6H_SF16,
        DecodeFormat::BC7_UNORM => decode::BC7_UNORM,

        // non-standard formats
        DecodeFormat::BC3_UNORM_RXGB => decode::BC3_UNORM_RXGB,
    }
}
