use std::{
    io::{Read, Seek, Write},
    num::NonZeroU8,
};

use crate::{
    cast,
    decode::{get_decoders, ReadSeek},
    detect,
    encode::get_encoders,
    Channels, ColorFormat, DecodeError, Dithering, Dx9PixelFormat, DxgiFormat, EncodeError,
    EncodeOptions, FormatError, FourCC, Header, MaskPixelFormat, PixelFormatFlags, Precision, Rect,
    Size, SizeMultiple,
};

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
impl Format {
    /// Returns the format of the surfaces from a DDS header.
    pub fn from_header(header: &Header) -> Result<Format, FormatError> {
        match header {
            Header::Dx9(dx9) => match &dx9.pixel_format {
                Dx9PixelFormat::FourCC(four_cc) => detect::four_cc_to_supported(*four_cc)
                    .ok_or(FormatError::UnsupportedFourCC(*four_cc)),
                Dx9PixelFormat::Mask(pixel_format) => {
                    detect::pixel_format_to_supported(pixel_format)
                        .ok_or(FormatError::UnsupportedPixelFormat)
                }
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

    /// Decodes the image data of a surface from the given reader and writes it
    /// to the given output buffer.
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
        let reader = reader as &mut dyn ReadSeek;
        let decoders = get_decoders(*self);
        decoders.decode_rect(color, reader, size, rect, output, row_pitch)
    }

    pub fn encode<W: Write>(
        &self,
        writer: &mut W,
        size: Size,
        color: ColorFormat,
        data: &[u8],
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        if let Some(encoders) = get_encoders(*self) {
            encoders.encode(data, size.width, color, writer, options)
        } else {
            Err(EncodeError::UnsupportedFormat(*self))
        }
    }
    pub fn encode_u8<W: Write>(
        &self,
        writer: &mut W,
        size: Size,
        channels: Channels,
        data: &[u8],
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        self.encode(
            writer,
            size,
            ColorFormat::new(channels, Precision::U8),
            data,
            options,
        )
    }
    pub fn encode_u16<W: Write>(
        &self,
        writer: &mut W,
        size: Size,
        channels: Channels,
        data: &[u16],
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        self.encode(
            writer,
            size,
            ColorFormat::new(channels, Precision::U16),
            cast::as_bytes(data),
            options,
        )
    }
    pub fn encode_f32<W: Write>(
        &self,
        writer: &mut W,
        size: Size,
        channels: Channels,
        data: &[f32],
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        self.encode(
            writer,
            size,
            ColorFormat::new(channels, Precision::F32),
            cast::as_bytes(data),
            options,
        )
    }

    /// Returns information about the encoding support of this format.
    ///
    /// If the format does not support encoding, `None` is returned.
    pub const fn encoding(self) -> Option<EncodingSupport> {
        if let Some(encoders) = get_encoders(self) {
            Some(EncodingSupport {
                dithering: encoders.supported_dithering(),
                split_height: encoders.split_height,
                local_dithering: encoders.local_dithering(),
                size_multiple: encoders.size_multiple,
            })
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
impl TryFrom<Format> for Dx9PixelFormat {
    type Error = ();

    fn try_from(value: Format) -> Result<Self, Self::Error> {
        if let Ok(four_cc) = FourCC::try_from(value) {
            return Ok(four_cc.into());
        }

        // TODO: more formats
        match value {
            Format::B8G8R8_UNORM => Ok(Self::Mask(MaskPixelFormat {
                flags: PixelFormatFlags::RGB,
                rgb_bit_count: 24,
                r_bit_mask: 0x00FF0000,
                g_bit_mask: 0x0000FF00,
                b_bit_mask: 0x000000FF,
                a_bit_mask: 0,
            })),
            Format::R8G8B8_UNORM => Ok(Self::Mask(MaskPixelFormat {
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

/// Describes the extent of support for encoding a format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodingSupport {
    /// Whether and what type of dithering is supported.
    pub dithering: Dithering,
    /// The split height for the image format.
    ///
    /// Encoding most formats is trivially parallelizable, by splitting the
    /// image into chunks by lines, encoding each chunk separately, and writing
    /// the encoded chunks to the output stream in order.
    ///
    /// This value specifies how many lines need to be grouped together for
    /// correct encoding. E.g. `BC1_UNORM` requires 4 lines to be grouped
    /// together, meaning that all chunks (except the last one) must have a
    /// height that is a multiple of 4. So e.g. an image with a height of 10
    /// pixels can split into chunks with heights of 4-4-2, 8-2, 4-6, or 10.
    ///
    /// Note that most dithering will produce different (but not necessarily
    /// incorrect) results if the image is split into chunks. However, all BCn
    /// formats implement block-based local dithering, meaning that the dithering
    /// is the same whether the image is split or not. See
    /// [`EncodingSupport::local_dithering`].
    pub split_height: Option<NonZeroU8>,
    /// Whether the format supports local dithering.
    ///
    /// Most formats implement global error diffusing dithering for best quality.
    /// However, this prevents parallel encoding of the image, as the dithering
    /// error of one chunk depends on the dithering error of the previous chunk.
    /// It's still possible to encode the image in parallel, but the dither
    /// pattern may reveal the chunk seams.
    ///
    /// Local dithering on the other hand will attempt to diffuse the error
    /// within a small region of the image. E.g. `BC1_UNORM` will dither within
    /// a 4x4 block. This allows the image to be split into chunks and encoded
    /// in parallel without revealing the chunk seams.
    ///
    /// `self.dithering == Dithering::None` implies `self.local_dithering == false`.
    pub local_dithering: bool,
    /// The size multiple of the encoded image.
    ///
    /// If the dimensions of the image are not multiples of this size, the
    /// encoder with return an error.
    pub size_multiple: SizeMultiple,
}
