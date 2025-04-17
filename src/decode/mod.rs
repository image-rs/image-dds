mod astc;
mod bc;
mod bc6;
mod bc7;
mod bcn_util;
mod bi_planar;
mod decoder;
mod read_write;
mod sub_sampled;
mod uncompressed;

use std::io::{Read, Seek};

use astc::*;
use bc::*;
use bi_planar::*;
pub(crate) use decoder::*;
use sub_sampled::*;
use uncompressed::*;

use crate::{ColorFormat, DecodingError, Format, ImageViewMut, Rect, Size};

pub(crate) const fn get_decoders(format: Format) -> DecoderSet {
    match format {
        // uncompressed formats
        Format::R8G8B8_UNORM => R8G8B8_UNORM,
        Format::B8G8R8_UNORM => B8G8R8_UNORM,
        Format::R8G8B8A8_UNORM => R8G8B8A8_UNORM,
        Format::R8G8B8A8_SNORM => R8G8B8A8_SNORM,
        Format::B8G8R8A8_UNORM => B8G8R8A8_UNORM,
        Format::B8G8R8X8_UNORM => B8G8R8X8_UNORM,
        Format::B5G6R5_UNORM => B5G6R5_UNORM,
        Format::B5G5R5A1_UNORM => B5G5R5A1_UNORM,
        Format::B4G4R4A4_UNORM => B4G4R4A4_UNORM,
        Format::A4B4G4R4_UNORM => A4B4G4R4_UNORM,
        Format::R8_SNORM => R8_SNORM,
        Format::R8_UNORM => R8_UNORM,
        Format::R8G8_UNORM => R8G8_UNORM,
        Format::R8G8_SNORM => R8G8_SNORM,
        Format::A8_UNORM => A8_UNORM,
        Format::R16_UNORM => R16_UNORM,
        Format::R16_SNORM => R16_SNORM,
        Format::R16G16_UNORM => R16G16_UNORM,
        Format::R16G16_SNORM => R16G16_SNORM,
        Format::R16G16B16A16_UNORM => R16G16B16A16_UNORM,
        Format::R16G16B16A16_SNORM => R16G16B16A16_SNORM,
        Format::R10G10B10A2_UNORM => R10G10B10A2_UNORM,
        Format::R11G11B10_FLOAT => R11G11B10_FLOAT,
        Format::R9G9B9E5_SHAREDEXP => R9G9B9E5_SHAREDEXP,
        Format::R16_FLOAT => R16_FLOAT,
        Format::R16G16_FLOAT => R16G16_FLOAT,
        Format::R16G16B16A16_FLOAT => R16G16B16A16_FLOAT,
        Format::R32_FLOAT => R32_FLOAT,
        Format::R32G32_FLOAT => R32G32_FLOAT,
        Format::R32G32B32_FLOAT => R32G32B32_FLOAT,
        Format::R32G32B32A32_FLOAT => R32G32B32A32_FLOAT,
        Format::R10G10B10_XR_BIAS_A2_UNORM => R10G10B10_XR_BIAS_A2_UNORM,
        Format::AYUV => AYUV,
        Format::Y410 => Y410,
        Format::Y416 => Y416,

        // sub-sampled formats
        Format::R1_UNORM => R1_UNORM,
        Format::R8G8_B8G8_UNORM => R8G8_B8G8_UNORM,
        Format::G8R8_G8B8_UNORM => G8R8_G8B8_UNORM,
        Format::UYVY => UYVY,
        Format::YUY2 => YUY2,
        Format::Y210 => Y210,
        Format::Y216 => Y216,

        // bi-planar formats
        Format::NV12 => NV12,
        Format::P010 => P010,
        Format::P016 => P016,

        // block compression formats
        Format::BC1_UNORM => BC1_UNORM,
        Format::BC2_UNORM => BC2_UNORM,
        Format::BC2_UNORM_PREMULTIPLIED_ALPHA => BC2_UNORM_PREMULTIPLIED_ALPHA,
        Format::BC3_UNORM => BC3_UNORM,
        Format::BC3_UNORM_PREMULTIPLIED_ALPHA => BC3_UNORM_PREMULTIPLIED_ALPHA,
        Format::BC4_UNORM => BC4_UNORM,
        Format::BC4_SNORM => BC4_SNORM,
        Format::BC5_UNORM => BC5_UNORM,
        Format::BC5_SNORM => BC5_SNORM,
        Format::BC6H_UF16 => BC6H_UF16,
        Format::BC6H_SF16 => BC6H_SF16,
        Format::BC7_UNORM => BC7_UNORM,

        // ASTC formats
        Format::ASTC_4X4_UNORM => ASTC_4X4_UNORM,
        Format::ASTC_5X4_UNORM => ASTC_5X4_UNORM,
        Format::ASTC_5X5_UNORM => ASTC_5X5_UNORM,
        Format::ASTC_6X5_UNORM => ASTC_6X5_UNORM,
        Format::ASTC_6X6_UNORM => ASTC_6X6_UNORM,
        Format::ASTC_8X5_UNORM => ASTC_8X5_UNORM,
        Format::ASTC_8X6_UNORM => ASTC_8X6_UNORM,
        Format::ASTC_8X8_UNORM => ASTC_8X8_UNORM,
        Format::ASTC_10X5_UNORM => ASTC_10X5_UNORM,
        Format::ASTC_10X6_UNORM => ASTC_10X6_UNORM,
        Format::ASTC_10X8_UNORM => ASTC_10X8_UNORM,
        Format::ASTC_10X10_UNORM => ASTC_10X10_UNORM,
        Format::ASTC_12X10_UNORM => ASTC_12X10_UNORM,
        Format::ASTC_12X12_UNORM => ASTC_12X12_UNORM,

        // non-standard formats
        Format::BC3_UNORM_RXGB => BC3_UNORM_RXGB,
        Format::BC3_UNORM_NORMAL => BC3_UNORM_NORMAL,
    }
}

/// Decodes the image data of a surface from the given reader and writes it
/// to the given output buffer.
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
/// If the operation fails and returns an error **other** than an IO error,
/// the position of the reader remains unchanged.
///
/// ## Panics
///
/// This method will only panic in the given reader panics while reading.
pub fn decode(
    reader: &mut dyn Read,
    image: ImageViewMut,
    format: Format,
    options: &DecodeOptions,
) -> Result<(), DecodingError> {
    get_decoders(format).decode(reader, image, options)
}

/// Decodes a rectangle of the image data of a surface from the given reader
/// and writes it to the given output buffer.
///
/// ## Row pitch and the output buffer
///
/// The `row_pitch` parameter specifies the number of bytes between the start
/// of one row and the start of the next row in the output buffer.
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
/// If the operation fails and returns an error **other** than an IO error,
/// the position of the reader remains unchanged.
///
/// ## Panics
///
/// This method will only panic in the given reader panics while reading.
#[allow(clippy::too_many_arguments)]
pub fn decode_rect<R: Read + Seek>(
    reader: &mut R,
    output: &mut [u8],
    row_pitch: usize,
    color: ColorFormat,
    size: Size,
    rect: Rect,
    format: Format,
    options: &DecodeOptions,
) -> Result<(), DecodingError> {
    let reader = reader as &mut dyn ReadSeek;
    let decoders = get_decoders(format);
    decoders.decode_rect(color, reader, size, rect, output, row_pitch, options)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// The maximum amount of memory that the decoder is allowed to allocate.
    ///
    /// If the decoder needs to allocate more memory than this limit, it will
    /// return [`DecodeError::MemoryLimitExceeded`].
    ///
    /// While most decoders can make do with a few kilobytes of stack memory,
    /// some formats require a variable amount of memory depending on the size
    /// of the image. For example,
    /// [`NV12`](https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering#nv12)
    /// is a bi-planar format and decoding it generally requires reading the
    /// entire Y plane into memory. This can be a problem for large images.
    ///
    /// Default: 33 MiB
    ///
    /// (The default was chosen to be large enough to decode 4K `NV12`, `P016`,
    /// and `P010` images. All other formats require at most 256 KiB for 16K
    /// images.)
    pub memory_limit: usize,
}
impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            memory_limit: 33 * 1024 * 1024,
        }
    }
}
