use std::{io::Write, num::NonZeroU8};

use crate::{EncodeError, Format, ImageView, SizeMultiple};

mod bc;
mod bc1;
mod bc4;
mod bcn_util;
mod bi_planar;
mod encoder;
mod sub_sampled;
mod uncompressed;

use bc::*;
use bi_planar::*;
pub(crate) use encoder::EncoderSet;
use sub_sampled::*;
use uncompressed::*;

pub(crate) const fn get_encoders(format: Format) -> Option<EncoderSet> {
    Some(match format {
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

        // ASTC formats
        Format::ASTC_4X4_UNORM
        | Format::ASTC_5X4_UNORM
        | Format::ASTC_5X5_UNORM
        | Format::ASTC_6X5_UNORM
        | Format::ASTC_6X6_UNORM
        | Format::ASTC_8X5_UNORM
        | Format::ASTC_8X6_UNORM
        | Format::ASTC_8X8_UNORM
        | Format::ASTC_10X5_UNORM
        | Format::ASTC_10X6_UNORM
        | Format::ASTC_10X8_UNORM
        | Format::ASTC_10X10_UNORM
        | Format::ASTC_12X10_UNORM
        | Format::ASTC_12X12_UNORM => return None,

        // non-standard formats
        Format::BC3_UNORM_RXGB => BC3_UNORM_RXGB,

        // unsupported formats
        Format::BC6H_UF16 | Format::BC6H_SF16 | Format::BC7_UNORM => return None,
    })
}

pub fn encode(
    writer: &mut dyn Write,
    image: ImageView,
    format: Format,
    options: &EncodeOptions,
) -> Result<(), EncodeError> {
    if let Some(encoders) = get_encoders(format) {
        encoders.encode(writer, image, options)
    } else {
        Err(EncodeError::UnsupportedFormat(format))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Whether to enable dithering for specific channels.
    ///
    /// The dithering algorithm depends on the format. Uncompressed formats use
    /// Floyd-Steinberg dithering, while block-compressed formats use a modified
    /// version of the algorithm to dithering within a block.
    ///
    /// Notes:
    /// 1. Dithering is not supported for high-precision uncompressed formats
    ///    (>= 16 bits per pixel). This option will be ignored for those formats.
    /// 2. YUV formats are not supported.
    ///
    /// Default: [`Dithering::None`]
    pub dithering: Dithering,
    /// The error metric for block compression formats.
    ///
    /// Default: [`ErrorMetric::Uniform`]
    pub error_metric: ErrorMetric,
    /// The compression quality.
    ///
    /// This option is naturally ignored for uncompressed formats.
    ///
    /// Default: [`CompressionQuality::Normal`]
    pub quality: CompressionQuality,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            dithering: Dithering::None,
            error_metric: ErrorMetric::Uniform,
            quality: CompressionQuality::Normal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dithering {
    /// Dithering is disabled for all channels.
    None = 0b00,
    /// Dithering is enabled for all channels (RGBA).
    ColorAndAlpha = 0b11,
    /// Dithering is enabled only for color channels (RGB).
    Color = 0b01,
    /// Dithering is enabled only for the alpha channel.
    Alpha = 0b10,
}
impl Dithering {
    pub const fn new(color: bool, alpha: bool) -> Self {
        match (color, alpha) {
            (true, true) => Dithering::ColorAndAlpha,
            (true, false) => Dithering::Color,
            (false, true) => Dithering::Alpha,
            (false, false) => Dithering::None,
        }
    }

    pub const fn color(self) -> bool {
        matches!(self, Dithering::ColorAndAlpha | Dithering::Color)
    }
    pub const fn alpha(self) -> bool {
        matches!(self, Dithering::ColorAndAlpha | Dithering::Alpha)
    }

    pub(crate) fn intersect(self, other: Self) -> Self {
        match (self, other) {
            (Dithering::None, _) | (_, Dithering::None) => Dithering::None,
            (Dithering::ColorAndAlpha, other) | (other, Dithering::ColorAndAlpha) => other,
            (Dithering::Color, Dithering::Alpha) | (Dithering::Alpha, Dithering::Color) => {
                Dithering::None
            }
            (Dithering::Color, Dithering::Color) => Dithering::Color,
            (Dithering::Alpha, Dithering::Alpha) => Dithering::Alpha,
        }
    }
}
impl Default for Dithering {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorMetric {
    Uniform,
    Perceptual,
}
impl Default for ErrorMetric {
    fn default() -> Self {
        Self::Uniform
    }
}

/// The level of trade-off between compression quality and speed.
///
/// - `Fast`: Fast compression speed.
/// - `Normal`: Balanced compression speed and quality.
/// - `High`: Production-level quality.
/// - `Unreasonable`: Reference-level quality. The encoder will try to
///   brute-force the best possible encoding for the image. This may take 100x
///   longer than `High` while only producing marginally better results. This
///   mode should only be used to create reference images.
///
/// Note that `Fast`, `Normal`, and `High` are not guaranteed to produce the
/// same results across different versions of this crate. They try to produce
/// the best possible quality within a certain time frame. As such, the results
/// will improve over time as the encoder is optimized.
///
/// Currently, the rough time budget for each quality level meant for encoding
/// 1024x1024 RGBA 8-bit image data on a single thread is:
///
/// - `Fast`: 100ms
/// - `Normal`: 500ms
/// - `High`: 5s
///
/// Encoding DDS images is embarrassingly parallel, so using multiple cores
/// should make encoding roughly 4-10x faster on normal consumer hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CompressionQuality {
    Fast,
    Normal,
    High,
    Unreasonable,
}
impl Default for CompressionQuality {
    fn default() -> Self {
        Self::Normal
    }
}

/// The preferred group size when splitting an image into chunks for parallel
/// encoding.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PreferredGroupSize {
    EntireImage,
    Group {
        fast: u8,
        high: u8,
        unreasonable: u8,
    },
}
impl PreferredGroupSize {
    pub const fn group(fast: u64, high: u64, unreasonable: u64) -> Self {
        const fn log2(x: u64) -> u8 {
            64 - x.leading_zeros() as u8
        }

        Self::Group {
            fast: log2(fast),
            high: log2(high),
            unreasonable: log2(unreasonable),
        }
    }

    pub const fn combine(&self, other: Self) -> Self {
        const fn u8_min(a: u8, b: u8) -> u8 {
            if a > b {
                b
            } else {
                a
            }
        }

        match (*self, other) {
            (PreferredGroupSize::EntireImage, _) => other,
            (_, PreferredGroupSize::EntireImage) => *self,
            (
                PreferredGroupSize::Group {
                    fast: a,
                    high: b,
                    unreasonable: c,
                },
                PreferredGroupSize::Group {
                    fast: x,
                    high: y,
                    unreasonable: z,
                },
            ) => PreferredGroupSize::Group {
                fast: u8_min(a, x),
                high: u8_min(b, y),
                unreasonable: u8_min(c, z),
            },
        }
    }

    pub fn get_group_pixels(&self, quality: CompressionQuality) -> u64 {
        match *self {
            PreferredGroupSize::EntireImage => u64::MAX,
            PreferredGroupSize::Group {
                fast,
                high,
                unreasonable,
            } => {
                let size_log2 = match quality {
                    CompressionQuality::Fast => fast,
                    CompressionQuality::Normal => ((fast as u16 + high as u16) / 2) as u8,
                    CompressionQuality::High => high,
                    CompressionQuality::Unreasonable => unreasonable,
                };

                1 << size_log2.min(63)
            }
        }
    }
}

/// Describes the extent of support for encoding a format.
#[derive(Debug, Clone, Copy)]
pub struct EncodingSupport {
    dithering: Dithering,
    split_height: Option<NonZeroU8>,
    local_dithering: bool,
    size_multiple: SizeMultiple,
    group_size: PreferredGroupSize,
}

impl EncodingSupport {
    /// Whether and what type of dithering is supported.
    pub const fn dithering(&self) -> Dithering {
        self.dithering
    }
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
    /// [`crate::SplitSurface`] will automatically split the image into chunks
    /// of the correct height, so this value is only relevant if you are
    /// implementing your own encoder/splitter.
    ///
    /// Note that most dithering will produce different (but not necessarily
    /// incorrect) results if the image is split into chunks. However, all BCn
    /// formats implement block-based local dithering, meaning that the dithering
    /// is the same whether the image is split or not. See
    /// [`EncodingSupport::local_dithering()`].
    pub const fn split_height(&self) -> Option<NonZeroU8> {
        self.split_height
    }
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
    /// `self.dithering() == Dithering::None` implies `self.local_dithering() == false`.
    pub const fn local_dithering(&self) -> bool {
        self.local_dithering
    }
    /// The size multiple of the encoded image.
    ///
    /// If the dimensions of the image are not multiples of this size, the
    /// encoder with return an error.
    pub const fn size_multiple(&self) -> SizeMultiple {
        self.size_multiple
    }

    pub(crate) fn group_size(&self) -> PreferredGroupSize {
        self.group_size
    }
}
