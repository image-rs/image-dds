use std::{
    io::Write,
    num::{NonZeroU32, NonZeroU8},
};

use crate::{EncodingError, Format, ImageView, Progress, Size};

mod bc;
mod bc1;
mod bc4;
mod bc7;
mod bcn_util;
mod bi_planar;
mod encoder;
mod sub_sampled;
mod uncompressed;
mod write_util;

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

        // subsampled formats
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
        Format::BC7_UNORM => BC7_UNORM,

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
        Format::BC3_UNORM_NORMAL => BC3_UNORM_NORMAL,

        // unsupported formats
        Format::BC6H_UF16 | Format::BC6H_SF16 => return None,
    })
}

/// Encodes a single surfaces in the given format and writes the encoded pixel
/// data to the given writer.
///
/// If an error is returned (or the operation is cancelled), the writer may be
/// in an inconsistent state. Some, all, or none of the pixel data may have been
/// written.
///
/// If `progress` is cancelled before this function returns,
/// [`EncodingError::Cancelled`] is returned (unless another error occurs first).
/// Passing `None` as `progress` is equivalent to [`Progress::none()`].
///
/// ## Panics
///
/// Panics if:
///
/// 1. `writer` panics during writes.
/// 2. The underlying reporter function of `progress` panics (if given).
/// 3. An allocation fails.
pub fn encode(
    writer: &mut dyn Write,
    image: ImageView,
    format: Format,
    progress: Option<&mut Progress>,
    options: &EncodeOptions,
) -> Result<(), EncodingError> {
    let mut no_reporting = Progress::none();
    let progress = progress.unwrap_or(&mut no_reporting);

    // ending quickly if cancelled is a good property to have
    progress.check_cancelled()?;

    #[cfg(feature = "rayon")]
    if options.parallel {
        return encode_parallel(writer, image, format, progress, options);
    }

    let encoders = get_encoders(format).ok_or(EncodingError::UnsupportedFormat(format))?;
    encoders.encode(writer, image, progress, options)?;

    // error if the operation was cancelled
    progress.check_cancelled()
}

#[cfg(feature = "rayon")]
fn encode_parallel(
    writer: &mut dyn Write,
    image: ImageView,
    format: Format,
    mut progress: &mut Progress,
    options: &EncodeOptions,
) -> Result<(), EncodingError> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::{ParallelProgress, PixelInfo, SplitView};

    let mut options = options.clone();
    // don't cause an infinite loop
    options.parallel = false;

    let split = SplitView::new(image, format, &options);

    // optimization for single fragment
    if let Some(single) = split.single() {
        return encode(writer, single, format, Some(progress), &options);
    }

    // Prepare the parallel progress reporter. The +1 ensures that 100% is
    // reported only after everything written to disk.
    // Note: Parallel progress reporting is not supported for single-threaded
    // reporters. They will do nothing.
    let parallel_progress = ParallelProgress::new(&mut progress, image.height() as u64 + 1);

    let pixel_info = PixelInfo::from(format);
    let result: Result<Vec<Vec<u8>>, EncodingError> = (0..split.len())
        .into_par_iter()
        .map(|fragment_index| -> Result<Vec<u8>, EncodingError> {
            let fragment = split.get(fragment_index).expect("invalid fragment index");
            parallel_progress.check_cancelled()?;

            // allocate exactly the right amount of memory
            let bytes: usize = pixel_info
                .surface_bytes(fragment.size)
                .unwrap_or(u64::MAX)
                .try_into()
                .expect("too many bytes");
            let mut buffer: Vec<u8> = Vec::with_capacity(bytes);

            // Encode the fragment without any progress reporting or cancellation.
            // Fragments are typically very small and encoded quickly, so
            // reporting progress or checking for cancellation is not necessary.
            encode(&mut buffer, fragment, format, None, &options)?;

            parallel_progress.check_cancelled()?;
            parallel_progress.submit(fragment.height() as u64);

            debug_assert_eq!(buffer.len(), bytes);
            Ok(buffer)
        })
        .collect();

    let encoded_fragments = result?;
    for fragment in encoded_fragments {
        progress.check_cancelled()?;
        writer.write_all(&fragment)?;
    }

    // report completion
    progress.checked_report(1.0)
}

/// Options for encoding images.
///
/// ## See also
///
/// - [`encode`]
/// - [`Encoder::options`](crate::Encoder::options)
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
    /// For a full list of formats and their support for dithering, see the
    /// [`Format` documentation](crate::Format).
    ///
    /// Default: [`Dithering::None`]
    pub dithering: Dithering,
    /// The error metric for block compression formats.
    ///
    /// This is currently only supported for BC1-3. All other formats will
    /// ignore this option and assume [`ErrorMetric::Uniform`].
    ///
    /// Default: [`ErrorMetric::Uniform`]
    pub error_metric: ErrorMetric,
    /// The compression quality.
    ///
    /// This option is naturally ignored for uncompressed formats.
    ///
    /// Default: [`CompressionQuality::Normal`]
    pub quality: CompressionQuality,
    /// Whether to use rayon for parallel encoding.
    ///
    /// Encoding certain formats can be very computationally intensive.
    /// Using rayon to parallelize the encoding can significantly speed up the
    /// encoding process.
    ///
    /// Even if this option is enabled, the encoder may still only use a single
    /// thread if either:
    ///
    /// 1. The `rayon` feature is not enabled.
    /// 2. The format does not support parallel encoding.
    /// 3. The image is small enough that the overhead of parallelization
    ///    outweighs the benefits.
    ///
    /// Default: `true`
    pub parallel: bool,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            dithering: Dithering::None,
            error_metric: ErrorMetric::Uniform,
            quality: CompressionQuality::Normal,
            parallel: true,
        }
    }
}

/// Specifies whether dithering is enabled/supported for color and/or alpha
/// channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Dithering {
    /// Dithering is disabled for all channels.
    #[default]
    None = 0b00,
    /// Dithering is enabled for all channels (RGBA).
    ColorAndAlpha = 0b11,
    /// Dithering is enabled only for color channels (RGB).
    Color = 0b01,
    /// Dithering is enabled only for the alpha channel.
    Alpha = 0b10,
}
impl Dithering {
    /// Creates a new `Dithering` given the channels it is enabled for.
    pub const fn new(color: bool, alpha: bool) -> Self {
        match (color, alpha) {
            (true, true) => Dithering::ColorAndAlpha,
            (true, false) => Dithering::Color,
            (false, true) => Dithering::Alpha,
            (false, false) => Dithering::None,
        }
    }

    /// Whether dithering is enabled for color channels.
    pub const fn color(self) -> bool {
        matches!(self, Dithering::ColorAndAlpha | Dithering::Color)
    }
    /// Whether dithering is enabled for the alpha channel.
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

/// The error metric encoders use to optimize compressed formats.
///
/// Not all formats support all error metrics. See
/// [`EncodeOptions::error_metric`] for more details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ErrorMetric {
    /// The error of all channels is weighted equally.
    #[default]
    Uniform,
    /// Color is assumed to be sRGB, and the error is calculated in a
    /// perceptually uniform color space.
    ///
    /// [Oklab](https://en.wikipedia.org/wiki/Oklab_color_space) is currently
    /// used as the perceptually uniform color space.
    Perceptual,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum CompressionQuality {
    Fast,
    #[default]
    Normal,
    High,
    Unreasonable,
}

/// The preferred fragment size (=number of pixel in a fragment) when splitting
/// an image into chunks for parallel encoding.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PreferredFragmentSize {
    /// Splitting the image is not preferred. The whole image should be
    /// encoded as a single fragment.
    EntireImage,
    Fragment {
        fast: u8,
        high: u8,
        unreasonable: u8,
    },
}
impl PreferredFragmentSize {
    pub const fn new(fast: u64, high: u64, unreasonable: u64) -> Self {
        const fn log2(x: u64) -> u8 {
            debug_assert!(x != 0);
            debug_assert!(x.is_power_of_two());

            64 - x.leading_zeros() as u8 - 1
        }

        Self::Fragment {
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
            (PreferredFragmentSize::EntireImage, _) => other,
            (_, PreferredFragmentSize::EntireImage) => *self,
            (
                PreferredFragmentSize::Fragment {
                    fast: a,
                    high: b,
                    unreasonable: c,
                },
                PreferredFragmentSize::Fragment {
                    fast: x,
                    high: y,
                    unreasonable: z,
                },
            ) => PreferredFragmentSize::Fragment {
                fast: u8_min(a, x),
                high: u8_min(b, y),
                unreasonable: u8_min(c, z),
            },
        }
    }

    /// Returns the preferred number of pixels in a fragment for the given
    /// compression quality.
    pub fn get_preferred(&self, quality: CompressionQuality) -> u64 {
        match *self {
            PreferredFragmentSize::EntireImage => u64::MAX,
            PreferredFragmentSize::Fragment {
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

pub(crate) type SizeMultiple = (NonZeroU8, NonZeroU8);
const SIZE_MUL_2X2: SizeMultiple = {
    let Some(two) = NonZeroU8::new(2) else {
        unreachable!()
    };
    (two, two)
};

/// Describes the extent of support for encoding a format.
#[derive(Debug, Clone, Copy)]
pub struct EncodingSupport {
    dithering: Dithering,
    split_height: Option<NonZeroU8>,
    local_dithering: bool,
    size_multiple: Option<SizeMultiple>,
    pub(crate) fragment_size: PreferredFragmentSize,
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
    /// [`SplitView`](crate::SplitView) will automatically split the image into chunks
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
    /// The size multiple of the encoded image, if any.
    ///
    /// Some formats require the image to be a multiple of a certain size.
    /// E.g. `NV12` requires the image to be a multiple of 2x2 pixels, meaning that
    /// the width and height of the image must be even.
    ///
    /// Formats that can encode images of any size will return `None`.
    ///
    /// If `Some` is returned, the width and height of the returned value are
    /// both non-zero, and at least one of them is greater than 1.
    ///
    /// Use [`EncodingSupport::supports_size()`] to check if a given image size
    /// is supported by the format.
    ///
    /// Note: Size requirements also apply to mipmaps. This means that formats
    /// with size requirements do **not** support full mipmap chains, since full
    /// chains always end with a 1x1 mip.
    ///
    /// # Example
    ///
    /// ```
    /// # use dds::*;
    /// # use std::num::NonZeroU32;
    /// let format = Format::NV12;
    /// let encoding = format.encoding_support().unwrap();
    ///
    /// assert_eq!(
    ///     encoding.size_multiple(),
    ///     Some((NonZeroU32::new(2).unwrap(), NonZeroU32::new(2).unwrap()))
    /// );
    /// assert_eq!(encoding.supports_size(Size::new(2, 2)), true);
    /// assert_eq!(encoding.supports_size(Size::new(3, 2)), false);
    /// ```
    pub const fn size_multiple(&self) -> Option<(NonZeroU32, NonZeroU32)> {
        if let Some((w, h)) = self.size_multiple {
            let (Some(w), Some(h)) = (
                NonZeroU32::new(w.get() as u32),
                NonZeroU32::new(h.get() as u32),
            ) else {
                unreachable!()
            };
            Some((w, h))
        } else {
            None
        }
    }
    /// Whether the given image size is supported for encoding.
    ///
    /// This will always return `true` if [`EncodingSupport::size_multiple()`]
    /// is `None`.
    pub const fn supports_size(&self, size: Size) -> bool {
        if let Some((w, h)) = self.size_multiple {
            size.width % w.get() as u32 == 0 && size.height % h.get() as u32 == 0
        } else {
            true
        }
    }
}
