use std::io::Read;

use crate::{detect, util::div_ceil, DecodeError, DxgiFormat, FourCC, Header};

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
    pub(crate) const VARIANTS: u32 = 4;

    /// Returns the number of channels.
    pub const fn count(&self) -> u8 {
        match self {
            Self::Grayscale | Self::Alpha => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
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
    pub(crate) const VARIANTS: u32 = 3;

    /// Returns the size of a single value of this precision in bytes.
    pub const fn size(&self) -> u8 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
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
    BC1_ALPHA_UNORM,
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
    pub fn from_dxgi(dxgi: DxgiFormat) -> Option<SupportedFormat> {
        detect::dxgi_format_to_supported(dxgi)
    }
    /// Returns the format of a surface from a FourCC code.
    ///
    /// `None` if the FourCC code is not supported for decoding.
    pub fn from_four_cc(four_cc: FourCC) -> Option<SupportedFormat> {
        detect::four_cc_to_supported(four_cc)
    }

    /// The number and type of (color) channels in the surface.
    ///
    /// If the channels of a format cannot be accurately described by
    /// [`Channels`], the next larger type is used. For example, a format with
    /// only R and G channels will be described as [`Channels::Rgb`].
    pub fn channels(&self) -> Channels {
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
    pub fn precision(&self) -> Precision {
        decoders::get_decoders(*self).main().precision
    }

    /// A list of all channels this formats supports for decoding.
    ///
    /// This list is guaranteed to be without duplicates and to contain
    /// `self.channels()`.
    pub fn supported_channels(&self) -> &'static [Channels] {
        decoders::get_decoders(*self).supported_channels
    }
    /// A list of all precisions this formats supports for decoding.
    ///
    /// This list is guaranteed to be without duplicates and to contain
    /// `self.precision()`.
    pub fn supported_precisions(&self) -> &'static [Precision] {
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
        if let Some(decoder) = decoders
            .iter()
            .find(|d| d.channels == channels && d.precision == precision)
        {
            size.check_buffer_len(channels, precision, output)?;
            (decoder.decode_fn)(reader, size, output)
        } else {
            Err(DecodeError::UnsupportedColorTypePrecision(
                *self, channels, precision,
            ))
        }
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
            // Casting [u16] to [u8] cannot panic
            bytemuck::cast_slice_mut(output),
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
            // Casting [f32] to [u8] cannot panic
            bytemuck::cast_slice_mut(output),
        )
    }

    /// Returns the number of bytes required to store a surface of the given dimensions.
    ///
    /// If the number of bytes overflows a `usize`, `None` is returned.
    pub(crate) fn get_surface_bytes(&self, width: u32, height: u32) -> Option<u64> {
        // this cannot overflow
        let pixels = width as u64 * height as u64;

        match self {
            // 1 bytes per pixel
            Self::R8_UNORM | Self::R8_SNORM | Self::A8_UNORM => Some(pixels),
            // 2 bytes per pixel
            Self::B5G6R5_UNORM
            | Self::B5G5R5A1_UNORM
            | Self::B4G4R4A4_UNORM
            | Self::R8G8_UNORM
            | Self::R8G8_SNORM
            | Self::R16_UNORM
            | Self::R16_SNORM
            | Self::R16_FLOAT => pixels.checked_mul(2),
            // 3 bytes per pixel
            Self::R8G8B8_UNORM | Self::B8G8R8_UNORM => pixels.checked_mul(3),
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
            | Self::R10G10B10_XR_BIAS_A2_UNORM => pixels.checked_mul(4),
            // 8 bytes per pixel
            Self::R16G16B16A16_UNORM
            | Self::R16G16B16A16_SNORM
            | Self::R16G16B16A16_FLOAT
            | Self::R32G32_FLOAT => pixels.checked_mul(8),
            // 12 bytes per pixel
            Self::R32G32B32_FLOAT => pixels.checked_mul(12),
            // 16 bytes per pixel
            Self::R32G32B32A32_FLOAT => pixels.checked_mul(16),

            // sub-sampled formats
            Self::R8G8_B8G8_UNORM | Self::G8R8_G8B8_UNORM => {
                // 4 bytes per one 2x1 block
                let blocks_x = div_ceil(width, 2);
                let blocks_y = height;
                let blocks = u64::checked_mul(blocks_x as u64, blocks_y as u64)?;
                blocks.checked_mul(4)
            }

            // block compression formats
            Self::BC1_ALPHA_UNORM | Self::BC4_UNORM | Self::BC4_SNORM => {
                // 8 bytes per one 4x4 block
                let blocks_x = div_ceil(width, 4);
                let blocks_y = div_ceil(height, 4);
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
                let blocks_x = div_ceil(width, 4);
                let blocks_y = div_ceil(height, 4);
                let blocks = u64::checked_mul(blocks_x as u64, blocks_y as u64)?;
                blocks.checked_mul(16)
            }
        }
    }
}

pub struct Size {
    pub width: u32,
    pub height: u32,
}
impl Size {
    fn check_buffer_len(
        &self,
        channels: Channels,
        precision: Precision,
        buf: &[u8],
    ) -> Result<(), DecodeError> {
        // overflow isn't possible here
        let bytes_per_pixel = channels.count() as usize * precision.size() as usize;
        // saturate to usize::MAX on overflow
        let required_bytes = usize::saturating_mul(self.width as usize, self.height as usize)
            .saturating_mul(bytes_per_pixel);

        if buf.len() != required_bytes {
            Err(DecodeError::UnexpectedBufferSize {
                expected: required_bytes,
                actual: buf.len(),
            })
        } else {
            Ok(())
        }
    }
}
impl From<(u32, u32)> for Size {
    fn from((width, height): (u32, u32)) -> Self {
        Self { width, height }
    }
}

mod decoders {
    use std::io::Read;

    use crate::DecodeError;

    use super::{Channels, Precision, Size, SupportedFormat};

    type DecodeFn =
        fn(reader: &mut dyn Read, size: Size, output: &mut [u8]) -> Result<(), DecodeError>;
    pub struct Decoder {
        pub channels: Channels,
        pub precision: Precision,
        pub decode_fn: DecodeFn,
    }
    impl Decoder {
        pub const fn new(channels: Channels, precision: Precision, decode_fn: DecodeFn) -> Self {
            Self {
                channels,
                precision,
                decode_fn,
            }
        }
    }

    fn noop_decode(
        _reader: &mut dyn Read,
        _size: Size,
        _output: &mut [u8],
    ) -> Result<(), DecodeError> {
        Ok(())
    }

    pub struct DecodersInfo {
        pub decoders: &'static [Decoder],
        pub supported_channels: &'static [Channels],
        pub supported_precisions: &'static [Precision],
    }
    impl DecodersInfo {
        pub const fn main(&self) -> &'static Decoder {
            &self.decoders[0]
        }
        const fn verify(&self) {
            // 1. The list must be non-empty.
            assert!(!self.decoders.is_empty());

            // 2. No color channel-precision combination may be repeated.
            {
                let mut bitset: u32 = 0;
                let mut i = 0;
                while i < self.decoders.len() {
                    let decoder = &self.decoders[i];

                    let key =
                        decoder.channels as u32 * Precision::VARIANTS + decoder.precision as u32;
                    assert!(key < 32);

                    let bit_mask = 1 << key;
                    if bitset & bit_mask != 0 {
                        panic!("Repeated color channel-precision combination");
                    }
                    bitset |= bit_mask;

                    i += 1;
                }
            }

            // 3. Color channel-precision combination must be exhaustive.
            let mut channels_bitset: u32 = 0;
            let mut precision_bitset: u32 = 0;
            {
                let mut i = 0;
                while i < self.decoders.len() {
                    let decoder = &self.decoders[i];

                    channels_bitset |= 1 << decoder.channels as u32;
                    precision_bitset |= 1 << decoder.precision as u32;

                    i += 1;
                }

                let channels_count = channels_bitset.count_ones();
                let precision_count = precision_bitset.count_ones();
                // the expected number of decoders IF all combinations are present
                let expected = channels_count * precision_count;
                if self.decoders.len() != expected as usize {
                    panic!("Missing color channel-precision combination");
                }
            }

            // Supported channels must match decoder channels
            {
                let mut supported_bitset: u32 = 0;
                let mut i = 0;
                while i < self.supported_channels.len() {
                    let color = self.supported_channels[i] as u32;
                    supported_bitset |= 1 << color;
                    i += 1;
                }

                if supported_bitset != channels_bitset {
                    panic!("Supported channels do not match decoder channels");
                }
                if supported_bitset.count_ones() != self.supported_channels.len() as u32 {
                    panic!("Supported channels should contain no duplicates");
                }
            }

            // Supported precisions must match decoder precisions
            {
                let mut supported_bitset: u32 = 0;
                let mut i = 0;
                while i < self.supported_precisions.len() {
                    let precision = self.supported_precisions[i] as u32;
                    supported_bitset |= 1 << precision;
                    i += 1;
                }

                if supported_bitset != precision_bitset {
                    panic!("Supported precisions do not match decoder precisions");
                }
                if supported_bitset.count_ones() != self.supported_precisions.len() as u32 {
                    panic!("Supported precisions should contain no duplicates");
                }
            }
        }
    }
    pub const fn get_decoders(format: SupportedFormat) -> DecodersInfo {
        use Channels::*;
        use Precision::*;

        // INFO
        //
        // Decoder lists have a few requirements that are checked at compile
        // time. The requirements are:
        //
        // 1. There must be at least one decoder.
        // 2. No color channels-precision combination may be repeated.
        // 3. Color channels-precision combination must be exhaustive. This means
        //    that if there exists a combination with channels C and precision P,
        //    there must be a decoder for combination (C,P).
        //
        // Additionally, the first decoder will be treated as the MAIN decoder.
        // This means that its channels and precision will be used as the
        // default channels and precision for the format.

        /// A helper macro to make it easier to define a const array of decoders.
        ///
        /// This macro can be used in two ways:
        ///
        /// For a single decoder:
        ///
        /// ```rust
        /// decoders!(Rgb, U8, my_decode_fn)
        /// ```
        ///
        /// For multiple decoders:
        ///
        /// ```rust
        /// decoders!(
        ///     [Rgb, Rgba], // supported colors
        ///     [U8],        // supported precisions
        ///     [(Rgba, U8, decode_rgba_fn), (Rgb, U8, decode_rgb_fn)],
        /// )
        /// ```
        macro_rules! decoders {
            ([$($ct:expr),*], [$($cp:expr),*], [$(($c:expr, $p:expr, $d:expr)),* $(,)? ] $(,)?) => {{
                const DECODERS: &[Decoder] = &[$(Decoder::new($c, $p, $d)),*];

                const INFO: DecodersInfo = {
                    let info = DecodersInfo {
                        decoders: DECODERS,
                        supported_channels: &[$($ct),*],
                        supported_precisions: &[$($cp),*],
                    };
                    info.verify();
                    info
                };

                INFO
            }};
            ($c:ident, $p:ident, $d:expr) => {{
                const INFO: DecodersInfo = {
                    const DECODER: Decoder = Decoder::new($c, $p, $d);

                    let info = DecodersInfo {
                        decoders: &[DECODER],
                        supported_channels: &[$c],
                        supported_precisions: &[$p],
                    };
                    info.verify();
                    info
                };

                INFO
            }};
        }

        match format {
            // uncompressed formats
            SupportedFormat::R8G8B8_UNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::B8G8R8_UNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::R8G8B8A8_UNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::R8G8B8A8_SNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::B8G8R8A8_UNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::B8G8R8X8_UNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::B5G6R5_UNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::B5G5R5A1_UNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::B4G4R4A4_UNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::R8_SNORM => decoders!(Grayscale, U8, noop_decode),
            SupportedFormat::R8_UNORM => decoders!(Grayscale, U8, noop_decode),
            SupportedFormat::R8G8_UNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::R8G8_SNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::A8_UNORM => decoders!(Alpha, U8, noop_decode),
            SupportedFormat::R16_UNORM => decoders!(Grayscale, U16, noop_decode),
            SupportedFormat::R16_SNORM => decoders!(Grayscale, U16, noop_decode),
            SupportedFormat::R16G16_UNORM => decoders!(Rgb, U16, noop_decode),
            SupportedFormat::R16G16_SNORM => decoders!(Rgb, U16, noop_decode),
            SupportedFormat::R16G16B16A16_UNORM => decoders!(Rgba, U16, noop_decode),
            SupportedFormat::R16G16B16A16_SNORM => decoders!(Rgba, U16, noop_decode),
            SupportedFormat::R10G10B10A2_UNORM => decoders!(Rgba, U16, noop_decode),
            SupportedFormat::R11G11B10_FLOAT => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::R9G9B9E5_SHAREDEXP => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::R16_FLOAT => decoders!(Grayscale, F32, noop_decode),
            SupportedFormat::R16G16_FLOAT => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::R16G16B16A16_FLOAT => decoders!(Rgba, F32, noop_decode),
            SupportedFormat::R32_FLOAT => decoders!(Grayscale, F32, noop_decode),
            SupportedFormat::R32G32_FLOAT => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::R32G32B32_FLOAT => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::R32G32B32A32_FLOAT => decoders!(Rgba, F32, noop_decode),
            SupportedFormat::R10G10B10_XR_BIAS_A2_UNORM => decoders!(Rgba, F32, noop_decode),

            // sub-sampled formats
            SupportedFormat::R8G8_B8G8_UNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::G8R8_G8B8_UNORM => decoders!(Rgb, U8, noop_decode),

            // block compression formats
            SupportedFormat::BC1_ALPHA_UNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::BC2_UNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::BC3_UNORM => decoders!(Rgba, U8, noop_decode),
            SupportedFormat::BC4_UNORM => decoders!(Grayscale, U8, noop_decode),
            SupportedFormat::BC4_SNORM => decoders!(Grayscale, U8, noop_decode),
            SupportedFormat::BC5_UNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::BC5_SNORM => decoders!(Rgb, U8, noop_decode),
            SupportedFormat::BC6H_UF16 => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::BC6H_SF16 => decoders!(Rgb, F32, noop_decode),
            SupportedFormat::BC7_UNORM => decoders!(
                [Rgb, Rgba],
                [U8],
                [(Rgba, U8, noop_decode), (Rgb, U8, noop_decode)],
            ),
        }
    }
}
