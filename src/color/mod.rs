use std::mem::size_of;

use crate::{cast, Size};

pub(crate) mod ch;
mod formats;
mod oklab;

pub(crate) use formats::*;
pub(crate) use oklab::*;

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
    /// The number of different channel variants.
    pub(crate) const COUNT: usize = 4;

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
    /// Returns the size of a single value of this precision in bytes.
    pub const fn size(&self) -> u8 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}

/// A color format with a specific number of channels and precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorFormat {
    pub channels: Channels,
    pub precision: Precision,
}
impl ColorFormat {
    pub const fn new(channels: Channels, precision: Precision) -> Self {
        Self {
            channels,
            precision,
        }
    }

    /// The number of bytes per pixel in the decoded surface/output buffer.
    ///
    /// This is calculated as simply `channels.count() * precision.size()`.
    pub const fn bytes_per_pixel(&self) -> u8 {
        self.channels.count() * self.precision.size()
    }

    /// The number of bytes per pixel in a decoded surface/output buffer.
    ///
    /// If the number if bytes is larger than `isize::MAX`, `None` is returned.
    pub fn buffer_size(&self, size: Size) -> Option<usize> {
        let bytes_per_pixel = self.bytes_per_pixel() as u64;
        let pixels = size.pixels();

        let bytes = pixels.checked_mul(bytes_per_pixel)?;
        if bytes < isize::MAX as u64 {
            Some(bytes as usize)
        } else {
            None
        }
    }

    /// Returns a unique key for this color format.
    ///
    /// The key is guaranteed to be less than 32.
    pub(crate) const fn key(&self) -> u8 {
        let key = self.precision as u8 * Channels::COUNT as u8 + self.channels as u8;
        debug_assert!(key < 32);
        key
    }

    pub const GRAYSCALE_U8: Self = Self::new(Channels::Grayscale, Precision::U8);
    pub const GRAYSCALE_U16: Self = Self::new(Channels::Grayscale, Precision::U16);
    pub const GRAYSCALE_F32: Self = Self::new(Channels::Grayscale, Precision::F32);

    pub const ALPHA_U8: Self = Self::new(Channels::Alpha, Precision::U8);
    pub const ALPHA_U16: Self = Self::new(Channels::Alpha, Precision::U16);
    pub const ALPHA_F32: Self = Self::new(Channels::Alpha, Precision::F32);

    pub const RGB_U8: Self = Self::new(Channels::Rgb, Precision::U8);
    pub const RGB_U16: Self = Self::new(Channels::Rgb, Precision::U16);
    pub const RGB_F32: Self = Self::new(Channels::Rgb, Precision::F32);

    pub const RGBA_U8: Self = Self::new(Channels::Rgba, Precision::U8);
    pub const RGBA_U16: Self = Self::new(Channels::Rgba, Precision::U16);
    pub const RGBA_F32: Self = Self::new(Channels::Rgba, Precision::F32);
}
impl core::fmt::Display for ColorFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} {:?}", self.channels, self.precision)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ColorFormatSet {
    data: u16,
}
impl ColorFormatSet {
    pub const U8: Self = Self::from_slice(&[
        ColorFormat::GRAYSCALE_U8,
        ColorFormat::ALPHA_U8,
        ColorFormat::RGB_U8,
        ColorFormat::RGBA_U8,
    ]);
    pub const U16: Self = Self::from_slice(&[
        ColorFormat::GRAYSCALE_U16,
        ColorFormat::ALPHA_U16,
        ColorFormat::RGB_U16,
        ColorFormat::RGBA_U16,
    ]);
    pub const F32: Self = Self::from_slice(&[
        ColorFormat::GRAYSCALE_F32,
        ColorFormat::ALPHA_F32,
        ColorFormat::RGB_F32,
        ColorFormat::RGBA_F32,
    ]);

    pub const EMPTY: Self = Self { data: 0 };
    pub const ALL: Self = Self {
        data: Self::U8.data | Self::U16.data | Self::F32.data,
    };

    pub const fn from_precision(precision: Precision) -> Self {
        match precision {
            Precision::U8 => Self::U8,
            Precision::U16 => Self::U16,
            Precision::F32 => Self::F32,
        }
    }
    pub const fn from_single(format: ColorFormat) -> Self {
        Self {
            data: 1 << format.key(),
        }
    }
    pub const fn from_slice(formats: &[ColorFormat]) -> Self {
        let mut data = 0;

        let mut i = 0;
        while i < formats.len() {
            data |= 1 << formats[i].key();
            i += 1;
        }

        Self { data }
    }

    pub const fn is_all(self) -> bool {
        self.data == Self::ALL.data
    }
    pub const fn len(self) -> u8 {
        self.data.count_ones() as u8
    }

    pub const fn contains(&self, format: ColorFormat) -> bool {
        self.data & (1 << format.key()) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            data: self.data | other.data,
        }
    }
    #[allow(dead_code)]
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            data: self.data & other.data,
        }
    }
}

pub(crate) trait Norm: Copy + Default {
    const ZERO: Self;
    const HALF: Self;
    const ONE: Self;
}
impl Norm for u8 {
    const ZERO: Self = 0;
    const HALF: Self = 128;
    const ONE: Self = u8::MAX;
}
impl Norm for u16 {
    const ZERO: Self = 0;
    const HALF: Self = 32768;
    const ONE: Self = u16::MAX;
}
impl Norm for f32 {
    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;
}

pub(crate) trait WithPrecision {
    const PRECISION: Precision;
}
impl WithPrecision for u8 {
    const PRECISION: Precision = Precision::U8;
}
impl WithPrecision for u16 {
    const PRECISION: Precision = Precision::U16;
}
impl WithPrecision for f32 {
    const PRECISION: Precision = Precision::F32;
}

pub(crate) fn convert_channels<Precision>(
    from: Channels,
    to: Channels,
    from_buffer: &[u8],
    to_buffer: &mut [u8],
) where
    Precision: Norm + cast::Castable + cast::IntoNeBytes,
    [Precision; 1]: cast::IntoNeBytes,
    [Precision; 3]: cast::IntoNeBytes,
    [Precision; 4]: cast::IntoNeBytes,
{
    fn map<From, To>(from_buffer: &[u8], to_buffer: &mut [u8], f: impl Fn(From) -> To)
    where
        From: cast::IntoNeBytes,
        To: cast::IntoNeBytes,
    {
        let from_chunked: &[From::Bytes] =
            cast::from_bytes(from_buffer).expect("invalid from buffer");
        let to_chunked: &mut [To::Bytes] =
            cast::from_bytes_mut(to_buffer).expect("invalid to buffer");
        debug_assert!(from_chunked.len() == to_chunked.len());

        for (from, to) in from_chunked.iter().zip(to_chunked) {
            *to = f(From::from_ne_bytes(*from)).into_ne_bytes();
        }
    }

    fn fill<Precision: cast::Castable + Copy + cast::IntoNeBytes>(
        to_buffer: &mut [u8],
        value: Precision,
    ) {
        let decoded: &mut [Precision::Bytes] =
            cast::from_bytes_mut(to_buffer).expect("invalid to buffer");

        decoded.fill(value.into_ne_bytes());
    }

    use ch::*;
    use Channels::*;

    debug_assert!(from_buffer.len() % (size_of::<Precision>() * from.count() as usize) == 0);
    debug_assert!(to_buffer.len() % (size_of::<Precision>() * to.count() as usize) == 0);
    debug_assert_eq!(
        from_buffer.len() / from.count() as usize,
        to_buffer.len() / to.count() as usize
    );

    match (from, to) {
        (Grayscale, Grayscale) | (Alpha, Alpha) | (Rgb, Rgb) | (Rgba, Rgba) => {
            to_buffer.copy_from_slice(from_buffer);
        }

        (Grayscale, Alpha) | (Rgb, Alpha) => fill(to_buffer, Precision::ONE),
        (Alpha, Grayscale) | (Alpha, Rgb) => fill(to_buffer, Precision::ZERO),

        (Grayscale, Rgb) => map(from_buffer, to_buffer, grayscale_to_rgb::<Precision>),
        (Grayscale, Rgba) => map(from_buffer, to_buffer, grayscale_to_rgba::<Precision>),
        (Alpha, Rgba) => map(from_buffer, to_buffer, alpha_to_rgba::<Precision>),
        (Rgb, Grayscale) => map(from_buffer, to_buffer, rgb_to_grayscale::<Precision>),
        (Rgb, Rgba) => map(from_buffer, to_buffer, rgb_to_rgba::<Precision>),
        (Rgba, Grayscale) => map(from_buffer, to_buffer, rgba_to_grayscale::<Precision>),
        (Rgba, Alpha) => map(from_buffer, to_buffer, rgba_to_alpha::<Precision>),
        (Rgba, Rgb) => map(from_buffer, to_buffer, rgba_to_rgb::<Precision>),
    }
}
pub(crate) fn convert_channels_for(
    from: ColorFormat,
    to: Channels,
    from_buffer: &[u8],
    to_buffer: &mut [u8],
) {
    match from.precision {
        Precision::U8 => convert_channels::<u8>(from.channels, to, from_buffer, to_buffer),
        Precision::U16 => convert_channels::<u16>(from.channels, to, from_buffer, to_buffer),
        Precision::F32 => convert_channels::<f32>(from.channels, to, from_buffer, to_buffer),
    }
}

pub(crate) fn as_rgba_f32<'a>(
    from: ColorFormat,
    from_buffer: &'a [u8],
    to_buffer: &'a mut [[f32; 4]],
) -> &'a [[f32; 4]] {
    if from == ColorFormat::RGBA_F32 {
        if let Some(slice) = cast::from_bytes(from_buffer) {
            return slice;
        }
    }

    convert_to_rgba_f32(from, from_buffer, to_buffer);
    to_buffer
}
pub(crate) fn convert_to_rgba_f32(
    from: ColorFormat,
    from_buffer: &[u8],
    to_buffer: &mut [[f32; 4]],
) {
    let channels = from.channels;
    let precision = from.precision;

    debug_assert!(from_buffer.len() % (precision.size() as usize * channels.count() as usize) == 0);
    debug_assert_eq!(
        from_buffer.len() / (precision.size() as usize * channels.count() as usize),
        to_buffer.len()
    );

    match precision {
        // Use exact conversions. This function is mainly used by encoders,
        // where performance isn't as important and exactness makes testing a
        // lot easier.
        Precision::U8 => convert_t_to_rgba_f32(channels, from_buffer, to_buffer, n8::f32),
        Precision::U16 => convert_t_to_rgba_f32(channels, from_buffer, to_buffer, n16::f32),
        Precision::F32 => {
            // since the precision is already f32, we just need to convert
            // channels
            convert_channels::<f32>(
                channels,
                Channels::Rgba,
                from_buffer,
                cast::as_bytes_mut(to_buffer),
            );
        }
    };
}
fn convert_t_to_rgba_f32<T>(
    from: Channels,
    from_buffer: &[u8],
    to_buffer: &mut [[f32; 4]],
    to_f32: impl (Fn(T) -> f32) + Copy,
) where
    T: cast::Castable + cast::IntoNeBytes,
{
    fn map<const C: usize, T>(
        from_buffer: &[u8],
        to_buffer: &mut [[f32; 4]],
        to_f32: impl Fn(T) -> f32,
        f: impl Fn([f32; C]) -> [f32; 4],
    ) where
        T: cast::Castable + cast::IntoNeBytes,
    {
        let from_chunked: &[[T::Bytes; C]] =
            cast::from_bytes(from_buffer).expect("invalid from buffer");
        debug_assert!(from_chunked.len() == to_buffer.len());

        for (from, to) in from_chunked.iter().zip(to_buffer) {
            *to = f(from.map(|c| to_f32(T::from_ne_bytes(c))));
        }
    }

    use ch::*;
    use Channels::*;

    match from {
        Grayscale => map(from_buffer, to_buffer, to_f32, grayscale_to_rgba),
        Alpha => map(from_buffer, to_buffer, to_f32, alpha_to_rgba),
        Rgb => map(from_buffer, to_buffer, to_f32, rgb_to_rgba),
        Rgba => map(from_buffer, to_buffer, to_f32, |pixel| pixel),
    }
}
