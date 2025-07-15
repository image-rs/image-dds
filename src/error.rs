use std::num::NonZeroU32;

use crate::{
    header::{DxgiFormat, FourCC, Header},
    Format,
};

#[derive(Debug)]
#[non_exhaustive]
pub enum FormatError {
    UnsupportedDxgiFormat(DxgiFormat),
    UnsupportedFourCC(FourCC),
    UnsupportedPixelFormat,
}
impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::UnsupportedDxgiFormat(format) => {
                write!(f, "Unsupported DXGI format: {format:?}")
            }
            FormatError::UnsupportedFourCC(four_cc) => {
                write!(f, "Unsupported FourCC pixel format: {four_cc:?}")
            }
            FormatError::UnsupportedPixelFormat => {
                write!(f, "Unsupported pixel format in the DDS header")
            }
        }
    }
}
impl std::error::Error for FormatError {}

#[derive(Debug)]
#[non_exhaustive]
pub enum LayoutError {
    /// The decoder only supports up to 255 mipmaps.
    ///
    /// In practice, texture will have at most 32 mipmaps, so this limitation
    /// should only affect invalid/malicious files.
    TooManyMipMaps(u32),
    /// A volume/texture 3D without a depth.
    MissingDepth,
    /// The width, height, or depth of the texture is zero.
    ZeroDimension,
    /// The `array_size` field is too large.
    ///
    /// It either exceeds a user-defined limit or causes an overflow for cube map faces.
    ArraySizeTooBig(u32),
    /// The header of the DDS file describes a data section that is too large.
    ///
    /// I.e. it is possible for the header to describe a texture that requires >2^64
    /// bytes of memory.
    DataLayoutTooBig,
    /// The faces of a cube map must always be 2D textures.
    InvalidCubeMapFaces,
}
impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::TooManyMipMaps(mipmaps) => {
                write!(
                    f,
                    "Too many mipmaps ({mipmaps}), the maximum supported is 255"
                )
            }
            LayoutError::MissingDepth => {
                write!(f, "Missing depth for a texture 3D/volume")
            }
            LayoutError::ZeroDimension => {
                write!(f, "The width, height, or depth of the texture is zero")
            }
            LayoutError::ArraySizeTooBig(size) => {
                write!(f, "Array size {size} is too large")
            }
            LayoutError::DataLayoutTooBig => {
                write!(f, "Data layout described by the header is too large")
            }
            LayoutError::InvalidCubeMapFaces => {
                write!(f, "Cube map faces must be 2D textures")
            }
        }
    }
}
impl std::error::Error for LayoutError {}

#[derive(Debug)]
#[non_exhaustive]
pub enum DecodingError {
    /// When decoding a rectangle, the rectangle is out of bounds of the size
    /// of the image.
    RectOutOfBounds,
    /// Returned by [`Decoder::read_surface_rect`](crate::Decoder::read_surface_rect)
    /// and [`decode()`](crate::decode()) when the size of the image does not
    /// match the size of the rectangle.
    UnexpectedRectSize,

    /// Returned by [`Decoder::read_surface`](crate::Decoder::read_surface)
    /// when the user tries to decode a surface into an image that is not the
    /// same size as the surface.
    UnexpectedSurfaceSize,
    /// When decoding a volume texture, it is not allowed to skip mipmaps
    /// within a volume.
    ///
    /// See [`Decoder::skip_mipmaps`](crate::Decoder::skip_mipmaps) for more
    /// details.
    CannotSkipMipmapsInVolume,
    /// There are no further surfaces to decode.
    NoMoreSurfaces,
    /// This error is returned by [`Decoder::read_cube_map`](crate::Decoder::read_cube_map)
    /// when the user tries to read a DDS file that isn't a cube map.
    NotACubeMap,

    /// The decoder has exceeded its memory limit.
    MemoryLimitExceeded,

    Layout(LayoutError),
    Format(FormatError),
    Header(HeaderError),
    Io(std::io::Error),
}

impl std::fmt::Display for DecodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodingError::RectOutOfBounds => {
                write!(f, "Rectangle is out of bounds of the image size")
            }
            DecodingError::UnexpectedRectSize => {
                write!(f, "Unexpected rectangle size")
            }
            DecodingError::UnexpectedSurfaceSize => {
                write!(f, "Unexpected surface size")
            }
            DecodingError::CannotSkipMipmapsInVolume => {
                write!(f, "Cannot skip mipmaps within a volume texture")
            }
            DecodingError::NoMoreSurfaces => {
                write!(f, "No more surfaces to decode")
            }
            DecodingError::NotACubeMap => {
                write!(f, "The DDS file is not a cube map")
            }

            DecodingError::MemoryLimitExceeded => {
                write!(f, "Memory limit exceeded")
            }

            DecodingError::Layout(error) => write!(f, "{error}"),
            DecodingError::Format(error) => write!(f, "{error}"),
            DecodingError::Header(error) => write!(f, "Header error: {error}"),
            DecodingError::Io(error) => write!(f, "I/O error: {error}"),
        }
    }
}

impl From<LayoutError> for DecodingError {
    fn from(error: LayoutError) -> Self {
        DecodingError::Layout(error)
    }
}
impl From<FormatError> for DecodingError {
    fn from(error: FormatError) -> Self {
        DecodingError::Format(error)
    }
}
impl From<HeaderError> for DecodingError {
    fn from(error: HeaderError) -> Self {
        DecodingError::Header(error)
    }
}
impl From<std::io::Error> for DecodingError {
    fn from(error: std::io::Error) -> Self {
        DecodingError::Io(error)
    }
}

impl std::error::Error for DecodingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodingError::Layout(error) => Some(error),
            DecodingError::Format(error) => Some(error),
            DecodingError::Header(error) => Some(error),
            DecodingError::Io(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum HeaderError {
    InvalidMagicBytes([u8; 4]),
    InvalidHeaderSize(u32),
    InvalidPixelFormatSize(u32),
    InvalidRgbBitCount(u32),
    InvalidDxgiFormat(u32),
    InvalidResourceDimension(u32),
    InvalidAlphaMode(u32),
    InvalidArraySizeForTexture3D(u32),

    Io(std::io::Error),
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeaderError::InvalidMagicBytes(bytes) => {
                write!(
                    f,
                    "Invalid magic bytes {:?}, expected {:?} (ASCII: 'DDS ')",
                    bytes,
                    Header::MAGIC
                )
            }
            HeaderError::InvalidHeaderSize(size) => {
                write!(f, "Invalid DDS header size of {size}, expected 124")
            }
            HeaderError::InvalidPixelFormatSize(size) => {
                write!(
                    f,
                    "Invalid DDS header pixel format size of {size}, expected 32"
                )
            }
            HeaderError::InvalidRgbBitCount(count) => {
                write!(
                    f,
                    "Invalid DDS header pixel format rgb_bit_count of {count}, expected 8, 16, 24, or 32"
                )
            }
            HeaderError::InvalidDxgiFormat(format) => {
                write!(f, "Invalid DXGI format {format} in DX10 header extension")
            }
            HeaderError::InvalidResourceDimension(dimension) => {
                let label = match dimension {
                    0 => " (Unknown)",
                    1 => " (Buffer)",
                    2 => " (Texture1D)",
                    3 => " (Texture2D)",
                    4 => " (Texture3D)",
                    _ => "",
                };
                write!(
                    f,
                    "Invalid resource dimension {dimension}{label} in DX10 header extension"
                )
            }
            HeaderError::InvalidAlphaMode(mode) => {
                write!(f, "Invalid alpha mode {mode} in DX10 header extension")
            }
            HeaderError::InvalidArraySizeForTexture3D(array_size) => {
                write!(
                    f,
                    "Invalid array size {array_size} for a texture 3D in DX10 header extension"
                )
            }

            HeaderError::Io(error) => write!(f, "I/O error: {error}"),
        }
    }
}

impl From<std::io::Error> for HeaderError {
    fn from(error: std::io::Error) -> Self {
        HeaderError::Io(error)
    }
}
impl std::error::Error for HeaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HeaderError::Io(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum EncodingError {
    /// Returned by [`Encoder`](crate::Encoder) and [`encode()`](crate::encode()) when the format
    /// does not support encoding.
    UnsupportedFormat(Format),
    InvalidSize(NonZeroU32, NonZeroU32),

    /// Returned by [`Encoder`](crate::Encoder) when the user tries to write a surface
    /// with a size that is different from the size declared in the header.
    UnexpectedSurfaceSize,
    /// Returned by [`Encoder`](crate::Encoder) when the encoder has already written all
    /// surfaces declared in the header, but the user attempts to write
    /// additional surfaces.
    TooManySurfaces,
    /// Returned by [`Encoder::finish()`](crate::Encoder::finish()) when the encoder has not
    /// written all surfaces declared in the header.
    MissingSurfaces,

    Layout(LayoutError),
    Io(std::io::Error),
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EncodingError::UnsupportedFormat(format) => {
                write!(f, "Unsupported format for encoding: {format:?}")
            }
            EncodingError::InvalidSize(width, height) => {
                write!(f, "Size must be a multiple of {width}x{height}")
            }

            EncodingError::UnexpectedSurfaceSize => {
                write!(f, "Unexpected surface size")
            }
            EncodingError::TooManySurfaces => {
                write!(f, "Too many surfaces are attempted to written")
            }
            EncodingError::MissingSurfaces => write!(f, "Not enough surfaces have been written"),

            EncodingError::Layout(err) => write!(f, "Layout error: {err}"),
            EncodingError::Io(err) => write!(f, "IO error: {err}"),
        }
    }
}
impl std::error::Error for EncodingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EncodingError::Layout(err) => Some(err),
            EncodingError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<LayoutError> for EncodingError {
    fn from(err: LayoutError) -> Self {
        EncodingError::Layout(err)
    }
}
impl From<std::io::Error> for EncodingError {
    fn from(err: std::io::Error) -> Self {
        EncodingError::Io(err)
    }
}
