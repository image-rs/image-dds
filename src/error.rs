use crate::{ColorFormat, DxgiFormat, Format, FourCC, Header};

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
                write!(f, "DXGI format {:?} is not supported for decoding", format)
            }
            FormatError::UnsupportedFourCC(four_cc) => {
                write!(f, "Unsupported {:?} in DX10 header extension", four_cc)
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
pub enum DecodeError {
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
    /// I.e. it is possible for the header to describe a texture that requires
    /// >2^64 bytes of memory.
    DataLayoutTooBig,
    UnexpectedBufferSize {
        expected: usize,
    },

    /// When decoding a rectangle, the rectangle is out of bounds of the size
    /// of the image.
    RectOutOfBounds,
    /// When decoding a rectangle, the row pitch is too small.
    ///
    /// A row pitch must be at least `color.bytes_per_pixel() * rect.width` bytes.
    RowPitchTooSmall {
        required_minimum: usize,
    },
    /// When decoding a rectangle, the buffer is too small.
    ///
    /// A buffer much have at least `row_pitch * rect.height` bytes.
    RectBufferTooSmall {
        required_minimum: usize,
    },

    Format(FormatError),
    Header(HeaderError),
    Io(std::io::Error),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::TooManyMipMaps(mipmaps) => {
                write!(
                    f,
                    "Too many mipmaps ({}), the maximum supported is 32",
                    mipmaps
                )
            }
            DecodeError::MissingDepth => {
                write!(f, "Missing depth for a texture 3D/volume")
            }
            DecodeError::ZeroDimension => {
                write!(f, "The width, height, or depth of the texture is zero")
            }
            DecodeError::ArraySizeTooBig(size) => {
                write!(f, "Array size {} is too large", size)
            }
            DecodeError::DataLayoutTooBig => {
                write!(f, "Data layout described by the header is too large")
            }
            DecodeError::UnexpectedBufferSize { expected } => {
                write!(f, "Unexpected buffer size: expected {} bytes", expected)
            }

            DecodeError::RectOutOfBounds => {
                write!(f, "Rectangle is out of bounds of the image size")
            }
            DecodeError::RowPitchTooSmall { required_minimum } => {
                write!(
                    f,
                    "Row pitch too small: Must be at least `color.bytes_per_pixel() * rect.width` == {} bytes",
                    required_minimum
                )
            }
            DecodeError::RectBufferTooSmall { required_minimum } => {
                write!(
                    f,
                    "Buffer too small for rectangle: required at least {} bytes",
                    required_minimum
                )
            }

            DecodeError::Format(error) => write!(f, "{}", error),
            DecodeError::Header(error) => write!(f, "Header error: {}", error),
            DecodeError::Io(error) => write!(f, "I/O error: {}", error),
        }
    }
}

impl From<FormatError> for DecodeError {
    fn from(error: FormatError) -> Self {
        DecodeError::Format(error)
    }
}
impl From<HeaderError> for DecodeError {
    fn from(error: HeaderError) -> Self {
        DecodeError::Header(error)
    }
}
impl From<std::io::Error> for DecodeError {
    fn from(error: std::io::Error) -> Self {
        DecodeError::Io(error)
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::Format(error) => Some(error),
            DecodeError::Header(error) => Some(error),
            DecodeError::Io(error) => Some(error),
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
                write!(f, "Invalid DDS header size of {}, expected 124", size)
            }
            HeaderError::InvalidPixelFormatSize(size) => {
                write!(
                    f,
                    "Invalid DDS header pixel format size of {}, expected 32",
                    size
                )
            }
            HeaderError::InvalidDxgiFormat(format) => {
                write!(f, "Invalid DXGI format {} in DX10 header extension", format)
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
                    "Invalid resource dimension {}{} in DX10 header extension",
                    dimension, label
                )
            }
            HeaderError::InvalidAlphaMode(mode) => {
                write!(f, "Invalid alpha mode {} in DX10 header extension", mode)
            }
            HeaderError::InvalidArraySizeForTexture3D(array_size) => {
                write!(
                    f,
                    "Invalid array size {} for a texture 3D in DX10 header extension",
                    array_size
                )
            }

            HeaderError::Io(error) => write!(f, "I/O error: {}", error),
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
pub enum EncodeError {
    UnsupportedFormat(Format),
    InvalidLines,
    Io(std::io::Error),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EncodeError::UnsupportedFormat(format) => {
                write!(f, "Unsupported format: {:?}", format)
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

impl From<std::io::Error> for EncodeError {
    fn from(err: std::io::Error) -> Self {
        EncodeError::Io(err)
    }
}
