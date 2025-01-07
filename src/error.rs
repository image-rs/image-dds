use crate::{Channels, DxgiFormat, FourCC, Precision, SupportedFormat};

#[derive(Debug)]
#[non_exhaustive]
pub enum DecodeError {
    UnsupportedDxgiFormat(DxgiFormat),
    UnsupportedFourCC(FourCC),
    UnsupportedPixelFormat,
    TooManyMipMaps(u32),
    /// A volume/texture 3D without a depth.
    MissingDepth,
    /// The width, height, or depth of the texture is zero.
    ZeroDimension,
    /// The header of the DDS file describes a data section that is too large.
    ///
    /// I.e. it is possible for the header to describe a texture that requires
    /// >2^64 bytes of memory.
    DataLayoutTooBig,
    UnsupportedColorTypePrecision(SupportedFormat, Channels, Precision),
    UnexpectedBufferSize {
        expected: usize,
        actual: usize,
    },

    Header(HeaderError),
    Io(std::io::Error),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::UnsupportedDxgiFormat(format) => {
                write!(f, "DXGI format {:?} is not supported for decoding", format)
            }
            DecodeError::UnsupportedFourCC(four_cc) => {
                let bytes = four_cc.0.to_le_bytes();
                let ascii = if bytes.iter().all(|&b| b.is_ascii_alphanumeric()) {
                    let mut ascii = " (ASCII: ".to_string();
                    for &b in &bytes {
                        ascii.push(b as char);
                    }
                    ascii.push(')');
                    ascii
                } else {
                    String::new()
                };

                write!(
                    f,
                    "Unsupported FourCC code {}{} in DX10 header extension",
                    four_cc.0, ascii
                )
            }
            DecodeError::UnsupportedPixelFormat => {
                write!(f, "Unsupported pixel format in the DDS header")
            }
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
            DecodeError::DataLayoutTooBig => {
                write!(f, "Data layout described by the header is too large")
            }
            DecodeError::UnsupportedColorTypePrecision(format, color_type, color_precision) => {
                write!(
                    f,
                    "{:?} x {:?} is not supported for format {:?}. Supported are {:?} x {:?}",
                    color_type,
                    color_precision,
                    format,
                    format.supported_channels(),
                    format.supported_precisions()
                )
            }
            DecodeError::UnexpectedBufferSize { expected, actual } => {
                write!(
                    f,
                    "Unexpected buffer size: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }

            DecodeError::Header(error) => write!(f, "Header error: {}", error),
            DecodeError::Io(error) => write!(f, "I/O error: {}", error),
        }
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
    InvalidArraySizeForTexture3D(u32),

    Io(std::io::Error),
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeaderError::InvalidMagicBytes(bytes) => {
                write!(
                    f,
                    "Invalid magic bytes {:?}, expected [68, 68, 83, 32] (ASCII: 'DDS ')",
                    bytes
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
