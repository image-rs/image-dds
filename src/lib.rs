#![forbid(unsafe_code)]

mod cast;
mod decode;
mod detect;
mod error;
mod format;
mod header;
mod layout;
mod tiny_set;
mod util;

use std::io::Read;

pub use error::*;
pub use format::*;
pub use header::*;
pub use layout::*;
pub use tiny_set::*;

/// Additional options for the DDS decoder specifying how to read and interpret
/// the header.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Options {
    /// Whether magic bytes should be skipped when reading the header.
    ///
    /// DDS files typically start with the magic bytes `"DDS "`. By default, the
    /// decoder will check for these bytes and error if they are not present.
    ///
    /// If this is set to `true`, the decoder assume that the magic bytes are
    /// not present and immediately start reading the header. This can be used
    /// to read DDS files without magic bytes.
    ///
    /// Defaults to `false`.
    pub skip_magic_bytes: bool,

    /// The maximum allowed value of the `array_size` field in the header.
    ///
    /// DDS files support texture arrays and the `array_size` field denotes the
    /// number of textures in the array. The only exception for this are cube
    /// maps where `array_size` denotes the number of cube maps instead, meaning
    /// that the DDS file will contain `array_size * 6` textures (6 faces per
    /// cube map).
    ///
    /// Since `array_size` is defined by the file, it is possible for a
    /// malicious or corrupted file to contain a very large value. For security
    /// reasons, this option can be used to limit the maximum allowed value.
    ///
    /// To disable this limit, set this to `u32::MAX`.
    ///
    /// Defaults to `4096`.
    pub max_array_size: u32,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            skip_magic_bytes: false,
            max_array_size: 4096,
        }
    }
}

pub struct DdsDecoder {
    header: Header,
    format: SupportedFormat,
    layout: DataLayout,
}

impl DdsDecoder {
    /// Creates a new decoder by reading the header from the given reader.
    ///
    /// This is equivalent to calling `Decoder::new_with(r, Options::default())`.
    /// See [`Self::new_with`] for more details.
    pub fn new<R: Read>(r: &mut R) -> Result<Self, DecodeError> {
        Self::new_with(r, Options::default())
    }
    /// Creates a new decoder with the given options by reading the header from the given reader.
    ///
    /// If this operations succeeds, the given reader will be positioned at the start of the data
    /// section. All offsets in [`DataLayout`] are relative to this position.
    pub fn new_with<R: Read>(r: &mut R, options: Options) -> Result<Self, DecodeError> {
        // magic bytes
        if !options.skip_magic_bytes {
            Header::read_magic(r)?;
        }

        // header
        let header = Header::read(r)?;

        Self::from_header_with(header, options)
    }

    pub fn from_header(header: Header) -> Result<Self, DecodeError> {
        Self::from_header_with(header, Options::default())
    }
    pub fn from_header_with(header: Header, options: Options) -> Result<Self, DecodeError> {
        // enforce `array_size` limit
        if let Some(dxt10) = &header.dxt10 {
            if dxt10.array_size > options.max_array_size {
                return Err(DecodeError::ArraySizeTooBig(dxt10.array_size));
            }
        }

        // detect format
        let format = SupportedFormat::from_header(&header)?;

        // data layout
        let layout = DataLayout::from_header(&header, format)?;

        Ok(Self {
            header,
            format,
            layout,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn format(&self) -> SupportedFormat {
        self.format
    }
    pub fn layout(&self) -> &DataLayout {
        &self.layout
    }
}
