#![forbid(unsafe_code)]

mod cast;
mod color;
mod decode;
mod detect;
mod encode;
mod error;
mod format;
mod header;
mod layout;
mod pixel;
mod util;

use std::{io::Read, num::NonZeroU8};

pub use color::*;
pub use encode::{CompressionQuality, Dithering, EncodeOptions, ErrorMetric};
pub use error::*;
pub use format::*;
pub use header::*;
pub use layout::*;
pub use pixel::*;

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

    /// Whether to allow certain invalid DDS files to be read.
    ///
    /// Certain older software may generate DDS files that do not strictly
    /// adhere to the DDS specification and may contain invalid values in the
    /// header. By default, the decoder will reject such files.
    ///
    /// If this option is set to `true`, the decoder will (1) ignore invalid
    /// header values that would otherwise cause the decoder to reject the file
    /// and (2) attempt to fix the header to read the file correctly. To fix the
    /// header, [`Options::file_len`] must be provided.
    ///
    /// Defaults to `false`.
    pub permissive: bool,

    /// The length of the file in bytes.
    ///
    /// This length includes the magic bytes, header, and data section. Even if
    /// [`Options::skip_magic_bytes`] is set to `true`, the length must include
    /// the magic bytes.
    ///
    /// The purpose of this option is to provide more information, which enables
    /// the decoder to read certain invalid DDS files if [`Options::permissive`]
    /// is set to `true`. If [`Options::permissive`] is set to `false`, this
    /// option will be ignored.
    ///
    /// If this option is set incorrectly (i.e. this length is not equal to the
    /// actual length of the file), the decoder may misinterpret certain valid
    /// and invalid DDS files.
    ///
    /// Defaults to `None`.
    ///
    /// ### Usage
    ///
    /// The most common way to set this option is to use the file metadata:
    ///
    /// ```no_run
    /// let mut file = std::fs::File::open("example.dds").unwrap();
    ///
    /// let mut options = ddsd::Options::default();
    /// options.permissive = true;
    /// options.file_len = file.metadata().ok().map(|m| m.len());
    /// ```
    pub file_len: Option<u64>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            skip_magic_bytes: false,
            max_array_size: 4096,
            permissive: false,
            file_len: None,
        }
    }
}

pub struct DdsDecoder {
    header: Header,
    format: Format,
    layout: DataLayout,
}

impl DdsDecoder {
    /// Creates a new decoder by reading the header from the given reader.
    ///
    /// This is equivalent to calling `Decoder::new_with(r, Options::default())`.
    /// See [`Self::new_with`] for more details.
    pub fn new<R: Read>(r: &mut R) -> Result<Self, DecodeError> {
        Self::new_with(r, &Options::default())
    }
    /// Creates a new decoder with the given options by reading the header from the given reader.
    ///
    /// If this operations succeeds, the given reader will be positioned at the start of the data
    /// section. All offsets in [`DataLayout`] are relative to this position.
    pub fn new_with<R: Read>(r: &mut R, options: &Options) -> Result<Self, DecodeError> {
        Self::from_header_with(Header::read(r, options)?, options)
    }

    pub fn from_header(header: Header) -> Result<Self, DecodeError> {
        Self::from_header_with(header, &Options::default())
    }
    pub fn from_header_with(header: Header, options: &Options) -> Result<Self, DecodeError> {
        // enforce `array_size` limit
        if let Some(dxt10) = header.dx10() {
            if dxt10.array_size > options.max_array_size {
                return Err(DecodeError::ArraySizeTooBig(dxt10.array_size));
            }
        }

        // detect format
        let format = Format::from_header(&header)?;

        // data layout
        let layout = DataLayout::from_header_with(&header, format.into())?;

        Ok(Self {
            header,
            format,
            layout,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn format(&self) -> Format {
        self.format
    }
    pub fn layout(&self) -> &DataLayout {
        &self.layout
    }

    /// Whether the texture is in sRGB color space.
    ///
    /// This can only be `true` for DX10+ DDS files. Legacy (DX9) formats cannot
    /// specify the color space and are assumed to be linear.
    pub fn is_srgb(&self) -> bool {
        if let Some(dx10) = self.header.dx10() {
            dx10.dxgi_format.is_srgb()
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}
impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
    pub const fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub const fn is_multiple_of(&self, multiple: SizeMultiple) -> bool {
        self.width % multiple.width_multiple.get() as u32 == 0
            && self.height % multiple.height_multiple.get() as u32 == 0
    }
    pub const fn round_down_to_multiple(&self, multiple: SizeMultiple) -> Self {
        Self {
            width: self.width - self.width % multiple.width_multiple.get() as u32,
            height: self.height - self.height % multiple.height_multiple.get() as u32,
        }
    }
}
impl From<(u32, u32)> for Size {
    fn from((width, height): (u32, u32)) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SizeMultiple {
    pub width_multiple: NonZeroU8,
    pub height_multiple: NonZeroU8,
}
impl SizeMultiple {
    pub const ONE: Self = Self::new(1, 1);
    pub const M2_2: Self = Self::new(2, 2);

    const fn new(width_multiple: u8, height_multiple: u8) -> Self {
        Self {
            width_multiple: NonZeroU8::new(width_multiple).unwrap(),
            height_multiple: NonZeroU8::new(height_multiple).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
impl Rect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub const fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    /// Returns `true` if this rectangle is completely within the bounds of the
    /// given size.
    ///
    /// This means that `self.x + self.width <= size.width` and
    /// `self.y + self.height <= size.height`.
    pub(crate) fn is_within_bounds(&self, size: Size) -> bool {
        // use u64 to prevent overflow
        let end_x = self.x as u64 + self.width as u64;
        let end_y = self.y as u64 + self.height as u64;
        end_x <= size.width as u64 && end_y <= size.height as u64
    }
}
