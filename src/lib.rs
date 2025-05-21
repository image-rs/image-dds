//! # DDS
//!
//! A Rust library for decoding and encoding DDS (DirectDraw Surface) files.
//!
//! ## The DDS format
//!
//! DDS is a container format for storing compressed and uncompressed textures,
//! cube maps, volumes, buffers, and arrays of the before. It is used by
//! DirectX, OpenGL, Vulkan, and other graphics APIs.
//!
//! A DDS files has 2 parts: a header and a data section. The header describes
//! the type of data in the file (e.g. a BC1-compressed 100x200px texture) and
//! determines the layout of the data section. The data section then contains
//! the binary pixel data.
//!
//! ## Features
//!
//! - `rayon` (default): Parallel encoding using the `rayon` crate.
//!
//!   This feature will enable parallel encoding of DDS files. Both the
//!   high-level [`Encoder`] and low-level [`encode()`] functions will use this
//!   feature to speed up processing.
//!
//! All features marked with "(default)" are enabled by default.
//!
//! ## Usage
//!
//! ### Decoding
//!
//! The [`Decoder`] type is the high-level interface for decoding DDS files.
//!
//! The most common case, a single image, can be decoded as follows:
//!
//! ```no_run
//! # use dds::*;
//! # use std::fs::File;
//! let file = File::open("path/to/file.dds").unwrap();
//! let mut decoder = Decoder::new(file).unwrap();
//! // make sure the file is a single image
//! assert!(decoder.layout().texture().is_some());
//! // prepare a buffer to decode into
//! let mut data = vec![0u8; decoder.main_size().pixels() as usize * 4];
//! // create an image view from the buffer
//! let view = ImageViewMut::new(&mut data, decoder.main_size(), ColorFormat::RGBA_U8).unwrap();
//! // decode the image into the buffer
//! decoder.read_surface(view).unwrap();
//! ```
//!
//! Cube maps can be detected using `decoder.layout().is_cube_map()` and decoded
//! with [`Decoder::read_cube_map`].
//!
//! Volumes have to be read one depth slice at a time using [`Decoder::read_surface`].
//!
//! It is also possible to decode a rectangle of a surface using
//! [`Decoder::read_surface_rect`].
//!
//! ### Encoding
//!
//! Since the data of a DDS file is determined by the header, the first step to
//! encoding a DDS file is to create a header. See the documentation of
//! the [`crate::header`] module for more details.
//!
//! ```no_run
//! # use dds::{*, header::*};
//! # use std::fs::File;
//! fn save_rgba_image(
//!     file: &mut File,
//!     image_data: &[u8],
//!     width: u32,
//!     height: u32,
//! ) -> Result<(), EncodingError> {
//!     let format = Format::BC1_UNORM;
//!     let header = Header::new_image(width, height, format);
//!
//!     let mut encoder = Encoder::new(file, format, &header)?;
//!     encoder.options.quality = CompressionQuality::Fast;
//!
//!     let view = ImageView::new(image_data, Size::new(width, height), ColorFormat::RGBA_U8)
//!         .expect("invalid image data");
//!     encoder.write_surface(view)?;
//!     encoder.finish()?;
//!     Ok(())
//! }
//! ```
//!
//! Note the use of [`Encoder::finish()`]. This method will verify that the
//! file contains all necessary data.
//!
//! To create DDS files with mipmaps, use [`Encoder::write_surface_with`]:
//!
//! ```no_run
//! # use dds::{*, header::*};
//! # use std::fs::File;
//! fn save_rgba_image_with_mipmaps(
//!     file: &mut File,
//!     image_data: &[u8],
//!     width: u32,
//!     height: u32,
//! ) -> Result<(), EncodingError> {
//!     let format = Format::BC1_UNORM;
//!     let header = Header::new_image(width, height, format).with_mipmaps();
//!
//!     let mut encoder = Encoder::new(file, format, &header)?;
//!     encoder.options.quality = CompressionQuality::Fast;
//!
//!     let view = ImageView::new(image_data, Size::new(width, height), ColorFormat::RGBA_U8)
//!         .expect("invalid image data");
//!     let write_options = WriteOptions {
//!         generate_mipmaps: true,
//!         ..Default::default()
//!     };
//!     encoder.write_surface_with(view, None, &write_options)?;
//!     encoder.finish()?;
//!     Ok(())
//! }
//! ```
//!
//! Cube maps can be created by encoding their 6 faces in the order:
//! +X -X +Y -Y +Z -Z.
//!
//! Volumes have to be encoded one depth slice at a time using [`Encoder::write_surface`].
//!
//! ### Progress reporting
//!
//! The decoder is generally so fast that progress reporting is not needed.
//!
//! The encoder, however, can take a long time to encode large images. Use the
//! `progress` parameter of the [`Encoder::write_surface_with`] method to get
//! periodic updates on the encoding progress. See the [`Progress`] type for
//! more details.
//!
//! ### Low-level API
//!
//! Besides the `Encoder` and `Decoder` types, the library also exposes a low-level
//! API for encoding and decoding DDS surfaces. It should generally not be
//! necessary to use this API.
//!
//! The [`encode()`] and [`decode()`] functions are used to encode and decode a
//! single DDS surface. The [`SplitSurface`] type can be used to split a surface
//! into multiple fragments for parallel encoding.

#![forbid(unsafe_code)]

mod cast;
mod color;
mod decode;
mod decoder;
mod detect;
mod encode;
mod encoder;
mod error;
mod format;
pub mod header;
mod iter;
mod layout;
mod pixel;
mod progress;
mod resize;
mod split;
mod util;

pub use color::*;
pub use decode::{decode, decode_rect, DecodeOptions};
pub use decoder::*;
pub use encode::{
    encode, CompressionQuality, Dithering, EncodeOptions, EncodingSupport, ErrorMetric,
};
pub use encoder::*;
pub use error::*;
pub use format::*;
pub use layout::*;
pub use pixel::*;
pub use progress::*;
pub use split::*;

/// A borrowed slice of image data.
#[derive(Clone, Copy)]
pub struct ImageView<'a> {
    data: &'a [u8],
    size: Size,
    color: ColorFormat,
}
impl<'a> ImageView<'a> {
    /// Creates a new image view from the given data, size, and color format.
    ///
    /// The data must be the correct size for the given size and color format.
    /// If `data.len() != size.pixels() * color.bytes_per_pixel()`,
    /// then `None` is returned.
    pub fn new(data: &'a [u8], size: Size, color: ColorFormat) -> Option<Self> {
        if data.len() as u64 != size.pixels().saturating_mul(color.bytes_per_pixel() as u64) {
            return None;
        }
        Some(Self { data, size, color })
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    pub fn size(&self) -> Size {
        self.size
    }
    pub fn width(&self) -> u32 {
        self.size.width
    }
    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn color(&self) -> ColorFormat {
        self.color
    }
    pub fn row_pitch(&self) -> usize {
        self.size.width as usize * self.color.bytes_per_pixel() as usize
    }
}

/// A borrowed mutable slice of image data.
pub struct ImageViewMut<'a> {
    data: &'a mut [u8],
    size: Size,
    color: ColorFormat,
}
impl<'a> ImageViewMut<'a> {
    /// Creates a new image view from the given data, size, and color format.
    ///
    /// The data must be the correct size for the given size and color format.
    /// If `data.len() != size.pixels() * color.bytes_per_pixel()`,
    /// then `None` is returned.
    pub fn new(data: &'a mut [u8], size: Size, color: ColorFormat) -> Option<Self> {
        if data.len() as u64 != size.pixels().saturating_mul(color.bytes_per_pixel() as u64) {
            return None;
        }
        Some(Self { data, size, color })
    }

    pub fn data(&mut self) -> &mut [u8] {
        self.data
    }

    pub fn size(&self) -> Size {
        self.size
    }
    pub fn width(&self) -> u32 {
        self.size.width
    }
    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn color(&self) -> ColorFormat {
        self.color
    }
    pub fn row_pitch(&self) -> usize {
        self.size.width as usize * self.color.bytes_per_pixel() as usize
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
    /// Whether the area of this size is zero.
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
    /// The number of pixels in this size.
    pub const fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns the size of the mipmap at the given level.
    ///
    /// Level 0 is the original size. All returned sizes are at least 1x1.
    ///
    /// ```
    /// # use dds::Size;
    /// let size = Size::new(256, 100);
    /// assert_eq!(size.get_mipmap(0), size);
    /// assert_eq!(size.get_mipmap(1), Size::new(128, 50));
    /// assert_eq!(size.get_mipmap(2), Size::new(64, 25));
    /// assert_eq!(size.get_mipmap(3), Size::new(32, 12));
    /// assert_eq!(size.get_mipmap(4), Size::new(16, 6));
    /// assert_eq!(size.get_mipmap(5), Size::new(8, 3));
    /// assert_eq!(size.get_mipmap(6), Size::new(4, 1));
    /// assert_eq!(size.get_mipmap(7), Size::new(2, 1));
    /// assert_eq!(size.get_mipmap(8), Size::new(1, 1));
    /// assert_eq!(size.get_mipmap(9), Size::new(1, 1));
    /// // From now on, it's always 1x1
    /// assert_eq!(size.get_mipmap(255), Size::new(1, 1));
    /// ```
    pub const fn get_mipmap(&self, level: u8) -> Self {
        Self {
            width: util::get_mipmap_size(self.width, level).get(),
            height: util::get_mipmap_size(self.height, level).get(),
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
