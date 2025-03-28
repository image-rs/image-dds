//! Functionality for reading, parsing, and writing DDS headers.
//!
//! This module revolves around the [`Header`] enum, which represents a parsed
//! DDS header. [`Header::read`] and [`Header::write`] can be used to read and
//! write DDS headers from and to disk. [`Header::new_image`] and co. can be
//! used to create new headers. To use specifically DX9 or DX10 headers, use
//! [`Dx9Header`] and [`Dx10Header`] directly.
//!
//! [`RawHeader`] is a low-level representation of an unparsed DDS header. It is
//! bit-for-bit what is on disk. You rarely need to interact with this type, but
//! it can useful for manually detecting and parsing non-standard DDS files.
//!
//! # Creating a header
//!
//! This will create the header for a 128x256 BC1 image without mipmaps:
//!
//! ```
//! # use dds::{*, header::*};
//! let header = Header::new_image(128, 256, Format::BC1_UNORM);
//! assert_eq!(header.width(), 128);
//! assert_eq!(header.height(), 256);
//! assert_eq!(header.depth(), None);
//! assert_eq!(header.mipmap_count().get(), 1);
//! assert_eq!(header.dx10().unwrap().dxgi_format, DxgiFormat::BC1_UNORM);
//! ```
//!
//! To specify mipmaps, use [`Header::with_mipmap_count`] to set a specific
//! number of mipmaps, or [`Header::with_mipmaps`] to automatically set the
//! mipmap count based on the dimensions of the image:
//!
//! ```
//! # use dds::{*, header::*};
//! # use std::num::NonZeroU32;
//! let header = Header::new_image(128, 256, Format::BC1_UNORM).with_mipmaps();
//! assert_eq!(header.mipmap_count().get(), 9);
//!
//! // The layout of the image is:
//! let layout = DataLayout::from_header(&header).unwrap();
//! let texture = layout.texture().unwrap();
//! let mut mipmap_sizes = texture.iter_mips().map(|m| m.size());
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(128, 256)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(64, 128)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(32, 64)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(16, 32)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(8, 16)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(4, 8)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(2, 4)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(1, 2)));
//! assert_eq!(mipmap_sizes.next(), Some(Size::new(1, 1)));
//! assert_eq!(mipmap_sizes.next(), None);
//! ```
//!
//! You can also set a specific number of mipmaps with
//! [`Header::with_mipmap_count`]:
//!
//! ```
//! # use dds::{*, header::*};
//! # use std::num::NonZeroU32;
//! let header = Header::new_image(128, 256, Format::BC1_UNORM)
//!     .with_mipmap_count(NonZeroU32::new(4).unwrap());
//! assert_eq!(header.mipmap_count().get(), 4);
//! ```
//!
//! Lastly, if you need more control over the header, use [`Dx9Header`] and
//! [`Dx10Header`] directly.

use crate::{
    cast,
    detect::{dxgi_to_four_cc, dxgi_to_masked, four_cc_to_dxgi, masked_to_dxgi},
    util::{get_maximum_mipmap_count, read_u32_le_array, NON_ZERO_U32_ONE},
    CubeMapFaces, DataLayout, DataRegion, Format, HeaderError, PixelInfo, Size,
};
use bitflags::bitflags;
use std::{
    io::{Read, Write},
    num::NonZeroU32,
};

/// An unparsed DDS header without magic bytes.
///
/// See [`Header`] for a parsed version.
///
/// See <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header>
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RawHeader {
    /// Size of structure. This member must be set to 124.
    pub size: u32,
    /// Flags to indicate which members contain valid data.
    pub flags: DdsFlags,
    pub height: u32,
    pub width: u32,
    /// The pitch or number of bytes per scan line in an uncompressed texture;
    /// the total number of bytes in the top level texture for a compressed texture.
    pub pitch_or_linear_size: u32,
    /// Depth of a volume texture (in pixels), otherwise unused.
    pub depth: u32,
    /// Number of mipmap levels, otherwise unused.
    pub mipmap_count: u32,
    pub reserved1: [u32; 11],
    pub pixel_format: RawPixelFormat,
    pub caps: Caps,
    pub caps2: Caps2,
    pub caps3: u32,
    pub caps4: u32,
    pub reserved2: u32,
    pub dx10: Option<RawDx10Header>,
}
/// An unparsed DDS pixel format.
///
/// See [`Dx9PixelFormat`] for a parsed version.
///
/// See <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat>
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RawPixelFormat {
    /// Structure size; set to 32 (bytes).
    pub size: u32,
    /// Values which indicate what type of data is in the surface.
    pub flags: PixelFormatFlags,
    pub four_cc: FourCC,
    pub rgb_bit_count: u32,
    pub r_bit_mask: u32,
    pub g_bit_mask: u32,
    pub b_bit_mask: u32,
    pub a_bit_mask: u32,
}
/// An unparsed DDS DX10 header.
///
/// See [`Dx10Header`] for a parsed version.
///
/// See <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10>
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RawDx10Header {
    pub dxgi_format: u32,
    pub resource_dimension: u32,
    pub misc_flag: MiscFlags,
    pub array_size: u32,
    /// Contains additional metadata (formerly was reserved). The lower 3 bits indicate the alpha mode of the associated resource. The upper 29 bits are reserved and are typically 0.
    pub misc_flags2: u32,
}

impl RawHeader {
    pub(crate) const SIZE: u32 = 124;
    pub(crate) const INTS: usize = Self::SIZE as usize / 4;

    /// Reads the raw header **without** magic bytes from a reader.
    ///
    /// This will not do any form of validation whatsoever. The way for this
    /// operation to fail is for the given reader to error.
    pub fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buffer: [u32; RawHeader::INTS] = Default::default();
        read_u32_le_array(reader, &mut buffer)?;

        let mut header: Self = Self {
            size: buffer[0],
            flags: DdsFlags::from_bits_retain(buffer[1]),
            height: buffer[2],
            width: buffer[3],
            pitch_or_linear_size: buffer[4],
            depth: buffer[5],
            mipmap_count: buffer[6],
            reserved1: [
                buffer[7], buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13],
                buffer[14], buffer[15], buffer[16], buffer[17],
            ],
            pixel_format: RawPixelFormat {
                size: buffer[18],
                flags: PixelFormatFlags::from_bits_retain(buffer[19]),
                four_cc: FourCC(buffer[20]),
                rgb_bit_count: buffer[21],
                r_bit_mask: buffer[22],
                g_bit_mask: buffer[23],
                b_bit_mask: buffer[24],
                a_bit_mask: buffer[25],
            },
            caps: Caps::from_bits_retain(buffer[26]),
            caps2: Caps2::from_bits_retain(buffer[27]),
            caps3: buffer[28],
            caps4: buffer[29],
            reserved2: buffer[30],
            dx10: None,
        };

        if header.pixel_format.flags.contains(PixelFormatFlags::FOURCC)
            && header.pixel_format.four_cc == FourCC::DX10
        {
            let buffer = &mut buffer[..5];
            read_u32_le_array(reader, buffer)?;
            header.dx10 = Some(RawDx10Header {
                dxgi_format: buffer[0],
                resource_dimension: buffer[1],
                misc_flag: MiscFlags::from_bits_retain(buffer[2]),
                array_size: buffer[3],
                misc_flags2: buffer[4],
            });
        }

        Ok(header)
    }

    /// Write the raw header **without** magic bytes to a writer.
    ///
    /// This will not do any form of validation whatsoever. The way for this
    /// operation to fail is for the given reader to error.
    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let mut buffer: [u32; 36] = [
            self.size,
            self.flags.bits(),
            self.height,
            self.width,
            self.pitch_or_linear_size,
            self.depth,
            self.mipmap_count,
            self.reserved1[0],
            self.reserved1[1],
            self.reserved1[2],
            self.reserved1[3],
            self.reserved1[4],
            self.reserved1[5],
            self.reserved1[6],
            self.reserved1[7],
            self.reserved1[8],
            self.reserved1[9],
            self.reserved1[10],
            self.pixel_format.size,
            self.pixel_format.flags.bits(),
            self.pixel_format.four_cc.0,
            self.pixel_format.rgb_bit_count,
            self.pixel_format.r_bit_mask,
            self.pixel_format.g_bit_mask,
            self.pixel_format.b_bit_mask,
            self.pixel_format.a_bit_mask,
            self.caps.bits(),
            self.caps2.bits(),
            self.caps3,
            self.caps4,
            self.reserved2,
            // fill in the DXT10 header later
            0,
            0,
            0,
            0,
            0,
        ];
        let selection = if let Some(dx10) = &self.dx10 {
            buffer[31] = dx10.dxgi_format;
            buffer[32] = dx10.resource_dimension;
            buffer[33] = dx10.misc_flag.bits();
            buffer[34] = dx10.array_size;
            buffer[35] = dx10.misc_flags2;
            &mut buffer[..]
        } else {
            &mut buffer[..31]
        };
        let bytes = cast::as_bytes_mut(selection);
        cast::slice_ne_to_le_32(bytes);
        writer.write_all(bytes)?;
        Ok(())
    }
}

impl RawPixelFormat {
    const SIZE: u32 = 32;

    fn new_four_cc(four_cc: FourCC) -> RawPixelFormat {
        Self {
            size: Self::SIZE,
            flags: PixelFormatFlags::FOURCC,
            four_cc,
            rgb_bit_count: 0,
            r_bit_mask: 0,
            g_bit_mask: 0,
            b_bit_mask: 0,
            a_bit_mask: 0,
        }
    }
    fn new_mask(mask: &MaskPixelFormat) -> RawPixelFormat {
        Self {
            size: Self::SIZE,
            flags: mask.flags,
            four_cc: FourCC::NONE,
            rgb_bit_count: mask.rgb_bit_count.into(),
            r_bit_mask: mask.r_bit_mask,
            g_bit_mask: mask.g_bit_mask,
            b_bit_mask: mask.b_bit_mask,
            a_bit_mask: mask.a_bit_mask,
        }
    }
}

impl RawDx10Header {
    pub(crate) const SIZE: u32 = 20;
}

/// A parsed header, split by version.
///
/// <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header>
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Header {
    Dx9(Dx9Header),
    Dx10(Dx10Header),
}
/// DX9-specific header data.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dx9Header {
    /// Surface height (in pixels).
    pub height: u32,
    /// Surface width (in pixels).
    pub width: u32,
    /// Depth of a volume texture (in pixels).
    pub depth: Option<u32>,
    /// Number of mipmap levels.
    pub mipmap_count: NonZeroU32,
    /// Additional detail about the surfaces stored.
    pub caps2: Caps2,
    pub pixel_format: Dx9PixelFormat,
}
/// DX9 pixel format.
///
/// DDS files define their pixel format either with a (legacy) `DDS_PIXELFORMAT`
/// structure or with a `DXGI_FORMAT` from the Direct3D 10 and later APIs. This
/// enum represents all cases in a single type.
///
/// <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat>
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dx9PixelFormat {
    FourCC(FourCC),
    Mask(MaskPixelFormat),
}
/// The RGBA mask for reading color data.
///
/// For more information, see <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat>.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskPixelFormat {
    /// Values which indicate what type of data is in the surface.
    ///
    /// The flag [`PixelFormatFlags::FOURCC`] must **not** be set.
    pub flags: PixelFormatFlags,
    /// Number of bits in an RGB (possibly including alpha) format. Valid when dwFlags includes DDPF_RGB, DDPF_LUMINANCE, or DDPF_YUV.
    pub rgb_bit_count: RgbBitCount,
    /// Red (or luminance or Y) mask for reading color data. For instance, given the A8R8G8B8 format, the red mask would be 0x00ff0000.
    pub r_bit_mask: u32,
    /// Green (or U) mask for reading color data. For instance, given the A8R8G8B8 format, the green mask would be 0x0000ff00.
    pub g_bit_mask: u32,
    /// Blue (or V) mask for reading color data. For instance, given the A8R8G8B8 format, the blue mask would be 0x000000ff.
    pub b_bit_mask: u32,
    /// Alpha mask for reading alpha data. dwFlags must include DDPF_ALPHAPIXELS or DDPF_ALPHA. For instance, given the A8R8G8B8 format, the alpha mask would be 0xff000000.
    pub a_bit_mask: u32,
}
/// Valid values for [`MaskPixelFormat::rgb_bit_count`].
///
/// There are only 4 possible valid values: 8, 16, 24, and 32. This is because
/// the number of bits must:
///
/// 1. be greater than 0,
/// 2. be divisible to be a whole number of bytes, and
/// 3. be at most 32 because the masks don't support more than that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RgbBitCount {
    Count8 = 8,
    Count16 = 16,
    Count24 = 24,
    Count32 = 32,
}
impl TryFrom<u32> for RgbBitCount {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(RgbBitCount::Count8),
            16 => Ok(RgbBitCount::Count16),
            24 => Ok(RgbBitCount::Count24),
            32 => Ok(RgbBitCount::Count32),
            _ => Err(value),
        }
    }
}
impl From<RgbBitCount> for u32 {
    fn from(value: RgbBitCount) -> Self {
        value as u32
    }
}

/// DDS header extension to handle resource arrays, DXGI pixel formats that don't map to the legacy Microsoft DirectDraw pixel format structures, and additional metadata.
///
/// <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10>
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dx10Header {
    /// Surface height (in pixels).
    pub height: u32,
    /// Surface width (in pixels).
    pub width: u32,
    /// Depth of a volume texture (in pixels).
    pub depth: Option<u32>,
    /// Number of mipmap levels.
    pub mipmap_count: NonZeroU32,

    /// The surface pixel format.
    pub dxgi_format: DxgiFormat,
    /// Identifies the type of resource.
    pub resource_dimension: ResourceDimension,
    /// Identifies other, less common options for resources.
    ///
    /// The following value for this member is a subset of the values in the D3D10_RESOURCE_MISC_FLAG or D3D11_RESOURCE_MISC_FLAG enumeration.
    pub misc_flag: MiscFlags,
    /// The number of elements in the array.
    ///
    /// For a 2D texture that is also a cube-map texture, this number represents the number of cubes. This number is the same as the number in the NumCubes member of D3D10_TEXCUBE_ARRAY_SRV1 or D3D11_TEXCUBE_ARRAY_SRV). In this case, the DDS file contains arraySize*6 2D textures. For more information about this case, see the miscFlag description.
    ///
    /// For a 3D texture, you must set this number to 1.
    pub array_size: u32,
    /// The alpha mode of the associated resource.
    pub alpha_mode: AlphaMode,
}

/// Options specifying how to read and interpret a DDS header.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ParseOptions {
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

    /// Whether to allow certain invalid DDS files to be read.
    ///
    /// Certain older software may generate DDS files that do not strictly
    /// adhere to the DDS specification and may contain invalid values in the
    /// header. By default, the decoder will reject such files.
    ///
    /// If this option is set to `true`, the decoder will (1) ignore invalid
    /// header values that would otherwise cause the decoder to reject the file
    /// and (2) attempt to fix the header to read the file correctly. To fix the
    /// header, [`Self::file_len`] must be provided.
    ///
    /// Defaults to `false`.
    pub permissive: bool,

    /// The length of the file in bytes.
    ///
    /// This length includes the magic bytes, header, and data section. Even if
    /// [`Self::skip_magic_bytes`] is set to `true`, the length must include
    /// the magic bytes.
    ///
    /// The purpose of this option is to provide more information, which enables
    /// the decoder to read certain invalid DDS files if [`Self::permissive`]
    /// is set to `true`. If [`Self::permissive`] is set to `false`, this
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
    /// # use dds::header::*;
    /// let mut file = std::fs::File::open("example.dds").unwrap();
    /// let file_len = file.metadata().ok().map(|m| m.len());
    /// let options = ParseOptions::new_permissive(file_len);
    /// ```
    pub file_len: Option<u64>,
}
impl ParseOptions {
    pub fn new_permissive(file_len: Option<u64>) -> Self {
        Self {
            permissive: true,
            file_len,
            ..Default::default()
        }
    }
}
#[allow(clippy::derivable_impls)]
impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            skip_magic_bytes: false,
            permissive: false,
            file_len: None,
        }
    }
}

impl From<Dx9Header> for Header {
    fn from(header: Dx9Header) -> Self {
        Self::Dx9(header)
    }
}
impl From<Dx10Header> for Header {
    fn from(header: Dx10Header) -> Self {
        Self::Dx10(header)
    }
}
impl From<FourCC> for Dx9PixelFormat {
    fn from(four_cc: FourCC) -> Self {
        Self::FourCC(four_cc)
    }
}
impl From<MaskPixelFormat> for Dx9PixelFormat {
    fn from(mask: MaskPixelFormat) -> Self {
        Self::Mask(mask)
    }
}

impl Header {
    pub const fn width(&self) -> u32 {
        match self {
            Header::Dx9(header) => header.width,
            Header::Dx10(header) => header.width,
        }
    }
    pub const fn height(&self) -> u32 {
        match self {
            Header::Dx9(header) => header.height,
            Header::Dx10(header) => header.height,
        }
    }
    pub const fn size(&self) -> Size {
        Size::new(self.width(), self.height())
    }
    pub const fn depth(&self) -> Option<u32> {
        match self {
            Header::Dx9(header) => header.depth,
            Header::Dx10(header) => header.depth,
        }
    }
    pub const fn mipmap_count(&self) -> NonZeroU32 {
        match self {
            Header::Dx9(header) => header.mipmap_count,
            Header::Dx10(header) => header.mipmap_count,
        }
    }

    /// The [`Dx10Header::array_size`] value, or 1 if it's a DX9 header.
    pub const fn array_size(&self) -> u32 {
        match self {
            Self::Dx9(_) => 1,
            Self::Dx10(header) => header.array_size,
        }
    }
    /// Returns [`Dx10Header::alpha_mode`] or [`Dx9Header::alpha_mode`].
    pub const fn alpha_mode(&self) -> AlphaMode {
        match self {
            Self::Dx9(header) => header.alpha_mode(),
            Self::Dx10(header) => header.alpha_mode,
        }
    }

    /// Whether the color format is in sRGB color space.
    ///
    /// This can only be `true` for DX10 header. Legacy (DX9) formats cannot
    /// specify the color space and are assumed to be linear.
    pub const fn is_srgb(&self) -> bool {
        if let Self::Dx10(dx10) = self {
            dx10.dxgi_format.is_srgb()
        } else {
            false
        }
    }
    /// Whether this header describes a cube map.
    ///
    /// Note: DX9 supports partial cube maps, which will also return `true`.
    /// DX10 only supports full cube maps.
    pub const fn is_cube_map(&self) -> bool {
        match self {
            Self::Dx9(dx9) => dx9.is_cube_map(),
            Self::Dx10(dx10) => dx10.is_cube_map(),
        }
    }
    /// Whether this header describes a volume texture.
    pub const fn is_volume(&self) -> bool {
        match self {
            Self::Dx9(dx9) => dx9.is_volume(),
            Self::Dx10(dx10) => dx10.is_volume(),
        }
    }

    pub const fn dx9(&self) -> Option<&Dx9Header> {
        match self {
            Header::Dx9(dx9) => Some(dx9),
            _ => None,
        }
    }
    pub const fn dx10(&self) -> Option<&Dx10Header> {
        match self {
            Header::Dx10(dx10) => Some(dx10),
            _ => None,
        }
    }

    /// Returns the size of the header (including the DX10 header extension if
    /// any) in bytes. This does **not** include the magic bytes at the start
    /// of the file.
    ///
    /// This is useful for calculating the offset to the pixel data.
    ///
    /// The returned value will be 144 for DX10 DDS files and 124 for legacy
    /// files.
    pub const fn byte_len(&self) -> usize {
        let mut size = RawHeader::SIZE;
        if self.dx10().is_some() {
            size += RawDx10Header::SIZE;
        }
        size as usize
    }

    /// Creates a new header for a 2D texture with the given dimensions and
    /// format.
    ///
    /// This will prefer DX10 headers if the format is supported by DX10.
    ///
    /// The mipmap count is set to 1.
    pub fn new_image(width: u32, height: u32, format: Format) -> Self {
        if let Ok(dxgi) = DxgiFormat::try_from(format) {
            Self::Dx10(Dx10Header::new_image(width, height, dxgi))
        } else {
            Self::Dx9(Dx9Header::new_image(
                width,
                height,
                format.try_into().unwrap(),
            ))
        }
    }
    /// Creates a new header for a 3D texture with the given dimensions and
    /// format.
    ///
    /// This will prefer DX10 headers if the format is supported by DX10.
    ///
    /// The mipmap count is set to 1.
    pub fn new_volume(width: u32, height: u32, depth: u32, format: Format) -> Self {
        if let Ok(dxgi) = DxgiFormat::try_from(format) {
            Self::Dx10(Dx10Header::new_volume(width, height, depth, dxgi))
        } else {
            Self::Dx9(Dx9Header::new_volume(
                width,
                height,
                depth,
                format.try_into().unwrap(),
            ))
        }
    }
    /// Creates a new header for a cube map with the given dimensions and
    /// format.
    ///
    /// This will prefer DX10 headers if the format is supported by DX10.
    ///
    /// The mipmap count is set to 1.
    pub fn new_cube_map(width: u32, height: u32, format: Format) -> Self {
        if let Ok(dxgi) = DxgiFormat::try_from(format) {
            Self::Dx10(Dx10Header::new_cube_map(width, height, dxgi))
        } else {
            Self::Dx9(Dx9Header::new_cube_map(
                width,
                height,
                format.try_into().unwrap(),
            ))
        }
    }

    /// A builder-pattern-style method to set the width and height of the
    /// header.
    ///
    /// Depth will be set to `None`. If you want to leave the depth unchanged
    /// or change it as well, use [`Header::with_dimensions`].
    pub fn with_size(mut self, size: Size) -> Header {
        match &mut self {
            Header::Dx9(header) => {
                header.width = size.width;
                header.height = size.height;
                header.depth = None;
            }
            Header::Dx10(header) => {
                header.width = size.width;
                header.height = size.height;
                header.depth = None;
            }
        }
        self
    }
    /// A builder-pattern-style method to set the width, height, and depth of
    /// the header.
    ///
    /// For headers of 2D textures, [`Header::with_size`] is more appropriate.
    pub fn with_dimensions(mut self, width: u32, height: u32, depth: Option<u32>) -> Header {
        match &mut self {
            Header::Dx9(header) => {
                header.width = width;
                header.height = height;
                header.depth = depth;
            }
            Header::Dx10(header) => {
                header.width = width;
                header.height = height;
                header.depth = depth;
            }
        }
        self
    }
    /// A builder-pattern-style method to set the mipmap count of the header.
    ///
    /// For the an easier way to enable mipmapping, use [`Header::with_mipmaps`].
    pub fn with_mipmap_count(mut self, mipmap_count: NonZeroU32) -> Header {
        match &mut self {
            Header::Dx9(header) => header.mipmap_count = mipmap_count,
            Header::Dx10(header) => header.mipmap_count = mipmap_count,
        }
        self
    }
    /// A builder-pattern-style method to set the mipmap count of the header
    /// such that the last mipmap level has the dimensions 1x1 (or 1x1x1).
    /// E.g. for 64x256 image, the mipmap count will be set to 9.
    pub fn with_mipmaps(self) -> Header {
        let max = get_maximum_mipmap_count(
            self.width()
                .max(self.height())
                .max(self.depth().unwrap_or(1)),
        );

        self.with_mipmap_count(max)
    }

    /// Converts this header into a DX9 header if possible. If the header is a
    /// DX9 header already, it will be returned as is.
    pub fn to_dx9(&self) -> Option<Dx9Header> {
        match self {
            Header::Dx9(dx9_header) => Some(dx9_header.clone()),
            Header::Dx10(dx10_header) => dx10_header.to_dx9(),
        }
    }
    /// Converts this header into a DX10 header if possible. If the header is a
    /// DX10 header already, it will be returned as is.
    pub fn to_dx10(&self) -> Option<Dx10Header> {
        match self {
            Header::Dx9(dx9_header) => dx9_header.to_dx10(),
            Header::Dx10(dx10_header) => Some(dx10_header.clone()),
        }
    }

    fn fix_based_on_file_len(&mut self, options: &ParseOptions) -> Option<()> {
        fn get_expected_data_len(header: &Header, options: &ParseOptions) -> Option<u64> {
            let non_data = Header::MAGIC.len() + header.byte_len();
            options.file_len?.checked_sub(non_data as u64)
        }

        // Prepare the necessary information
        let expected_data_len = get_expected_data_len(self, options)?;
        let pixel_info = PixelInfo::from_header(self).ok()?;
        let test = move |header: &Header| {
            if let Ok(layout) = DataLayout::from_header_with(header, pixel_info) {
                layout.data_len() == expected_data_len
            } else {
                false
            }
        };

        // The common is that the header is already correct
        if test(self) {
            return Some(());
        }

        // Some DX10 writers set array_size=0 for "arrays" with one element.
        // https://github.com/microsoft/DirectXTex/pull/490
        //
        // Note: Unlike the other fixes, this directly change the header even if it
        // doesn't match the expected data length. This is because
        // `expected_data_len > 0` always implies `array_size > 0`, so we know that
        // `array_size = 0` is wrong, no matter what.
        if let Header::Dx10(dx10) = self {
            if expected_data_len > 0 && dx10.array_size == 0 {
                dx10.array_size = 1;

                // update the current layout since we directly changed the header
                if test(self) {
                    return Some(());
                }
            }
        }

        // Some DDS files containing a single cube map have array_size set to 6.
        // This is incorrect and likely stems from an incorrect MS DDS docs example.
        // https://github.com/MicrosoftDocs/win32/pull/1970
        if let Some(dx10) = self.dx10() {
            if dx10.array_size == 6
                && dx10.resource_dimension == ResourceDimension::Texture2D
                && dx10.misc_flag.contains(MiscFlags::TEXTURE_CUBE)
            {
                let mut new_header = self.clone();
                if let Header::Dx10(dx10) = &mut new_header {
                    dx10.array_size = 1;
                }

                if test(&new_header) {
                    *self = new_header;
                    return Some(());
                }
            }
        }

        // Sometimes, the mipmap count is incorrect. We can try to fix this by
        // simply guessing the correct mipmap count.
        let mipmap = self.mipmap_count().get();
        let max_levels = get_maximum_mipmap_count(
            self.width()
                .max(self.height())
                .max(self.depth().unwrap_or(1)),
        );
        let guesses = [
            1,                // it's very common for DDS images to have no mipmaps
            max_levels.get(), // or a full mipmap chain
            mipmap - 1,       // otherwise, it could be an off-by-one error
            mipmap.saturating_add(1),
        ];
        for guess in guesses.into_iter().filter_map(NonZeroU32::new) {
            let new_header = self.clone().with_mipmap_count(guess);

            if test(&new_header) {
                *self = new_header;
                return Some(());
            }
        }

        // sadly, we couldn't fix it
        None
    }

    /// The magic bytes (`'DDS '`) at the start of every DDS file.
    pub const MAGIC: [u8; 4] = *b"DDS ";

    /// The magic bytes `'DDS '` are at the start of every DDS file. This
    /// function reads the magic bytes and returns `Ok` if they are correct.
    ///
    /// See [`Header::MAGIC`] for the expected magic bytes.
    pub fn read_magic<R: Read>(reader: &mut R) -> Result<(), HeaderError> {
        let mut buffer = [0; 4];
        reader.read_exact(&mut buffer)?;

        if buffer != Self::MAGIC {
            return Err(HeaderError::InvalidMagicBytes(buffer));
        }

        Ok(())
    }

    /// Reads the header from a reader.
    ///
    /// Magic bytes are read by default and can be turned off with
    /// [`ParseOptions::skip_magic_bytes`].
    ///
    /// If the header is read successfully, the reader will be at the start of the pixel data.
    pub fn read<R: Read>(reader: &mut R, options: &ParseOptions) -> Result<Self, HeaderError> {
        if !options.skip_magic_bytes {
            Self::read_magic(reader)?;
        }

        let raw = RawHeader::read(reader)?;
        Self::from_raw(&raw, options)
    }

    pub fn from_raw(raw: &RawHeader, options: &ParseOptions) -> Result<Self, HeaderError> {
        // verify header size
        if raw.size != RawHeader::SIZE {
            if options.permissive && raw.size == 24 {
                // Some DDS files from the game Stalker 2 have their header size
                // set to 24 instead of 124. This is likely a typo in the source
                // code from the DDS encoder they used.
                // https://github.com/microsoft/DirectXTex/issues/399
            } else {
                return Err(HeaderError::InvalidHeaderSize(raw.size));
            }
        }

        let flags = raw.flags;
        let height = raw.height;
        let width = raw.width;
        let depth = if flags.contains(DdsFlags::DEPTH) {
            Some(raw.depth)
        } else {
            None
        };

        let mipmap_count = if flags.contains(DdsFlags::MIPMAP_COUNT)
            || raw.caps.contains(Caps::COMPLEX)
            || raw.caps.contains(Caps::MIPMAP)
        {
            raw.mipmap_count
        } else {
            1
        };
        let mipmap_count = NonZeroU32::new(mipmap_count).unwrap_or(NON_ZERO_U32_ONE);

        // this always has to be parsed to throw an error if it's invalid
        let pixel_format = Dx9PixelFormat::from_raw(&raw.pixel_format, options)?;

        let mut header = if let Some(dx10) = &raw.dx10 {
            let dxgi_format =
                DxgiFormat::try_from(dx10.dxgi_format).map_err(HeaderError::InvalidDxgiFormat)?;
            let resource_dimension = ResourceDimension::try_from(dx10.resource_dimension)
                .map_err(HeaderError::InvalidResourceDimension)?;

            let misc_flag = dx10.misc_flag;

            let raw_alpha_mode = dx10.misc_flags2 & 0b111;
            let alpha_mode = if let Ok(alpha_mode) = AlphaMode::try_from(raw_alpha_mode) {
                alpha_mode
            } else if options.permissive {
                AlphaMode::Unknown
            } else {
                return Err(HeaderError::InvalidAlphaMode(raw_alpha_mode));
            };

            let mut array_size = dx10.array_size;
            if resource_dimension == ResourceDimension::Texture3D && array_size != 1 {
                if options.permissive {
                    array_size = 1;
                } else {
                    return Err(HeaderError::InvalidArraySizeForTexture3D(array_size));
                }
            }

            // DX10 header
            Header::Dx10(Dx10Header {
                height,
                width,
                depth,
                mipmap_count,
                dxgi_format,
                resource_dimension,
                misc_flag,
                array_size,
                alpha_mode,
            })
        } else {
            // DX9 header
            Header::Dx9(Dx9Header {
                height,
                width,
                depth,
                mipmap_count,
                caps2: raw.caps2,
                pixel_format,
            })
        };

        if options.permissive {
            _ = header.fix_based_on_file_len(options);
        }

        Ok(header)
    }

    /// Writes the header including magic bytes.
    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&Self::MAGIC)?;

        let raw = self.to_raw();
        raw.write(writer)
    }

    pub fn to_raw(&self) -> RawHeader {
        let mut flags = DdsFlags::REQUIRED | DdsFlags::MIPMAP_COUNT;
        let mut caps = Caps::REQUIRED;

        if self.mipmap_count().get() > 1 {
            caps |= Caps::MIPMAP | Caps::COMPLEX;
        }
        if self.depth().is_some() {
            flags |= DdsFlags::DEPTH;
        }

        // We can only calculate the pitch or linear size if we know the byte
        // size and layout of the pixel data.
        let mut pitch_or_linear_size = 0;
        if let Ok(pixel_info) = PixelInfo::from_header(self) {
            if let PixelInfo::Fixed { bytes_per_pixel } = pixel_info {
                let pitch = self.width().checked_mul(bytes_per_pixel as u32);
                if let Some(pitch) = pitch {
                    pitch_or_linear_size = pitch;
                    flags |= DdsFlags::PITCH;
                }
            } else {
                let linear_size: Option<u32> = pixel_info
                    .surface_bytes(self.size())
                    .and_then(|size| size.try_into().ok());
                if let Some(linear_size) = linear_size {
                    pitch_or_linear_size = linear_size;
                    flags |= DdsFlags::LINEAR_SIZE;
                }
            }
        }

        let (caps2, pixel_format, dx10) = match self {
            Header::Dx9(dx9_header) => {
                let format = match &dx9_header.pixel_format {
                    Dx9PixelFormat::FourCC(four_cc) => RawPixelFormat::new_four_cc(*four_cc),
                    Dx9PixelFormat::Mask(mask_pixel_format) => {
                        RawPixelFormat::new_mask(mask_pixel_format)
                    }
                };

                (dx9_header.caps2, format, None)
            }
            Header::Dx10(dx10_header) => {
                let mut caps2 = Caps2::empty();
                if dx10_header.resource_dimension == ResourceDimension::Texture3D {
                    caps2 |= Caps2::VOLUME;
                }
                if dx10_header.misc_flag.contains(MiscFlags::TEXTURE_CUBE) {
                    caps2 |= Caps2::CUBE_MAP | Caps2::CUBE_MAP_ALL_FACES;
                }

                let dx10 = RawDx10Header {
                    dxgi_format: dx10_header.dxgi_format.into(),
                    resource_dimension: dx10_header.resource_dimension.into(),
                    misc_flag: dx10_header.misc_flag,
                    array_size: dx10_header.array_size,
                    misc_flags2: dx10_header.alpha_mode.into(),
                };

                (caps2, RawPixelFormat::new_four_cc(FourCC::DX10), Some(dx10))
            }
        };

        RawHeader {
            size: RawHeader::SIZE,
            flags,
            height: self.height(),
            width: self.width(),
            pitch_or_linear_size,
            depth: self.depth().unwrap_or(1),
            mipmap_count: self.mipmap_count().get(),
            reserved1: [0; 11],
            pixel_format,
            caps,
            caps2,
            caps3: 0,
            caps4: 0,
            reserved2: 0,
            dx10,
        }
    }
}

impl Dx9PixelFormat {
    fn from_raw(raw: &RawPixelFormat, options: &ParseOptions) -> Result<Self, HeaderError> {
        let size = raw.size;
        if size != RawPixelFormat::SIZE {
            if options.permissive && size == 0 {
                // Some DDS files have their pixel format size set to 0.
                // https://github.com/microsoft/DirectXTex/issues/392
            } else if options.permissive && size == 24 {
                // Some DDS files from the game Flat Out 2 have their pixel
                // format size set to 24 instead of 32. This is likely a bug in
                // the program that created the DDS files.
                // https://github.com/microsoft/DirectXTex/issues/392
            } else {
                return Err(HeaderError::InvalidPixelFormatSize(size));
            }
        }

        let mut flags = raw.flags;
        let four_cc = raw.four_cc;
        let rgb_bit_count = raw.rgb_bit_count;

        if options.permissive
            && rgb_bit_count == 0
            && four_cc != FourCC::NONE
            && !flags.contains(PixelFormatFlags::FOURCC)
        {
            // Some old DDS files from Unreal Tournament 2004 have no flags set,
            // an rgb bit count of 0, and use four CC. These files are invalid
            // and format detection will fail for them, so we need to fix the
            // header here. Since those files do use four CC, we just set the
            // missing flag.
            // https://github.com/microsoft/DirectXTex/pull/371
            flags |= PixelFormatFlags::FOURCC;
        }

        let format = if flags.contains(PixelFormatFlags::FOURCC) {
            Dx9PixelFormat::FourCC(four_cc)
        } else {
            let rgb_bit_count = match RgbBitCount::try_from(rgb_bit_count) {
                Ok(valid) => valid,
                Err(invalid) => return Err(HeaderError::InvalidRgbBitCount(invalid)),
            };

            Dx9PixelFormat::Mask(MaskPixelFormat {
                flags,
                rgb_bit_count,
                r_bit_mask: raw.r_bit_mask,
                g_bit_mask: raw.g_bit_mask,
                b_bit_mask: raw.b_bit_mask,
                a_bit_mask: raw.a_bit_mask,
            })
        };

        Ok(format)
    }
}

impl Dx9Header {
    /// Returns the alpha mode of the pixel format.
    ///
    /// This will mostly be `AlphaMode::Unknown` because DX9 doesn't have a
    /// concept of alpha modes. However, `DXT2` and `DXT4` formats are specified
    /// to be premultiplied alpha and will return `AlphaMode::Premultiplied`.
    pub const fn alpha_mode(&self) -> AlphaMode {
        if let Dx9PixelFormat::FourCC(four_cc) = self.pixel_format {
            if four_cc.0 == FourCC::DXT2.0 || four_cc.0 == FourCC::DXT4.0 {
                return AlphaMode::Premultiplied;
            }
        }
        AlphaMode::Unknown
    }

    /// Whether this header describes a cube map by checking for the
    /// [`Caps2::CUBE_MAP`] flag.
    ///
    /// Note: DX9 supports partial cube maps, which will also return `true`.
    /// See [`Dx9Header::cube_map_faces`].
    pub const fn is_cube_map(&self) -> bool {
        self.caps2.contains(Caps2::CUBE_MAP)
    }
    /// Returns the cube map faces iff this header describes a cube map.
    pub fn cube_map_faces(&self) -> Option<CubeMapFaces> {
        if self.caps2.contains(Caps2::CUBE_MAP) {
            Some(self.caps2.into())
        } else {
            None
        }
    }

    /// Whether this header describes a volume texture by checking for the
    /// [`Caps2::VOLUME`] flag.
    pub const fn is_volume(&self) -> bool {
        self.caps2.contains(Caps2::VOLUME)
    }

    /// Creates a new header for DX10 texture 2D with the given dimensions and
    /// format.
    ///
    /// The mipmap count is set to 1.
    pub const fn new_image(width: u32, height: u32, format: Dx9PixelFormat) -> Self {
        Self {
            height,
            width,
            depth: None,
            mipmap_count: NON_ZERO_U32_ONE,
            caps2: Caps2::empty(),
            pixel_format: format,
        }
    }
    /// Creates a new header for DX10 texture 3D with the given dimensions and
    /// format.
    ///
    /// The mipmap count is set to 1.
    pub const fn new_volume(width: u32, height: u32, depth: u32, format: Dx9PixelFormat) -> Self {
        Self {
            height,
            width,
            depth: Some(depth),
            mipmap_count: NON_ZERO_U32_ONE,
            caps2: Caps2::VOLUME,
            pixel_format: format,
        }
    }
    /// Creates a new header for DX9 cube map with the given dimensions and
    /// format.
    ///
    /// The mipmap count is set to 1 and the cube map faces are set.
    pub const fn new_cube_map(width: u32, height: u32, format: Dx9PixelFormat) -> Self {
        Self {
            height,
            width,
            depth: None,
            mipmap_count: NON_ZERO_U32_ONE,
            caps2: Caps2::CUBE_MAP.union(Caps2::CUBE_MAP_ALL_FACES),
            pixel_format: format,
        }
    }

    /// A builder-pattern-style method to set the width and height of the
    /// header.
    ///
    /// Depth will be set to `None`. If you want to leave the depth unchanged
    /// or change it as well, use [`Self::with_dimensions`].
    pub fn with_size(mut self, size: Size) -> Self {
        self.width = size.width;
        self.height = size.height;
        self.depth = None;
        self
    }
    /// A builder-pattern-style method to set the width, height, and depth of
    /// the header.
    ///
    /// For headers of 2D textures, [`Self::with_size`] is more appropriate.
    pub fn with_dimensions(mut self, width: u32, height: u32, depth: Option<u32>) -> Self {
        self.width = width;
        self.height = height;
        self.depth = depth;
        self
    }
    /// A builder-pattern-style method to set the mipmap count of the header.
    pub fn with_mipmap_count(mut self, mipmap_count: NonZeroU32) -> Self {
        self.mipmap_count = mipmap_count;
        self
    }
    /// A builder-pattern-style method to set the cube map faces of the header.
    ///
    /// This will set the [`Caps2::CUBE_MAP`] flag and the flags of all given
    /// faces. Faces not given will have their flags unset.
    pub fn with_cube_map_faces(mut self, faces: CubeMapFaces) -> Self {
        self.caps2 =
            (self.caps2 & !Caps2::CUBE_MAP_ALL_FACES) | Caps2::CUBE_MAP | Caps2::from(faces);
        self
    }
    /// A builder-pattern-style method to set the pixel format of the header.
    pub fn with_pixel_format(mut self, pixel_format: Dx9PixelFormat) -> Self {
        self.pixel_format = pixel_format;
        self
    }

    pub fn to_dx10(&self) -> Option<Dx10Header> {
        let alpha_mode = self.alpha_mode();

        let dxgi_format = match &self.pixel_format {
            Dx9PixelFormat::FourCC(four_cc) => {
                // special handling for DXT2 and DXT4
                if *four_cc == FourCC::DXT2 {
                    DxgiFormat::BC2_UNORM
                } else if *four_cc == FourCC::DXT4 {
                    DxgiFormat::BC3_UNORM
                } else if let Some(dxgi) = four_cc_to_dxgi(*four_cc) {
                    dxgi
                } else {
                    return None;
                }
            }
            Dx9PixelFormat::Mask(mask_pixel_format) => masked_to_dxgi(mask_pixel_format)?,
        };

        if self.caps2.contains(Caps2::CUBE_MAP) && !self.caps2.contains(Caps2::CUBE_MAP_ALL_FACES) {
            // DX10 requires all faces to be present
            return None;
        }

        let resource_dimension = if self.caps2.contains(Caps2::VOLUME) {
            ResourceDimension::Texture3D
        } else {
            ResourceDimension::Texture2D
        };
        let misc_flag = if self.caps2.contains(Caps2::CUBE_MAP) {
            MiscFlags::TEXTURE_CUBE
        } else {
            MiscFlags::empty()
        };

        Some(Dx10Header {
            height: self.height,
            width: self.width,
            depth: self.depth,
            mipmap_count: self.mipmap_count,
            dxgi_format,
            resource_dimension,
            misc_flag,
            array_size: 1,
            alpha_mode,
        })
    }
}

impl Dx10Header {
    /// Whether this header describes a cube map by checking for the
    /// [`MiscFlags::TEXTURE_CUBE`] flag.
    ///
    /// Note: DX10 guarantees that cube maps have exactly 6 faces.
    pub const fn is_cube_map(&self) -> bool {
        self.misc_flag.contains(MiscFlags::TEXTURE_CUBE)
    }
    /// Whether this header describes a volume texture by checking for
    /// [`ResourceDimension::Texture3D`].
    pub const fn is_volume(&self) -> bool {
        matches!(self.resource_dimension, ResourceDimension::Texture3D)
    }

    const fn pick_alpha_mode(dxgi: DxgiFormat) -> AlphaMode {
        if dxgi.has_alpha() {
            AlphaMode::Straight
        } else {
            AlphaMode::Unknown
        }
    }
    /// Creates a new header for DX10 texture 2D with the given dimensions and
    /// format.
    ///
    /// The mipmap count is set to 1 and the alpha mode is set to unknown.
    pub const fn new_image(width: u32, height: u32, format: DxgiFormat) -> Self {
        Self {
            height,
            width,
            depth: None,
            mipmap_count: NON_ZERO_U32_ONE,
            dxgi_format: format,
            resource_dimension: ResourceDimension::Texture2D,
            misc_flag: MiscFlags::empty(),
            array_size: 1,
            alpha_mode: Self::pick_alpha_mode(format),
        }
    }
    /// Creates a new header for DX10 texture 3D with the given dimensions and
    /// format.
    ///
    /// The mipmap count is set to 1 and the alpha mode is set to unknown.
    pub const fn new_volume(width: u32, height: u32, depth: u32, format: DxgiFormat) -> Self {
        Self {
            height,
            width,
            depth: Some(depth),
            mipmap_count: NON_ZERO_U32_ONE,
            dxgi_format: format,
            resource_dimension: ResourceDimension::Texture3D,
            misc_flag: MiscFlags::empty(),
            array_size: 1,
            alpha_mode: Self::pick_alpha_mode(format),
        }
    }
    /// Creates a new header for DX10 cube map with the given dimensions and
    /// format.
    ///
    /// The mipmap count is set to 1 and the alpha mode is set to unknown.
    pub const fn new_cube_map(width: u32, height: u32, format: DxgiFormat) -> Self {
        Self {
            height,
            width,
            depth: None,
            mipmap_count: NON_ZERO_U32_ONE,
            dxgi_format: format,
            resource_dimension: ResourceDimension::Texture2D,
            misc_flag: MiscFlags::TEXTURE_CUBE,
            array_size: 1,
            alpha_mode: Self::pick_alpha_mode(format),
        }
    }

    /// A builder-pattern-style method to set the width and height of the
    /// header.
    ///
    /// Depth will be set to `None`. If you want to leave the depth unchanged
    /// or change it as well, use [`Self::with_dimensions`].
    pub fn with_size(mut self, size: Size) -> Self {
        self.width = size.width;
        self.height = size.height;
        self.depth = None;
        self
    }
    /// A builder-pattern-style method to set the width, height, and depth of
    /// the header.
    ///
    /// For headers of 2D textures, [`Self::with_size`] is more appropriate.
    pub fn with_dimensions(mut self, width: u32, height: u32, depth: Option<u32>) -> Self {
        self.width = width;
        self.height = height;
        self.depth = depth;
        self
    }
    /// A builder-pattern-style method to set the mipmap count of the header.
    pub fn with_mipmap_count(mut self, mipmap_count: NonZeroU32) -> Self {
        self.mipmap_count = mipmap_count;
        self
    }
    /// A builder-pattern-style method to set the DXGI format of the header.
    ///
    /// The alpha mode will be set to `AlphaMode::Straight` if the format has
    /// alpha, and `AlphaMode::Unknown` otherwise.
    pub fn with_dxgi_format(mut self, dxgi_format: DxgiFormat) -> Self {
        self.dxgi_format = dxgi_format;
        self.alpha_mode = Self::pick_alpha_mode(dxgi_format);
        self
    }
    /// A builder-pattern-style method to set the resource dimension of the
    /// header.
    pub fn with_resource_dimension(mut self, resource_dimension: ResourceDimension) -> Self {
        self.resource_dimension = resource_dimension;
        self
    }
    /// A builder-pattern-style method to set the misc flags of the header.
    /// This will overwrite all current flags.
    pub fn with_misc_flags(mut self, misc_flags: MiscFlags) -> Self {
        self.misc_flag = misc_flags;
        self
    }
    /// A builder-pattern-style method to set the array size of the header.
    pub fn with_array_size(mut self, array_size: u32) -> Self {
        self.array_size = array_size;
        self
    }
    /// A builder-pattern-style method to set the alpha mode of the header.
    pub fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }

    pub fn to_dx9(&self) -> Option<Dx9Header> {
        fn to_dx9_format(
            mut dxgi_format: DxgiFormat,
            alpha_mode: AlphaMode,
        ) -> Option<Dx9PixelFormat> {
            // convert the format to linear before, because DX9 doesn't have
            // any sRGB formats
            dxgi_format = dxgi_format.to_linear();

            // special case for DXT2 and DXT4
            if alpha_mode == AlphaMode::Premultiplied {
                if dxgi_format == DxgiFormat::BC2_UNORM {
                    return Some(FourCC::DXT2.into());
                }
                if dxgi_format == DxgiFormat::BC3_UNORM {
                    return Some(FourCC::DXT4.into());
                }
            }

            dxgi_to_four_cc(dxgi_format)
                .map(Dx9PixelFormat::FourCC)
                .or_else(|| dxgi_to_masked(dxgi_format).map(Dx9PixelFormat::Mask))
        }

        if self.array_size != 1 {
            // DX9 does not support texture arrays
            return None;
        }
        if self.misc_flag.contains(MiscFlags::TEXTURE_CUBE)
            && self.resource_dimension != ResourceDimension::Texture2D
        {
            // A cube maps with needs to have 2D textures as sides
            return None;
        }

        let mut caps2 = Caps2::empty();
        if self.resource_dimension == ResourceDimension::Texture3D {
            caps2 |= Caps2::VOLUME;
        }
        if self.misc_flag.contains(MiscFlags::TEXTURE_CUBE) {
            caps2 |= Caps2::CUBE_MAP | Caps2::CUBE_MAP_ALL_FACES;
        }

        let format = to_dx9_format(self.dxgi_format, self.alpha_mode)?;

        Some(Dx9Header {
            height: self.height,
            width: self.width,
            depth: self.depth,
            mipmap_count: self.mipmap_count,
            caps2,
            pixel_format: format,
        })
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct DdsFlags: u32 {
        /// Required in every .dds file.
        const CAPS = 0x1;
        /// Required in every .dds file.
        const HEIGHT = 0x2;
        /// Required in every .dds file.
        const WIDTH = 0x4;
        /// Required when pitch is provided for an uncompressed texture.
        const PITCH = 0x8;
        /// Required in every .dds file.
        const PIXEL_FORMAT = 0x1000;
        /// Required in a mipmapped texture.
        const MIPMAP_COUNT = 0x20000;
        /// Required when pitch is provided for a compressed texture.
        const LINEAR_SIZE = 0x80000;
        /// Required in a depth texture.
        const DEPTH = 0x800000;

        /// Required in every .dds file.
        const REQUIRED = Self::CAPS.bits()
            | Self::HEIGHT.bits()
            | Self::WIDTH.bits()
            | Self::PIXEL_FORMAT.bits();
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Caps: u32 {
        /// Optional; must be used on any file that contains more than one surface (a mipmap, a cubic environment map, or mipmapped volume texture).
        const COMPLEX = 0x8;
        /// Optional; should be used for a mipmap.
        const MIPMAP = 0x400000;
        /// Required
        const TEXTURE = 0x1000;

        /// Required for all.
        const REQUIRED = Self::TEXTURE.bits();
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Caps2: u32 {
        /// Required for a cube map.
        const CUBE_MAP = 0x200;
        /// Required when these surfaces are stored in a cube map.
        const CUBE_MAP_POSITIVE_X = 0x400;
        /// Required when these surfaces are stored in a cube map.
        const CUBE_MAP_NEGATIVE_X = 0x800;
        /// Required when these surfaces are stored in a cube map.
        const CUBE_MAP_POSITIVE_Y = 0x1000;
        /// Required when these surfaces are stored in a cube map.
        const CUBE_MAP_NEGATIVE_Y = 0x2000;
        /// Required when these surfaces are stored in a cube map.
        const CUBE_MAP_POSITIVE_Z = 0x4000;
        /// Required when these surfaces are stored in a cube map.
        const CUBE_MAP_NEGATIVE_Z = 0x8000;
        /// Required for a volume texture.
        const VOLUME = 0x200000;

        /// Although Direct3D 9 supports partial cube-maps, Direct3D 10, 10.1, and 11 require that you define all six cube-map faces (that is, you must set DDS_CUBEMAP_ALLFACES).
        const CUBE_MAP_ALL_FACES = Self::CUBE_MAP_POSITIVE_X.bits()
            | Self::CUBE_MAP_NEGATIVE_X.bits()
            | Self::CUBE_MAP_POSITIVE_Y.bits()
            | Self::CUBE_MAP_NEGATIVE_Y.bits()
            | Self::CUBE_MAP_POSITIVE_Z.bits()
            | Self::CUBE_MAP_NEGATIVE_Z.bits();
    }

    /// Values which indicate what type of data is in the surface.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PixelFormatFlags: u32 {
        // Official docs are outdated. The following constants are from (1) the
        // official docs and (2) the source code of DirectXTex:
        // https://github.com/microsoft/DirectXTex/blob/af1c8b3cb4cae9354a7aade2f999ebf97d46e4fb/DirectXTex/DDS.h#L42

        /// Texture contains alpha data; dwRGBAlphaBitMask contains valid data.
        const ALPHAPIXELS = 0x1;
        /// Used in some older DDS files for alpha channel only uncompressed data (dwRGBBitCount contains the alpha channel bitcount; dwABitMask contains valid data)
        const ALPHA = 0x2;
        /// Texture contains compressed RGB data; dwFourCC contains valid data.
        const FOURCC = 0x4;
        const PAL8 = 0x20;
        /// Texture contains uncompressed RGB data; dwRGBBitCount and the RGB masks (dwRBitMask, dwGBitMask, dwBBitMask) contain valid data.
        const RGB = 0x40;
        const RGBA = Self::RGB.bits() | Self::ALPHAPIXELS.bits();
        /// Used in some older DDS files for YUV uncompressed data (dwRGBBitCount contains the YUV bit count; dwRBitMask contains the Y mask, dwGBitMask contains the U mask, dwBBitMask contains the V mask)
        const YUV = 0x200;
        /// Used in some older DDS files for single channel color uncompressed data (dwRGBBitCount contains the luminance channel bit count; dwRBitMask contains the channel mask). Can be combined with DDPF_ALPHAPIXELS for a two channel DDS file.
        const LUMINANCE = 0x20000;
        const LUMINANCE_ALPHA = Self::LUMINANCE.bits() | Self::ALPHAPIXELS.bits();
        const BUMP_LUMINANCE = 0x40000;
        /// While DirectXTex calls this flag `BUMPDUDV` (bumpmap dUdV), this just says that the texture contains SNORM data. Which channels the texture contains depends on which bit masks are non-zero. All dw*BitMask fields contain valid data.
        const BUMP_DUDV = 0x80000;
    }

    /// Identifies other, less common options for resources.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct MiscFlags: u32 {
        /// Sets a resource to be a cube texture created from a Texture2DArray that contains 6 textures.
        const TEXTURE_CUBE = 0x4;
    }
}

/// The alpha mode of the associated texture.
///
/// This is most often `Unknown`, even in DX10 headers.
///
/// <https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// Alpha channel content is unknown. This is the value for legacy files, which typically is assumed to be 'straight' alpha.
    Unknown = 0,
    /// Any alpha channel content is presumed to use straight alpha.
    Straight = 1,
    /// Any alpha channel content is using premultiplied alpha. The only legacy file formats that indicate this information are 'DX2' and 'DX4'.
    Premultiplied = 2,
    /// Any alpha channel content is all set to fully opaque.
    Opaque = 3,
    /// Any alpha channel content is being used as a 4th channel and is not intended to represent transparency (straight or premultiplied).
    Custom = 4,
}
impl TryFrom<u32> for AlphaMode {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(AlphaMode::Unknown),
            1 => Ok(AlphaMode::Straight),
            2 => Ok(AlphaMode::Premultiplied),
            3 => Ok(AlphaMode::Opaque),
            4 => Ok(AlphaMode::Custom),
            _ => Err(value),
        }
    }
}
impl From<AlphaMode> for u32 {
    fn from(value: AlphaMode) -> Self {
        value as u32
    }
}

/// Identifies the type of resource being used.
///
/// <https://learn.microsoft.com/en-us/windows/win32/api/d3d10/ne-d3d10-d3d10_resource_dimension>
/// <https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_resource_dimension>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceDimension {
    // Unknown = 0,
    // Buffer = 1,
    Texture1D = 2,
    Texture2D = 3,
    Texture3D = 4,
}
impl TryFrom<u32> for ResourceDimension {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(ResourceDimension::Texture1D),
            3 => Ok(ResourceDimension::Texture2D),
            4 => Ok(ResourceDimension::Texture3D),
            _ => Err(value),
        }
    }
}
impl From<ResourceDimension> for u32 {
    fn from(value: ResourceDimension) -> Self {
        value as u32
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub u32);

impl FourCC {
    pub const NONE: Self = FourCC(0);

    pub const DXT1: Self = FourCC(u32::from_le_bytes(*b"DXT1"));
    pub const DXT2: Self = FourCC(u32::from_le_bytes(*b"DXT2"));
    pub const DXT3: Self = FourCC(u32::from_le_bytes(*b"DXT3"));
    pub const DXT4: Self = FourCC(u32::from_le_bytes(*b"DXT4"));
    pub const DXT5: Self = FourCC(u32::from_le_bytes(*b"DXT5"));
    pub const RXGB: Self = FourCC(u32::from_le_bytes(*b"RXGB"));

    pub const DX10: Self = FourCC(u32::from_le_bytes(*b"DX10"));

    pub const ATI1: Self = FourCC(u32::from_le_bytes(*b"ATI1"));
    pub const BC4U: Self = FourCC(u32::from_le_bytes(*b"BC4U"));
    pub const BC4S: Self = FourCC(u32::from_le_bytes(*b"BC4S"));

    pub const ATI2: Self = FourCC(u32::from_le_bytes(*b"ATI2"));
    pub const BC5U: Self = FourCC(u32::from_le_bytes(*b"BC5U"));
    pub const BC5S: Self = FourCC(u32::from_le_bytes(*b"BC5S"));

    pub const RGBG: Self = FourCC(u32::from_le_bytes(*b"RGBG"));
    pub const GRGB: Self = FourCC(u32::from_le_bytes(*b"GRGB"));

    pub const YUY2: Self = FourCC(u32::from_le_bytes(*b"YUY2"));
    pub const UYVY: Self = FourCC(u32::from_le_bytes(*b"UYVY"));
}

impl From<u32> for FourCC {
    fn from(value: u32) -> Self {
        FourCC(value)
    }
}
impl From<FourCC> for u32 {
    fn from(value: FourCC) -> Self {
        value.0
    }
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self.0.to_le_bytes();
        if bytes.iter().all(|&b| b.is_ascii_alphanumeric()) {
            write!(
                f,
                "FourCC({}{}{}{})",
                bytes[0] as char, bytes[1] as char, bytes[2] as char, bytes[3] as char
            )
        } else {
            write!(f, "FourCC(0x{:x})", self.0)
        }
    }
}

/// Resource data formats, including fully-typed and typeless formats. A list
/// of modifiers at the bottom of the page more fully describes each format
/// type.
///
/// <https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format>
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DxgiFormat(u8);
impl DxgiFormat {
    pub const fn is_srgb(self) -> bool {
        self.0 != self.to_linear().0
    }
    pub const fn to_srgb(self) -> DxgiFormat {
        match self {
            DxgiFormat::BC1_UNORM => DxgiFormat::BC1_UNORM_SRGB,
            DxgiFormat::BC2_UNORM => DxgiFormat::BC2_UNORM_SRGB,
            DxgiFormat::BC3_UNORM => DxgiFormat::BC3_UNORM_SRGB,
            DxgiFormat::BC7_UNORM => DxgiFormat::BC7_UNORM_SRGB,

            DxgiFormat::R8G8B8A8_UNORM => DxgiFormat::R8G8B8A8_UNORM_SRGB,
            DxgiFormat::B8G8R8A8_UNORM => DxgiFormat::B8G8R8A8_UNORM_SRGB,
            DxgiFormat::B8G8R8X8_UNORM => DxgiFormat::B8G8R8X8_UNORM_SRGB,

            DxgiFormat::ASTC_4X4_UNORM => DxgiFormat::ASTC_4X4_UNORM_SRGB,
            DxgiFormat::ASTC_5X4_UNORM => DxgiFormat::ASTC_5X4_UNORM_SRGB,
            DxgiFormat::ASTC_5X5_UNORM => DxgiFormat::ASTC_5X5_UNORM_SRGB,
            DxgiFormat::ASTC_6X5_UNORM => DxgiFormat::ASTC_6X5_UNORM_SRGB,
            DxgiFormat::ASTC_6X6_UNORM => DxgiFormat::ASTC_6X6_UNORM_SRGB,
            DxgiFormat::ASTC_8X5_UNORM => DxgiFormat::ASTC_8X5_UNORM_SRGB,
            DxgiFormat::ASTC_8X6_UNORM => DxgiFormat::ASTC_8X6_UNORM_SRGB,
            DxgiFormat::ASTC_8X8_UNORM => DxgiFormat::ASTC_8X8_UNORM_SRGB,
            DxgiFormat::ASTC_10X5_UNORM => DxgiFormat::ASTC_10X5_UNORM_SRGB,
            DxgiFormat::ASTC_10X6_UNORM => DxgiFormat::ASTC_10X6_UNORM_SRGB,
            DxgiFormat::ASTC_10X8_UNORM => DxgiFormat::ASTC_10X8_UNORM_SRGB,
            DxgiFormat::ASTC_10X10_UNORM => DxgiFormat::ASTC_10X10_UNORM_SRGB,
            DxgiFormat::ASTC_12X10_UNORM => DxgiFormat::ASTC_12X10_UNORM_SRGB,
            DxgiFormat::ASTC_12X12_UNORM => DxgiFormat::ASTC_12X12_UNORM_SRGB,

            _ => self,
        }
    }
    pub const fn to_linear(self) -> DxgiFormat {
        match self {
            DxgiFormat::BC1_UNORM_SRGB => DxgiFormat::BC1_UNORM,
            DxgiFormat::BC2_UNORM_SRGB => DxgiFormat::BC2_UNORM,
            DxgiFormat::BC3_UNORM_SRGB => DxgiFormat::BC3_UNORM,
            DxgiFormat::BC7_UNORM_SRGB => DxgiFormat::BC7_UNORM,

            DxgiFormat::R8G8B8A8_UNORM_SRGB => DxgiFormat::R8G8B8A8_UNORM,
            DxgiFormat::B8G8R8A8_UNORM_SRGB => DxgiFormat::B8G8R8A8_UNORM,
            DxgiFormat::B8G8R8X8_UNORM_SRGB => DxgiFormat::B8G8R8X8_UNORM,

            DxgiFormat::ASTC_4X4_UNORM_SRGB => DxgiFormat::ASTC_4X4_UNORM,
            DxgiFormat::ASTC_5X4_UNORM_SRGB => DxgiFormat::ASTC_5X4_UNORM,
            DxgiFormat::ASTC_5X5_UNORM_SRGB => DxgiFormat::ASTC_5X5_UNORM,
            DxgiFormat::ASTC_6X5_UNORM_SRGB => DxgiFormat::ASTC_6X5_UNORM,
            DxgiFormat::ASTC_6X6_UNORM_SRGB => DxgiFormat::ASTC_6X6_UNORM,
            DxgiFormat::ASTC_8X5_UNORM_SRGB => DxgiFormat::ASTC_8X5_UNORM,
            DxgiFormat::ASTC_8X6_UNORM_SRGB => DxgiFormat::ASTC_8X6_UNORM,
            DxgiFormat::ASTC_8X8_UNORM_SRGB => DxgiFormat::ASTC_8X8_UNORM,
            DxgiFormat::ASTC_10X5_UNORM_SRGB => DxgiFormat::ASTC_10X5_UNORM,
            DxgiFormat::ASTC_10X6_UNORM_SRGB => DxgiFormat::ASTC_10X6_UNORM,
            DxgiFormat::ASTC_10X8_UNORM_SRGB => DxgiFormat::ASTC_10X8_UNORM,
            DxgiFormat::ASTC_10X10_UNORM_SRGB => DxgiFormat::ASTC_10X10_UNORM,
            DxgiFormat::ASTC_12X10_UNORM_SRGB => DxgiFormat::ASTC_12X10_UNORM,
            DxgiFormat::ASTC_12X12_UNORM_SRGB => DxgiFormat::ASTC_12X12_UNORM,

            _ => self,
        }
    }

    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            DxgiFormat::R32G32B32A32_TYPELESS
                | DxgiFormat::R32G32B32A32_FLOAT
                | DxgiFormat::R32G32B32A32_UINT
                | DxgiFormat::R32G32B32A32_SINT
                | DxgiFormat::R16G16B16A16_TYPELESS
                | DxgiFormat::R16G16B16A16_FLOAT
                | DxgiFormat::R16G16B16A16_UNORM
                | DxgiFormat::R16G16B16A16_UINT
                | DxgiFormat::R16G16B16A16_SNORM
                | DxgiFormat::R16G16B16A16_SINT
                | DxgiFormat::R10G10B10A2_TYPELESS
                | DxgiFormat::R10G10B10A2_UNORM
                | DxgiFormat::R10G10B10A2_UINT
                | DxgiFormat::R8G8B8A8_TYPELESS
                | DxgiFormat::R8G8B8A8_UNORM
                | DxgiFormat::R8G8B8A8_UNORM_SRGB
                | DxgiFormat::R8G8B8A8_UINT
                | DxgiFormat::R8G8B8A8_SNORM
                | DxgiFormat::R8G8B8A8_SINT
                | DxgiFormat::A8_UNORM
                | DxgiFormat::BC1_TYPELESS
                | DxgiFormat::BC1_UNORM
                | DxgiFormat::BC1_UNORM_SRGB
                | DxgiFormat::BC2_TYPELESS
                | DxgiFormat::BC2_UNORM
                | DxgiFormat::BC2_UNORM_SRGB
                | DxgiFormat::BC3_TYPELESS
                | DxgiFormat::BC3_UNORM
                | DxgiFormat::BC3_UNORM_SRGB
                | DxgiFormat::B5G5R5A1_UNORM
                | DxgiFormat::B8G8R8A8_UNORM
                | DxgiFormat::R10G10B10_XR_BIAS_A2_UNORM
                | DxgiFormat::B8G8R8A8_TYPELESS
                | DxgiFormat::B8G8R8A8_UNORM_SRGB
                | DxgiFormat::BC7_TYPELESS
                | DxgiFormat::BC7_UNORM
                | DxgiFormat::BC7_UNORM_SRGB
                | DxgiFormat::AYUV
                | DxgiFormat::AI44
                | DxgiFormat::IA44
                | DxgiFormat::A8P8
                | DxgiFormat::B4G4R4A4_UNORM
                | DxgiFormat::A4B4G4R4_UNORM
                | DxgiFormat::ASTC_4X4_TYPELESS
                | DxgiFormat::ASTC_4X4_UNORM
                | DxgiFormat::ASTC_4X4_UNORM_SRGB
                | DxgiFormat::ASTC_5X4_TYPELESS
                | DxgiFormat::ASTC_5X4_UNORM
                | DxgiFormat::ASTC_5X4_UNORM_SRGB
                | DxgiFormat::ASTC_5X5_TYPELESS
                | DxgiFormat::ASTC_5X5_UNORM
                | DxgiFormat::ASTC_5X5_UNORM_SRGB
                | DxgiFormat::ASTC_6X5_TYPELESS
                | DxgiFormat::ASTC_6X5_UNORM
                | DxgiFormat::ASTC_6X5_UNORM_SRGB
                | DxgiFormat::ASTC_6X6_TYPELESS
                | DxgiFormat::ASTC_6X6_UNORM
                | DxgiFormat::ASTC_6X6_UNORM_SRGB
                | DxgiFormat::ASTC_8X5_TYPELESS
                | DxgiFormat::ASTC_8X5_UNORM
                | DxgiFormat::ASTC_8X5_UNORM_SRGB
                | DxgiFormat::ASTC_8X6_TYPELESS
                | DxgiFormat::ASTC_8X6_UNORM
                | DxgiFormat::ASTC_8X6_UNORM_SRGB
                | DxgiFormat::ASTC_8X8_TYPELESS
                | DxgiFormat::ASTC_8X8_UNORM
                | DxgiFormat::ASTC_8X8_UNORM_SRGB
                | DxgiFormat::ASTC_10X5_TYPELESS
                | DxgiFormat::ASTC_10X5_UNORM
                | DxgiFormat::ASTC_10X5_UNORM_SRGB
                | DxgiFormat::ASTC_10X6_TYPELESS
                | DxgiFormat::ASTC_10X6_UNORM
                | DxgiFormat::ASTC_10X6_UNORM_SRGB
                | DxgiFormat::ASTC_10X8_TYPELESS
                | DxgiFormat::ASTC_10X8_UNORM
                | DxgiFormat::ASTC_10X8_UNORM_SRGB
                | DxgiFormat::ASTC_10X10_TYPELESS
                | DxgiFormat::ASTC_10X10_UNORM
                | DxgiFormat::ASTC_10X10_UNORM_SRGB
                | DxgiFormat::ASTC_12X10_TYPELESS
                | DxgiFormat::ASTC_12X10_UNORM
                | DxgiFormat::ASTC_12X10_UNORM_SRGB
                | DxgiFormat::ASTC_12X12_TYPELESS
                | DxgiFormat::ASTC_12X12_UNORM
                | DxgiFormat::ASTC_12X12_UNORM_SRGB
        )
    }

    #[allow(unused)]
    pub(crate) fn all() -> impl Iterator<Item = DxgiFormat> {
        (0..192).filter_map(|i| DxgiFormat::try_from(i).ok())
    }
}
impl TryFrom<u32> for DxgiFormat {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        // NOTE: This implementation is NOT generated by the marco for
        // performance and code size reasons. On virtually any optimization
        // level, the below code translates to around 6 instructions, while a
        // generated match arm (0 | 1 | 2 | ... | 115 | 130 | 131 | 132 => ...)
        // translates to a LUT on -O3 and a jump table with 133 entries on
        // <= -O2, -Os, and -Oz. It's slower and takes up vastly more binary
        // size.
        match value {
            0..=115
            | 130..=135
            | 137..=139
            | 141..=143
            | 145..=147
            | 149..=151
            | 153..=155
            | 157..=159
            | 161..=163
            | 165..=167
            | 169..=171
            | 173..=175
            | 177..=179
            | 181..=183
            | 185..=187
            | 191 => Ok(DxgiFormat(value as u8)),
            _ => Err(value),
        }
    }
}
impl From<DxgiFormat> for u32 {
    fn from(value: DxgiFormat) -> Self {
        value.0 as u32
    }
}

macro_rules! define_dxgi_formats {
    ($($name:ident = $n:literal),+) => {
        impl DxgiFormat {
            $(pub const $name: DxgiFormat = DxgiFormat($n);)+
        }

        impl std::fmt::Debug for DxgiFormat {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let name = match *self {
                    $(Self::$name => stringify!($name),)+
                    _ => {
                        return write!(f, "DxgiFormat({})", self.0);
                    }
                };
                write!(f, "{} ({})", name, self.0)
            }
        }
    };
}
define_dxgi_formats!(
    UNKNOWN = 0,
    R32G32B32A32_TYPELESS = 1,
    R32G32B32A32_FLOAT = 2,
    R32G32B32A32_UINT = 3,
    R32G32B32A32_SINT = 4,
    R32G32B32_TYPELESS = 5,
    R32G32B32_FLOAT = 6,
    R32G32B32_UINT = 7,
    R32G32B32_SINT = 8,
    R16G16B16A16_TYPELESS = 9,
    R16G16B16A16_FLOAT = 10,
    R16G16B16A16_UNORM = 11,
    R16G16B16A16_UINT = 12,
    R16G16B16A16_SNORM = 13,
    R16G16B16A16_SINT = 14,
    R32G32_TYPELESS = 15,
    R32G32_FLOAT = 16,
    R32G32_UINT = 17,
    R32G32_SINT = 18,
    R32G8X24_TYPELESS = 19,
    D32_FLOAT_S8X24_UINT = 20,
    R32_FLOAT_X8X24_TYPELESS = 21,
    X32_TYPELESS_G8X24_UINT = 22,
    R10G10B10A2_TYPELESS = 23,
    R10G10B10A2_UNORM = 24,
    R10G10B10A2_UINT = 25,
    R11G11B10_FLOAT = 26,
    R8G8B8A8_TYPELESS = 27,
    R8G8B8A8_UNORM = 28,
    R8G8B8A8_UNORM_SRGB = 29,
    R8G8B8A8_UINT = 30,
    R8G8B8A8_SNORM = 31,
    R8G8B8A8_SINT = 32,
    R16G16_TYPELESS = 33,
    R16G16_FLOAT = 34,
    R16G16_UNORM = 35,
    R16G16_UINT = 36,
    R16G16_SNORM = 37,
    R16G16_SINT = 38,
    R32_TYPELESS = 39,
    D32_FLOAT = 40,
    R32_FLOAT = 41,
    R32_UINT = 42,
    R32_SINT = 43,
    R24G8_TYPELESS = 44,
    D24_UNORM_S8_UINT = 45,
    R24_UNORM_X8_TYPELESS = 46,
    X24_TYPELESS_G8_UINT = 47,
    R8G8_TYPELESS = 48,
    R8G8_UNORM = 49,
    R8G8_UINT = 50,
    R8G8_SNORM = 51,
    R8G8_SINT = 52,
    R16_TYPELESS = 53,
    R16_FLOAT = 54,
    D16_UNORM = 55,
    R16_UNORM = 56,
    R16_UINT = 57,
    R16_SNORM = 58,
    R16_SINT = 59,
    R8_TYPELESS = 60,
    R8_UNORM = 61,
    R8_UINT = 62,
    R8_SNORM = 63,
    R8_SINT = 64,
    A8_UNORM = 65,
    R1_UNORM = 66,
    R9G9B9E5_SHAREDEXP = 67,
    R8G8_B8G8_UNORM = 68,
    G8R8_G8B8_UNORM = 69,
    BC1_TYPELESS = 70,
    BC1_UNORM = 71,
    BC1_UNORM_SRGB = 72,
    BC2_TYPELESS = 73,
    BC2_UNORM = 74,
    BC2_UNORM_SRGB = 75,
    BC3_TYPELESS = 76,
    BC3_UNORM = 77,
    BC3_UNORM_SRGB = 78,
    BC4_TYPELESS = 79,
    BC4_UNORM = 80,
    BC4_SNORM = 81,
    BC5_TYPELESS = 82,
    BC5_UNORM = 83,
    BC5_SNORM = 84,
    B5G6R5_UNORM = 85,
    B5G5R5A1_UNORM = 86,
    B8G8R8A8_UNORM = 87,
    B8G8R8X8_UNORM = 88,
    R10G10B10_XR_BIAS_A2_UNORM = 89,
    B8G8R8A8_TYPELESS = 90,
    B8G8R8A8_UNORM_SRGB = 91,
    B8G8R8X8_TYPELESS = 92,
    B8G8R8X8_UNORM_SRGB = 93,
    BC6H_TYPELESS = 94,
    BC6H_UF16 = 95,
    BC6H_SF16 = 96,
    BC7_TYPELESS = 97,
    BC7_UNORM = 98,
    BC7_UNORM_SRGB = 99,
    AYUV = 100,
    Y410 = 101,
    Y416 = 102,
    NV12 = 103,
    P010 = 104,
    P016 = 105,
    OPAQUE_420 = 106,
    YUY2 = 107,
    Y210 = 108,
    Y216 = 109,
    NV11 = 110,
    AI44 = 111,
    IA44 = 112,
    P8 = 113,
    A8P8 = 114,
    B4G4R4A4_UNORM = 115,
    P208 = 130,
    V208 = 131,
    V408 = 132,
    ASTC_4X4_TYPELESS = 133,
    ASTC_4X4_UNORM = 134,
    ASTC_4X4_UNORM_SRGB = 135,
    ASTC_5X4_TYPELESS = 137,
    ASTC_5X4_UNORM = 138,
    ASTC_5X4_UNORM_SRGB = 139,
    ASTC_5X5_TYPELESS = 141,
    ASTC_5X5_UNORM = 142,
    ASTC_5X5_UNORM_SRGB = 143,
    ASTC_6X5_TYPELESS = 145,
    ASTC_6X5_UNORM = 146,
    ASTC_6X5_UNORM_SRGB = 147,
    ASTC_6X6_TYPELESS = 149,
    ASTC_6X6_UNORM = 150,
    ASTC_6X6_UNORM_SRGB = 151,
    ASTC_8X5_TYPELESS = 153,
    ASTC_8X5_UNORM = 154,
    ASTC_8X5_UNORM_SRGB = 155,
    ASTC_8X6_TYPELESS = 157,
    ASTC_8X6_UNORM = 158,
    ASTC_8X6_UNORM_SRGB = 159,
    ASTC_8X8_TYPELESS = 161,
    ASTC_8X8_UNORM = 162,
    ASTC_8X8_UNORM_SRGB = 163,
    ASTC_10X5_TYPELESS = 165,
    ASTC_10X5_UNORM = 166,
    ASTC_10X5_UNORM_SRGB = 167,
    ASTC_10X6_TYPELESS = 169,
    ASTC_10X6_UNORM = 170,
    ASTC_10X6_UNORM_SRGB = 171,
    ASTC_10X8_TYPELESS = 173,
    ASTC_10X8_UNORM = 174,
    ASTC_10X8_UNORM_SRGB = 175,
    ASTC_10X10_TYPELESS = 177,
    ASTC_10X10_UNORM = 178,
    ASTC_10X10_UNORM_SRGB = 179,
    ASTC_12X10_TYPELESS = 181,
    ASTC_12X10_UNORM = 182,
    ASTC_12X10_UNORM_SRGB = 183,
    ASTC_12X12_TYPELESS = 185,
    ASTC_12X12_UNORM = 186,
    ASTC_12X12_UNORM_SRGB = 187,
    A4B4G4R4_UNORM = 191
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxgi_format_srgb() {
        for dxgi in DxgiFormat::all() {
            assert_eq!(dxgi.to_linear(), dxgi.to_srgb().to_linear());
            assert_eq!(dxgi.to_srgb(), dxgi.to_linear().to_srgb());
            assert!(!dxgi.to_linear().is_srgb());

            if dxgi.is_srgb() {
                assert_eq!(dxgi, dxgi.to_srgb());
                assert_ne!(dxgi, dxgi.to_linear());
            } else {
                assert_eq!(dxgi, dxgi.to_linear());
            }
        }
    }
}
