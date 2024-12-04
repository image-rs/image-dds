use crate::{util::read_u32s, DdsDecodeError, DxgiFormat};
use bitflags::bitflags;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::io::Read;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct HeaderReadOptions {
    pub magic: MagicBytes,
}
impl Default for HeaderReadOptions {
    fn default() -> Self {
        Self {
            magic: MagicBytes::Check,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MagicBytes {
    Check,
    Skip,
}

/// The DDS header + DX10 header if any.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FullHeader {
    pub header: DdsHeader,
    pub header_dxt10: Option<DdsHeaderDxt10>,
}
impl FullHeader {
    pub const MAGIC: [u8; 4] = *b"DDS ";

    pub fn new(header: DdsHeader, header_dxt10: Option<DdsHeaderDxt10>) -> Self {
        Self {
            header,
            header_dxt10,
        }
    }

    pub fn read<R: Read>(reader: &mut R) -> Result<Self, DdsDecodeError> {
        Self::read_with_options(reader, &Default::default())
    }
    pub fn read_with_options<R: Read>(
        reader: &mut R,
        options: &HeaderReadOptions,
    ) -> Result<Self, DdsDecodeError> {
        match options.magic {
            MagicBytes::Skip => (),
            MagicBytes::Check => {
                let mut magic = [0u8; 4];
                reader.read_exact(&mut magic)?;
                assert_eq!(magic, Self::MAGIC);
            }
        };

        let header = DdsHeader::read(reader)?;
        let header_dxt10 = if header.pixel_format.four_cc == FourCC::DX10 {
            Some(DdsHeaderDxt10::read(reader)?)
        } else {
            None
        };

        Ok(Self {
            header,
            header_dxt10,
        })
    }
}

/// The DDS_HEADER structure contains information about the dimensions, format, and mipmap count of a texture.
///
/// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DdsHeader {
    /// Size of structure. This member must be set to 124.
    pub size: u32,
    /// Flags to indicate which members contain valid data.
    pub flags: DdsFlags,
    /// Surface height (in pixels).
    pub height: u32,
    /// Surface width (in pixels).
    pub width: u32,
    /// The pitch or number of bytes per scan line in an uncompressed texture; the total number of bytes in the top level texture for a compressed texture. For information about how to compute the pitch, see the DDS File Layout section of the [Programming Guide for DDS](https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide).
    pub pitch_or_linear_size: u32,
    /// Depth of a volume texture (in pixels), otherwise unused.
    pub depth: u32,
    /// Number of mipmap levels, otherwise unused.
    pub mipmap_count: u32,
    /// Unused.
    pub reserved1: [u32; 11],
    pub pixel_format: DdsPixelFormat,
    /// Specifies the complexity of the surfaces stored.
    pub caps: DdsCaps,
    /// Additional detail about the surfaces stored.
    pub caps2: DdsCaps2,
    /// Unused.
    pub caps3: u32,
    /// Unused.
    pub caps4: u32,
    /// Unused.
    pub reserved2: u32,
}
impl DdsHeader {
    pub const SIZE: usize = 124;
    const INTS: usize = Self::SIZE / 4;

    pub fn read<R: Read>(reader: &mut R) -> Result<Self, DdsDecodeError> {
        let mut buffer = [0; Self::INTS];
        read_u32s(reader, &mut buffer)?;
        Self::read_buffer(buffer)
    }
    fn read_buffer(buffer: [u32; Self::INTS]) -> Result<Self, DdsDecodeError> {
        if buffer[0] != Self::SIZE as u32 {
            return Err(DdsDecodeError::InvalidHeader(
                "Invalid DdsHeader size, expected 124",
            ));
        }
        if buffer[18] != DdsPixelFormat::SIZE as u32 {
            return Err(DdsDecodeError::InvalidHeader(
                "Invalid DdsPixelFormat size, expected 32",
            ));
        }

        Ok(Self {
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
            pixel_format: DdsPixelFormat {
                size: buffer[18],
                flags: DdsPixelFormatFlags::from_bits_retain(buffer[19]),
                four_cc: buffer[20].into(),
                rgb_bit_count: buffer[21],
                r_bit_mask: buffer[22],
                g_bit_mask: buffer[23],
                b_bit_mask: buffer[24],
                a_bit_mask: buffer[25],
            },
            caps: DdsCaps::from_bits_retain(buffer[26]),
            caps2: DdsCaps2::from_bits_retain(buffer[27]),
            caps3: buffer[28],
            caps4: buffer[29],
            reserved2: buffer[30],
        })
    }

    pub fn get_depth(&self) -> Option<u32> {
        if self.flags.contains(DdsFlags::DEPTH) {
            Some(self.depth)
        } else {
            None
        }
    }
    pub fn get_mipmap_count(&self) -> Option<u32> {
        if self.flags.contains(DdsFlags::MIPMAP_COUNT)
            || self.caps.contains(DdsCaps::COMPLEX)
            || self.caps.contains(DdsCaps::MIPMAP)
        {
            Some(self.mipmap_count)
        } else {
            None
        }
    }
}

/// The DDS_PIXELFORMAT structure describes the pixel format of the surface or volume texture.
///
/// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DdsPixelFormat {
    /// Size of structure. This member must be set to 32.
    pub size: u32,
    /// Values which indicate what type of data is in the surface.
    pub flags: DdsPixelFormatFlags,
    /// Four-character codes for specifying compressed or custom formats. Possible values include: DXT1, DXT2, DXT3, DXT4, or DXT5. A FourCC of DX10 indicates the prescense of the DDS_HEADER_DXT10 extended header, and the dxgiFormat member of that structure indicates the true format. When using a four-character code, dwFlags must include DDPF_FOURCC.
    pub four_cc: FourCC,
    /// Number of bits in an RGB (possibly including alpha) format. Valid when dwFlags includes DDPF_RGB, DDPF_LUMINANCE, or DDPF_YUV.
    pub rgb_bit_count: u32,
    /// Red (or luminance or Y) mask for reading color data. For instance, given the A8R8G8B8 format, the red mask would be 0x00ff0000.
    pub r_bit_mask: u32,
    /// Green (or U) mask for reading color data. For instance, given the A8R8G8B8 format, the green mask would be 0x0000ff00.
    pub g_bit_mask: u32,
    /// Blue (or V) mask for reading color data. For instance, given the A8R8G8B8 format, the blue mask would be 0x000000ff.
    pub b_bit_mask: u32,
    /// Alpha mask for reading alpha data. dwFlags must include DDPF_ALPHAPIXELS or DDPF_ALPHA. For instance, given the A8R8G8B8 format, the alpha mask would be 0xff000000.
    pub a_bit_mask: u32,
}
impl DdsPixelFormat {
    pub const SIZE: usize = 32;
    const INTS: usize = Self::SIZE / 4;
}

/// DDS header extension to handle resource arrays, DXGI pixel formats that don't map to the legacy Microsoft DirectDraw pixel format structures, and additional metadata.
///
/// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DdsHeaderDxt10 {
    /// The surface pixel format.
    pub dxgi_format: DxgiFormat,
    /// Identifies the type of resource.
    pub resource_dimension: ResourceDimension,
    pub misc_flag: u32,
    pub array_size: u32,
    pub misc_flags2: u32,
}
impl DdsHeaderDxt10 {
    pub const SIZE: usize = 20;
    const INTS: usize = 5;

    pub fn read<R: Read>(reader: &mut R) -> Result<Self, DdsDecodeError> {
        let mut buffer = [0; Self::INTS];
        read_u32s(reader, &mut buffer)?;
        Self::read_buffer(buffer)
    }
    fn read_buffer(buffer: [u32; Self::INTS]) -> Result<Self, DdsDecodeError> {
        use DdsDecodeError::InvalidHeader;

        let dxgi_format = DxgiFormat::try_from(buffer[0])
            .map_err(|_| InvalidHeader("Invalid DXGI format in DdsHeaderDxt10"))?;
        let resource_dimension = ResourceDimension::try_from(buffer[1])
            .map_err(|_| InvalidHeader("Invalid resource dimension in DdsHeaderDxt10"))?;

        Ok(Self {
            dxgi_format,
            resource_dimension,
            misc_flag: buffer[2],
            array_size: buffer[3],
            misc_flags2: buffer[4],
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
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct DdsCaps: u32 {
        /// Optional; must be used on any file that contains more than one surface (a mipmap, a cubic environment map, or mipmapped volume texture).
        const COMPLEX = 0x8;
        /// Optional; should be used for a mipmap.
        const MIPMAP = 0x400000;
        /// Required
        const TEXTURE = 0x1000;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct DdsCaps2: u32 {
        /// Required for a cube map.
        const CUBEMAP = 0x200;
        /// Required when these surfaces are stored in a cube map.
        const CUBEMAP_POSITIVEX = 0x400;
        /// Required when these surfaces are stored in a cube map.
        const CUBEMAP_NEGATIVEX = 0x800;
        /// Required when these surfaces are stored in a cube map.
        const CUBEMAP_POSITIVEY = 0x1000;
        /// Required when these surfaces are stored in a cube map.
        const CUBEMAP_NEGATIVEY = 0x2000;
        /// Required when these surfaces are stored in a cube map.
        const CUBEMAP_POSITIVEZ = 0x4000;
        /// Required when these surfaces are stored in a cube map.
        const CUBEMAP_NEGATIVEZ = 0x8000;
        /// Required for a volume texture.
        const VOLUME = 0x200000;

        /// Although Direct3D 9 supports partial cube-maps, Direct3D 10, 10.1, and 11 require that you define all six cube-map faces (that is, you must set DDS_CUBEMAP_ALLFACES).
        const CUBEMAP_ALL_FACES = Self::CUBEMAP_POSITIVEX.bits()
            | Self::CUBEMAP_NEGATIVEX.bits()
            | Self::CUBEMAP_POSITIVEY.bits()
            | Self::CUBEMAP_NEGATIVEY.bits()
            | Self::CUBEMAP_POSITIVEZ.bits()
            | Self::CUBEMAP_NEGATIVEZ.bits();
    }

    /// Values which indicate what type of data is in the surface.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct DdsPixelFormatFlags: u32 {
        /// Texture contains alpha data; dwRGBAlphaBitMask contains valid data.
        const ALPHAPIXELS = 0x1;
        /// Used in some older DDS files for alpha channel only uncompressed data (dwRGBBitCount contains the alpha channel bitcount; dwABitMask contains valid data)
        const ALPHA = 0x2;
        /// Texture contains compressed RGB data; dwFourCC contains valid data.
        const FOURCC = 0x4;
        /// Texture contains uncompressed RGB data; dwRGBBitCount and the RGB masks (dwRBitMask, dwGBitMask, dwBBitMask) contain valid data.
        const RGB = 0x40;
        /// Used in some older DDS files for YUV uncompressed data (dwRGBBitCount contains the YUV bit count; dwRBitMask contains the Y mask, dwGBitMask contains the U mask, dwBBitMask contains the V mask)
        const YUV = 0x200;
        /// Used in some older DDS files for single channel color uncompressed data (dwRGBBitCount contains the luminance channel bit count; dwRBitMask contains the channel mask). Can be combined with DDPF_ALPHAPIXELS for a two channel DDS file.
        const LUMINANCE = 0x20000;
    }
}

/// Identifies the type of resource being used.
///
/// https://learn.microsoft.com/en-us/windows/win32/api/d3d10/ne-d3d10-d3d10_resource_dimension
/// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_resource_dimension
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoPrimitive, TryFromPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ResourceDimension {
    Unknown = 0,
    Buffer = 1,
    Texture1D = 2,
    Texture2D = 3,
    Texture3D = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FourCC(pub u32);

impl FourCC {
    pub const NONE: Self = FourCC(0);

    pub const DXT1: Self = FourCC(u32::from_le_bytes(*b"DXT1"));
    pub const DXT2: Self = FourCC(u32::from_le_bytes(*b"DXT2"));
    pub const DXT3: Self = FourCC(u32::from_le_bytes(*b"DXT3"));
    pub const DXT4: Self = FourCC(u32::from_le_bytes(*b"DXT4"));
    pub const DXT5: Self = FourCC(u32::from_le_bytes(*b"DXT5"));

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
impl From<[u8; 4]> for FourCC {
    fn from(value: [u8; 4]) -> Self {
        FourCC(u32::from_le_bytes(value))
    }
}
impl From<&[u8; 4]> for FourCC {
    fn from(value: &[u8; 4]) -> Self {
        FourCC(u32::from_le_bytes(*value))
    }
}

impl TryFrom<FourCC> for DxgiFormat {
    type Error = ();

    fn try_from(value: FourCC) -> Result<Self, Self::Error> {
        match value {
            FourCC::DXT1 => Ok(DxgiFormat::BC1_UNORM),
            FourCC::DXT2 => Ok(DxgiFormat::BC2_UNORM),
            FourCC::DXT3 => Ok(DxgiFormat::BC2_UNORM),
            FourCC::DXT4 => Ok(DxgiFormat::BC3_UNORM),
            FourCC::DXT5 => Ok(DxgiFormat::BC3_UNORM),

            FourCC::ATI1 => Ok(DxgiFormat::BC4_UNORM),
            FourCC::BC4U => Ok(DxgiFormat::BC4_UNORM),
            FourCC::BC4S => Ok(DxgiFormat::BC4_SNORM),

            FourCC::ATI2 => Ok(DxgiFormat::BC5_UNORM),
            FourCC::BC5U => Ok(DxgiFormat::BC5_UNORM),
            FourCC::BC5S => Ok(DxgiFormat::BC5_SNORM),

            FourCC::RGBG => Ok(DxgiFormat::R8G8_B8G8_UNORM),
            FourCC::GRGB => Ok(DxgiFormat::G8R8_G8B8_UNORM),

            FourCC::YUY2 => Ok(DxgiFormat::YUY2),

            FourCC(36) => Ok(DxgiFormat::R16G16B16A16_UNORM),
            FourCC(110) => Ok(DxgiFormat::R16G16B16A16_SNORM),
            FourCC(111) => Ok(DxgiFormat::R16_FLOAT),
            FourCC(112) => Ok(DxgiFormat::R16G16_FLOAT),
            FourCC(113) => Ok(DxgiFormat::R16G16B16A16_FLOAT),
            FourCC(114) => Ok(DxgiFormat::R32_FLOAT),
            FourCC(115) => Ok(DxgiFormat::R32G32_FLOAT),
            FourCC(116) => Ok(DxgiFormat::R32G32B32A32_FLOAT),

            _ => Err(()),
        }
    }
}
