use crate::{util::read_u32_le_array, HeaderError, Size};
use bitflags::bitflags;
use std::io::Read;

/// The DDS header and the DX10 extension header if any.
///
/// This structure contains parsed data. Using by the decoder.
///
/// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Header {
    /// Flags to indicate which members contain valid data.
    pub flags: DdsFlags,
    /// Surface height (in pixels).
    pub height: u32,
    /// Surface width (in pixels).
    pub width: u32,
    /// Depth of a volume texture (in pixels).
    pub depth: Option<u32>,
    /// Number of mipmap levels.
    pub mipmap_count: Option<u32>,
    // /// Unused.
    // pub reserved1: [u32; 11],
    pub pixel_format: PixelFormat,
    /// Specifies the complexity of the surfaces stored.
    pub caps: DdsCaps,
    /// Additional detail about the surfaces stored.
    pub caps2: DdsCaps2,
    // /// Unused.
    // pub caps3: u32,
    // /// Unused.
    // pub caps4: u32,
    // /// Unused.
    // pub reserved2: u32,
    /// Optional DX10 extension header.
    pub dxt10: Option<HeaderDxt10>,
}

impl Header {
    const SIZE: usize = 124;
    const INTS: usize = Self::SIZE / 4;

    /// The magic bytes (`'DDS '`) at the start of every DDS file.
    pub const MAGIC: [u8; 4] = *b"DDS ";

    /// The magic bytes `'DDS '` are at the start of every DDS file. This
    /// function reads the magic bytes and checks if they are correct.
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

    /// Reads the header without magic bytes from a reader.
    ///
    /// If the header is read successfully, the reader will be at the start of the pixel data.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, HeaderError> {
        let buffer: [u32; Self::INTS] = read_u32_le_array(reader)?;

        if buffer[0] != Self::SIZE as u32 {
            return Err(HeaderError::InvalidHeaderSize(buffer[0]));
        }

        let flags = DdsFlags::from_bits_retain(buffer[1]);

        let height = buffer[2];
        let width = buffer[3];
        let depth = if flags.contains(DdsFlags::DEPTH) {
            Some(buffer[5])
        } else {
            None
        };

        let pixel_format = PixelFormat::read_buffer([
            buffer[18], buffer[19], buffer[20], buffer[21], buffer[22], buffer[23], buffer[24],
            buffer[25],
        ])?;

        let caps = DdsCaps::from_bits_retain(buffer[26]);
        let caps2 = DdsCaps2::from_bits_retain(buffer[27]);

        let mipmap_count = if flags.contains(DdsFlags::MIPMAP_COUNT)
            || caps.contains(DdsCaps::COMPLEX | DdsCaps::MIPMAP)
        {
            Some(buffer[6])
        } else {
            None
        };

        let dxt10 = if pixel_format.four_cc == Some(FourCC::DX10) {
            let dx10_buffer = read_u32_le_array(reader)?;
            let dxt10 = HeaderDxt10::read_buffer(dx10_buffer)?;
            Some(dxt10)
        } else {
            None
        };

        Ok(Self {
            flags,
            height,
            width,
            depth,
            mipmap_count,
            pixel_format,
            caps,
            caps2,
            dxt10,
        })
    }

    pub fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }
}

/// The DDS_PIXELFORMAT structure describes the pixel format of the surface or volume texture.
///
/// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PixelFormat {
    /// Values which indicate what type of data is in the surface.
    pub flags: PixelFormatFlags,
    /// Four-character codes for specifying compressed or custom formats. Possible values include: DXT1, DXT2, DXT3, DXT4, or DXT5.
    ///
    /// This is `None` if `flags` does not contain [`PixelFormatFlags::FOURCC`].
    pub four_cc: Option<FourCC>,
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
impl PixelFormat {
    const SIZE: usize = 32;
    const INTS: usize = Self::SIZE / 4;

    fn read_buffer(buffer: [u32; Self::INTS]) -> Result<Self, HeaderError> {
        if buffer[0] != PixelFormat::SIZE as u32 {
            return Err(HeaderError::InvalidPixelFormatSize(buffer[0]));
        }

        let flags = PixelFormatFlags::from_bits_retain(buffer[1]);
        let four_cc = if flags.contains(PixelFormatFlags::FOURCC) {
            Some(FourCC::from(buffer[2]))
        } else {
            None
        };

        Ok(Self {
            flags,
            four_cc,
            rgb_bit_count: buffer[3],
            r_bit_mask: buffer[4],
            g_bit_mask: buffer[5],
            b_bit_mask: buffer[6],
            a_bit_mask: buffer[7],
        })
    }

    /// Creates a new pixel format with the given four-character code.
    ///
    /// This is a convenience function for creating a pixel format with the
    /// [`PixelFormatFlags::FOURCC`] flag set.
    pub fn new_four_cc(four_cc: FourCC) -> Self {
        Self {
            flags: PixelFormatFlags::FOURCC,
            four_cc: Some(four_cc),
            rgb_bit_count: 0,
            r_bit_mask: 0,
            g_bit_mask: 0,
            b_bit_mask: 0,
            a_bit_mask: 0,
        }
    }
}

/// DDS header extension to handle resource arrays, DXGI pixel formats that don't map to the legacy Microsoft DirectDraw pixel format structures, and additional metadata.
///
/// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HeaderDxt10 {
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
    /// Contains additional metadata (formerly was reserved). The lower 3 bits indicate the alpha mode of the associated resource. The upper 29 bits are reserved and are typically 0.
    pub misc_flags2: MiscFlags2,
}
impl HeaderDxt10 {
    const SIZE: usize = 20;
    const INTS: usize = Self::SIZE / 4;

    fn read_buffer(buffer: [u32; Self::INTS]) -> Result<Self, HeaderError> {
        let dxgi_format =
            DxgiFormat::try_from(buffer[0]).map_err(HeaderError::InvalidDxgiFormat)?;
        let resource_dimension = ResourceDimension::try_from(buffer[1])
            .map_err(HeaderError::InvalidResourceDimension)?;

        let misc_flag = MiscFlags::from_bits_retain(buffer[2]);
        let misc_flags2 = MiscFlags2::from_bits_retain(buffer[4]);

        let array_size = buffer[3];
        if resource_dimension == ResourceDimension::Texture3D && array_size != 1 {
            return Err(HeaderError::InvalidArraySizeForTexture3D(array_size));
        }

        Ok(Self {
            dxgi_format,
            resource_dimension,
            misc_flag,
            array_size,
            misc_flags2,
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
    pub struct DdsCaps: u32 {
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
    pub struct DdsCaps2: u32 {
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
        /// Texture contains uncompressed RGB data; dwRGBBitCount and the RGB masks (dwRBitMask, dwGBitMask, dwBBitMask) contain valid data.
        const RGB = 0x40;
        const RGBA = Self::RGB.bits() | Self::ALPHAPIXELS.bits();
        /// Used in some older DDS files for YUV uncompressed data (dwRGBBitCount contains the YUV bit count; dwRBitMask contains the Y mask, dwGBitMask contains the U mask, dwBBitMask contains the V mask)
        const YUV = 0x200;
        /// Used in some older DDS files for single channel color uncompressed data (dwRGBBitCount contains the luminance channel bit count; dwRBitMask contains the channel mask). Can be combined with DDPF_ALPHAPIXELS for a two channel DDS file.
        const LUMINANCE = 0x20000;
        const LUMINANCE_ALPHA = Self::LUMINANCE.bits() | Self::ALPHAPIXELS.bits();
        const PAL8 = 0x20;
        /// While DirectXTex calls this flag `BUMPDUDV` (bumpmap dUdV), this just says that the texture contains SNORM data. Which channels the texture contains depends on which bit masks are non-zero. All dw*BitMask fields contain valid data.
        const SNORM = 0x80000;
    }

    /// Identifies other, less common options for resources.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct MiscFlags: u32 {
        /// Sets a resource to be a cube texture created from a Texture2DArray that contains 6 textures.
        const TEXTURE_CUBE = 0x4;
    }

    /// Additional metadata.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct MiscFlags2: u32 {
        /// Alpha channel content is unknown. This is the value for legacy files, which typically is assumed to be 'straight' alpha.
        const ALPHA_MODE_UNKNOWN = 0x0;
        /// Any alpha channel content is presumed to use straight alpha.
        const ALPHA_MODE_STRAIGHT = 0x1;
        /// Any alpha channel content is using premultiplied alpha. The only legacy file formats that indicate this information are 'DX2' and 'DX4'.
        const ALPHA_MODE_PREMULTIPLIED = 0x2;
        /// Any alpha channel content is all set to fully opaque.
        const ALPHA_MODE_OPAQUE = 0x3;
        /// Any alpha channel content is being used as a 4th channel and is not intended to represent transparency (straight or premultiplied).
        const ALPHA_MODE_CUSTOM = 0x4;
    }
}

/// Identifies the type of resource being used.
///
/// https://learn.microsoft.com/en-us/windows/win32/api/d3d10/ne-d3d10-d3d10_resource_dimension
/// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_resource_dimension
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self.0.to_le_bytes();
        if bytes.iter().all(|&b| b.is_ascii_alphanumeric()) {
            write!(
                f,
                "FourCC(0x{:x}; {}{}{}{})",
                self.0, bytes[0] as char, bytes[1] as char, bytes[2] as char, bytes[3] as char
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
/// https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DxgiFormat(u8);
impl DxgiFormat {
    pub fn is_srgb(&self) -> bool {
        matches!(
            *self,
            DxgiFormat::BC1_UNORM_SRGB
                | DxgiFormat::BC2_UNORM_SRGB
                | DxgiFormat::BC3_UNORM_SRGB
                | DxgiFormat::BC7_UNORM_SRGB
                | DxgiFormat::R8G8B8A8_UNORM_SRGB
                | DxgiFormat::B8G8R8A8_UNORM_SRGB
                | DxgiFormat::B8G8R8X8_UNORM_SRGB
        )
    }
}

macro_rules! define_dxgi_formats {
    ($($name:ident - $n:literal),+) => {
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
    UNKNOWN - 0,
    R32G32B32A32_TYPELESS - 1,
    R32G32B32A32_FLOAT - 2,
    R32G32B32A32_UINT - 3,
    R32G32B32A32_SINT - 4,
    R32G32B32_TYPELESS - 5,
    R32G32B32_FLOAT - 6,
    R32G32B32_UINT - 7,
    R32G32B32_SINT - 8,
    R16G16B16A16_TYPELESS - 9,
    R16G16B16A16_FLOAT - 10,
    R16G16B16A16_UNORM - 11,
    R16G16B16A16_UINT - 12,
    R16G16B16A16_SNORM - 13,
    R16G16B16A16_SINT - 14,
    R32G32_TYPELESS - 15,
    R32G32_FLOAT - 16,
    R32G32_UINT - 17,
    R32G32_SINT - 18,
    R32G8X24_TYPELESS - 19,
    D32_FLOAT_S8X24_UINT - 20,
    R32_FLOAT_X8X24_TYPELESS - 21,
    X32_TYPELESS_G8X24_UINT - 22,
    R10G10B10A2_TYPELESS - 23,
    R10G10B10A2_UNORM - 24,
    R10G10B10A2_UINT - 25,
    R11G11B10_FLOAT - 26,
    R8G8B8A8_TYPELESS - 27,
    R8G8B8A8_UNORM - 28,
    R8G8B8A8_UNORM_SRGB - 29,
    R8G8B8A8_UINT - 30,
    R8G8B8A8_SNORM - 31,
    R8G8B8A8_SINT - 32,
    R16G16_TYPELESS - 33,
    R16G16_FLOAT - 34,
    R16G16_UNORM - 35,
    R16G16_UINT - 36,
    R16G16_SNORM - 37,
    R16G16_SINT - 38,
    R32_TYPELESS - 39,
    D32_FLOAT - 40,
    R32_FLOAT - 41,
    R32_UINT - 42,
    R32_SINT - 43,
    R24G8_TYPELESS - 44,
    D24_UNORM_S8_UINT - 45,
    R24_UNORM_X8_TYPELESS - 46,
    X24_TYPELESS_G8_UINT - 47,
    R8G8_TYPELESS - 48,
    R8G8_UNORM - 49,
    R8G8_UINT - 50,
    R8G8_SNORM - 51,
    R8G8_SINT - 52,
    R16_TYPELESS - 53,
    R16_FLOAT - 54,
    D16_UNORM - 55,
    R16_UNORM - 56,
    R16_UINT - 57,
    R16_SNORM - 58,
    R16_SINT - 59,
    R8_TYPELESS - 60,
    R8_UNORM - 61,
    R8_UINT - 62,
    R8_SNORM - 63,
    R8_SINT - 64,
    A8_UNORM - 65,
    R1_UNORM - 66,
    R9G9B9E5_SHAREDEXP - 67,
    R8G8_B8G8_UNORM - 68,
    G8R8_G8B8_UNORM - 69,
    BC1_TYPELESS - 70,
    BC1_UNORM - 71,
    BC1_UNORM_SRGB - 72,
    BC2_TYPELESS - 73,
    BC2_UNORM - 74,
    BC2_UNORM_SRGB - 75,
    BC3_TYPELESS - 76,
    BC3_UNORM - 77,
    BC3_UNORM_SRGB - 78,
    BC4_TYPELESS - 79,
    BC4_UNORM - 80,
    BC4_SNORM - 81,
    BC5_TYPELESS - 82,
    BC5_UNORM - 83,
    BC5_SNORM - 84,
    B5G6R5_UNORM - 85,
    B5G5R5A1_UNORM - 86,
    B8G8R8A8_UNORM - 87,
    B8G8R8X8_UNORM - 88,
    R10G10B10_XR_BIAS_A2_UNORM - 89,
    B8G8R8A8_TYPELESS - 90,
    B8G8R8A8_UNORM_SRGB - 91,
    B8G8R8X8_TYPELESS - 92,
    B8G8R8X8_UNORM_SRGB - 93,
    BC6H_TYPELESS - 94,
    BC6H_UF16 - 95,
    BC6H_SF16 - 96,
    BC7_TYPELESS - 97,
    BC7_UNORM - 98,
    BC7_UNORM_SRGB - 99,
    AYUV - 100,
    Y410 - 101,
    Y416 - 102,
    NV12 - 103,
    P010 - 104,
    P016 - 105,
    OPAQUE_420 - 106,
    YUY2 - 107,
    Y210 - 108,
    Y216 - 109,
    NV11 - 110,
    AI44 - 111,
    IA44 - 112,
    P8 - 113,
    A8P8 - 114,
    B4G4R4A4_UNORM - 115,
    P208 - 130,
    V208 - 131,
    V408 - 132
);

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
            0..=115 | 130..=132 => Ok(DxgiFormat(value as u8)),
            _ => Err(value),
        }
    }
}
impl From<DxgiFormat> for u32 {
    fn from(value: DxgiFormat) -> Self {
        value.0 as u32
    }
}
