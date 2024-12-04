use bitflags::bitflags;

use crate::{
    DdsCaps2, DdsDecodeError, DdsHeaderDxt10, DxgiFormat, FourCC, FullHeader, ResourceDimension,
};

pub trait ByteLength {
    /// Returns the number of bytes this object occupies in the data section of a DDS file.
    fn byte_len(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SurfaceDescriptor {
    pub width: u32,
    pub height: u32,
    pub offset: usize,
    pub len: usize,
}
impl ByteLength for SurfaceDescriptor {
    fn byte_len(&self) -> usize {
        self.len
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VolumeDescriptor {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub offset: usize,
    pub slice_len: usize,
}
impl VolumeDescriptor {
    /// Iterates over all depth slices of the volume.
    pub fn iter_slices(&self) -> impl Iterator<Item = SurfaceDescriptor> + '_ {
        (0..self.depth).map(move |depth| SurfaceDescriptor {
            width: self.width,
            height: self.height,
            offset: self.offset + depth as usize * self.slice_len,
            len: self.slice_len,
        })
    }
}
impl ByteLength for VolumeDescriptor {
    fn byte_len(&self) -> usize {
        self.depth as usize * self.slice_len
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Texture {
    pub main: SurfaceDescriptor,
    pub mips: Vec<SurfaceDescriptor>,
}
impl Texture {
    pub fn iter_surfaces(&self) -> impl Iterator<Item = SurfaceDescriptor> + '_ {
        std::iter::once(&self.main).chain(self.mips.iter()).cloned()
    }
}
impl ByteLength for Texture {
    fn byte_len(&self) -> usize {
        self.main.byte_len() + self.mips.iter().map(|mip| mip.byte_len()).sum::<usize>()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Volume {
    pub main: VolumeDescriptor,
    pub mips: Vec<VolumeDescriptor>,
}
impl Volume {
    pub fn iter_volumes(&self) -> impl Iterator<Item = VolumeDescriptor> + '_ {
        std::iter::once(&self.main).chain(self.mips.iter()).cloned()
    }
}
impl ByteLength for Volume {
    fn byte_len(&self) -> usize {
        self.main.byte_len() + self.mips.iter().map(|mip| mip.byte_len()).sum::<usize>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureArrayKind {
    /// An array of textures.
    Textures,
    /// An array of cube maps.
    ///
    /// The number of textures in the array is a multiple of 6.
    CubeMaps,
    /// An array of at most 5 sides of a cube maps.
    PartialCubeMap(CubeMapSide),
}
bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct CubeMapSide: u8 {
        const POSITIVE_X = 0b0000_0001;
        const NEGATIVE_X = 0b0000_0010;
        const POSITIVE_Y = 0b0000_0100;
        const NEGATIVE_Y = 0b0000_1000;
        const POSITIVE_Z = 0b0001_0000;
        const NEGATIVE_Z = 0b0010_0000;
        const ALL = Self::POSITIVE_X.bits() | Self::NEGATIVE_X.bits() | Self::POSITIVE_Y.bits() | Self::NEGATIVE_Y.bits() | Self::POSITIVE_Z.bits() | Self::NEGATIVE_Z.bits();
    }
}
impl From<DdsCaps2> for CubeMapSide {
    fn from(value: DdsCaps2) -> Self {
        let faces = (value & DdsCaps2::CUBEMAP_ALL_FACES).bits();
        CubeMapSide::from_bits_truncate((faces >> 10) as u8)
    }
}
impl CubeMapSide {
    /// Returns the number of cube map sides set in this bit mask.
    pub fn count(&self) -> u32 {
        self.contains(CubeMapSide::POSITIVE_X) as u32
            + self.contains(CubeMapSide::NEGATIVE_X) as u32
            + self.contains(CubeMapSide::POSITIVE_Y) as u32
            + self.contains(CubeMapSide::NEGATIVE_Y) as u32
            + self.contains(CubeMapSide::POSITIVE_Z) as u32
            + self.contains(CubeMapSide::NEGATIVE_Z) as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureArray {
    pub textures: Vec<Texture>,
    pub kind: TextureArrayKind,
}
impl TextureArray {
    pub fn iter_textures(&self) -> impl Iterator<Item = &Texture> {
        self.textures.iter()
    }
}
impl ByteLength for TextureArray {
    fn byte_len(&self) -> usize {
        self.textures.iter().map(|texture| texture.byte_len()).sum()
    }
}

trait Chainable: ByteLength {
    fn offset(&self) -> usize;
    fn next_offset(&self) -> usize {
        self.offset() + self.byte_len()
    }
}
impl Chainable for SurfaceDescriptor {
    fn offset(&self) -> usize {
        self.offset
    }
}
impl Chainable for VolumeDescriptor {
    fn offset(&self) -> usize {
        self.offset
    }
}
impl Chainable for Texture {
    fn offset(&self) -> usize {
        self.main.offset
    }
}
fn create_chain<T: Chainable>(
    range: std::ops::Range<u32>,
    offset: usize,
    f: impl Fn(usize, u32) -> T,
) -> impl Iterator<Item = T> {
    let mut offset = offset;
    range.map(move |i| {
        let result = f(offset, i);
        offset = result.next_offset();
        result
    })
}
fn create_chain_u8<T: Chainable>(
    range: std::ops::Range<u8>,
    offset: usize,
    f: impl Fn(usize, u8) -> T,
) -> impl Iterator<Item = T> {
    let mut offset = offset;
    range.map(move |i| {
        let result = f(offset, i);
        offset = result.next_offset();
        result
    })
}

fn get_mip_size(main_size: u32, level: u8) -> u32 {
    let size = main_size >> level;
    if size == 0 {
        1
    } else {
        size
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SurfaceLayoutInfo {
    width: u32,
    height: u32,
    mip_count: u8,
    pitch: PitchInfo,
}
impl SurfaceLayoutInfo {
    fn from_header(header: &FullHeader) -> Result<Self, DdsDecodeError> {
        let pitch = PitchInfo::from_header(header)?;

        // width and height are always required, so we can read them without checking `flags`.
        let width = header.header.width;
        let height = header.header.height;

        let mip_count = u32::max(1, header.header.get_mipmap_count().unwrap_or(1));
        if mip_count > 32 {
            return Err(DdsDecodeError::TooManyMipMaps(mip_count));
        }

        Ok(Self {
            width,
            height,
            mip_count: mip_count as u8,
            pitch,
        })
    }

    fn create_desc(&self, offset: usize, level: u8) -> SurfaceDescriptor {
        assert!(level < self.mip_count);

        let width = get_mip_size(self.width, level);
        let height = get_mip_size(self.height, level);

        SurfaceDescriptor {
            width,
            height,
            offset,
            len: self.pitch.get_byte_length(width, height),
        }
    }

    fn create(&self, offset: usize) -> Texture {
        let main = self.create_desc(offset, 0);
        let mips = create_chain_u8(1..self.mip_count, main.next_offset(), |offset, level| {
            self.create_desc(offset, level)
        })
        .collect();

        Texture { main, mips }
    }

    fn create_many(&self, count: u32) -> Vec<Texture> {
        create_chain(0..count, 0, |offset, _| self.create(offset)).collect()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VolumeLayoutInfo {
    width: u32,
    height: u32,
    depth: u32,
    mip_count: u8,
    pitch: PitchInfo,
}
impl VolumeLayoutInfo {
    fn from_header(header: &FullHeader) -> Result<Self, DdsDecodeError> {
        let SurfaceLayoutInfo {
            width,
            height,
            mip_count,
            pitch,
        } = SurfaceLayoutInfo::from_header(header)?;

        let depth = header
            .header
            .get_depth()
            .ok_or_else(|| DdsDecodeError::InvalidHeader("3D texture without depth"))?;

        Ok(Self {
            width,
            height,
            depth,
            mip_count,
            pitch,
        })
    }

    fn create_desc(&self, offset: usize, level: u8) -> VolumeDescriptor {
        assert!(level < self.mip_count);

        let width = get_mip_size(self.width, level);
        let height = get_mip_size(self.height, level);
        let depth = get_mip_size(self.depth, level);

        VolumeDescriptor {
            width,
            height,
            depth,
            offset,
            slice_len: self.pitch.get_byte_length(width, height),
        }
    }

    fn create(&self, offset: usize) -> Volume {
        let main = self.create_desc(offset, 0);
        let mips = create_chain_u8(1..self.mip_count, main.next_offset(), |offset, level| {
            self.create_desc(offset, level)
        })
        .collect();

        Volume { main, mips }
    }
}

/// This describes the layout of surfaces and volumes in the data section of a DDS file.
pub enum DataLayout {
    Texture(Texture),
    Volume(Volume),
    TextureArray(TextureArray),
}
impl DataLayout {
    pub fn from_header(header: &FullHeader) -> Result<Self, DdsDecodeError> {
        if let Some(ref header_dxt10) = header.header_dxt10 {
            // DirectX 10+

            match header_dxt10.resource_dimension {
                ResourceDimension::Unknown | ResourceDimension::Buffer => {
                    Err(DdsDecodeError::InvalidHeader("Invalid resource dimension"))
                }
                ResourceDimension::Texture1D => {
                    let mut info = SurfaceLayoutInfo::from_header(header)?;
                    info.height = 1;
                    let array_size = header_dxt10.array_size;

                    if array_size == 1 {
                        Ok(Self::Texture(info.create(0)))
                    } else {
                        Ok(Self::TextureArray(TextureArray {
                            textures: info.create_many(array_size),
                            kind: TextureArrayKind::Textures,
                        }))
                    }
                }
                ResourceDimension::Texture2D => {
                    let info = SurfaceLayoutInfo::from_header(header)?;
                    let array_size = header_dxt10.array_size;
                    let is_cube_map = header_dxt10.misc_flag & 0x4 != 0;

                    if is_cube_map {
                        // "For a 2D texture that is also a cube-map texture, array_size represents the number of cubes."
                        return Ok(Self::TextureArray(TextureArray {
                            textures: info.create_many(array_size * 6),
                            kind: TextureArrayKind::CubeMaps,
                        }));
                    }

                    if array_size == 1 {
                        Ok(Self::Texture(info.create(0)))
                    } else {
                        Ok(Self::TextureArray(TextureArray {
                            textures: info.create_many(array_size),
                            kind: TextureArrayKind::Textures,
                        }))
                    }
                }
                ResourceDimension::Texture3D => {
                    let info = VolumeLayoutInfo::from_header(header)?;
                    Ok(Self::Volume(info.create(0)))
                }
            }
        } else {
            // DirectX <=9

            if header.header.caps2.contains(DdsCaps2::VOLUME) {
                let info = VolumeLayoutInfo::from_header(header)?;
                Ok(Self::Volume(info.create(0)))
            } else if header.header.caps2.contains(DdsCaps2::CUBEMAP) {
                let info = SurfaceLayoutInfo::from_header(header)?;
                let sides = CubeMapSide::from(header.header.caps2);
                let side_count = sides.count();

                Ok(Self::TextureArray(TextureArray {
                    textures: info.create_many(side_count),
                    kind: if side_count == 6 {
                        TextureArrayKind::CubeMaps
                    } else {
                        TextureArrayKind::PartialCubeMap(sides)
                    },
                }))
            } else {
                let info = SurfaceLayoutInfo::from_header(header)?;
                Ok(Self::Texture(info.create(0)))
            }
        }
    }
}
impl ByteLength for DataLayout {
    fn byte_len(&self) -> usize {
        match self {
            DataLayout::Texture(texture) => texture.byte_len(),
            DataLayout::Volume(volume) => volume.byte_len(),
            DataLayout::TextureArray(textures) => textures.byte_len(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PitchInfo {
    BlockCompressed {
        block_size: u8,
    },
    Other {
        bits_per_pixel: u8,
    },
    SubSampled {
        sub_sampling: SubSampling,
        bits_per_pixel: u8,
    },
}
impl PitchInfo {
    pub fn from_header(header: &FullHeader) -> Result<Self, DdsDecodeError> {
        if let Some(DdsHeaderDxt10 { dxgi_format, .. }) = header.header_dxt10 {
            return match Self::from_dxgi(dxgi_format) {
                Some(pitcher) => Ok(pitcher),
                None => Err(DdsDecodeError::UnsupportedDxgiFormat(dxgi_format)),
            };
        }

        let four_cc = header.header.pixel_format.four_cc;

        // try to convert FourCC to DxgiFormat to get a pitcher
        if let Some(pitcher) = four_cc.try_into().ok().and_then(Self::from_dxgi) {
            return Ok(pitcher);
        }

        // at this point, we only have to deal with a few legacy formats
        match four_cc {
            FourCC::UYVY | FourCC::YUY2 | FourCC::RGBG | FourCC::GRGB => {
                Ok(PitchInfo::SubSampled {
                    sub_sampling: SubSampling::X2,
                    bits_per_pixel: 16,
                })
            }
            _ => Err(DdsDecodeError::UnsupportedFourCC(four_cc)),
        }
    }
    pub fn from_dxgi(format: DxgiFormat) -> Option<PitchInfo> {
        if let Some(block_size) = format.block_size() {
            Some(PitchInfo::BlockCompressed { block_size })
        } else if let Some(bits_per_pixel) = format.bits_per_pixel() {
            let bits_per_pixel = bits_per_pixel.into();
            if let Some(sub_sampling) = format.into() {
                Some(PitchInfo::SubSampled {
                    sub_sampling,
                    bits_per_pixel,
                })
            } else {
                Some(PitchInfo::Other { bits_per_pixel })
            }
        } else {
            None
        }
    }

    pub fn scan_line_length(&self, width: u32) -> usize {
        let width = width as u64;
        let pitch = match self {
            PitchInfo::BlockCompressed { block_size } => {
                u64::max(1, (width + 3) / 4) * (*block_size as u64)
            }
            PitchInfo::Other { bits_per_pixel } => (width * (*bits_per_pixel as u64) + 7) / 8,
            PitchInfo::SubSampled {
                sub_sampling,
                bits_per_pixel,
            } => {
                let bits_per_pixel = *bits_per_pixel as u64;
                match sub_sampling {
                    SubSampling::X2 => {
                        assert_eq!(bits_per_pixel % 4, 0);
                        ((width + 1) / 2) * (bits_per_pixel / 4)
                    }
                    SubSampling::X2Y2 => {
                        assert_eq!(bits_per_pixel % 2, 0);
                        ((width + 1) / 2) * (bits_per_pixel / 2)
                    }
                    SubSampling::X4 => {
                        assert_eq!(bits_per_pixel % 2, 0);
                        ((width + 3) / 4) * (bits_per_pixel / 2)
                    }
                }
            }
        };
        pitch as usize
    }
    pub fn number_of_scan_lines(&self, height: u32) -> usize {
        match self {
            PitchInfo::BlockCompressed { .. } => (height as usize + 3) / 4,
            PitchInfo::SubSampled {
                sub_sampling: SubSampling::X2Y2,
                ..
            } => {
                // Because of sub sampling, each scan line covers 2 pixel lines
                (height as usize + 1) / 2
            }
            _ => height as usize,
        }
    }

    pub fn get_byte_length(&self, width: u32, height: u32) -> usize {
        self.scan_line_length(width) * self.number_of_scan_lines(height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SubSampling {
    /// 2:1 downsampling in the horizontal direction (4:2:2).
    X2,
    /// 2:1 downsampling in both directions (4:2:0).
    X2Y2,
    /// 4:1 downsampling in the horizontal direction (4:1:1).
    X4,
}
impl SubSampling {
    pub fn x(&self) -> u32 {
        match self {
            SubSampling::X2 | SubSampling::X2Y2 => 2,
            SubSampling::X4 => 4,
        }
    }
    pub fn y(&self) -> u32 {
        match self {
            SubSampling::X2Y2 => 2,
            _ => 1,
        }
    }
}
impl From<DxgiFormat> for Option<SubSampling> {
    fn from(value: DxgiFormat) -> Self {
        match value {
            DxgiFormat::R8G8_B8G8_UNORM
            | DxgiFormat::G8R8_G8B8_UNORM
            | DxgiFormat::YUY2
            | DxgiFormat::Y210
            | DxgiFormat::Y216 => Some(SubSampling::X2),
            DxgiFormat::NV12 | DxgiFormat::P010 | DxgiFormat::P016 | DxgiFormat::OPAQUE_420 => {
                Some(SubSampling::X2Y2)
            }
            DxgiFormat::NV11 => Some(SubSampling::X4),
            _ => None,
        }
    }
}
impl From<FourCC> for Option<SubSampling> {
    fn from(value: FourCC) -> Self {
        if let Ok(value) = DxgiFormat::try_from(value) {
            return Self::from(value);
        }
        match value {
            FourCC::UYVY => Some(SubSampling::X2),
            _ => None,
        }
    }
}
