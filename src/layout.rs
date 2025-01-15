use bitflags::bitflags;

use crate::{DdsCaps2, DecodeError, Header, MiscFlags, PixelInfo, ResourceDimension, Size};

pub trait DataRegion {
    /// The number of bytes this object occupies in the data section of a DDS file.
    ///
    /// It is guaranteed that `self.offset() + self.len() <= u64::MAX`.
    fn data_len(&self) -> u64;
    /// The byte offset of this object in the data section of a DDS file.
    fn data_offset(&self) -> u64;
    /// The byte offset of the byte after this object in the data section of a DDS file.
    ///
    /// This is equivalent to `self.offset() + self.len()`.
    fn data_end(&self) -> u64 {
        self.data_offset() + self.data_len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceDescriptor {
    width: u32,
    height: u32,
    offset: u64,
    len: u64,
}
impl SurfaceDescriptor {
    /// Internal constructor.
    ///
    /// This **assumes** that the arguments are valid and only performs checks
    /// in debug.
    fn new(width: u32, height: u32, offset: u64, len: u64) -> Self {
        debug_assert!(width > 0);
        debug_assert!(height > 0);
        debug_assert!(len > 0);
        debug_assert!(offset.checked_add(len).is_some());

        Self {
            width,
            height,
            offset,
            len,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }
}
impl DataRegion for SurfaceDescriptor {
    fn data_len(&self) -> u64 {
        self.len
    }
    fn data_offset(&self) -> u64 {
        self.offset
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VolumeDescriptor {
    width: u32,
    height: u32,
    depth: u32,
    offset: u64,
    slice_len: u64,
}
impl VolumeDescriptor {
    fn new(width: u32, height: u32, depth: u32, offset: u64, slice_len: u64) -> Self {
        debug_assert!(width > 0);
        debug_assert!(height > 0);
        debug_assert!(depth > 0);
        debug_assert!(slice_len > 0);
        // check that `offset + len` does not overflow
        debug_assert!(slice_len
            .checked_mul(depth as u64)
            .and_then(|len| offset.checked_add(len))
            .is_some());

        Self {
            width,
            height,
            depth,
            offset,
            slice_len,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn depth(&self) -> u32 {
        self.depth
    }
    pub fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    /// Iterates over all depth slices of the volume.
    ///
    /// To get the depth value of a slice, use `.enumerate()`. Example:
    ///
    /// ```no_run
    /// # use ddsd::{VolumeDescriptor, DataRegion};
    /// # fn get_volume() -> VolumeDescriptor { todo!() }
    /// let volume: VolumeDescriptor = get_volume();
    /// for (depth, slice) in volume.iter_depth_slices().enumerate() {
    ///     println!("Slice {} starts at {}", depth, slice.data_offset());
    /// }
    /// ```
    pub fn iter_depth_slices(&self) -> impl Iterator<Item = SurfaceDescriptor> {
        let Self {
            width,
            height,
            offset,
            slice_len,
            ..
        } = *self;

        (0..self.depth).map(move |depth| SurfaceDescriptor {
            width,
            height,
            offset: offset + depth as u64 * slice_len,
            len: slice_len,
        })
    }
}
impl DataRegion for VolumeDescriptor {
    fn data_len(&self) -> u64 {
        // Cannot overflow. See `VolumeDescriptor::new`.
        self.slice_len * self.depth as u64
    }
    fn data_offset(&self) -> u64 {
        self.offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Texture {
    main: SurfaceDescriptor,
    mipmaps: u8,
    pixels: PixelInfo,
    len: u64,
}
impl Texture {
    /// Creates a new texture at offset 0.
    fn create_at_offset_0(
        width: u32,
        height: u32,
        mipmaps: u8,
        pixels: PixelInfo,
    ) -> Result<Self, DecodeError> {
        // at least one mipmap
        debug_assert!(mipmaps > 0);

        // zero dimensions
        if width == 0 || height == 0 {
            return Err(DecodeError::ZeroDimension);
        }

        // create main surface
        let main = SurfaceDescriptor::new(
            width,
            height,
            0,
            pixels
                .surface_bytes(Size::new(width, height))
                .ok_or(DecodeError::DataLayoutTooBig)?,
        );

        // compute len
        let mut len = main.data_len();
        for level in 1..mipmaps {
            let width = get_mip_size(main.width(), level);
            let height = get_mip_size(main.height(), level);
            // this technically cannot overflow, because mip_len <= main.len,
            // but being conservative is better here
            let mip_len = pixels
                .surface_bytes(Size::new(width, height))
                .ok_or(DecodeError::DataLayoutTooBig)?;
            // this might overflow, so we check it
            len = len
                .checked_add(mip_len)
                .ok_or(DecodeError::DataLayoutTooBig)?;
        }

        Ok(Self {
            main,
            mipmaps,
            pixels,
            len,
        })
    }

    /// The level 0 mipmap of this texture.
    pub fn main(&self) -> SurfaceDescriptor {
        self.main
    }
    pub fn mipmaps(&self) -> u8 {
        self.mipmaps
    }
    pub fn get(&self, index: u8) -> Option<SurfaceDescriptor> {
        self.iter_surfaces().nth(index as usize)
    }
    pub fn iter_surfaces(&self) -> impl Iterator<Item = SurfaceDescriptor> {
        let mut offset = self.main.data_offset();
        let width_0 = self.main.width();
        let height_0 = self.main.height();
        let pixels = self.pixels;
        (0..self.mipmaps).map(move |level| {
            let width = get_mip_size(width_0, level);
            let height = get_mip_size(height_0, level);
            // Panic Safety: This cannot overflow, because we already checked in the constructor
            let len = pixels.surface_bytes(Size::new(width, height)).unwrap();
            let surface = SurfaceDescriptor::new(width, height, offset, len);
            offset += len;
            surface
        })
    }

    /// Internal method. This **assumes** that `offset + len` does not overflow.
    fn set_offset(&mut self, offset: u64) {
        self.main.offset = offset;
    }
}
impl DataRegion for Texture {
    fn data_len(&self) -> u64 {
        self.len
    }
    fn data_offset(&self) -> u64 {
        self.main.data_offset()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Volume {
    main: VolumeDescriptor,
    mipmaps: u8,
    pixels: PixelInfo,
    len: u64,
}
impl Volume {
    /// Creates a new volume at offset 0.
    fn create_at_offset_0(
        width: u32,
        height: u32,
        depth: u32,
        mipmaps: u8,
        pixels: PixelInfo,
    ) -> Result<Self, DecodeError> {
        // at least one mipmap
        debug_assert!(mipmaps > 0);

        // zero dimensions
        if width == 0 || height == 0 || depth == 0 {
            return Err(DecodeError::ZeroDimension);
        }

        // create main volume
        let main_slice_len = pixels
            .surface_bytes(Size::new(width, height))
            .ok_or(DecodeError::DataLayoutTooBig)?;
        if main_slice_len.checked_mul(depth as u64).is_none() {
            return Err(DecodeError::DataLayoutTooBig);
        }
        let main = VolumeDescriptor::new(width, height, depth, 0, main_slice_len);

        // compute len
        let mut len = main.data_len();
        for level in 1..mipmaps {
            let width = get_mip_size(main.width(), level);
            let height = get_mip_size(main.height(), level);
            let depth = get_mip_size(main.height(), level);
            // this technically cannot overflow, because mip_len <= main.len,
            // but being conservative is better here
            let mip_slice_len = pixels
                .surface_bytes(Size::new(width, height))
                .ok_or(DecodeError::DataLayoutTooBig)?;
            let mip_len = mip_slice_len
                .checked_mul(depth as u64)
                .ok_or(DecodeError::DataLayoutTooBig)?;
            // this might overflow, so we check it
            len = len
                .checked_add(mip_len)
                .ok_or(DecodeError::DataLayoutTooBig)?;
        }

        Ok(Self {
            main,
            mipmaps,
            pixels,
            len,
        })
    }

    pub fn main(&self) -> VolumeDescriptor {
        self.main
    }
    pub fn mipmaps(&self) -> u8 {
        self.mipmaps
    }
    pub fn get(&self, index: u8) -> Option<VolumeDescriptor> {
        self.iter_volumes().nth(index as usize)
    }
    pub fn iter_volumes(&self) -> impl Iterator<Item = VolumeDescriptor> {
        let mut offset = self.main.data_offset();
        let width_0 = self.main.width();
        let height_0 = self.main.height();
        let depth_0 = self.main.depth();
        let pixels = self.pixels;
        (0..self.mipmaps).map(move |level| {
            let width = get_mip_size(width_0, level);
            let height = get_mip_size(height_0, level);
            let depth = get_mip_size(depth_0, level);
            // Panic Safety: This cannot overflow, because we already checked in the constructor
            let slice_len = pixels.surface_bytes(Size::new(width, height)).unwrap();
            let volume = VolumeDescriptor::new(width, height, depth, offset, slice_len);
            offset += depth as u64 * slice_len;
            volume
        })
    }
}
impl DataRegion for Volume {
    fn data_len(&self) -> u64 {
        self.len
    }
    fn data_offset(&self) -> u64 {
        // if there is a volume, it is the only object in the data section
        0
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
        let faces = (value & DdsCaps2::CUBE_MAP_ALL_FACES).bits();
        CubeMapSide::from_bits_truncate((faces >> 10) as u8)
    }
}
impl CubeMapSide {
    /// Returns the number of cube map sides set in this bit mask.
    pub fn count(&self) -> u32 {
        self.bits().count_ones()
    }
}

/// An array of textures or (partial) cube maps.
///
/// The array is guaranteed to have at least one texture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureArray {
    kind: TextureArrayKind,
    array_len: u32,
    first: Texture,
}
impl TextureArray {
    fn new(kind: TextureArrayKind, array_len: u32, first: Texture) -> Result<Self, DecodeError> {
        // must start at offset 0
        debug_assert_eq!(first.data_offset(), 0);

        let total_len = first.data_len().checked_mul(array_len as u64);
        if total_len.is_none() {
            return Err(DecodeError::DataLayoutTooBig);
        }

        Ok(Self {
            kind,
            array_len,
            first,
        })
    }

    pub fn kind(&self) -> TextureArrayKind {
        self.kind
    }

    pub fn is_empty(&self) -> bool {
        self.array_len == 0
    }
    pub fn len(&self) -> usize {
        self.array_len as usize
    }
    pub fn first(&self) -> Texture {
        self.first.clone()
    }
    pub fn get(&self, index: usize) -> Option<Texture> {
        if index < self.array_len as usize {
            let mut texture = self.first.clone();
            // Panic Safety: this can't overflow, because we checked in the constructor
            texture.set_offset(index as u64 * texture.data_len());
            Some(texture)
        } else {
            None
        }
    }
    pub fn iter_textures(&self) -> impl Iterator<Item = Texture> {
        let mut texture = self.first.clone();
        let texture_len = texture.data_len();
        (0..self.array_len).map(move |i| {
            // Panic Safety: this can't overflow, because we checked in the constructor
            texture.set_offset(i as u64 * texture_len);
            texture.clone()
        })
    }
}
impl DataRegion for TextureArray {
    fn data_len(&self) -> u64 {
        // Panic Safety: this can't overflow, because we checked in the constructor
        self.first.data_len() * self.array_len as u64
    }
    fn data_offset(&self) -> u64 {
        0
    }
}

/// The type and layout of the surfaces/volumes in the data section of a DDS file.
#[derive(Debug, Clone)]
pub enum DataLayout {
    Texture(Texture),
    Volume(Volume),
    TextureArray(TextureArray),
}
impl DataLayout {
    pub fn from_header(header: &Header) -> Result<Self, DecodeError> {
        Self::from_header_with(header, PixelInfo::from_header(header)?)
    }
    pub fn from_header_with(header: &Header, pixel_info: PixelInfo) -> Result<Self, DecodeError> {
        if let Some(ref header_dxt10) = header.dxt10 {
            // DirectX 10+

            match header_dxt10.resource_dimension {
                ResourceDimension::Texture1D => {
                    let mut info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                    info.height = 1;
                    let array_size = header_dxt10.array_size;

                    if array_size == 1 {
                        Ok(Self::Texture(info.create()?))
                    } else {
                        Ok(Self::TextureArray(
                            info.create_array(TextureArrayKind::Textures, array_size)?,
                        ))
                    }
                }
                ResourceDimension::Texture2D => {
                    let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                    let array_size = header_dxt10.array_size;
                    let is_cube_map = header_dxt10.misc_flag.contains(MiscFlags::TEXTURE_CUBE);

                    if is_cube_map {
                        // "For a 2D texture that is also a cube-map texture, array_size represents the number of cubes."
                        let cube_map_faces = array_size
                            .checked_mul(6)
                            .ok_or(DecodeError::ArraySizeTooBig(array_size))?;
                        return Ok(Self::TextureArray(
                            info.create_array(TextureArrayKind::CubeMaps, cube_map_faces)?,
                        ));
                    }

                    if array_size == 1 {
                        Ok(Self::Texture(info.create()?))
                    } else {
                        Ok(Self::TextureArray(
                            info.create_array(TextureArrayKind::Textures, array_size)?,
                        ))
                    }
                }
                ResourceDimension::Texture3D => {
                    let info = VolumeLayoutInfo::from_header(header, pixel_info)?;
                    Ok(Self::Volume(info.create()?))
                }
            }
        } else {
            // DirectX <=9

            if header.caps2.contains(DdsCaps2::VOLUME) {
                let info = VolumeLayoutInfo::from_header(header, pixel_info)?;
                Ok(Self::Volume(info.create()?))
            } else if header.caps2.contains(DdsCaps2::CUBE_MAP) {
                let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                let sides = CubeMapSide::from(header.caps2);
                let side_count = sides.count();

                let kind = if side_count == 6 {
                    TextureArrayKind::CubeMaps
                } else {
                    TextureArrayKind::PartialCubeMap(sides)
                };
                Ok(Self::TextureArray(info.create_array(kind, side_count)?))
            } else {
                let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                Ok(Self::Texture(info.create()?))
            }
        }
    }
}
impl DataRegion for DataLayout {
    fn data_len(&self) -> u64 {
        match self {
            DataLayout::Texture(texture) => texture.data_len(),
            DataLayout::Volume(volume) => volume.data_len(),
            DataLayout::TextureArray(textures) => textures.data_len(),
        }
    }
    fn data_offset(&self) -> u64 {
        // the data layout describes the entire data section, so it has to start at 0
        0
    }
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
    mipmaps: u8,
    pixels: PixelInfo,
}
impl SurfaceLayoutInfo {
    fn from_header(header: &Header, pixels: PixelInfo) -> Result<Self, DecodeError> {
        let mipmaps = header.mipmap_count.unwrap_or(1).max(1);
        if mipmaps > 32 {
            return Err(DecodeError::TooManyMipMaps(mipmaps));
        }

        Ok(Self {
            width: header.width,
            height: header.height,
            mipmaps: mipmaps as u8,
            pixels,
        })
    }

    fn create(&self) -> Result<Texture, DecodeError> {
        Texture::create_at_offset_0(self.width, self.height, self.mipmaps, self.pixels)
    }

    fn create_array(
        &self,
        kind: TextureArrayKind,
        array_len: u32,
    ) -> Result<TextureArray, DecodeError> {
        TextureArray::new(kind, array_len, self.create()?)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VolumeLayoutInfo {
    width: u32,
    height: u32,
    depth: u32,
    mipmaps: u8,
    pixels: PixelInfo,
}
impl VolumeLayoutInfo {
    fn from_header(header: &Header, pixels: PixelInfo) -> Result<Self, DecodeError> {
        let mipmaps = header.mipmap_count.unwrap_or(1).max(1);
        if mipmaps > 32 {
            return Err(DecodeError::TooManyMipMaps(mipmaps));
        }

        Ok(Self {
            width: header.width,
            height: header.height,
            depth: header.depth.ok_or(DecodeError::MissingDepth)?,
            mipmaps: mipmaps as u8,
            pixels,
        })
    }

    fn create(&self) -> Result<Volume, DecodeError> {
        Volume::create_at_offset_0(
            self.width,
            self.height,
            self.depth,
            self.mipmaps,
            self.pixels,
        )
    }
}
