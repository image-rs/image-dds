use std::num::NonZeroU8;

use bitflags::bitflags;

use crate::header::{DdsCaps2, Header, ResourceDimension};
use crate::{util::get_mipmap_size, DecodeError, PixelInfo, Size};

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

    pub fn get_depth_slice(&self, depth: u32) -> Option<SurfaceDescriptor> {
        if depth < self.depth {
            Some(SurfaceDescriptor {
                width: self.width,
                height: self.height,
                offset: self.offset + depth as u64 * self.slice_len,
                len: self.slice_len,
            })
        } else {
            None
        }
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
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
    len: u64,
}
impl Texture {
    /// Creates a new texture at offset 0.
    fn create_at_offset_0(
        width: u32,
        height: u32,
        mipmaps: NonZeroU8,
        pixels: PixelInfo,
    ) -> Result<Self, DecodeError> {
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
        for level in 1..mipmaps.get() {
            // this technically cannot overflow, because mip_len <= main.len,
            // but being conservative is better here
            let mip_len = pixels
                .surface_bytes(main.size().get_mipmap(level))
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

    pub fn pixel_info(&self) -> PixelInfo {
        self.pixels
    }

    /// The level 0 mipmap of this texture.
    pub fn main(&self) -> SurfaceDescriptor {
        self.main
    }
    pub fn mipmaps(&self) -> u8 {
        self.mipmaps.get()
    }
    pub fn get(&self, level: u8) -> Option<SurfaceDescriptor> {
        self.iter_mips().nth(level as usize)
    }
    pub fn iter_mips(&self) -> impl Iterator<Item = SurfaceDescriptor> {
        let mut offset = self.main.data_offset();
        let size_0 = self.main.size();
        let pixels = self.pixels;
        (0..self.mipmaps.get()).map(move |level| {
            let size = size_0.get_mipmap(level);
            // Panic Safety: This cannot overflow, because we already checked in the constructor
            let len = pixels.surface_bytes(size).unwrap();
            let surface = SurfaceDescriptor::new(size.width, size.height, offset, len);
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
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
    len: u64,
}
impl Volume {
    /// Creates a new volume at offset 0.
    fn create_at_offset_0(
        width: u32,
        height: u32,
        depth: u32,
        mipmaps: NonZeroU8,
        pixels: PixelInfo,
    ) -> Result<Self, DecodeError> {
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
        for level in 1..mipmaps.get() {
            let width = get_mipmap_size(main.width(), level);
            let height = get_mipmap_size(main.height(), level);
            let depth = get_mipmap_size(main.height(), level);
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

    pub fn pixel_info(&self) -> PixelInfo {
        self.pixels
    }

    pub fn main(&self) -> VolumeDescriptor {
        self.main
    }
    pub fn mipmaps(&self) -> u8 {
        self.mipmaps.get()
    }
    pub fn get(&self, level: u8) -> Option<VolumeDescriptor> {
        self.iter_mips().nth(level as usize)
    }
    pub fn iter_mips(&self) -> impl Iterator<Item = VolumeDescriptor> {
        let mut offset = self.main.data_offset();
        let width_0 = self.main.width();
        let height_0 = self.main.height();
        let depth_0 = self.main.depth();
        let pixels = self.pixels;
        (0..self.mipmaps.get()).map(move |level| {
            let width = get_mipmap_size(width_0, level);
            let height = get_mipmap_size(height_0, level);
            let depth = get_mipmap_size(depth_0, level);
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
    PartialCubeMap(CubeMapFaces),
}
bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct CubeMapFaces: u8 {
        const POSITIVE_X = 0b0000_0001;
        const NEGATIVE_X = 0b0000_0010;
        const POSITIVE_Y = 0b0000_0100;
        const NEGATIVE_Y = 0b0000_1000;
        const POSITIVE_Z = 0b0001_0000;
        const NEGATIVE_Z = 0b0010_0000;
        const ALL = Self::POSITIVE_X.bits() | Self::NEGATIVE_X.bits() | Self::POSITIVE_Y.bits() | Self::NEGATIVE_Y.bits() | Self::POSITIVE_Z.bits() | Self::NEGATIVE_Z.bits();
    }
}
impl From<DdsCaps2> for CubeMapFaces {
    fn from(value: DdsCaps2) -> Self {
        let faces = (value & DdsCaps2::CUBE_MAP_ALL_FACES).bits();
        CubeMapFaces::from_bits_truncate((faces >> 10) as u8)
    }
}
impl CubeMapFaces {
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
    pub(crate) first: Texture,
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

    pub fn pixel_info(&self) -> PixelInfo {
        self.first.pixel_info()
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
    pub fn iter(&self) -> impl Iterator<Item = Texture> {
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
        match header {
            Header::Dx10(dx10) => {
                match dx10.resource_dimension {
                    ResourceDimension::Texture1D => {
                        let mut info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                        info.height = 1;
                        let array_size = dx10.array_size;

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
                        let array_size = dx10.array_size;

                        if dx10.is_cube_map() {
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
            }
            Header::Dx9(dx9) => {
                if dx9.is_volume() {
                    let info = VolumeLayoutInfo::from_header(header, pixel_info)?;
                    Ok(Self::Volume(info.create()?))
                } else if let Some(faces) = dx9.cube_map_faces() {
                    let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                    let face_count = faces.count();

                    let kind = if face_count == 6 {
                        TextureArrayKind::CubeMaps
                    } else {
                        TextureArrayKind::PartialCubeMap(faces)
                    };
                    Ok(Self::TextureArray(info.create_array(kind, face_count)?))
                } else {
                    let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                    Ok(Self::Texture(info.create()?))
                }
            }
        }
    }

    pub fn texture(&self) -> Option<&Texture> {
        match self {
            DataLayout::Texture(texture) => Some(texture),
            _ => None,
        }
    }
    pub fn volume(&self) -> Option<&Volume> {
        match self {
            DataLayout::Volume(volume) => Some(volume),
            _ => None,
        }
    }
    pub fn texture_array(&self) -> Option<&TextureArray> {
        match self {
            DataLayout::TextureArray(array) => Some(array),
            _ => None,
        }
    }

    pub fn pixel_info(&self) -> PixelInfo {
        match self {
            DataLayout::Texture(texture) => texture.pixel_info(),
            DataLayout::Volume(volume) => volume.pixel_info(),
            DataLayout::TextureArray(array) => array.pixel_info(),
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SurfaceLayoutInfo {
    width: u32,
    height: u32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
}
impl SurfaceLayoutInfo {
    fn from_header(header: &Header, pixels: PixelInfo) -> Result<Self, DecodeError> {
        let mipmaps = header.mipmap_count().get();
        if mipmaps > 32 {
            return Err(DecodeError::TooManyMipMaps(mipmaps));
        }
        let mipmaps = NonZeroU8::new(mipmaps as u8).unwrap();

        Ok(Self {
            width: header.width(),
            height: header.height(),
            mipmaps,
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
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
}
impl VolumeLayoutInfo {
    fn from_header(header: &Header, pixels: PixelInfo) -> Result<Self, DecodeError> {
        let mipmaps = header.mipmap_count().get();
        if mipmaps > 255 {
            return Err(DecodeError::TooManyMipMaps(mipmaps));
        }
        let mipmaps = NonZeroU8::new(mipmaps as u8).unwrap();

        Ok(Self {
            width: header.width(),
            height: header.height(),
            depth: header.depth().ok_or(DecodeError::MissingDepth)?,
            mipmaps,
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
