use std::num::{NonZeroU32, NonZeroU8};

use bitflags::bitflags;

use crate::header::{Caps2, Header, ResourceDimension};
use crate::DecodeError;
use crate::{
    util::{get_mipmap_size, NON_ZERO_U32_ONE},
    LayoutError, PixelInfo, Size,
};

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
    width: NonZeroU32,
    height: NonZeroU32,
    offset: u64,
    len: u64,
}
impl SurfaceDescriptor {
    /// Internal constructor.
    ///
    /// This **assumes** that the arguments are valid and only performs checks
    /// in debug.
    fn new(width: NonZeroU32, height: NonZeroU32, offset: u64, len: u64) -> Self {
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
        self.width.get()
    }
    pub fn height(&self) -> u32 {
        self.height.get()
    }
    pub fn size(&self) -> Size {
        Size::new(self.width(), self.height())
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
    width: NonZeroU32,
    height: NonZeroU32,
    depth: NonZeroU32,
    offset: u64,
    slice_len: u64,
}
impl VolumeDescriptor {
    fn new(
        width: NonZeroU32,
        height: NonZeroU32,
        depth: NonZeroU32,
        offset: u64,
        slice_len: u64,
    ) -> Self {
        debug_assert!(slice_len > 0);
        // check that `offset + len` does not overflow
        debug_assert!(slice_len
            .checked_mul(depth.get() as u64)
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
        self.width.get()
    }
    pub fn height(&self) -> u32 {
        self.height.get()
    }
    pub fn depth(&self) -> u32 {
        self.depth.get()
    }
    pub fn size(&self) -> Size {
        Size::new(self.width(), self.height())
    }

    pub fn get_depth_slice(&self, depth: u32) -> Option<SurfaceDescriptor> {
        if depth < self.depth() {
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
    /// # use dds::{VolumeDescriptor, DataRegion};
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

        (0..self.depth()).map(move |depth| SurfaceDescriptor {
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
        self.slice_len * self.depth() as u64
    }
    fn data_offset(&self) -> u64 {
        self.offset
    }
}

fn to_short_len(len: u64) -> Option<NonZeroU32> {
    len.try_into().ok().and_then(NonZeroU32::new)
}
fn get_texture_len(
    width: NonZeroU32,
    height: NonZeroU32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
) -> Option<u64> {
    let size = Size::new(width.get(), height.get());

    let mut len: u64 = 0;
    for level in 0..mipmaps.get() {
        let mip_len = pixels.surface_bytes(size.get_mipmap(level))?;
        len = len.checked_add(mip_len)?;
    }
    Some(len)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Texture {
    width: NonZeroU32,
    height: NonZeroU32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
    offset_index: u32,
    // A cache for data length. This is used to avoid recomputing the length
    // when the length is isn't too large.
    short_len: Option<NonZeroU32>,
}
impl Texture {
    /// Creates a new texture at offset 0.
    fn create_at_offset_0(
        width: NonZeroU32,
        height: NonZeroU32,
        mipmaps: NonZeroU8,
        pixels: PixelInfo,
    ) -> Result<Self, LayoutError> {
        // Check that length and all other calculations do not overflow
        let len =
            get_texture_len(width, height, mipmaps, pixels).ok_or(LayoutError::DataLayoutTooBig)?;

        Ok(Self {
            width,
            height,
            mipmaps,
            pixels,
            offset_index: 0,
            short_len: to_short_len(len),
        })
    }

    pub fn pixel_info(&self) -> PixelInfo {
        self.pixels
    }

    /// The level 0 size of this texture.
    fn size(&self) -> Size {
        Size::new(self.width.get(), self.height.get())
    }

    /// The level 0 mipmap of this texture.
    pub fn main(&self) -> SurfaceDescriptor {
        // PANIC SAFETY: This cannot overflow, because we already checked in the constructor
        let len = self.pixels.surface_bytes(self.size()).unwrap();
        SurfaceDescriptor::new(self.width, self.height, self.data_offset(), len)
    }
    pub fn mipmaps(&self) -> u8 {
        self.mipmaps.get()
    }
    pub fn get(&self, level: u8) -> Option<SurfaceDescriptor> {
        self.iter_mips().nth(level as usize)
    }
    pub fn iter_mips(&self) -> impl Iterator<Item = SurfaceDescriptor> {
        let mut offset = self.data_offset();
        let size_0 = self.size();
        let pixels = self.pixels;
        (0..self.mipmaps.get()).map(move |level| {
            let width = get_mipmap_size(size_0.width, level);
            let height = get_mipmap_size(size_0.height, level);
            let size = Size::new(width.get(), height.get());
            // Panic Safety: This cannot overflow, because we already checked in the constructor
            let len = pixels.surface_bytes(size).unwrap();
            let surface = SurfaceDescriptor::new(width, height, offset, len);
            offset += len;
            surface
        })
    }

    /// Internal method. This **assumes** that `offset + len` does not overflow.
    fn set_offset_index(&mut self, index: u32) {
        self.offset_index = index;
    }
}
impl DataRegion for Texture {
    fn data_len(&self) -> u64 {
        if let Some(short_len) = self.short_len {
            short_len.get() as u64
        } else {
            // Panic Safety: This cannot overflow, because we already checked in the constructor
            get_texture_len(self.width, self.height, self.mipmaps, self.pixels).unwrap()
        }
    }
    fn data_offset(&self) -> u64 {
        self.offset_index as u64 * self.data_len()
    }
    fn data_end(&self) -> u64 {
        (self.offset_index as u64 + 1) * self.data_len()
    }
}

fn get_volume_len(
    width: NonZeroU32,
    height: NonZeroU32,
    depth: NonZeroU32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
) -> Option<u64> {
    let mut len: u64 = 0;
    for level in 0..mipmaps.get() {
        let width = get_mipmap_size(width.get(), level);
        let height = get_mipmap_size(height.get(), level);
        let depth = get_mipmap_size(depth.get(), level);

        let slice_size = Size::new(width.get(), height.get());
        let slice_len = pixels.surface_bytes(slice_size)?;
        let mip_len = slice_len.checked_mul(depth.get() as u64)?;

        len = len.checked_add(mip_len)?;
    }

    Some(len)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Volume {
    width: NonZeroU32,
    height: NonZeroU32,
    depth: NonZeroU32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
}
impl Volume {
    /// Creates a new volume at offset 0.
    fn create_at_offset_0(
        width: NonZeroU32,
        height: NonZeroU32,
        depth: NonZeroU32,
        mipmaps: NonZeroU8,
        pixels: PixelInfo,
    ) -> Result<Self, LayoutError> {
        // compute the length of the entire volume (including mips) to check
        // for overflows, so we can assume no overflows in the rest of the code
        _ = get_volume_len(width, height, depth, mipmaps, pixels)
            .ok_or(LayoutError::DataLayoutTooBig)?;

        Ok(Self {
            width,
            height,
            depth,
            mipmaps,
            pixels,
        })
    }

    pub fn pixel_info(&self) -> PixelInfo {
        self.pixels
    }

    /// The level 0 mipmap of this volume.
    pub fn main(&self) -> VolumeDescriptor {
        let slice_size = Size::new(self.width.get(), self.height.get());
        // Panic Safety: This cannot overflow, because we already checked in the constructor
        let slice_len = self.pixels.surface_bytes(slice_size).unwrap();

        VolumeDescriptor {
            width: self.width,
            height: self.height,
            depth: self.depth,
            offset: 0,
            slice_len,
        }
    }
    pub fn mipmaps(&self) -> u8 {
        self.mipmaps.get()
    }
    pub fn get(&self, level: u8) -> Option<VolumeDescriptor> {
        self.iter_mips().nth(level as usize)
    }
    pub fn iter_mips(&self) -> impl Iterator<Item = VolumeDescriptor> {
        let mut offset = 0;
        let width_0 = self.width.get();
        let height_0 = self.height.get();
        let depth_0 = self.depth.get();
        let pixels = self.pixels;
        (0..self.mipmaps.get()).map(move |level| {
            let width = get_mipmap_size(width_0, level);
            let height = get_mipmap_size(height_0, level);
            let depth = get_mipmap_size(depth_0, level);
            let slice_size = Size::new(width.get(), height.get());
            // Panic Safety: This cannot overflow, because we already checked in the constructor
            let slice_len = pixels.surface_bytes(slice_size).unwrap();
            let volume = VolumeDescriptor::new(width, height, depth, offset, slice_len);
            offset += depth.get() as u64 * slice_len;
            volume
        })
    }
}
impl DataRegion for Volume {
    fn data_len(&self) -> u64 {
        // Panic Safety: This cannot overflow, because we already checked in the constructor
        get_volume_len(
            self.width,
            self.height,
            self.depth,
            self.mipmaps,
            self.pixels,
        )
        .unwrap()
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
    /// A bitset representing which faces of a cube map are present.
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
impl From<Caps2> for CubeMapFaces {
    fn from(value: Caps2) -> Self {
        let faces = (value & Caps2::CUBE_MAP_ALL_FACES).bits();
        CubeMapFaces::from_bits_truncate((faces >> 10) as u8)
    }
}
impl From<CubeMapFaces> for Caps2 {
    fn from(value: CubeMapFaces) -> Self {
        let faces = value.bits() & 0b11_1111;
        Caps2::from_bits_truncate((faces as u32) << 10)
    }
}
impl CubeMapFaces {
    /// Returns the number of cube map sides set in this bit mask.
    pub fn count(&self) -> u32 {
        self.bits().count_ones()
    }
}

/// An array of textures or (partial) cube maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureArray {
    kind: TextureArrayKind,
    array_len: u32,

    width: NonZeroU32,
    height: NonZeroU32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
    texture_short_len: Option<NonZeroU32>,
}
impl TextureArray {
    fn new(kind: TextureArrayKind, array_len: u32, first: Texture) -> Result<Self, LayoutError> {
        // must start at offset 0
        debug_assert_eq!(first.data_offset(), 0);

        // Check that the entire array does not overflow
        _ = first
            .data_len()
            .checked_mul(array_len as u64)
            .ok_or(LayoutError::DataLayoutTooBig)?;

        Ok(Self {
            kind,
            array_len,

            width: first.width,
            height: first.height,
            mipmaps: first.mipmaps,
            pixels: first.pixels,
            texture_short_len: first.short_len,
        })
    }

    pub fn pixel_info(&self) -> PixelInfo {
        self.pixels
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

    pub fn size(&self) -> Size {
        Size::new(self.width.get(), self.height.get())
    }

    pub(crate) fn first(&self) -> Texture {
        Texture {
            width: self.width,
            height: self.height,
            mipmaps: self.mipmaps,
            pixels: self.pixels,
            offset_index: 0,
            short_len: self.texture_short_len,
        }
    }
    pub fn get(&self, index: usize) -> Option<Texture> {
        if index < self.array_len as usize {
            let mut texture = self.first();
            texture.set_offset_index(index as u32);
            Some(texture)
        } else {
            None
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = Texture> {
        let mut texture = self.first();
        (0..self.array_len).map(move |index| {
            texture.set_offset_index(index);
            texture
        })
    }
}
impl DataRegion for TextureArray {
    fn data_len(&self) -> u64 {
        // Panic Safety: this can't overflow, because we checked in the constructor
        self.first().data_len() * self.array_len as u64
    }
    fn data_offset(&self) -> u64 {
        0
    }
}

/// The type and layout of the surfaces/volumes in the data section of a DDS file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataLayout {
    Texture(Texture),
    Volume(Volume),
    TextureArray(TextureArray),
}
impl DataLayout {
    pub fn from_header(header: &Header) -> Result<Self, DecodeError> {
        let layout = Self::from_header_with(header, PixelInfo::from_header(header)?)?;
        Ok(layout)
    }
    pub fn from_header_with(header: &Header, pixel_info: PixelInfo) -> Result<Self, LayoutError> {
        match header {
            Header::Dx10(dx10) => {
                if dx10.is_cube_map() {
                    if dx10.resource_dimension != ResourceDimension::Texture2D {
                        return Err(LayoutError::InvalidCubeMapDimensions);
                    }

                    let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                    let array_size = dx10.array_size;

                    // "For a 2D texture that is also a cube-map texture, array_size represents the number of cubes."
                    let cube_map_faces = array_size
                        .checked_mul(6)
                        .ok_or(LayoutError::ArraySizeTooBig(array_size))?;
                    return Ok(Self::TextureArray(
                        info.create_array(TextureArrayKind::CubeMaps, cube_map_faces)?,
                    ));
                }

                match dx10.resource_dimension {
                    ResourceDimension::Texture1D | ResourceDimension::Texture2D => {
                        let mut info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                        if dx10.resource_dimension == ResourceDimension::Texture1D {
                            info.height = NON_ZERO_U32_ONE;
                        }
                        let array_size = dx10.array_size;

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
                if let Some(faces) = dx9.cube_map_faces() {
                    if dx9.is_volume() {
                        return Err(LayoutError::InvalidCubeMapDimensions);
                    }

                    let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                    let face_count = faces.count();

                    let kind = if face_count == 6 {
                        TextureArrayKind::CubeMaps
                    } else {
                        TextureArrayKind::PartialCubeMap(faces)
                    };
                    Ok(Self::TextureArray(info.create_array(kind, face_count)?))
                } else if dx9.is_volume() {
                    let info = VolumeLayoutInfo::from_header(header, pixel_info)?;
                    Ok(Self::Volume(info.create()?))
                } else {
                    let info = SurfaceLayoutInfo::from_header(header, pixel_info)?;
                    Ok(Self::Texture(info.create()?))
                }
            }
        }
    }

    /// The size of the level 0 object.
    ///
    /// For single textures and texture arrays, this will return the size of the
    /// texture (mipmap level 0). For cube maps, this will return the size of
    /// the individual faces (mipmap level 0). For volume textures, this will
    /// return the size of the first depth slice (mipmap level 0).
    pub fn main_size(&self) -> Size {
        match self {
            DataLayout::Texture(texture) => texture.size(),
            DataLayout::Volume(volume) => volume.main().size(),
            DataLayout::TextureArray(texture_array) => texture_array.size(),
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

fn parse_dimension(dim: u32) -> Result<NonZeroU32, LayoutError> {
    NonZeroU32::new(dim).ok_or(LayoutError::ZeroDimension)
}
fn parse_mipmap_count(mipmaps: NonZeroU32) -> Result<NonZeroU8, LayoutError> {
    NonZeroU8::try_from(mipmaps).map_err(|_| LayoutError::TooManyMipMaps(mipmaps.get()))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SurfaceLayoutInfo {
    width: NonZeroU32,
    height: NonZeroU32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
}
impl SurfaceLayoutInfo {
    fn from_header(header: &Header, pixels: PixelInfo) -> Result<Self, LayoutError> {
        Ok(Self {
            width: parse_dimension(header.width())?,
            height: parse_dimension(header.height())?,
            mipmaps: parse_mipmap_count(header.mipmap_count())?,
            pixels,
        })
    }

    fn create(&self) -> Result<Texture, LayoutError> {
        Texture::create_at_offset_0(self.width, self.height, self.mipmaps, self.pixels)
    }

    fn create_array(
        &self,
        kind: TextureArrayKind,
        array_len: u32,
    ) -> Result<TextureArray, LayoutError> {
        TextureArray::new(kind, array_len, self.create()?)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VolumeLayoutInfo {
    width: NonZeroU32,
    height: NonZeroU32,
    depth: NonZeroU32,
    mipmaps: NonZeroU8,
    pixels: PixelInfo,
}
impl VolumeLayoutInfo {
    fn from_header(header: &Header, pixels: PixelInfo) -> Result<Self, LayoutError> {
        Ok(Self {
            width: parse_dimension(header.width())?,
            height: parse_dimension(header.height())?,
            depth: parse_dimension(header.depth().ok_or(LayoutError::MissingDepth)?)?,
            mipmaps: parse_mipmap_count(header.mipmap_count())?,
            pixels,
        })
    }

    fn create(&self) -> Result<Volume, LayoutError> {
        Volume::create_at_offset_0(
            self.width,
            self.height,
            self.depth,
            self.mipmaps,
            self.pixels,
        )
    }
}
