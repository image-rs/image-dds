use bitflags::bitflags;

use crate::{DdsCaps2, DecodeError, Header, MiscFlags, ResourceDimension, Size, SupportedFormat};

pub trait Region {
    /// The number of bytes this object occupies in the data section of a DDS file.
    ///
    /// It is guaranteed that `self.offset() + self.len() <= u64::MAX`.
    fn byte_len(&self) -> u64;
    /// The byte offset of this object in the data section of a DDS file.
    fn byte_offset(&self) -> u64;
    /// The byte offset of the byte after this object in the data section of a DDS file.
    ///
    /// This is equivalent to `self.offset() + self.len()`.
    fn byte_end(&self) -> u64 {
        self.byte_offset() + self.byte_len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SurfaceDescriptor {
    width: u32,
    height: u32,
    offset: u64,
    len: u64,
}
impl SurfaceDescriptor {
    fn new(width: u32, height: u32, offset: u64, len: u64) -> Result<Self, DecodeError> {
        debug_assert!(width > 0);
        debug_assert!(height > 0);
        debug_assert!(len > 0);

        if offset.checked_add(len).is_none() {
            return Err(DecodeError::DataLayoutTooBig);
        }

        Ok(Self {
            width,
            height,
            offset,
            len,
        })
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
impl Region for SurfaceDescriptor {
    fn byte_len(&self) -> u64 {
        self.len
    }
    fn byte_offset(&self) -> u64 {
        self.offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VolumeDescriptor {
    width: u32,
    height: u32,
    depth: u32,
    offset: u64,
    slice_len: u64,
}
impl VolumeDescriptor {
    fn new(
        width: u32,
        height: u32,
        depth: u32,
        offset: u64,
        slice_len: u64,
    ) -> Result<Self, DecodeError> {
        debug_assert!(width > 0);
        debug_assert!(height > 0);
        debug_assert!(depth > 0);
        debug_assert!(slice_len > 0);

        // check that `offset + len` does not overflow
        let end = slice_len
            .checked_mul(depth as u64)
            .and_then(|len| offset.checked_add(len));
        if end.is_none() {
            return Err(DecodeError::DataLayoutTooBig);
        }

        Ok(Self {
            width,
            height,
            depth,
            offset,
            slice_len,
        })
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
    /// # use ddsd::{VolumeDescriptor, Region};
    /// # fn get_volume() -> VolumeDescriptor { todo!() }
    /// let volume: VolumeDescriptor = get_volume();
    /// for (depth, slice) in volume.iter_depth_slices().enumerate() {
    ///     println!("Slice {} starts at {}", depth, slice.byte_offset());
    /// }
    /// ```
    pub fn iter_depth_slices(&self) -> impl Iterator<Item = SurfaceDescriptor> {
        let Self {
            width,
            height,
            offset,
            slice_len,
            ..
        } = self.clone();
        (0..self.depth).map(move |depth| SurfaceDescriptor {
            width,
            height,
            offset: offset + depth as u64 * slice_len,
            len: slice_len,
        })
    }
}
impl Region for VolumeDescriptor {
    fn byte_len(&self) -> u64 {
        // Cannot overflow. See `VolumeDescriptor::new`.
        self.slice_len * self.depth as u64
    }
    fn byte_offset(&self) -> u64 {
        self.offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Texture {
    surfaces: Box<[SurfaceDescriptor]>,
}
impl Texture {
    fn new(surfaces: Vec<SurfaceDescriptor>) -> Self {
        // we need at least one surface
        debug_assert!(!surfaces.is_empty());
        // mips must be sorted by size
        debug_assert!(surfaces
            .windows(2)
            .all(|pair| pair[0].width >= pair[1].width && pair[0].height >= pair[1].height));
        // must be packed
        debug_assert!(surfaces
            .windows(2)
            .all(|pair| pair[0].byte_end() == pair[1].byte_offset()));

        Self {
            surfaces: surfaces.into_boxed_slice(),
        }
    }

    pub fn main(&self) -> &SurfaceDescriptor {
        &self.surfaces[0]
    }
    pub fn surfaces(&self) -> &[SurfaceDescriptor] {
        self.surfaces.as_ref()
    }
}
impl Region for Texture {
    fn byte_len(&self) -> u64 {
        let last = &self.surfaces[self.surfaces.len() - 1];
        last.byte_end() - self.main().byte_offset()
    }
    fn byte_offset(&self) -> u64 {
        self.main().byte_offset()
    }
    fn byte_end(&self) -> u64 {
        let last = &self.surfaces[self.surfaces.len() - 1];
        last.byte_end()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Volume {
    volumes: Box<[VolumeDescriptor]>,
}
impl Volume {
    fn new(volumes: Vec<VolumeDescriptor>) -> Self {
        // we need at least one volume
        debug_assert!(!volumes.is_empty());
        // mips must be sorted by size
        debug_assert!(volumes.windows(2).all(|pair| pair[0].width >= pair[1].width
            && pair[0].height >= pair[1].height
            && pair[0].depth >= pair[1].depth));
        // the offset of the next surface must be offset + len of the previous surface
        debug_assert!(volumes
            .windows(2)
            .all(|pair| pair[0].byte_end() == pair[1].byte_offset()));
        // volume must start at offset 0
        debug_assert_eq!(volumes[0].byte_offset(), 0);

        Self {
            volumes: volumes.into_boxed_slice(),
        }
    }

    pub fn main(&self) -> &VolumeDescriptor {
        &self.volumes[0]
    }
    pub fn volumes(&self) -> &[VolumeDescriptor] {
        self.volumes.as_ref()
    }
}
impl Region for Volume {
    fn byte_len(&self) -> u64 {
        let last = &self.volumes[self.volumes.len() - 1];
        last.byte_end()
    }
    fn byte_offset(&self) -> u64 {
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
        self.contains(CubeMapSide::POSITIVE_X) as u32
            + self.contains(CubeMapSide::NEGATIVE_X) as u32
            + self.contains(CubeMapSide::POSITIVE_Y) as u32
            + self.contains(CubeMapSide::NEGATIVE_Y) as u32
            + self.contains(CubeMapSide::POSITIVE_Z) as u32
            + self.contains(CubeMapSide::NEGATIVE_Z) as u32
    }
}

/// An array of textures or (partial) cube maps.
///
/// The array is guaranteed to have at least one texture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureArray {
    kind: TextureArrayKind,
    // TODO: This needs a new design.
    // Allocating potentially 2^32-1 textures is neither fast nor secure.
    // Ideally, we would only store the first texture and generate the rest on demand,
    // maybe even in a way to reuse the same allocation.
    textures: Box<[Texture]>,
}
impl TextureArray {
    fn new(kind: TextureArrayKind, textures: Vec<Texture>) -> Self {
        // we need at least one texture
        debug_assert!(!textures.is_empty());
        // the offset of the next surface must be offset + len of the previous surface
        debug_assert!(textures
            .windows(2)
            .all(|pair| pair[0].byte_end() == pair[1].byte_offset()));
        // must start at offset 0
        debug_assert_eq!(textures[0].byte_offset(), 0);

        Self {
            kind,
            textures: textures.into_boxed_slice(),
        }
    }

    pub fn kind(&self) -> TextureArrayKind {
        self.kind
    }
    pub fn textures(&self) -> &[Texture] {
        self.textures.as_ref()
    }
}
impl Region for TextureArray {
    fn byte_len(&self) -> u64 {
        let last = &self.textures[self.textures.len() - 1];
        last.byte_end()
    }
    fn byte_offset(&self) -> u64 {
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
    pub fn from_header(
        header: &Header,
        pixel_format: SupportedFormat,
    ) -> Result<Self, DecodeError> {
        if let Some(ref header_dxt10) = header.dxt10 {
            // DirectX 10+

            match header_dxt10.resource_dimension {
                ResourceDimension::Texture1D => {
                    let mut info = SurfaceLayoutInfo::from_header(header, pixel_format)?;
                    info.height = 1;
                    let array_size = header_dxt10.array_size;

                    if array_size == 1 {
                        Ok(Self::Texture(info.create(0)?))
                    } else {
                        Ok(Self::TextureArray(TextureArray::new(
                            TextureArrayKind::Textures,
                            info.create_many(array_size)?,
                        )))
                    }
                }
                ResourceDimension::Texture2D => {
                    let info = SurfaceLayoutInfo::from_header(header, pixel_format)?;
                    // TODO: Limit array_size
                    let array_size = header_dxt10.array_size;
                    let is_cube_map = header_dxt10.misc_flag.contains(MiscFlags::TEXTURE_CUBE);

                    if is_cube_map {
                        // "For a 2D texture that is also a cube-map texture, array_size represents the number of cubes."
                        return Ok(Self::TextureArray(TextureArray::new(
                            TextureArrayKind::CubeMaps,
                            info.create_many(array_size * 6)?,
                        )));
                    }

                    if array_size == 1 {
                        Ok(Self::Texture(info.create(0)?))
                    } else {
                        Ok(Self::TextureArray(TextureArray::new(
                            TextureArrayKind::Textures,
                            info.create_many(array_size)?,
                        )))
                    }
                }
                ResourceDimension::Texture3D => {
                    let info = VolumeLayoutInfo::from_header(header, pixel_format)?;
                    Ok(Self::Volume(info.create()?))
                }
            }
        } else {
            // DirectX <=9

            if header.caps2.contains(DdsCaps2::VOLUME) {
                let info = VolumeLayoutInfo::from_header(header, pixel_format)?;
                Ok(Self::Volume(info.create()?))
            } else if header.caps2.contains(DdsCaps2::CUBE_MAP) {
                let info = SurfaceLayoutInfo::from_header(header, pixel_format)?;
                let sides = CubeMapSide::from(header.caps2);
                let side_count = sides.count();

                Ok(Self::TextureArray(TextureArray::new(
                    if side_count == 6 {
                        TextureArrayKind::CubeMaps
                    } else {
                        TextureArrayKind::PartialCubeMap(sides)
                    },
                    info.create_many(side_count)?,
                )))
            } else {
                let info = SurfaceLayoutInfo::from_header(header, pixel_format)?;
                Ok(Self::Texture(info.create(0)?))
            }
        }
    }
}
impl Region for DataLayout {
    fn byte_len(&self) -> u64 {
        match self {
            DataLayout::Texture(texture) => texture.byte_len(),
            DataLayout::Volume(volume) => volume.byte_len(),
            DataLayout::TextureArray(textures) => textures.byte_len(),
        }
    }
    fn byte_offset(&self) -> u64 {
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
    pixel_format: SupportedFormat,
}
impl SurfaceLayoutInfo {
    fn from_header(header: &Header, pixel_format: SupportedFormat) -> Result<Self, DecodeError> {
        // width and height are always required, so we can read them without checking `flags`.
        let width = header.width;
        let height = header.height;

        if width == 0 || height == 0 {
            return Err(DecodeError::ZeroDimension);
        }

        let mipmaps = header.mipmap_count.unwrap_or(1).max(1);
        if mipmaps > 32 {
            return Err(DecodeError::TooManyMipMaps(mipmaps));
        }

        Ok(Self {
            width,
            height,
            mipmaps: mipmaps as u8,
            pixel_format,
        })
    }

    fn create_desc(&self, offset: u64, level: u8) -> Result<SurfaceDescriptor, DecodeError> {
        assert!(level < self.mipmaps);

        let width = get_mip_size(self.width, level);
        let height = get_mip_size(self.height, level);
        let len = self
            .pixel_format
            .get_surface_bytes(Size::new(width, height))
            .ok_or(DecodeError::DataLayoutTooBig)?;

        SurfaceDescriptor::new(width, height, offset, len)
    }

    fn create(&self, mut offset: u64) -> Result<Texture, DecodeError> {
        let mut surfaces = Vec::with_capacity(self.mipmaps as usize);
        for level in 0..self.mipmaps {
            let surface = self.create_desc(offset, level)?;
            offset = surface.byte_end();
            surfaces.push(surface);
        }

        Ok(Texture::new(surfaces))
    }

    fn create_many(&self, count: u32) -> Result<Vec<Texture>, DecodeError> {
        debug_assert!(count > 0);

        let mut textures = Vec::with_capacity(count as usize);
        let mut offset = 0;
        for _ in 0..count {
            let texture = self.create(offset)?;
            offset = texture.byte_end();
            textures.push(texture);
        }

        Ok(textures)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VolumeLayoutInfo {
    width: u32,
    height: u32,
    depth: u32,
    mipmaps: u8,
    pixel_format: SupportedFormat,
}
impl VolumeLayoutInfo {
    fn from_header(header: &Header, pixel_format: SupportedFormat) -> Result<Self, DecodeError> {
        let SurfaceLayoutInfo {
            width,
            height,
            mipmaps,
            pixel_format,
        } = SurfaceLayoutInfo::from_header(header, pixel_format)?;

        let depth = header.depth.ok_or(DecodeError::MissingDepth)?;

        if width == 0 || height == 0 || depth == 0 {
            return Err(DecodeError::ZeroDimension);
        }

        Ok(Self {
            width,
            height,
            depth,
            mipmaps,
            pixel_format,
        })
    }

    fn create_desc(&self, offset: u64, level: u8) -> Result<VolumeDescriptor, DecodeError> {
        assert!(level < self.mipmaps);

        let width = get_mip_size(self.width, level);
        let height = get_mip_size(self.height, level);
        let depth = get_mip_size(self.depth, level);
        let slice_len = self
            .pixel_format
            .get_surface_bytes(Size::new(width, height))
            .ok_or(DecodeError::DataLayoutTooBig)?;

        VolumeDescriptor::new(width, height, depth, offset, slice_len)
    }

    fn create(&self) -> Result<Volume, DecodeError> {
        let mut volumes = Vec::with_capacity(self.mipmaps as usize);
        let mut offset = 0;
        for level in 0..self.mipmaps {
            let surface = self.create_desc(offset, level)?;
            offset = surface.byte_end();
            volumes.push(surface);
        }

        Ok(Volume::new(volumes))
    }
}
