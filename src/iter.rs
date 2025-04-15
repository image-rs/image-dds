use crate::{DataLayout, DataRegion, Size, SurfaceDescriptor, Texture, Volume};

#[derive(Debug, Clone, Copy)]
pub struct SurfaceInfo<'a> {
    size: Size,
    len: u64,
    pub(crate) mipmap_level: u8,
    valid_until: std::marker::PhantomData<&'a ()>,
}
impl SurfaceInfo<'_> {
    fn new(desc: SurfaceDescriptor, mipmap_level: u8) -> Self {
        Self {
            size: desc.size(),
            len: desc.data_len(),
            mipmap_level,
            valid_until: std::marker::PhantomData,
        }
    }

    /// The size of the surface.
    pub fn size(&self) -> Size {
        self.size
    }

    /// The length in bytes on disk of the surface.
    pub fn data_len(&self) -> u64 {
        self.len
    }

    /// Whether this surface is has a mipmapping level greater than 0.
    ///
    /// For textures and texture arrays, this means that the texture is not
    /// a level 0 texture. For volumes, this means that the surface is not a
    /// depth slice of the level 0 volume.
    pub fn is_mipmap(&self) -> bool {
        self.mipmap_level != 0
    }
}

pub(crate) enum SurfaceIterator {
    Texture(TextureSurfaceIterator),
    Volume(VolumeSurfaceIterator),
}
impl SurfaceIterator {
    pub fn new(layout: DataLayout) -> Self {
        match layout {
            DataLayout::Texture(texture) => {
                SurfaceIterator::Texture(TextureSurfaceIterator::new(texture, 1))
            }
            DataLayout::Volume(volume) => {
                SurfaceIterator::Volume(VolumeSurfaceIterator::new(volume))
            }
            DataLayout::TextureArray(texture_array) => SurfaceIterator::Texture(
                TextureSurfaceIterator::new(texture_array.first(), texture_array.len() as u32),
            ),
        }
    }

    pub fn current(&self) -> Option<SurfaceInfo> {
        match self {
            Self::Texture(iter) => iter.current(),
            Self::Volume(iter) => iter.current(),
        }
    }

    pub fn advance(&mut self) {
        match self {
            Self::Texture(iter) => iter.advance(),
            Self::Volume(iter) => iter.advance(),
        }
    }

    pub fn skip_mipmaps(&mut self) -> Result<u64, ()> {
        match self {
            Self::Texture(iter) => Ok(iter.skip_mipmaps()),
            Self::Volume(iter) => iter.skip_mipmaps(),
        }
    }
}

pub(crate) struct TextureSurfaceIterator {
    first: Texture,
    len: u32,
    current_index: u32,
    current_level: u8,
}
impl TextureSurfaceIterator {
    fn new(first: Texture, len: u32) -> Self {
        Self {
            first,
            len,
            current_index: 0,
            current_level: 0,
        }
    }

    fn current(&self) -> Option<SurfaceInfo> {
        if self.current_index < self.len {
            let desc = self.first.get(self.current_level);
            debug_assert!(desc.is_some());
            Some(SurfaceInfo::new(desc?, self.current_level))
        } else {
            None
        }
    }

    fn advance(&mut self) {
        if self.current_index < self.len {
            // this can never overflow, because we ensure that
            // `current_level < first.mipmaps()`
            let next_level = self.current_level + 1;
            if next_level < self.first.mipmaps() {
                self.current_level = next_level;
            } else {
                self.current_index += 1;
                self.current_level = 0;
            }
        }
    }

    fn skip_mipmaps(&mut self) -> u64 {
        if self.current_index < self.len && self.current_level != 0 {
            let mut skipped_bytes = 0;
            for surface in self.first.iter_mips().skip(self.current_level as usize) {
                skipped_bytes += surface.data_len();
            }

            self.current_index += 1;
            self.current_level = 0;

            skipped_bytes
        } else {
            0
        }
    }
}

pub(crate) struct VolumeSurfaceIterator {
    volume: Volume,
    current_level: u8,
    current_depth: u32,
}
impl VolumeSurfaceIterator {
    fn new(volume: Volume) -> Self {
        Self {
            volume,
            current_level: 0,
            current_depth: 0,
        }
    }

    fn current(&self) -> Option<SurfaceInfo> {
        let v = self.volume.get(self.current_level)?;
        debug_assert!(self.current_depth < v.depth());
        let desc = v.get_depth_slice(self.current_depth);
        debug_assert!(desc.is_some());
        Some(SurfaceInfo::new(desc?, self.current_level))
    }

    fn advance(&mut self) {
        if let Some(v) = self.volume.get(self.current_level) {
            let next_depth = self.current_depth + 1;
            if next_depth < v.depth() {
                self.current_depth = next_depth;
            } else {
                self.current_level += 1;
                self.current_depth = 0;
            }
        }
    }

    fn skip_mipmaps(&mut self) -> Result<u64, ()> {
        // we cannot skip anything within a volume
        if self.current_depth != 0 {
            return Err(());
        }

        // don't move at the start or end of a volume
        if self.current_level == 0 || self.current_level >= self.volume.mipmaps() {
            return Ok(0);
        }

        let mut skipped_bytes = 0;
        for surface in self.volume.iter_mips().skip(self.current_level as usize) {
            skipped_bytes += surface.data_len();
        }

        self.current_level = self.volume.mipmaps();

        Ok(skipped_bytes)
    }
}
