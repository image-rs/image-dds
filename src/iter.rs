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

#[derive(Clone, PartialEq)]
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

    pub fn rewind(&mut self) {
        match self {
            Self::Texture(iter) => iter.rewind(),
            Self::Volume(iter) => iter.rewind(),
        }
    }

    pub fn skip_mipmaps(&mut self) -> Result<u64, ()> {
        match self {
            Self::Texture(iter) => Ok(iter.skip_mipmaps()),
            Self::Volume(iter) => iter.skip_mipmaps(),
        }
    }

    /// How many bytes have been read so far to reach the current surface.
    pub fn elapsed_bytes(&self) -> u64 {
        match self {
            Self::Texture(iter) => iter.elapsed_bytes(),
            Self::Volume(iter) => iter.elapsed_bytes(),
        }
    }
}

#[derive(Clone, PartialEq)]
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

    fn rewind(&mut self) {
        if self.current_level > 0 {
            self.current_level -= 1;
        } else if self.current_index > 0 {
            self.current_index -= 1;
            self.current_level = self.first.mipmaps() - 1;
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

    pub fn elapsed_bytes(&self) -> u64 {
        // start with the full textures already read
        let mut bytes = self.first.data_len() * self.current_index as u64;
        // add the mipmaps of the current texture
        for level in 0..self.current_level {
            bytes += self.first.get(level).unwrap().data_len();
        }
        bytes
    }
}

#[derive(Clone, PartialEq)]
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

    fn rewind(&mut self) {
        if self.current_depth > 0 {
            self.current_depth -= 1;
        } else if self.current_level > 0 {
            self.current_level -= 1;
            let v = self.volume.get(self.current_level).unwrap();
            self.current_depth = v.depth() - 1;
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

    pub fn elapsed_bytes(&self) -> u64 {
        let mut bytes = 0;

        // start with the full volumes already read
        for i in 0..self.current_level {
            bytes += self.volume.get(i).unwrap().data_len();
        }
        // add the depth slices of the current slice
        let current_volume = self.volume.get(self.current_level);
        if let Some(v) = current_volume {
            let depth_slice_bytes = v.get_depth_slice(0).map_or(0, |s| s.data_len());
            bytes += depth_slice_bytes * self.current_depth as u64;
        }

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts that the iterator works the same as a manual iteration
    fn assert_iter(layout: DataLayout) {
        /// advances the iterator and checks that advance and rewind are implemented correctly
        fn advance(iter: &mut SurfaceIterator) {
            let state_a = iter.clone();
            iter.advance();
            let mut state_b = iter.clone();
            state_b.rewind();
            assert!(state_a == state_b);
        }

        let mut iter = SurfaceIterator::new(layout);
        let mut elapsed_bytes = 0;

        match layout {
            DataLayout::Texture(texture) => {
                for mip in texture.iter_mips() {
                    let surface = iter.current().unwrap();
                    assert_eq!(surface.size(), mip.size());
                    assert_eq!(surface.data_len(), mip.data_len());
                    advance(&mut iter);
                    elapsed_bytes += mip.data_len();
                    assert_eq!(elapsed_bytes, iter.elapsed_bytes());
                }
            }
            DataLayout::TextureArray(array) => {
                for texture in array.iter() {
                    for mip in texture.iter_mips() {
                        let surface = iter.current().unwrap();
                        assert_eq!(surface.size(), mip.size());
                        assert_eq!(surface.data_len(), mip.data_len());
                        advance(&mut iter);
                        elapsed_bytes += mip.data_len();
                        assert_eq!(elapsed_bytes, iter.elapsed_bytes());
                    }
                }
            }
            DataLayout::Volume(volume) => {
                for mip in volume.iter_mips() {
                    for slice in mip.iter_depth_slices() {
                        let surface = iter.current().unwrap();
                        assert_eq!(surface.size(), slice.size());
                        assert_eq!(surface.data_len(), slice.data_len());
                        advance(&mut iter);
                        elapsed_bytes += slice.data_len();
                        assert_eq!(elapsed_bytes, iter.elapsed_bytes());
                    }
                }
            }
        }

        assert!(iter.current().is_none());
        assert_eq!(iter.elapsed_bytes(), layout.data_len());
    }

    #[test]
    fn test_surface_iterator() {
        use crate::{header::*, *};

        fn test(header: impl Into<Header>) {
            assert_iter(DataLayout::from_header(&header.into()).unwrap());
        }

        test(Header::new_image(100, 300, Format::BC1_UNORM));
        test(Header::new_image(100, 300, Format::BC1_UNORM).with_mipmap_count(5));
        test(Header::new_image(100, 300, Format::BC1_UNORM).with_mipmaps());
        test(Header::new_cube_map(100, 300, Format::BC1_UNORM));
        test(Header::new_cube_map(100, 300, Format::BC1_UNORM).with_mipmap_count(5));
        test(Header::new_cube_map(100, 300, Format::BC1_UNORM).with_mipmaps());
        test(Header::new_volume(100, 300, 500, Format::BC1_UNORM));
        test(Header::new_volume(100, 300, 500, Format::BC1_UNORM).with_mipmap_count(5));
        test(Header::new_volume(100, 300, 500, Format::BC1_UNORM).with_mipmaps());
    }
}
