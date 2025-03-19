use std::io::{Read, Seek};

use crate::{
    decode, decode_rect,
    header::{Header, ParseOptions},
    ColorFormat, DataLayout, DataRegion, DecodeError, DecodeOptions, Format, Rect, Size,
    SurfaceDescriptor, Texture, Volume,
};

/// Information about the header, pixel format, and data layout of a DDS file.
///
/// This is immutable since the data layout and format depend on the header.
/// In particular, the data layout is guaranteed to be generated from the
/// header.
#[derive(Debug, Clone)]
pub struct DdsInfo {
    header: Header,
    format: Format,
    layout: DataLayout,
}

impl DdsInfo {
    /// Creates a new decoder by reading the header from the given reader.
    ///
    /// This is equivalent to calling `Decoder::read_with_options(r, ParseOptions::default())`.
    /// See [`Self::read_with_options`] for more details.
    pub fn read<R: Read>(r: &mut R) -> Result<Self, DecodeError> {
        Self::read_with_options(r, &ParseOptions::default())
    }
    /// Creates a new decoder with the given options by reading the header from the given reader.
    ///
    /// If this operations succeeds, the given reader will be positioned at the start of the data
    /// section. All offsets in [`DataLayout`] are relative to this position.
    pub fn read_with_options<R: Read>(
        r: &mut R,
        options: &ParseOptions,
    ) -> Result<Self, DecodeError> {
        let header = Header::read(r, options)?;
        Self::new(header)
    }

    pub fn new(header: Header) -> Result<Self, DecodeError> {
        // detect format
        let format = Format::from_header(&header)?;

        Self::new_with_format(header, format)
    }
    pub fn new_with_format(header: Header, format: Format) -> Result<Self, DecodeError> {
        // data layout
        let layout = DataLayout::from_header_with(&header, format.into())?;

        Ok(Self {
            header,
            format,
            layout,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn format(&self) -> Format {
        self.format
    }
    pub fn layout(&self) -> &DataLayout {
        &self.layout
    }
}

/// A decoder for reading the pixel data of a DDS file.
pub struct Decoder<R> {
    reader: R,

    info: DdsInfo,
    iter: SurfaceIterator,
    pub options: DecodeOptions,
}
impl<R> Decoder<R> {
    pub fn new(reader: R) -> Result<Self, DecodeError>
    where
        R: Read,
    {
        Self::new_with_options(reader, &ParseOptions::default())
    }
    pub fn new_with_options(mut reader: R, options: &ParseOptions) -> Result<Self, DecodeError>
    where
        R: Read,
    {
        let info = DdsInfo::read_with_options(&mut reader, options)?;

        Self::from_info(reader, info)
    }

    pub fn from_info(reader: R, info: DdsInfo) -> Result<Self, DecodeError> {
        Ok(Self {
            reader,
            iter: SurfaceIterator::new(info.layout()),
            info,
            options: DecodeOptions::default(),
        })
    }

    pub fn info(&self) -> &DdsInfo {
        &self.info
    }
    pub fn format(&self) -> Format {
        self.info.format()
    }
    pub fn layout(&self) -> &DataLayout {
        self.info.layout()
    }

    /// The size of the level 0 object.
    ///
    /// For single textures and texture arrays, this will return the size of the
    /// texture (mipmap level 0). For cube maps, this will return the size of
    /// the individual faces (mipmap level 0). For volume textures, this will
    /// return the size of the first depth slice (mipmap level 0).
    pub fn main_size(&self) -> Size {
        self.info.header().size()
    }
    /// The native color of the DDS file.
    ///
    /// See [`Format::precision`] for more information about the precision of
    /// the color format.
    pub fn native_color(&self) -> ColorFormat {
        self.info.format().color()
    }

    pub fn into_reader(self) -> R {
        self.reader
    }

    /// Returns information about the next surface.
    ///
    /// The returned value is not valid after calling `next_surface`.
    ///
    /// If there are no more surfaces, `None` is returned.
    pub fn surface_info(&self) -> Option<SurfaceInfo<'_>> {
        self.iter.current()
    }

    /// Reads the next surface into the given buffer.
    ///
    /// The next surface is determined by the data layout of the DDS file. For
    /// volume textures, this function will read the next depth slice.
    pub fn read_surface(&mut self, buffer: &mut [u8], color: ColorFormat) -> Result<(), DecodeError>
    where
        R: Read,
    {
        let current = self.iter.current().ok_or(DecodeError::NoMoreSurfaces)?;

        decode(
            &mut self.reader,
            self.info.format,
            current.size,
            color,
            buffer,
            &self.options,
        )?;

        self.iter.advance();
        Ok(())
    }

    /// Reads a rectangle of the next surface into the given buffer.
    ///
    /// Similarly to [`Decoder::read_surface`], this operation will consume the
    /// current surface and advance to the next one. It is not possible to read
    /// multiple rectangles from the same surface. If this is what you want to
    /// do, use the [`decode_rect`] function instead.
    pub fn read_surface_rect(
        &mut self,
        buffer: &mut [u8],
        row_pitch: usize,
        rect: Rect,
        color: ColorFormat,
    ) -> Result<(), DecodeError>
    where
        R: Read + Seek,
    {
        let current = self.iter.current().ok_or(DecodeError::NoMoreSurfaces)?;

        decode_rect(
            &mut self.reader,
            self.info.format,
            current.size,
            rect,
            color,
            buffer,
            row_pitch,
            &self.options,
        )?;

        self.iter.advance();
        Ok(())
    }

    /// Skips ahead to the next level 0 object.
    ///
    /// The main use case for this function is to skip mipmaps between cube map
    /// faces and elements of a texture array.
    ///
    /// Volume textures are not allowed to call this function within a volume.
    /// It's only valid to call this function at the start or end of a volume.
    /// Because of this, it can only be used to skip to the end of the file for
    /// volumes.
    ///
    /// Notes:
    ///
    /// - If the DDS file does not contain any mipmaps, this is a no-op.
    /// - Calling this at the start or end of a DDS file is a no-op.
    pub fn skip_mipmaps(&mut self) -> Result<(), DecodeError>
    where
        R: Seek,
    {
        if let Ok(skip) = self.iter.skip_mipmaps() {
            if skip > 0 {
                self.reader.seek(std::io::SeekFrom::Current(skip as i64))?;
            }
            Ok(())
        } else {
            Err(DecodeError::CannotSkipMipmapsInVolume)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SurfaceInfo<'a> {
    size: Size,
    len: u64,
    valid_until: std::marker::PhantomData<&'a ()>,
}
impl SurfaceInfo<'_> {
    fn from_descriptor(desc: SurfaceDescriptor) -> Self {
        Self {
            size: desc.size(),
            len: desc.data_len(),
            valid_until: std::marker::PhantomData,
        }
    }

    pub fn size(&self) -> Size {
        self.size
    }
    pub fn data_len(&self) -> u64 {
        self.len
    }
}

enum SurfaceIterator {
    Texture(TextureSurfaceIterator),
    Volume(VolumeSurfaceIterator),
}
impl SurfaceIterator {
    fn new(layout: &DataLayout) -> Self {
        match layout {
            DataLayout::Texture(texture) => {
                SurfaceIterator::Texture(TextureSurfaceIterator::new(texture.clone(), 1))
            }
            DataLayout::Volume(volume) => {
                SurfaceIterator::Volume(VolumeSurfaceIterator::new(volume.clone()))
            }
            DataLayout::TextureArray(texture_array) => {
                SurfaceIterator::Texture(TextureSurfaceIterator::new(
                    texture_array.first.clone(),
                    texture_array.len() as u32,
                ))
            }
        }
    }

    fn current(&self) -> Option<SurfaceInfo> {
        match self {
            Self::Texture(iter) => iter.current(),
            Self::Volume(iter) => iter.current(),
        }
    }

    fn advance(&mut self) {
        match self {
            Self::Texture(iter) => iter.advance(),
            Self::Volume(iter) => iter.advance(),
        }
    }

    fn skip_mipmaps(&mut self) -> Result<u64, ()> {
        match self {
            Self::Texture(iter) => Ok(iter.skip_mipmaps()),
            Self::Volume(iter) => iter.skip_mipmaps(),
        }
    }
}

struct TextureSurfaceIterator {
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
            Some(SurfaceInfo::from_descriptor(desc?))
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

struct VolumeSurfaceIterator {
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
        Some(SurfaceInfo::from_descriptor(desc?))
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
