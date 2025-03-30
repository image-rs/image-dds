use std::io::{Read, Seek};

use crate::{
    decode, decode_rect,
    header::{Header, ParseOptions},
    iter::{SurfaceInfo, SurfaceIterator},
    util, ColorFormat, CubeMapFaces, DataLayout, DecodeError, DecodeOptions, Format, ImageViewMut,
    Rect, Size, TextureArrayKind,
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
    pub fn layout(&self) -> DataLayout {
        self.layout
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
    pub fn layout(&self) -> DataLayout {
        self.info.layout()
    }

    /// The size of the level 0 object.
    ///
    /// For single textures and texture arrays, this will return the size of the
    /// texture (mipmap level 0). For cube maps, this will return the size of
    /// the individual faces (mipmap level 0). For volume textures, this will
    /// return the size of the first depth slice (mipmap level 0).
    pub fn main_size(&self) -> Size {
        self.info.layout().main_size()
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

    /// Returns information about the surface about to be read.
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
    pub fn read_surface(&mut self, image: ImageViewMut) -> Result<(), DecodeError>
    where
        R: Read,
    {
        let current = self.iter.current().ok_or(DecodeError::NoMoreSurfaces)?;
        if image.size() != current.size() {
            return Err(DecodeError::UnexpectedSurfaceSize);
        }

        decode(&mut self.reader, image, self.info.format, &self.options)?;

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
            buffer,
            row_pitch,
            color,
            current.size(),
            rect,
            self.info.format,
            &self.options,
        )?;

        self.iter.advance();
        Ok(())
    }

    /// Skips over the next surface.
    ///
    /// This behaves the same as [`Decoder::read_surface_rect`] when decoding
    /// an empty rectangle.
    pub fn skip_surface(&mut self) -> Result<(), DecodeError>
    where
        R: Seek,
    {
        let current = self.iter.current().ok_or(DecodeError::NoMoreSurfaces)?;

        util::io_skip_exact(&mut self.reader, current.data_len())?;

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

    /// Reads all faces of a cube map into the given image and skips any
    /// mipmaps of those faces.
    ///
    /// The faces of the cube map are arranged in a shape of an unfolded cube
    /// like this:
    ///
    /// ```txt
    /// +----+----+----+----+  `.
    /// |    | +Y |    |    |   |
    /// +----+----+----+----+   |
    /// | -X | +Z | +X | -Z |   | 3 * height
    /// +----+----+----+----+   |
    /// |    | -Y |    |    |   |
    /// +----+----+----+----+  ´
    /// .___________________.
    ///      4 * width
    /// ```
    ///
    /// As such, the output image buffer is expected to have a size of
    /// `width * 4` by `height * 3`, where `width` and `height` are the
    /// dimensions of a single cube map face.
    ///
    /// For partial cube maps, only the faces that are present in the DDS file
    /// are read. The faces are arranged in the same order as for full cube
    /// maps.
    pub fn read_cube_map(&mut self, image: ImageViewMut) -> Result<(), DecodeError>
    where
        R: Read + Seek,
    {
        let layout = self.layout();
        let texture_array = layout.texture_array().ok_or(DecodeError::NotACubeMap)?;
        let faces = match texture_array.kind() {
            TextureArrayKind::Textures => return Err(DecodeError::NotACubeMap),
            TextureArrayKind::CubeMaps => CubeMapFaces::ALL,
            TextureArrayKind::PartialCubeMap(cube_map_faces) => cube_map_faces,
        };

        let face_size = texture_array.size();
        let image_width = face_size.width.checked_mul(4);
        let image_height = face_size.height.checked_mul(3);
        if image_width != Some(image.width()) || image_height != Some(image.height()) {
            return Err(DecodeError::UnexpectedSurfaceSize);
        }

        let face_offsets = [
            (CubeMapFaces::POSITIVE_X, 2, 1),
            (CubeMapFaces::NEGATIVE_X, 0, 1),
            (CubeMapFaces::POSITIVE_Y, 1, 0),
            (CubeMapFaces::NEGATIVE_Y, 1, 2),
            (CubeMapFaces::POSITIVE_Z, 1, 1),
            (CubeMapFaces::NEGATIVE_Z, 3, 1),
        ];

        let color = image.color();
        let bytes_per_pixel = color.bytes_per_pixel() as usize;
        let row_pitch = image.row_pitch();
        let rect = Rect::new(0, 0, face_size.width, face_size.height);
        let image_bytes = image.data;

        for (_, x, y) in face_offsets
            .into_iter()
            .filter(|(face, _, _)| faces.contains(*face))
        {
            let current = self.iter.current().ok_or(DecodeError::NoMoreSurfaces)?;
            if current.size() != face_size {
                return Err(DecodeError::UnexpectedSurfaceSize);
            }

            let offset_x = x * face_size.width;
            let offset_y = y * face_size.height;
            let offset = offset_y as usize * row_pitch + offset_x as usize * bytes_per_pixel;

            self.read_surface_rect(&mut image_bytes[offset..], row_pitch, rect, color)?;
            self.skip_mipmaps()?;
        }

        Ok(())
    }
}
