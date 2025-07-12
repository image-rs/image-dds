use std::io::{Read, Seek};

use crate::{
    decode, decode_rect,
    header::{Header, ParseOptions},
    iter::{SurfaceInfo, SurfaceIterator},
    util, ColorFormat, CubeMapFaces, DataLayout, DecodeOptions, DecodingError, Format,
    ImageViewMut, Rect, Size, TextureArrayKind,
};

/// A decoder for reading the pixel data of a DDS file.
///
/// See [crate-level documentation](crate) for usage examples.
pub struct Decoder<R> {
    reader: R,

    header: Header,
    format: Format,
    layout: DataLayout,

    iter: SurfaceIterator,
    pub options: DecodeOptions,
}
impl<R> Decoder<R> {
    /// Creates a new decoder from the given reader.
    ///
    /// Same as [`Self::new_with_options`] with default options.
    pub fn new(reader: R) -> Result<Self, DecodingError>
    where
        R: Read,
    {
        Self::new_with_options(reader, &ParseOptions::default())
    }
    /// Creates a new decoder from the given reader.
    ///
    /// This will read the header from the reader. How the header is read can
    /// be configured with the given options.
    pub fn new_with_options(mut reader: R, options: &ParseOptions) -> Result<Self, DecodingError>
    where
        R: Read,
    {
        let header = Header::read(&mut reader, options)?;
        Self::from_header(reader, header)
    }

    /// Creates a new decoder from the given reader and header.
    ///
    /// Same as [`Self::from_header_with`] with the [`Format`] being detected
    /// from the given header.
    pub fn from_header(reader: R, header: Header) -> Result<Self, DecodingError> {
        let format = Format::from_header(&header)?;
        Self::from_header_with(reader, header, format)
    }
    /// Creates a new decoder from the given reader, header, and format.
    ///
    /// Calling this method will NOT read data from the reader. The header and
    /// format are used to determine the layout of the data in the DDS file.
    /// The reader will only be used again when reading surfaces or seeking.
    pub fn from_header_with(
        reader: R,
        header: Header,
        format: Format,
    ) -> Result<Self, DecodingError> {
        let layout = DataLayout::from_header_with(&header, format.into())?;

        Ok(Self {
            reader,

            header,
            format,
            layout,

            iter: SurfaceIterator::new(layout),
            options: DecodeOptions::default(),
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

    /// The size of the level 0 object.
    ///
    /// For single textures and texture arrays, this will return the size of the
    /// texture (mipmap level 0). For cube maps, this will return the size of
    /// the individual faces (mipmap level 0). For volume textures, this will
    /// return the size of the first depth slice (mipmap level 0).
    pub fn main_size(&self) -> Size {
        self.layout.main_size()
    }
    /// The native color of the DDS file.
    ///
    /// See [`Format::precision`] for more information about the precision of
    /// the color format.
    pub fn native_color(&self) -> ColorFormat {
        self.format.color()
    }

    /// Reads the next surface into the given buffer.
    ///
    /// The next surface is determined by the data layout of the DDS file. For
    /// volume textures, this function will read the next depth slice. See
    /// [`Self::surface_info`] for more information about the next surface.
    ///
    /// If [Self::is_done] is true, this function will return an error.
    pub fn read_surface(&mut self, image: ImageViewMut) -> Result<(), DecodingError>
    where
        R: Read,
    {
        let current = self.iter.current().ok_or(DecodingError::NoMoreSurfaces)?;
        if image.size() != current.size() {
            return Err(DecodingError::UnexpectedSurfaceSize);
        }

        decode(&mut self.reader, image, self.format, &self.options)?;

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
    ) -> Result<(), DecodingError>
    where
        R: Read + Seek,
    {
        let current = self.iter.current().ok_or(DecodingError::NoMoreSurfaces)?;

        decode_rect(
            &mut self.reader,
            buffer,
            row_pitch,
            color,
            current.size(),
            rect,
            self.format,
            &self.options,
        )?;

        self.iter.advance();
        Ok(())
    }

    /// Skips over the next surface.
    ///
    /// Returns an error if there are no more surfaces.
    pub fn skip_surface(&mut self) -> Result<(), DecodingError>
    where
        R: Seek,
    {
        let current = self.iter.current().ok_or(DecodingError::NoMoreSurfaces)?;

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
    pub fn skip_mipmaps(&mut self) -> Result<(), DecodingError>
    where
        R: Seek,
    {
        if let Ok(skip) = self.iter.skip_mipmaps() {
            if skip > 0 {
                self.reader.seek(std::io::SeekFrom::Current(skip as i64))?;
            }
            Ok(())
        } else {
            Err(DecodingError::CannotSkipMipmapsInVolume)
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
    /// dimensions of a single cube map face. (See [`Self::main_size`] for the
    /// size of a single face.)
    ///
    /// For partial cube maps, only the faces that are present in the DDS file
    /// are read. The faces are arranged in the same order as for full cube
    /// maps.
    ///
    /// It's recommended to use `self.layout().is_cube_map()` to determine
    /// whether the DDS file is a cube map or not.
    pub fn read_cube_map(&mut self, mut image: ImageViewMut) -> Result<(), DecodingError>
    where
        R: Read + Seek,
    {
        let layout = self.layout();
        let texture_array = layout.texture_array().ok_or(DecodingError::NotACubeMap)?;
        let faces = match texture_array.kind() {
            TextureArrayKind::Textures => return Err(DecodingError::NotACubeMap),
            TextureArrayKind::CubeMaps => CubeMapFaces::ALL,
            TextureArrayKind::PartialCubeMap(cube_map_faces) => cube_map_faces,
        };

        let face_size = texture_array.size();
        let image_width = face_size.width.checked_mul(4);
        let image_height = face_size.height.checked_mul(3);
        if image_width != Some(image.width()) || image_height != Some(image.height()) {
            return Err(DecodingError::UnexpectedSurfaceSize);
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
        let image_bytes = image.data();

        for (_, x, y) in face_offsets
            .into_iter()
            .filter(|(face, _, _)| faces.contains(*face))
        {
            let current = self.iter.current().ok_or(DecodingError::NoMoreSurfaces)?;
            if current.size() != face_size {
                return Err(DecodingError::UnexpectedSurfaceSize);
            }

            let offset_x = x * face_size.width;
            let offset_y = y * face_size.height;
            let offset = offset_y as usize * row_pitch + offset_x as usize * bytes_per_pixel;

            self.read_surface_rect(&mut image_bytes[offset..], row_pitch, rect, color)?;
            self.skip_mipmaps()?;
        }

        Ok(())
    }

    /// Moves to the reader back the previous surface, allowing it to be read
    /// again.
    ///
    /// If there is no previous surface, this function will do nothing.
    ///
    /// Note that this operation does **not** bring the decoder into a known
    /// (working) state after an error occurred.
    pub fn rewind_to_previous_surface(&mut self) -> Result<(), DecodingError>
    where
        R: Seek,
    {
        let current_bytes = self.iter.elapsed_bytes();
        self.iter.rewind();
        let previous_bytes = self.iter.elapsed_bytes();
        let seek = previous_bytes as i64 - current_bytes as i64;
        self.reader.seek(std::io::SeekFrom::Current(seek))?;

        Ok(())
    }

    /// Moves to the reader to the start of the data section of the DDS file,
    /// making it possible to read the DDS file again.
    ///
    /// Note that this operation does **not** bring the decoder into a known
    /// (working) state after an error occurred.
    pub fn rewind_to_start(&mut self) -> Result<(), DecodingError>
    where
        R: Seek,
    {
        let elapsed_bytes = self.iter.elapsed_bytes();
        let seek = -(elapsed_bytes as i64);
        self.reader.seek(std::io::SeekFrom::Current(seek))?;

        self.iter = SurfaceIterator::new(self.layout);

        Ok(())
    }

    /// Returns information about the surface about to be read.
    ///
    /// The returned value is not valid after calling `next_surface`.
    ///
    /// If there are no more surfaces, `None` is returned.
    ///
    /// Use [`Self::is_done`] instead of checking for `None` to determine if the
    /// encoder is done reading.
    pub fn surface_info(&self) -> Option<SurfaceInfo<'_>> {
        self.iter.current()
    }
    /// Returns whether all surfaces that have been read.
    ///
    /// If `true` is returned, then attempting to read more surfaces will
    /// always result in an error.
    ///
    /// See [`Self::read_surface`] for more information about reading surfaces,
    /// and [`Self::surface_info`] for more information about the next
    /// surface.
    pub fn is_done(&self) -> bool {
        self.iter.current().is_none()
    }
    /// Returns the underlying reader.
    pub fn into_reader(self) -> R {
        self.reader
    }
}
