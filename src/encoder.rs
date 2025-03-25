use std::{io::Write, ops::Range};

use crate::{
    encode,
    header::Header,
    iter::{SurfaceInfo, SurfaceIterator},
    ColorFormat, DataLayout, Dithering, EncodeError, EncodeOptions, Format, Size,
};

fn split_surface_into_lines(
    size: Size,
    format: Format,
    options: &EncodeOptions,
) -> Vec<Range<u32>> {
    if size.is_empty() {
        return Vec::new();
    }

    if let Some(support) = format.encoding_support() {
        if let Some(split_height) = support.split_height {
            // dithering destroys our ability to split the surface into lines,
            // because it would create visible seams
            if support.local_dithering
                || options.dithering.intersect(support.dithering) == Dithering::None
            {
                let mut lines = Vec::new();
                let mut y: u32 = 0;
                while y < size.height {
                    let end = (y + split_height.get() as u32).min(size.height);
                    lines.push(y..end);
                    y = end;
                }
                return lines;
            }
        }
    }

    // can't split the surface
    let range = 0..size.height;
    vec![range]
}

pub struct Encoder<W> {
    writer: W,
    format: Format,
    layout: DataLayout,
    iter: SurfaceIterator,
    pub options: EncodeOptions,
}
impl<W> Encoder<W> {
    pub fn new(mut writer: W, format: Format, header: &Header) -> Result<Self, EncodeError>
    where
        W: Write,
    {
        let layout = DataLayout::from_header_with(header, format.into())?;

        header.write(&mut writer)?;

        Ok(Self {
            writer,
            format,
            layout,
            iter: SurfaceIterator::new(layout),
            options: EncodeOptions::default(),
        })
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
        self.format().color()
    }

    pub fn into_writer(self) -> W {
        self.writer
    }

    /// Returns information about the surface about to be written.
    ///
    /// The returned value is not valid after calling `write_surface`.
    ///
    /// If there are no more surfaces, `None` is returned.
    pub fn surface_info(&self) -> Option<SurfaceInfo<'_>> {
        self.iter.current()
    }

    /// Writes the next surface.
    ///
    /// The next surface is determined by the data layout of the DDS file. For
    /// volume textures, this function will write the next depth slice.
    ///
    /// See [`Self::surface_info`] for more information about the surface.
    pub fn write_surface(
        &mut self,
        buffer: &mut [u8],
        color: ColorFormat,
    ) -> Result<(), EncodeError>
    where
        W: Write,
    {
        let current = self.iter.current().ok_or(EncodeError::TooManySurfaces)?;

        encode(
            &mut self.writer,
            self.format,
            current.size(),
            color,
            buffer,
            &self.options,
        )?;

        self.iter.advance();
        Ok(())
    }

    /// Writes the next surface.
    ///
    /// The next surface is determined by the data layout of the DDS file. For
    /// volume textures, this function will write the next depth slice.
    ///
    /// See [`Self::surface_info`] for more information about the surface.
    pub fn write_surface_with(
        &mut self,
        buffer: &mut [u8],
        color: ColorFormat,
        progress: impl FnMut(f32),
        options: WriteOptions,
    ) -> Result<(), EncodeError>
    where
        W: Write,
    {
        let current = self.iter.current().ok_or(EncodeError::TooManySurfaces)?;

        encode(
            &mut self.writer,
            self.format,
            current.size(),
            color,
            buffer,
            &self.options,
        )?;

        self.iter.advance();
        Ok(())
    }
}

pub struct WriteOptions {
    /// Whether to generate mipmaps for the texture.
    ///
    /// Since the encoder knows exactly how many mipmaps are needed, it will
    /// generate all mipmaps until the next level 0 object or EOF.
    pub generate_mipmaps: bool,
    /// Whether the alpha channel (if any) is straight alpha.
    ///
    /// This is important when generating mipmaps. Resizing RGBA with straight
    /// alpha requires that the alpha channel is premultiplied into the color
    /// channels before resizing and then unpremultiplied after resizing. This
    /// is necessary to avoid color bleeding.
    ///
    /// If the alpha channel is premultiplied alpha or custom (e.g. like in
    /// channel-packed textures), this option should be set to `false`.
    ///
    /// If this option is set to `false`, all channels will be resized
    /// independently of each other.
    pub resize_straight_alpha: bool,
}
