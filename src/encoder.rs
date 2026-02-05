use std::io::Write;

use crate::{
    encode,
    header::Header,
    iter::{SurfaceInfo, SurfaceIterator},
    resize::{AlignedBuffer, Aligner, ResizeState},
    ColorFormat, DataLayout, EncodeOptions, EncodingError, Format, ImageView, Progress,
    ProgressRange, Size,
};

/// An encoder for DDS files.
///
/// See [crate-level documentation](crate) for usage examples.
pub struct Encoder<W> {
    // internal state
    writer: W,
    format: Format,
    layout: DataLayout,
    iter: SurfaceIterator,

    /// The encoding options used to encode surfaces.
    ///
    /// Default: [`EncodeOptions::default()`]
    pub options: EncodeOptions,
    /// Options regarding automatic mipmap generation.
    ///
    /// Set `self.mipmaps.generate = false` to disable automatic mipmap
    /// generation.
    ///
    /// Default: [`MipmapOptions::default()`]
    pub mipmaps: MipmapOptions,

    // internal cache for resizing
    mipmap_cache: MipmapCache,
}
impl<W> Encoder<W> {
    /// Creates a new encoder and immediately writes the header to the writer.
    ///
    /// The format and header and given separately, because certain older and
    /// non-standard formats cannot be detected from the header alone. I.e.
    /// `BC3_UNORM` and `BC3_UNORM_NORMAL` have the same header. If you are
    /// only using commonly-used formats, you can use [`Format::from_header`]
    /// to detect the format.
    ///
    /// If the given format does not support encoding,
    /// [`EncodingError::UnsupportedFormat`] is returned.
    pub fn new(mut writer: W, format: Format, header: &Header) -> Result<Self, EncodingError>
    where
        W: Write,
    {
        if format.encoding_support().is_none() {
            return Err(EncodingError::UnsupportedFormat(format));
        }

        let layout = DataLayout::from_header_with(header, format.into())?;

        header.write(&mut writer)?;

        Ok(Self {
            writer,
            format,
            layout,
            iter: SurfaceIterator::new(layout),
            options: EncodeOptions::default(),
            mipmaps: MipmapOptions::default(),
            mipmap_cache: MipmapCache::new(),
        })
    }

    /// Creates a new encoder for a single image with the given size and format.
    ///
    /// If `mipmaps` is `true`, a full mipmap chain will be declared in the
    /// header.
    ///
    /// The header is created using [`Header::new_image`] and immediately
    /// written to the writer. For more control over the header, use
    /// [`Encoder::new`].
    pub fn new_image(
        writer: W,
        size: Size,
        format: Format,
        mipmaps: bool,
    ) -> Result<Self, EncodingError>
    where
        W: Write,
    {
        let mut header = Header::new_image(size.width, size.height, format);
        if mipmaps {
            header = header.with_mipmaps();
        }

        Self::new(writer, format, &header)
    }

    /// The format of the pixel data.
    pub fn format(&self) -> Format {
        self.format
    }
    /// The layout of the data section.
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

    /// Writes the next surface.
    ///
    /// The next surface is determined by the data layout of the DDS file. For
    /// volume textures, this function will write the next depth slice.
    ///
    /// See [`Self::surface_info`] for more information about the surface.
    ///
    /// If [`self.mipmaps.generate`](MipmapOptions::generate) is set to `true`
    /// and the header specifies mipmaps, this function will automatically
    /// generate mipmaps for the surface. The only exception is volume depth
    /// slices, which are not supported for mipmap generation.
    pub fn write_surface(&mut self, image: ImageView) -> Result<(), EncodingError>
    where
        W: Write,
    {
        let mut progress = Progress::none();
        self.write_surface_impl(image, &mut progress)
    }

    /// Writes the next surface with progress.
    ///
    /// Behaves like [`Self::write_surface`], but also reports progress.
    /// Single-threaded progress reporter may cause negatively impact performance.
    pub fn write_surface_with_progress(
        &mut self,
        image: ImageView,
        progress: &mut Progress,
    ) -> Result<(), EncodingError>
    where
        W: Write,
    {
        self.write_surface_impl(image, progress)
    }

    fn write_surface_impl(
        &mut self,
        image: ImageView,
        progress: &mut Progress,
    ) -> Result<(), EncodingError>
    where
        W: Write,
    {
        // Get information about the current surface.
        let current = self.iter.current().ok_or(EncodingError::TooManySurfaces)?;
        if current.size() != image.size() {
            return Err(EncodingError::UnexpectedSurfaceSize);
        }

        // Figure out how many mipmaps we'll generate ahead of time.
        let mipmaps_to_generate = if self.mipmaps.generate && !self.layout.is_volume() {
            let mipmap_levels = self.layout.mipmaps();
            mipmap_levels.saturating_sub(current.mipmap_level + 1)
        } else {
            0
        };

        let get_level_progress_range = move |level: u8| -> ProgressRange {
            if mipmaps_to_generate == 0 {
                ProgressRange::FULL
            } else {
                // This is how much progress the main surface accounts for.
                // I determined this value experimentally, so that 50% progress
                // roughly aligns with 50% execution time.
                let main_surface = 0.6_f32; // 60%
                let start = 1.0 - (1.0 - main_surface).powi(level as i32);
                let end = 1.0 - (1.0 - main_surface).powi(level as i32 + 1);
                ProgressRange::from_to(start, end)
            }
        };

        // write the main surface
        encode(
            &mut self.writer,
            image,
            self.format,
            Some(&mut progress.sub_range(get_level_progress_range(0))),
            &self.options,
        )?;
        self.iter.advance();

        // write all mipmaps, if any
        if mipmaps_to_generate > 0 {
            progress.check_cancelled()?;

            // gather the sizes of all mipmaps to generate
            let mut mipmap_sizes: Vec<Size> = Vec::with_capacity(16);
            while let Some(mipmap) = self.iter.current() {
                if !mipmap.is_mipmap() {
                    break;
                }
                mipmap_sizes.push(mipmap.size());
                self.iter.advance();
            }

            let mut level = 0;
            self.mipmap_cache
                .generate(image, &mipmap_sizes, self.mipmaps, |mipmap| {
                    level += 1;
                    encode(
                        &mut self.writer,
                        mipmap,
                        self.format,
                        Some(&mut progress.sub_range(get_level_progress_range(level))),
                        &self.options,
                    )
                })?;
        }

        // report 100% progress
        progress.checked_report(1.0)?;

        Ok(())
    }

    /// Returns information about the surface about to be written.
    ///
    /// The returned value only valid until the next call to
    /// [`Self::write_surface`] or [`Self::write_surface_with_progress`].
    ///
    /// If there are no more surfaces, `None` is returned.
    ///
    /// Use [`Self::is_done`] instead of checking for `None` to determine if the
    /// encoder is done writing.
    pub fn surface_info(&self) -> Option<SurfaceInfo<'_>> {
        self.iter.current()
    }
    /// Returns whether all surfaces that have been written.
    ///
    /// If `true` is returned, then attempting to write more surfaces will
    /// always result in an error.
    ///
    /// See [`Self::write_surface`] for more information about writing surfaces
    /// and [`Self::surface_info`] for more information about the current
    /// surface.
    pub fn is_done(&self) -> bool {
        self.iter.current().is_none()
    }
    /// Checks that the encoder is done writing and flushes the writer.
    ///
    /// Instead of dropping the encoder, this method should be used to ensure
    /// that (1) the DDS file is valid and (2) all data is written to the
    /// writer.
    ///
    /// If you need the writer after this call, use [`Self::into_writer`].
    ///
    /// This will return [`EncodingError::MissingSurfaces`] if some surfaces are
    /// yet to be written. See [`Self::is_done`].
    pub fn finish(mut self) -> Result<(), EncodingError>
    where
        W: Write,
    {
        if !self.is_done() {
            return Err(EncodingError::MissingSurfaces);
        }
        self.writer.flush()?;
        Ok(())
    }
    /// Returns the underlying writer.
    ///
    /// Using this method may result in the creation of invalid DDS files if
    /// the encoder is not done yet. See [`Self::is_done`].
    ///
    /// Preferably, use [`Self::finish`] to ensure that the DDS file is valid.
    pub fn into_writer(self) -> W {
        self.writer
    }
}

/// The filter to use when resizing images for mipmap generation.
///
/// ## See also
///
/// - [`MipmapOptions::resize_filter`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ResizeFilter {
    /// Nearest neighbor interpolation (=point filtering).
    Nearest,
    /// Box (also called area or binning).
    ///
    /// This is the default filter, because it produces mipmaps that are
    /// generally free of artifacts and sharp (without being over sharpened).
    #[default]
    Box,
    /// Triangle filtering (=linear interpolation).
    Triangle,
    /// Mitchell interpolation.
    Mitchell,
    /// Lanczos interpolation with a radius of 3.
    Lanczos3,
}

/// Options for automatic mipmap generation in [`Encoder`].
#[derive(Debug, Clone, Copy)]
pub struct MipmapOptions {
    /// Whether to generate mipmaps for the texture.
    ///
    /// Since the encoder knows exactly how many mipmaps are needed, it will
    /// generate all mipmaps until the next level 0 object or EOF.
    ///
    /// Note: Generating mipmaps for volume depth slices is not supported. This
    /// will **NOT** result in an error. Instead, the encoder will silently
    /// ignore the option and assume mipmap generation is disabled.
    ///
    /// Default: `true`
    pub generate: bool,
    /// Whether the alpha channel (if any) is straight alpha transparency.
    ///
    /// This is important when generating mipmaps. Resizing RGBA with straight
    /// alpha requires that the alpha channel is premultiplied into the color
    /// channels before resizing and then un-premultiplied after resizing. This
    /// is necessary to avoid color bleeding.
    ///
    /// If the alpha channel is premultiplied alpha transparency or custom
    /// (e.g. like in channel-packed textures), this option should be set to
    /// `false`.
    ///
    /// If this option is set to `false`, all channels will be resized
    /// independently of each other. If set to `true`, the alpha channel will
    /// be interpreted as straight alpha transparency and handled accordingly.
    ///
    /// Default: `true`
    pub resize_straight_alpha: bool,
    /// The filter to use when resizing the texture to generate mipmaps.
    ///
    /// Default: [`ResizeFilter::Box`]
    pub resize_filter: ResizeFilter,
}
impl Default for MipmapOptions {
    fn default() -> Self {
        Self {
            generate: true,
            resize_straight_alpha: true,
            resize_filter: ResizeFilter::Box,
        }
    }
}

struct MipmapCache {
    aligner: Aligner,
    resizer: ResizeState,
}
impl MipmapCache {
    fn new() -> Self {
        Self {
            aligner: Aligner::new(),
            resizer: ResizeState::new(),
        }
    }

    fn generate(
        &mut self,
        image: ImageView,
        sizes: &[Size],
        options: MipmapOptions,
        f: impl FnMut(ImageView) -> Result<(), EncodingError>,
    ) -> Result<(), EncodingError> {
        #[cfg(debug_assertions)]
        {
            // check that sizes are decreasing
            let mut last_size = image.size();
            for &size in sizes {
                debug_assert!(
                    size.width <= last_size.width && size.height <= last_size.height,
                    "Mipmap sizes must be in decreasing order"
                );
                last_size = size;
            }
        }

        // decide which path to take

        if options.resize_filter == ResizeFilter::Nearest {
            // Fast path for custom Nearest implementation.
            return self.generate_nearest(image, sizes, f);
        }

        if sizes
            .iter()
            .chain(&[image.size])
            .all(|s| s.width.is_power_of_two() && s.height.is_power_of_two())
        {
            // Mipmap generation behaves more nicely when all sizes are powers of two.

            if options.resize_filter == ResizeFilter::Box {
                // Box filter behaves well enough that we can generate each
                // mipmap from the previous one.
                return self.generate_from_previous(image, sizes, options, f);
            }

            return self.generate_from_previous_two(image, sizes, options, f);
        }

        self.generate_from_source(image, sizes, options, true, f)
    }

    /// Generates each mipmap directly from the source image.
    fn generate_nearest(
        &mut self,
        image: ImageView,
        sizes: &[Size],
        mut f: impl FnMut(ImageView) -> Result<(), EncodingError>,
    ) -> Result<(), EncodingError> {
        // otherwise, generate mipmaps sequentially
        for &mipmap_size in sizes {
            let mipmap = crate::resize::resize_nearest(image, mipmap_size);
            f(mipmap.as_view().as_image_view())?;
        }

        Ok(())
    }

    /// Generates each mipmap directly from the source image.
    fn generate_from_source(
        &mut self,
        image: ImageView,
        sizes: &[Size],
        options: MipmapOptions,
        parallel: bool,
        mut f: impl FnMut(ImageView) -> Result<(), EncodingError>,
    ) -> Result<(), EncodingError> {
        let straight_alpha = options.resize_straight_alpha;
        let filter = options.resize_filter;

        let src = self.aligner.align(image);

        // when rayon is available, generate mipmaps in parallel
        #[cfg(feature = "rayon")]
        if parallel {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

            let precomputed_mipmaps: Vec<AlignedBuffer> = sizes
                .par_iter()
                .map(|&mipmap_size| crate::resize::resize(src, mipmap_size, straight_alpha, filter))
                .collect();

            for mipmap in precomputed_mipmaps {
                f(mipmap.as_view().as_image_view())?;
            }
            return Ok(());
        }

        // otherwise, generate mipmaps sequentially
        for &mipmap_size in sizes {
            let mipmap = self
                .resizer
                .resize(src, mipmap_size, straight_alpha, filter);

            f(mipmap.as_image_view())?;
        }

        Ok(())
    }

    /// Generates each mipmap from the previous mipmap.
    fn generate_from_previous(
        &mut self,
        image: ImageView,
        sizes: &[Size],
        options: MipmapOptions,
        mut f: impl FnMut(ImageView) -> Result<(), EncodingError>,
    ) -> Result<(), EncodingError> {
        let straight_alpha = options.resize_straight_alpha;
        let filter = options.resize_filter;

        let src = self.aligner.align(image);

        let first_mipmap = crate::resize::resize(src, sizes[0], straight_alpha, filter);
        f(first_mipmap.as_view().as_image_view())?;

        let mut prev_mipmap = first_mipmap;

        for &mipmap_size in &sizes[1..] {
            let next_mipmap =
                crate::resize::resize(prev_mipmap.as_view(), mipmap_size, straight_alpha, filter);

            f(next_mipmap.as_view().as_image_view())?;

            prev_mipmap = next_mipmap;
        }

        Ok(())
    }

    /// Generates each mipmap from the previous previous mipmap.
    fn generate_from_previous_two(
        &mut self,
        image: ImageView,
        sizes: &[Size],
        options: MipmapOptions,
        mut f: impl FnMut(ImageView) -> Result<(), EncodingError>,
    ) -> Result<(), EncodingError> {
        let straight_alpha = options.resize_straight_alpha;
        let filter = options.resize_filter;

        let src = self.aligner.align(image);

        let first_mipmap = crate::resize::resize(src, sizes[0], straight_alpha, filter);
        f(first_mipmap.as_view().as_image_view())?;

        if sizes.len() == 1 {
            return Ok(());
        }

        let second_mipmap = crate::resize::resize(src, sizes[1], straight_alpha, filter);
        f(second_mipmap.as_view().as_image_view())?;

        let mut prev_prev_mipmap = first_mipmap;
        let mut prev_mipmap = second_mipmap;

        for &mipmap_size in &sizes[2..] {
            let next_mipmap = crate::resize::resize(
                prev_prev_mipmap.as_view(),
                mipmap_size,
                straight_alpha,
                filter,
            );

            f(next_mipmap.as_view().as_image_view())?;

            prev_prev_mipmap = prev_mipmap;
            prev_mipmap = next_mipmap;
        }

        Ok(())
    }
}
