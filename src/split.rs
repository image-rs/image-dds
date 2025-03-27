use std::{io::Write, ops::Range};

use crate::{encode, Dithering, EncodeError, EncodeOptions, Format, ImageView, Size};

/// This implements the main logic for splitting a surface into lines.
fn split_surface_into_lines(
    size: Size,
    format: Format,
    options: &EncodeOptions,
) -> Option<Vec<Range<u32>>> {
    if size.is_empty() {
        return None;
    }

    let support = format.encoding_support()?;
    let split_height = support.split_height()?;

    // dithering destroys our ability to split the surface into lines,
    // because it would create visible seams
    if !support.local_dithering()
        && options.dithering.intersect(support.dithering()) != Dithering::None
    {
        return None;
    }

    let group_pixels = support
        .group_size()
        .get_group_pixels(options.quality)
        .max(1);
    if group_pixels >= size.pixels() {
        // the image is small enough that it's not worth splitting
        return None;
    }

    let group_height = u64::clamp(
        (group_pixels / size.width as u64) / split_height.get() as u64 * split_height.get() as u64,
        split_height.get() as u64,
        u32::MAX as u64,
    ) as u32;

    let mut lines = Vec::new();
    let mut y: u32 = 0;
    while y < size.height {
        let end = y.saturating_add(group_height).min(size.height);
        lines.push(y..end);
        y = end;
    }

    Some(lines)
}

pub struct SplitSurface<'a> {
    fragments: Box<[ImageView<'a>]>,
    format: Format,
    options: EncodeOptions,
}

impl<'a> SplitSurface<'a> {
    /// Creates a new `SplitSurface` with exactly one fragment that covers the whole surface.
    pub fn from_single_fragment(
        image: ImageView<'a>,
        format: Format,
        options: &EncodeOptions,
    ) -> Self {
        Self {
            fragments: vec![image].into_boxed_slice(),
            format,
            options: options.clone(),
        }
    }

    pub fn new(image: ImageView<'a>, format: Format, options: &EncodeOptions) -> Self {
        if let Some(ranges) = split_surface_into_lines(image.size(), format, options) {
            let row_pitch = image.row_pitch();

            let fragments = ranges
                .into_iter()
                .map(move |range| {
                    let start = range.start as usize * row_pitch;
                    let end = range.end as usize * row_pitch;
                    let height = range.end - range.start;
                    ImageView::new(
                        &image.data[start..end],
                        Size::new(image.width(), height),
                        image.color,
                    )
                    .expect("invalid split")
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();

            Self {
                fragments,
                format,
                options: options.clone(),
            }
        } else {
            Self::from_single_fragment(image, format, options)
        }
    }

    /// If this split surface consists of only a single fragment, returns that
    /// fragment.
    pub fn single(&self) -> Option<&ImageView<'a>> {
        if self.fragments.len() == 1 {
            self.fragments.first()
        } else {
            None
        }
    }

    pub fn format(&self) -> Format {
        self.format
    }
    pub fn options(&self) -> &EncodeOptions {
        &self.options
    }
    /// The list of fragments that make up the surface.
    ///
    /// The list is guaranteed to be non-empty.
    pub fn fragments(&self) -> &[ImageView<'a>] {
        &self.fragments
    }

    /// Encodes a single fragment to the writer.
    pub fn encode_fragment(
        &self,
        writer: &mut dyn Write,
        fragment: &ImageView<'a>,
    ) -> Result<(), EncodeError> {
        encode(writer, *fragment, self.format, &self.options)
    }

    /// Encodes all fragments to the writer.
    ///
    /// This will encode the fragments in parallel (if the `rayon` feature is enabled).
    pub fn encode(&self, writer: &mut dyn Write) -> Result<(), EncodeError> {
        self.encode_impl(writer)
    }
    #[cfg(feature = "rayon")]
    fn encode_impl(&self, writer: &mut dyn Write) -> Result<(), EncodeError> {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        // optimization for single fragment
        if let Some(single) = self.single() {
            return self.encode_fragment(writer, single);
        }

        let pixel_info = crate::PixelInfo::from(self.format);

        let result: Result<Vec<Vec<u8>>, EncodeError> = self
            .fragments
            .par_iter()
            .map(|fragment| -> Result<Vec<u8>, EncodeError> {
                let bytes: usize = pixel_info
                    .surface_bytes(fragment.size)
                    .unwrap_or(u64::MAX)
                    .try_into()
                    .expect("too many bytes");
                let mut buffer: Vec<u8> = Vec::with_capacity(bytes);

                self.encode_fragment(&mut buffer, fragment)?;

                debug_assert_eq!(buffer.len(), bytes);
                Ok(buffer)
            })
            .collect();

        let encoded_fragments = result?;
        for fragment in encoded_fragments {
            writer.write_all(&fragment)?;
        }

        Ok(())
    }
    #[cfg(not(feature = "rayon"))]
    fn encode_impl(&self, writer: &mut dyn Write) -> Result<(), EncodeError> {
        for fragment in &self.fragments {
            self.encode_fragment(writer, fragment)?;
        }
        Ok(())
    }
}

/// This function has the same API as [`encode()`], but it will automatically
/// make use of the `rayon` feature to parallelize the encoding of the surface.
///
/// If the `rayon` feature is not enabled, this function will behave exactly
/// like [`encode()`].
pub fn split_encode(
    writer: &mut dyn Write,
    image: ImageView,
    format: Format,
    options: &EncodeOptions,
) -> Result<(), EncodeError> {
    // Only actually split the surface when rayon is available.
    // If we don't get to encode in parallel, splitting is pure overhead.
    #[cfg(feature = "rayon")]
    {
        let split = SplitSurface::new(image, format, options);
        split.encode(writer)
    }
    #[cfg(not(feature = "rayon"))]
    {
        encode(writer, image, format, options)
    }
}
