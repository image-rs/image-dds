use std::{io::Write, ops::Range};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{encode, ColorFormat, Dithering, EncodeError, EncodeOptions, Format, PixelInfo, Size};

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

pub struct SurfaceFragment<'a> {
    pub data: &'a [u8],
    pub size: Size,
    pub color: ColorFormat,
}

pub struct SplitSurface<'a> {
    fragments: Box<[SurfaceFragment<'a>]>,
    format: Format,
    options: EncodeOptions,
}

impl<'a> SplitSurface<'a> {
    /// Creates a new `SplitSurface` with exactly one fragment that covers the whole surface.
    pub fn from_single_fragment(
        data: &'a [u8],
        size: Size,
        color: ColorFormat,
        format: Format,
        options: &EncodeOptions,
    ) -> Self {
        // TODO: check that data is valid
        Self {
            fragments: vec![SurfaceFragment { data, size, color }].into_boxed_slice(),
            format,
            options: options.clone(),
        }
    }

    pub fn new(
        data: &'a [u8],
        size: Size,
        color: ColorFormat,
        format: Format,
        options: &EncodeOptions,
    ) -> Self {
        // TODO: think over this panic
        assert_eq!(
            data.len() as u64,
            size.pixels().saturating_mul(color.bytes_per_pixel() as u64)
        );

        if let Some(ranges) = split_surface_into_lines(size, format, options) {
            let stride = size.width as usize * color.bytes_per_pixel() as usize;

            let fragments = ranges
                .into_iter()
                .map(move |range| {
                    let start = range.start as usize * stride;
                    let end = range.end as usize * stride;
                    let height = range.end - range.start;
                    SurfaceFragment {
                        data: &data[start..end],
                        size: Size::new(size.width, height),
                        color,
                    }
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();

            Self {
                fragments,
                format,
                options: options.clone(),
            }
        } else {
            Self::from_single_fragment(data, size, color, format, options)
        }
    }

    /// If this split surface consists of only a single fragment, returns that
    /// fragment.
    pub fn single(&self) -> Option<&SurfaceFragment<'a>> {
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
    pub fn fragments(&self) -> &[SurfaceFragment<'a>] {
        &self.fragments
    }

    /// Encodes a single fragment to the writer.
    pub fn encode_fragment(
        &self,
        writer: &mut dyn Write,
        fragment: &SurfaceFragment<'a>,
    ) -> Result<(), EncodeError> {
        encode(
            writer,
            self.format,
            fragment.size,
            fragment.color,
            fragment.data,
            &self.options,
        )
    }

    /// Encodes all fragments to the writer.
    ///
    /// This will encode the fragments in parallel (if the `rayon` feature is enabled).
    pub fn encode(&self, writer: &mut dyn Write) -> Result<(), EncodeError> {
        // optimization for single fragment
        if let Some(single) = self.single() {
            return self.encode_fragment(writer, single);
        }

        let pixel_info = PixelInfo::from(self.format);

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
}
