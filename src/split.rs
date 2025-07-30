use std::ops::Range;

use crate::{util, Dithering, EncodeOptions, Format, ImageView, Offset, Size};

/// This implements the main logic for splitting a surface into lines.
fn split_surface_into_lines(
    size: Size,
    format: Format,
    options: &EncodeOptions,
) -> Option<impl Iterator<Item = Range<u32>>> {
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

    let groups = util::div_ceil(size.height, group_height);
    debug_assert!(groups >= 2);

    Some((0..groups).map(move |i| {
        let start = i * group_height;
        let end = ((i + 1) * group_height).min(size.height);
        start..end
    }))
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
            Self {
                fragments: ranges
                    .map(|range| {
                        image.cropped(
                            Offset::new(0, range.start),
                            Size::new(image.width(), range.end - range.start),
                        )
                    })
                    .collect(),
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
}
