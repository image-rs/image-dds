use std::num::NonZeroU32;

use crate::{util, Dithering, EncodeOptions, Format, ImageView, Size};

/// An [`ImageView`] that has been split into horizontal fragments.
///
/// ## Purpose
///
/// The main use case of this type is to allow end users to implement custom
/// concurrent/parallel encoding schemes. The parallel encoding implemented by
/// [`encode`](crate::encode()) may not fit every use case, so this type can be
/// used to split a single [`ImageView`] into multiple fragments that can be
/// encoded independently.
///
/// Note that split views depend on the [`Format`] and [`EncodeOptions`] that
/// are used to create them. The encoding fragments with different formats
/// or options may yield unexpected results.
pub struct SplitView<'a> {
    image: ImageView<'a>,
    len: u32,
    group_height: Option<NonZeroU32>,
}

impl<'a> SplitView<'a> {
    /// Creates a new `SplitView` with exactly one fragment that covers the whole surface.
    pub fn new_single(image: ImageView<'a>) -> Self {
        Self {
            image,
            len: 1,
            group_height: None,
        }
    }

    /// Creates a new `SplitView` from the given image, format, and options.
    pub fn new(image: ImageView<'a>, format: Format, options: &EncodeOptions) -> Self {
        if let Some(group_height) = get_group_height(image.size(), format, options) {
            let len = util::div_ceil(image.height(), group_height.get());

            Self {
                image,
                len,
                group_height: Some(group_height),
            }
        } else {
            Self::new_single(image)
        }
    }

    /// Returns the number of fragments that make up this split surface.
    ///
    /// This is guaranteed to be at least 1.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn get(&self, index: u32) -> Option<ImageView<'a>> {
        if index >= self.len {
            return None;
        }

        if let Some(group_height) = self.group_height {
            let start_y = index * group_height.get();
            let end_y = start_y
                .saturating_add(group_height.get())
                .min(self.image.height());
            debug_assert!(start_y < self.image.height());

            let fragment_height = end_y - start_y;
            let start = start_y as usize * self.image.row_pitch();
            let end = end_y as usize * self.image.row_pitch();

            Some(
                ImageView::new(
                    &self.image.data[start..end],
                    Size::new(self.image.width(), fragment_height),
                    self.image.color,
                )
                .expect("invalid split"),
            )
        } else {
            Some(self.image)
        }
    }

    /// If this split surface consists of only a single fragment, returns that
    /// fragment.
    pub fn single(&self) -> Option<ImageView<'a>> {
        if self.len() == 1 {
            Some(self.image)
        } else {
            None
        }
    }
}

fn get_group_height(size: Size, format: Format, options: &EncodeOptions) -> Option<NonZeroU32> {
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

    // it's important that the group height is divisible by the split height,
    let split_height_64 = split_height.get() as u64;
    let group_height_maybe_zero =
        u32::try_from((group_pixels / size.width as u64) / split_height_64 * split_height_64)
            .ok()?;

    let group_height = NonZeroU32::new(group_height_maybe_zero).unwrap_or(split_height.into());

    Some(group_height)
}
