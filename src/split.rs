use std::num::NonZeroU32;

use crate::{util, Dithering, EncodeOptions, Format, ImageView, Offset, Size};

/// An [`ImageView`] that has been split into horizontal fragments.
///
/// ## Purpose
///
/// The primary purpose of this type is to enable the creation of custom
/// concurrent or parallel encoding schemes. The parallel encoding implemented by
/// [`encode`](crate::encode()) may not fit every use case, so this type can be
/// used to encode fragments of an image independently of each other.
///
/// Note: Split views are specific to the [`Format`] and [`EncodeOptions`] used
/// to create them. Encoding fragments with different formats or options may
/// yield unexpected results.
pub struct SplitView<'a> {
    image: ImageView<'a>,
    len: u32,
    fragment_height: Option<NonZeroU32>,
}

impl<'a> SplitView<'a> {
    /// Creates a new `SplitView` with exactly one fragment that covers the whole surface.
    pub fn new_single(image: ImageView<'a>) -> Self {
        Self {
            image,
            len: 1,
            fragment_height: None,
        }
    }

    /// Creates a new `SplitView` from the given image, format, and options.
    pub fn new(image: ImageView<'a>, format: Format, options: &EncodeOptions) -> Self {
        if let Some(fragment_height) = get_fragment_height(image.size(), format, options) {
            let len = util::div_ceil(image.height(), fragment_height.get());

            Self {
                image,
                len,
                fragment_height: Some(fragment_height),
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

    /// Returns the fragment at the given index, or `None` if the index is out
    /// of bounds.
    pub fn get(&self, index: u32) -> Option<ImageView<'a>> {
        if index >= self.len {
            return None;
        }

        if let Some(full_fragment_height) = self.fragment_height {
            let start_y = index * full_fragment_height.get();
            let end_y = start_y
                .saturating_add(full_fragment_height.get())
                .min(self.image.height());
            debug_assert!(start_y < self.image.height());

            // calculate the actual height of this fragment
            // this can be smaller than fragment_height for the last fragment
            let fragment_height = end_y - start_y;

            Some(self.image.cropped(
                Offset::new(0, start_y),
                Size::new(self.image.width(), fragment_height),
            ))
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

fn get_fragment_height(size: Size, format: Format, options: &EncodeOptions) -> Option<NonZeroU32> {
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

    let fragment_pixels = support.fragment_size.get_preferred(options.quality).max(1);
    if fragment_pixels >= size.pixels() {
        // the image is small enough that it's not worth splitting
        return None;
    }

    // it's important that the fragment height is divisible by the split height,
    let split_height_64 = split_height.get() as u64;
    let fragment_height_or_zero =
        u32::try_from((fragment_pixels / size.width as u64) / split_height_64 * split_height_64)
            .ok()?;

    let fragment_height = NonZeroU32::new(fragment_height_or_zero).unwrap_or(split_height.into());

    Some(fragment_height)
}
