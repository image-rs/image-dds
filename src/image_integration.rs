use std::ops::Deref;

use image::{ColorType, DynamicImage, ImageBuffer, Pixel};

use crate::{cast, ColorFormat, ImageView, ImageViewMut, Size};

/// Create a new `ImageView` from a raw byte slice from an `image` buffer.
///
/// `image`'s buffers aren't guaranteed to have the exact length needed (they
/// may be longer), so this does some slicing to ensure the length is correct.
fn to_view(bytes: &[u8], size: Size, color: ColorFormat) -> ImageView<'_> {
    let len = color.buffer_size(size).expect("size too big to allocate");
    ImageView::new(&bytes[..len], size, color).unwrap()
}
fn to_view_mut(bytes: &mut [u8], size: Size, color: ColorFormat) -> ImageViewMut<'_> {
    let len = color.buffer_size(size).expect("size too big to allocate");
    ImageViewMut::new(&mut bytes[..len], size, color).unwrap()
}

/// A trait for pixel types that correspond to [`ColorFormat`]s.
pub trait SupportedPixel: Pixel {
    /// The corresponding `ColorFormat` for this pixel type.
    const COLOR_FORMAT: ColorFormat;

    /// Converts a slice of this pixel's subpixel type to a byte slice.
    fn subpixels_as_bytes(subpixels: &[Self::Subpixel]) -> &[u8];
    /// Converts a mutable slice of this pixel's subpixel type to a mutable byte slice.
    fn subpixels_as_bytes_mut(subpixels: &mut [Self::Subpixel]) -> &mut [u8];
}

macro_rules! impl_supported_pixel {
    ($color:expr, $pixel:ty) => {
        impl SupportedPixel for $pixel {
            const COLOR_FORMAT: ColorFormat = $color;

            fn subpixels_as_bytes(subpixels: &[Self::Subpixel]) -> &[u8] {
                cast::as_bytes(subpixels)
            }
            fn subpixels_as_bytes_mut(subpixels: &mut [Self::Subpixel]) -> &mut [u8] {
                cast::as_bytes_mut(subpixels)
            }
        }
    };
}
impl_supported_pixel!(ColorFormat::GRAYSCALE_U8, image::Luma<u8>);
impl_supported_pixel!(ColorFormat::GRAYSCALE_U16, image::Luma<u16>);
impl_supported_pixel!(ColorFormat::GRAYSCALE_F32, image::Luma<f32>);
impl_supported_pixel!(ColorFormat::RGB_U8, image::Rgb<u8>);
impl_supported_pixel!(ColorFormat::RGB_U16, image::Rgb<u16>);
impl_supported_pixel!(ColorFormat::RGB_F32, image::Rgb<f32>);
impl_supported_pixel!(ColorFormat::RGBA_U8, image::Rgba<u8>);
impl_supported_pixel!(ColorFormat::RGBA_U16, image::Rgba<u16>);
impl_supported_pixel!(ColorFormat::RGBA_F32, image::Rgba<f32>);

impl<'a, P, Container> From<&'a ImageBuffer<P, Container>> for ImageView<'a>
where
    P: SupportedPixel,
    Container: Deref<Target = [P::Subpixel]>,
{
    /// Creates an `ImageView` of an `image::ImageBuffer` reference.
    ///
    /// ## Panic
    ///
    /// Panics if the underlying container of the image buffer has less than
    /// `width * height * channels` elements. Such image buffers are generally
    /// considered invalid by `image`, but they are possible to create.
    fn from(image: &'a ImageBuffer<P, Container>) -> Self {
        let size = Size::new(image.width(), image.height());
        let slice: &[P::Subpixel] = image.deref();
        let bytes: &[u8] = P::subpixels_as_bytes(slice);
        to_view(bytes, size, P::COLOR_FORMAT)
    }
}
impl<'a, P, Container> From<&'a mut ImageBuffer<P, Container>> for ImageViewMut<'a>
where
    P: SupportedPixel,
    Container: Deref<Target = [P::Subpixel]> + std::convert::AsMut<[P::Subpixel]>,
{
    /// Creates an `ImageViewMut` of a mutable `image::ImageBuffer` reference.
    ///
    /// ## Panic
    ///
    /// Panics if the underlying container of the image buffer has less than
    /// `width * height * channels` elements. Such image buffers are generally
    /// considered invalid by `image`, but they are possible to create.
    fn from(image: &'a mut ImageBuffer<P, Container>) -> Self {
        let size = Size::new(image.width(), image.height());
        let slice: &mut [P::Subpixel] = image.as_flat_samples_mut().samples;
        let bytes: &mut [u8] = P::subpixels_as_bytes_mut(slice);
        to_view_mut(bytes, size, P::COLOR_FORMAT)
    }
}

/// Error type for when a `DynamicImage`'s color type can't be represented by `ColorFormat`.
#[derive(Debug)]
pub struct UnsupportedColorType;

impl<'a> TryFrom<&'a DynamicImage> for ImageView<'a> {
    type Error = UnsupportedColorType;

    /// Creates an `ImageView` of an `image::DynamicImage` reference.
    ///
    /// Returns an error if the `DynamicImage`'s color type can't be represented
    /// exactly by [`ColorFormat`].
    ///
    /// ## Panic
    ///
    /// Panics if the underlying container of the image buffer has less than
    /// `width * height * channels` elements. Such image buffers are generally
    /// considered invalid by `image`, but they are possible to create.
    fn try_from(image: &'a DynamicImage) -> Result<Self, Self::Error> {
        let color = ColorFormat::try_from(image.color()).map_err(|_| UnsupportedColorType)?;
        let size = Size::new(image.width(), image.height());
        Ok(to_view(image.as_bytes(), size, color))
    }
}
impl<'a> TryFrom<&'a mut DynamicImage> for ImageViewMut<'a> {
    type Error = UnsupportedColorType;

    /// Creates an `ImageViewMut` of a mutable `image::DynamicImage` reference.
    ///
    /// Returns an error if the `DynamicImage`'s color type can't be represented
    /// exactly by [`ColorFormat`].
    ///
    /// ## Panic
    ///
    /// Panics if the underlying container of the image buffer has less than
    /// `width * height * channels` elements. Such image buffers are generally
    /// considered invalid by `image`, but they are possible to create.
    fn try_from(image: &'a mut DynamicImage) -> Result<Self, Self::Error> {
        Ok(match image {
            DynamicImage::ImageLuma8(image_buffer) => Self::from(image_buffer),
            DynamicImage::ImageRgb8(image_buffer) => Self::from(image_buffer),
            DynamicImage::ImageRgba8(image_buffer) => Self::from(image_buffer),
            DynamicImage::ImageLuma16(image_buffer) => Self::from(image_buffer),
            DynamicImage::ImageRgb16(image_buffer) => Self::from(image_buffer),
            DynamicImage::ImageRgba16(image_buffer) => Self::from(image_buffer),
            DynamicImage::ImageRgb32F(image_buffer) => Self::from(image_buffer),
            DynamicImage::ImageRgba32F(image_buffer) => Self::from(image_buffer),
            _ => return Err(UnsupportedColorType),
        })
    }
}

// Color format conversions

impl TryFrom<ColorFormat> for ColorType {
    type Error = ColorFormat;

    fn try_from(color: ColorFormat) -> Result<Self, Self::Error> {
        Ok(match color {
            ColorFormat::GRAYSCALE_U8 => ColorType::L8,
            ColorFormat::GRAYSCALE_U16 => ColorType::L16,
            ColorFormat::RGB_U8 => ColorType::Rgb8,
            ColorFormat::RGB_U16 => ColorType::Rgb16,
            ColorFormat::RGB_F32 => ColorType::Rgb32F,
            ColorFormat::RGBA_U8 => ColorType::Rgba8,
            ColorFormat::RGBA_U16 => ColorType::Rgba16,
            ColorFormat::RGBA_F32 => ColorType::Rgba32F,
            _ => return Err(color),
        })
    }
}
impl TryFrom<ColorType> for ColorFormat {
    type Error = ColorType;

    fn try_from(color: ColorType) -> Result<Self, Self::Error> {
        Ok(match color {
            ColorType::L8 => ColorFormat::GRAYSCALE_U8,
            ColorType::L16 => ColorFormat::GRAYSCALE_U16,
            ColorType::Rgb8 => ColorFormat::RGB_U8,
            ColorType::Rgb16 => ColorFormat::RGB_U16,
            ColorType::Rgb32F => ColorFormat::RGB_F32,
            ColorType::Rgba8 => ColorFormat::RGBA_U8,
            ColorType::Rgba16 => ColorFormat::RGBA_U16,
            ColorType::Rgba32F => ColorFormat::RGBA_F32,
            _ => return Err(color),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::*;

    const COLOR_TYPES: &[ColorType] = &[
        ColorType::L8,
        ColorType::La8,
        ColorType::Rgb8,
        ColorType::Rgba8,
        ColorType::L16,
        ColorType::La16,
        ColorType::Rgb16,
        ColorType::Rgba16,
        ColorType::Rgb32F,
        ColorType::Rgba32F,
    ];
    const COLOR_FORMATS: &[ColorFormat] = &[
        ColorFormat::ALPHA_U8,
        ColorFormat::GRAYSCALE_U8,
        ColorFormat::RGB_U8,
        ColorFormat::RGBA_U8,
        ColorFormat::ALPHA_U16,
        ColorFormat::GRAYSCALE_U16,
        ColorFormat::RGB_U16,
        ColorFormat::RGBA_U16,
        ColorFormat::ALPHA_F32,
        ColorFormat::GRAYSCALE_F32,
        ColorFormat::RGB_F32,
        ColorFormat::RGBA_F32,
    ];

    #[test]
    fn test_image_buffer_to_view() {
        let image_buffer = ImageBuffer::from_fn(2, 2, |x, y| {
            let r = x as u8;
            let g = y as u8;
            let b = 0;
            Rgb([r, g, b])
        });

        let view = ImageView::from(&image_buffer);
        assert_eq!(view.size(), Size::new(2, 2));
        assert_eq!(view.color(), ColorFormat::RGB_U8);
        assert_eq!(view.data(), &[0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);

        let dyn_image = DynamicImage::from(image_buffer);

        let view = ImageView::try_from(&dyn_image).unwrap();
        assert_eq!(view.size(), Size::new(2, 2));
        assert_eq!(view.color(), ColorFormat::RGB_U8);
        assert_eq!(view.data(), &[0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    }

    #[test]
    fn test_view_color() {
        assert_eq!(
            ImageView::from(&GrayImage::new(10, 10)).color(),
            ColorFormat::GRAYSCALE_U8
        );
        assert_eq!(
            ImageView::from(&RgbImage::new(10, 10)).color(),
            ColorFormat::RGB_U8
        );
        assert_eq!(
            ImageView::from(&RgbaImage::new(10, 10)).color(),
            ColorFormat::RGBA_U8
        );
        assert_eq!(
            ImageView::from(&Rgb32FImage::new(10, 10)).color(),
            ColorFormat::RGB_F32
        );
        assert_eq!(
            ImageView::from(&Rgba32FImage::new(10, 10)).color(),
            ColorFormat::RGBA_F32
        );

        for &color_type in COLOR_TYPES {
            assert_eq!(
                ImageView::try_from(&DynamicImage::new(10, 10, color_type))
                    .ok()
                    .map(|view| view.color()),
                ColorFormat::try_from(color_type).ok()
            );
        }
    }
    #[test]
    fn test_view_mut_color() {
        assert_eq!(
            ImageViewMut::from(&mut GrayImage::new(10, 10)).color(),
            ColorFormat::GRAYSCALE_U8
        );
        assert_eq!(
            ImageViewMut::from(&mut RgbImage::new(10, 10)).color(),
            ColorFormat::RGB_U8
        );
        assert_eq!(
            ImageViewMut::from(&mut RgbaImage::new(10, 10)).color(),
            ColorFormat::RGBA_U8
        );
        assert_eq!(
            ImageViewMut::from(&mut Rgb32FImage::new(10, 10)).color(),
            ColorFormat::RGB_F32
        );
        assert_eq!(
            ImageViewMut::from(&mut Rgba32FImage::new(10, 10)).color(),
            ColorFormat::RGBA_F32
        );

        for &color_type in COLOR_TYPES {
            assert_eq!(
                ImageViewMut::try_from(&mut DynamicImage::new(10, 10, color_type))
                    .ok()
                    .map(|view| view.color()),
                ColorFormat::try_from(color_type).ok()
            );
        }
    }

    #[test]
    fn test_container_too_long() {
        let image_buffer = GrayImage::from_vec(2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(image_buffer.len(), 9);
        assert_eq!(image_buffer.as_raw().len(), 9);

        let view = ImageView::from(&image_buffer);
        assert_eq!(view.size(), Size::new(2, 2));
        assert_eq!(view.color(), ColorFormat::GRAYSCALE_U8);
        assert_eq!(view.data(), &[1, 2, 3, 4]);

        let dyn_image = DynamicImage::from(image_buffer);
        assert_eq!(dyn_image.as_bytes().len(), 9);

        let view = ImageView::try_from(&dyn_image).unwrap();
        assert_eq!(view.size(), Size::new(2, 2));
        assert_eq!(view.color(), ColorFormat::GRAYSCALE_U8);
        assert_eq!(view.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_color_conversion() {
        for &color_type in COLOR_TYPES {
            if let Ok(color_format) = ColorFormat::try_from(color_type) {
                assert_eq!(ColorType::try_from(color_format), Ok(color_type));
            }
        }

        for &color_format in COLOR_FORMATS {
            if let Ok(color_type) = ColorType::try_from(color_format) {
                assert_eq!(ColorFormat::try_from(color_type), Ok(color_format));
            }
        }
    }
}
