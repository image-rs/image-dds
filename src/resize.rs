use fast_image_resize::{
    images::{Image, ImageRef},
    PixelType,
};

use crate::{Channels, ColorFormat, Precision, ResizeFilter, Size};

pub(crate) struct Aligner {
    buffer: Vec<u8>,
}
impl Aligner {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn align<'a>(
        &'a mut self,
        data: &'a [u8],
        size: Size,
        color: ColorFormat,
    ) -> AlignedView<'a> {
        let pixel = to_pixel_type(color);
        let bytes_per_pixel = pixel.size();
        debug_assert_eq!(bytes_per_pixel, color.bytes_per_pixel() as usize);
        debug_assert_eq!(size.pixels() as usize * bytes_per_pixel, data.len());

        let view = if let Ok(src) = ImageRef::new(size.width, size.height, data, pixel) {
            src
        } else {
            // the image data isn't aligned, so we need to copy it to an aligned buffer
            let src_slice = get_aligned_slice(&mut self.buffer, size, bytes_per_pixel);
            src_slice.copy_from_slice(data);
            ImageRef::new(size.width, size.height, src_slice, pixel).expect("image should be valid")
        };

        AlignedView { view, pixel }
    }
}

pub(crate) struct AlignedView<'a> {
    view: ImageRef<'a>,
    pixel: PixelType,
}

pub(crate) struct ResizeState {
    inner: fast_image_resize::Resizer,
    dest_buffer: Vec<u8>,
}
impl ResizeState {
    pub fn new() -> Self {
        Self {
            inner: fast_image_resize::Resizer::new(),
            dest_buffer: Vec::new(),
        }
    }

    pub fn resize<'a>(
        &'a mut self,
        src: &AlignedView,
        new_size: Size,
        straight_alpha: bool,
        filter: ResizeFilter,
    ) -> &'a [u8] {
        let bytes_per_pixel = src.pixel.size();

        // prepare the destination buffer
        let dest_slice = get_aligned_slice(&mut self.dest_buffer, new_size, bytes_per_pixel);
        let mut dst = Image::from_slice_u8(new_size.width, new_size.height, dest_slice, src.pixel)
            .expect("image should be valid");

        // the actual resizing
        let options = fast_image_resize::ResizeOptions {
            mul_div_alpha: straight_alpha,
            algorithm: to_resize_algorithm(filter),
            ..Default::default()
        };

        self.inner
            .resize(&src.view, &mut dst, &options)
            .expect("resize should always succeed");

        dest_slice
    }
}

fn get_aligned_slice(buffer: &mut Vec<u8>, size: Size, bytes_per_pixel: usize) -> &mut [u8] {
    let slice_len = size.pixels() as usize * bytes_per_pixel;
    let align_to = 16;

    // we want the buffer to slightly larger than the slice, so we have
    // some space to align the slice
    let buffer_len = size.pixels() as usize * bytes_per_pixel + align_to;
    if buffer.len() < buffer_len {
        buffer.resize(buffer_len, 0);
    }

    // figure out the offset which aligns the slice
    let mut aligned_offset = 0;
    for offset in 0..align_to {
        let slice = &mut buffer[offset..offset + slice_len];
        if is_aligned(slice, align_to) {
            aligned_offset = offset;
            break;
        }
    }

    &mut buffer[aligned_offset..aligned_offset + slice_len]
}

fn is_aligned(slice: &[u8], alignment: usize) -> bool {
    (slice.as_ptr() as usize) % alignment == 0
}

fn to_pixel_type(color: ColorFormat) -> PixelType {
    match (color.precision, color.channels) {
        (Precision::U8, Channels::Grayscale | Channels::Alpha) => PixelType::U8,
        (Precision::U8, Channels::Rgb) => PixelType::U8x3,
        (Precision::U8, Channels::Rgba) => PixelType::U8x4,
        (Precision::U16, Channels::Grayscale | Channels::Alpha) => PixelType::U16,
        (Precision::U16, Channels::Rgb) => PixelType::U16x3,
        (Precision::U16, Channels::Rgba) => PixelType::U16x4,
        (Precision::F32, Channels::Grayscale | Channels::Alpha) => PixelType::F32,
        (Precision::F32, Channels::Rgb) => PixelType::F32x3,
        (Precision::F32, Channels::Rgba) => PixelType::F32x4,
    }
}
fn to_resize_algorithm(filter: ResizeFilter) -> fast_image_resize::ResizeAlg {
    use fast_image_resize::{FilterType, ResizeAlg};

    match filter {
        ResizeFilter::Nearest => ResizeAlg::Nearest,
        ResizeFilter::Box => ResizeAlg::Convolution(FilterType::Box),
        ResizeFilter::Triangle => ResizeAlg::Convolution(FilterType::Bilinear),
        ResizeFilter::Mitchell => ResizeAlg::Convolution(FilterType::Mitchell),
        ResizeFilter::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
    }
}
