use crate::{cast, ColorFormat, ImageView, Precision, ResizeFilter, Size};

use resize::{Filter, Resizer};

pub(crate) struct Aligner {
    buffer: Vec<u8>,
}
impl Aligner {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn align<'a>(&'a mut self, image: ImageView<'a>) -> AlignedView<'a> {
        let size = image.size();
        let color = image.color();
        let data = image.data();

        let bytes_per_pixel = color.bytes_per_pixel() as usize;

        let view = if !image.is_contiguous() {
            // Right now, the implementation assumes that the data to be
            // contiguous, so we need to copy it to an aligned buffer line by line.
            let aligned_slice = get_aligned_slice(&mut self.buffer, size, bytes_per_pixel);
            let bytes_per_row = size.width as usize * bytes_per_pixel;
            for (y, data_row) in image.rows().enumerate() {
                debug_assert_eq!(data_row.len(), bytes_per_row);
                let a_start = y * bytes_per_row;
                let a_end = a_start + bytes_per_row;
                aligned_slice[a_start..a_end].copy_from_slice(data_row);
            }
            aligned_slice
        } else if is_aligned(data, color.precision.size() as usize) {
            data
        } else {
            // the image data isn't aligned, so we need to copy it to an aligned buffer
            let aligned_slice = get_aligned_slice(&mut self.buffer, size, bytes_per_pixel);
            aligned_slice.copy_from_slice(data);
            aligned_slice
        };

        AlignedView { view, size, color }
    }
}

pub(crate) struct AlignedView<'a> {
    view: &'a [u8],
    size: Size,
    color: ColorFormat,
}

pub(crate) struct ResizeState {
    dest_buffer: Vec<u8>,
}
impl ResizeState {
    pub fn new() -> Self {
        Self {
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
        let bytes_per_pixel = src.color.bytes_per_pixel() as usize;

        // prepare the destination buffer
        let dest_slice = get_aligned_slice(&mut self.dest_buffer, new_size, bytes_per_pixel);

        let filter = to_resize_filter_type(filter);
        let args = Args {
            size: src.size,
            src_bytes: src.view,
            new_size,
            dst_bytes: dest_slice,
            filter,
        };

        use Precision::*;
        match (src.color.precision, src.color.channels.count()) {
            (U8, 1) => resize_typed::<Pixel<[u8; 1]>>(args),
            (U16, 1) => resize_typed::<Pixel<[u16; 1]>>(args),
            (F32, 1) => resize_typed::<Pixel<[f32; 1]>>(args),
            (U8, 3) => resize_typed::<Pixel<[u8; 3]>>(args),
            (U16, 3) => resize_typed::<Pixel<[u16; 3]>>(args),
            (F32, 3) => resize_typed::<Pixel<[f32; 3]>>(args),
            (U8, 4) => {
                if straight_alpha {
                    resize_typed::<StraightAlpha<[u8; 4]>>(args)
                } else {
                    resize_typed::<Pixel<[u8; 4]>>(args)
                }
            }
            (U16, 4) => {
                if straight_alpha {
                    resize_typed::<StraightAlpha<[u16; 4]>>(args)
                } else {
                    resize_typed::<Pixel<[u16; 4]>>(args)
                }
            }
            (F32, 4) => {
                if straight_alpha {
                    resize_typed::<StraightAlpha<[f32; 4]>>(args)
                } else {
                    resize_typed::<Pixel<[f32; 4]>>(args)
                }
            }
            _ => unreachable!(),
        }

        dest_slice
    }
}

struct Args<'a, 'b> {
    size: Size,
    src_bytes: &'a [u8],
    new_size: Size,
    dst_bytes: &'b mut [u8],
    filter: resize::Type,
}
fn resize_typed<P>(
    Args {
        size,
        src_bytes,
        new_size,
        dst_bytes,
        filter,
    }: Args,
) where
    P: resize::PixelFormat + Default,
    P::InputPixel: cast::Castable,
    P::OutputPixel: cast::Castable,
{
    let src_slice: &[P::InputPixel] = cast::from_bytes(src_bytes).expect("invalid source data");
    let dst_slice: &mut [P::OutputPixel] =
        cast::from_bytes_mut(dst_bytes).expect("invalid destination data");

    let mut resizes: Resizer<P> = resize::Resizer::new(
        size.width as usize,
        size.height as usize,
        new_size.width as usize,
        new_size.height as usize,
        P::default(),
        filter,
    )
    .unwrap();

    resizes.resize(src_slice, dst_slice).unwrap();
}

fn get_aligned_slice(buffer: &mut Vec<u8>, size: Size, bytes_per_pixel: usize) -> &mut [u8] {
    let slice_len = size.pixels() as usize * bytes_per_pixel;
    let align_to = 4;

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

fn to_resize_filter_type(filter: ResizeFilter) -> resize::Type {
    match filter {
        ResizeFilter::Nearest => resize::Type::Point,
        ResizeFilter::Box => resize::Type::Custom(Filter::box_filter(1.0)),
        ResizeFilter::Triangle => resize::Type::Triangle,
        ResizeFilter::Mitchell => resize::Type::Mitchell,
        ResizeFilter::Lanczos3 => resize::Type::Lanczos3,
    }
}

use pixel::{Pixel, StraightAlpha};
mod pixel {
    use glam::Vec4;

    pub(crate) struct Pixel<T> {
        _marker: std::marker::PhantomData<T>,
    }
    impl<T> Default for Pixel<T> {
        fn default() -> Self {
            Self {
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub(crate) trait SelectAccumulator {
        type Accumulator: Accumulator;
    }
    pub(crate) struct Selector<const N: usize>;
    impl SelectAccumulator for Selector<1> {
        type Accumulator = f32;
    }
    impl SelectAccumulator for Selector<3> {
        type Accumulator = Vec4;
    }
    impl SelectAccumulator for Selector<4> {
        type Accumulator = Vec4;
    }

    pub(crate) trait Accumulator: Copy + Send + Sync {
        fn zero() -> Self;
        fn add_scaled(&mut self, input: Self, scale: f32);
    }
    impl Accumulator for f32 {
        fn zero() -> Self {
            0.0
        }
        fn add_scaled(&mut self, input: Self, scale: f32) {
            *self += input * scale;
        }
    }
    impl Accumulator for Vec4 {
        fn zero() -> Self {
            Vec4::ZERO
        }
        fn add_scaled(&mut self, input: Self, scale: f32) {
            *self += input * scale;
        }
    }

    pub(crate) trait IntoAccumulator<T> {
        fn to_accumulator(self) -> T;
        fn to_value(acc: T) -> Self;
    }
    impl IntoAccumulator<f32> for [u8; 1] {
        fn to_accumulator(self) -> f32 {
            self[0] as f32
        }
        fn to_value(acc: f32) -> Self {
            [(acc + 0.5) as u8]
        }
    }
    impl IntoAccumulator<f32> for [u16; 1] {
        fn to_accumulator(self) -> f32 {
            self[0] as f32
        }
        fn to_value(acc: f32) -> Self {
            [(acc + 0.5) as u16]
        }
    }
    impl IntoAccumulator<f32> for [f32; 1] {
        fn to_accumulator(self) -> f32 {
            self[0]
        }
        fn to_value(acc: f32) -> Self {
            [acc]
        }
    }
    impl IntoAccumulator<Vec4> for [u8; 3] {
        fn to_accumulator(self) -> Vec4 {
            Vec4::new(self[0] as f32, self[1] as f32, self[2] as f32, 0.0)
        }
        fn to_value(acc: Vec4) -> Self {
            [
                (acc.x + 0.5) as u8,
                (acc.y + 0.5) as u8,
                (acc.z + 0.5) as u8,
            ]
        }
    }
    impl IntoAccumulator<Vec4> for [u16; 3] {
        fn to_accumulator(self) -> Vec4 {
            Vec4::new(self[0] as f32, self[1] as f32, self[2] as f32, 0.0)
        }
        fn to_value(acc: Vec4) -> Self {
            [
                (acc.x + 0.5) as u16,
                (acc.y + 0.5) as u16,
                (acc.z + 0.5) as u16,
            ]
        }
    }
    impl IntoAccumulator<Vec4> for [f32; 3] {
        fn to_accumulator(self) -> Vec4 {
            Vec4::new(self[0], self[1], self[2], 0.0)
        }
        fn to_value(acc: Vec4) -> Self {
            [acc.x, acc.y, acc.z]
        }
    }
    impl IntoAccumulator<Vec4> for [u8; 4] {
        fn to_accumulator(self) -> Vec4 {
            Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                self[3] as f32,
            )
        }
        fn to_value(acc: Vec4) -> Self {
            [
                (acc.x + 0.5) as u8,
                (acc.y + 0.5) as u8,
                (acc.z + 0.5) as u8,
                (acc.w + 0.5) as u8,
            ]
        }
    }
    impl IntoAccumulator<Vec4> for [u16; 4] {
        fn to_accumulator(self) -> Vec4 {
            Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                self[3] as f32,
            )
        }
        fn to_value(acc: Vec4) -> Self {
            [
                (acc.x + 0.5) as u16,
                (acc.y + 0.5) as u16,
                (acc.z + 0.5) as u16,
                (acc.w + 0.5) as u16,
            ]
        }
    }
    impl IntoAccumulator<Vec4> for [f32; 4] {
        fn to_accumulator(self) -> Vec4 {
            Vec4::new(self[0], self[1], self[2], self[3])
        }
        fn to_value(acc: Vec4) -> Self {
            [acc.x, acc.y, acc.z, acc.w]
        }
    }

    impl<T, const N: usize> resize::PixelFormat for Pixel<[T; N]>
    where
        T: Send + Sync + Copy,
        [T; N]: Default,
        Selector<N>: SelectAccumulator,
        [T; N]: IntoAccumulator<<Selector<N> as SelectAccumulator>::Accumulator>,
    {
        type InputPixel = [T; N];

        type OutputPixel = [T; N];

        type Accumulator = <Selector<N> as SelectAccumulator>::Accumulator;

        fn new() -> Self::Accumulator {
            Self::Accumulator::zero()
        }

        fn add(&self, acc: &mut Self::Accumulator, inp: Self::InputPixel, coeff: f32) {
            acc.add_scaled(inp.to_accumulator(), coeff);
        }

        fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
            acc.add_scaled(inp, coeff);
        }

        fn into_pixel(&self, acc: Self::Accumulator) -> Self::OutputPixel {
            Self::OutputPixel::to_value(acc)
        }
    }

    pub(crate) struct StraightAlpha<T> {
        _marker: std::marker::PhantomData<T>,
    }
    impl<T> Default for StraightAlpha<T> {
        fn default() -> Self {
            Self {
                _marker: std::marker::PhantomData,
            }
        }
    }
    pub(crate) trait IntoStraightAlphaAccumulator<T> {
        fn to_accumulator(self) -> T;
        fn to_value(acc: T) -> Self;
    }
    impl IntoStraightAlphaAccumulator<Vec4> for [u8; 4] {
        fn to_accumulator(self) -> Vec4 {
            let a = self[3] as f32;
            Vec4::new(
                self[0] as f32 * a,
                self[1] as f32 * a,
                self[2] as f32 * a,
                a,
            )
        }
        fn to_value(acc: Vec4) -> Self {
            let a = acc.w;
            let a_out = (a + 0.5) as u8;
            if a_out == 0 {
                return [0, 0, 0, 0];
            }
            let a_r = a.recip();
            [
                (acc.x * a_r + 0.5) as u8,
                (acc.y * a_r + 0.5) as u8,
                (acc.z * a_r + 0.5) as u8,
                a_out,
            ]
        }
    }
    impl IntoStraightAlphaAccumulator<Vec4> for [u16; 4] {
        fn to_accumulator(self) -> Vec4 {
            let a = self[3] as f32;
            Vec4::new(
                self[0] as f32 * a,
                self[1] as f32 * a,
                self[2] as f32 * a,
                a,
            )
        }
        fn to_value(acc: Vec4) -> Self {
            let a = acc.w;
            let a_out = (a + 0.5) as u16;
            if a_out == 0 {
                return [0, 0, 0, 0];
            }
            let a_r = a.recip();
            [
                (acc.x * a_r + 0.5) as u16,
                (acc.y * a_r + 0.5) as u16,
                (acc.z * a_r + 0.5) as u16,
                a_out,
            ]
        }
    }
    impl IntoStraightAlphaAccumulator<Vec4> for [f32; 4] {
        fn to_accumulator(self) -> Vec4 {
            let a = self[3];
            Vec4::new(self[0] * a, self[1] * a, self[2] * a, a)
        }
        fn to_value(acc: Vec4) -> Self {
            let a = acc.w;
            if a == 0.0 {
                return [0.0, 0.0, 0.0, 0.0];
            }
            let a_r = 1.0 / a.recip();
            [acc.x * a_r, acc.y * a_r, acc.z * a_r, a]
        }
    }

    impl<T> resize::PixelFormat for StraightAlpha<[T; 4]>
    where
        T: Send + Sync + Copy,
        [T; 4]: Default + IntoStraightAlphaAccumulator<Vec4>,
    {
        type InputPixel = [T; 4];

        type OutputPixel = [T; 4];

        type Accumulator = Vec4;

        fn new() -> Self::Accumulator {
            Self::Accumulator::zero()
        }

        fn add(&self, acc: &mut Self::Accumulator, inp: Self::InputPixel, coeff: f32) {
            acc.add_scaled(inp.to_accumulator(), coeff);
        }

        fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
            acc.add_scaled(inp, coeff);
        }

        fn into_pixel(&self, acc: Self::Accumulator) -> Self::OutputPixel {
            Self::OutputPixel::to_value(acc)
        }
    }
}
