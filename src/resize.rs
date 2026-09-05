use crate::{cast, Channels, ColorFormat, ImageView, Precision, ResizeFilter, Size};

use resize::{Filter, Resizer};

pub(crate) use aligned_types::*;

mod aligned_types {
    use crate::{cast, resize::is_aligned, ColorFormat, ImageView, Size};

    pub(crate) type AlignTo = u32;

    #[derive(Clone, Copy)]
    pub(crate) struct AlignedView<'a> {
        view: &'a [u8],
        size: Size,
        color: ColorFormat,
    }
    impl<'a> AlignedView<'a> {
        pub fn new(view: &'a [u8], size: Size, color: ColorFormat) -> Self {
            debug_assert_eq!(view.len(), color.buffer_size(size).expect("Invalid size"));
            debug_assert!(is_aligned(view, color.precision.size() as usize));

            Self { view, size, color }
        }

        pub fn view(&self) -> &'a [u8] {
            self.view
        }
        pub fn size(&self) -> Size {
            self.size
        }
        pub fn color(&self) -> ColorFormat {
            self.color
        }

        pub fn as_image_view(&self) -> ImageView<'a> {
            ImageView::new(self.view, self.size, self.color).expect("invalid aligned view")
        }
    }

    pub(crate) struct AlignedBuffer {
        buffer: Vec<AlignTo>,
        size: Size,
        color: ColorFormat,
    }
    impl AlignedBuffer {
        pub fn new(buffer: Vec<AlignTo>, size: Size, color: ColorFormat) -> Self {
            debug_assert!(
                buffer.len() * std::mem::size_of::<AlignTo>()
                    >= color.buffer_size(size).expect("Invalid size"),
                "Buffer too small for aligned buffer"
            );

            Self {
                buffer,
                size,
                color,
            }
        }

        pub fn as_view(&self) -> AlignedView<'_> {
            let byte_slice = cast::as_bytes(&self.buffer);
            let len = self.color.buffer_size(self.size).expect("Invalid size");

            AlignedView {
                view: &byte_slice[..len],
                size: self.size,
                color: self.color,
            }
        }
    }
}

pub(crate) struct Aligner {
    buffer: Vec<AlignTo>,
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
            let aligned_slice = get_aligned_slice(&mut self.buffer, size, color);
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
            let aligned_slice = get_aligned_slice(&mut self.buffer, size, color);
            aligned_slice.copy_from_slice(data);
            aligned_slice
        };

        AlignedView::new(view, size, color)
    }
}

pub(crate) struct ResizeState {
    dest_buffer: Vec<AlignTo>,
}
impl ResizeState {
    pub fn new() -> Self {
        Self {
            dest_buffer: Vec::new(),
        }
    }

    pub fn resize<'a>(
        &'a mut self,
        src: AlignedView,
        new_size: Size,
        straight_alpha: bool,
        gamma_correction: bool,
        filter: ResizeFilter,
    ) -> AlignedView<'a> {
        let color = src.color();

        // prepare the destination buffer
        let dest_slice = get_aligned_slice(&mut self.dest_buffer, new_size, color);

        resize_into(
            src,
            dest_slice,
            new_size,
            straight_alpha,
            gamma_correction,
            filter,
        );

        AlignedView::new(dest_slice, new_size, color)
    }
}

pub(crate) fn resize(
    src: AlignedView,
    new_size: Size,
    straight_alpha: bool,
    gamma_correction: bool,
    filter: ResizeFilter,
) -> AlignedBuffer {
    let color = src.color();

    // prepare the destination buffer
    let mut dest_buffer: Vec<AlignTo> = Vec::new();
    let dest_slice = get_aligned_slice(&mut dest_buffer, new_size, color);

    resize_into(
        src,
        dest_slice,
        new_size,
        straight_alpha,
        gamma_correction,
        filter,
    );

    AlignedBuffer::new(dest_buffer, new_size, color)
}

fn get_aligned_slice(buffer: &mut Vec<AlignTo>, size: Size, color: ColorFormat) -> &mut [u8] {
    let slice_len = color
        .buffer_size(size)
        .expect("size too big for aligned slice");

    // reserve enough space in the buffer
    let buffer_len = slice_len.div_ceil(std::mem::size_of::<AlignTo>());
    if buffer.len() < buffer_len {
        buffer.resize(buffer_len, 0);
    }

    &mut cast::as_bytes_mut(buffer.as_mut_slice())[..slice_len]
}
fn is_aligned(slice: &[u8], alignment: usize) -> bool {
    (slice.as_ptr() as usize) % alignment == 0
}

fn resize_into(
    src: AlignedView,
    dest_slice: &mut [u8],
    new_size: Size,
    straight_alpha: bool,
    gamma_correction: bool,
    filter: ResizeFilter,
) {
    let color = src.color();

    debug_assert_eq!(
        dest_slice.len(),
        color
            .buffer_size(new_size)
            .expect("invalid size for destination slice")
    );
    debug_assert!(is_aligned(src.view(), color.precision.size() as usize));
    debug_assert!(is_aligned(dest_slice, color.precision.size() as usize));

    let args = Args {
        src_size: src.size(),
        src_bytes: src.view(),
        dst_size: new_size,
        dst_bytes: dest_slice,
        filter,
    };

    fn dispatch<G: GammaCorrection>(args: Args, color: ColorFormat, straight_alpha: bool) {
        use Channels as C;
        use Precision as P;

        if straight_alpha && color.channels == C::Rgba {
            return match color.precision {
                P::U8 => resize_typed::<StraightAlpha<[u8; 4], G>>(args),
                P::U16 => resize_typed::<StraightAlpha<[u16; 4], G>>(args),
                P::F32 => resize_typed::<StraightAlpha<[f32; 4], G>>(args),
            };
        }

        match (color.precision, color.channels) {
            (P::U8, C::Alpha | C::Grayscale) => resize_typed::<Pixel<[u8; 1], G>>(args),
            (P::U16, C::Alpha | C::Grayscale) => resize_typed::<Pixel<[u16; 1], G>>(args),
            (P::F32, C::Alpha | C::Grayscale) => resize_typed::<Pixel<[f32; 1], G>>(args),
            (P::U8, C::Rgb) => resize_typed::<Pixel<[u8; 3], G>>(args),
            (P::U16, C::Rgb) => resize_typed::<Pixel<[u16; 3], G>>(args),
            (P::F32, C::Rgb) => resize_typed::<Pixel<[f32; 3], G>>(args),
            (P::U8, C::Rgba) => resize_typed::<Pixel<[u8; 4], G>>(args),
            (P::U16, C::Rgba) => resize_typed::<Pixel<[u16; 4], G>>(args),
            (P::F32, C::Rgba) => resize_typed::<Pixel<[f32; 4], G>>(args),
        }
    }

    if gamma_correction && color.channels != Channels::Alpha {
        dispatch::<ApplyGamma>(args, color, straight_alpha);
    } else {
        dispatch::<IgnoreGamma>(args, color, straight_alpha);
    }
}

struct Args<'a, 'b> {
    src_size: Size,
    src_bytes: &'a [u8],
    dst_size: Size,
    dst_bytes: &'b mut [u8],
    filter: ResizeFilter,
}
fn resize_typed<P>(
    Args {
        src_size,
        src_bytes,
        dst_size,
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
        src_size.width as usize,
        src_size.height as usize,
        dst_size.width as usize,
        dst_size.height as usize,
        P::default(),
        to_resize_filter_type(filter),
    )
    .expect("failed to create resizer");

    resizes.resize(src_slice, dst_slice).expect("resize failed");
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

use pixel::{ApplyGamma, GammaCorrection, IgnoreGamma, Pixel, StraightAlpha};
mod pixel {
    use glam::Vec4;

    pub(crate) struct Pixel<T, Gamma> {
        _marker: std::marker::PhantomData<(T, Gamma)>,
    }
    impl<T, Gamma> Default for Pixel<T, Gamma> {
        fn default() -> Self {
            Self {
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub(crate) struct IgnoreGamma;
    pub(crate) struct ApplyGamma;
    pub(crate) trait GammaCorrection: Send + Sync {
        fn to_linear_vec4(input: Vec4) -> Vec4 {
            input
        }
        fn to_linear_f32(input: f32) -> f32 {
            input
        }

        fn from_linear_vec4(input: Vec4) -> Vec4 {
            input
        }
        fn from_linear_f32(input: f32) -> f32 {
            input
        }
    }
    impl GammaCorrection for IgnoreGamma {}
    // Instead of a proper sRGB <-> linear RGB conversion or gamma 2.2, we use gamma 2.0 here.
    // This is visually very similar to gamma 2.2 but a lot faster to compute.
    impl GammaCorrection for ApplyGamma {
        fn to_linear_vec4(input: Vec4) -> Vec4 {
            let mut out = input * input;
            out.w = input.w; // don't apply gamma correction to alpha channel
            out
        }
        fn to_linear_f32(input: f32) -> f32 {
            input * input
        }

        fn from_linear_vec4(input: Vec4) -> Vec4 {
            // TODO: Hope and pray that this mess gets optimized to a sqrt_ps instruction by the compiler
            let mut out = input.max(Vec4::ZERO).powf(0.5);
            out.w = input.w;
            out
        }
        fn from_linear_f32(input: f32) -> f32 {
            input.max(0.0).sqrt()
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
        fn to_accumulator<G: GammaCorrection>(self) -> T;
        fn to_value<G: GammaCorrection>(acc: T) -> Self;
    }
    impl IntoAccumulator<f32> for [u8; 1] {
        fn to_accumulator<G: GammaCorrection>(self) -> f32 {
            G::to_linear_f32(self[0] as f32)
        }
        fn to_value<G: GammaCorrection>(acc: f32) -> Self {
            [(G::from_linear_f32(acc) + 0.5) as u8]
        }
    }
    impl IntoAccumulator<f32> for [u16; 1] {
        fn to_accumulator<G: GammaCorrection>(self) -> f32 {
            G::to_linear_f32(self[0] as f32)
        }
        fn to_value<G: GammaCorrection>(acc: f32) -> Self {
            [(G::from_linear_f32(acc) + 0.5) as u16]
        }
    }
    impl IntoAccumulator<f32> for [f32; 1] {
        fn to_accumulator<G: GammaCorrection>(self) -> f32 {
            G::to_linear_f32(self[0])
        }
        fn to_value<G: GammaCorrection>(acc: f32) -> Self {
            [G::from_linear_f32(acc)]
        }
    }
    impl IntoAccumulator<Vec4> for [u8; 3] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            G::to_linear_vec4(Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                0.0,
            ))
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let out = G::from_linear_vec4(acc) + 0.5;
            [out.x as u8, out.y as u8, out.z as u8]
        }
    }
    impl IntoAccumulator<Vec4> for [u16; 3] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            G::to_linear_vec4(Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                0.0,
            ))
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let out = G::from_linear_vec4(acc) + 0.5;
            [out.x as u16, out.y as u16, out.z as u16]
        }
    }
    impl IntoAccumulator<Vec4> for [f32; 3] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            G::to_linear_vec4(Vec4::new(self[0], self[1], self[2], 0.0))
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let out = G::from_linear_vec4(acc);
            [out.x, out.y, out.z]
        }
    }
    impl IntoAccumulator<Vec4> for [u8; 4] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            G::to_linear_vec4(Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                self[3] as f32,
            ))
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let out = G::from_linear_vec4(acc) + 0.5;
            [out.x as u8, out.y as u8, out.z as u8, out.w as u8]
        }
    }
    impl IntoAccumulator<Vec4> for [u16; 4] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            G::to_linear_vec4(Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                self[3] as f32,
            ))
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let out = G::from_linear_vec4(acc) + 0.5;
            [out.x as u16, out.y as u16, out.z as u16, out.w as u16]
        }
    }
    impl IntoAccumulator<Vec4> for [f32; 4] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            G::to_linear_vec4(Vec4::new(self[0], self[1], self[2], self[3]))
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let out = G::from_linear_vec4(acc);
            [out.x, out.y, out.z, out.w]
        }
    }

    impl<T, G, const N: usize> resize::PixelFormat for Pixel<[T; N], G>
    where
        T: Send + Sync + Copy,
        G: GammaCorrection,
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
            acc.add_scaled(inp.to_accumulator::<G>(), coeff);
        }

        fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
            acc.add_scaled(inp, coeff);
        }

        fn into_pixel(&self, acc: Self::Accumulator) -> Self::OutputPixel {
            Self::OutputPixel::to_value::<G>(acc)
        }
    }

    pub(crate) struct StraightAlpha<T, G> {
        _marker: std::marker::PhantomData<(T, G)>,
    }
    impl<T, G> Default for StraightAlpha<T, G> {
        fn default() -> Self {
            Self {
                _marker: std::marker::PhantomData,
            }
        }
    }
    pub(crate) trait IntoStraightAlphaAccumulator<T> {
        fn to_accumulator<G: GammaCorrection>(self) -> T;
        fn to_value<G: GammaCorrection>(acc: T) -> Self;
    }
    const ALPHA_EPSILON: f32 = 0.01;
    impl IntoStraightAlphaAccumulator<Vec4> for [u8; 4] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            let v = G::to_linear_vec4(Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                self[3] as f32,
            ));
            v * Vec4::new(v.w, v.w, v.w, 1.0)
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let a = acc.w;
            let a_r = if a <= ALPHA_EPSILON { 0.0 } else { a.recip() };
            let f = Vec4::new(a_r, a_r, a_r, 1.0);
            let out = G::from_linear_vec4(acc * f) + 0.5;
            [out.x as u8, out.y as u8, out.z as u8, out.w as u8]
        }
    }
    impl IntoStraightAlphaAccumulator<Vec4> for [u16; 4] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            let v = G::to_linear_vec4(Vec4::new(
                self[0] as f32,
                self[1] as f32,
                self[2] as f32,
                self[3] as f32,
            ));
            v * Vec4::new(v.w, v.w, v.w, 1.0)
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let a = acc.w;
            let a_r = if a <= ALPHA_EPSILON { 0.0 } else { a.recip() };
            let f = Vec4::new(a_r, a_r, a_r, 1.0);
            let out = G::from_linear_vec4(acc * f) + 0.5;
            [out.x as u16, out.y as u16, out.z as u16, out.w as u16]
        }
    }
    impl IntoStraightAlphaAccumulator<Vec4> for [f32; 4] {
        fn to_accumulator<G: GammaCorrection>(self) -> Vec4 {
            let v = G::to_linear_vec4(Vec4::new(self[0], self[1], self[2], self[3]));
            v * Vec4::new(v.w, v.w, v.w, 1.0)
        }
        fn to_value<G: GammaCorrection>(acc: Vec4) -> Self {
            let a = acc.w;
            let a_r = if a <= 0.0 { 0.0 } else { a.recip() };
            let f = Vec4::new(a_r, a_r, a_r, 1.0);
            let out = G::from_linear_vec4(acc * f).max(Vec4::ZERO);
            [out.x, out.y, out.z, out.w]
        }
    }

    impl<T, G> resize::PixelFormat for StraightAlpha<[T; 4], G>
    where
        T: Send + Sync + Copy,
        G: GammaCorrection,
        [T; 4]: Default + IntoStraightAlphaAccumulator<Vec4>,
    {
        type InputPixel = [T; 4];

        type OutputPixel = [T; 4];

        type Accumulator = Vec4;

        fn new() -> Self::Accumulator {
            Self::Accumulator::zero()
        }

        fn add(&self, acc: &mut Self::Accumulator, inp: Self::InputPixel, coeff: f32) {
            acc.add_scaled(inp.to_accumulator::<G>(), coeff);
        }

        fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
            acc.add_scaled(inp, coeff);
        }

        fn into_pixel(&self, acc: Self::Accumulator) -> Self::OutputPixel {
            Self::OutputPixel::to_value::<G>(acc)
        }
    }
}
