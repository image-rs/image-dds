use dds::*;

use crate::util::{as_bytes, as_bytes_mut, cast_slice, cast_slice_mut, Castable};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Image<T> {
    pub data: Vec<T>,
    pub channels: Channels,
    pub size: Size,
}
impl<T> Image<T> {
    #[track_caller]
    pub fn new(data: Vec<T>, channels: Channels, size: Size) -> Self {
        assert_eq!(
            data.len(),
            size.pixels() as usize * channels.count() as usize,
            "Data size doesn't match image size"
        );

        Self {
            data,
            channels,
            size,
        }
    }
    pub fn new_empty(channels: Channels, size: Size) -> Self
    where
        T: Default + Copy,
    {
        let data = vec![T::default(); size.pixels() as usize * channels.count() as usize];
        Self::new(data, channels, size)
    }

    pub fn stride(&self) -> usize {
        self.size.width as usize * self.channels.count() as usize * std::mem::size_of::<T>()
    }

    pub fn as_bytes(&self) -> &[u8]
    where
        T: Castable,
    {
        as_bytes(&self.data)
    }
    pub fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        T: Castable,
    {
        as_bytes_mut(&mut self.data)
    }

    pub fn precision(&self) -> Precision
    where
        T: WithPrecision,
    {
        T::PRECISION
    }
    pub fn color(&self) -> ColorFormat
    where
        T: WithPrecision,
    {
        ColorFormat::new(self.channels, T::PRECISION)
    }

    pub fn view(&'_ self) -> ImageView<'_>
    where
        T: Castable + WithPrecision,
    {
        ImageView::new(self.as_bytes(), self.size, self.color()).unwrap()
    }
    pub fn view_mut(&'_ mut self) -> ImageViewMut<'_>
    where
        T: Castable + WithPrecision,
    {
        let size = self.size;
        let color = self.color();
        ImageViewMut::new(self.as_bytes_mut(), size, color).unwrap()
    }

    pub fn to_channels(&self, channels: Channels) -> Image<T>
    where
        T: Copy + Default + Castable + Norm,
    {
        Image::new(
            convert_channels(&self.data, self.channels, channels),
            channels,
            self.size,
        )
    }

    pub fn cropped(&self, new_size: Size) -> Image<T>
    where
        T: Copy,
    {
        if new_size == self.size {
            return self.clone();
        }
        assert!(new_size.width <= self.size.width);
        assert!(new_size.height <= self.size.height);

        let new_width = new_size.width as usize;
        let new_height = new_size.height as usize;
        let new_stride = new_width * self.channels.count() as usize;

        let mut new_data = Vec::with_capacity(new_stride * new_height);
        for y in 0..new_height {
            let src_offset = y * self.size.width as usize * self.channels.count() as usize;
            let dst_offset = y * new_stride;
            new_data.extend_from_slice(&self.data[src_offset..src_offset + new_stride]);
        }

        Image::new(new_data, self.channels, new_size)
    }
}
impl Image<u8> {
    pub fn to_u16(&self) -> Image<u16> {
        Image::new(
            self.data.iter().map(|&x| x as u16 * 257).collect(),
            self.channels,
            self.size,
        )
    }
    pub fn to_f32(&self) -> Image<f32> {
        Image::new(
            self.data.iter().map(|&x| x as f32 / 255.0).collect(),
            self.channels,
            self.size,
        )
    }
}

pub trait WithPrecision {
    const PRECISION: Precision;
}
impl WithPrecision for u8 {
    const PRECISION: Precision = Precision::U8;
}
impl WithPrecision for u16 {
    const PRECISION: Precision = Precision::U16;
}
impl WithPrecision for f32 {
    const PRECISION: Precision = Precision::F32;
}

pub trait Norm {
    const NORM_ONE: Self;
    const NORM_ZERO: Self;
}
impl Norm for u8 {
    const NORM_ONE: Self = u8::MAX;
    const NORM_ZERO: Self = 0;
}
impl Norm for u16 {
    const NORM_ONE: Self = u16::MAX;
    const NORM_ZERO: Self = 0;
}
impl Norm for f32 {
    const NORM_ONE: Self = 1.0;
    const NORM_ZERO: Self = 0.0;
}

pub fn convert_channels<T>(data: &[T], from: Channels, to: Channels) -> Vec<T>
where
    T: Copy + Default + Castable + Norm,
{
    if from == to {
        return data.to_vec();
    }

    fn convert<const N: usize, const M: usize, T>(
        data: &[T],
        f: impl Fn([T; N]) -> [T; M],
    ) -> Vec<T>
    where
        T: Copy + Default + Castable,
    {
        let pixels = data.len() / N;
        let mut result: Vec<T> = vec![Default::default(); pixels * M];

        let data_n: &[[T; N]] = cast_slice(data);
        let result_m: &mut [[T; M]] = cast_slice_mut(&mut result);

        for (i, o) in data_n.iter().zip(result_m.iter_mut()) {
            *o = f(*i);
        }

        result
    }

    match (from, to) {
        // already handled
        (Channels::Grayscale, Channels::Grayscale)
        | (Channels::Alpha, Channels::Alpha)
        | (Channels::Rgb, Channels::Rgb)
        | (Channels::Rgba, Channels::Rgba) => unreachable!(),

        (Channels::Grayscale, Channels::Alpha) => convert(data, |[_]| [T::NORM_ONE]),
        (Channels::Grayscale, Channels::Rgb) => convert(data, |[g]| [g, g, g]),
        (Channels::Grayscale, Channels::Rgba) => convert(data, |[g]| [g, g, g, T::NORM_ONE]),
        (Channels::Alpha, Channels::Grayscale) => convert(data, |[_]| [T::NORM_ZERO]),
        (Channels::Alpha, Channels::Rgb) => {
            convert(data, |[_]| [T::NORM_ZERO, T::NORM_ZERO, T::NORM_ZERO])
        }
        (Channels::Alpha, Channels::Rgba) => {
            convert(data, |[a]| [T::NORM_ZERO, T::NORM_ZERO, T::NORM_ZERO, a])
        }
        (Channels::Rgb, Channels::Grayscale) => convert(data, |[r, _, _]| [r]),
        (Channels::Rgb, Channels::Alpha) => convert(data, |[_, _, _]| [T::NORM_ONE]),
        (Channels::Rgb, Channels::Rgba) => convert(data, |[r, g, b]| [r, g, b, T::NORM_ONE]),
        (Channels::Rgba, Channels::Grayscale) => convert(data, |[r, _, _, _]| [r]),
        (Channels::Rgba, Channels::Alpha) => convert(data, |[_, _, _, a]| [a]),
        (Channels::Rgba, Channels::Rgb) => convert(data, |[r, g, b, _]| [r, g, b]),
    }
}
