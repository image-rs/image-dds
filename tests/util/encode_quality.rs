use bitflags::bitflags;
use dds::*;

use crate::util::{cast_slice, Image};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricChannel {
    R = 0,
    G = 1,
    B = 2,
    A = 3,
    Gray = 4,
    L = 5,
    C = 6,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MetricChannelSet: u8 {
        const R = 1 << MetricChannel::R as u8;
        const G = 1 << MetricChannel::G as u8;
        const B = 1 << MetricChannel::B as u8;
        const A = 1 << MetricChannel::A as u8;
        const GRAY = 1 << MetricChannel::Gray as u8;
        const LUM = 1 << MetricChannel::L as u8;
        const COLOR = 1 << MetricChannel::C as u8;
        const RGB = Self::R.bits() | Self::G.bits() | Self::B.bits();
        const RGBA = Self::RGB.bits() | Self::A.bits();
    }
}

impl From<MetricChannel> for MetricChannelSet {
    fn from(channel: MetricChannel) -> Self {
        MetricChannelSet::from_bits(1 << channel as u8).unwrap()
    }
}
impl std::ops::BitOr<MetricChannel> for MetricChannelSet {
    type Output = MetricChannelSet;
    fn bitor(self, rhs: MetricChannel) -> Self::Output {
        self | MetricChannelSet::from(rhs)
    }
}
impl std::ops::BitOr<MetricChannel> for MetricChannel {
    type Output = MetricChannelSet;
    fn bitor(self, rhs: MetricChannel) -> Self::Output {
        MetricChannelSet::from(self) | MetricChannelSet::from(rhs)
    }
}

#[derive(Clone)]
pub struct Metrics {
    pub channel: MetricChannel,
    pub mse: f64,
    /// This is the MSE of the image after a small blur
    pub mse_blur: f64,
    pub region_error: f64,
}
impl Metrics {
    pub fn psnr(&self) -> f64 {
        -10.0 * self.mse.log(10.0)
    }
    pub fn psnr_blur(&self) -> f64 {
        -10.0 * self.mse_blur.log(10.0)
    }
}
pub fn measure_compression_quality(
    org: &Image<f32>,
    compressed: &Image<f32>,
    channels: MetricChannelSet,
) -> Vec<Metrics> {
    let org = org.to_channels(Channels::Rgba);
    let compressed = compressed.to_channels(Channels::Rgba);

    assert!(org.size == compressed.size);
    assert!(org.channels == compressed.channels);
    assert!(org.data.len() == compressed.data.len());
    let width = org.size.width as usize;
    let height = org.size.height as usize;

    fn calculate_mse<T, F>(org: &[T], compressed: &[T], get_value: F) -> f64
    where
        T: Copy,
        F: Fn(T) -> f64,
    {
        let mut mse = 0.0;
        for (&o, &c) in org.iter().zip(compressed.iter()) {
            let diff = get_value(o) - get_value(c);
            mse += diff * diff;
        }
        mse /= org.len() as f64;
        mse
    }
    fn box_blur<T, F>(image: &[T], width: usize, height: usize, get_value: F) -> Vec<f64>
    where
        T: Copy,
        F: Fn(T) -> f64,
    {
        let mut blurred: Vec<f64> = image.iter().map(|&x| get_value(x)).collect();

        const GAUSS_WEIGHTS: [f64; 5] = {
            let raw = [1.0, 4.0, 6.0, 4.0, 1.0];
            let sum = raw[0] + raw[1] + raw[2] + raw[3] + raw[4];
            [
                raw[0] / sum,
                raw[1] / sum,
                raw[2] / sum,
                raw[3] / sum,
                raw[4] / sum,
            ]
        };
        fn weigh(values: [f64; 5]) -> f64 {
            values[0] * GAUSS_WEIGHTS[0]
                + values[1] * GAUSS_WEIGHTS[1]
                + values[2] * GAUSS_WEIGHTS[2]
                + values[3] * GAUSS_WEIGHTS[3]
                + values[4] * GAUSS_WEIGHTS[4]
        }

        // Pass 1: horizontal
        for y in 0..height {
            let index_base = y * width;
            let mut prev_prev = blurred[index_base];
            let mut prev = blurred[index_base];
            let last_index = index_base + width - 1;
            for x in 0..width {
                let current = blurred[index_base + x];
                let next = blurred[last_index.min(index_base + x + 1)];
                let next_next = blurred[last_index.min(index_base + x + 2)];
                let sum = weigh([prev_prev, prev, current, next, next_next]);
                prev_prev = prev;
                prev = current;
                blurred[index_base + x] = sum;
            }
        }

        // Pass 2: vertical
        for x in 0..width {
            let mut prev_prev = blurred[x];
            let mut prev = blurred[x];
            let last_index = (height - 1) * width + x;
            for y in 0..height {
                let index = y * width + x;
                let current = blurred[index];
                let next = blurred[last_index.min(index + width)];
                let next_next = blurred[last_index.min(index + 2 * width)];
                let sum = weigh([prev_prev, prev, current, next, next_next]);
                prev_prev = prev;
                prev = current;
                blurred[index] = sum;
            }
        }

        blurred
    }
    fn calculate_metrics<T, F>(
        org: &[T],
        compressed: &[T],
        width: usize,
        height: usize,
        channel: MetricChannel,
        get_value: F,
    ) -> Metrics
    where
        T: Copy,
        F: Copy + Fn(T) -> f64,
    {
        // PSNR
        let mse = calculate_mse(org, compressed, get_value);

        // blurred PSNR
        let blurred_org = box_blur(org, width, height, get_value);
        let blurred_compressed = box_blur(compressed, width, height, get_value);
        let mse_blur = calculate_mse(&blurred_org, &blurred_compressed, |x| x);

        // region error is just the absolute average error per 4x4 region
        const REGION_SIZE: usize = 4;
        let mut region_error = 0.0;
        for region_y in 0..height / REGION_SIZE {
            for region_x in 0..width / REGION_SIZE {
                let mut region = 0.0;
                for y in 0..REGION_SIZE {
                    for x in 0..REGION_SIZE {
                        let i = (region_y * REGION_SIZE + y) * width + region_x * REGION_SIZE + x;
                        let diff = get_value(org[i]) - get_value(compressed[i]);
                        region += diff;
                    }
                }
                region_error += region.abs() / (REGION_SIZE * REGION_SIZE) as f64;
            }
        }
        region_error /= (width / REGION_SIZE * height / REGION_SIZE) as f64;

        Metrics {
            channel,
            mse,
            mse_blur,
            region_error,
        }
    }

    #[allow(clippy::excessive_precision)]
    fn rgb_to_l(r: f32, g: f32, b: f32) -> f32 {
        // OKLab
        fn srgb_to_linear(c: f32) -> f32 {
            if c >= 0.04045 {
                ((c + 0.055) / 1.055).powf(2.4)
            } else {
                c / 12.92
            }
        }
        fn cbrt(x: f32) -> f32 {
            // This is the fast cbrt approximation from the oklab crate.
            // Source: https://gitlab.com/kornelski/oklab/-/blob/d3c074f154187dd5c0642119a6402a6c0753d70c/oklab/src/lib.rs#L61
            // Author: Kornel (https://gitlab.com/kornelski/)
            const B: u32 = 709957561;
            const C: f32 = 5.4285717010e-1;
            const D: f32 = -7.0530611277e-1;
            const E: f32 = 1.4142856598e+0;
            const F: f32 = 1.6071428061e+0;
            const G: f32 = 3.5714286566e-1;

            let mut t = f32::from_bits((x.to_bits() / 3).wrapping_add(B));
            let s = C + (t * t) * (t / x);
            t *= G + F / (s + E + D / s);
            t
        }

        let [r, g, b] = [r, g, b].map(srgb_to_linear);

        let mut l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
        let mut m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
        let mut s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

        l = cbrt(l);
        m = cbrt(m);
        s = cbrt(s);

        l * 0.2104542553 + m * 0.7936177850 + s * -0.0040720468
    }

    // compute metrics
    let mut metrics: Vec<Metrics> = Vec::new();

    let org: &[[f32; 4]] = cast_slice(&org.data);
    let compressed: &[[f32; 4]] = cast_slice(&compressed.data);
    let calc = |ch: MetricChannel| -> Metrics {
        macro_rules! calc {
            ($f:expr) => {
                calculate_metrics(org, compressed, width, height, ch, $f)
            };
        }

        match ch {
            MetricChannel::R => calc!(|x| x[0] as f64),
            MetricChannel::G => calc!(|x| x[1] as f64),
            MetricChannel::B => calc!(|x| x[2] as f64),
            MetricChannel::A => calc!(|x| x[3] as f64),
            MetricChannel::Gray => {
                calc!(|[r, g, b, _]| (r as f64 + g as f64 + b as f64) * (1.0 / 3.0))
            }
            MetricChannel::L => calc!(|[r, g, b, a]| (rgb_to_l(r, g, b) * a) as f64),
            MetricChannel::C => unreachable!(),
        }
    };

    for ch in [
        MetricChannel::L,
        MetricChannel::C,
        MetricChannel::R,
        MetricChannel::G,
        MetricChannel::B,
        MetricChannel::A,
        MetricChannel::Gray,
    ] {
        // skip channels already covered
        if metrics.iter().any(|m| m.channel == ch) {
            continue;
        }

        if channels.contains(ch.into()) {
            if ch == MetricChannel::C {
                // special case: color = RGB
                let r = calc(MetricChannel::R);
                let g = calc(MetricChannel::G);
                let b = calc(MetricChannel::B);
                metrics.push(Metrics {
                    channel: MetricChannel::C,
                    mse: (r.mse + g.mse + b.mse) / 3.0,
                    mse_blur: (r.mse_blur + g.mse_blur + b.mse_blur) / 3.0,
                    region_error: (r.region_error + g.region_error + b.region_error) / 3.0,
                });
                if channels.contains(MetricChannel::R.into()) {
                    metrics.push(r);
                }
                if channels.contains(MetricChannel::G.into()) {
                    metrics.push(g);
                }
                if channels.contains(MetricChannel::B.into()) {
                    metrics.push(b);
                }
            } else {
                metrics.push(calc(ch));
            }
        }
    }

    metrics
}
