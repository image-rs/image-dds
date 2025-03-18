use crate::util::closure_types3;
use crate::{yuv10, yuv16, yuv8, WithPrecision};
use crate::{Channels::*, ColorFormat};

use super::read_write::{
    for_each_bi_planar, for_each_bi_planar_rect, process_bi_planar_helper, BiPlaneInfo, PlaneRange,
};
use super::{Args, DecoderSet, DirectDecoder, RArgs};

// helpers

macro_rules! underlying {
    ($channels:expr, $out:ty, $p1:ty, $p2:ty, $f:expr) => {{
        const CHANNELS: usize = $channels.count() as usize;
        type OutPixel = [$out; CHANNELS];
        type Plane1 = $p1;
        type Plane2 = $p2;

        const INFO: BiPlaneInfo = BiPlaneInfo {
            plane1_element_size: std::mem::size_of::<Plane1>() as u8,
            plane2_element_size: std::mem::size_of::<Plane2>() as u8,
            sub_sampling: (2, 2),
        };
        const SUB_SAMPLING_X: usize = INFO.sub_sampling.0 as usize;

        fn process_bi_planar(plane1: &[u8], plane2: &[u8], decoded: &mut [u8], range: PlaneRange) {
            let f = closure_types3::<
                [Plane1; SUB_SAMPLING_X],
                Plane2,
                u8,
                [OutPixel; SUB_SAMPLING_X],
                _,
            >($f);
            process_bi_planar_helper(plane1, plane2, decoded, range, f)
        }

        const NATIVE_COLOR: ColorFormat =
            ColorFormat::new($channels, <$out as WithPrecision>::PRECISION);

        DirectDecoder::new_with_all_channels(
            NATIVE_COLOR,
            |Args(r, out, context)| {
                for_each_bi_planar(r, out, context, NATIVE_COLOR, INFO, process_bi_planar)
            },
            |RArgs(r, out, row_pitch, rect, context)| {
                for_each_bi_planar_rect(
                    r,
                    out,
                    row_pitch,
                    context,
                    rect,
                    NATIVE_COLOR,
                    INFO,
                    process_bi_planar,
                )
            },
        )
    }};
}

macro_rules! rgb {
    ($out:ty, p1 = $p1:ty, p2 = $p2:ty, $f:expr) => {
        underlying!(Rgb, $out, $p1, $p2, $f)
    };
}

// decoders

pub(crate) const NV12: DecoderSet = DecoderSet::new(&[
    rgb!(u8, p1 = u8, p2 = [u8; 2], |y, [u, v], _| y
        .map(|y| yuv8::n8([y, u, v]))),
    rgb!(u16, p1 = u8, p2 = [u8; 2], |y, [u, v], _| y
        .map(|y| yuv8::n16([y, u, v]))),
    rgb!(f32, p1 = u8, p2 = [u8; 2], |y, [u, v], _| y
        .map(|y| yuv8::f32([y, u, v]))),
]);

fn to10(yuv: [u16; 3]) -> [u16; 3] {
    yuv.map(|v| v >> 6)
}
pub(crate) const P010: DecoderSet = DecoderSet::new(&[
    rgb!(u16, p1 = u16, p2 = [u16; 2], |y, [u, v], _| y
        .map(|y| yuv10::n16(to10([y, u, v])))),
    rgb!(u8, p1 = u16, p2 = [u16; 2], |y, [u, v], _| y
        .map(|y| yuv10::n8(to10([y, u, v])))),
    rgb!(f32, p1 = u16, p2 = [u16; 2], |y, [u, v], _| y
        .map(|y| yuv10::f32(to10([y, u, v])))),
]);

pub(crate) const P016: DecoderSet = DecoderSet::new(&[
    rgb!(u16, p1 = u16, p2 = [u16; 2], |y, [u, v], _| y
        .map(|y| yuv16::n16([y, u, v]))),
    rgb!(u8, p1 = u16, p2 = [u16; 2], |y, [u, v], _| y
        .map(|y| yuv16::n8([y, u, v]))),
    rgb!(f32, p1 = u16, p2 = [u16; 2], |y, [u, v], _| y
        .map(|y| yuv16::f32([y, u, v]))),
]);
