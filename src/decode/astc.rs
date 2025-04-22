use super::read_write::{
    for_each_block_rect_untyped, for_each_block_untyped, general_process_blocks, PixelRange,
};
use super::{Args, Decoder, DecoderSet, RArgs};
use crate::{Channels, ColorFormat, NormConvert, WithPrecision};

// helpers

fn decode_astc_block<const PIXELS: usize, T: Default + Copy>(
    block_size: (usize, usize),
) -> impl Fn([u8; 16]) -> [[T; 4]; PIXELS]
where
    u8: NormConvert<T>,
{
    // The inner function isn't generic over PIXELS. This brings down the
    // number of instantiations of astc_decode::astc_decode_block from 14*3=42
    // to 3. This saves around 110KiB in the final binary.
    fn decode_into<T>(bytes: &[u8; 16], footprint: astc_decode::Footprint, out: &mut [[T; 4]])
    where
        u8: NormConvert<T>,
    {
        debug_assert_eq!(
            footprint.block_width() as usize * footprint.block_height() as usize,
            out.len()
        );

        let width = footprint.block_width() as usize;
        astc_decode::astc_decode_block(bytes, footprint, |x, y, [r, g, b, a]| {
            out[y as usize * width + x as usize] = [
                NormConvert::to(r),
                NormConvert::to(g),
                NormConvert::to(b),
                NormConvert::to(a),
            ];
        });
    }

    debug_assert_eq!(PIXELS, block_size.0 * block_size.1);
    let footprint = astc_decode::Footprint::new(block_size.0 as u32, block_size.1 as u32);

    move |bytes| {
        let mut block = [[T::default(); 4]; PIXELS];
        decode_into(&bytes, footprint, &mut block);
        block
    }
}

macro_rules! astc_decoder {
    ($out:ty, $block_w:literal, $block_h:literal) => {{
        const BLOCK_WIDTH: usize = $block_w;
        const BLOCK_HEIGHT: usize = $block_h;
        const BLOCK_PIXELS: usize = BLOCK_WIDTH * BLOCK_HEIGHT;
        const BYTES_PER_BLOCK: usize = 16;
        const CHANNELS: usize = 4;
        type OutPixel = [$out; CHANNELS];

        fn process_blocks(
            encoded_blocks: &[u8],
            decoded: &mut [u8],
            stride: usize,
            range: PixelRange,
        ) {
            let f = decode_astc_block::<BLOCK_PIXELS, $out>((BLOCK_WIDTH, BLOCK_HEIGHT));
            general_process_blocks::<
                BLOCK_WIDTH,
                BLOCK_HEIGHT,
                BLOCK_PIXELS,
                BYTES_PER_BLOCK,
                OutPixel,
            >(encoded_blocks, decoded, stride, range, f)
        }

        const NATIVE_COLOR: ColorFormat =
            ColorFormat::new(Channels::Rgba, <$out as WithPrecision>::PRECISION);

        Decoder::new_with_all_channels(
            NATIVE_COLOR,
            |Args(r, out, context)| {
                for_each_block_untyped::<BLOCK_WIDTH, BLOCK_HEIGHT, BYTES_PER_BLOCK, OutPixel>(
                    r,
                    out,
                    context,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
            |RArgs(r, out, row_pitch, rect, context)| {
                for_each_block_rect_untyped::<BLOCK_WIDTH, BLOCK_HEIGHT, BYTES_PER_BLOCK>(
                    r,
                    out,
                    row_pitch,
                    context,
                    rect,
                    NATIVE_COLOR,
                    process_blocks,
                )
            },
        )
    }};
}

macro_rules! astc {
    ($w:literal, $h:literal) => {
        DecoderSet::new(&[
            astc_decoder!(u8, $w, $h),
            astc_decoder!(u16, $w, $h),
            astc_decoder!(f32, $w, $h),
        ])
    };
}

// decoders

pub(crate) const ASTC_4X4_UNORM: DecoderSet = astc!(4, 4);
pub(crate) const ASTC_5X4_UNORM: DecoderSet = astc!(5, 4);
pub(crate) const ASTC_5X5_UNORM: DecoderSet = astc!(5, 5);
pub(crate) const ASTC_6X5_UNORM: DecoderSet = astc!(6, 5);
pub(crate) const ASTC_6X6_UNORM: DecoderSet = astc!(6, 6);
pub(crate) const ASTC_8X5_UNORM: DecoderSet = astc!(8, 5);
pub(crate) const ASTC_8X6_UNORM: DecoderSet = astc!(8, 6);
pub(crate) const ASTC_8X8_UNORM: DecoderSet = astc!(8, 8);
pub(crate) const ASTC_10X5_UNORM: DecoderSet = astc!(10, 5);
pub(crate) const ASTC_10X6_UNORM: DecoderSet = astc!(10, 6);
pub(crate) const ASTC_10X8_UNORM: DecoderSet = astc!(10, 8);
pub(crate) const ASTC_10X10_UNORM: DecoderSet = astc!(10, 10);
pub(crate) const ASTC_12X10_UNORM: DecoderSet = astc!(12, 10);
pub(crate) const ASTC_12X12_UNORM: DecoderSet = astc!(12, 12);
