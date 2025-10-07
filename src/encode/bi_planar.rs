// helpers

use std::num::NonZeroU32;

use super::{
    encoder::{Args, Encoder, EncoderSet},
    EncodeOptions,
};
use crate::{
    cast::{self, ToLe},
    encode::write_util::for_each_f32_rgba_rows,
    util, yuv10, yuv16, yuv8, EncodingError,
};

#[allow(clippy::type_complexity)]
fn bi_planar_universal<P1: ToLe + cast::Castable + Default + Copy, P2: ToLe + cast::Castable>(
    args: Args,
    encode_macro_pixel: fn([[f32; 4]; 4], &EncodeOptions) -> ([P1; 4], P2),
) -> Result<(), EncodingError> {
    const BLOCK_WIDTH: usize = 2;
    const BLOCK_HEIGHT: usize = 2;

    let Args {
        image,
        writer,
        options,
        progress,
        ..
    } = args;
    let width = image.width() as usize;
    let height = image.height() as usize;

    if width % BLOCK_WIDTH != 0 || height % BLOCK_HEIGHT != 0 {
        return Err(EncodingError::InvalidSize(
            NonZeroU32::new(BLOCK_WIDTH as u32).unwrap(),
            NonZeroU32::new(BLOCK_HEIGHT as u32).unwrap(),
        ));
    }

    let mut plane1_buffer = vec![P1::default(); width * BLOCK_HEIGHT];
    let mut plane2: Vec<P2> = Vec::new();

    let line_group_count = util::div_ceil(height, BLOCK_HEIGHT);
    let report_frequency = util::div_ceil(1024 * 1024, width * BLOCK_HEIGHT);
    let mut group_index = 0;
    for_each_f32_rgba_rows(image, BLOCK_HEIGHT, |rows| -> Result<(), EncodingError> {
        // occasionally report progress
        progress.checked_report_if(
            group_index % report_frequency == 0,
            group_index as f32 / line_group_count as f32,
        )?;
        group_index += 1;

        // handle full blocks
        for macro_x in 0..width / BLOCK_WIDTH {
            let mut block = [[0_f32; 4]; 4];
            for y in 0..BLOCK_HEIGHT {
                for x in 0..BLOCK_WIDTH {
                    block[y * BLOCK_WIDTH + x] = rows[y * width + macro_x * BLOCK_WIDTH + x];
                }
            }

            let (p1, p2) = encode_macro_pixel(block, &options);

            for y in 0..BLOCK_HEIGHT {
                for x in 0..BLOCK_WIDTH {
                    plane1_buffer[y * width + macro_x * BLOCK_WIDTH + x] = p1[y * BLOCK_WIDTH + x];
                }
            }
            plane2.push(p2);
        }

        P1::to_le(&mut plane1_buffer);
        writer.write_all(cast::as_bytes(&plane1_buffer))?;

        Ok(())
    })?;

    progress.check_cancelled()?;
    P2::to_le(&mut plane2);
    writer.write_all(cast::as_bytes(&plane2))?;

    Ok(())
}

// encoders

pub(crate) const NV12: EncoderSet = EncoderSet::new_bi_planar(&[Encoder::new_universal(|args| {
    bi_planar_universal(args, |block, _| {
        let block_yuv = block.map(|[r, g, b, _]| yuv8::from_rgb_f32([r, g, b]));

        let block_y = block_yuv.map(|yuv| yuv[0]);
        let u = block_yuv.iter().map(|yuv| yuv[1] as u16).sum::<u16>() / 4;
        let v = block_yuv.iter().map(|yuv| yuv[2] as u16).sum::<u16>() / 4;

        (block_y, [u as u8, v as u8])
    })
})]);

pub(crate) const P010: EncoderSet = EncoderSet::new_bi_planar(&[Encoder::new_universal(|args| {
    bi_planar_universal(args, |block, _| {
        let block_yuv = block.map(|[r, g, b, _]| yuv10::from_rgb_f32([r, g, b]));

        let block_y = block_yuv.map(|yuv| yuv[0] << 6);
        let u = block_yuv.iter().map(|yuv| yuv[1]).sum::<u16>() / 4;
        let v = block_yuv.iter().map(|yuv| yuv[2]).sum::<u16>() / 4;

        (block_y, [u << 6, v << 6])
    })
})]);

pub(crate) const P016: EncoderSet = EncoderSet::new_bi_planar(&[Encoder::new_universal(|args| {
    bi_planar_universal(args, |block, _| {
        let block_yuv = block.map(|[r, g, b, _]| yuv16::from_rgb_f32([r, g, b]));

        let block_y = block_yuv.map(|yuv| yuv[0]);
        let u = block_yuv.iter().map(|yuv| yuv[1] as u32).sum::<u32>() / 4;
        let v = block_yuv.iter().map(|yuv| yuv[2] as u32).sum::<u32>() / 4;

        (block_y, [u as u16, v as u16])
    })
})]);
