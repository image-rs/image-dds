// helpers

use crate::{
    cast::{self, ToLe},
    convert_to_rgba_f32, yuv10, yuv16, yuv8, ColorFormatSet, EncodeError, SizeMultiple,
};

use super::{
    encoder::{Args, Encoder, EncoderSet, Flags},
    EncodeOptions,
};

fn bi_planar_universal<P1: ToLe + cast::Castable + Default + Copy, P2: ToLe + cast::Castable>(
    args: Args,
    encode_macro_pixel: fn([[f32; 4]; 4], &EncodeOptions) -> ([P1; 4], P2),
) -> Result<(), EncodeError> {
    const BLOCK_WIDTH: usize = 2;
    const BLOCK_HEIGHT: usize = 2;

    let Args {
        data,
        color,
        writer,
        width,
        height,
        options,
        ..
    } = args;
    let bytes_per_pixel = color.bytes_per_pixel() as usize;

    if width % BLOCK_WIDTH != 0 || height % BLOCK_HEIGHT != 0 {
        return Err(EncodeError::InvalidSize(SizeMultiple::new(
            BLOCK_WIDTH as u8,
            BLOCK_HEIGHT as u8,
        )));
    }

    let mut intermediate_buffer = vec![[0_f32; 4]; width * BLOCK_HEIGHT];
    let mut plane1_buffer = vec![P1::default(); width * BLOCK_HEIGHT];
    let mut plane2: Vec<P2> = Vec::new();

    let row_pitch = width * bytes_per_pixel;
    for line_group in data.chunks(row_pitch * BLOCK_HEIGHT) {
        debug_assert!(line_group.len() % row_pitch == 0);
        let rows_in_group = line_group.len() / row_pitch;

        // fill the intermediate buffer
        convert_to_rgba_f32(
            color,
            line_group,
            &mut intermediate_buffer[..rows_in_group * width],
        );

        // handle full blocks
        for macro_x in 0..width / BLOCK_WIDTH {
            let mut block = [[0_f32; 4]; 4];
            for y in 0..BLOCK_HEIGHT {
                for x in 0..BLOCK_WIDTH {
                    block[y * BLOCK_WIDTH + x] =
                        intermediate_buffer[y * width + macro_x * BLOCK_WIDTH + x];
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
        writer.write_all(cast::as_bytes(&plane1_buffer[..rows_in_group * width]))?;
    }

    P2::to_le(&mut plane2);
    writer.write_all(cast::as_bytes(&plane2))?;

    Ok(())
}

// encoders

pub const NV12: EncoderSet = EncoderSet::new(&[Encoder {
    color_formats: ColorFormatSet::ALL,
    flags: Flags::empty(),
    encode: |args| {
        bi_planar_universal(args, |block, _| {
            let block_yuv = block.map(|[r, g, b, _]| yuv8::from_rgb_f32([r, g, b]));

            let block_y = block_yuv.map(|yuv| yuv[0]);
            let u = block_yuv.iter().map(|yuv| yuv[1] as u16).sum::<u16>() / 4;
            let v = block_yuv.iter().map(|yuv| yuv[2] as u16).sum::<u16>() / 4;

            (block_y, [u as u8, v as u8])
        })
    },
}]);

pub const P010: EncoderSet = EncoderSet::new(&[Encoder {
    color_formats: ColorFormatSet::ALL,
    flags: Flags::empty(),
    encode: |args| {
        bi_planar_universal(args, |block, _| {
            let block_yuv = block.map(|[r, g, b, _]| yuv10::from_rgb_f32([r, g, b]));

            let block_y = block_yuv.map(|yuv| yuv[0] << 6);
            let u = block_yuv.iter().map(|yuv| yuv[1]).sum::<u16>() / 4;
            let v = block_yuv.iter().map(|yuv| yuv[2]).sum::<u16>() / 4;

            (block_y, [u << 6, v << 6])
        })
    },
}]);

pub const P016: EncoderSet = EncoderSet::new(&[Encoder {
    color_formats: ColorFormatSet::ALL,
    flags: Flags::empty(),
    encode: |args| {
        bi_planar_universal(args, |block, _| {
            let block_yuv = block.map(|[r, g, b, _]| yuv16::from_rgb_f32([r, g, b]));

            let block_y = block_yuv.map(|yuv| yuv[0]);
            let u = block_yuv.iter().map(|yuv| yuv[1] as u32).sum::<u32>() / 4;
            let v = block_yuv.iter().map(|yuv| yuv[2] as u32).sum::<u32>() / 4;

            (block_y, [u as u16, v as u16])
        })
    },
}]);
