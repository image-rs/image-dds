// helpers

use crate::{cast, ch, convert_to_rgba_f32, util, ColorFormatSet};

use super::{
    bc4,
    write::{BaseEncoder, Flags},
    Args, DecodedArgs, EncodeError, EncodeOptions,
};

fn block_universal<
    const BLOCK_WIDTH: usize,
    const BLOCK_HEIGHT: usize,
    const BLOCK_BYTES: usize,
>(
    args: Args,
    encode_block: fn(&[[f32; 4]], usize, &EncodeOptions, &mut [u8; BLOCK_BYTES]),
) -> Result<(), EncodeError> {
    let DecodedArgs {
        data,
        color,
        writer,
        width,
        options,
        ..
    } = DecodedArgs::from(args)?;
    let bytes_per_pixel = color.bytes_per_pixel() as usize;

    let mut intermediate_buffer = vec![[0_f32; 4]; width * BLOCK_HEIGHT];
    let mut encoded_buffer = vec![[0_u8; BLOCK_BYTES]; util::div_ceil(width, BLOCK_WIDTH)];

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
        for i in 0..(BLOCK_HEIGHT - rows_in_group) {
            // copy the first line to fill the rest
            intermediate_buffer.copy_within(..width, (rows_in_group + i) * width);
        }

        // handle full blocks
        for block_index in 0..width / BLOCK_WIDTH {
            let block_start = block_index * BLOCK_WIDTH;
            let block = &intermediate_buffer[block_start..];
            let encoded = &mut encoded_buffer[block_index];

            encode_block(block, width, &options, encoded);
        }

        // handle last partial block
        if width % BLOCK_WIDTH != 0 {
            let block_index = width / BLOCK_WIDTH;
            let block_start = block_index * BLOCK_WIDTH;
            let block_width = width - block_start;

            // fill block data
            let mut block_data = vec![[0_f32; 4]; BLOCK_WIDTH * BLOCK_HEIGHT];
            for i in 0..BLOCK_HEIGHT {
                let row = &mut block_data[i * BLOCK_WIDTH..(i + 1) * BLOCK_WIDTH];
                let partial_row = &intermediate_buffer[block_start + i * width..][..block_width];
                row[..block_width].copy_from_slice(partial_row);
                let last = partial_row.last().copied().unwrap_or_default();
                row[block_width..].fill(last);
            }

            let encoded = &mut encoded_buffer[block_index];
            encode_block(&block_data, BLOCK_WIDTH, &options, encoded);
        }

        writer.write_all(cast::as_bytes(&encoded_buffer))?;
    }

    Ok(())
}

fn get_4x4_grayscale(data: &[[f32; 4]], row_pitch: usize) -> [f32; 16] {
    let mut block = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            block[i * 4 + j] = ch::rgba_to_grayscale(data[i * row_pitch + j])[0];
        }
    }
    block
}

fn get_4x4_select_channel<const CHANNEL: usize>(data: &[[f32; 4]], row_pitch: usize) -> [f32; 16] {
    let mut block = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            block[i * 4 + j] = data[i * row_pitch + j][CHANNEL];
        }
    }
    block
}

// encoders

fn handle_bc4(data: &[[f32; 4]], row_pitch: usize, options: bc4::Bc4Options) -> [u8; 8] {
    let block = get_4x4_grayscale(data, row_pitch);
    bc4::compress_bc4_block(block, options)
}

pub const BC4_UNORM: &[BaseEncoder] = &[BaseEncoder {
    color_formats: ColorFormatSet::ALL,
    flags: Flags::DITHER_COLOR,
    encode: |args| {
        block_universal::<4, 4, 8>(args, |data, row_pitch, options, out| {
            let options = bc4::Bc4Options {
                dither: options.dither.color(),
                snorm: false,
            };
            *out = handle_bc4(data, row_pitch, options);
        })
    },
}];

pub const BC4_SNORM: &[BaseEncoder] = &[BaseEncoder {
    color_formats: ColorFormatSet::ALL,
    flags: Flags::DITHER_COLOR,
    encode: |args| {
        block_universal::<4, 4, 8>(args, |data, row_pitch, options, out| {
            let options = bc4::Bc4Options {
                dither: options.dither.color(),
                snorm: true,
            };
            *out = handle_bc4(data, row_pitch, options);
        })
    },
}];

fn handle_bc5(data: &[[f32; 4]], row_pitch: usize, options: bc4::Bc4Options) -> [u8; 16] {
    let red_block = get_4x4_select_channel::<0>(data, row_pitch);
    let green_block = get_4x4_select_channel::<1>(data, row_pitch);

    let red = bc4::compress_bc4_block(red_block, options);
    let green = bc4::compress_bc4_block(green_block, options);

    let mut out = [0; 16];
    out[0..8].copy_from_slice(&red);
    out[8..].copy_from_slice(&green);
    out
}

pub const BC5_UNORM: &[BaseEncoder] = &[BaseEncoder {
    color_formats: ColorFormatSet::ALL,
    flags: Flags::DITHER_COLOR,
    encode: |args| {
        block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
            let options = bc4::Bc4Options {
                dither: options.dither.color(),
                snorm: false,
            };
            *out = handle_bc5(data, row_pitch, options);
        })
    },
}];

pub const BC5_SNORM: &[BaseEncoder] = &[BaseEncoder {
    color_formats: ColorFormatSet::ALL,
    flags: Flags::DITHER_COLOR,
    encode: |args| {
        block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
            let options = bc4::Bc4Options {
                dither: options.dither.color(),
                snorm: true,
            };
            *out = handle_bc5(data, row_pitch, options);
        })
    },
}];
