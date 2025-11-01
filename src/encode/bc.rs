// helpers

use glam::Vec4;

use crate::{
    cast, ch,
    encode::{bc7::Bc7Modes, bcn_util::Quantized, write_util::for_each_f32_rgba_rows},
    n4,
    util::{self, clamp_0_1},
    Dithering, EncodingError,
};

use super::{
    bc1, bc4, bc7, bcn_util,
    encoder::{Args, Encoder, EncoderSet, Flags},
    CompressionQuality, EncodeOptions, ErrorMetric, PreferredFragmentSize,
};

fn block_universal<
    const BLOCK_WIDTH: usize,
    const BLOCK_HEIGHT: usize,
    const BLOCK_BYTES: usize,
>(
    args: Args,
    encode_block: fn(&[[f32; 4]], usize, &EncodeOptions, &mut [u8; BLOCK_BYTES]),
) -> Result<(), EncodingError> {
    let Args {
        image,
        writer,
        options,
        progress,
        ..
    } = args;
    let width = image.width() as usize;
    let height = image.height() as usize;

    let mut encoded_buffer = vec![[0_u8; BLOCK_BYTES]; util::div_ceil(width, BLOCK_WIDTH)];

    // Report frequencies were chosen manually.
    // I just tried to pick frequencies such that every quality level reports
    // <50 times per second in non-parallel mode. These numbers may need to be
    // adjusted later as algorithms improve and get faster.
    let report_frequency = match options.quality {
        CompressionQuality::Fast => 8192,
        CompressionQuality::Normal => 4096,
        CompressionQuality::High => 2048,
        CompressionQuality::Unreasonable => 256,
    };
    let block_count = util::div_ceil(width, BLOCK_WIDTH) * util::div_ceil(height, BLOCK_HEIGHT);
    let mut block_index: usize = 0;
    let mut report_block = || -> Result<(), EncodingError> {
        progress.checked_report_if(
            block_index % report_frequency == 0,
            block_index as f32 / block_count as f32,
        )?;
        block_index += 1;
        Ok(())
    };

    for_each_f32_rgba_rows(image, BLOCK_HEIGHT, |rows| {
        // handle full blocks
        #[allow(clippy::needless_range_loop)]
        for block_index in 0..width / BLOCK_WIDTH {
            let block_start = block_index * BLOCK_WIDTH;
            let block = &rows[block_start..];
            let encoded = &mut encoded_buffer[block_index];

            encode_block(block, width, &options, encoded);
            report_block()?;
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
                let partial_row = &rows[block_start + i * width..][..block_width];
                row[..block_width].copy_from_slice(partial_row);
                let last = partial_row.last().copied().unwrap_or_default();
                row[block_width..].fill(last);
            }

            let encoded = &mut encoded_buffer[block_index];
            encode_block(&block_data, BLOCK_WIDTH, &options, encoded);
            report_block()?;
        }

        writer.write_all(cast::as_bytes(&encoded_buffer))?;

        Ok(())
    })
}

fn get_4x4_rgba(data: &[[f32; 4]], row_pitch: usize) -> [[f32; 4]; 16] {
    let mut block: [[f32; 4]; 16] = [[0.0; 4]; 16];
    for i in 0..4 {
        for j in 0..4 {
            block[i * 4 + j] = data[i * row_pitch + j];
        }
    }
    block
}
fn get_4x4_rgba_vec4(data: &[[f32; 4]], row_pitch: usize) -> [Vec4; 16] {
    let mut block: [Vec4; 16] = [Vec4::ZERO; 16];
    for i in 0..4 {
        for j in 0..4 {
            block[i * 4 + j] = Vec4::from_array(data[i * row_pitch + j]);
        }
    }
    block
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

fn get_alpha(data: &[[f32; 4]; 16]) -> [f32; 16] {
    data.map(|x| x[3])
}
fn pre_multiply_alpha(data: &mut [[f32; 4]]) {
    for pixel in data.iter_mut() {
        let [r, g, b, a] = pixel.map(clamp_0_1);
        pixel[0] = r * a;
        pixel[1] = g * a;
        pixel[2] = b * a;
        pixel[3] = a;
    }
}

fn concat_blocks(left: [u8; 8], right: [u8; 8]) -> [u8; 16] {
    let mut out = [0; 16];
    out[0..8].copy_from_slice(&left);
    out[8..].copy_from_slice(&right);
    out
}

const BC1_FRAGMENT_SIZE: PreferredFragmentSize =
    PreferredFragmentSize::new(64 * 64, 16 * 16, 16 * 16);
const BC3_FRAGMENT_SIZE: PreferredFragmentSize = BC1_FRAGMENT_SIZE.combine(BC4_FRAGMENT_SIZE);
const BC4_FRAGMENT_SIZE: PreferredFragmentSize =
    PreferredFragmentSize::new(64 * 64, 32 * 32, 8 * 8);
// TODO: adjust this later
const BC7_FRAGMENT_SIZE: PreferredFragmentSize =
    PreferredFragmentSize::new(16 * 16, 16 * 16, 16 * 16);

// encoders

fn get_bc1_options(options: &EncodeOptions) -> bc1::Bc1Options {
    bc1::Bc1Options {
        dither: options.dithering.color(),
        perceptual: options.error_metric == ErrorMetric::Perceptual,
        opaque_always_p4: options.quality <= CompressionQuality::Normal,
        fit_optimal: options.quality >= CompressionQuality::Normal,
        refine: options.quality >= CompressionQuality::Normal,
        refine_max_iter: match options.quality {
            CompressionQuality::Fast => 0,
            CompressionQuality::Normal => 0,
            CompressionQuality::High => 4,
            CompressionQuality::Unreasonable => 10,
        },
        quantization: match options.quality {
            CompressionQuality::Fast => bcn_util::Quantization::ChannelWiseOptimized,
            _ => bcn_util::Quantization::ChannelWise,
        },
        ..bc1::Bc1Options::default()
    }
}
pub(crate) const BC1_UNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 8>(args, |data, row_pitch, options, out| {
        let bc1_options = get_bc1_options(options);
        let mut block = get_4x4_rgba(data, row_pitch);

        if options.dithering.alpha() {
            let alpha = get_alpha(&block);
            bcn_util::block_dither(&alpha, |i, pixel| {
                let alpha = if pixel >= 0.5 { 1.0 } else { 0.0 };
                block[i][3] = alpha;
                alpha
            });
        }

        *out = bc1::compress_bc1_block(block, bc1_options);
    })
})
.add_flags(Flags::DITHER_ALL)
.with_fragment_size(BC1_FRAGMENT_SIZE)]);

fn bc2_alpha(alpha: [f32; 16], options: &EncodeOptions) -> [u8; 8] {
    let mut indexes: u64 = 0;
    let mut set_value = |i: usize, value: u8| {
        debug_assert!(value < 16);
        indexes |= (value as u64) << (i * 4);
    };

    if options.dithering.alpha() {
        bcn_util::block_dither(&alpha, |i, pixel| {
            let value = n4::from_f32(pixel);
            set_value(i, value);
            n4::f32(value)
        });
    } else {
        for (i, pixel) in alpha.into_iter().enumerate() {
            let value = n4::from_f32(pixel);
            set_value(i, value);
        }
    }

    indexes.to_le_bytes()
}

pub(crate) const BC2_UNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
        let (bc1_options, _) = get_bc3_options(options);

        let block = get_4x4_rgba(data, row_pitch);

        let alpha_block = bc2_alpha(get_alpha(&block), options);
        let bc1_block = bc1::compress_bc1_block(block, bc1_options);

        *out = concat_blocks(alpha_block, bc1_block);
    })
})
.add_flags(Flags::DITHER_ALL)
.with_fragment_size(BC1_FRAGMENT_SIZE)]);

pub(crate) const BC2_UNORM_PREMULTIPLIED_ALPHA: EncoderSet =
    EncoderSet::new_bc(&[Encoder::new_universal(|args| {
        block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
            let (bc1_options, _) = get_bc3_options(options);

            let mut block = get_4x4_rgba(data, row_pitch);
            pre_multiply_alpha(&mut block);

            let alpha_block = bc2_alpha(get_alpha(&block), options);
            let bc1_block = bc1::compress_bc1_block(block, bc1_options);

            *out = concat_blocks(alpha_block, bc1_block);
        })
    })
    .add_flags(Flags::DITHER_ALL)
    .with_fragment_size(BC1_FRAGMENT_SIZE)]);

fn get_bc3_options(options: &EncodeOptions) -> (bc1::Bc1Options, bc4::Bc4Options) {
    let mut bc1_options = get_bc1_options(options);
    bc1_options.no_p3_default = true;

    let mut bc4_options = get_bc4_options(options);
    bc4_options.snorm = false;

    (bc1_options, bc4_options)
}
pub(crate) const BC3_UNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
        let (bc1_options, bc4_options) = get_bc3_options(options);

        let block = get_4x4_rgba(data, row_pitch);

        let bc4_block = bc4::compress_bc4_block(get_alpha(&block), bc4_options);
        let bc1_block = bc1::compress_bc1_block(block, bc1_options);

        *out = concat_blocks(bc4_block, bc1_block);
    })
})
.add_flags(Flags::DITHER_ALL)
.with_fragment_size(BC3_FRAGMENT_SIZE)]);

pub(crate) const BC3_UNORM_PREMULTIPLIED_ALPHA: EncoderSet =
    EncoderSet::new_bc(&[Encoder::new_universal(|args| {
        block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
            let (bc1_options, bc4_options) = get_bc3_options(options);

            let mut block = get_4x4_rgba(data, row_pitch);
            pre_multiply_alpha(&mut block);

            let bc4_block = bc4::compress_bc4_block(get_alpha(&block), bc4_options);
            let bc1_block = bc1::compress_bc1_block(block, bc1_options);

            *out = concat_blocks(bc4_block, bc1_block);
        })
    })
    .add_flags(Flags::DITHER_ALL)
    .with_fragment_size(BC3_FRAGMENT_SIZE)]);

pub(crate) const BC3_UNORM_RXGB: EncoderSet =
    EncoderSet::new_bc(&[Encoder::new_universal(|args| {
        block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
            let (bc1_options, bc4_options) = get_bc3_options(options);

            let block_r = get_4x4_select_channel::<0>(data, row_pitch);
            let mut block_gb = get_4x4_rgba(data, row_pitch);
            block_gb.iter_mut().for_each(|pixel| {
                // It's important to set the R channel to 1 (aka 255) to be
                // compatible with NVTT's AGBR format, which just swaps the R
                // and A channels. If the R channel is 0, the block will be
                // considered transparent.
                //
                // Setting R=A would be another option, but this would result in
                // worse compressed artifacts.
                pixel[0] = 1.0;
            });

            let bc4_block = bc4::compress_bc4_block(block_r, bc4_options);
            let bc1_block = bc1::compress_bc1_block(block_gb, bc1_options);

            *out = concat_blocks(bc4_block, bc1_block);
        })
    })
    .add_flags(Flags::DITHER_COLOR)
    .with_fragment_size(BC3_FRAGMENT_SIZE)]);

pub(crate) const BC3_UNORM_NORMAL: EncoderSet =
    EncoderSet::new_bc(&[Encoder::new_universal(|args| {
        block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
            let (bc1_options, bc4_options) = get_bc3_options(options);

            let block_a = get_4x4_select_channel::<0>(data, row_pitch);
            let mut block_rgb = get_4x4_rgba(data, row_pitch);
            block_rgb.iter_mut().for_each(|pixel| {
                pixel[0] = 1.0;
                pixel[2] = 0.0;
                pixel[3] = 1.0;
            });

            let bc4_block = bc4::compress_bc4_block(block_a, bc4_options);
            let bc1_block = bc1::compress_bc1_block(block_rgb, bc1_options);

            *out = concat_blocks(bc4_block, bc1_block);
        })
    })
    .add_flags(Flags::DITHER_COLOR)
    .with_fragment_size(BC3_FRAGMENT_SIZE)]);

fn handle_bc4(data: &[[f32; 4]], row_pitch: usize, options: bc4::Bc4Options) -> [u8; 8] {
    let block = get_4x4_grayscale(data, row_pitch);
    bc4::compress_bc4_block(block, options)
}
fn get_bc4_options(options: &EncodeOptions) -> bc4::Bc4Options {
    bc4::Bc4Options {
        dither: options.dithering.color(),
        snorm: false,
        brute_force: options.quality == CompressionQuality::Unreasonable,
        use_inter4: options.quality > CompressionQuality::Fast,
        use_inter4_heuristic: true,
        high_quality_quantize: options.quality >= CompressionQuality::High,
        fast_iter: options.quality <= CompressionQuality::Normal,
        refine: options.quality >= CompressionQuality::Normal,
    }
}

pub(crate) const BC4_UNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 8>(args, |data, row_pitch, options, out| {
        let mut options = get_bc4_options(options);
        options.snorm = false;
        *out = handle_bc4(data, row_pitch, options);
    })
})
.add_flags(Flags::DITHER_COLOR)
.with_fragment_size(BC4_FRAGMENT_SIZE)]);

pub(crate) const BC4_SNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 8>(args, |data, row_pitch, options, out| {
        let mut options = get_bc4_options(options);
        options.snorm = true;
        *out = handle_bc4(data, row_pitch, options);
    })
})
.add_flags(Flags::DITHER_COLOR)
.with_fragment_size(BC4_FRAGMENT_SIZE)]);

fn handle_bc5(data: &[[f32; 4]], row_pitch: usize, options: bc4::Bc4Options) -> [u8; 16] {
    let red_block = get_4x4_select_channel::<0>(data, row_pitch);
    let green_block = get_4x4_select_channel::<1>(data, row_pitch);

    let red = bc4::compress_bc4_block(red_block, options);
    let green = bc4::compress_bc4_block(green_block, options);

    concat_blocks(red, green)
}

pub(crate) const BC5_UNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
        let mut options = get_bc4_options(options);
        options.snorm = false;
        *out = handle_bc5(data, row_pitch, options);
    })
})
.add_flags(Flags::DITHER_COLOR)
.with_fragment_size(BC4_FRAGMENT_SIZE)]);

pub(crate) const BC5_SNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
        let mut options = get_bc4_options(options);
        options.snorm = true;
        *out = handle_bc5(data, row_pitch, options);
    })
})
.add_flags(Flags::DITHER_COLOR)
.with_fragment_size(BC4_FRAGMENT_SIZE)]);

pub(crate) const BC7_UNORM: EncoderSet = EncoderSet::new_bc(&[Encoder::new_universal(|args| {
    block_universal::<4, 4, 16>(args, |data, row_pitch, options, out| {
        let block_vec: [Vec4; 16] = get_4x4_rgba_vec4(data, row_pitch);

        let block = if options.dithering == Dithering::None {
            block_vec.map(bc7::Rgba::round)
        } else {
            // This is just going to be simple 2x2 ordered dithering.
            // This is enough to raise the color depth to approx 10 bits for smooth gradients.
            let dither_color = if options.dithering.color() { 1.0 } else { 0.0 };
            let dither_alpha = if options.dithering.alpha() { 1.0 } else { 0.0 };
            let dither_mask = Vec4::new(dither_color, dither_color, dither_color, dither_alpha);

            let mut block = [bc7::Rgba::new(0, 0, 0, 0); 16];

            const DITHER_MATRIX: [f32; 4] = [-0.5, 0.0, 0.25, -0.25];

            for y in 0..4 {
                for x in 0..4 {
                    let index = y * 4 + x;
                    let pixel = block_vec[index];

                    // 2x2 ordered dithering
                    let dither_index = (x % 2) + (y % 2) * 2;
                    let dither_weight = DITHER_MATRIX[dither_index] * (1. / 255.);
                    let dither = dither_weight * dither_mask;

                    block[index] = bc7::Rgba::round(pixel + dither);
                }
            }

            block
        };

        let options = bc7::Bc7Options {
            allowed_modes: match options.quality {
                CompressionQuality::Fast => Bc7Modes::MODE0 | Bc7Modes::MODE4 | Bc7Modes::MODE6,
                CompressionQuality::Normal => {
                    Bc7Modes::MODE1
                        | Bc7Modes::MODE3
                        | Bc7Modes::MODE4
                        | Bc7Modes::MODE5
                        | Bc7Modes::MODE6
                        | Bc7Modes::MODE7
                }
                CompressionQuality::High | CompressionQuality::Unreasonable => Bc7Modes::all(),
            },
            max_partitions: 64,
            max_partition_candidates: match options.quality {
                CompressionQuality::Fast => 1,
                CompressionQuality::Normal => 1,
                CompressionQuality::High => 2,
                CompressionQuality::Unreasonable => 64,
            },
            fast_mode_0_partitions: options.quality == CompressionQuality::Fast,
            quantization: match options.quality {
                CompressionQuality::Fast => bcn_util::Quantization::ChannelWiseOptimized,
                CompressionQuality::Normal => bcn_util::Quantization::ChannelWiseOptimized,
                CompressionQuality::High => bcn_util::Quantization::ChannelWise,
                CompressionQuality::Unreasonable => bcn_util::Quantization::ChannelWise,
            },
            allow_color_rotation: true,
            max_color_rotations: match options.quality {
                CompressionQuality::Fast => 1,
                CompressionQuality::Normal => 1,
                CompressionQuality::High => 2,
                CompressionQuality::Unreasonable => 4,
            },
            fit_optimal: true,
            max_refinement_iters: match options.quality {
                CompressionQuality::Fast => 0,
                CompressionQuality::Normal => 0,
                CompressionQuality::High => 1,
                CompressionQuality::Unreasonable => 8,
            },
            max_p_bit_combinations: match options.quality {
                CompressionQuality::Fast => 1,
                CompressionQuality::Normal => 1,
                CompressionQuality::High => 2,
                CompressionQuality::Unreasonable => 4,
            },
            force_modes: Bc7Modes::empty(),
        };

        *out = bc7::compress_bc7_block(block, options);
    })
})
.add_flags(Flags::DITHER_ALL)
.with_fragment_size(BC7_FRAGMENT_SIZE)]);
