use crate::{
    decode::bcn_util::{BitStream, Indexes, PARTITION_SET_2, PARTITION_SET_3},
    util::unlikely_branch,
};

pub(crate) fn decode_bc7_block(block: [u8; 16]) -> [[u8; 4]; 16] {
    let mut stream = BitStream::new(block);
    // initialize the output to all 0s, aka transparent black
    let mut output = [[0_u8; 4]; 16];

    let mode = extract_mode(&mut stream);

    // This sort of static dispatch is necessary for performance, as it enables
    // the compiler to vectorize the actual pixel interpolation loop, which is
    // the most performance-critical part of the decoding process.

    match mode {
        0 => mode_subset_3::<0>(&mut output, stream),
        1 => mode_subset_2::<1>(&mut output, stream),
        2 => mode_subset_3::<2>(&mut output, stream),
        3 => mode_subset_2::<3>(&mut output, stream),
        4 => mode_4(&mut output, stream),
        5 => mode_5(&mut output, stream),
        6 => mode_6(&mut output, stream),
        7 => mode_subset_2::<7>(&mut output, stream),
        8.. => {
            unlikely_branch();
            // To quote the spec: Mode 8 (LSB 0x00) is reserved and should not
            // be used by the encoder. If this mode is given to the hardware,
            // an all 0 block will be returned.
        }
    }

    output
}

#[inline(always)]
fn mode_subset_2<const MODE: u8>(output: &mut [[u8; 4]; 16], mut stream: BitStream) {
    debug_assert!(MODE == 1 || MODE == 3 || MODE == 7);

    let partition_set_id = extract_partition_set_id(MODE, &mut stream);
    let subset_index_map = PARTITION_SET_2[partition_set_id as usize];

    // get fully decoded endpoints
    let [color0_s0, color1_s0, color0_s1, color1_s1] = get_end_points_4(MODE, &mut stream);

    let index_bits = match MODE {
        1 => 3,
        3 | 7 => 2,
        _ => unreachable!(),
    };

    let indexes = Indexes::new_p2(index_bits, &mut stream, subset_index_map.fixup_index_2);

    for pixel_index in 0..16 {
        let subset_index = subset_index_map.get_subset_index(pixel_index);

        // an `if` turns out to be faster than indexing here
        let [color0, color1] = if subset_index == 0 {
            [color0_s0, color1_s0]
        } else {
            [color0_s1, color1_s1]
        };

        let index = indexes.get_index(pixel_index);

        let r = interpolate_2_or_3(color0[0], color1[0], index, index_bits);
        let g = interpolate_2_or_3(color0[1], color1[1], index, index_bits);
        let b = interpolate_2_or_3(color0[2], color1[2], index, index_bits);
        let a = interpolate_2_or_3(color0[3], color1[3], index, index_bits);

        output[pixel_index as usize] = [r, g, b, a];
    }
}
#[inline(always)]
fn mode_subset_3<const MODE: u8>(output: &mut [[u8; 4]; 16], mut stream: BitStream) {
    debug_assert!(MODE == 0 || MODE == 2);

    let partition_set_id = extract_partition_set_id(MODE, &mut stream);
    let subset_index_map = PARTITION_SET_3[partition_set_id as usize];

    // get fully decoded endpoints
    let endpoints = get_end_points_6(MODE, &mut stream);

    let index_bits = match MODE {
        0 => 3,
        2 => 2,
        _ => unreachable!(),
    };

    let indexes = Indexes::new_p3(
        index_bits,
        &mut stream,
        subset_index_map.fixup_index_2,
        subset_index_map.fixup_index_3,
    );

    for pixel_index in 0..16 {
        // The `.min(2)` allows LLVM to prove that bounds checks are unnecessary
        let subset_index = subset_index_map.get_subset_index(pixel_index).min(2);

        // endpoints are now complete.
        let endpoint_start = endpoints[2 * subset_index as usize];
        let endpoint_end = endpoints[2 * subset_index as usize + 1];

        let index = indexes.get_index(pixel_index);

        let r = interpolate_2_or_3(endpoint_start[0], endpoint_end[0], index, index_bits);
        let g = interpolate_2_or_3(endpoint_start[1], endpoint_end[1], index, index_bits);
        let b = interpolate_2_or_3(endpoint_start[2], endpoint_end[2], index, index_bits);
        let a = interpolate_2_or_3(endpoint_start[3], endpoint_end[3], index, index_bits);

        output[pixel_index as usize] = [r, g, b, a];
    }
}
fn mode_4(output: &mut [[u8; 4]; 16], mut stream: BitStream) {
    // rotation and index mode as one read
    let rotation_and_index_mode = stream.consume_bits(3);
    // extract rotation bits
    let rotation = rotation_and_index_mode & 0b11;
    // index mode
    let index_mode = rotation_and_index_mode & 0b100 != 0;

    // get fully decoded endpoints
    let [color0, color1] = get_end_points_2(4, &mut stream);

    let color_indexes = Indexes::new_p1(2, &mut stream);
    let alpha_indexes = Indexes::new_p1(3, &mut stream);

    for pixel_index in 0..16 {
        let color_index = color_indexes.get_index(pixel_index);
        let alpha_index = alpha_indexes.get_index(pixel_index);

        let mut color_weight = get_weight_2(color_index);
        let mut alpha_weight = get_weight_3(alpha_index);

        if index_mode {
            std::mem::swap(&mut color_weight, &mut alpha_weight);
        }

        output[pixel_index as usize] =
            interpolate_colors_alpha(color0, color1, color_weight, alpha_weight);
    }

    swap_channels(output, rotation);
}
fn mode_5(output: &mut [[u8; 4]; 16], mut stream: BitStream) {
    let rotation = stream.consume_bits(2);

    // get fully decoded endpoints
    let [color0, color1] = get_end_points_2(5, &mut stream);

    let color_indexes = Indexes::new_p1(2, &mut stream);
    let alpha_indexes = Indexes::new_p1(2, &mut stream);

    for pixel_index in 0..16 {
        let color_index = color_indexes.get_index(pixel_index);
        let alpha_index = alpha_indexes.get_index(pixel_index);

        let color_weight = get_weight_2(color_index);
        let alpha_weight = get_weight_2(alpha_index);

        output[pixel_index as usize] =
            interpolate_colors_alpha(color0, color1, color_weight, alpha_weight);
    }

    swap_channels(output, rotation);
}
fn mode_6(output: &mut [[u8; 4]; 16], mut stream: BitStream) {
    // get fully decoded endpoints
    let [color0, color1] = get_end_points_2(6, &mut stream);

    let indexes = Indexes::new_p1(4, &mut stream);

    for pixel_index in 0..16 {
        let index = indexes.get_index(pixel_index);
        let weight = get_weight_4(index);
        output[pixel_index as usize] = interpolate_colors(color0, color1, weight);
    }
}

fn extract_mode(stream: &mut BitStream) -> u8 {
    // instead of doing it in a loopty loop, just count trailing zeros
    let mode = stream.low_u8().trailing_zeros() as u8;
    stream.skip(mode + 1);
    mode
}
fn extract_partition_set_id(mode: u8, stream: &mut BitStream) -> u8 {
    debug_assert!(matches!(mode, 0 | 1 | 2 | 3 | 7));
    let bits = if mode == 0 { 4 } else { 6 };
    stream.consume_bits(bits)
}

#[inline]
fn promote(mut number: u8, number_bits: u8) -> u8 {
    debug_assert!((4..8).contains(&number_bits));
    number <<= 8 - number_bits;
    number |= number >> number_bits;
    number
}
#[inline(always)]
fn get_end_points_2(mode: u8, stream: &mut BitStream) -> [[u8; 4]; 2] {
    #![allow(clippy::needless_range_loop)]
    let mut output: [[u8; 4]; 2] = Default::default();

    match mode {
        4 => {
            let r = [0_u8; 2].map(|_| stream.consume_bits(5));
            let g = [0_u8; 2].map(|_| stream.consume_bits(5));
            let b = [0_u8; 2].map(|_| stream.consume_bits(5));
            let a = [0_u8; 2].map(|_| stream.consume_bits(6));

            for i in 0..2 {
                output[i] = [
                    promote(r[i], 5),
                    promote(g[i], 5),
                    promote(b[i], 5),
                    promote(a[i], 6),
                ];
            }
        }
        5 => {
            let r = [0_u8; 2].map(|_| stream.consume_bits(7));
            let g = [0_u8; 2].map(|_| stream.consume_bits(7));
            let b = [0_u8; 2].map(|_| stream.consume_bits(7));
            let a = [0_u8; 2].map(|_| stream.consume_bits(8));

            for i in 0..2 {
                output[i] = [promote(r[i], 7), promote(g[i], 7), promote(b[i], 7), a[i]];
            }
        }
        6 => {
            let mut r = [0_u8; 2].map(|_| stream.consume_bits(7));
            let mut g = [0_u8; 2].map(|_| stream.consume_bits(7));
            let mut b = [0_u8; 2].map(|_| stream.consume_bits(7));
            let mut a = [0_u8; 2].map(|_| stream.consume_bits(7));

            // each endpoint has its own p
            for i in 0..2 {
                let p = stream.consume_bit() as u8;
                r[i] = (r[i] << 1) | p;
                g[i] = (g[i] << 1) | p;
                b[i] = (b[i] << 1) | p;
                a[i] = (a[i] << 1) | p;
            }

            for i in 0..2 {
                output[i] = [r[i], g[i], b[i], a[i]];
            }
        }
        _ => unreachable!(),
    };

    output
}
#[inline(always)]
fn get_end_points_4(mode: u8, stream: &mut BitStream) -> [[u8; 4]; 4] {
    let mut output: [[u8; 4]; 4] = Default::default();

    match mode {
        1 => {
            let mut r = [0_u8; 4].map(|_| stream.consume_bits(6));
            let mut g = [0_u8; 4].map(|_| stream.consume_bits(6));
            let mut b = [0_u8; 4].map(|_| stream.consume_bits(6));

            // p is shared between endpoints of the same subset
            for i in 0..2 {
                let p = stream.consume_bit() as u8;
                let i0 = i * 2;
                let i1 = i0 + 1;
                r[i0] = (r[i0] << 1) | p;
                g[i0] = (g[i0] << 1) | p;
                b[i0] = (b[i0] << 1) | p;
                r[i1] = (r[i1] << 1) | p;
                g[i1] = (g[i1] << 1) | p;
                b[i1] = (b[i1] << 1) | p;
            }

            for i in 0..4 {
                output[i] = [promote(r[i], 7), promote(g[i], 7), promote(b[i], 7), 255];
            }
        }
        3 => {
            let mut r = [0_u8; 4].map(|_| stream.consume_bits(7));
            let mut g = [0_u8; 4].map(|_| stream.consume_bits(7));
            let mut b = [0_u8; 4].map(|_| stream.consume_bits(7));

            // each endpoint has its own p
            for i in 0..4 {
                let p = stream.consume_bit() as u8;
                r[i] = (r[i] << 1) | p;
                g[i] = (g[i] << 1) | p;
                b[i] = (b[i] << 1) | p;
            }

            for i in 0..4 {
                output[i] = [r[i], g[i], b[i], 255];
            }
        }
        7 => {
            let mut r = [0_u8; 4].map(|_| stream.consume_bits(5));
            let mut g = [0_u8; 4].map(|_| stream.consume_bits(5));
            let mut b = [0_u8; 4].map(|_| stream.consume_bits(5));
            let mut a = [0_u8; 4].map(|_| stream.consume_bits(5));

            // each endpoint has its own p
            for i in 0..4 {
                let p = stream.consume_bit() as u8;
                r[i] = (r[i] << 1) | p;
                g[i] = (g[i] << 1) | p;
                b[i] = (b[i] << 1) | p;
                a[i] = (a[i] << 1) | p;
            }

            for i in 0..4 {
                output[i] = [
                    promote(r[i], 6),
                    promote(g[i], 6),
                    promote(b[i], 6),
                    promote(a[i], 6),
                ];
            }
        }
        _ => unreachable!(),
    };

    output
}
#[inline(always)]
fn get_end_points_6(mode: u8, stream: &mut BitStream) -> [[u8; 4]; 6] {
    let mut output: [[u8; 4]; 6] = Default::default();

    match mode {
        0 => {
            let mut r = [0_u8; 6].map(|_| stream.consume_bits(4));
            let mut g = [0_u8; 6].map(|_| stream.consume_bits(4));
            let mut b = [0_u8; 6].map(|_| stream.consume_bits(4));

            // each endpoint has its own p
            for i in 0..6 {
                let p = stream.consume_bit() as u8;
                r[i] = (r[i] << 1) | p;
                g[i] = (g[i] << 1) | p;
                b[i] = (b[i] << 1) | p;
            }

            for i in 0..6 {
                output[i] = [promote(r[i], 5), promote(g[i], 5), promote(b[i], 5), 255];
            }
        }
        2 => {
            let r = [0_u8; 6].map(|_| stream.consume_bits(5));
            let g = [0_u8; 6].map(|_| stream.consume_bits(5));
            let b = [0_u8; 6].map(|_| stream.consume_bits(5));

            for i in 0..6 {
                output[i] = [promote(r[i], 5), promote(g[i], 5), promote(b[i], 5), 255];
            }
        }
        _ => unreachable!(),
    };

    output
}

fn swap_channels(pixels: &mut [[u8; 4]; 16], rotation: u8) {
    // Decode the 2 color rotation bits as follows:
    // 00 - Block format is Scalar(A) Vector(RGB) - no swapping
    // 01 - Block format is Scalar(R) Vector(AGB) - swap A and R
    // 10 - Block format is Scalar(G) Vector(RAB) - swap A and G
    // 11 - Block format is Scalar(B) Vector(RGA) - swap A and B
    match rotation {
        1 => pixels.iter_mut().for_each(|p| p.swap(0, 3)),
        2 => pixels.iter_mut().for_each(|p| p.swap(1, 3)),
        3 => pixels.iter_mut().for_each(|p| p.swap(2, 3)),
        _ => {}
    };
}

// Weights are all multiplied by 4 compared to the original ones. This changes
// the interpolation formula from
//   ((64-w)*e0 + w*e1 + 32) >> 6
// to
//   ((256-w)*e0 + w*e1 + 128) >> 8
// The nice thing about this is that intermediate results still fit into u16,
// but the compiler can optimize away the `>> 8`.
const WEIGHTS_2: [u16; 4] = [0, 84, 172, 256];
const WEIGHTS_3: [u16; 8] = [0, 36, 72, 108, 148, 184, 220, 256];
const WEIGHTS_4: [u16; 16] = [
    0, 16, 36, 52, 68, 84, 104, 120, 136, 152, 172, 188, 204, 220, 240, 256,
];

fn interpolate_2_or_3(e0: u8, e1: u8, index: u8, index_bits: u8) -> u8 {
    let weight = match index_bits {
        2 => WEIGHTS_2[index as usize],
        3 => WEIGHTS_3[index as usize],
        _ => unreachable!(),
    };
    let w0 = 256 - weight;
    let w1 = weight;
    ((w0 * e0 as u16 + w1 * e1 as u16 + 128) >> 8) as u8
}

#[inline]
fn get_weight_4(index: u8) -> u16 {
    WEIGHTS_4[index as usize]
}
#[inline]
fn get_weight_3(index: u8) -> u16 {
    WEIGHTS_3[index as usize]
}
#[inline]
fn get_weight_2(index: u8) -> u16 {
    WEIGHTS_2[index as usize]
}
#[inline]
fn interpolate_colors(color0: [u8; 4], color1: [u8; 4], weight: u16) -> [u8; 4] {
    let w0 = 256 - weight;
    let w1 = weight;
    [
        ((w0 * color0[0] as u16 + w1 * color1[0] as u16 + 128) >> 8) as u8,
        ((w0 * color0[1] as u16 + w1 * color1[1] as u16 + 128) >> 8) as u8,
        ((w0 * color0[2] as u16 + w1 * color1[2] as u16 + 128) >> 8) as u8,
        ((w0 * color0[3] as u16 + w1 * color1[3] as u16 + 128) >> 8) as u8,
    ]
}
#[inline]
fn interpolate_colors_alpha(
    color0: [u8; 4],
    color1: [u8; 4],
    color_weight: u16,
    alpha_weight: u16,
) -> [u8; 4] {
    let wc0 = 256 - color_weight;
    let wc1 = color_weight;
    let wa0 = 256 - alpha_weight;
    let wa1 = alpha_weight;
    [
        ((wc0 * color0[0] as u16 + wc1 * color1[0] as u16 + 128) >> 8) as u8,
        ((wc0 * color0[1] as u16 + wc1 * color1[1] as u16 + 128) >> 8) as u8,
        ((wc0 * color0[2] as u16 + wc1 * color1[2] as u16 + 128) >> 8) as u8,
        ((wa0 * color0[3] as u16 + wa1 * color1[3] as u16 + 128) >> 8) as u8,
    ]
}
