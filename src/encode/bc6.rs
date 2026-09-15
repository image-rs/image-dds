#![allow(clippy::needless_range_loop)]

use bitflags::bitflags;
use glam::Vec3A;

pub(crate) use crate::bcn_data::BC6HFormat;
use crate::{
    bcn_data::{EndPointPair, IntColor, ModeOne, ModeTwo, Subset2Map, PARTITION_SET_2},
    decode::bc6,
    encode::bcn_util::{
        line3_fit_endpoints, refine_endpoints, ColorLine3, CompressedIndexList, IndexList,
        Quantization, RefinementOptions,
    },
};

const F16_MAX: f32 = 65504.0;

bitflags! {
#[derive(Copy, Clone, Debug)]
pub(crate) struct Bc6Modes: u32 {
    // Mode TWO
    const M10_555 = 1 << ModeTwo::M10_555 as u8;
    const M7_666 = 1 << ModeTwo::M7_666 as u8;
    const M11_544 = 1 << ModeTwo::M11_544 as u8;
    const M11_454 = 1 << ModeTwo::M11_454 as u8;
    const M11_445 = 1 << ModeTwo::M11_445 as u8;
    const M9_555 = 1 << ModeTwo::M9_555 as u8;
    const M8_655 = 1 << ModeTwo::M8_655 as u8;
    const M8_565 = 1 << ModeTwo::M8_565 as u8;
    const M8_556 = 1 << ModeTwo::M8_556 as u8;
    const M6_666 = 1 << ModeTwo::M6_666 as u8;

    const MODE_TWO = Self::M10_555.bits()
        | Self::M7_666.bits()
        | Self::M11_544.bits()
        | Self::M11_454.bits()
        | Self::M11_445.bits()
        | Self::M9_555.bits()
        | Self::M8_655.bits()
        | Self::M8_565.bits()
        | Self::M8_556.bits()
        | Self::M6_666.bits();

    // Mode ONE
    const M10_10 = 1 << ModeOne::M10_10 as u8;
    const M11_9 = 1 << ModeOne::M11_9 as u8;
    const M12_8 = 1 << ModeOne::M12_8 as u8;
    const M16_4 = 1 << ModeOne::M16_4 as u8;

    const MODE_ONE = Self::M10_10.bits()
        | Self::M11_9.bits()
        | Self::M12_8.bits()
        | Self::M16_4.bits();
    }
}
impl Bc6Modes {
    fn has_mode_one(&self, mode: ModeOne) -> bool {
        self.contains(Bc6Modes::from_bits_retain(1 << mode as u8))
    }
    fn has_mode_two(&self, mode: ModeTwo) -> bool {
        self.contains(Bc6Modes::from_bits_retain(1 << mode as u8))
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Bc6Options {
    pub format: BC6HFormat,
    pub modes: Bc6Modes,
    /// How many of the top-scoring partitions to consider during compression.
    pub top_partitions: u8,
    pub quantization: Quantization,
    /// Mode two has 2 families of modes that only differ in how many bits are used to encode delta.
    /// Namely, M8_xxx and M11_xxx. If e.g. the encoder finds a block compression with mode M8_565,
    /// then compressions with M8_655 and M8_556 likely also exist and are very similar in quality.
    ///
    /// If set to `true`, the encoder will pick the first successful M8_xxx or M11_xxx mode and
    /// skip its delta variants.
    pub pick_first_8_11: bool,
    /// When enabled, the encoder will use a fast heuristic to check whether the block's endpoints
    /// can be transformed before performing full quantization + compression.
    ///
    /// Speeds up compression but comes at a minor quality loss.
    pub check_transformable: bool,
}

pub(crate) fn compress_bc6_block(mut block: [Vec3A; 16], options: Bc6Options) -> [u8; 16] {
    clamp_to_range(&mut block, options.format);

    // get RGB range
    let mut min = block[0];
    let mut max = block[0];
    for &c in &block[1..] {
        min = min.min(c);
        max = max.max(c);
    }

    if min == max {
        return compress_single_color(min, options.format);
    }

    let mut best = Compressed::invalid();

    if options.modes.intersects(Bc6Modes::MODE_ONE) {
        let one = compress_bc6_one(&block, (min, max), &options);
        best = best.better(one);
    }

    if options.modes.intersects(Bc6Modes::MODE_TWO) {
        let two = compress_bc6_two(&block, &options);
        best = best.better(two);
    }

    best.block
}

fn get_indexes_error_one(
    a: Int16Color,
    b: Int16Color,
    block: &[Vec3A; 16],
    format: BC6HFormat,
) -> f32 {
    let palette = create_palette_one_from_int16(a, b, format);
    get_block_error_p16(block, &palette)
}
fn get_indexes_error_two(a: Int16Color, b: Int16Color, block: &[Vec3A], format: BC6HFormat) -> f32 {
    let palette = create_palette_two_from_int16(a, b, format);
    get_block_error_p8(block, &palette)
}
fn create_palette_one_from_int16(
    Int16Color(a): Int16Color,
    Int16Color(b): Int16Color,
    format: BC6HFormat,
) -> [Vec3A; 16] {
    let mut palette: [IntColor<u16>; 16] = Default::default();
    // interpolate
    for i in 0..16 {
        let w = bc6::WEIGHT_4[i] as i32;
        palette[i].r = bc6::finish_unquantize((a.r * (64 - w) + b.r * w + 32) >> 6, format);
        palette[i].g = bc6::finish_unquantize((a.g * (64 - w) + b.g * w + 32) >> 6, format);
        palette[i].b = bc6::finish_unquantize((a.b * (64 - w) + b.b * w + 32) >> 6, format);
    }

    palette.map(half3_to_vec3)
}
fn create_palette_two_from_int16(
    Int16Color(a): Int16Color,
    Int16Color(b): Int16Color,
    format: BC6HFormat,
) -> [Vec3A; 8] {
    let mut palette: [IntColor<u16>; 8] = Default::default();
    // interpolate
    for i in 0..8 {
        let w = bc6::WEIGHT_3[i] as i32;
        palette[i].r = bc6::finish_unquantize((a.r * (64 - w) + b.r * w + 32) >> 6, format);
        palette[i].g = bc6::finish_unquantize((a.g * (64 - w) + b.g * w + 32) >> 6, format);
        palette[i].b = bc6::finish_unquantize((a.b * (64 - w) + b.b * w + 32) >> 6, format);
    }

    palette.map(half3_to_vec3)
}
fn get_block_error_p8(block: &[Vec3A], palette: &[Vec3A; 8]) -> f32 {
    let mut error = 0.0;
    for &color in block {
        let mut min_dist = f32::MAX;
        for &p in palette {
            let distance = (color - p).length_squared();
            min_dist = min_dist.min(distance);
        }
        error += min_dist;
    }
    error
}
fn get_block_error_p16(block: &[Vec3A; 16], palette: &[Vec3A; 16]) -> f32 {
    let mut error = 0.0;
    for &color in block {
        let mut min_dist = f32::MAX;
        for &p in palette {
            let distance = (color - p).length_squared();
            min_dist = min_dist.min(distance);
        }
        error += min_dist;
    }
    error
}

/// `endpoints` must be quantized (but NOT transformed) according to the given mode
fn get_indexes_one(
    endpoints: EndPointPair,
    block: &[Vec3A; 16],
    mode: ModeOne,
    format: BC6HFormat,
) -> (IndexList<4>, f32) {
    let palette =
        crate::decode::bc6::generate_f16_palette_one(endpoints.a, endpoints.b, mode, format);
    let palette = palette.map(half3_to_vec3);

    let mut error = 0.0;
    let mut indexes = IndexList::new();

    for i in 0..16 {
        let (index, e) = find_closest_palette_index(block[i], &palette);
        indexes.set(i, index);
        error += e;
    }

    (indexes, error)
}
/// `endpoints` must be quantized (but NOT transformed) according to the given mode
fn get_indexes_two(
    endpoints: EndPointPair,
    block: &[Vec3A],
    mode: ModeTwo,
    format: BC6HFormat,
) -> (IndexList<3>, f32) {
    let palette =
        crate::decode::bc6::generate_f16_palette_two(endpoints.a, endpoints.b, mode, format);
    let palette = palette.map(half3_to_vec3);

    let mut error = 0.0;
    let mut indexes = IndexList::new();

    for (i, &color) in block.iter().enumerate() {
        let (index, e) = find_closest_palette_index(color, &palette);
        indexes.set(i, index);
        error += e;
    }

    (indexes, error)
}

fn find_closest_palette_index(color: Vec3A, palette: &[Vec3A]) -> (u8, f32) {
    assert!(palette.len() >= 4 && palette.len() <= 16);
    let mut closest_index = 0;
    let mut closest_distance = f32::MAX;
    for (i, &p) in palette.iter().enumerate() {
        let distance = (color - p).length_squared();
        if distance < closest_distance {
            closest_distance = distance;
            closest_index = i;
        }
    }
    (closest_index as u8, closest_distance)
}

fn compress_bc6_one(
    block: &[Vec3A; 16],
    (min, max): (Vec3A, Vec3A),
    options: &Bc6Options,
) -> Compressed {
    let (e0, e1) = line3_fit_endpoints(block, 0.95);

    let (e0, e1) = refine_endpoints(
        e0,
        e1,
        RefinementOptions {
            step_initial: min.distance(max) * 0.2,
            step_decay: 0.5,
            step_min: 0.0,
            max_iter: 5,
        },
        |(e0, e1)| {
            let a = quantize_vec3_to_int16_color(e0, options.format);
            let b = quantize_vec3_to_int16_color(e1, options.format);
            get_indexes_error_one(a, b, block, options.format)
        },
    );

    let a = quantize_vec3_to_int16_color(e0, options.format);
    let b = quantize_vec3_to_int16_color(e1, options.format);

    let mut best = Compressed::invalid();

    for mode in [
        ModeOne::M10_10,
        ModeOne::M11_9,
        ModeOne::M12_8,
        ModeOne::M16_4,
    ] {
        if !options.modes.has_mode_one(mode) {
            continue;
        }

        if let Some(candidate) = quantize_and_index_one((a, b), block, mode, options) {
            best = best.better(candidate)
        } else {
            // Since the transform failed, we know that transforms for modes
            // with even lower bit counts for b will also fail.
            break;
        }
    }

    best
}

fn quantize_and_index_one(
    (a_16, b_16): (Int16Color, Int16Color),
    block: &[Vec3A; 16],
    mode: ModeOne,
    options: &Bc6Options,
) -> Option<Compressed> {
    let a0_bits = mode.a0_bit_count();

    // check whether this mode's transformation is even possible
    if options.check_transformable
        && mode.transformed()
        && !can_compress_endpoints_one(
            EndPointPair {
                a: quantize_round(a_16, a0_bits, options.format),
                b: quantize_round(b_16, a0_bits, options.format),
            },
            mode,
        )
    {
        return None;
    }

    let mut pair = quantize_endpoints(a_16, b_16, a0_bits, options, |a, b| {
        get_indexes_error_one(a, b, block, options.format)
    });

    let (indexes, error) = get_indexes_one(pair, block, mode, options.format);

    let (compressed_indexes, swap) = indexes.compress_p1();
    if swap {
        std::mem::swap(&mut pair.a, &mut pair.b);
    }

    let compressed_pair = compress_endpoints_one(pair, mode)?;

    Some(Compressed::mode_one(
        mode,
        compressed_pair,
        compressed_indexes,
        error,
    ))
}

fn quantize_endpoints(
    a_16: Int16Color,
    b_16: Int16Color,
    bits: u8,
    options: &Bc6Options,
    compute_error: impl Fn(Int16Color, Int16Color) -> f32,
) -> EndPointPair {
    let (a, b) = pick_best_quantized(
        a_16,
        b_16,
        bits,
        options.format,
        options.quantization,
        |a, b| {
            let a_16 = unquantize(a, bits, options.format);
            let b_16 = unquantize(b, bits, options.format);
            compute_error(a_16, b_16)
        },
    );
    EndPointPair { a, b }
}

/// A thin wrapper around `IntColor<i32>` that represents a 16-bit "stretched float" color.
#[derive(Clone, Copy)]
struct Int16Color(IntColor<i32>);
impl IntColor<i32> {
    #[inline]
    fn to_int16(self, format: BC6HFormat) -> Int16Color {
        #[cfg(debug_assertions)]
        {
            let range = match format {
                BC6HFormat::UnsignedF16 => 0..65536,
                BC6HFormat::SignedF16 => -32768..32768,
            };
            debug_assert!(range.contains(&self.r), "r={} out of range", self.r);
            debug_assert!(range.contains(&self.g), "g={} out of range", self.g);
            debug_assert!(range.contains(&self.b), "b={} out of range", self.b);
        }

        Int16Color(self)
    }
}

fn unquantize(a: IntColor<i32>, bits: u8, format: BC6HFormat) -> Int16Color {
    IntColor::new(
        bc6::unquantize(a.r, bits, format),
        bc6::unquantize(a.g, bits, format),
        bc6::unquantize(a.b, bits, format),
    )
    .to_int16(format)
}
fn quantize_round(a: Int16Color, bits: u8, format: BC6HFormat) -> IntColor<i32> {
    IntColor::new(
        quantize_int16_to_bitwidth_round(a.0.r, bits, format),
        quantize_int16_to_bitwidth_round(a.0.g, bits, format),
        quantize_int16_to_bitwidth_round(a.0.b, bits, format),
    )
}
fn pick_best_quantized(
    a_16: Int16Color,
    b_16: Int16Color,
    bits: u8,
    format: BC6HFormat,
    quantization: Quantization,
    mut error_metric: impl FnMut(IntColor<i32>, IntColor<i32>) -> f32,
) -> (IntColor<i32>, IntColor<i32>) {
    fn full_range(a: Int16Color, bits: u8, format: BC6HFormat) -> (IntColor<i32>, IntColor<i32>) {
        let r = quantize_int16_to_bitwidth_floor_ceil(a.0.r, bits, format);
        let g = quantize_int16_to_bitwidth_floor_ceil(a.0.g, bits, format);
        let b = quantize_int16_to_bitwidth_floor_ceil(a.0.b, bits, format);
        debug_assert!(r.0 <= r.1);
        debug_assert!(g.0 <= g.1);
        debug_assert!(b.0 <= b.1);
        (IntColor::new(r.0, g.0, b.0), IntColor::new(r.1, g.1, b.1))
    }
    fn optimized_range(
        a: Int16Color,
        bits: u8,
        format: BC6HFormat,
    ) -> (IntColor<i32>, IntColor<i32>) {
        let (mut floor, mut ceil) = full_range(a, bits, format);

        let floor_16 = unquantize(floor, bits, format);
        let ceil_16 = unquantize(ceil, bits, format);

        for c in 0..3 {
            let floor_16 = floor_16.0.get_channel(c);
            let ceil_16 = ceil_16.0.get_channel(c);
            let exact_16 = a.0.get_channel(c);
            // This equivalent to the hard-coded cull threshold of 0.25
            let cull_threshold = (ceil_16 - floor_16) >> 4;
            if (floor_16 - exact_16).abs() <= cull_threshold {
                // close to floor, so skip ceil
                ceil.set_channel(c, floor.get_channel(c));
            } else if (ceil_16 - exact_16).abs() <= cull_threshold {
                // close to ceil, so skip floor
                floor.set_channel(c, ceil.get_channel(c));
            }
        }

        (floor, ceil)
    }

    if bits >= 15 {
        // no quantization is done for 16 bits
        return (a_16.0, b_16.0);
    }

    let mut best = (
        quantize_round(a_16, bits, format),
        quantize_round(b_16, bits, format),
    );

    if quantization == Quantization::Round {
        // For simple rounding, we don't need to optimize at all
        return best;
    }

    let mut best_error = error_metric(best.0, best.1);

    let get_range = match quantization {
        Quantization::ChannelWiseOptimized => optimized_range,
        _ => full_range,
    };

    let (c0_min, c0_max) = get_range(a_16, bits, format);
    let (c1_min, c1_max) = get_range(b_16, bits, format);

    // Channel-wise optimization
    for c in 0..3 {
        let skip0 = best.0.get_channel(c);
        let skip1 = best.1.get_channel(c);
        for channel0 in c0_min.get_channel(c)..=c0_max.get_channel(c) {
            for channel1 in c1_min.get_channel(c)..=c1_max.get_channel(c) {
                if channel0 == skip0 && channel1 == skip1 {
                    continue;
                }
                let (mut c0, mut c1) = best;

                c0.set_channel(c, channel0);
                c1.set_channel(c, channel1);
                let error = error_metric(c0, c1);
                if error < best_error {
                    best = (c0, c1);
                    best_error = error;
                }
            }
        }
    }
    best
}

fn compress_bc6_two(block: &[Vec3A; 16], options: &Bc6Options) -> Compressed {
    let mut best = Compressed::invalid();
    for partition in score_partitions(block, 32)
        .into_iter()
        .take(options.top_partitions as usize)
    {
        let subset = PARTITION_SET_2[partition as usize];

        let mut reordered = *block;
        subset.sort_block(&mut reordered);
        let split_index = subset.count_zeros() as usize;

        // optimize endpoints for subsets
        let block_s0 = &reordered[..split_index];
        let block_s1 = &reordered[split_index..];
        let e_s0 = get_refined_endpoints(block_s0, options);
        let e_s1 = get_refined_endpoints(block_s1, options);

        let a0_16 = quantize_vec3_to_int16_color(e_s0.0, options.format);
        let b0_16 = quantize_vec3_to_int16_color(e_s0.1, options.format);
        let a1_16 = quantize_vec3_to_int16_color(e_s1.0, options.format);
        let b1_16 = quantize_vec3_to_int16_color(e_s1.1, options.format);

        let mut last_failed_bit_count = 100;
        let mut m8_success = false;
        let mut m11_success = false;
        for mode in [
            ModeTwo::M6_666,
            ModeTwo::M7_666,
            ModeTwo::M8_556,
            ModeTwo::M8_565,
            ModeTwo::M8_655,
            ModeTwo::M9_555,
            ModeTwo::M10_555,
            ModeTwo::M11_445,
            ModeTwo::M11_454,
            ModeTwo::M11_544,
        ] {
            let a0_bit_count = mode.a0_bit_count();

            if a0_bit_count > last_failed_bit_count {
                // If a mode with a lower bit count for a0 failed already, then
                // we know that the transform for this mode will fail as well.
                // Similar reasoning as in mode one. See `compress_bc6_one`.
                break;
            }

            if !options.modes.has_mode_two(mode) {
                continue;
            }

            // If we already have a compression for M8_xxx and M11_xxx, then skip
            // similar modes as they are very unlikely to produce a better result.
            if options.pick_first_8_11
                && (m8_success && a0_bit_count == 8 || m11_success && a0_bit_count == 11)
            {
                continue;
            }

            if let Some(compressed) = quantize_and_index_two(
                [(a0_16, b0_16), (a1_16, b1_16)],
                block_s0,
                block_s1,
                partition,
                subset,
                mode,
                options,
            ) {
                best = best.better(compressed);
                m8_success |= a0_bit_count == 8;
                m11_success |= a0_bit_count == 11;
            } else {
                last_failed_bit_count = a0_bit_count;
            }
        }
    }
    best
}
fn score_partitions(block: &[Vec3A; 16], max_partitions: u8) -> [u8; 32] {
    let compute_error_rgb =
        move |colors: &[Vec3A]| -> f32 { ColorLine3::new(colors).sum_dist_sq(colors) };

    let mut scored: [(u8, f32); 32] = std::array::from_fn(move |i| {
        let partition = i as u8;
        let subset = PARTITION_SET_2[partition as usize];

        if partition >= max_partitions {
            return (partition, f32::INFINITY);
        }

        let split_index = subset.count_zeros() as usize;

        let mut reordered = *block;
        subset.sort_block(&mut reordered);
        let error = compute_error_rgb(&reordered[..split_index])
            + compute_error_rgb(&reordered[split_index..]);

        (partition, error)
    });

    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    scored.map(|(p, _)| p)
}
fn get_refined_endpoints(block: &[Vec3A], options: &Bc6Options) -> (Vec3A, Vec3A) {
    let (e0, e1) = line3_fit_endpoints(block, 0.95);

    refine_endpoints(
        e0,
        e1,
        RefinementOptions {
            step_initial: e0.distance(e1) * 0.25,
            step_decay: 0.5,
            step_min: 0.0,
            max_iter: 5,
        },
        |(e0, e1)| {
            let a = quantize_vec3_to_int16_color(e0, options.format);
            let b = quantize_vec3_to_int16_color(e1, options.format);
            get_indexes_error_two(a, b, block, options.format)
        },
    )
}

fn quantize_and_index_two(
    [(a0_16, b0_16), (a1_16, b1_16)]: [(Int16Color, Int16Color); 2],
    block_s0: &[Vec3A],
    block_s1: &[Vec3A],
    partition: u8,
    subset: Subset2Map,
    mode: ModeTwo,
    options: &Bc6Options,
) -> Option<Compressed> {
    let a0_bits = mode.a0_bit_count();

    // check whether the endpoints can possibly be compressed before quantizing
    if options.check_transformable
        && mode.transformed()
        && can_compress_endpoints_two(
            [
                EndPointPair {
                    a: quantize_round(a0_16, a0_bits, options.format),
                    b: quantize_round(b0_16, a0_bits, options.format),
                },
                EndPointPair {
                    a: quantize_round(a1_16, a0_bits, options.format),
                    b: quantize_round(b1_16, a0_bits, options.format),
                },
            ],
            mode,
        ) == Some(false)
    {
        return None;
    }

    // quantize endpoints
    let mut pair_s0 = quantize_endpoints(a0_16, b0_16, a0_bits, options, |a, b| {
        get_indexes_error_two(a, b, block_s0, options.format)
    });
    let mut pair_s1 = quantize_endpoints(a1_16, b1_16, a0_bits, options, |a, b| {
        get_indexes_error_two(a, b, block_s1, options.format)
    });

    let (indexes_s0, s0_error) = get_indexes_two(pair_s0, block_s0, mode, options.format);
    let (indexes_s1, s1_error) = get_indexes_two(pair_s1, block_s1, mode, options.format);

    let error = s0_error + s1_error;
    let index = IndexList::merge2(subset, indexes_s0, indexes_s1);

    let (compressed_index, swaps) = index.compress_p2(subset);
    if swaps[0] {
        std::mem::swap(&mut pair_s0.a, &mut pair_s0.b);
    }
    if swaps[1] {
        std::mem::swap(&mut pair_s1.a, &mut pair_s1.b);
    }

    let compressed_endpoints = compress_endpoints_two([pair_s0, pair_s1], mode)?;

    Some(Compressed::mode_two(
        mode,
        compressed_endpoints,
        partition,
        compressed_index,
        error,
    ))
}

/// Clamps number to range ([0,MAX] for unsigned and [-Max,Max] for signed) and maps NaN to zero.
///
/// Note: While signed does support +-Inf, I do not want to support them, because these values don't
/// work with MSE for obvious reasons. There are ways around this (e.g. mapping to +-Inf to +-K,
/// where K is some large value outside the range fp16 but within fp32), but I don't want to think
/// about that complexity for now.
fn clamp_to_range(block: &mut [Vec3A; 16], format: BC6HFormat) {
    match format {
        BC6HFormat::UnsignedF16 => {
            for c in block {
                *c = c.max(Vec3A::ZERO).min(Vec3A::splat(F16_MAX));
            }
        }
        BC6HFormat::SignedF16 => {
            for c in block.iter_mut() {
                let mask = c.is_nan_mask();
                *c = Vec3A::select(mask, Vec3A::ZERO, *c);
            }
            for c in block {
                *c = c.max(Vec3A::splat(-F16_MAX)).min(Vec3A::splat(F16_MAX));
            }
        }
    }
}
#[test]
fn test_clamp_to_range() {
    fn clamp(color: Vec3A, format: BC6HFormat) -> Vec3A {
        let mut block = [color; 16];
        clamp_to_range(&mut block, format);
        block[0]
    }

    // Nan, +-Inf
    assert_eq!(
        clamp(
            Vec3A::new(f32::NAN, f32::NEG_INFINITY, f32::INFINITY),
            BC6HFormat::UnsignedF16
        ),
        Vec3A::new(0.0, 0.0, F16_MAX)
    );
    assert_eq!(
        clamp(
            Vec3A::new(f32::NAN, f32::NEG_INFINITY, f32::INFINITY),
            BC6HFormat::SignedF16
        ),
        Vec3A::new(0.0, -F16_MAX, F16_MAX)
    );

    // normal numbers
    assert_eq!(
        clamp(Vec3A::new(-1.0, 1.0, 1_000_000.0), BC6HFormat::UnsignedF16),
        Vec3A::new(0.0, 1.0, F16_MAX)
    );
    assert_eq!(
        clamp(Vec3A::new(-1.0, 1.0, 1_000_000.0), BC6HFormat::SignedF16),
        Vec3A::new(-1.0, 1.0, F16_MAX)
    );
}

/// Single color compression is pretty easy for BC6H. Just use mode One 16_4 and store the color
/// components directly in a0. The rest is all zero.
fn compress_single_color(color: Vec3A, format: BC6HFormat) -> [u8; 16] {
    Compressed::mode_one(
        ModeOne::M16_4,
        EndPointPair {
            a: quantize_vec3_to_int16_color(color, format).0,
            b: IntColor::new(0, 0, 0),
        },
        IndexList::<4>::constant(0).compress_p1().0,
        0.0,
    )
    .block
}

#[derive(Copy, Clone, Debug)]
struct Compressed {
    block: [u8; 16],
    error: f32,
}
impl Compressed {
    fn better(self, other: Self) -> Self {
        if self.error <= other.error {
            self
        } else {
            other
        }
    }

    const fn invalid() -> Self {
        Self {
            block: [0; 16],
            error: f32::INFINITY,
        }
    }

    fn mode_one(mode: ModeOne, pair: EndPointPair, list: CompressedIndexList, error: f32) -> Self {
        let mut stream = BitStream::new();
        stream.write_mode_one(mode);

        let a_bits = mode.a0_bit_count();
        let b_bits = mode.b0_bit_count();
        let a = pair.a.bit_and((1 << a_bits) - 1);
        let b = pair.b.bit_and((1 << b_bits) - 1);

        // first 30 bits are the low 10 bits of a
        stream.write_u64(a.r as u64 & 1023, 10);
        stream.write_u64(a.g as u64 & 1023, 10);
        stream.write_u64(a.b as u64 & 1023, 10);

        let a_ext_bits = a_bits - 10;
        let a_ext_mask = (1 << a_ext_bits) - 1;
        stream.write_u64(b.r as u64, b_bits);
        stream.write_u16_rev(((a.r >> 10) & a_ext_mask) as u16, a_ext_bits);
        stream.write_u64(b.g as u64, b_bits);
        stream.write_u16_rev(((a.g >> 10) & a_ext_mask) as u16, a_ext_bits);
        stream.write_u64(b.b as u64, b_bits);
        stream.write_u16_rev(((a.b >> 10) & a_ext_mask) as u16, a_ext_bits);

        debug_assert_eq!(list.bits, 63);
        stream.write_u64(list.compressed_indexes, 63);

        Compressed {
            block: stream.finish(),
            error,
        }
    }
    fn mode_two(
        mode: ModeTwo,
        endpoints: [EndPointPair; 2],
        partition: u8,
        list: CompressedIndexList,
        error: f32,
    ) -> Self {
        let mut stream = BitStream::new();
        stream.write_mode_two(mode);

        write_compressed_endpoints_two(mode, endpoints, &mut stream);

        stream.write_u64(partition as u64, 5);

        debug_assert_eq!(list.bits, 46);
        stream.write_u64(list.compressed_indexes, 46);

        Compressed {
            block: stream.finish(),
            error,
        }
    }
}

/// Modified version of `decode::bc6::extract_compressed_endpoints_two` for writing.
fn write_compressed_endpoints_two(
    mode: ModeTwo,
    endpoints: [EndPointPair; 2],
    stream: &mut BitStream,
) {
    let w = endpoints[0].a;
    let x = endpoints[0].b;
    let y = endpoints[1].a;
    let z = endpoints[1].b;

    /// This is a macro to translate the expressions from 19.5.13.5 table 2
    /// into useable code.
    ///
    /// It has 2 modes:
    /// 1. Single bit mode: `gy[4]   == write!(g, y, 4)`
    /// 2. Range mode:      `rw[9:0] == write!(r, w, 9..0)`
    ///
    /// Note that these are NOT normal Rust ranges, I'm just misappropriating
    /// their syntax.
    macro_rules! write {
        ($i1:ident, $i2:ident, $index:literal) => {
            stream.write_u64((($i2.$i1 >> $index) & 1) as u64, 1);
        };
        ($i1:ident, $i2:ident, $high:literal .. 0) => {
            let bits = $high + 1;
            stream.write_u64(($i2.$i1 & ((1 << bits) - 1)) as u64, bits);
        };
    }

    match mode {
        ModeTwo::M10_555 => {
            write!(g, y, 4);
            write!(b, y, 4);
            write!(b, z, 4);
            write!(r, w, 9..0);
            write!(g, w, 9..0);
            write!(b, w, 9..0);
            write!(r, x, 4..0);
            write!(g, z, 4);
            write!(g, y, 3..0);
            write!(g, x, 4..0);
            write!(b, z, 0);
            write!(g, z, 3..0);
            write!(b, x, 4..0);
            write!(b, z, 1);
            write!(b, y, 3..0);
            write!(r, y, 4..0);
            write!(b, z, 2);
            write!(r, z, 4..0);
            write!(b, z, 3);
        }
        ModeTwo::M7_666 => {
            write!(g, y, 5);
            write!(g, z, 4);
            write!(g, z, 5);
            write!(r, w, 6..0);
            write!(b, z, 0);
            write!(b, z, 1);
            write!(b, y, 4);
            write!(g, w, 6..0);
            write!(b, y, 5);
            write!(b, z, 2);
            write!(g, y, 4);
            write!(b, w, 6..0);
            write!(b, z, 3);
            write!(b, z, 5);
            write!(b, z, 4);
            write!(r, x, 5..0);
            write!(g, y, 3..0);
            write!(g, x, 5..0);
            write!(g, z, 3..0);
            write!(b, x, 5..0);
            write!(b, y, 3..0);
            write!(r, y, 5..0);
            write!(r, z, 5..0);
        }
        ModeTwo::M11_544 => {
            write!(r, w, 9..0);
            write!(g, w, 9..0);
            write!(b, w, 9..0);
            write!(r, x, 4..0);
            write!(r, w, 10);
            write!(g, y, 3..0);
            write!(g, x, 3..0);
            write!(g, w, 10);
            write!(b, z, 0);
            write!(g, z, 3..0);
            write!(b, x, 3..0);
            write!(b, w, 10);
            write!(b, z, 1);
            write!(b, y, 3..0);
            write!(r, y, 4..0);
            write!(b, z, 2);
            write!(r, z, 4..0);
            write!(b, z, 3);
        }
        ModeTwo::M11_454 => {
            write!(r, w, 9..0);
            write!(g, w, 9..0);
            write!(b, w, 9..0);
            write!(r, x, 3..0);
            write!(r, w, 10);
            write!(g, z, 4);
            write!(g, y, 3..0);
            write!(g, x, 4..0);
            write!(g, w, 10);
            write!(g, z, 3..0);
            write!(b, x, 3..0);
            write!(b, w, 10);
            write!(b, z, 1);
            write!(b, y, 3..0);
            write!(r, y, 3..0);
            write!(b, z, 0);
            write!(b, z, 2);
            write!(r, z, 3..0);
            write!(g, y, 4);
            write!(b, z, 3);
        }
        ModeTwo::M11_445 => {
            write!(r, w, 9..0);
            write!(g, w, 9..0);
            write!(b, w, 9..0);
            write!(r, x, 3..0);
            write!(r, w, 10);
            write!(b, y, 4);
            write!(g, y, 3..0);
            write!(g, x, 3..0);
            write!(g, w, 10);
            write!(b, z, 0);
            write!(g, z, 3..0);
            write!(b, x, 4..0);
            write!(b, w, 10);
            write!(b, y, 3..0);
            write!(r, y, 3..0);
            write!(b, z, 1);
            write!(b, z, 2);
            write!(r, z, 3..0);
            write!(b, z, 4);
            write!(b, z, 3);
        }
        ModeTwo::M9_555 => {
            write!(r, w, 8..0);
            write!(b, y, 4);
            write!(g, w, 8..0);
            write!(g, y, 4);
            write!(b, w, 8..0);
            write!(b, z, 4);
            write!(r, x, 4..0);
            write!(g, z, 4);
            write!(g, y, 3..0);
            write!(g, x, 4..0);
            write!(b, z, 0);
            write!(g, z, 3..0);
            write!(b, x, 4..0);
            write!(b, z, 1);
            write!(b, y, 3..0);
            write!(r, y, 4..0);
            write!(b, z, 2);
            write!(r, z, 4..0);
            write!(b, z, 3);
        }
        ModeTwo::M8_655 => {
            write!(r, w, 7..0);
            write!(g, z, 4);
            write!(b, y, 4);
            write!(g, w, 7..0);
            write!(b, z, 2);
            write!(g, y, 4);
            write!(b, w, 7..0);
            write!(b, z, 3);
            write!(b, z, 4);
            write!(r, x, 5..0);
            write!(g, y, 3..0);
            write!(g, x, 4..0);
            write!(b, z, 0);
            write!(g, z, 3..0);
            write!(b, x, 4..0);
            write!(b, z, 1);
            write!(b, y, 3..0);
            write!(r, y, 5..0);
            write!(r, z, 5..0);
        }
        ModeTwo::M8_565 => {
            write!(r, w, 7..0);
            write!(b, z, 0);
            write!(b, y, 4);
            write!(g, w, 7..0);
            write!(g, y, 5);
            write!(g, y, 4);
            write!(b, w, 7..0);
            write!(g, z, 5);
            write!(b, z, 4);
            write!(r, x, 4..0);
            write!(g, z, 4);
            write!(g, y, 3..0);
            write!(g, x, 5..0);
            write!(g, z, 3..0);
            write!(b, x, 4..0);
            write!(b, z, 1);
            write!(b, y, 3..0);
            write!(r, y, 4..0);
            write!(b, z, 2);
            write!(r, z, 4..0);
            write!(b, z, 3);
        }
        ModeTwo::M8_556 => {
            write!(r, w, 7..0);
            write!(b, z, 1);
            write!(b, y, 4);
            write!(g, w, 7..0);
            write!(b, y, 5);
            write!(g, y, 4);
            write!(b, w, 7..0);
            write!(b, z, 5);
            write!(b, z, 4);
            write!(r, x, 4..0);
            write!(g, z, 4);
            write!(g, y, 3..0);
            write!(g, x, 4..0);
            write!(b, z, 0);
            write!(g, z, 3..0);
            write!(b, x, 5..0);
            write!(b, y, 3..0);
            write!(r, y, 4..0);
            write!(b, z, 2);
            write!(r, z, 4..0);
            write!(b, z, 3);
        }
        ModeTwo::M6_666 => {
            write!(r, w, 5..0);
            write!(g, z, 4);
            write!(b, z, 0);
            write!(b, z, 1);
            write!(b, y, 4);
            write!(g, w, 5..0);
            write!(g, y, 5);
            write!(b, y, 5);
            write!(b, z, 2);
            write!(g, y, 4);
            write!(b, w, 5..0);
            write!(g, z, 5);
            write!(b, z, 3);
            write!(b, z, 5);
            write!(b, z, 4);
            write!(r, x, 5..0);
            write!(g, y, 3..0);
            write!(g, x, 5..0);
            write!(g, z, 3..0);
            write!(b, x, 5..0);
            write!(b, y, 3..0);
            write!(r, y, 5..0);
            write!(r, z, 5..0);
        }
    }
}

struct BitStream {
    data: u128,
    bits: u8,
}
impl BitStream {
    fn new() -> Self {
        Self { data: 0, bits: 0 }
    }

    #[inline(always)]
    fn write_u64(&mut self, value: u64, bits: u8) {
        debug_assert!(bits < 64);
        debug_assert!(value < (1 << bits));

        self.data |= (value as u128) << self.bits;
        self.bits += bits;
    }
    #[inline(always)]
    fn write_u16_rev(&mut self, value: u16, bits: u8) {
        debug_assert!(bits < 16);
        debug_assert!(value < (1 << bits));

        if bits == 0 {
            return;
        }

        let value = value.reverse_bits() >> (16 - bits);
        self.write_u64(value as u64, bits);
    }

    fn write_mode_one(&mut self, mode: ModeOne) {
        self.write_u64(mode as u64, 5);
    }
    fn write_mode_two(&mut self, mode: ModeTwo) {
        let bits = match mode {
            ModeTwo::M10_555 | ModeTwo::M7_666 => 2,
            _ => 5,
        };
        self.write_u64(mode as u64, bits);
    }

    fn finish(self) -> [u8; 16] {
        debug_assert!(self.bits == 128);
        self.data.to_le_bytes()
    }
}

/// Inverse function of `unquantize`
fn quantize_int16_to_bitwidth_round(
    mut component: i32,
    u_bits_per_comp: u8,
    format: BC6HFormat,
) -> i32 {
    match format {
        BC6HFormat::UnsignedF16 => {
            debug_assert!((0..65536).contains(&component));

            if u_bits_per_comp >= 15 {
                component
            } else if component <= 0 {
                0
            } else if component >= 0xffff {
                (1 << u_bits_per_comp) - 1
            } else {
                (component << u_bits_per_comp) >> 16
            }
        }
        BC6HFormat::SignedF16 => {
            debug_assert!((-32768..32768).contains(&component));

            if u_bits_per_comp >= 16 {
                component
            } else {
                let mut s = false;
                if component < 0 {
                    s = true;
                    component = -component;
                }

                let q = if component == 0 {
                    0
                } else if component >= 0x7FFF {
                    (1 << (u_bits_per_comp - 1)) - 1
                } else {
                    (component << (u_bits_per_comp - 1)) >> 15
                };

                if s {
                    -q
                } else {
                    q
                }
            }
        }
    }
}
/// Same as `quantize_int16_to_bitwidth_round`, but instead of rounding, it returns the floor and ceil values.
fn quantize_int16_to_bitwidth_floor_ceil(
    mut component: i32,
    u_bits_per_comp: u8,
    format: BC6HFormat,
) -> (i32, i32) {
    match format {
        BC6HFormat::UnsignedF16 => {
            debug_assert!((0..65536).contains(&component));

            let max = (1 << u_bits_per_comp) - 1;
            if u_bits_per_comp >= 15 {
                (component, component)
            } else if component <= 0 {
                (0, 0)
            } else if component >= 0xffff {
                (max, max)
            } else {
                let floor = ((component << u_bits_per_comp) - 0x8000).max(0) >> 16;
                let ceil = (((component << u_bits_per_comp) + 0x7fff) >> 16).min(max);
                (floor, ceil)
            }
        }
        BC6HFormat::SignedF16 => {
            debug_assert!((-32768..32768).contains(&component));

            if u_bits_per_comp >= 16 {
                (component, component)
            } else {
                let mut s = false;
                if component < 0 {
                    s = true;
                    component = -component;
                }

                let max = (1 << (u_bits_per_comp - 1)) - 1;
                let (floor, ceil) = if component == 0 {
                    (0, 0)
                } else if component >= 0x7FFF {
                    (max, max)
                } else {
                    let floor = ((component << (u_bits_per_comp - 1)) - 0x4000).max(0) >> 15;
                    let ceil = (((component << (u_bits_per_comp - 1)) + 0x3fff) >> 15).min(max);
                    (floor, ceil)
                };

                if s {
                    (-ceil, -floor)
                } else {
                    (floor, ceil)
                }
            }
        }
    }
}
/// Inverse function of `finish_unquantize`
fn quantize_half_to_int16(half: u16, format: BC6HFormat) -> i32 {
    match format {
        BC6HFormat::UnsignedF16 => {
            debug_assert!(half & 0x8000 == 0, "Must be unsigned");
            // undo scaling by 31/64 (round nearest)
            ((half as i32 * 64 + 15) / 31).min(0xffff)
        }
        BC6HFormat::SignedF16 => {
            let negative = half & 0x8000 != 0;
            let unsigned = (half & 0x7FFF) as i32;
            // undo scaling by 31/32 (round nearest)
            let component = ((unsigned * 32 + 15) / 31).min(0x7fff);
            // undo sign
            if negative {
                -component
            } else {
                component
            }
        }
    }
}

fn f32_to_half(mut value: f32, format: BC6HFormat) -> u16 {
    debug_assert!(!value.is_nan());

    #[allow(clippy::manual_clamp)]
    if format == BC6HFormat::UnsignedF16 {
        value = value.max(0.0).min(F16_MAX);
    }

    crate::color::fp16::from_f32(value)
}
fn quantize_vec3_to_int16_color(value: Vec3A, format: BC6HFormat) -> Int16Color {
    IntColor::new(
        quantize_half_to_int16(f32_to_half(value.x, format), format),
        quantize_half_to_int16(f32_to_half(value.y, format), format),
        quantize_half_to_int16(f32_to_half(value.z, format), format),
    )
    .to_int16(format)
}

fn half3_to_vec3(half: IntColor<u16>) -> Vec3A {
    Vec3A::new(
        crate::color::fp16::f32(half.r),
        crate::color::fp16::f32(half.g),
        crate::color::fp16::f32(half.b),
    )
}

fn can_compress_endpoints_one(endpoints: EndPointPair, mode: ModeOne) -> bool {
    // TODO: Implement this more efficiently
    compress_endpoints_one(endpoints, mode).is_some()
}
fn compress_endpoints_one(
    EndPointPair { a, b }: EndPointPair,
    mode: ModeOne,
) -> Option<EndPointPair> {
    let a_bits = mode.a0_bit_count();
    let a_mask = (1 << a_bits) - 1;

    if !mode.transformed() {
        return Some(EndPointPair {
            a: a.bit_and(a_mask),
            b: b.bit_and(a_mask),
        });
    }

    let b_bits = mode.b0_bit_count();
    let b = get_transformed_endpoint(a, b, a_bits, [b_bits; 3])?;

    Some(EndPointPair {
        a: a.bit_and(a_mask),
        b,
    })
}
/// If `true`, the endpoints can be transformed even if swapped.
/// If `false`, the endpoints cannot be transformed even if swapped.
/// If `None`, it depends on which way they are swapped whether the endpoints can be transformed.
fn can_compress_endpoints_two(endpoints: [EndPointPair; 2], mode: ModeTwo) -> Option<bool> {
    if !mode.transformed() {
        return Some(true);
    }

    // In mode two, two swaps are possible: for the endpoints of subset 0 and subset 1.
    // Importantly, the swap of subset 1 doesn't matter, since endpoints a1 and b1 are both transformed relative to a0.
    // The swap for subset 0 is important, so we must assume that it happens

    let a_bits = mode.a0_bit_count();
    let delta_bits = mode.delta_bit_count();
    let can_transform = move |a0: IntColor<i32>, other: IntColor<i32>| -> bool {
        get_transformed_endpoint(a0, other, a_bits, delta_bits).is_some()
    };

    if !can_transform(endpoints[0].a, endpoints[0].b) {
        // If the first subset cannot be transformed, then the endpoints cannot be transformed even if swapped.
        // This *technically* isn't true, but almost. If a-b is 2^(n-1) for n-bit delta, then the
        // transformation would succeed if swapped, since -2^(n-1) can be represented by a signed n-bit integer.
        // That said, these cases are rare in practice and not worth checking here.
        return Some(false);
    }

    // test both swap directions
    let swap_non = can_transform(endpoints[0].a, endpoints[1].a)
        && can_transform(endpoints[0].a, endpoints[1].b);
    let swap_yes = can_transform(endpoints[0].b, endpoints[1].a)
        && can_transform(endpoints[0].b, endpoints[1].b);

    // If both swaps agree on whether the endpoints can be transformed, return that result.
    if swap_non == swap_yes {
        return Some(swap_non);
    }
    None
}
fn compress_endpoints_two(
    endpoints: [EndPointPair; 2],
    mode: ModeTwo,
) -> Option<[EndPointPair; 2]> {
    let a_bits = mode.a0_bit_count();
    let a0_mask = (1 << a_bits) - 1;

    if !mode.transformed() {
        return Some(endpoints.map(|e| EndPointPair {
            a: e.a.bit_and(a0_mask),
            b: e.b.bit_and(a0_mask),
        }));
    }

    let a0 = endpoints[0].a;
    let b0 = endpoints[0].b;
    let a1 = endpoints[1].a;
    let b1 = endpoints[1].b;

    let delta_bits = mode.delta_bit_count();
    let b0 = get_transformed_endpoint(a0, b0, a_bits, delta_bits)?;
    let a1 = get_transformed_endpoint(a0, a1, a_bits, delta_bits)?;
    let b1 = get_transformed_endpoint(a0, b1, a_bits, delta_bits)?;

    Some([
        EndPointPair {
            a: a0.bit_and(a0_mask),
            b: b0,
        },
        EndPointPair { a: a1, b: b1 },
    ])
}
fn get_transformed_endpoint(
    a: IntColor<i32>,
    b: IntColor<i32>,
    a_bits: u8,
    b_bits: [u8; 3],
) -> Option<IntColor<i32>> {
    let r = get_transform_diff(a.r, b.r, a_bits);
    let g = get_transform_diff(a.g, b.g, a_bits);
    let b = get_transform_diff(a.b, b.b, a_bits);

    let r_max = 1 << (b_bits[0] - 1);
    let g_max = 1 << (b_bits[1] - 1);
    let b_max = 1 << (b_bits[2] - 1);

    if -r_max <= r && r < r_max && -g_max <= g && g < g_max && -b_max <= b && b < b_max {
        return Some(IntColor::new(
            r & ((1 << b_bits[0]) - 1),
            g & ((1 << b_bits[1]) - 1),
            b & ((1 << b_bits[2]) - 1),
        ));
    }

    None
}
fn get_transform_diff(mut a: i32, mut b: i32, a_bits: u8) -> i32 {
    let mask = (1 << a_bits) - 1;
    let shift = 1 << (a_bits - 1);

    a &= mask;
    b &= mask;

    let diff1 = b - a;
    let diff2 = ((b + shift) & mask) - ((a + shift) & mask);

    let diff1_abs = diff1.abs();
    let diff2_abs = diff2.abs();

    if diff1_abs < diff2_abs {
        diff1
    } else if diff2_abs < diff1_abs {
        diff2
    } else {
        // If they only differ by sign, pick the smaller one.
        // This is important, because e.g. 4-bit values can represent -8 but not +8.
        diff1.min(diff2)
    }
}
#[test]
fn test_get_transform_diff() {
    // unsigned 8 bit
    for a in 0..256 {
        for b in 0..256 {
            let diff = get_transform_diff(a, b, 8);
            let b_after_transform = (a + diff) & 0xff;
            assert_eq!(b_after_transform, b);
        }
    }

    // signed 8 bit
    for a in -128..128 {
        for b in -128..128 {
            let diff = get_transform_diff(a, b, 8);
            let b_after_transform = sign_extend((a + diff) & 0xff, 8);
            assert_eq!(b_after_transform, b);
        }
    }

    fn sign_extend(x: i32, bit_count: u8) -> i32 {
        debug_assert!(bit_count > 0);
        debug_assert!(bit_count < 32);

        // check that all bits outsize bit_count are zero
        debug_assert_eq!(x & !((1 << bit_count) - 1), 0);

        let shift = 32 - bit_count;
        (x << shift) >> shift
    }
}
