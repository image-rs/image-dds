#![allow(clippy::needless_range_loop)]

use bitflags::bitflags;
use glam::{IVec4, Vec3A, Vec4};

use crate::{
    bcn_data::{
        Subset2Map, Subset3Map, PARTITION_SET_2, PARTITION_SET_3, PARTITION_SET_3_DUPLICATES,
        PARTITION_SET_3_DUPLICATE_COUNT,
    },
    encode::bcn_util::{self, ColorLine3, ColorLine4, Quantization, Quantized, WithChannels},
};

bitflags! {
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct Bc7Modes: u8 {
        const MODE0 = 1 << 0;
        const MODE1 = 1 << 1;
        const MODE2 = 1 << 2;
        const MODE3 = 1 << 3;
        const MODE4 = 1 << 4;
        const MODE5 = 1 << 5;
        const MODE6 = 1 << 6;
        const MODE7 = 1 << 7;
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Bc7Options {
    /// Allowed BC7 modes for compression.
    pub allowed_modes: Bc7Modes,

    /// Instead of considering the full range of 64 partitions, only consider
    /// the first K partitions.
    ///
    /// This is not the same as `max_partition_candidates`. The encoder will
    /// first score the first `max_partitions` partitions and then try the best
    /// `max_partition_candidates` of those.
    ///
    /// Must be between 1 and 64.
    pub max_partitions: u8,
    /// The maximum number of partitions to try per mode.
    ///
    /// Must be between 1 and 64.
    pub max_partition_candidates: u8,
    /// Instead of using the 16 partitions for mode 0, only a smaller subset of 10.
    ///
    /// This only applies when mode 1 is enabled, mode 2 is disabled, and the
    /// block is opaque.
    pub fast_mode_0_partitions: bool,

    pub quantization: Quantization,

    /// If true, the encoder will consider color channel rotations for modes 4 and 5.
    pub allow_color_rotation: bool,

    /// If color rotation for mode 4 and 5 is allowed, this controls how many
    /// rotations are tried.
    ///
    /// If set to 4, all rotations are tried. IF set to less then 4, a heuristic
    /// is used to pick the best rotations to try.
    ///
    /// Must be between 1 and 4.
    pub max_color_rotations: u8,

    /// The maximum number of refinement iterations for endpoint optimization.
    pub max_refinement_iters: u8,

    /// The maximum number of p-bit combinations to try per subset per mode.
    ///
    /// If this number is less then 4, a heuristic is used to pick a good p-bit
    /// (or two) instead of trying all combinations.
    ///
    /// Must be 1, 2, or 4.
    pub max_p_bit_combinations: u8,

    /// Whether to least squares fit the endpoints before refinement.
    pub fit_optimal: bool,

    /// Forces the encoder to use exactly the specified BC7 modes. If none are
    /// specified, this option is ignored.
    ///
    /// This is mainly for testing purposes.
    pub force_modes: Bc7Modes,
}

pub(crate) fn compress_bc7_block(block: [Rgba<8>; 16], options: Bc7Options) -> [u8; 16] {
    let stats = BlockStats::new(&block);

    // a block of a single color can always be compressed exactly
    if let Some(color) = stats.single_color() {
        return compress_single_color(color).block;
    }

    let mut modes = Bc7Modes::empty();

    // for modes, we want to first determine which modes are sensible to use and
    // then filter by allowed modes
    modes |= Bc7Modes::MODE4 | Bc7Modes::MODE5; // always useful
    if stats.opaque() {
        // modes exclusively for opaque blocks
        modes |= Bc7Modes::MODE0 | Bc7Modes::MODE1 | Bc7Modes::MODE2 | Bc7Modes::MODE3;
    } else {
        // modes exclusively for transparent blocks
        // (mode 7 is strictly worse than mode 3 for opaque blocks)
        modes |= Bc7Modes::MODE7;
    }
    // Mode 6 uses combined Color+Alpha. This is a problem for blocks that
    // contain both opaque and partially transparent pixels, because the opaque
    // pixels will be become partially transparent as well. Since we want to
    // ensure that opaque pixels stay opaque, we only try mode 6 if the block
    // is fully opaque or doesn't contain any opaque pixels. Constant alpha is
    // also fine.
    // Lastly, if no other alpha mode is allowed, it's better than nothing.
    if stats.single_alpha().is_some()
        || stats.max.a != 255
        || (stats.min.a != 255
            && !options
                .allowed_modes
                .intersects(Bc7Modes::MODE4 | Bc7Modes::MODE5 | Bc7Modes::MODE7))
    {
        modes |= Bc7Modes::MODE6;
    }

    // Filter by allowed modes
    modes &= options.allowed_modes;
    if modes.is_empty() {
        // no allowed modes left, fall back to all modes
        modes = options.allowed_modes;
        debug_assert!(!modes.is_empty());
    }

    // Overwrite with force mode (if any)
    if !options.force_modes.is_empty() {
        modes = options.force_modes;
    }

    let mut partitions = PartitionSelect::new(&block, stats, modes, options);
    let mut rotations = RotationSelect::new(&block, stats);

    let mut best = Compressed::invalid();
    if modes.contains(Bc7Modes::MODE0) {
        best = best.better(compress_mode0(&mut partitions, options));
    }
    if modes.contains(Bc7Modes::MODE1) {
        best = best.better(compress_mode1(&mut partitions, options));
    }
    if modes.contains(Bc7Modes::MODE2) {
        best = best.better(compress_mode2(&mut partitions, options));
    }
    if modes.contains(Bc7Modes::MODE3) {
        best = best.better(compress_mode3(&mut partitions, options));
    }
    if modes.contains(Bc7Modes::MODE4) {
        best = best.better(compress_mode4(&mut rotations, options));
    }
    if modes.contains(Bc7Modes::MODE5) {
        best = best.better(compress_mode5(&mut rotations, options));
    }
    if modes.contains(Bc7Modes::MODE6) {
        best = best.better(compress_mode6(block, options));
    }
    if modes.contains(Bc7Modes::MODE7) {
        best = best.better(compress_mode7(&mut partitions, options));
    }

    best.block
}

/// Solid-color blocks can be encoded exactly.
///
/// https://fgiesen.wordpress.com/2024/11/03/bc7-optimal-solid-color-blocks/
fn compress_single_color(color: Rgba<8>) -> Compressed {
    fn optimize(c: u8) -> (u8, u8) {
        (c >> 1, if c < 128 { c + 1 } else { c - 1 } >> 1)
    }

    let (c0_r, c1_r) = optimize(color.r);
    let (c0_g, c1_g) = optimize(color.g);
    let (c0_b, c1_b) = optimize(color.b);

    let c0 = Rgb::new(c0_r, c0_g, c0_b);
    let c1 = Rgb::new(c1_r, c1_g, c1_b);

    Compressed::mode5(
        0, // exact, so 0 error
        Rotation::None,
        [c0, c1],
        IndexList::<2>::constant(1),
        [Alpha::new(color.a); 2],
        // the index for alpha doesn't matter since both endpoints are the same
        IndexList::<2>::constant(1),
    )
}

fn compress_mode0(partitions: &mut PartitionSelect, options: Bc7Options) -> Compressed {
    let block = partitions.block.map(|p| p.color());

    partitions.pick_best_partition_3(true, options, |partition| {
        let subset = PARTITION_SET_3[partition as usize];

        let mut reordered = block;
        subset.sort_block(&mut reordered);
        let split_index = subset.count_zeros() as usize;
        let split_index2 = split_index + subset.count_ones() as usize;

        // subset0 and subset1
        let (error_s0, [e0_s0, e1_s0], [p0_s0, p1_s0], indexes_s0) =
            compress_rgb(&reordered[..split_index], UniquePBits, options);
        let (error_s1, [e0_s1, e1_s1], [p0_s1, p1_s1], indexes_s1) =
            compress_rgb(&reordered[split_index..split_index2], UniquePBits, options);
        let (error_s2, [e0_s2, e1_s2], [p0_s2, p1_s2], indexes_s2) =
            compress_rgb(&reordered[split_index2..], UniquePBits, options);

        Compressed::mode0(
            error_s0 + error_s1 + error_s2,
            partition,
            [e0_s0, e1_s0, e0_s1, e1_s1, e0_s2, e1_s2],
            [p0_s0, p1_s0, p0_s1, p1_s1, p0_s2, p1_s2],
            IndexList::merge3(subset, indexes_s0, indexes_s1, indexes_s2),
        )
    })
}
fn compress_mode1(partitions: &mut PartitionSelect, options: Bc7Options) -> Compressed {
    let block_rgb = partitions.block.map(|p| p.color());

    partitions.pick_best_partition_2(options, |partition| {
        let subset = PARTITION_SET_2[partition as usize];

        let mut reordered = block_rgb;
        subset.sort_block(&mut reordered);
        let split_index = subset.count_zeros() as usize;

        // subset0 and subset1
        let (error_s0, [e0_s0, e1_s0], p_s0, indexes_s0) =
            compress_rgb(&reordered[..split_index], SharedPBit, options);
        let (error_s1, [e0_s1, e1_s1], p_s1, indexes_s1) =
            compress_rgb(&reordered[split_index..], SharedPBit, options);

        Compressed::mode1(
            error_s0 + error_s1,
            partition,
            [e0_s0, e1_s0, e0_s1, e1_s1],
            [p_s0, p_s1],
            IndexList::merge2(subset, indexes_s0, indexes_s1),
        )
    })
}
fn compress_mode2(partitions: &mut PartitionSelect, options: Bc7Options) -> Compressed {
    let block = partitions.block.map(|p| p.color());

    partitions.pick_best_partition_3(false, options, |partition| {
        let subset = PARTITION_SET_3[partition as usize];

        let mut reordered = block;
        subset.sort_block(&mut reordered);
        let split_index = subset.count_zeros() as usize;
        let split_index2 = split_index + subset.count_ones() as usize;

        // subset0 and subset1
        let (error_s0, [e0_s0, e1_s0], _, indexes_s0) =
            compress_rgb(&reordered[..split_index], NoPBit, options);
        let (error_s1, [e0_s1, e1_s1], _, indexes_s1) =
            compress_rgb(&reordered[split_index..split_index2], NoPBit, options);
        let (error_s2, [e0_s2, e1_s2], _, indexes_s2) =
            compress_rgb(&reordered[split_index2..], NoPBit, options);

        Compressed::mode2(
            error_s0 + error_s1 + error_s2,
            partition,
            [e0_s0, e1_s0, e0_s1, e1_s1, e0_s2, e1_s2],
            IndexList::merge3(subset, indexes_s0, indexes_s1, indexes_s2),
        )
    })
}
fn compress_mode3(partitions: &mut PartitionSelect, options: Bc7Options) -> Compressed {
    let block_rgb = partitions.block.map(|p| p.color());

    partitions.pick_best_partition_2(options, |partition| {
        let subset = PARTITION_SET_2[partition as usize];

        let mut reordered = block_rgb;
        subset.sort_block(&mut reordered);
        let split_index = subset.count_zeros() as usize;

        // subset0 and subset1
        let (error_s0, [e0_s0, e1_s0], [p0_s0, p1_s0], indexes_s0) =
            compress_rgb(&reordered[..split_index], UniquePBits, options);
        let (error_s1, [e0_s1, e1_s1], [p0_s1, p1_s1], indexes_s1) =
            compress_rgb(&reordered[split_index..], UniquePBits, options);

        Compressed::mode3(
            error_s0 + error_s1,
            partition,
            [e0_s0, e1_s0, e0_s1, e1_s1],
            [p0_s0, p1_s0, p0_s1, p1_s1],
            IndexList::merge2(subset, indexes_s0, indexes_s1),
        )
    })
}
fn compress_mode4(rotations: &mut RotationSelect, options: Bc7Options) -> Compressed {
    let block = rotations.block;
    let stats = rotations.stats;

    rotations.pick_best(options, |r| {
        let c3a2 = {
            let (error, color, color_indexes, alpha, alpha_indexes) =
                compress_color_separate_alpha_with_rotation(*block, stats, options, r);
            Compressed::mode4(
                error,
                r,
                IndexMode::C3A2,
                color,
                alpha_indexes,
                alpha,
                color_indexes,
            )
        };

        // If the separated channel is a constant value, then we don't want to
        // waste the 3-bit per pixel indexes on it.
        let c = r.channel();
        if stats.min.get(c) == stats.max.get(c) {
            return c3a2;
        }

        let c2a3 = {
            let (error, color, color_indexes, alpha, alpha_indexes) =
                compress_color_separate_alpha_with_rotation(*block, stats, options, r);
            Compressed::mode4(
                error,
                r,
                IndexMode::C2A3,
                color,
                color_indexes,
                alpha,
                alpha_indexes,
            )
        };
        c2a3.better(c3a2)
    })
}
fn compress_mode5(rotations: &mut RotationSelect, options: Bc7Options) -> Compressed {
    let block = rotations.block;
    let stats = rotations.stats;

    rotations.pick_best(options, |r| {
        let (error, color, color_indexes, alpha, alpha_indexes) =
            compress_color_separate_alpha_with_rotation(*block, stats, options, r);
        Compressed::mode5(error, r, color, color_indexes, alpha, alpha_indexes)
    })
}
fn compress_mode6(block: [Rgba<8>; 16], options: Bc7Options) -> Compressed {
    let (error, [e0, e1], [p0, p1], indexes) = compress_rgba(&block, options);
    Compressed::mode6(error, [e0, e1], [p0, p1], indexes)
}
fn compress_mode7(partitions: &mut PartitionSelect, options: Bc7Options) -> Compressed {
    partitions.pick_best_partition_2(options, |partition| {
        let subset = PARTITION_SET_2[partition as usize];

        let mut reordered = *partitions.block;
        subset.sort_block(&mut reordered);
        let split_index = subset.count_zeros() as usize;

        // subset0 and subset1
        let (error_s0, [e0_s0, e1_s0], [p0_s0, p1_s0], indexes_s0) =
            compress_rgba(&reordered[..split_index], options);
        let (error_s1, [e0_s1, e1_s1], [p0_s1, p1_s1], indexes_s1) =
            compress_rgba(&reordered[split_index..], options);

        Compressed::mode7(
            error_s0 + error_s1,
            partition,
            [e0_s0, e1_s0, e0_s1, e1_s1],
            [p0_s0, p1_s0, p0_s1, p1_s1],
            IndexList::merge2(subset, indexes_s0, indexes_s1),
        )
    })
}

/// Strategy for which rotations to use for mode 4 and mode 5.
struct RotationSelect<'a> {
    block: &'a [Rgba<8>; 16],
    stats: BlockStats,

    forced_rotation_cache: Option<Option<Rotation>>,
    rotations_cache: Option<[Rotation; 4]>,
}
impl<'a> RotationSelect<'a> {
    fn new(block: &'a [Rgba<8>; 16], stats: BlockStats) -> RotationSelect<'a> {
        RotationSelect {
            block,
            stats,

            forced_rotation_cache: None,
            rotations_cache: None,
        }
    }

    fn get_forced_rotation(&mut self, options: Bc7Options) -> Option<Rotation> {
        let stats = self.stats;
        let block = self.block;

        *self.forced_rotation_cache.get_or_insert_with(|| {
            // Color rotation may be disabled
            if !options.allow_color_rotation {
                return Some(Rotation::None);
            }

            // Having a low error for alpha is important for visual quality, so we only
            // want to swap alpha with other channels if there isn't much in the alpha
            // channel.
            const ALPHA_THRESHOLD: u8 = 16;
            if stats.max.a.abs_diff(stats.min.a) > ALPHA_THRESHOLD {
                return Some(Rotation::None);
            }

            // If the color is approximately grayscale, don't try swapping.
            // It will only cause discolorations.
            const COLOR_VARIANCE_THRESHOLD: u8 = 8;
            if stats.is_grayscale
                || block.iter().all(|p| {
                    let gr = p.g.abs_diff(p.r);
                    let gb = p.g.abs_diff(p.b);
                    gr.max(gb) < COLOR_VARIANCE_THRESHOLD
                })
            {
                return Some(Rotation::None);
            }

            None
        })
    }
    fn get_rotations(&mut self, options: Bc7Options) -> &[Rotation; 4] {
        let stats = self.stats;
        let block = self.block;

        const ALL: [Rotation; 4] = [Rotation::None, Rotation::AR, Rotation::AG, Rotation::AB];

        self.rotations_cache.get_or_insert_with(|| {
            if options.max_color_rotations >= 4 {
                return ALL;
            }

            let mut scored: [(Rotation, f32); 4] = ALL.map(|r| {
                let channel = r.channel();
                if stats.min.get(channel) == stats.max.get(channel) {
                    // no point in swapping with a constant channel
                    return (r, f32::INFINITY);
                }

                let block3 = match r {
                    Rotation::None => block.map(|p| p.color().to_vec()),
                    Rotation::AR => block.map(|p| Rgb::<8>::new(p.a, p.g, p.b).to_vec()),
                    Rotation::AG => block.map(|p| Rgb::<8>::new(p.r, p.a, p.b).to_vec()),
                    Rotation::AB => block.map(|p| Rgb::<8>::new(p.r, p.g, p.a).to_vec()),
                };

                let error = squared_error_line_3(&block3);
                (r, error)
            });
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

            scored.map(|(r, _)| r)
        })
    }

    fn pick_best(
        &mut self,
        options: Bc7Options,
        mut f: impl FnMut(Rotation) -> Compressed,
    ) -> Compressed {
        if let Some(r) = self.get_forced_rotation(options) {
            return f(r);
        }

        let stats = self.stats;
        let rotations = self.get_rotations(options);

        let mut best = Compressed::invalid();
        for &r in rotations.iter().take(options.max_color_rotations as usize) {
            let channel = r.channel();
            if stats.min.get(channel) == stats.max.get(channel) {
                // no point in swapping with a constant channel
                continue;
            }

            best = best.better(f(r))
        }

        best
    }
}
/// Compression function for mode 4 and mode 5.
fn compress_color_separate_alpha_with_rotation<
    const C: u8,
    const A: u8,
    const IC: u8,
    const IA: u8,
>(
    mut block: [Rgba<8>; 16],
    mut stats: BlockStats,
    options: Bc7Options,
    rotation: Rotation,
) -> (
    u32,
    [Rgb<C>; 2],
    IndexList<IC>,
    [Alpha<A>; 2],
    IndexList<IA>,
) {
    // Apply rotation
    if rotation != Rotation::None {
        block = rotation.apply(block);
        stats = rotation.apply_stats(stats);
    }

    // RGB
    let rgb = block.map(|p| p.color());
    let rgb_vec = rgb.map(|p| p.to_vec());
    let (c0, c1) = bcn_util::line3_fit_endpoints(&rgb_vec, 0.9);
    let (c0, c1) = fit_optimal_endpoints_rgb::<C, IC>(c0, c1, &rgb, &rgb_vec, options);
    let (c0, c1) = refine_along_line3(c0, c1, 1 << IC, options, |(min, max)| {
        closest_error_rgb::<IC>(Rgb::round(min), Rgb::round(max), &rgb)
    });
    let (e0, e1) = options
        .quantization
        .pick_best(c0, c1, |e0: Rgb<C>, e1: Rgb<C>| {
            closest_error_rgb::<IC>(e0.promote(), e1.promote(), &rgb)
        });
    let (color_indexes, color_error) = closest_rgb::<IC>(e0.promote(), e1.promote(), &rgb);
    let color_endpoints = [e0, e1];

    // Alpha
    let alpha_pixels = block.map(|p| p.alpha());
    let (alpha_endpoints, alpha_indexes, alpha_error) = if let Some(a) = stats.single_alpha() {
        let a_f = a as f32 * (1.0 / 255.0);
        let round = Alpha::<A>::round(a_f);
        if round.promote().a == a {
            ([round; 2], IndexList::<IA>::constant(0), 0) // exact, so 0 error
        } else {
            let floor = Alpha::<A>::floor(a_f);
            let ceil = Alpha::<A>::ceil(a_f);
            let (indexes, error) =
                closest_alpha::<IA>(floor.promote(), ceil.promote(), &alpha_pixels);
            ([floor, ceil], indexes, error)
        }
    } else {
        // We want opaque pixels to stay opaque no matter what.
        let force_opaque = rotation == Rotation::None && stats.max.a == 255;
        let c = (stats.min.alpha().to_vec(), stats.max.alpha().to_vec());
        let c = fit_optimal_endpoints_alpha::<A, IA>(c.0, c.1, &alpha_pixels, options);
        let c = bcn_util::refine_endpoints(
            c.0,
            c.1,
            bcn_util::RefinementOptions {
                step_initial: (c.1 - c.0) * 0.2,
                step_decay: 0.5,
                step_min: 1.0 / 255.0,
                max_iter: options.max_refinement_iters as u32,
            },
            |(min, max)| {
                if force_opaque && max < 1.0 {
                    return u32::MAX;
                }
                closest_error_alpha::<IA>(Alpha::floor(min), Alpha::ceil(max), &alpha_pixels)
            },
        );
        let a_min: Alpha<A> = Alpha::floor(c.0);
        let a_max: Alpha<A> = Alpha::ceil(c.1);
        let (alpha_indexes, alpha_error) =
            closest_alpha::<IA>(a_min.promote(), a_max.promote(), &alpha_pixels);
        ([a_min, a_max], alpha_indexes, alpha_error)
    };

    (
        color_error + alpha_error,
        color_endpoints,
        color_indexes,
        alpha_endpoints,
        alpha_indexes,
    )
}

fn compress_rgba<const B: u8, const I: u8>(
    block: &[Rgba<8>],
    options: Bc7Options,
) -> (u32, [Rgba<B>; 2], [bool; 2], IndexList<I>) {
    debug_assert!(block.len() <= 16);
    debug_assert!(block.len() >= 2);

    // RGBA
    let mut rgba_vec = [Vec4::ZERO; 16];
    for (i, p) in block.iter().enumerate() {
        rgba_vec[i] = p.to_vec();
    }
    let rgba_vec = &rgba_vec[..block.len()];

    let (c0, c1) = bcn_util::line4_fit_endpoints(rgba_vec, 0.9);
    let (c0, c1) = fit_optimal_endpoints_rgba::<B, I>(c0, c1, block, rgba_vec, options);
    let (c0, c1) = refine_along_line4(c0, c1, 1 << I, options, |(min, max)| {
        closest_error_rgba::<I>(Rgba::round(min), Rgba::round(max), block)
    });

    // Quantize
    // If all pixels are opaque, the p-bits have to be (1,1) to keep them opaque.
    let opaque = block.iter().all(|p| p.a == 255);
    let possible_p_bits: Option<&[[bool; 2]]> = if opaque { Some(&[[true, true]]) } else { None };

    let estimate = estimate_rgba::<B>(c0, c1);
    let (error, (es, indexes), p) =
        UniquePBits.pick_best(possible_p_bits, estimate, options, |p| {
            let (e0, e1) = options
                .quantization
                .pick_best(c0, c1, |e0: Rgba<B>, e1: Rgba<B>| {
                    let e8 = p.promote_rgba(e0, e1);
                    closest_error_rgba::<I>(e8[0], e8[1], block)
                });
            let e8 = p.promote_rgba(e0, e1);
            let (indexes, error) = closest_rgba::<I>(e8[0], e8[1], block);
            (error, ([e0, e1], indexes))
        });

    (error, es, p, indexes)
}
fn compress_rgb<const B: u8, const I: u8, State: PBitState>(
    block: &[Rgb<8>],
    p_bit: impl PBitHandling<State = State>,
    options: Bc7Options,
) -> (u32, [Rgb<B>; 2], State, IndexList<I>) {
    debug_assert!(block.len() <= 16);
    debug_assert!(block.len() >= 2);

    // RGB
    let mut rgb_vec = [Vec3A::ZERO; 16];
    for (i, p) in block.iter().enumerate() {
        rgb_vec[i] = p.to_vec();
    }
    let rgb_vec = &rgb_vec[..block.len()];

    let (c0, c1) = bcn_util::line3_fit_endpoints(rgb_vec, 0.9);
    let (c0, c1) = fit_optimal_endpoints_rgb::<B, I>(c0, c1, block, rgb_vec, options);
    let (c0, c1) = refine_along_line3(c0, c1, 1 << I, options, |(min, max)| {
        closest_error_rgb::<I>(Rgb::round(min), Rgb::round(max), block)
    });

    // pick the best p-bit configuration
    let estimate = estimate_rgb::<B>(c0, c1);
    let (error, (es, indexes), p) = p_bit.pick_best(None, estimate, options, |p| {
        let (e0, e1) = options
            .quantization
            .pick_best(c0, c1, |e0: Rgb<B>, e1: Rgb<B>| {
                let e8 = p.promote_rgb(e0, e1);
                closest_error_rgb::<I>(e8[0], e8[1], block)
            });
        let e8 = p.promote_rgb(e0, e1);
        let (indexes, error) = closest_rgb::<I>(e8[0], e8[1], block);
        (error, ([e0, e1], indexes))
    });

    (error, es, p, indexes)
}

trait EstimatePQuantizationError<const B: u8> {
    fn estimate(&self, p: impl PBitState) -> f32;

    fn get_best<P: PBitState + Copy>(&self, states: &[P]) -> P {
        let mut best_state = states[0];
        let mut best_error = self.estimate(best_state);

        for &p in &states[1..] {
            let error = self.estimate(p);
            if error < best_error {
                best_error = error;
                best_state = p;
            }
        }

        best_state
    }
    fn get_best_2<P: PBitState + Copy>(&self, states: &[P]) -> [P; 2] {
        debug_assert!(states.len() >= 2);

        let mut best_states = [states[0], states[1]];
        let mut best_error = [self.estimate(best_states[0]), self.estimate(best_states[1])];

        if best_error[1] < best_error[0] {
            best_error.swap(0, 1);
            best_states.swap(0, 1);
        }

        for &p in &states[2..] {
            let error = self.estimate(p);
            if error < best_error[0] {
                best_error[1] = best_error[0];
                best_states[1] = best_states[0];
                best_error[0] = error;
                best_states[0] = p;
            } else if error < best_error[1] {
                best_error[1] = error;
                best_states[1] = p;
            }
        }

        best_states
    }
}
impl<const B: u8> EstimatePQuantizationError<B> for (Vec3A, Vec3A) {
    fn estimate(&self, p: impl PBitState) -> f32 {
        let (c0, c1) = *self;
        let [e0, e1] = p.promote_rgb(Rgb::<B>::round(c0), Rgb::<B>::round(c1));
        let err0 = e0.to_vec().distance_squared(c0);
        let err1 = e1.to_vec().distance_squared(c1);
        err0 + err1
    }
}
impl<const B: u8> EstimatePQuantizationError<B> for (Vec4, Vec4) {
    fn estimate(&self, p: impl PBitState) -> f32 {
        let (c0, c1) = *self;
        let [e0, e1] = p.promote_rgba(Rgba::<B>::round(c0), Rgba::<B>::round(c1));
        let err0 = e0.to_vec().distance_squared(c0);
        let err1 = e1.to_vec().distance_squared(c1);
        err0 + err1
    }
}
fn estimate_rgb<const B: u8>(c0: Vec3A, c1: Vec3A) -> impl EstimatePQuantizationError<B> {
    (c0, c1)
}
fn estimate_rgba<const B: u8>(c0: Vec4, c1: Vec4) -> impl EstimatePQuantizationError<B> {
    (c0, c1)
}

trait PBitHandling {
    type State: PBitState + Copy + 'static;
    const ALL: &'static [Self::State];

    fn pick_best<const B: u8, T>(
        &self,
        all: Option<&[Self::State]>,
        e: impl EstimatePQuantizationError<B>,
        options: Bc7Options,
        f: impl Fn(Self::State) -> (u32, T),
    ) -> (u32, T, Self::State) {
        let all = all.unwrap_or(Self::ALL);

        debug_assert!(!all.is_empty());
        if all.len() == 1 || all.len() <= options.max_p_bit_combinations as usize {
            return self.pick_best_of_directly(all, f);
        }

        if options.max_p_bit_combinations == 1 {
            let best = e.get_best(all);
            return self.pick_best_of_directly(&[best], f);
        }
        if options.max_p_bit_combinations == 2 {
            let best = e.get_best_2(all);
            return self.pick_best_of_directly(&best, f);
        }

        self.pick_best_of_directly(Self::ALL, f)
    }
    fn pick_best_of_directly<T>(
        &self,
        possibilities: &[Self::State],
        f: impl Fn(Self::State) -> (u32, T),
    ) -> (u32, T, Self::State) {
        debug_assert!(!possibilities.is_empty());

        let mut best_error;
        let mut best_t;
        let mut best_state = possibilities[0];
        (best_error, best_t) = f(best_state);

        for &p in &possibilities[1..] {
            let (error, t) = f(p);
            if error < best_error {
                best_error = error;
                best_t = t;
                best_state = p;
            }
        }

        (best_error, best_t, best_state)
    }
}
trait PBitState {
    fn promote_rgb<const B: u8>(&self, c0: Rgb<B>, c1: Rgb<B>) -> [Rgb<8>; 2];
    fn promote_rgba<const B: u8>(&self, c0: Rgba<B>, c1: Rgba<B>) -> [Rgba<8>; 2];
}

struct UniquePBits;
impl PBitHandling for UniquePBits {
    type State = [bool; 2];
    const ALL: &'static [[bool; 2]] = &[[false, false], [false, true], [true, false], [true, true]];
}
impl PBitState for [bool; 2] {
    fn promote_rgb<const B: u8>(&self, c0: Rgb<B>, c1: Rgb<B>) -> [Rgb<8>; 2] {
        [c0.p_promote(self[0]), c1.p_promote(self[1])]
    }
    fn promote_rgba<const B: u8>(&self, c0: Rgba<B>, c1: Rgba<B>) -> [Rgba<8>; 2] {
        [c0.p_promote(self[0]), c1.p_promote(self[1])]
    }
}

struct SharedPBit;
impl PBitHandling for SharedPBit {
    type State = bool;
    const ALL: &'static [bool] = &[false, true];
}
impl PBitState for bool {
    fn promote_rgb<const B: u8>(&self, c0: Rgb<B>, c1: Rgb<B>) -> [Rgb<8>; 2] {
        [c0.p_promote(*self), c1.p_promote(*self)]
    }
    fn promote_rgba<const B: u8>(&self, c0: Rgba<B>, c1: Rgba<B>) -> [Rgba<8>; 2] {
        [c0.p_promote(*self), c1.p_promote(*self)]
    }
}

#[derive(Copy, Clone)]
struct NoPBit;
impl PBitHandling for NoPBit {
    type State = NoPBit;
    const ALL: &'static [NoPBit] = &[NoPBit];
}
impl PBitState for NoPBit {
    fn promote_rgb<const B: u8>(&self, c0: Rgb<B>, c1: Rgb<B>) -> [Rgb<8>; 2] {
        [c0.promote(), c1.promote()]
    }
    fn promote_rgba<const B: u8>(&self, c0: Rgba<B>, c1: Rgba<B>) -> [Rgba<8>; 2] {
        [c0.promote(), c1.promote()]
    }
}

fn fit_optimal_endpoints_rgba<const B: u8, const I: u8>(
    c0: Vec4,
    c1: Vec4,
    block: &[Rgba<8>],
    block_vec: &[Vec4],
    options: Bc7Options,
) -> (Vec4, Vec4) {
    if !options.fit_optimal {
        return (c0, c1);
    }

    debug_assert_eq!(block.len(), block_vec.len());

    // If the endpoints are very close, then quantization artifacts have a
    // significant effect on which indexes are chosen per pixel. This is a
    // problem here, because we don't know the final quantized endpoints yet
    // and have to guess them. The closer the endpoints are, the more the
    // error in our guess matters. Since this method optimizes based on indexes,
    // if the indexes aren't good, it will return garbage.
    // This minimum distance is somewhat arbitrary. It seems to work, so I
    // haven't looked into other value.
    let min_dist = (1_u32 << I) as f32 / (1_u32 << B) as f32;
    if c0.distance_squared(c1) < min_dist * min_dist {
        return (c0, c1);
    }

    let (e0, e1): (Rgba<B>, Rgba<B>) = Quantization::wide(c0, c1);
    let index_list: IndexList<I> = closest_rgba(e0.promote(), e1.promote(), block).0;
    let weights = index_list.to_weights();

    let optimal: (Vec4, Vec4) = bcn_util::least_squares_weights(block_vec, &weights[..block.len()]);

    (optimal.0, optimal.1)
}
fn fit_optimal_endpoints_rgb<const B: u8, const I: u8>(
    c0: Vec3A,
    c1: Vec3A,
    block: &[Rgb<8>],
    block_vec: &[Vec3A],
    options: Bc7Options,
) -> (Vec3A, Vec3A) {
    if !options.fit_optimal {
        return (c0, c1);
    }

    debug_assert_eq!(block.len(), block_vec.len());

    // If the endpoints are very close, then quantization artifacts have a
    // significant effect on which indexes are chosen per pixel. This is a
    // problem here, because we don't know the final quantized endpoints yet
    // and have to guess them. The closer the endpoints are, the more the
    // error in our guess matters. Since this method optimizes based on indexes,
    // if the indexes aren't good, it will return garbage.
    // This minimum distance is somewhat arbitrary. It seems to work, so I
    // haven't looked into other value.
    let min_dist = (1_u32 << I) as f32 / (1_u32 << B) as f32;
    if c0.distance_squared(c1) < min_dist * min_dist {
        return (c0, c1);
    }

    let (e0, e1): (Rgb<B>, Rgb<B>) = Quantization::wide(c0, c1);
    let index_list: IndexList<I> = closest_rgb(e0.promote(), e1.promote(), block).0;
    let weights = index_list.to_weights();

    let optimal: (Vec3A, Vec3A) =
        bcn_util::least_squares_weights(block_vec, &weights[..block.len()]);

    (optimal.0, optimal.1)
}
fn fit_optimal_endpoints_alpha<const B: u8, const I: u8>(
    mut min: f32,
    mut max: f32,
    block: &[Alpha<8>; 16],
    options: Bc7Options,
) -> (f32, f32) {
    if !options.fit_optimal {
        return (min, max);
    }

    let (e0, e1) = (Alpha::<B>::round(min), Alpha::<B>::round(max));
    let index_list: IndexList<I> = closest_alpha(e0.promote(), e1.promote(), block).0;
    let weights = index_list.to_weights();

    let weights: [Vec4; 4] = [
        Vec4::new(weights[0], weights[1], weights[2], weights[3]),
        Vec4::new(weights[4], weights[5], weights[6], weights[7]),
        Vec4::new(weights[8], weights[9], weights[10], weights[11]),
        Vec4::new(weights[12], weights[13], weights[14], weights[15]),
    ];

    let alpha = block.map(|a| a.to_vec());
    let alpha: [Vec4; 4] = [
        Vec4::new(alpha[0], alpha[1], alpha[2], alpha[3]),
        Vec4::new(alpha[4], alpha[5], alpha[6], alpha[7]),
        Vec4::new(alpha[8], alpha[9], alpha[10], alpha[11]),
        Vec4::new(alpha[12], alpha[13], alpha[14], alpha[15]),
    ];

    (min, max) = bcn_util::least_squares_weights_f32_vec4(&alpha, &weights);
    min = min.clamp(0.0, 1.0);
    max = max.clamp(0.0, 1.0);

    (min, max)
}

fn refine_along_line3(
    min: Vec3A,
    max: Vec3A,
    palette_size: u8,
    options: Bc7Options,
    compute_error: impl Fn((Vec3A, Vec3A)) -> u32,
) -> (Vec3A, Vec3A) {
    let dist = min.distance(max);
    if dist < 0.0001 {
        return (min, max);
    }
    let mid = (min + max) * 0.5;
    let min_dir = (min - mid) * 2.0;
    let max_dir = (max - mid) * 2.0;
    let get_min_max = |min_t: f32, max_t: f32| {
        let min = mid + min_dir * min_t;
        let max = mid + max_dir * max_t;
        (min, max)
    };

    let options = bcn_util::RefinementOptions {
        step_initial: match palette_size {
            4 => 0.2,
            8 => 0.1,
            _ => unreachable!(),
        },
        step_decay: 0.5,
        step_min: 0.0025 / dist,
        max_iter: options.max_refinement_iters as u32,
    };

    let (min_t, max_t) = bcn_util::refine_endpoints(0.5, 0.5, options, move |(min_t, max_t)| {
        compute_error(get_min_max(min_t, max_t))
    });

    get_min_max(min_t, max_t)
}
fn refine_along_line4(
    min: Vec4,
    max: Vec4,
    palette_size: u8,
    options: Bc7Options,
    compute_error: impl Fn((Vec4, Vec4)) -> u32,
) -> (Vec4, Vec4) {
    let dist = min.distance(max);
    if dist < 0.0001 {
        return (min, max);
    }
    let mid = (min + max) * 0.5;
    let min_dir = (min - mid) * 2.0;
    let max_dir = (max - mid) * 2.0;
    let get_min_max = |min_t: f32, max_t: f32| {
        let min = mid + min_dir * min_t;
        let max = mid + max_dir * max_t;
        (min, max)
    };

    let options = bcn_util::RefinementOptions {
        step_initial: match palette_size {
            4 => 0.2,
            8 => 0.1,
            16 => 0.1,
            _ => unreachable!(),
        },
        step_decay: 0.5,
        step_min: 0.0025 / dist,
        max_iter: options.max_refinement_iters as u32,
    };

    let (min_t, max_t) = bcn_util::refine_endpoints(0.5, 0.5, options, move |(min_t, max_t)| {
        compute_error(get_min_max(min_t, max_t))
    });

    get_min_max(min_t, max_t)
}

fn closest_rgb<const I: u8>(e0: Rgb<8>, e1: Rgb<8>, pixels: &[Rgb<8>]) -> (IndexList<I>, u32) {
    debug_assert!(I == 2 || I == 3);

    fn closest<const N: usize, const I: u8>(
        palette: [Rgb<8>; N],
        pixels: &[Rgb<8>],
    ) -> (IndexList<I>, u32) {
        debug_assert_eq!(N, 1 << I);

        let mut indexes = IndexList::<I>::new();
        let mut error = 0_u32;
        for (i, &p) in pixels.iter().enumerate() {
            let mut best_index = 0;
            let mut best_dist = u32::MAX;
            for (j, &c) in palette.iter().enumerate() {
                let dist = p.dist_sq(c);
                if dist < best_dist {
                    best_dist = dist;
                    best_index = j;
                }
            }
            indexes.set(i, best_index as u8);
            error += best_dist;
        }
        (indexes, error)
    }

    match I {
        2 => {
            let palette: [Rgb<8>; 4] = [
                e0,
                interpolate_rgb::<2>(e0, e1, 1),
                interpolate_rgb::<2>(e0, e1, 2),
                e1,
            ];
            closest(palette, pixels)
        }
        3 => {
            let palette: [Rgb<8>; 8] = [
                e0,
                interpolate_rgb::<3>(e0, e1, 1),
                interpolate_rgb::<3>(e0, e1, 2),
                interpolate_rgb::<3>(e0, e1, 3),
                interpolate_rgb::<3>(e0, e1, 4),
                interpolate_rgb::<3>(e0, e1, 5),
                interpolate_rgb::<3>(e0, e1, 6),
                e1,
            ];
            closest(palette, pixels)
        }
        _ => unreachable!(),
    }
}
fn closest_error_rgb<const I: u8>(e0: Rgb<8>, e1: Rgb<8>, pixels: &[Rgb<8>]) -> u32 {
    debug_assert!(I == 2 || I == 3);

    fn error<const N: usize>(palette: [Rgb<8>; N], pixels: &[Rgb<8>]) -> u32 {
        let mut error = 0_u32;
        for &p in pixels {
            let mut best_dist = u32::MAX;
            for &c in &palette {
                best_dist = best_dist.min(c.dist_sq(p));
            }
            error += best_dist;
        }
        error
    }

    match I {
        2 => {
            let palette: [Rgb<8>; 4] = [
                e0,
                interpolate_rgb::<2>(e0, e1, 1),
                interpolate_rgb::<2>(e0, e1, 2),
                e1,
            ];
            error(palette, pixels)
        }
        3 => {
            let palette: [Rgb<8>; 8] = [
                e0,
                interpolate_rgb::<3>(e0, e1, 1),
                interpolate_rgb::<3>(e0, e1, 2),
                interpolate_rgb::<3>(e0, e1, 3),
                interpolate_rgb::<3>(e0, e1, 4),
                interpolate_rgb::<3>(e0, e1, 5),
                interpolate_rgb::<3>(e0, e1, 6),
                e1,
            ];
            error(palette, pixels)
        }
        _ => unreachable!(),
    }
}
fn closest_rgba<const I: u8>(e0: Rgba<8>, e1: Rgba<8>, pixels: &[Rgba<8>]) -> (IndexList<I>, u32) {
    debug_assert!(I == 2 || I == 4);

    fn closest<const N: usize, const I: u8>(
        palette: [Rgba<8>; N],
        pixels: &[Rgba<8>],
    ) -> (IndexList<I>, u32) {
        debug_assert_eq!(N, 1 << I);

        let mut indexes = IndexList::<I>::new();
        let mut error = 0_u32;
        for (i, &p) in pixels.iter().enumerate() {
            let mut best_index = 0;
            let mut best_dist = u32::MAX;
            for (j, &c) in palette.iter().enumerate() {
                let dist = c.dist_sq(p);
                if dist < best_dist {
                    best_dist = dist;
                    best_index = j;
                }
            }
            indexes.set(i, best_index as u8);
            error += best_dist;
        }
        (indexes, error)
    }

    match I {
        2 => {
            let palette: [Rgba<8>; 4] = [
                e0,
                interpolate_rgba::<2>(e0, e1, 1),
                interpolate_rgba::<2>(e0, e1, 2),
                e1,
            ];
            closest(palette, pixels)
        }
        4 => {
            let palette: [Rgba<8>; 16] = [
                e0,
                interpolate_rgba::<4>(e0, e1, 1),
                interpolate_rgba::<4>(e0, e1, 2),
                interpolate_rgba::<4>(e0, e1, 3),
                interpolate_rgba::<4>(e0, e1, 4),
                interpolate_rgba::<4>(e0, e1, 5),
                interpolate_rgba::<4>(e0, e1, 6),
                interpolate_rgba::<4>(e0, e1, 7),
                interpolate_rgba::<4>(e0, e1, 8),
                interpolate_rgba::<4>(e0, e1, 9),
                interpolate_rgba::<4>(e0, e1, 10),
                interpolate_rgba::<4>(e0, e1, 11),
                interpolate_rgba::<4>(e0, e1, 12),
                interpolate_rgba::<4>(e0, e1, 13),
                interpolate_rgba::<4>(e0, e1, 14),
                e1,
            ];
            closest(palette, pixels)
        }
        _ => unreachable!(),
    }
}
fn closest_error_rgba<const I: u8>(e0: Rgba<8>, e1: Rgba<8>, pixels: &[Rgba<8>]) -> u32 {
    debug_assert!(I == 2 || I == 4);

    fn error<const N: usize>(palette: [Rgba<8>; N], pixels: &[Rgba<8>]) -> u32 {
        let mut error = 0_u32;
        for &p in pixels {
            let mut best_dist = u32::MAX;
            for &c in &palette {
                best_dist = best_dist.min(c.dist_sq(p));
            }
            error += best_dist;
        }
        error
    }

    fn error_16(e0: Rgba<8>, e1: Rgba<8>, pixels: &[Rgba<8>]) -> u32 {
        if e0 == e1 {
            return pixels.iter().map(move |p| p.dist_sq(e0)).sum();
        }

        // This approximates the true error calculation by projecting
        // the pixel onto the line defined by e0 and e1. The line parameter t
        // t is quantized using rounding to nearest.
        // This is only an approximation because round to nearest is not the
        // correct quantization. (See the weights table and observe that weights
        // are not equally spaced.) This error could be corrected by simply
        // testing the 2 nearest quantized t values. However, the addition perf
        // cost is not worth the small accuracy gain.
        let (e0_8, e1_8) = (e0, e1);
        let e0 = IVec4::new(e0.r as i32, e0.g as i32, e0.b as i32, e0.a as i32);
        let e1 = IVec4::new(e1.r as i32, e1.g as i32, e1.b as i32, e1.a as i32);
        let d = e1 - e0;
        let d_len_sq = d.length_squared();
        let d_len_sq_half = d_len_sq >> 1;
        debug_assert!(d_len_sq > 0);

        let mut error = 0_u32;
        for &p in pixels {
            let q = IVec4::new(p.r as i32, p.g as i32, p.b as i32, p.a as i32);
            let v = q - e0;
            let t = (v.dot(d) * 15 + d_len_sq_half) / d_len_sq;
            let t = t.clamp(0, 15) as u8;
            let c = interpolate_rgba::<4>(e0_8, e1_8, t);
            error += c.dist_sq(p);
        }
        error
    }

    match I {
        2 => {
            let palette: [Rgba<8>; 4] = [
                e0,
                interpolate_rgba::<2>(e0, e1, 1),
                interpolate_rgba::<2>(e0, e1, 2),
                e1,
            ];
            error(palette, pixels)
        }
        4 => error_16(e0, e1, pixels),
        _ => unreachable!(),
    }
}
fn closest_alpha<const I: u8>(
    e0: Alpha<8>,
    e1: Alpha<8>,
    pixels: &[Alpha<8>; 16],
) -> (IndexList<I>, u32) {
    debug_assert!(I == 2 || I == 3);

    fn closest<const N: usize, const I: u8>(
        palette: [Alpha<8>; N],
        pixels: &[Alpha<8>; 16],
    ) -> (IndexList<I>, u32) {
        debug_assert_eq!(N, 1 << I);

        let mut indexes = IndexList::<I>::new();
        let mut error = 0_u32;
        for i in 0..16 {
            let p = pixels[i];
            let mut best_index = 0;
            let mut best_dist = u32::MAX;
            for (j, &c) in palette.iter().enumerate() {
                let d = p.a as i32 - c.a as i32;
                let dist = (d * d) as u32;
                if dist < best_dist {
                    best_dist = dist;
                    best_index = j;
                }
            }
            indexes.set(i, best_index as u8);
            error += best_dist;
        }
        (indexes, error)
    }

    match I {
        2 => {
            let palette: [Alpha<8>; 4] = [
                e0,
                interpolate_alpha::<2>(e0, e1, 1),
                interpolate_alpha::<2>(e0, e1, 2),
                e1,
            ];
            closest(palette, pixels)
        }
        3 => {
            let palette: [Alpha<8>; 8] = [
                e0,
                interpolate_alpha::<3>(e0, e1, 1),
                interpolate_alpha::<3>(e0, e1, 2),
                interpolate_alpha::<3>(e0, e1, 3),
                interpolate_alpha::<3>(e0, e1, 4),
                interpolate_alpha::<3>(e0, e1, 5),
                interpolate_alpha::<3>(e0, e1, 6),
                e1,
            ];
            closest(palette, pixels)
        }
        _ => unreachable!(),
    }
}
fn closest_error_alpha<const I: u8>(min: Alpha<8>, max: Alpha<8>, pixels: &[Alpha<8>; 16]) -> u32 {
    debug_assert!(I == 2 || I == 3);

    debug_assert!(min.a <= max.a);
    if min.a >= max.a {
        debug_assert!(min.a == max.a);
        let a = min.a;
        pixels
            .iter()
            .map(|p| {
                let d = p.a.abs_diff(a) as u32;
                d * d
            })
            .sum()
    } else {
        debug_assert!(min.a < max.a);

        fn error<const N: usize>(pixels: &[Alpha<8>; 16], interpolated: [u8; N]) -> u32 {
            pixels
                .iter()
                .map(|p| {
                    let mut best = 255_u8;
                    for &a in &interpolated {
                        let d = p.a.abs_diff(a);
                        best = best.min(d);
                    }
                    let d = best as u32;
                    d * d
                })
                .sum()
        }

        match I {
            2 => {
                let interpolated = [
                    min.a,
                    interpolate::<2>(min.a, max.a, 1),
                    interpolate::<2>(min.a, max.a, 2),
                    max.a,
                ];
                error(pixels, interpolated)
            }
            3 => {
                let interpolated = [
                    min.a,
                    interpolate::<3>(min.a, max.a, 1),
                    interpolate::<3>(min.a, max.a, 2),
                    interpolate::<3>(min.a, max.a, 3),
                    interpolate::<3>(min.a, max.a, 4),
                    interpolate::<3>(min.a, max.a, 5),
                    interpolate::<3>(min.a, max.a, 6),
                    max.a,
                ];
                error(pixels, interpolated)
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Compressed {
    block: [u8; 16],
    error: u32,
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
            error: u32::MAX,
        }
    }

    fn mode0(
        error: u32,
        partition: u8,
        mut rgb: [Rgb<4>; 6],
        mut p: [bool; 6],
        indexes: IndexList<3>,
    ) -> Self {
        debug_assert!(partition < 16);
        let subset = PARTITION_SET_3[partition as usize];
        let (indexes, [swap0, swap1, swap2]) = indexes.compress_p3(subset);

        if swap0 {
            rgb.swap(0, 1);
            p.swap(0, 1);
        }
        if swap1 {
            rgb.swap(2, 3);
            p.swap(2, 3);
        }
        if swap2 {
            rgb.swap(4, 5);
            p.swap(4, 5);
        }

        let mut stream = BitStream::new();
        stream.write_mode(0);
        stream.write_u64(partition as u64, 4);
        stream.write_endpoints_rgb(&rgb);
        stream.write_endpoints_p(&p);
        stream.write_indexes(indexes);
        let block = stream.finish();

        Self { block, error }
    }
    fn mode1(
        error: u32,
        partition: u8,
        mut rgb: [Rgb<6>; 4],
        p: [bool; 2],
        indexes: IndexList<3>,
    ) -> Self {
        debug_assert!(partition < 64);
        let subset = PARTITION_SET_2[partition as usize];
        let (indexes, [swap0, swap1]) = indexes.compress_p2(subset);

        if swap0 {
            rgb.swap(0, 1);
        }
        if swap1 {
            rgb.swap(2, 3);
        }

        let mut stream = BitStream::new();
        stream.write_mode(1);
        stream.write_u64(partition as u64, 6);
        stream.write_endpoints_rgb(&rgb);
        stream.write_endpoints_p(&p);
        stream.write_indexes(indexes);
        let block = stream.finish();

        Self { block, error }
    }
    fn mode2(error: u32, partition: u8, mut rgb: [Rgb<5>; 6], indexes: IndexList<2>) -> Self {
        debug_assert!(partition < 64);
        let subset = PARTITION_SET_3[partition as usize];
        let (indexes, [swap0, swap1, swap2]) = indexes.compress_p3(subset);

        if swap0 {
            rgb.swap(0, 1);
        }
        if swap1 {
            rgb.swap(2, 3);
        }
        if swap2 {
            rgb.swap(4, 5);
        }

        let mut stream = BitStream::new();
        stream.write_mode(2);
        stream.write_u64(partition as u64, 6);
        stream.write_endpoints_rgb(&rgb);
        stream.write_indexes(indexes);
        let block = stream.finish();

        Self { block, error }
    }
    fn mode3(
        error: u32,
        partition: u8,
        mut rgb: [Rgb<7>; 4],
        mut p: [bool; 4],
        indexes: IndexList<2>,
    ) -> Self {
        debug_assert!(partition < 64);
        let subset = PARTITION_SET_2[partition as usize];
        let (indexes, [swap0, swap1]) = indexes.compress_p2(subset);

        if swap0 {
            rgb.swap(0, 1);
            p.swap(0, 1);
        }
        if swap1 {
            rgb.swap(2, 3);
            p.swap(2, 3);
        }

        let mut stream = BitStream::new();
        stream.write_mode(3);
        stream.write_u64(partition as u64, 6);
        stream.write_endpoints_rgb(&rgb);
        stream.write_endpoints_p(&p);
        stream.write_indexes(indexes);
        let block = stream.finish();

        Self { block, error }
    }
    fn mode4(
        error: u32,
        rotation: Rotation,
        index_mode: IndexMode,
        mut color: [Rgb<5>; 2],
        color_indexes: IndexList<2>,
        mut alpha: [Alpha<6>; 2],
        alpha_indexes: IndexList<3>,
    ) -> Self {
        let (color_indexes, color_swap) = color_indexes.compress_p1();
        let (alpha_indexes, alpha_swap) = alpha_indexes.compress_p1();

        let swapped = index_mode == IndexMode::C3A2;

        if color_swap && !swapped || alpha_swap && swapped {
            color.swap(0, 1);
        }
        if alpha_swap && !swapped || color_swap && swapped {
            alpha.swap(0, 1);
        }

        let mut stream = BitStream::new();
        stream.write_mode(4);
        stream.write_rotation(rotation);
        stream.write_u64(index_mode as u64, 1);
        stream.write_endpoints_rgb(&color);
        stream.write_endpoints_alpha(&alpha);
        stream.write_indexes(color_indexes);
        stream.write_indexes(alpha_indexes);
        let block = stream.finish();

        Self { block, error }
    }
    fn mode5(
        error: u32,
        rotation: Rotation,
        mut color: [Rgb<7>; 2],
        color_indexes: IndexList<2>,
        mut alpha: [Alpha<8>; 2],
        alpha_indexes: IndexList<2>,
    ) -> Self {
        let (color_indexes, color_swap) = color_indexes.compress_p1();
        let (alpha_indexes, alpha_swap) = alpha_indexes.compress_p1();

        if color_swap {
            color.swap(0, 1);
        }
        if alpha_swap {
            alpha.swap(0, 1);
        }

        let mut stream = BitStream::new();
        stream.write_mode(5);
        stream.write_rotation(rotation);
        stream.write_endpoints_rgb(&color);
        stream.write_endpoints_alpha(&alpha);
        stream.write_indexes(color_indexes);
        stream.write_indexes(alpha_indexes);
        let block = stream.finish();

        Self { block, error }
    }
    fn mode6(error: u32, mut rgba: [Rgba<7>; 2], mut p: [bool; 2], indexes: IndexList<4>) -> Self {
        let (indexes, swap) = indexes.compress_p1();

        if swap {
            rgba.swap(0, 1);
            p.swap(0, 1);
        }

        let mut stream = BitStream::new();
        stream.write_mode(6);
        stream.write_endpoints_rgba(&rgba);
        stream.write_endpoints_p(&p);
        stream.write_indexes(indexes);
        let block = stream.finish();

        Self { block, error }
    }
    fn mode7(
        error: u32,
        partition: u8,
        mut rgba: [Rgba<5>; 4],
        mut p: [bool; 4],
        indexes: IndexList<2>,
    ) -> Self {
        debug_assert!(partition < 64);
        let subset = PARTITION_SET_2[partition as usize];
        let (indexes, [swap0, swap1]) = indexes.compress_p2(subset);

        if swap0 {
            rgba.swap(0, 1);
            p.swap(0, 1);
        }
        if swap1 {
            rgba.swap(2, 3);
            p.swap(2, 3);
        }

        let mut stream = BitStream::new();
        stream.write_mode(7);
        stream.write_u64(partition as u64, 6);
        stream.write_endpoints_rgba(&rgba);
        stream.write_endpoints_p(&p);
        stream.write_indexes(indexes);
        let block = stream.finish();

        Self { block, error }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum IndexMode {
    /// This is the default for mode 4. The color index is 2-bit and the alpha index is 3-bit.
    C2A3 = 0,
    /// Swapped index mode. The color index is 3-bit and the alpha index is 2-bit.
    C3A2 = 1,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Rotation {
    None = 0,
    AR = 1,
    AG = 2,
    AB = 3,
}
impl Rotation {
    fn swap_r(p: Rgba<8>) -> Rgba<8> {
        Rgba::new(p.a, p.g, p.b, p.r)
    }
    fn swap_g(p: Rgba<8>) -> Rgba<8> {
        Rgba::new(p.r, p.a, p.b, p.g)
    }
    fn swap_b(p: Rgba<8>) -> Rgba<8> {
        Rgba::new(p.r, p.g, p.a, p.b)
    }

    fn apply<const N: usize>(self, colors: [Rgba<8>; N]) -> [Rgba<8>; N] {
        match self {
            Rotation::None => colors,
            Rotation::AR => colors.map(Self::swap_r),
            Rotation::AG => colors.map(Self::swap_g),
            Rotation::AB => colors.map(Self::swap_b),
        }
    }
    fn apply_stats(self, stats: BlockStats) -> BlockStats {
        let [min, max] = self.apply([stats.min, stats.max]);

        BlockStats {
            min,
            max,
            is_grayscale: self == Self::None && stats.is_grayscale,
        }
    }

    fn channel(self) -> usize {
        match self {
            Rotation::None => 3, // alpha
            Rotation::AR => 0,   // red
            Rotation::AG => 1,   // green
            Rotation::AB => 2,   // blue
        }
    }
}

/// A list of 16 indexes each using I bits.
///
/// I must be 2, 3, or 4.
struct IndexList<const I: u8> {
    indexes: u64,
}
impl<const I: u8> IndexList<I> {
    const MAX_INDEX: u8 = (1 << I) - 1;
    const INDEXES_MASK: u64 = if I == 4 {
        u64::MAX
    } else {
        (1 << (I * 16)) - 1
    };

    const fn new() -> Self {
        debug_assert!(I == 2 || I == 3 || I == 4);
        Self { indexes: 0 }
    }
    const CONSTANT_MULTIPLE: u64 = {
        let mut m = 0;
        let mut i = 0;
        while i < 16 {
            m |= 1 << (i * I);
            i += 1;
        }
        m
    };
    fn constant(value: u8) -> Self {
        debug_assert!(I == 2 || I == 3 || I == 4);
        debug_assert!(value <= Self::MAX_INDEX);

        Self {
            indexes: value as u64 * Self::CONSTANT_MULTIPLE,
        }
    }

    fn get(&self, index: usize) -> u8 {
        debug_assert!(index < 16);
        ((self.indexes >> (index * I as usize)) & Self::MAX_INDEX as u64) as u8
    }
    fn set(&mut self, index: usize, value: u8) {
        debug_assert!(index < 16);
        debug_assert!(value <= Self::MAX_INDEX);
        debug_assert!(self.get(index) == 0, "Cannot set an index twice.");
        self.indexes |= (value as u64) << (index * I as usize);
    }

    fn to_weights(&self) -> [f32; 16] {
        let mut weights: [f32; 16] = [0.0; 16];
        let f = 1.0 / ((1_u8 << I) - 1) as f32;
        for i in 0..16 {
            weights[i] = self.get(i) as f32 * f;
        }
        weights
    }

    /// Compresses the index list and returns whether the endpoints need to be swapped.
    fn compress_p1(mut self) -> (CompressedIndexList, bool) {
        let swap = self.ensure_msb_zero(0, Self::INDEXES_MASK);
        let compressed = Self::compress_single_index(self.indexes, 0);
        (
            CompressedIndexList {
                compressed_indexes: compressed,
                bits: 16 * I - 1,
            },
            swap,
        )
    }
    fn compress_p2(mut self, subset: Subset2Map) -> (CompressedIndexList, [bool; 2]) {
        debug_assert!(I == 2 || I == 3);

        let p2_fixup = subset.fixup_index_2;
        debug_assert!(0 < p2_fixup && p2_fixup < 16);

        let mask_s1 = match I {
            2 => bits_repeat2_u16(subset.subset_indexes) as u64,
            3 => bits_repeat3_u16(subset.subset_indexes),
            _ => unreachable!(),
        };
        debug_assert!(mask_s1 & Self::INDEXES_MASK == mask_s1);

        let swap1 = self.ensure_msb_zero(0, mask_s1 ^ Self::INDEXES_MASK);
        let swap2 = self.ensure_msb_zero(p2_fixup, mask_s1);

        let mut compressed = self.indexes;
        compressed = Self::compress_single_index(compressed, p2_fixup);
        compressed = Self::compress_single_index(compressed, 0);
        (
            CompressedIndexList {
                compressed_indexes: compressed,
                bits: 16 * I - 2,
            },
            [swap1, swap2],
        )
    }
    fn compress_p3(mut self, subset: Subset3Map) -> (CompressedIndexList, [bool; 3]) {
        debug_assert!(I == 2 || I == 3);

        fn get_mask<const I: u8>(subset: Subset3Map, value: u8) -> u64 {
            let mut mask = 0_u64;
            let element_mask = (1 << I) - 1;
            for i in 0..16 {
                if subset.get_subset_index(i) == value {
                    mask |= element_mask << (i * I);
                }
            }
            mask
        }

        let p2_fixup = subset.fixup_index_2;
        let p3_fixup = subset.fixup_index_3;
        debug_assert!(0 < p2_fixup && p2_fixup < p3_fixup && p3_fixup < 16);

        // Fix up indexes are stored ordered by ascending numeric value, not
        // subset index. But here we need them ordered by subset index.
        let (mut s1_index, mut s2_index) = (p2_fixup, p3_fixup);
        if subset.get_subset_index(s1_index) == 2 {
            std::mem::swap(&mut s1_index, &mut s2_index);
        }
        let swap1 = self.ensure_msb_zero(0, get_mask::<I>(subset, 0));
        let swap2 = self.ensure_msb_zero(s1_index, get_mask::<I>(subset, 1));
        let swap3 = self.ensure_msb_zero(s2_index, get_mask::<I>(subset, 2));

        let mut compressed = self.indexes;
        compressed = Self::compress_single_index(compressed, p3_fixup);
        compressed = Self::compress_single_index(compressed, p2_fixup);
        compressed = Self::compress_single_index(compressed, 0);
        (
            CompressedIndexList {
                compressed_indexes: compressed,
                bits: 16 * I - 3,
            },
            [swap1, swap2, swap3],
        )
    }

    /// This method makes the MSB of the index-th value 0 by possibly flipping
    /// all bits. Returns `true` if the bits were flipped, `false` otherwise.
    fn ensure_msb_zero(&mut self, index: u8, subset_mask: u64) -> bool {
        debug_assert!(I == 2 || I == 3 || I == 4);
        debug_assert!(index < 16);

        // the MSB of the index-th value has to be 0
        let msb_mask = 1 << (I - 1 + index * I);
        let swap = (self.indexes & msb_mask) != 0;

        // if the MSB is 1, flip all bits
        if swap {
            self.indexes ^= subset_mask;
        }

        swap
    }
    fn compress_single_index(indexes: u64, index: u8) -> u64 {
        debug_assert!(I == 2 || I == 3 || I == 4);
        debug_assert!(index < 16);

        // the MSB of the index-th value has to be 0
        let msb_mask = 1 << (I - 1 + index * I);
        // now the MSB of the given index value is 0, so we can drop it
        debug_assert!((indexes & msb_mask) == 0);
        let before_mask = msb_mask - 1;
        let after_mask = !before_mask << 1;

        (indexes & before_mask) | ((indexes & after_mask) >> 1)
    }

    fn merge2(subset: Subset2Map, s0: Self, s1: Self) -> Self {
        let mut indexes = Self::new();
        let mut s0_index = 0;
        let mut s1_index = 0;
        for i in 0..16 {
            let value;
            if subset.get_subset_index(i) == 0 {
                value = s0.get(s0_index);
                s0_index += 1;
            } else {
                value = s1.get(s1_index);
                s1_index += 1;
            }
            indexes.set(i as usize, value);
        }
        indexes
    }
    fn merge3(subset: Subset3Map, s0: Self, s1: Self, s2: Self) -> Self {
        let mut indexes = Self::new();
        let mut s0_index = 0;
        let mut s1_index = 0;
        let mut s2_index = 0;
        for i in 0..16 {
            let value;
            match subset.get_subset_index(i) {
                0 => {
                    value = s0.get(s0_index);
                    s0_index += 1;
                }
                1 => {
                    value = s1.get(s1_index);
                    s1_index += 1;
                }
                2 => {
                    value = s2.get(s2_index);
                    s2_index += 1;
                }
                _ => unreachable!(),
            }
            indexes.set(i as usize, value);
        }
        indexes
    }
}

struct CompressedIndexList {
    compressed_indexes: u64,
    bits: u8,
}

#[repr(C, align(4))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Rgba<const B: u8> {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}
const _: () = {
    assert!(std::mem::size_of::<Rgba<8>>() == std::mem::size_of::<u32>());
    assert!(std::mem::align_of::<Rgba<8>>() == std::mem::align_of::<u32>());
};
impl<const B: u8> Rgba<B> {
    const MAX: u8 = ((1_u32 << B) - 1) as u8;

    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        debug_assert!(0 < B && B <= 8);
        debug_assert!(r <= Self::MAX);
        debug_assert!(g <= Self::MAX);
        debug_assert!(b <= Self::MAX);
        debug_assert!(a <= Self::MAX);
        Self { r, g, b, a }
    }

    pub fn to_u32(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, self.a])
    }

    pub fn color(self) -> Rgb<B> {
        Rgb::new(self.r, self.g, self.b)
    }
    pub fn alpha(self) -> Alpha<B> {
        Alpha::new(self.a)
    }

    pub fn promote(self) -> Rgba<8> {
        if B == 8 {
            Rgba::new(self.r, self.g, self.b, self.a)
        } else {
            Rgba::new(
                promote(self.r, B),
                promote(self.g, B),
                promote(self.b, B),
                promote(self.a, B),
            )
        }
    }
    pub fn p_promote(self, p: bool) -> Rgba<8> {
        debug_assert!(B < 8);
        let r = (self.r << 1) | (p as u8);
        let g = (self.g << 1) | (p as u8);
        let b = (self.b << 1) | (p as u8);
        let a = (self.a << 1) | (p as u8);
        if B == 7 {
            Rgba::new(r, g, b, a)
        } else {
            Rgba::new(
                promote(r, B + 1),
                promote(g, B + 1),
                promote(b, B + 1),
                promote(a, B + 1),
            )
        }
    }

    #[inline(always)]
    pub fn dist_sq(self, other: Rgba<8>) -> u32 {
        let dr = self.r as i32 - other.r as i32;
        let dg = self.g as i32 - other.g as i32;
        let db = self.b as i32 - other.b as i32;
        let da = self.a as i32 - other.a as i32;
        (dr * dr + dg * dg + db * db + da * da) as u32
    }
}
impl<const B: u8> PartialEq for Rgba<B> {
    fn eq(&self, other: &Self) -> bool {
        self.to_u32() == other.to_u32()
    }
}
impl<const B: u8> Eq for Rgba<B> {}
impl<const B: u8> Quantized for Rgba<B> {
    type V = Vec4;

    fn round(v: Vec4) -> Self {
        Rgba::new(
            channel_round::<B>(v.x),
            channel_round::<B>(v.y),
            channel_round::<B>(v.z),
            channel_round::<B>(v.w),
        )
    }
    fn floor(v: Vec4) -> Self {
        Rgba::new(
            channel_floor::<B>(v.x),
            channel_floor::<B>(v.y),
            channel_floor::<B>(v.z),
            channel_floor::<B>(v.w),
        )
    }
    fn ceil(v: Vec4) -> Self {
        Rgba::new(
            channel_ceil::<B>(v.x),
            channel_ceil::<B>(v.y),
            channel_ceil::<B>(v.z),
            channel_ceil::<B>(v.w),
        )
    }

    #[inline(always)]
    fn to_vec(self) -> Vec4 {
        let p = self.promote();
        Vec4::new(p.r as f32, p.g as f32, p.b as f32, p.a as f32) * (1.0 / 255.0)
    }
}
impl<const B: u8> WithChannels for Rgba<B> {
    type E = u8;
    const CHANNELS: usize = 4;

    fn get(&self, channel: usize) -> Self::E {
        match channel {
            0 => self.r,
            1 => self.g,
            2 => self.b,
            3 => self.a,
            _ => unreachable!(),
        }
    }
    fn set(&mut self, channel: usize, value: Self::E) {
        match channel {
            0 => self.r = value,
            1 => self.g = value,
            2 => self.b = value,
            3 => self.a = value,
            _ => unreachable!(),
        }
    }
}

#[repr(C, align(4))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Rgb<const B: u8> {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}
const _: () = {
    assert!(std::mem::size_of::<Rgb<8>>() == std::mem::size_of::<u32>());
    assert!(std::mem::align_of::<Rgb<8>>() == std::mem::align_of::<u32>());
};
impl<const B: u8> Rgb<B> {
    const MAX: u8 = ((1_u32 << B) - 1) as u8;

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        debug_assert!(0 < B && B <= 8);
        debug_assert!(r <= Self::MAX);
        debug_assert!(g <= Self::MAX);
        debug_assert!(b <= Self::MAX);
        Self { r, g, b }
    }

    pub fn to_u32(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, 0])
    }

    pub fn promote(self) -> Rgb<8> {
        if B == 8 {
            Rgb::new(self.r, self.g, self.b)
        } else {
            Rgb::new(promote(self.r, B), promote(self.g, B), promote(self.b, B))
        }
    }
    pub fn p_promote(self, p: bool) -> Rgb<8> {
        debug_assert!(B < 8);
        let r = (self.r << 1) | (p as u8);
        let g = (self.g << 1) | (p as u8);
        let b = (self.b << 1) | (p as u8);
        if B == 7 {
            Rgb::new(r, g, b)
        } else {
            Rgb::new(promote(r, B + 1), promote(g, B + 1), promote(b, B + 1))
        }
    }

    #[inline(always)]
    pub fn dist_sq(self, other: Rgb<8>) -> u32 {
        let dr = self.r as i32 - other.r as i32;
        let dg = self.g as i32 - other.g as i32;
        let db = self.b as i32 - other.b as i32;
        (dr * dr + dg * dg + db * db) as u32
    }
}
impl<const B: u8> PartialEq for Rgb<B> {
    fn eq(&self, other: &Self) -> bool {
        self.to_u32() == other.to_u32()
    }
}
impl<const B: u8> Eq for Rgb<B> {}
impl<const B: u8> Quantized for Rgb<B> {
    type V = Vec3A;

    fn round(v: Vec3A) -> Self {
        Rgb::new(
            channel_round::<B>(v.x),
            channel_round::<B>(v.y),
            channel_round::<B>(v.z),
        )
    }
    fn floor(v: Vec3A) -> Self {
        Rgb::new(
            channel_floor::<B>(v.x),
            channel_floor::<B>(v.y),
            channel_floor::<B>(v.z),
        )
    }
    fn ceil(v: Vec3A) -> Self {
        Rgb::new(
            channel_ceil::<B>(v.x),
            channel_ceil::<B>(v.y),
            channel_ceil::<B>(v.z),
        )
    }

    #[inline(always)]
    fn to_vec(self) -> Vec3A {
        let p = self.promote();
        Vec3A::new(p.r as f32, p.g as f32, p.b as f32) * (1.0 / 255.0)
    }
}
impl<const B: u8> WithChannels for Rgb<B> {
    type E = u8;
    const CHANNELS: usize = 3;

    fn get(&self, channel: usize) -> Self::E {
        match channel {
            0 => self.r,
            1 => self.g,
            2 => self.b,
            _ => unreachable!(),
        }
    }
    fn set(&mut self, channel: usize, value: Self::E) {
        match channel {
            0 => self.r = value,
            1 => self.g = value,
            2 => self.b = value,
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Alpha<const B: u8> {
    pub a: u8,
}
impl<const B: u8> Alpha<B> {
    const MAX: u8 = ((1_u32 << B) - 1) as u8;

    pub const fn new(a: u8) -> Self {
        debug_assert!(0 < B && B <= 8);
        debug_assert!(a <= Self::MAX);
        Self { a }
    }

    pub fn promote(self) -> Alpha<8> {
        if B == 8 {
            Alpha::new(self.a)
        } else {
            Alpha::new(promote(self.a, B))
        }
    }
}
impl<const B: u8> Quantized for Alpha<B> {
    type V = f32;

    #[inline(always)]
    fn round(v: f32) -> Self {
        Alpha::new(channel_round::<B>(v))
    }
    #[inline(always)]
    fn floor(v: f32) -> Self {
        Alpha::new(channel_floor::<B>(v))
    }
    #[inline(always)]
    fn ceil(v: f32) -> Self {
        Alpha::new(channel_ceil::<B>(v))
    }

    #[inline(always)]
    fn to_vec(self) -> f32 {
        self.promote().a as f32 * (1.0 / 255.0)
    }
}
impl<const B: u8> WithChannels for Alpha<B> {
    type E = u8;
    const CHANNELS: usize = 1;

    #[inline(always)]
    fn get(&self, channel: usize) -> Self::E {
        debug_assert!(channel == 0);
        self.a
    }
    #[inline(always)]
    fn set(&mut self, channel: usize, value: Self::E) {
        debug_assert!(channel == 0);
        self.a = value;
    }
}

fn channel_round<const B: u8>(v: f32) -> u8 {
    match B {
        8 => (v * 255.0 + 0.5) as u8,
        4 => (v * 15.0 + 0.5).min(15.0) as u8,
        5..=7 => {
            let max = ((1_u32 << B) - 1) as u8;
            let v = v.clamp(0.0, 1.0);
            let mut nearest = (v * max as f32 + 0.5) as u8;
            let nearest_err = (channel_to_vec::<B>(nearest) - v).abs();
            if nearest > 0 && (channel_to_vec::<B>(nearest - 1) - v).abs() < nearest_err {
                nearest -= 1;
            } else if nearest < max && (channel_to_vec::<B>(nearest + 1) - v).abs() < nearest_err {
                nearest += 1;
            }
            nearest
        }
        _ => unreachable!(),
    }
}
fn channel_floor<const B: u8>(v: f32) -> u8 {
    match B {
        8 => (v * 255.0) as u8,
        4 => (v * 15.0).min(15.0) as u8,
        5..=7 => {
            let max = ((1_u32 << B) - 1) as u8;
            let v = v.clamp(0.0, 1.0);
            let mut floor = (v * max as f32) as u8;
            if floor > 0 && channel_to_vec::<B>(floor) > v {
                floor -= 1;
            } else if floor < max && channel_to_vec::<B>(floor + 1) < v {
                floor += 1;
            }
            floor
        }
        _ => unreachable!(),
    }
}
fn channel_ceil<const B: u8>(v: f32) -> u8 {
    const CEIL: f32 = 0.9999;
    match B {
        8 => (v * 255.0 + CEIL) as u8,
        4 => (v * 15.0 + CEIL).min(15.0) as u8,
        5..=7 => {
            let max = ((1_u32 << B) - 1) as u8;
            let v = v.clamp(0.0, 1.0);
            let mut ceil = (v * max as f32 + CEIL).min(max as f32) as u8;
            if ceil < max && channel_to_vec::<B>(ceil) < v {
                ceil += 1;
            } else if ceil > 0 && channel_to_vec::<B>(ceil - 1) > v {
                ceil -= 1;
            }
            ceil
        }
        _ => unreachable!(),
    }
}
#[inline(always)]
fn channel_to_vec<const B: u8>(c: u8) -> f32 {
    let u = if B == 8 { c } else { promote(c, B) };
    u as f32 * (1.0 / 255.0)
}
#[inline]
fn promote(mut number: u8, number_bits: u8) -> u8 {
    debug_assert!((4..8).contains(&number_bits));
    number <<= 8 - number_bits;
    number |= number >> number_bits;
    number
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

fn interpolate<const W: usize>(e0: u8, e1: u8, index: u8) -> u8 {
    let weight = match W {
        2 => WEIGHTS_2[index as usize],
        3 => WEIGHTS_3[index as usize],
        4 => WEIGHTS_4[index as usize],
        _ => unreachable!(),
    };
    let w0 = 256 - weight;
    let w1 = weight;
    ((w0 * e0 as u16 + w1 * e1 as u16 + 128) >> 8) as u8
}
fn interpolate_rgba<const W: usize>(e0: Rgba<8>, e1: Rgba<8>, index: u8) -> Rgba<8> {
    let weight = match W {
        2 => WEIGHTS_2[index as usize],
        3 => WEIGHTS_3[index as usize],
        4 => WEIGHTS_4[index as usize],
        _ => unreachable!(),
    };
    let w0 = 256 - weight;
    let w1 = weight;

    Rgba::new(
        ((w0 * e0.r as u16 + w1 * e1.r as u16 + 128) >> 8) as u8,
        ((w0 * e0.g as u16 + w1 * e1.g as u16 + 128) >> 8) as u8,
        ((w0 * e0.b as u16 + w1 * e1.b as u16 + 128) >> 8) as u8,
        ((w0 * e0.a as u16 + w1 * e1.a as u16 + 128) >> 8) as u8,
    )
}
fn interpolate_rgb<const W: usize>(e0: Rgb<8>, e1: Rgb<8>, index: u8) -> Rgb<8> {
    let weight = match W {
        2 => WEIGHTS_2[index as usize],
        3 => WEIGHTS_3[index as usize],
        4 => WEIGHTS_4[index as usize],
        _ => unreachable!(),
    };
    let w0 = 256 - weight;
    let w1 = weight;

    Rgb::new(
        ((w0 * e0.r as u16 + w1 * e1.r as u16 + 128) >> 8) as u8,
        ((w0 * e0.g as u16 + w1 * e1.g as u16 + 128) >> 8) as u8,
        ((w0 * e0.b as u16 + w1 * e1.b as u16 + 128) >> 8) as u8,
    )
}
fn interpolate_alpha<const W: usize>(e0: Alpha<8>, e1: Alpha<8>, index: u8) -> Alpha<8> {
    Alpha::new(interpolate::<W>(e0.a, e1.a, index))
}

#[derive(Clone, Copy)]
struct BlockStats {
    min: Rgba<8>,
    max: Rgba<8>,
    /// Whether all pixels are opaque and grayscale.
    is_grayscale: bool,
}
impl BlockStats {
    fn new(block: &[Rgba<8>; 16]) -> Self {
        let mut min = Rgba::new(255, 255, 255, 255);
        let mut max = Rgba::new(0, 0, 0, 0);
        for &pixel in block {
            min.r = min.r.min(pixel.r);
            min.g = min.g.min(pixel.g);
            min.b = min.b.min(pixel.b);
            min.a = min.a.min(pixel.a);

            max.r = max.r.max(pixel.r);
            max.g = max.g.max(pixel.g);
            max.b = max.b.max(pixel.b);
            max.a = max.a.max(pixel.a);
        }

        let is_grayscale = min.a == 255 && block.iter().all(|&p| p.r == p.g && p.r == p.b);

        Self {
            min,
            max,
            is_grayscale,
        }
    }

    fn single_color(&self) -> Option<Rgba<8>> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }
    fn single_alpha(&self) -> Option<u8> {
        if self.min.a == self.max.a {
            Some(self.max.a)
        } else {
            None
        }
    }

    /// Returns whether Alpha is 255 everywhere.
    fn opaque(&self) -> bool {
        self.min.a == 255
    }
}

/// This type has the responsibility of selecting which partitions to try
/// for modes with 2 or 3 subsets.
///
/// The current strategy involves scoring all possible partitions based on
/// some error metric per subset and then picking the best N partitions to try.
///
/// The error metric works as follows:
/// - For grayscale blocks: Squared error from the mean. This is fast to compute
///   and works okay.
/// - For color blocks(with and without) alpha: Compute the best-fitting line
///   and use the squared error from that line. Line fitting isn't cheap, but
///   closely approximates the error from actual endpoint quantization and
///   index selection.
struct PartitionSelect<'a> {
    has_alpha: bool,
    is_grayscale: bool,
    block: &'a [Rgba<8>; 16],
    modes: Bc7Modes,
    max_partitions: u8,
    fast_mode_0: bool,

    cache_partitions_2: Option<[u8; 64]>,
    cache_partitions_3_64: Option<[u8; 64]>,
    cache_partitions_3_16: Option<[u8; 16]>,
}
impl<'a> PartitionSelect<'a> {
    fn new(
        block: &'a [Rgba<8>; 16],
        stats: BlockStats,
        modes: Bc7Modes,
        options: Bc7Options,
    ) -> Self {
        Self {
            has_alpha: !stats.opaque(),
            is_grayscale: stats.is_grayscale,
            block,
            modes,
            max_partitions: options.max_partitions,
            fast_mode_0: options.fast_mode_0_partitions && !modes.contains(Bc7Modes::MODE2),

            cache_partitions_2: None,
            cache_partitions_3_64: None,
            cache_partitions_3_16: None,
        }
    }

    fn get_partitions_2(&mut self) -> &[u8; 64] {
        let has_alpha = self.has_alpha;
        let is_grayscale = self.is_grayscale;
        let block = self.block;
        let max_partitions = self.max_partitions;

        self.cache_partitions_2.get_or_insert_with(|| {
            let mut scored: [(u8, f32); 64] = std::array::from_fn(move |i| {
                let partition = i as u8;
                let subset = PARTITION_SET_2[partition as usize];

                if partition >= max_partitions {
                    return (partition, f32::INFINITY);
                }

                let split_index = subset.count_zeros() as usize;
                let error = if is_grayscale {
                    // GRAYSCALE
                    let mut reordered = block.map(|p| p.r as f32 * (1.0 / 255.0));
                    subset.sort_block(&mut reordered);
                    squared_error_grayscale(&reordered[..split_index])
                        + squared_error_grayscale(&reordered[split_index..])
                } else {
                    // RGB(A)
                    if has_alpha {
                        let mut reordered = block.map(|p| p.to_vec());
                        subset.sort_block(&mut reordered);
                        squared_error_line_4(&reordered[..split_index])
                            + squared_error_line_4(&reordered[split_index..])
                    } else {
                        let mut reordered = block.map(|p| p.color().to_vec());
                        subset.sort_block(&mut reordered);
                        squared_error_line_3(&reordered[..split_index])
                            + squared_error_line_3(&reordered[split_index..])
                    }
                };

                (partition, error)
            });

            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

            scored.map(|(p, _)| p)
        })
    }
    fn pick_best_partition_2(
        &mut self,
        options: Bc7Options,
        f: impl Fn(u8) -> Compressed,
    ) -> Compressed {
        let max_candidates = options.max_partition_candidates as usize;

        let mut best = Compressed::invalid();
        for &partition in self.get_partitions_2().iter().take(max_candidates) {
            best = best.better(f(partition));
        }
        best
    }

    fn fill_partitions_3(&mut self) {
        if self.cache_partitions_3_16.is_some() || self.cache_partitions_3_64.is_some() {
            return;
        }

        let has_mode_0 = self.modes.contains(Bc7Modes::MODE0);
        let has_mode_2 = self.modes.contains(Bc7Modes::MODE2);
        debug_assert!(has_mode_0 || has_mode_2);
        let max_partitions = self.max_partitions.min(if has_mode_2 { 64 } else { 16 });
        let enabled_partitions: u64 = if self.fast_mode_0 {
            debug_assert!(!has_mode_2);
            // This subset of 3 subset partitions works well in practice and
            // makes better use of caching.
            0b11_1111_1111_0000
        } else if max_partitions >= 64 {
            u64::MAX // all 64 partitions
        } else {
            (1_u64 << max_partitions) - 1
        };

        let is_grayscale = self.is_grayscale;
        let block = self.block;
        let mut error_cache = LineErrorCache3::new();

        let mut scored: [(u8, f32); 64] = std::array::from_fn(move |i| {
            let partition = i as u8;

            if enabled_partitions & (1 << partition) == 0 {
                return (partition, f32::INFINITY);
            }

            let error = if is_grayscale {
                error_cache.get_error(
                    partition,
                    block.map(|p| p.r as f32 * (1.0 / 255.0)),
                    squared_error_grayscale,
                )
            } else {
                error_cache.get_error(
                    partition,
                    block.map(|p| p.color().to_vec()),
                    squared_error_line_3,
                )
            };

            (partition, error)
        });

        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let partitions_64: [u8; 64] = scored.map(|(p, _)| p);
        let mut iter_16 = partitions_64.iter().copied().filter(|p| *p < 16);
        let partitions_16: [u8; 16] = std::array::from_fn(move |_| iter_16.next().unwrap());

        self.cache_partitions_3_64 = Some(partitions_64);
        self.cache_partitions_3_16 = Some(partitions_16);
    }
    fn pick_best_partition_3(
        &mut self,
        is_mode_0: bool,
        options: Bc7Options,
        f: impl Fn(u8) -> Compressed,
    ) -> Compressed {
        let max_candidates = options.max_partition_candidates as usize;

        let current_mode = if is_mode_0 {
            Bc7Modes::MODE0
        } else {
            Bc7Modes::MODE2
        };
        debug_assert!(self.modes.contains(current_mode));
        self.fill_partitions_3();

        let partitions: &[u8] = if is_mode_0 {
            self.cache_partitions_3_16
                .as_ref()
                .expect("partitions_3_16 not filled")
        } else {
            self.cache_partitions_3_64
                .as_ref()
                .expect("partitions_3_64 not filled")
        };

        let mut best = Compressed::invalid();
        for &partition in partitions.iter().take(max_candidates) {
            best = best.better(f(partition));
        }
        best
    }
}
/// The 3 subset partitions have some duplicate subsets between them. This can
/// be exploited to cache line error computations.
struct LineErrorCache3 {
    cache: [f32; PARTITION_SET_3_DUPLICATE_COUNT],
}
impl LineErrorCache3 {
    fn new() -> Self {
        Self {
            cache: [f32::NAN; PARTITION_SET_3_DUPLICATE_COUNT],
        }
    }

    fn get_error<T: Copy>(
        &mut self,
        partition: u8,
        mut block: [T; 16],
        compute_error: impl Fn(&[T]) -> f32,
    ) -> f32 {
        let subset = PARTITION_SET_3[partition as usize];
        subset.sort_block(&mut block);
        let split_index = subset.count_zeros() as usize;
        let split_index2 = split_index + subset.count_ones() as usize;

        let s0 = &block[..split_index];
        let s1 = &block[split_index..split_index2];
        let s2 = &block[split_index2..];
        let [c0, c1, c2] = PARTITION_SET_3_DUPLICATES[partition as usize];

        let mut error = 0.0;
        for (s, cache_index) in [(s0, c0), (s1, c1), (s2, c2)] {
            if cache_index == 255 {
                // no duplicate line
                error += compute_error(s);
            } else {
                let cache = &mut self.cache[cache_index as usize];
                if cache.is_nan() {
                    *cache = compute_error(s);
                }
                error += *cache;
            }
        }
        error
    }
}

fn squared_error_line_3(colors: &[Vec3A]) -> f32 {
    ColorLine3::new(colors).sum_dist_sq(colors)
}
fn squared_error_line_4(colors: &[Vec4]) -> f32 {
    ColorLine4::new(colors).sum_dist_sq(colors)
}
fn squared_error_grayscale(colors: &[f32]) -> f32 {
    let mean = colors.iter().copied().sum::<f32>() / colors.len() as f32;

    let mut error = 0.0;
    for &c in colors {
        let d = c - mean;
        error += d * d;
    }
    error
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

    fn write_mode(&mut self, mode: u8) {
        debug_assert!(mode < 8);
        self.write_u64(1 << mode, mode + 1);
    }
    fn write_rotation(&mut self, rotation: Rotation) {
        self.write_u64(rotation as u64, 2);
    }
    fn write_indexes(&mut self, indexes: CompressedIndexList) {
        self.write_u64(indexes.compressed_indexes, indexes.bits);
    }
    fn write_endpoints_rgba<const B: u8>(&mut self, endpoints: &[Rgba<B>]) {
        debug_assert!(endpoints.len() % 2 == 0);
        endpoints.iter().for_each(|e| self.write_u64(e.r as u64, B));
        endpoints.iter().for_each(|e| self.write_u64(e.g as u64, B));
        endpoints.iter().for_each(|e| self.write_u64(e.b as u64, B));
        endpoints.iter().for_each(|e| self.write_u64(e.a as u64, B));
    }
    fn write_endpoints_rgb<const B: u8>(&mut self, endpoints: &[Rgb<B>]) {
        debug_assert!(endpoints.len() % 2 == 0);
        endpoints.iter().for_each(|e| self.write_u64(e.r as u64, B));
        endpoints.iter().for_each(|e| self.write_u64(e.g as u64, B));
        endpoints.iter().for_each(|e| self.write_u64(e.b as u64, B));
    }
    fn write_endpoints_alpha<const B: u8>(&mut self, endpoints: &[Alpha<B>]) {
        debug_assert!(endpoints.len() % 2 == 0);
        endpoints.iter().for_each(|e| self.write_u64(e.a as u64, B));
    }
    fn write_endpoints_p(&mut self, p: &[bool]) {
        debug_assert!(p.len() % 2 == 0);
        p.iter().for_each(|p| self.write_u64(*p as u64, 1));
    }

    fn finish(self) -> [u8; 16] {
        debug_assert!(self.bits == 128);
        self.data.to_le_bytes()
    }
}

/// Turns a 16-bit value into a 32-bit value where each bit is duplicated.
///
/// E.g. `0b1010` becomes `0b11001100`.
fn bits_repeat2_u16(x: u16) -> u32 {
    let mut x = x as u32;
    x = (x | (x << 8)) & 0x00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333;
    x = (x | (x << 1)) & 0x5555_5555;
    x | (x << 1)
}
/// Turns a 16-bit value into a 64-bit value where each bit is repeated 3 times.
///
/// E.g. `0b1010` becomes `0b111000111000`.
fn bits_repeat3_u16(x: u16) -> u64 {
    // TODO: find a faster algorithm
    let mut result: u64 = 0;
    for i in 0..16 {
        let bit = ((x >> i) & 1) * 0b111;
        result |= (bit as u64) << (i * 3);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_bits() {
        /// Reference implementation that repeats each bit `repeat` times.
        fn repeat(x: u64, repeat: u8) -> u64 {
            assert!(repeat > 0);
            let mut result = 0;
            for i in 0..64 / repeat {
                let bit = (x >> i) & 1;
                for j in 0..repeat {
                    result |= bit << (i * repeat + j);
                }
            }
            result
        }

        assert_eq!(repeat(0b0010_1100, 1), 0b0010_1100);
        assert_eq!(repeat(0b0010_1100, 2), 0b00001100_11110000);
        assert_eq!(repeat(0b0010_1100, 3), 0b000000111000_111111000000);
        assert_eq!(repeat(0b0010_1100, 4), 0b0000000011110000_1111111100000000);

        for x in 0..=u16::MAX {
            assert_eq!(repeat(x as u64, 1), x as u64, "Failed for {x:#b}");

            assert_eq!(
                repeat(x as u64, 2),
                bits_repeat2_u16(x) as u64,
                "Failed for {x:#b}"
            );

            assert_eq!(
                repeat(x as u64, 3),
                bits_repeat3_u16(x),
                "Failed for {x:#b}"
            );
        }
    }

    const EPS: f32 = 0.000001;
    #[test]
    fn round() {
        for v in (0..=1000).map(|i| i as f32 / 1000.) {
            macro_rules! test_b {
                ($b:literal) => {
                    let r = Alpha::<$b>::round(v);
                    let p1 = Alpha::<$b>::new(r.a.saturating_add(1).min(Alpha::<$b>::MAX));
                    let m1 = Alpha::<$b>::new(r.a.saturating_sub(1));

                    let diff_r = (r.to_vec() - v).abs();
                    let diff_p1 = (p1.to_vec() - v).abs();
                    let diff_m1 = (m1.to_vec() - v).abs();
                    assert!(
                        diff_r <= diff_p1 + EPS && diff_r <= diff_m1 + EPS,
                        "Failed for v={v} and B={}!\nr-1={} ({} Δ={})\nr  ={} ({} Δ={})\nr+1={} ({} Δ={})",
                        $b,
                        m1.a,
                        m1.to_vec(),
                        diff_m1,
                        r.a,
                        r.to_vec(),
                        diff_r,
                        p1.a,
                        p1.to_vec(),
                        diff_p1
                    );
                };
            }

            test_b!(8);
            test_b!(7);
            test_b!(6);
            test_b!(5);
            test_b!(4);
        }
    }
    #[test]
    fn floor() {
        for v in (0..=1000).map(|i| i as f32 / 1000.) {
            macro_rules! test_b {
                ($b:literal) => {
                    let f = Alpha::<$b>::floor(v);
                    let p1 = Alpha::<$b>::new(f.a.saturating_add(1).min(Alpha::<$b>::MAX));
                    let m1 = Alpha::<$b>::new(f.a.saturating_sub(1));

                    let v_f = f.to_vec();
                    let v_p1 = p1.to_vec();
                    let v_m1 = m1.to_vec();
                    assert!(
                        v_f <= v + EPS && v_p1 >= v - EPS && v_m1 <= v + EPS,
                        "Failed for v={v} and B={}!\nf-1={} ({})\nf  ={} ({})\nf+1={} ({})",
                        $b,
                        m1.a,
                        m1.to_vec(),
                        f.a,
                        f.to_vec(),
                        p1.a,
                        p1.to_vec()
                    );
                };
            }

            test_b!(8);
            test_b!(7);
            test_b!(6);
            test_b!(5);
            test_b!(4);
        }
    }
    #[test]
    fn ceil() {
        for v in (0..=1000).map(|i| i as f32 / 1000.) {
            macro_rules! test_b {
                ($b:literal) => {
                    let c = Alpha::<$b>::ceil(v);
                    let p1 = Alpha::<$b>::new(c.a.saturating_add(1).min(Alpha::<$b>::MAX));
                    let m1 = Alpha::<$b>::new(c.a.saturating_sub(1));

                    let v_c = c.to_vec();
                    let v_p1 = p1.to_vec();
                    let v_m1 = m1.to_vec();
                    assert!(
                        v_c >= v - EPS && v_p1 >= v - EPS && v_m1 <= v + EPS,
                        "Failed for v={v} and B={}!\nc-1={} ({})\nc  ={} ({})\nc+1={} ({})",
                        $b,
                        m1.a,
                        m1.to_vec(),
                        c.a,
                        c.to_vec(),
                        p1.a,
                        p1.to_vec()
                    );
                };
            }

            test_b!(8);
            test_b!(7);
            test_b!(6);
            test_b!(5);
            test_b!(4);
        }
    }
}
