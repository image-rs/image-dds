use crate::{
    decode::bcn_util::{BitStream, Indexes, Subset2Map, PARTITION_SET_2},
    util::unlikely_branch,
};

// Spec:
// https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#19.5.13%20BC6H%20/%20DXGI_FORMAT_BC6H

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BC6HFormat {
    UnsignedF16,
    SignedF16,
}

pub(crate) fn decode_bc6_block(block: [u8; 16], format: BC6HFormat) -> [[u16; 3]; 16] {
    let mut stream = BitStream::new(block);

    // initialize the output to all 0s, aka black
    let mut output = [[0_u16; 3]; 16];

    let mode = extract_mode(&mut stream);

    match mode {
        Mode::One(mode) => {
            let endpoints = extract_compressed_endpoints_one(mode, &mut stream);
            let index = Indexes::new_p1(4, &mut stream);

            let endpoints = decompress_endpoints_one(mode, format, endpoints);
            let precision = mode.a0_bit_count();

            let palette =
                generate_palette_unquantized_one(endpoints.a, endpoints.b, precision, format);

            for pixel_index in 0..16 {
                let index = index.get_index(pixel_index);

                let color = palette[index as usize];

                output[pixel_index as usize] = color.rgb();
            }
        }
        Mode::Two(mode) => {
            let endpoints = extract_compressed_endpoints_two(mode, &mut stream);
            let partition = extract_partition(&mut stream);
            let index = Indexes::new_p2(3, &mut stream, partition.fixup_index_2);

            let endpoints = decompress_endpoints_two(mode, format, endpoints);
            let precision = mode.a0_bit_count();

            let palette =
                endpoints.map(|e| generate_palette_unquantized_two(e.a, e.b, precision, format));

            for pixel_index in 0..16 {
                let index: u8 = index.get_index(pixel_index);
                let subset_index = partition.get_subset_index(pixel_index);

                let color = palette[subset_index as usize][index as usize];

                output[pixel_index as usize] = color.rgb();
            }
        }
        Mode::Invalid => {
            unlikely_branch();
            // Modes 10011, 10111, 11011, and 11111 are reserved and should not be
            // used by the encoder. If hardware is given these modes, the resulting
            // decompressed block must contain zeroes in all channels (except the
            // alpha channel).
            return output;
        }
    }

    output
}

#[derive(Clone, Copy)]
enum Mode {
    One(ModeOne),
    Two(ModeTwo),
    Invalid,
}

#[derive(Clone, Copy)]
enum ModeTwo {
    M10_555 = 0b00,
    M7_666 = 0b01,

    M11_544 = 0b00010,
    M11_454 = 0b00110,
    M11_445 = 0b01010,
    M9_555 = 0b01110,
    M8_655 = 0b10010,
    M8_565 = 0b10110,
    M8_556 = 0b11010,
    M6_666 = 0b11110,
}
impl ModeTwo {
    fn a0_bit_count(&self) -> u8 {
        match self {
            ModeTwo::M10_555 => 10,
            ModeTwo::M7_666 => 7,

            ModeTwo::M11_544 | ModeTwo::M11_454 | ModeTwo::M11_445 => 11,
            ModeTwo::M9_555 => 9,
            ModeTwo::M8_655 | ModeTwo::M8_565 | ModeTwo::M8_556 => 8,
            ModeTwo::M6_666 => 6,
        }
    }
    fn delta_bit_count(&self) -> (u8, u8, u8) {
        match self {
            ModeTwo::M10_555 => (5, 5, 5),
            ModeTwo::M7_666 => (6, 6, 6),

            ModeTwo::M11_544 => (5, 4, 4),
            ModeTwo::M11_454 => (4, 5, 4),
            ModeTwo::M11_445 => (4, 4, 5),
            ModeTwo::M9_555 => (5, 5, 5),
            ModeTwo::M8_655 => (6, 5, 5),
            ModeTwo::M8_565 => (5, 6, 5),
            ModeTwo::M8_556 => (5, 5, 6),
            ModeTwo::M6_666 => (6, 6, 6),
        }
    }

    fn transformed(&self) -> bool {
        !matches!(self, ModeTwo::M6_666)
    }
}
#[derive(Clone, Copy)]
enum ModeOne {
    M10_10 = 0b00,
    M11_9 = 0b01,
    M12_8 = 0b10,
    M16_4 = 0b11,
}
impl ModeOne {
    fn a0_bit_count(&self) -> u8 {
        match self {
            ModeOne::M10_10 => 10,
            ModeOne::M11_9 => 11,
            ModeOne::M12_8 => 12,
            ModeOne::M16_4 => 16,
        }
    }
    fn b0_bit_count(&self) -> u8 {
        20 - self.a0_bit_count()
    }

    fn transformed(&self) -> bool {
        !matches!(self, ModeOne::M10_10)
    }
}

fn extract_mode(stream: &mut BitStream) -> Mode {
    let low2 = stream.consume_bits(2);
    match low2 {
        0b00 => Mode::Two(ModeTwo::M10_555),
        0b01 => Mode::Two(ModeTwo::M7_666),
        0b10 => {
            let high3 = stream.consume_bits(3);
            let bits = (high3 << 2) | 0b10;
            Mode::Two(match bits {
                0b00010 => ModeTwo::M11_544,
                0b00110 => ModeTwo::M11_454,
                0b01010 => ModeTwo::M11_445,
                0b01110 => ModeTwo::M9_555,
                0b10010 => ModeTwo::M8_655,
                0b10110 => ModeTwo::M8_565,
                0b11010 => ModeTwo::M8_556,
                0b11110 => ModeTwo::M6_666,
                _ => unreachable!(),
            })
        }
        0b11 => {
            let high3 = stream.consume_bits(3);
            if high3 & 0b100 != 0 {
                Mode::Invalid
            } else {
                let high2 = high3 & 0b11;
                Mode::One(match high2 {
                    0b00 => ModeOne::M10_10,
                    0b01 => ModeOne::M11_9,
                    0b10 => ModeOne::M12_8,
                    0b11 => ModeOne::M16_4,
                    _ => unreachable!(),
                })
            }
        }
        _ => unreachable!(),
    }
}

fn extract_partition(stream: &mut BitStream) -> Subset2Map {
    let partition = stream.consume_bits(5);
    debug_assert!(partition < 32);
    PARTITION_SET_2[partition as usize]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct EndPointPair {
    a: IntColor<i32>,
    b: IntColor<i32>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct IntColor<T> {
    r: T,
    g: T,
    b: T,
    _pad: T,
}
impl<T> IntColor<T> {
    fn new(r: T, g: T, b: T) -> Self
    where
        T: Default,
    {
        Self {
            r,
            g,
            b,
            _pad: T::default(),
        }
    }
    fn rgb(&self) -> [T; 3]
    where
        T: Copy,
    {
        [self.r, self.g, self.b]
    }
}
impl IntColor<i32> {
    fn sign_extend_all(&mut self, bit_count: u8) {
        self.r = sign_extend(self.r, bit_count);
        self.g = sign_extend(self.g, bit_count);
        self.b = sign_extend(self.b, bit_count);
    }
    fn sign_extend(&mut self, r_bit_count: u8, g_bit_count: u8, b_bit_count: u8) {
        self.r = sign_extend(self.r, r_bit_count);
        self.g = sign_extend(self.g, g_bit_count);
        self.b = sign_extend(self.b, b_bit_count);
    }

    fn bit_and(&self, mask: i32) -> Self {
        Self {
            r: self.r & mask,
            g: self.g & mask,
            b: self.b & mask,
            _pad: 0, // TODO: maybe mask this too for better vectorization?
        }
    }
}
impl core::ops::Add for IntColor<i32> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        IntColor {
            r: self.r.wrapping_add(rhs.r),
            g: self.g.wrapping_add(rhs.g),
            b: self.b.wrapping_add(rhs.b),
            _pad: self._pad.wrapping_add(rhs._pad),
        }
    }
}

fn extract_compressed_endpoints_one(mode: ModeOne, stream: &mut BitStream) -> EndPointPair {
    let mut pair = EndPointPair::default();

    // rw -> endpt[0].A[0]    gw -> endpt[0].A[1]    bw -> endpt[0].A[2]
    // rx -> endpt[0].B[0]    gx -> endpt[0].B[1]    bx -> endpt[0].B[2]

    // the first 30 bits are the same for all ONE modes
    pair.a.r = stream.consume_bits_32(10);
    pair.a.g = stream.consume_bits_32(10);
    pair.a.b = stream.consume_bits_32(10);

    // how many bits are used per channel for A0.
    let a0_bit_count = mode.a0_bit_count();
    // how many bits are used per channel for B0.
    let b0_bit_count = 20 - a0_bit_count;

    let a0_extension = a0_bit_count - 10;

    pair.b.r = stream.consume_bits_32(b0_bit_count);
    pair.a.r |= (stream.consume_bits_rev(a0_extension) as i32) << 10;
    pair.b.g = stream.consume_bits_32(b0_bit_count);
    pair.a.g |= (stream.consume_bits_rev(a0_extension) as i32) << 10;
    pair.b.b = stream.consume_bits_32(b0_bit_count);
    pair.a.b |= (stream.consume_bits_rev(a0_extension) as i32) << 10;

    pair
}
fn decompress_endpoints_one(
    mode: ModeOne,
    format: BC6HFormat,
    mut pair: EndPointPair,
) -> EndPointPair {
    let a_bit_count = mode.a0_bit_count();
    let b_bit_count = mode.b0_bit_count();

    // sign extend the endpoints
    if format == BC6HFormat::SignedF16 {
        pair.a.sign_extend_all(a_bit_count);
    }
    if mode.transformed() || format == BC6HFormat::SignedF16 {
        pair.b.sign_extend_all(b_bit_count);
    }

    if mode.transformed() {
        // transform B0 -> B0+A0

        let mask = (1 << a_bit_count) - 1;
        pair.b = (pair.a + pair.b).bit_and(mask);
        if format == BC6HFormat::SignedF16 {
            pair.b.sign_extend_all(a_bit_count);
        }
    }

    pair
}

fn sign_extend(x: i32, bit_count: u8) -> i32 {
    debug_assert!(bit_count > 0);
    debug_assert!(bit_count < 32);

    // check that all bits outsize bit_count are zero
    debug_assert_eq!(x & !((1 << bit_count) - 1), 0);

    let shift = 32 - bit_count;
    (x << shift) >> shift
}

fn extract_compressed_endpoints_two(mode: ModeTwo, stream: &mut BitStream) -> [EndPointPair; 2] {
    let [mut w, mut x, mut y, mut z]: [IntColor<i32>; 4] = Default::default();

    // I hate BC6.

    /// This is a macro to translate the expressions from 19.5.13.5 table 2
    /// into useable code.
    ///
    /// It has 2 modes:
    /// 1. Single bit mode: `gy[4]   == consume!(g, y, 4)`
    /// 2. Range mode:      `rw[9:0] == consume!(r, w, 9..0)`
    ///
    /// Note that these are NOT normal Rust ranges, I'm just misappropriating
    /// their syntax.
    macro_rules! consume {
        ($i1:ident, $i2:ident, $index:literal) => {
            $i2.$i1 |= stream.consume_bits_32(1) << $index;
        };
        ($i1:ident, $i2:ident, $high:literal .. 0) => {
            $i2.$i1 |= stream.consume_bits_32($high + 1);
        };
    }

    match mode {
        ModeTwo::M10_555 => {
            consume!(g, y, 4);
            consume!(b, y, 4);
            consume!(b, z, 4);
            consume!(r, w, 9..0);
            consume!(g, w, 9..0);
            consume!(b, w, 9..0);
            consume!(r, x, 4..0);
            consume!(g, z, 4);
            consume!(g, y, 3..0);
            consume!(g, x, 4..0);
            consume!(b, z, 0);
            consume!(g, z, 3..0);
            consume!(b, x, 4..0);
            consume!(b, z, 1);
            consume!(b, y, 3..0);
            consume!(r, y, 4..0);
            consume!(b, z, 2);
            consume!(r, z, 4..0);
            consume!(b, z, 3);
        }
        ModeTwo::M7_666 => {
            consume!(g, y, 5);
            consume!(g, z, 4);
            consume!(g, z, 5);
            consume!(r, w, 6..0);
            consume!(b, z, 0);
            consume!(b, z, 1);
            consume!(b, y, 4);
            consume!(g, w, 6..0);
            consume!(b, y, 5);
            consume!(b, z, 2);
            consume!(g, y, 4);
            consume!(b, w, 6..0);
            consume!(b, z, 3);
            consume!(b, z, 5);
            consume!(b, z, 4);
            consume!(r, x, 5..0);
            consume!(g, y, 3..0);
            consume!(g, x, 5..0);
            consume!(g, z, 3..0);
            consume!(b, x, 5..0);
            consume!(b, y, 3..0);
            consume!(r, y, 5..0);
            consume!(r, z, 5..0);
        }
        ModeTwo::M11_544 => {
            consume!(r, w, 9..0);
            consume!(g, w, 9..0);
            consume!(b, w, 9..0);
            consume!(r, x, 4..0);
            consume!(r, w, 10);
            consume!(g, y, 3..0);
            consume!(g, x, 3..0);
            consume!(g, w, 10);
            consume!(b, z, 0);
            consume!(g, z, 3..0);
            consume!(b, x, 3..0);
            consume!(b, w, 10);
            consume!(b, z, 1);
            consume!(b, y, 3..0);
            consume!(r, y, 4..0);
            consume!(b, z, 2);
            consume!(r, z, 4..0);
            consume!(b, z, 3);
        }
        ModeTwo::M11_454 => {
            consume!(r, w, 9..0);
            consume!(g, w, 9..0);
            consume!(b, w, 9..0);
            consume!(r, x, 3..0);
            consume!(r, w, 10);
            consume!(g, z, 4);
            consume!(g, y, 3..0);
            consume!(g, x, 4..0);
            consume!(g, w, 10);
            consume!(g, z, 3..0);
            consume!(b, x, 3..0);
            consume!(b, w, 10);
            consume!(b, z, 1);
            consume!(b, y, 3..0);
            consume!(r, y, 3..0);
            consume!(b, z, 0);
            consume!(b, z, 2);
            consume!(r, z, 3..0);
            consume!(g, y, 4);
            consume!(b, z, 3);
        }
        ModeTwo::M11_445 => {
            consume!(r, w, 9..0);
            consume!(g, w, 9..0);
            consume!(b, w, 9..0);
            consume!(r, x, 3..0);
            consume!(r, w, 10);
            consume!(b, y, 4);
            consume!(g, y, 3..0);
            consume!(g, x, 3..0);
            consume!(g, w, 10);
            consume!(b, z, 0);
            consume!(g, z, 3..0);
            consume!(b, x, 4..0);
            consume!(b, w, 10);
            consume!(b, y, 3..0);
            consume!(r, y, 3..0);
            consume!(b, z, 1);
            consume!(b, z, 2);
            consume!(r, z, 3..0);
            consume!(b, z, 4);
            consume!(b, z, 3);
        }
        ModeTwo::M9_555 => {
            consume!(r, w, 8..0);
            consume!(b, y, 4);
            consume!(g, w, 8..0);
            consume!(g, y, 4);
            consume!(b, w, 8..0);
            consume!(b, z, 4);
            consume!(r, x, 4..0);
            consume!(g, z, 4);
            consume!(g, y, 3..0);
            consume!(g, x, 4..0);
            consume!(b, z, 0);
            consume!(g, z, 3..0);
            consume!(b, x, 4..0);
            consume!(b, z, 1);
            consume!(b, y, 3..0);
            consume!(r, y, 4..0);
            consume!(b, z, 2);
            consume!(r, z, 4..0);
            consume!(b, z, 3);
        }
        ModeTwo::M8_655 => {
            consume!(r, w, 7..0);
            consume!(g, z, 4);
            consume!(b, y, 4);
            consume!(g, w, 7..0);
            consume!(b, z, 2);
            consume!(g, y, 4);
            consume!(b, w, 7..0);
            consume!(b, z, 3);
            consume!(b, z, 4);
            consume!(r, x, 5..0);
            consume!(g, y, 3..0);
            consume!(g, x, 4..0);
            consume!(b, z, 0);
            consume!(g, z, 3..0);
            consume!(b, x, 4..0);
            consume!(b, z, 1);
            consume!(b, y, 3..0);
            consume!(r, y, 5..0);
            consume!(r, z, 5..0);
        }
        ModeTwo::M8_565 => {
            consume!(r, w, 7..0);
            consume!(b, z, 0);
            consume!(b, y, 4);
            consume!(g, w, 7..0);
            consume!(g, y, 5);
            consume!(g, y, 4);
            consume!(b, w, 7..0);
            consume!(g, z, 5);
            consume!(b, z, 4);
            consume!(r, x, 4..0);
            consume!(g, z, 4);
            consume!(g, y, 3..0);
            consume!(g, x, 5..0);
            consume!(g, z, 3..0);
            consume!(b, x, 4..0);
            consume!(b, z, 1);
            consume!(b, y, 3..0);
            consume!(r, y, 4..0);
            consume!(b, z, 2);
            consume!(r, z, 4..0);
            consume!(b, z, 3);
        }
        ModeTwo::M8_556 => {
            consume!(r, w, 7..0);
            consume!(b, z, 1);
            consume!(b, y, 4);
            consume!(g, w, 7..0);
            consume!(b, y, 5);
            consume!(g, y, 4);
            consume!(b, w, 7..0);
            consume!(b, z, 5);
            consume!(b, z, 4);
            consume!(r, x, 4..0);
            consume!(g, z, 4);
            consume!(g, y, 3..0);
            consume!(g, x, 4..0);
            consume!(b, z, 0);
            consume!(g, z, 3..0);
            consume!(b, x, 5..0);
            consume!(b, y, 3..0);
            consume!(r, y, 4..0);
            consume!(b, z, 2);
            consume!(r, z, 4..0);
            consume!(b, z, 3);
        }
        ModeTwo::M6_666 => {
            consume!(r, w, 5..0);
            consume!(g, z, 4);
            consume!(b, z, 0);
            consume!(b, z, 1);
            consume!(b, y, 4);
            consume!(g, w, 5..0);
            consume!(g, y, 5);
            consume!(b, y, 5);
            consume!(b, z, 2);
            consume!(g, y, 4);
            consume!(b, w, 5..0);
            consume!(g, z, 5);
            consume!(b, z, 3);
            consume!(b, z, 5);
            consume!(b, z, 4);
            consume!(r, x, 5..0);
            consume!(g, y, 3..0);
            consume!(g, x, 5..0);
            consume!(g, z, 3..0);
            consume!(b, x, 5..0);
            consume!(b, y, 3..0);
            consume!(r, y, 5..0);
            consume!(r, z, 5..0);
        }
    }

    [EndPointPair { a: w, b: x }, EndPointPair { a: y, b: z }]
}
fn decompress_endpoints_two(
    mode: ModeTwo,
    format: BC6HFormat,
    mut endpoints: [EndPointPair; 2],
) -> [EndPointPair; 2] {
    let a_bit_count = mode.a0_bit_count();
    let (delta_r, delta_g, delta_b) = mode.delta_bit_count();

    // sign extend the endpoints
    if format == BC6HFormat::SignedF16 {
        endpoints[0].a.sign_extend_all(a_bit_count);
    }
    if mode.transformed() || format == BC6HFormat::SignedF16 {
        endpoints[0].b.sign_extend(delta_r, delta_g, delta_b);
        endpoints[1].a.sign_extend(delta_r, delta_g, delta_b);
        endpoints[1].b.sign_extend(delta_r, delta_g, delta_b);
    }

    if mode.transformed() {
        // transform B0 -> B0+A0

        let a0 = endpoints[0].a;

        let mask = (1 << a_bit_count) - 1;
        endpoints[0].b = (endpoints[0].b + a0).bit_and(mask);
        endpoints[1].a = (endpoints[1].a + a0).bit_and(mask);
        endpoints[1].b = (endpoints[1].b + a0).bit_and(mask);

        if format == BC6HFormat::SignedF16 {
            endpoints[0].b.sign_extend_all(a_bit_count);
            endpoints[1].a.sign_extend_all(a_bit_count);
            endpoints[1].b.sign_extend_all(a_bit_count);
        }
    }

    endpoints
}

fn unquantize(mut component: i32, u_bits_per_comp: u8, format: BC6HFormat) -> i32 {
    let mut unq: i32;
    match format {
        BC6HFormat::UnsignedF16 => {
            if u_bits_per_comp >= 15 {
                unq = component;
            } else if component == 0 {
                unq = 0;
            } else if component == ((1 << u_bits_per_comp) - 1) {
                unq = 0xFFFF;
            } else {
                unq = ((component << 16) + 0x8000) >> u_bits_per_comp;
            }
        }
        BC6HFormat::SignedF16 => {
            if u_bits_per_comp >= 16 {
                unq = component;
            } else {
                let mut s = false;
                if component < 0 {
                    s = true;
                    component = -component;
                }

                if component == 0 {
                    unq = 0;
                } else if component >= ((1 << (u_bits_per_comp - 1)) - 1) {
                    unq = 0x7FFF;
                } else {
                    unq = ((component << 15) + 0x4000) >> (u_bits_per_comp - 1);
                }

                if s {
                    unq = -unq;
                }
            }
        }
    }

    unq
}

fn finish_unquantize(mut component: i32, format: BC6HFormat) -> u16 {
    match format {
        BC6HFormat::UnsignedF16 => {
            component = (component * 31) >> 6; // scale the magnitude by 31/64
            component as u16
        }
        BC6HFormat::SignedF16 => {
            component = if component < 0 {
                -(((-component) * 31) >> 5)
            } else {
                (component * 31) >> 5
            }; // scale the magnitude by 31/32
            let mut s = 0;
            if component < 0 {
                s = 0x8000;
                component = -component;
            }
            (s | component) as u16
        }
    }
}

const WEIGHT_3: [u8; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHT_4: [u8; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

// c1, c2: endpoints of a component
fn generate_palette_unquantized_one(
    c1: IntColor<i32>,
    c2: IntColor<i32>,
    prec: u8,
    format: BC6HFormat,
) -> [IntColor<u16>; 16] {
    let a = IntColor::new(
        unquantize(c1.r, prec, format),
        unquantize(c1.g, prec, format),
        unquantize(c1.b, prec, format),
    );
    let b = IntColor::new(
        unquantize(c2.r, prec, format),
        unquantize(c2.g, prec, format),
        unquantize(c2.b, prec, format),
    );

    let mut palette: [IntColor<u16>; 16] = Default::default();
    // interpolate
    for i in 0..16 {
        let w = WEIGHT_4[i] as i32;
        palette[i].r = finish_unquantize((a.r * (64 - w) + b.r * w + 32) >> 6, format);
        palette[i].g = finish_unquantize((a.g * (64 - w) + b.g * w + 32) >> 6, format);
        palette[i].b = finish_unquantize((a.b * (64 - w) + b.b * w + 32) >> 6, format);
    }

    palette
}
fn generate_palette_unquantized_two(
    c1: IntColor<i32>,
    c2: IntColor<i32>,
    prec: u8,
    format: BC6HFormat,
) -> [IntColor<u16>; 8] {
    let a = IntColor::new(
        unquantize(c1.r, prec, format),
        unquantize(c1.g, prec, format),
        unquantize(c1.b, prec, format),
    );
    let b = IntColor::new(
        unquantize(c2.r, prec, format),
        unquantize(c2.g, prec, format),
        unquantize(c2.b, prec, format),
    );

    let mut palette: [IntColor<u16>; 8] = Default::default();
    // interpolate
    for i in 0..8 {
        let w = WEIGHT_3[i] as i32;
        palette[i].r = finish_unquantize((a.r * (64 - w) + b.r * w + 32) >> 6, format);
        palette[i].g = finish_unquantize((a.g * (64 - w) + b.g * w + 32) >> 6, format);
        palette[i].b = finish_unquantize((a.b * (64 - w) + b.b * w + 32) >> 6, format);
    }

    palette
}
