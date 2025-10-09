use glam::Vec4;

pub(crate) fn compress_bc7_block(block: [Rgba<8>; 16]) -> [u8; 16] {
    let stats = BlockStats::new(&block);

    // a block of a single color can be compressed exactly
    if let Some(color) = stats.single_color() {
        return compress_single_color(color);
    }

    // TODO: just for testing
    compress_single_color(block[5])
}

/// https://fgiesen.wordpress.com/2024/11/03/bc7-optimal-solid-color-blocks/
fn compress_single_color(color: Rgba<8>) -> [u8; 16] {
    // TODO: implement this properly

    mode5(
        Rotation::None,
        [Rgb::new(color.r >> 1, color.g >> 1, color.b >> 1); 2],
        IndexList::constant(0),
        [Alpha::new(color.a); 2],
        IndexList::constant(0),
    )
}

fn mode5(
    rotation: Rotation,
    mut color: [Rgb<7>; 2],
    mut color_indexes: IndexList<2>,
    mut alpha: [Alpha<8>; 2],
    mut alpha_indexes: IndexList<2>,
) -> [u8; 16] {
    let mut stream = BitStream::new();
    stream.write_mode(5);
    stream.write_rotation(rotation);
    stream.write_endpoints_rgb(color);
    stream.write_endpoints_alpha(alpha);

    // TODO: indexes
    stream.write_u64(0, 31);
    stream.write_u64(0, 31);

    stream.finish()
}

fn swap<T: Copy>(array: &mut [T; 2]) {
    let tmp = array[0];
    array[0] = array[1];
    array[1] = tmp;
}

enum Rotation {
    None = 0,
    AR = 1,
    AG = 2,
    AB = 3,
}

/// A list of 16 indexes each using B bits.
///
/// B must be 2, 3, or 4.
struct IndexList<const B: u8> {
    indexes: u64,
}
impl<const B: u8> IndexList<B> {
    const MAX_INDEX: u8 = (1 << B) - 1;

    const fn new() -> Self {
        debug_assert!(B == 2 || B == 3 || B == 4);
        Self { indexes: 0 }
    }
    const CONSTANT_MULTIPLE: u64 = {
        let mut m = 0;
        let mut i = 0;
        while i < 16 {
            m |= 1 << (i * B);
            i += 1;
        }
        m
    };
    fn constant(value: u8) -> Self {
        debug_assert!(B == 2 || B == 3 || B == 4);
        debug_assert!(value <= Self::MAX_INDEX);

        Self {
            indexes: value as u64 * Self::CONSTANT_MULTIPLE,
        }
    }

    fn get(&self, index: usize) -> u8 {
        debug_assert!(index < 16);
        ((self.indexes >> (index * B as usize)) & Self::MAX_INDEX as u64) as u8
    }
    fn set(&mut self, index: usize, value: u8) {
        debug_assert!(index < 16);
        debug_assert!(value <= Self::MAX_INDEX);
        debug_assert!(self.get(index) == 0, "Cannot set an index twice.");
        self.indexes |= (value as u64) << (index * B as usize);
    }
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

    pub fn round(v: Vec4) -> Self {
        debug_assert!(0 < B && B <= 8);
        let x = v.min(Vec4::ONE) * (Self::MAX as f32) + 0.5;
        Self::new(x.x as u8, x.y as u8, x.z as u8, x.w as u8)
    }

    pub fn to_u32(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, self.a])
    }
    pub fn from_u32(x: u32) -> Self {
        let [r, g, b, a] = x.to_le_bytes();
        Self::new(r, g, b, a)
    }
}
impl<const B: u8> PartialEq for Rgba<B> {
    fn eq(&self, other: &Self) -> bool {
        self.to_u32() == other.to_u32()
    }
}
impl<const B: u8> Eq for Rgba<B> {}

#[repr(C, align(4))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Rgb<const B: u8> {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pad: u8,
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
        Self { r, g, b, pad: 0 }
    }

    pub fn to_u32(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, self.pad])
    }
    pub fn from_u32(x: u32) -> Self {
        let [r, g, b, _] = x.to_le_bytes();
        Self::new(r, g, b)
    }
}
impl<const B: u8> PartialEq for Rgb<B> {
    fn eq(&self, other: &Self) -> bool {
        self.to_u32() == other.to_u32()
    }
}
impl<const B: u8> Eq for Rgb<B> {}

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
}

struct BlockStats {
    and: Rgba<8>,
    or: Rgba<8>,
}
impl BlockStats {
    fn new(block: &[Rgba<8>; 16]) -> Self {
        let mut and: u32 = u32::MAX;
        let mut or: u32 = 0;
        for &pixel in block {
            let u = pixel.to_u32();
            and &= u;
            or |= u;
        }
        Self {
            and: Rgba::from_u32(and),
            or: Rgba::from_u32(or),
        }
    }

    fn single_color(&self) -> Option<Rgba<8>> {
        if self.and == self.or {
            Some(self.and)
        } else {
            None
        }
    }

    fn constant_alpha(&self) -> Option<u8> {
        if self.and.a == self.or.a {
            Some(self.and.a)
        } else {
            None
        }
    }
    /// Returns whether Alpha is 255 everywhere.
    fn opaque(&self) -> bool {
        self.and.a == 255
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

    fn write_mode(&mut self, mode: u8) {
        debug_assert!(mode < 8);
        self.write_u64(1 << mode, mode + 1);
    }
    fn write_rotation(&mut self, rotation: Rotation) {
        self.write_u64(rotation as u64, 2);
    }
    fn write_endpoints_rgb<const B: u8>(&mut self, endpoints: [Rgb<B>; 2]) {
        self.write_u64(endpoints[0].r as u64, B);
        self.write_u64(endpoints[1].r as u64, B);
        self.write_u64(endpoints[0].g as u64, B);
        self.write_u64(endpoints[1].g as u64, B);
        self.write_u64(endpoints[0].b as u64, B);
        self.write_u64(endpoints[1].b as u64, B);
    }
    fn write_endpoints_alpha<const B: u8>(&mut self, endpoints: [Alpha<B>; 2]) {
        self.write_u64(endpoints[0].a as u64, B);
        self.write_u64(endpoints[1].a as u64, B);
    }

    fn finish(self) -> [u8; 16] {
        debug_assert!(self.bits == 128);
        self.data.to_le_bytes()
    }
}
