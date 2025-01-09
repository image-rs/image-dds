//! Internal module for converting between different number formats.
//!
//! Most magic constants for the U/SNorm conversion are from:
//! https://rundevelopment.github.io/projects/multiply-add-constants-finder

#[derive(Debug, Clone, Copy)]
pub(crate) struct B5G6R5 {
    pub r5: u16,
    pub g6: u16,
    pub b5: u16,
}
impl B5G6R5 {
    #[inline(always)]
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self::from_u16(u16::from_le_bytes(bytes))
    }
    #[inline(always)]
    pub fn from_u16(u: u16) -> Self {
        Self {
            b5: u & 0x1F,
            g6: (u >> 5) & 0x3F,
            r5: (u >> 11) & 0x1F,
        }
    }
    #[inline(always)]
    pub fn to_n8(self) -> [u8; 3] {
        [
            n5::n8(self.r5 as u8),
            n6::n8(self.g6 as u8),
            n5::n8(self.b5 as u8),
        ]
    }
    #[inline(always)]
    pub fn to_n16(self) -> [u16; 3] {
        [
            n5::n16(self.r5 as u8),
            n6::n16(self.g6 as u8),
            n5::n16(self.b5 as u8),
        ]
    }
    #[inline(always)]
    pub fn to_f32(self) -> [f32; 3] {
        [
            n5::f32(self.r5 as u8),
            n6::f32(self.g6 as u8),
            n5::f32(self.b5 as u8),
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct B5G5R5A1 {
    pub r5: u16,
    pub g5: u16,
    pub b5: u16,
    pub a1: u16,
}
impl B5G5R5A1 {
    #[inline(always)]
    pub fn from_u16(u: u16) -> Self {
        Self {
            b5: u & 0x1F,
            g5: (u >> 5) & 0x1F,
            r5: (u >> 10) & 0x1F,
            a1: (u >> 15) & 0x1,
        }
    }
    #[inline(always)]
    pub fn to_n8(self) -> [u8; 4] {
        [
            n5::n8(self.r5 as u8),
            n5::n8(self.g5 as u8),
            n5::n8(self.b5 as u8),
            n1::n8(self.a1 as u8),
        ]
    }
    #[inline(always)]
    pub fn to_n16(self) -> [u16; 4] {
        [
            n5::n16(self.r5 as u8),
            n5::n16(self.g5 as u8),
            n5::n16(self.b5 as u8),
            n1::n16(self.a1 as u8),
        ]
    }
    #[inline(always)]
    pub fn to_f32(self) -> [f32; 4] {
        [
            n5::f32(self.r5 as u8),
            n5::f32(self.g5 as u8),
            n5::f32(self.b5 as u8),
            n1::f32(self.a1 as u8),
        ]
    }
}

/// Functions for converting **FROM Unorm1** values to other formats.
pub(crate) mod n1 {
    #[inline(always)]
    pub fn n8(x: u8) -> u8 {
        debug_assert!(x <= 1);
        if x == 0 {
            0
        } else {
            u8::MAX
        }
    }
    #[inline(always)]
    pub fn n16(x: u8) -> u16 {
        debug_assert!(x <= 1);
        if x == 0 {
            0
        } else {
            u16::MAX
        }
    }
    #[inline(always)]
    pub fn f32(x: u8) -> f32 {
        debug_assert!(x <= 1);
        if x == 0 {
            0.0
        } else {
            1.0
        }
    }
}

/// Functions for converting **FROM Unorm2** values to other formats.
pub(crate) mod n2 {
    #[inline(always)]
    pub fn n8(x: u8) -> u8 {
        debug_assert!(x <= 3);
        x * 85
    }
    #[inline(always)]
    pub fn n16(x: u8) -> u16 {
        debug_assert!(x <= 3);
        x as u16 * 21845
    }
    #[inline(always)]
    pub fn f32(x: u8) -> f32 {
        debug_assert!(x <= 3);
        // This turns out to be exact, so we don't need another method.
        const F: f32 = 1.0 / 3.0;
        x as f32 * F
    }
}

/// Functions for converting **FROM Unorm4** values to other formats.
pub(crate) mod n4 {
    #[inline(always)]
    pub fn n8(x: u8) -> u8 {
        debug_assert!(x <= 15);
        x * 17
    }
    #[inline(always)]
    pub fn n16(x: u8) -> u16 {
        debug_assert!(x <= 15);
        x as u16 * 4369
    }
    #[inline(always)]
    pub fn f32(x: u8) -> f32 {
        debug_assert!(x <= 15);
        const F: f32 = 1.0 / 15.0;
        x as f32 * F
    }
    #[inline(always)]
    pub fn f32_exact(x: u8) -> f32 {
        debug_assert!(x <= 15);
        // Adopted from: https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        // k0=3 was found by trial and error.
        const K0: f32 = 3.0;
        const K1: f32 = 1.0 / (15.0 * K0);
        (x as f32 * K0) * K1
    }
}

/// Functions for converting **FROM Unorm5** values to other formats.
pub(crate) mod n5 {
    #[inline(always)]
    pub fn n8(x: u8) -> u8 {
        debug_assert!(x <= 31);
        ((x as u16 * 2108 + 92) >> 8) as u8
    }
    #[inline(always)]
    pub fn n16(x: u8) -> u16 {
        debug_assert!(x <= 31);
        ((x as u32 * 138547200) >> 16) as u16
    }
    #[inline(always)]
    pub fn f32(x: u8) -> f32 {
        debug_assert!(x <= 31);
        const F: f32 = 1.0 / 31.0;
        x as f32 * F
    }
    #[inline(always)]
    pub fn f32_exact(x: u8) -> f32 {
        debug_assert!(x <= 31);
        // Adopted from: https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        // k0=3 was found by trial and error.
        const K0: f32 = 3.0;
        const K1: f32 = 1.0 / (31.0 * K0);
        (x as f32 * K0) * K1
    }
}

/// Functions for converting **FROM Unorm5** values to other formats.
pub(crate) mod n6 {
    #[inline(always)]
    pub fn n8(x: u8) -> u8 {
        debug_assert!(x <= 63);
        ((x as u16 * 1036 + 132) >> 8) as u8
    }
    #[inline(always)]
    pub fn n16(x: u8) -> u16 {
        debug_assert!(x <= 63);
        ((x as u32 * 68173056 + 30976) >> 16) as u16
    }
    #[inline(always)]
    pub fn f32(x: u8) -> f32 {
        debug_assert!(x <= 63);
        const F: f32 = 1.0 / 63.0;
        x as f32 * F
    }
    #[inline(always)]
    pub fn f32_exact(x: u8) -> f32 {
        debug_assert!(x <= 63);
        // Adopted from: https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        // k0=5 was found by trial and error.
        const K0: f32 = 5.0;
        const K1: f32 = 1.0 / (63.0 * K0);
        (x as f32 * K0) * K1
    }
}

/// Functions for converting **FROM Unorm8** values to other formats.
pub(crate) mod n8 {
    #[inline(always)]
    pub fn n16(x: u8) -> u16 {
        x as u16 * 257
    }
    #[inline(always)]
    pub fn f32(x: u8) -> f32 {
        const F: f32 = 1.0 / 255.0;
        x as f32 * F
    }
    #[inline(always)]
    pub fn f32_exact(x: u8) -> f32 {
        // https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        const K0: f32 = 3.0;
        const K1: f32 = 1.0 / (255.0 * K0);
        (x as f32 * K0) * K1
    }
}

/// Functions for converting **FROM Unorm10** values to other formats.
pub(crate) mod n10 {
    #[inline(always)]
    pub fn n8(x: u16) -> u8 {
        debug_assert!(x <= 1023);
        ((x as u32 * 16336 + 32656) >> 16) as u8
    }
    #[inline(always)]
    pub fn n16(x: u16) -> u16 {
        debug_assert!(x <= 1023);
        ((x as u32 * 4198340 + 32660) >> 16) as u16
    }
    #[inline(always)]
    pub fn f32(x: u16) -> f32 {
        debug_assert!(x <= 1023);
        const F: f32 = 1.0 / 1023.0;
        x as f32 * F
    }
    #[inline(always)]
    pub fn f32_exact(x: u16) -> f32 {
        debug_assert!(x <= 1023);
        // Adopted from: https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        // k0=85 was found by trial and error.
        const K0: f32 = 85.0;
        const K1: f32 = 1.0 / (1023.0 * K0);
        (x as f32 * K0) * K1
    }
}

/// Functions for converting **FROM Unorm16** values to other formats.
pub(crate) mod n16 {
    #[inline(always)]
    pub fn n8(x: u16) -> u8 {
        ((x as u32 * 255 + 32895) >> 16) as u8
    }
    #[inline(always)]
    pub fn f32(x: u16) -> f32 {
        const F: f32 = 1.0 / 65535.0;
        x as f32 * F
    }
    #[inline(always)]
    pub fn f32_exact(x: u16) -> f32 {
        // Adopted from https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        // I couldn't find any k0 that would work, so I used the infinite sum
        // approach from the article instead. This is slower, but oh well.
        const C0: f32 = 1.0 / 65536.0;
        const C1: f32 = (1.0 + 65536.0) / 65536.0 / 65536.0 / 65536.0;
        let temp = x as f32;
        (temp * C0) + (temp * C1)
    }
}

/// Functions for converting **FROM Snorm8** values to other formats.
pub(crate) mod s8 {
    /// Brings it in the range `[0, 254]`.
    #[inline(always)]
    fn norm(x: u8) -> u8 {
        // If you think that we can just do `x.wrapping_add(128)`, you'd be wrong.
        // https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format#format-modifiers
        // Both -128 and -127 map to -1.0. So we have to do more work:
        x.wrapping_add(128).saturating_sub(1)
    }
    #[inline(always)]
    pub fn n8(mut x: u8) -> u8 {
        x = norm(x);
        ((x as u16 * 258 + 2) >> 8) as u8
    }
    #[inline(always)]
    pub fn n16(mut x: u8) -> u16 {
        x = norm(x);
        ((x as u32 * 16909064 + 32520) >> 16) as u16
    }
    /// Unsigned f32.
    #[inline(always)]
    pub fn uf32(mut x: u8) -> f32 {
        x = norm(x);
        const F: f32 = 1.0 / 254.0;
        x as f32 * F
    }
    /// Unsigned f32.
    #[inline(always)]
    pub fn uf32_exact(mut x: u8) -> f32 {
        x = norm(x);
        // Adopted from: https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        // k0=31 was found by trial and error.
        const K0: f32 = 31.0;
        const K1: f32 = 1.0 / (254.0 * K0);
        (x as f32 * K0) * K1
    }
}

/// Functions for converting **FROM Snorm16** values to other formats.
pub(crate) mod s16 {
    /// Brings it in the range `[0, 65534]`.
    #[inline(always)]
    fn norm(x: u16) -> u16 {
        // Same for as for Snorm8.
        x.wrapping_add(32768).saturating_sub(1)
    }
    #[inline(always)]
    pub fn n8(mut x: u16) -> u8 {
        x = norm(x);
        ((x as u32 * 65282 + 8388354) >> 24) as u8
    }
    #[inline(always)]
    pub fn n16(mut x: u16) -> u16 {
        x = norm(x);
        ((x as u32 * 65538 + 2) >> 16) as u16
    }
    /// Unsigned f32.
    #[inline(always)]
    pub fn uf32(mut x: u16) -> f32 {
        x = norm(x);
        const F: f32 = 1.0 / 65534.0;
        x as f32 * F
    }
    /// Unsigned f32.
    #[inline(always)]
    pub fn uf32_exact(mut x: u16) -> f32 {
        x = norm(x);
        // Adopted from: https://fgiesen.wordpress.com/2024/11/06/exact-unorm8-to-float/
        // k0=73 was found by trial and error.
        const K0: f32 = 73.0;
        const K1: f32 = 1.0 / (65534.0 * K0);
        (x as f32 * K0) * K1
    }
}

pub(crate) fn f16_to_f32(half: u16) -> f32 {
    // https://stackoverflow.com/questions/36008434/how-can-i-decode-f16-to-f32-using-only-the-stable-standard-library
    let exp: u16 = half >> 10 & 0b1_1111;
    let mant: u16 = half & 0b11_1111_1111;
    let val: f32 = if exp == 0 {
        // denorm
        mant as f32 * 2.0_f32.powi(-24)
    } else if exp != 31 {
        (mant as f32 + 1024_f32) * 2.0_f32.powi(exp as i32 - 25)
    } else if mant == 0 {
        f32::INFINITY
    } else {
        f32::NAN
    };
    if half & 0x8000 != 0 {
        -val
    } else {
        val
    }
}
pub(crate) fn f11_to_f32(half: u16) -> f32 {
    // based on f16_to_f32
    let exp: u16 = half >> 6 & 0b1_1111;
    let mant: u16 = half & 0b11_1111;
    let val: f32 = if exp == 0 {
        // denorm
        mant as f32 * 2.0_f32.powi(-20)
    } else if exp != 31 {
        (mant as f32 + 64_f32) * 2.0_f32.powi(exp as i32 - 21)
    } else if mant == 0 {
        f32::INFINITY
    } else {
        f32::NAN
    };
    // no sign bit
    val
}
pub(crate) fn f10_to_f32(half: u16) -> f32 {
    // based on f16_to_f32
    let exp: u16 = half >> 5 & 0b1_1111;
    let mant: u16 = half & 0b1_1111;
    let val: f32 = if exp == 0 {
        // denorm
        mant as f32 * 2.0_f32.powi(-19)
    } else if exp != 31 {
        (mant as f32 + 32_f32) * 2.0_f32.powi(exp as i32 - 20)
    } else if mant == 0 {
        f32::INFINITY
    } else {
        f32::NAN
    };
    // no sign bit
    val
}

pub(crate) fn float3_to_bytes(floats: [f32; 3]) -> [u8; 12] {
    bytemuck::cast(floats)
}
pub(crate) fn float4_to_bytes(floats: [f32; 4]) -> [u8; 16] {
    bytemuck::cast(floats)
}

pub(crate) trait ToRgba {
    type Channel;
    fn to_rgba(self) -> [Self::Channel; 4];
}
impl ToRgba for [u8; 3] {
    type Channel = u8;

    #[inline(always)]
    fn to_rgba(self) -> [u8; 4] {
        [self[0], self[1], self[2], u8::MAX]
    }
}
impl ToRgba for [u16; 3] {
    type Channel = u16;

    #[inline(always)]
    fn to_rgba(self) -> [u16; 4] {
        [self[0], self[1], self[2], u16::MAX]
    }
}
impl ToRgba for [f32; 3] {
    type Channel = f32;

    #[inline(always)]
    fn to_rgba(self) -> [f32; 4] {
        [self[0], self[1], self[2], 1.0]
    }
}

pub(crate) trait SwapRB {
    fn swap_rb(self) -> Self;
}
impl<T> SwapRB for [T; 3] {
    #[inline(always)]
    fn swap_rb(self) -> Self {
        let [r, g, b] = self;
        [b, g, r]
    }
}
impl<T> SwapRB for [T; 4] {
    #[inline(always)]
    fn swap_rb(self) -> Self {
        let [r, g, b, a] = self;
        [b, g, r, a]
    }
}

#[cfg(test)]
mod test {
    macro_rules! test_to_unorm {
        ($t:ident, $name:ident, $convert:path, $max_in:expr) => {
            #[test]
            fn $name() {
                assert_eq!($convert(0), 0);
                assert_eq!($convert($max_in), $t::MAX);

                for x in 0..=$max_in {
                    let expected = (x as f64 * $t::MAX as f64 / $max_in as f64).round() as $t;
                    assert_eq!($convert(x), expected);
                }
            }
        };
    }
    test_to_unorm!(u8, n1_to_n8, super::n1::n8, 1);
    test_to_unorm!(u8, n2_to_n8, super::n2::n8, 3);
    test_to_unorm!(u8, n4_to_n8, super::n4::n8, 15);
    test_to_unorm!(u8, n5_to_n8, super::n5::n8, 31);
    test_to_unorm!(u8, n6_to_n8, super::n6::n8, 63);
    test_to_unorm!(u8, n10_to_n8, super::n10::n8, 1023);
    test_to_unorm!(u8, n16_to_n8, super::n16::n8, 65535);
    test_to_unorm!(u16, n1_to_n16, super::n1::n16, 1);
    test_to_unorm!(u16, n2_to_n16, super::n2::n16, 3);
    test_to_unorm!(u16, n4_to_n16, super::n4::n16, 15);
    test_to_unorm!(u16, n5_to_n16, super::n5::n16, 31);
    test_to_unorm!(u16, n6_to_n16, super::n6::n16, 63);
    test_to_unorm!(u16, n8_to_n16, super::n8::n16, 255);
    test_to_unorm!(u16, n10_to_n16, super::n10::n16, 1023);

    macro_rules! test_to_f32 {
        ($name:ident, $convert:path, $max_in:expr) => {
            #[test]
            fn $name() {
                assert_eq!($convert(0), 0.0);
                assert_eq!($convert($max_in), 1.0);

                for x in 0..=$max_in {
                    let expected = (x as f64 / $max_in as f64) as f32;
                    let actual = $convert(x);
                    if expected != actual {
                        let rel_err = (actual as f64 - expected as f64).abs() / expected as f64;
                        const MAX_REL_ERROR: f64 = 1.0 / $max_in as f64 / 100.0;
                        if rel_err > MAX_REL_ERROR {
                            assert_eq!(actual, expected, "failed for x={}, rel_err={}", x, rel_err);
                        }
                    }
                }
            }
        };
    }
    test_to_f32!(n1_to_f32, super::n1::f32, 1);
    test_to_f32!(n2_to_f32, super::n2::f32, 3);
    test_to_f32!(n4_to_f32, super::n4::f32, 15);
    test_to_f32!(n5_to_f32, super::n5::f32, 31);
    test_to_f32!(n6_to_f32, super::n6::f32, 63);
    test_to_f32!(n8_to_f32, super::n8::f32, 255);
    test_to_f32!(n10_to_f32, super::n10::f32, 1023);
    test_to_f32!(n16_to_f32, super::n16::f32, 65535);

    macro_rules! test_to_f32_exact {
        ($name:ident, $convert:path, $max_in:expr) => {
            #[test]
            fn $name() {
                assert_eq!($convert(0), 0.0);
                assert_eq!($convert($max_in), 1.0);

                for x in 0..=$max_in {
                    let expected = (x as f64 / $max_in as f64) as f32;
                    assert_eq!($convert(x), expected, "failed for x={}", x);
                }
            }
        };
    }
    test_to_f32_exact!(n1_to_f32_exact, super::n1::f32, 1);
    test_to_f32_exact!(n2_to_f32_exact, super::n2::f32, 3);
    test_to_f32_exact!(n4_to_f32_exact, super::n4::f32_exact, 15);
    test_to_f32_exact!(n5_to_f32_exact, super::n5::f32_exact, 31);
    test_to_f32_exact!(n6_to_f32_exact, super::n6::f32_exact, 63);
    test_to_f32_exact!(n8_to_f32_exact, super::n8::f32_exact, 255);
    test_to_f32_exact!(n10_to_f32_exact, super::n10::f32_exact, 1023);
    test_to_f32_exact!(n16_to_f32_exact, super::n16::f32_exact, 65535);

    macro_rules! test_snorm_to_unorm {
        ($in:ident / $in_unsigned:ident => $t:ident, $name:ident, $convert:path) => {
            #[test]
            fn $name() {
                assert_eq!($convert($in::MIN as $in_unsigned), 0);
                assert_eq!($convert(-$in::MAX as $in_unsigned), 0);
                assert_eq!($convert(0 as $in as $in_unsigned), $t::MAX / 2 + 1);
                assert_eq!($convert($in::MAX as $in_unsigned), $t::MAX);

                for x in 0..=$in_unsigned::MAX {
                    let xi = x as $in;
                    let expected = ((xi.max(-$in::MAX) as f64 / $in::MAX as f64 + 1.0) / 2.0
                        * $t::MAX as f64)
                        .round() as $t;
                    assert_eq!($convert(x), expected, "failed for x={} (u{})", xi, x);
                }
            }
        };
    }
    test_snorm_to_unorm!(i8/u8 => u8, s8_to_n8, super::s8::n8);
    test_snorm_to_unorm!(i16/u16 => u8, s16_to_n8, super::s16::n8);
    test_snorm_to_unorm!(i8/u8 => u16, s8_to_n16, super::s8::n16);
    test_snorm_to_unorm!(i16/u16 => u16, s16_to_n16, super::s16::n16);

    macro_rules! test_snorm_to_f32 {
        ($in:ident / $in_unsigned:ident, $name:ident, $convert:path) => {
            #[test]
            fn $name() {
                assert_eq!($convert($in::MIN as $in_unsigned), 0.0);
                assert_eq!($convert(-$in::MAX as $in_unsigned), 0.0);
                assert_eq!($convert(0 as $in as $in_unsigned), 0.5);
                assert_eq!($convert($in::MAX as $in_unsigned), 1.0);

                for x in 0..=$in_unsigned::MAX {
                    let xi = x as $in;
                    let expected =
                        ((xi.max(-$in::MAX) as f64 / $in::MAX as f64 + 1.0) / 2.0) as f32;
                    let actual = $convert(x);
                    if expected != actual {
                        let rel_err = (actual as f64 - expected as f64).abs() / expected as f64;
                        const MAX_REL_ERROR: f64 = 1.0 / $in::MAX as f64 / 100.0;
                        if rel_err > MAX_REL_ERROR {
                            assert_eq!(
                                actual, expected,
                                "failed for x={} (u{}), rel_err={}",
                                xi, x, rel_err
                            );
                        }
                    }
                }
            }
        };
    }
    test_snorm_to_f32!(i8 / u8, s8_to_uf32, super::s8::uf32);
    test_snorm_to_f32!(i16 / u16, s16_to_uf32, super::s16::uf32);

    macro_rules! test_snorm_to_f32_exact {
        ($in:ident / $in_unsigned:ident, $name:ident, $convert:path) => {
            #[test]
            fn $name() {
                assert_eq!($convert($in::MIN as $in_unsigned), 0.0);
                assert_eq!($convert(-$in::MAX as $in_unsigned), 0.0);
                assert_eq!($convert(0 as $in as $in_unsigned), 0.5);
                assert_eq!($convert($in::MAX as $in_unsigned), 1.0);

                for x in 0..=$in_unsigned::MAX {
                    let xi = x as $in;
                    let expected =
                        ((xi.max(-$in::MAX) as f64 / $in::MAX as f64 + 1.0) / 2.0) as f32;
                    assert_eq!($convert(x), expected, "failed for x={} (u{})", xi, x);
                }
            }
        };
    }
    test_snorm_to_f32_exact!(i8 / u8, s8_to_uf32_exact, super::s8::uf32_exact);
    test_snorm_to_f32_exact!(i16 / u16, s16_to_uf32_exact, super::s16::uf32_exact);
}
