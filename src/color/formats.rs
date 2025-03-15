//! Internal module for converting between different number formats.
//!
//! Most magic constants for the U/SNorm conversion are from:
//! https://rundevelopment.github.io/projects/multiply-add-constants-finder

use super::Norm;

#[derive(Debug, Clone, Copy)]
pub(crate) struct B5G6R5 {
    pub r5: u16,
    pub g6: u16,
    pub b5: u16,
}
impl B5G6R5 {
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

    // The nearest RGB8 color that represents `self * 2/3 + color * 1/3`.
    pub(crate) fn one_third_color_rgb8(self, color: Self) -> [u8; 3] {
        let r = self.r5 * 2 + color.r5;
        let g = self.g6 * 2 + color.g6;
        let b = self.b5 * 2 + color.b5;

        let r = ((r * 351 + 61) >> 7) as u8;
        let g = ((g as u32 * 2763 + 1039) >> 11) as u8;
        let b = ((b * 351 + 61) >> 7) as u8;
        [r, g, b]
    }
    // The nearest RGB8 color that represents `self * 1/2 + color * 1/2`.
    pub(crate) fn mid_color_rgb8(self, color: Self) -> [u8; 3] {
        let r = self.r5 + color.r5;
        let g = self.g6 + color.g6;
        let b = self.b5 + color.b5;

        let r = ((r * 1053 + 125) >> 8) as u8;
        let g = ((g as u32 * 4145 + 1019) >> 11) as u8;
        let b = ((b * 1053 + 125) >> 8) as u8;
        [r, g, b]
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

    pub fn from_f32(x: f32) -> u8 {
        if x >= 0.5 {
            1
        } else {
            0
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

    pub fn from_f32(x: f32) -> u8 {
        (x.min(1.0) * 3.0 + 0.5) as u8
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

    pub fn from_f32(x: f32) -> u8 {
        (x.min(1.0) * 15.0 + 0.5) as u8
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

    pub fn from_f32(x: f32) -> u8 {
        (x.min(1.0) * 31.0 + 0.5) as u8
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

    pub fn from_f32(x: f32) -> u8 {
        (x.min(1.0) * 63.0 + 0.5) as u8
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

    pub fn from_f32(x: f32) -> u8 {
        (x * 255.0 + 0.5) as u8
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

    pub fn from_f32(x: f32) -> u16 {
        (x.min(1.0) * 1023.0 + 0.5) as u16
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

    pub fn from_f32(x: f32) -> u16 {
        (x * 65535.0 + 0.5) as u16
    }
}

/// Functions for converting **FROM Snorm8** values to other formats.
pub(crate) mod s8 {
    /// Brings it in the range `[0, 254]`.
    #[inline(always)]
    pub fn norm(x: u8) -> u8 {
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

    pub fn from_n8(x: u8) -> u8 {
        // this computes round(x / 255 * 254)
        // range: 0-254
        let norm = ((x as u16 * 254 + 254) >> 8) as u8;
        from_norm(norm)
    }
    pub fn from_uf32(x: f32) -> u8 {
        let norm = (x.min(1.0) * 254.0 + 0.5) as u8;
        from_norm(norm)
    }
    // Converts a value in the range [0, 254] to SNORM8.
    pub fn from_norm(x: u8) -> u8 {
        debug_assert!(x <= 254);
        (x + 1).wrapping_sub(128)
    }
}

/// Functions for converting **FROM Snorm16** values to other formats.
pub(crate) mod s16 {
    /// Brings it in the range `[0, 65534]`.
    #[inline(always)]
    pub fn norm(x: u16) -> u16 {
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

    pub fn from_n16(x: u16) -> u16 {
        // this computes round(x / 65535 * 65534)
        // range: 0-254
        let norm = ((x as u32 * 65534 + 65534) >> 16) as u16;
        (norm + 1).wrapping_sub(32768)
    }
    pub fn from_uf32(x: f32) -> u16 {
        let norm = (x.min(1.0) * 65534.0 + 0.5) as u16;
        (norm + 1).wrapping_sub(32768)
    }
}

/// Functions for converting **FROM 10-bit XR_BIAS** values to other formats.
///
/// These are 2.8 fixed-point numbers, meaning 2 integer bits and 8 fractional
/// bits. These numbers are biased by -1.5 and then scaled by 256/510, resulting
/// in an effective range of `[-0.75294, 1.25294]`. XR (probably) means extended
/// range.
///
/// The conversion from 10-bit XR_BIAS to float is:
///
/// ```c
/// // source: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/xr-bias-to-float-conversion-rules
/// float XRtoFloat( UINT XRComponent ) {
///     // The & 0x3ff shows that only 10 bits contribute to the conversion.
///     return (float)( (XRComponent & 0x3ff) - 0x180 ) / 510.f;
/// }
/// ```
pub(crate) mod xr10 {
    #[inline(always)]
    pub fn n8(x: u16) -> u8 {
        // new range: [-384, 639] (or [-0.75294, 1.25294])
        let x = x as i16 - 0x180;
        // new range: [0, 510] (or [0.0, 1.0])
        let x = x.clamp(0, 510) as u16;
        // this is round(x / 510 * 255), but faster
        ((x + 1) >> 1) as u8
    }
    #[inline(always)]
    pub fn n16(x: u16) -> u16 {
        // new range: [-384, 639] (or [-0.75294, 1.25294])
        let x = x as i16 - 0x180;
        // new range: [0, 510] (or [0.0, 1.0])
        let x = x.clamp(0, 510) as u16;
        // this is round(x / 510 * 65535), but faster
        ((x as u32 * 8421376 + 65535) >> 16) as u16
    }
    #[inline(always)]
    pub fn f32(x: u16) -> f32 {
        // 0x180 == 1.5 in 2.8 fixed-point.
        const F: f32 = 1.0 / 510.0;
        (x as i16 - 0x180) as f32 * F
    }

    #[inline(always)]
    pub fn from_f32(x: f32) -> u16 {
        ((x * 510.0 + 384.5) as u16).min(1023)
    }
}

/// Functions for converting `f32` values to other formats.
pub(crate) mod fp {
    #[inline(always)]
    pub fn n8(x: f32) -> u8 {
        (x * 255.0 + 0.5) as u8
    }
    #[inline(always)]
    pub fn n16(x: f32) -> u16 {
        (x * 65535.0 + 0.5) as u16
    }
}

/// Functions for converting `f16` values to other formats.
pub(crate) mod fp16 {
    use crate::util::{two_powi, unlikely_branch};

    #[inline]
    pub fn n8(x: u16) -> u8 {
        // This is optimized implementation, combining fp16::f32 -> fp::n8 into one step.
        let exp: u16 = x >> 10 & 0b1_1111;
        let mant: u16 = x & 0b11_1111_1111;
        // Note: denorm all go to zero after rounding, so they don't need an extra branch.
        let val: u8 = if exp != 31 {
            ((mant as f32 + 1024_f32) * two_powi(exp as i8 - 25) * 255.0 + 0.5) as u8
        } else {
            unlikely_branch();
            if mant == 0 {
                // Inf goes to u8::MAX
                u8::MAX
            } else {
                // NaN goes to zero
                0
            }
        };
        if x & 0x8000 != 0 {
            // negative numbers go to zero
            0
        } else {
            val
        }
    }
    #[inline]
    pub fn n16(x: u16) -> u16 {
        // This is optimized implementation, combining fp16::f32 -> fp::n16 into one step.
        let exp: u16 = x >> 10 & 0b1_1111;
        let mant: u16 = x & 0b11_1111_1111;
        let val: u16 = if exp == 0 {
            // denorm
            unlikely_branch();
            const F: f32 = 65535.0 / 16777216.0;
            (mant as f32 * F + 0.5) as u16
        } else if exp != 31 {
            ((mant as f32 + 1024_f32) * two_powi(exp as i8 - 25) * 65535.0 + 0.5) as u16
        } else {
            unlikely_branch();
            if mant == 0 {
                // Inf goes to u16::MAX
                u16::MAX
            } else {
                // NaN goes to zero
                0
            }
        };
        if x & 0x8000 != 0 {
            // negative numbers go to zero
            0
        } else {
            val
        }
    }
    #[inline]
    pub fn f32(x: u16) -> f32 {
        // https://stackoverflow.com/questions/36008434/how-can-i-decode-f16-to-f32-using-only-the-stable-standard-library
        let exp: u16 = x >> 10 & 0b1_1111;
        let mant: u16 = x & 0b11_1111_1111;
        let val: f32 = if exp == 0 {
            // denorm
            unlikely_branch();
            mant as f32 * two_powi(-24)
        } else if exp != 31 {
            (mant as f32 + 1024_f32) * two_powi(exp as i8 - 25)
        } else {
            unlikely_branch();
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        };
        if x & 0x8000 != 0 {
            -val
        } else {
            val
        }
    }

    pub fn from_f32(value: f32) -> u16 {
        // Source: https://github.com/starkat99/half-rs/blob/2c4122db4e8f7d8ce030bb4b5ed8913bd6bbf2b1/src/binary16/arch.rs#L482
        // Author: Kathryn Long
        // License: MIT OR Apache-2.0

        // Convert to raw bytes
        let x: u32 = value.to_bits();

        // Extract IEEE754 components
        let sign = x & 0x8000_0000u32;
        let exp = x & 0x7F80_0000u32;
        let man = x & 0x007F_FFFFu32;

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7F80_0000u32 {
            // Set mantissa MSB for NaN (and also keep shifted mantissa bits)
            let nan_bit = if man == 0 { 0 } else { 0x0200u32 };
            return ((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 13)) as u16;
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = ((exp >> 23) as i32) - 127;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return (half_sign | 0x7C00u32) as u16;
        }

        // Check for underflow
        if half_exp <= 0 {
            // Check mantissa for what we can do
            if 14 - half_exp > 24 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return half_sign as u16;
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x0080_0000u32;
            let mut half_man = man >> (14 - half_exp);
            // Check for rounding (see comment above functions)
            let round_bit = 1 << (13 - half_exp);
            if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
                half_man += 1;
            }
            // No exponent for subnormals
            return (half_sign | half_man) as u16;
        }

        // Rebias the exponent
        let half_exp = (half_exp as u32) << 10;
        let half_man = man >> 13;
        // Check for rounding (see comment above functions)
        let round_bit = 0x0000_1000u32;
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            // Round it
            ((half_sign | half_exp | half_man) + 1) as u16
        } else {
            (half_sign | half_exp | half_man) as u16
        }
    }
}

/// Functions for converting `f16` values to other formats.
pub(crate) mod bc6h_uf16 {
    use crate::util::{two_powi, unlikely_branch};

    #[inline]
    pub fn n8(x: u16) -> u8 {
        // This is optimized implementation, combining fp16::f32 -> fp::n8 into one step.
        let exp: u16 = x >> 10 & 0b1_1111;
        let mant: u16 = x & 0b11_1111_1111;

        debug_assert!(x & 0x8000 == 0, "BC6H_UF16 values are positive.");
        debug_assert!(exp < 31, "BC6H_UF16 values cannot be +-INF and NaN.");

        // Note: denorm all go to zero after rounding, so they don't need an extra branch.
        ((mant as f32 + 1024_f32) * two_powi(exp as i8 - 25) * 255.0 + 0.5) as u8
    }
    #[inline]
    pub fn n16(x: u16) -> u16 {
        // This is optimized implementation, combining fp16::f32 -> fp::n16 into one step.
        let exp: u16 = x >> 10 & 0b1_1111;
        let mant: u16 = x & 0b11_1111_1111;

        debug_assert!(x & 0x8000 == 0, "BC6H_UF16 values are positive.");
        debug_assert!(exp < 31, "BC6H_UF16 values cannot be +-INF and NaN.");

        if exp == 0 {
            // denorm
            unlikely_branch();
            const F: f32 = 65535.0 / 16777216.0;
            (mant as f32 * F + 0.5) as u16
        } else {
            ((mant as f32 + 1024_f32) * two_powi(exp as i8 - 25) * 65535.0 + 0.5) as u16
        }
    }
    #[inline]
    pub fn f32(x: u16) -> f32 {
        // https://stackoverflow.com/questions/36008434/how-can-i-decode-f16-to-f32-using-only-the-stable-standard-library
        let exp: u16 = x >> 10 & 0b1_1111;
        let mant: u16 = x & 0b11_1111_1111;

        debug_assert!(x & 0x8000 == 0, "BC6H_UF16 values are positive.");
        debug_assert!(exp < 31, "BC6H_UF16 values cannot be +-INF and NaN.");

        if exp == 0 {
            // denorm
            unlikely_branch();
            mant as f32 * two_powi(-24)
        } else {
            (mant as f32 + 1024_f32) * two_powi(exp as i8 - 25)
        }
    }
}

/// This will round the mantissa of the given `f32` value to `n` bits (not
/// including the implicit 1 bit). Round half up is used.
///
/// All bits after the first `n` bits are set to zero.
#[inline]
fn f32_mantissa_round_half_up(n: u32, mut x: f32) -> f32 {
    debug_assert!(x >= 0.0);

    // This implements round to nearest even to a 6-bit mantissa.
    // It works as follows:
    // 1. Exact the f32 mantissa (23 bits). This will have a value of
    //    (1.)mmm... where mmm... is the mantissa.
    // 2. Truncate the mantissa to n+1 bits.
    // 3. Round-to-nearest-even only rounds up if the last 2 bits are
    //    0b11.
    // 4. If we DO NOT round up, truncate the mantissa to n bits. Done.
    // 5. If we DO round up, truncate the mantissa to n bits and add 1.
    //    This has to be done carefully, because the +1 might cause the
    //    mantissa to overflow n bits and increment the exponent.
    const MANTISSA_MASK: u32 = 0x007F_FFFF;

    // Add 0.5 * 2^(exp - n) for rounding

    // The f32 value with the mantissa all zero.
    // This represents a value of 2^exp
    let f_m0 = x.to_bits() & !MANTISSA_MASK;
    // The f32 value with only the (n+1)-th mantissa bit set
    // This represents a value of 2^exp + 2^(exp - n - 1)
    let f_mn = f_m0 | (1 << (23 - n - 1));
    let f_0_5 = f32::from_bits(f_mn) - f32::from_bits(f_m0);

    x += f_0_5;

    // truncate to n bits
    f32::from_bits(x.to_bits() & !(MANTISSA_MASK >> n))
}
fn f32_to_unsigned_fp_e5(n: u32, mut x: f32) -> u16 {
    if x.is_nan() {
        let nan: u16 = (1 << (n + 5)) - 1;
        return nan;
    }
    if x <= 0.0 {
        return 0;
    }

    if x.is_normal() {
        x = f32_mantissa_round_half_up(n, x);
    }

    let f16 = super::fp16::from_f32(x);
    let exp: u16 = f16 >> 10 & 0b1_1111;
    let mant: u16 = (f16 & 0b11_1111_1111) >> (10 - n);
    exp << n | mant
}
/// Functions for converting `f11` values to other formats.
pub(crate) mod fp11 {
    use crate::util::{two_powi, unlikely_branch};

    #[inline]
    pub fn n8(x: u16) -> u8 {
        let exp: u16 = x >> 6 & 0b1_1111;
        let mant: u16 = x & 0b11_1111;
        // no sign bit

        if exp != 31 {
            ((mant as f32 + 64_f32) * two_powi(exp as i8 - 21) * 255.0 + 0.5) as u8
        } else {
            unlikely_branch();
            if mant == 0 {
                255
            } else {
                0
            }
        }
    }
    #[inline]
    pub fn n16(x: u16) -> u16 {
        let exp: u16 = x >> 6 & 0b1_1111;
        let mant: u16 = x & 0b11_1111;
        // no sign bit

        if exp == 0 {
            // denorm
            // (mant as f32 * two_powi(-20) * 65535.0 + 0.5) as u16
            (mant + 7) >> 4
        } else if exp != 31 {
            ((mant as f32 + 64_f32) * two_powi(exp as i8 - 21) * 65535.0 + 0.5) as u16
        } else {
            unlikely_branch();
            if mant == 0 {
                u16::MAX
            } else {
                0
            }
        }
    }
    #[inline]
    pub fn f32(x: u16) -> f32 {
        // based on f16_to_f32
        let exp: u16 = x >> 6 & 0b1_1111;
        let mant: u16 = x & 0b11_1111;
        // no sign bit

        if exp == 0 {
            // denorm
            mant as f32 * two_powi(-20)
        } else if exp != 31 {
            (mant as f32 + 64_f32) * two_powi(exp as i8 - 21)
        } else {
            unlikely_branch();
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
    }

    #[inline]
    pub fn from_f32(x: f32) -> u16 {
        super::f32_to_unsigned_fp_e5(6, x)
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;

        #[test]
        fn creation() {
            // all negative values should go to zero
            let negative_values = [-0.0, -1.0, -1e-20, f32::NEG_INFINITY];
            for value in negative_values {
                assert_eq!(fp11::f32(fp11::from_f32(value)), 0.0, "value: {}", value);
            }

            // the following can be presented exactly
            let exact_values = [
                0.0,
                0.5,
                1.0,
                1.5,
                33.0,
                65.0,
                127.0,
                128.0,
                130.0,
                f32::INFINITY,
            ];
            for value in exact_values {
                let f11 = fp11::from_f32(value);
                let f32 = fp11::f32(f11);
                assert_eq!(value, f32, "value: {}", value);
            }
            assert_eq!(fp11::from_f32(f32::NAN), 0b11111_111111);
            assert!(fp11::f32(fp11::from_f32(f32::NAN)).is_nan());

            // inexact values are rounded round-to-nearest-even
            let inexact_values = [
                (129.0, 130.0),
                (131.0, 132.0),
                (133.0, 134.0),
                (135.0, 136.0),
                (137.0, 138.0),
                (255.0, 256.0),
                (257.0, 256.0),
            ];
            for (value, expected) in inexact_values {
                let f11 = fp11::from_f32(value);
                let f32 = fp11::f32(f11);
                assert_eq!(expected, f32, "value: {}", value);
            }
        }
    }
}

/// Functions for converting `f10` values to other formats.
pub(crate) mod fp10 {
    use crate::util::{two_powi, unlikely_branch};

    #[inline]
    pub fn n8(x: u16) -> u8 {
        let exp: u16 = x >> 5 & 0b1_1111;
        let mant: u16 = x & 0b1_1111;
        // no sign bit

        if exp != 31 {
            ((mant as f32 + 32_f32) * two_powi(exp as i8 - 20) * 255.0 + 0.5) as u8
        } else {
            unlikely_branch();
            if mant == 0 {
                255
            } else {
                0
            }
        }
    }
    #[inline]
    pub fn n16(x: u16) -> u16 {
        let exp: u16 = x >> 5 & 0b1_1111;
        let mant: u16 = x & 0b1_1111;
        // no sign bit

        if exp == 0 {
            // denorm
            // (mant as f32 * two_powi(-19) * 65535.0 + 0.5) as u16
            (mant + 3) >> 3
        } else if exp != 31 {
            ((mant as f32 + 32_f32) * two_powi(exp as i8 - 20) * 65535.0 + 0.5) as u16
        } else {
            unlikely_branch();
            if mant == 0 {
                u16::MAX
            } else {
                0
            }
        }
    }
    #[inline]
    pub fn f32(x: u16) -> f32 {
        // based on f16_to_f32
        let exp: u16 = x >> 5 & 0b1_1111;
        let mant: u16 = x & 0b1_1111;
        // no sign bit

        if exp == 0 {
            // denorm
            mant as f32 * two_powi(-19)
        } else if exp != 31 {
            (mant as f32 + 32_f32) * two_powi(exp as i8 - 20)
        } else {
            unlikely_branch();
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
    }

    #[inline]
    pub fn from_f32(x: f32) -> u16 {
        super::f32_to_unsigned_fp_e5(5, x)
    }
}

/// Optimized functions for the R9G9B9E5_SHAREDEXP format.
/// https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#3.2.2%20Floating%20Point%20Conversion
pub(crate) mod rgb9995f {
    use crate::util::two_powi;

    #[inline]
    pub fn f32(rgb: u32) -> [f32; 3] {
        let r_mant = rgb & 0x1FF;
        let g_mant = (rgb >> 9) & 0x1FF;
        let b_mant = (rgb >> 18) & 0x1FF;
        let exp = (rgb >> 27) & 0x1F;

        let f = two_powi(exp as i8 - 24);
        [r_mant as f32 * f, g_mant as f32 * f, b_mant as f32 * f]
    }
    #[inline]
    pub fn n8(rgb: u32) -> [u8; 3] {
        let r_mant = rgb & 0x1FF;
        let g_mant = (rgb >> 9) & 0x1FF;
        let b_mant = (rgb >> 18) & 0x1FF;
        let exp = (rgb >> 27) & 0x1F;

        // This is just the f32 conversion and f32 -> UNORM8 conversion
        // combined into one step.
        //
        // NOTE: I originally used a fixed-point math implementation, but
        // it was around 50% slower. I also looked into using f16 -> u8
        // hardware instructions (x86 f16c VCVTPH2PS), but this isn't
        // possible simply because the mantissa here has an *explicit* one
        // at the start. I also suspect that fixing up the one bit would make
        // R9G9B9E5 -> f16 -> f32 -> u8 slower than what I use below.

        let f = two_powi(exp as i8 - 24) * 255.0;
        [
            (r_mant as f32 * f + 0.5) as u8,
            (g_mant as f32 * f + 0.5) as u8,
            (b_mant as f32 * f + 0.5) as u8,
        ]
    }
    #[inline]
    pub fn n16(rgb: u32) -> [u16; 3] {
        let r_mant = rgb & 0x1FF;
        let g_mant = (rgb >> 9) & 0x1FF;
        let b_mant = (rgb >> 18) & 0x1FF;
        let exp = (rgb >> 27) & 0x1F;

        // This method is essentially the same as the above n8 method, so see
        // above for more information. The only difference is that denorms can
        // no longer fall through.

        let f = two_powi(exp as i8 - 24) * 65535.0;
        [
            (r_mant as f32 * f + 0.5) as u16,
            (g_mant as f32 * f + 0.5) as u16,
            (b_mant as f32 * f + 0.5) as u16,
        ]
    }

    #[inline]
    pub fn from_f32(rgb: [f32; 3]) -> u32 {
        // values are now either in range or NaN
        let [r, g, b] = rgb.map(|c| c.clamp(0.0, 65408.0));
        let max = r.max(g).max(b);

        if max.is_nan() || max == 0.0 || max.is_subnormal() {
            // all channels are either NaN or zero
            // sub-normal numbers also map to zero
            return 0;
        }

        // get the f32 exponent of max
        let raw_exp = max.to_bits() >> 23 & 0xFF;
        let mut exp = (raw_exp as i32 - 127 + 16).max(0) as u32;
        debug_assert!(exp <= 31);

        let f = two_powi(-(exp as i8 - 24));
        let mut r_mant = (r * f + 0.5) as u32;
        let mut g_mant = (g * f + 0.5) as u32;
        let mut b_mant = (b * f + 0.5) as u32;
        if r_mant == 512 || g_mant == 512 || b_mant == 512 {
            // This means that the mantissa overflowed to 10-bit while rounding.
            // So we need to increment the exponent and re-calculate the mantissas.
            exp += 1;
            debug_assert!(exp <= 31);

            let f = two_powi(-(exp as i8 - 24));
            r_mant = (r * f + 0.5) as u32;
            g_mant = (g * f + 0.5) as u32;
            b_mant = (b * f + 0.5) as u32;
        }
        debug_assert!(r_mant <= 511);
        debug_assert!(g_mant <= 511);
        debug_assert!(b_mant <= 511);

        r_mant | g_mant << 9 | b_mant << 18 | exp << 27
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;

        #[test]
        fn creation() {
            // these values are presented by zero
            let go_to_zero = [0.0, -1.0, 1e-20, f32::NAN, f32::NEG_INFINITY];
            for value in go_to_zero {
                assert_eq!(rgb9995f::f32(rgb9995f::from_f32([value, 0.0, 0.0]))[0], 0.0);
            }

            // all 9-bit values should be presented exactly
            for value in (0..=512).map(|v| v as f32) {
                let actual = rgb9995f::f32(rgb9995f::from_f32([value, 0.0, 0.0]));
                assert_eq!(actual, [value, 0.0, 0.0]);
            }

            // overflow when rounding should be handled correctly
            let actual = rgb9995f::f32(rgb9995f::from_f32([1023.0, 0.0, 0.0]));
            assert_eq!(actual, [1024.0, 0.0, 0.0]);

            // doesn't crash for any u16 values
            for value in (0..65556).map(|v| v as f32) {
                _ = rgb9995f::from_f32([value, 0.0, 1.0]);
            }
        }
    }
}

pub(crate) mod yuv8 {
    // https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering#converting-8-bit-yuv-to-rgb888

    pub fn n8(yuv: [u8; 3]) -> [u8; 3] {
        let [y, u, v] = yuv;

        let c = y as f32 - 16.0;
        let d = u as f32 - 128.0;
        let e = v as f32 - 128.0;

        let r = 1.164383 * c + 1.596027 * e;
        let g = 1.164383 * c - 0.391762 * d - 0.812968 * e;
        let b = 1.164383 * c + 2.017232 * d;

        let r = (r + 0.5) as u8;
        let g = (g + 0.5) as u8;
        let b = (b + 0.5) as u8;

        [r, g, b]
    }
    pub fn n16(yuv: [u8; 3]) -> [u16; 3] {
        f32(yuv).map(super::fp::n16)
    }
    pub fn f32(yuv: [u8; 3]) -> [f32; 3] {
        let [y, u, v] = yuv;

        let c = y as f32 - 16.0;
        let d = u as f32 - 128.0;
        let e = v as f32 - 128.0;

        let r = 1.164383 * c + 1.596027 * e;
        let g = 1.164383 * c - 0.391762 * d - 0.812968 * e;
        let b = 1.164383 * c + 2.017232 * d;

        const F: f32 = 1.0 / 255.0;
        let r = (r * F).clamp(0.0, 1.0);
        let g = (g * F).clamp(0.0, 1.0);
        let b = (b * F).clamp(0.0, 1.0);

        [r, g, b]
    }

    pub fn from_rgb_f32(rgb: [f32; 3]) -> [u8; 3] {
        let [r, g, b] = rgb.map(|c| c * 255.);

        let y = (0.256788 * r + 0.504129 * g + 0.097906 * b + (16. + 0.5)) as u8;
        let u = (-0.148223 * r - 0.290993 * g + 0.439216 * b + (128. + 0.5)) as u8;
        let v = (0.439216 * r - 0.367788 * g - 0.071427 * b + (128. + 0.5)) as u8;

        [y, u, v]
    }
}
pub(crate) mod yuv10 {
    // https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats

    pub fn n8(yuv: [u16; 3]) -> [u8; 3] {
        f32(yuv).map(super::fp::n8)
    }
    pub fn n16(yuv: [u16; 3]) -> [u16; 3] {
        f32(yuv).map(super::fp::n16)
    }
    pub fn f32(yuv: [u16; 3]) -> [f32; 3] {
        let [y, u, v] = yuv;

        let c = y as f32 - 64.0;
        let d = u as f32 - 512.0;
        let e = v as f32 - 512.0;

        let r = 1.164383 * c + 1.596027 * e;
        let g = 1.164383 * c - 0.391762 * d - 0.812968 * e;
        let b = 1.164383 * c + 2.017232 * d;

        const F: f32 = 1.0 / 1023.0;
        let r = (r * F).clamp(0.0, 1.0);
        let g = (g * F).clamp(0.0, 1.0);
        let b = (b * F).clamp(0.0, 1.0);

        [r, g, b]
    }

    pub fn from_rgb_f32(rgb: [f32; 3]) -> [u16; 3] {
        let [r, g, b] = rgb.map(|c| c * 1023.);

        let y = (0.256788 * r + 0.504129 * g + 0.097906 * b + (64. + 0.5)) as u16;
        let u = (-0.148223 * r - 0.290993 * g + 0.439216 * b + (512. + 0.5)) as u16;
        let v = (0.439216 * r - 0.367788 * g - 0.071427 * b + (512. + 0.5)) as u16;

        [y.min(1023), u.min(1023), v.min(1023)]
    }
}
pub(crate) mod yuv16 {
    // https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats

    pub fn n8(yuv: [u16; 3]) -> [u8; 3] {
        f32(yuv).map(super::fp::n8)
    }
    pub fn n16(yuv: [u16; 3]) -> [u16; 3] {
        f32(yuv).map(super::fp::n16)
    }
    pub fn f32(yuv: [u16; 3]) -> [f32; 3] {
        let [y, u, v] = yuv;

        let c = y as f32 - 4096.0;
        let d = u as f32 - 32768.0;
        let e = v as f32 - 32768.0;

        let r = 1.164383 * c + 1.596027 * e;
        let g = 1.164383 * c - 0.391762 * d - 0.812968 * e;
        let b = 1.164383 * c + 2.017232 * d;

        const F: f32 = 1.0 / 65535.0;
        let r = (r * F).clamp(0.0, 1.0);
        let g = (g * F).clamp(0.0, 1.0);
        let b = (b * F).clamp(0.0, 1.0);

        [r, g, b]
    }

    pub fn from_rgb_f32(rgb: [f32; 3]) -> [u16; 3] {
        let [r, g, b] = rgb.map(|c| c * 65535.);

        let y = (0.256788 * r + 0.504129 * g + 0.097906 * b + (4096. + 0.5)) as u16;
        let u = (-0.148223 * r - 0.290993 * g + 0.439216 * b + (32768. + 0.5)) as u16;
        let v = (0.439216 * r - 0.367788 * g - 0.071427 * b + (32768. + 0.5)) as u16;

        [y, u, v]
    }
}

pub(crate) trait ToRgba {
    type Channel;
    fn to_rgba(self) -> [Self::Channel; 4];
}
impl<T: Norm> ToRgba for [T; 3] {
    type Channel = T;

    #[inline(always)]
    fn to_rgba(self) -> [T; 4] {
        let [r, g, b] = self;
        [r, g, b, Norm::ONE]
    }
}
impl<T: Norm> ToRgba for [T; 1] {
    type Channel = T;

    #[inline(always)]
    fn to_rgba(self) -> [T; 4] {
        let [gray] = self;
        [gray, gray, gray, Norm::ONE]
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

pub(crate) trait NormConvert<To> {
    fn to(self) -> To;
}
impl<T> NormConvert<T> for T {
    #[inline(always)]
    fn to(self) -> T {
        self
    }
}
impl NormConvert<u16> for u8 {
    #[inline(always)]
    fn to(self) -> u16 {
        n8::n16(self)
    }
}
impl NormConvert<f32> for u8 {
    #[inline(always)]
    fn to(self) -> f32 {
        n8::f32(self)
    }
}
impl NormConvert<u8> for u16 {
    #[inline(always)]
    fn to(self) -> u8 {
        n16::n8(self)
    }
}
impl NormConvert<f32> for u16 {
    #[inline(always)]
    fn to(self) -> f32 {
        n16::f32(self)
    }
}
impl NormConvert<u8> for f32 {
    #[inline(always)]
    fn to(self) -> u8 {
        fp::n8(self)
    }
}
impl NormConvert<u16> for f32 {
    #[inline(always)]
    fn to(self) -> u16 {
        fp::n16(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

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

    #[test]
    fn fp_to_n8() {
        use super::fp;

        assert_eq!(fp::n8(f32::NEG_INFINITY), 0);
        assert_eq!(fp::n8(-1000.0), 0);
        assert_eq!(fp::n8(-1.0), 0);
        assert_eq!(fp::n8(0.0), 0);
        assert_eq!(fp::n8(0.5), 128);
        assert_eq!(fp::n8(1.0), 255);
        assert_eq!(fp::n8(1000.0), 255);
        assert_eq!(fp::n8(f32::INFINITY), 255);

        assert_eq!(fp::n8(f32::NAN), 0);
    }
    #[test]
    fn fp_to_n16() {
        use super::fp;

        assert_eq!(fp::n16(f32::NEG_INFINITY), 0);
        assert_eq!(fp::n16(-1000.0), 0);
        assert_eq!(fp::n16(-1.0), 0);
        assert_eq!(fp::n16(0.0), 0);
        assert_eq!(fp::n16(0.5), 32768);
        assert_eq!(fp::n16(1.0), u16::MAX);
        assert_eq!(fp::n16(1000.0), u16::MAX);
        assert_eq!(fp::n16(f32::INFINITY), u16::MAX);

        assert_eq!(fp::n16(f32::NAN), 0);
    }

    #[test]
    fn fp16_to_n8() {
        for i in 0..=u16::MAX {
            let expected = super::fp::n8(super::fp16::f32(i));
            let actual = super::fp16::n8(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }
    #[test]
    fn fp16_to_n16() {
        for i in 0..=u16::MAX {
            let expected = super::fp::n16(super::fp16::f32(i));
            let actual = super::fp16::n16(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }

    fn all_bc6h_uf16_values() -> impl Iterator<Item = u16> {
        (0..=u16::MAX).filter(|&x| {
            let value = super::fp16::f32(x);

            if value.is_sign_negative() {
                // BC6H_UF16 cannot negative values
                return false;
            }
            if !value.is_finite() {
                // BC6H_UF16 cannot produce +-Inf or NaN values.
                return false;
            }

            true
        })
    }
    #[test]
    fn bc6h_uf16_to_n8() {
        for i in all_bc6h_uf16_values() {
            let expected = super::fp16::n8(i);
            let actual = super::bc6h_uf16::n8(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }
    #[test]
    fn bc6h_uf16_to_n16() {
        for i in all_bc6h_uf16_values() {
            let expected = super::fp16::n16(i);
            let actual = super::bc6h_uf16::n16(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }
    #[test]
    fn bc6h_uf16_to_f32() {
        for i in all_bc6h_uf16_values() {
            let expected = super::fp16::f32(i);
            let actual = super::bc6h_uf16::f32(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }

    #[test]
    fn fp11_to_n8() {
        for i in 0..2048 {
            let expected = super::fp::n8(super::fp11::f32(i));
            let actual = super::fp11::n8(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }
    #[test]
    fn fp11_to_n16() {
        for i in 0..2048 {
            let expected = super::fp::n16(super::fp11::f32(i));
            let actual = super::fp11::n16(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }

    #[test]
    fn fp10_to_n8() {
        for i in 0..1024 {
            let expected = super::fp::n8(super::fp10::f32(i));
            let actual = super::fp10::n8(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }
    #[test]
    fn fp10_to_n16() {
        for i in 0..1024 {
            let expected = super::fp::n16(super::fp10::f32(i));
            let actual = super::fp10::n16(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }

    #[test]
    fn xr10_to_n8() {
        for i in 0..1024 {
            let expected = super::fp::n8(super::xr10::f32(i));
            let actual = super::xr10::n8(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }
    #[test]
    fn xr10_to_n16() {
        for i in 0..1024 {
            let expected = super::fp::n16(super::xr10::f32(i));
            let actual = super::xr10::n16(i);
            assert_eq!(actual, expected, "failed for i={}", i);
        }
    }

    #[test]
    fn rgb9995f_to_n8() {
        // This will exhaustively test all exponent and B mantissa values.
        // R and G won't be tested, since they behave the same as B.
        for e in 0..=31 {
            for b in 0..=255 {
                let i = e << 27 | b << 18;

                let expected = super::rgb9995f::f32(i).map(fp::n8);
                let actual = super::rgb9995f::n8(i);
                assert_eq!(actual, expected, "failed for exp={} mant={}", e, b);
            }
        }
    }
    #[test]
    fn rgb9995f_to_n16() {
        // This will exhaustively test all exponent and B mantissa values.
        // R and G won't be tested, since they behave the same as B.
        for e in 0..=31 {
            for b in 0..=255 {
                let i = e << 27 | b << 18;

                let expected = super::rgb9995f::f32(i).map(fp::n16);
                let actual = super::rgb9995f::n16(i);
                assert_eq!(actual, expected, "failed for exp={} mant={}", e, b);
            }
        }
    }
}
