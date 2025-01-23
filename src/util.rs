use crate::cast;

pub(crate) fn read_u32_le_array(
    reader: &mut impl std::io::Read,
    buffer: &mut [u32],
) -> std::io::Result<()> {
    reader.read_exact(cast::as_bytes_mut(buffer))?;
    for i in buffer.iter_mut() {
        *i = u32::from_le(*i);
    }
    Ok(())
}

pub(crate) fn le_to_native_endian_16(buf: &mut [u8]) {
    assert!(buf.len() % 2 == 0);

    if cfg!(target_endian = "big") {
        // TODO: optimize this for when the buffer is aligned to u16/u32/u64
        for i in (0..buf.len()).step_by(2) {
            buf.swap(i, i + 1);
        }
    }
}
pub(crate) fn le_to_native_endian_32(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);

    if cfg!(target_endian = "big") {
        // TODO: optimize this for when the buffer is aligned to u32/u64
        for i in (0..buf.len()).step_by(4) {
            buf.swap(i, i + 3);
            buf.swap(i + 1, i + 2);
        }
    }
}

/// An implementation of div_ceil to lower MSRV.
pub(crate) fn div_ceil<T>(a: T, b: T) -> T
where
    T: Copy
        + PartialEq
        + PartialOrd
        + From<u8>
        + std::ops::Div<Output = T>
        + std::ops::Rem<Output = T>
        + std::ops::Add<Output = T>
        + Unsigned,
{
    assert!(a >= T::from(0));
    assert!(b > T::from(0));

    let d = a / b;
    if a % b != T::from(0) {
        d + T::from(1)
    } else {
        d
    }
}

pub(crate) trait Unsigned {}
impl Unsigned for u8 {}
impl Unsigned for u16 {}
impl Unsigned for u32 {}
impl Unsigned for u64 {}
impl Unsigned for usize {}

/// Computes `2^exponent` as a float.
#[inline(always)]
pub(crate) fn two_powi(exponent: i8) -> f32 {
    // Ensure the exponent is within the range for f32
    // Exponent range for f32: -126 to 127 (since 2^127 is the max positive finite power of 2)
    debug_assert!(-126 <= exponent, "Exponent out of range for f32");

    let bits = (((exponent as i32) + 127) as u32) << 23;
    f32::from_bits(bits)
}

/// This is a hack to explicitly annotate the types of the closures.
pub(crate) fn closure_types<A, B, F: Fn(A) -> B>(f: F) -> F {
    f
}

/// This can be used to hint to the compiler that a branch is unlikely to be taken.
#[cold]
#[inline]
pub(crate) fn unlikely_branch() {}

#[cfg(test)]
mod test {
    #[test]
    fn div_ceil() {
        for a in 0..255 {
            for b in 1..255 {
                let expected = (a as f64 / b as f64).ceil() as u8;
                assert_eq!(super::div_ceil(a, b), expected, "a={}, b={}", a, b);
            }
        }
    }
    #[test]
    fn two_powi() {
        for i in -126..=127 {
            let expected = 2.0f32.powi(i as i32);
            let actual = super::two_powi(i);
            assert_eq!(actual, expected, "i={}", i);
        }
    }
}
