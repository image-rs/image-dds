use crate::cast;

pub(crate) fn read_u32_le_array<const N: usize>(
    reader: &mut impl std::io::Read,
) -> std::io::Result<[u32; N]> {
    let mut buffer = [0; N];
    reader.read_exact(cast::as_bytes_mut(&mut buffer))?;
    for i in buffer.iter_mut() {
        *i = u32::from_le(*i);
    }
    Ok(buffer)
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

/// Computes `(a as f64 / b as f64).round() as T`.
///
/// Results are NOT correct if `a + b/2` overflows `T`.
#[inline(always)]
pub(crate) fn div_round_fast<T>(a: T, b: T) -> T
where
    T: Copy + From<u8> + std::ops::Add<T, Output = T> + std::ops::Div<T, Output = T> + Unsigned,
{
    (a + b / 2.into()) / b
}

pub(crate) trait Unsigned {}
impl Unsigned for u8 {}
impl Unsigned for u16 {}
impl Unsigned for u32 {}
impl Unsigned for u64 {}
impl Unsigned for usize {}

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
    fn div_round_fast() {
        for a in 0..32 {
            for b in 1..32 {
                let expected = (a as f64 / b as f64).round() as u8;
                assert_eq!(super::div_round_fast(a, b), expected, "a={}, b={}", a, b);
            }
        }
    }
}
