pub(crate) fn read_u32_le_array<const N: usize>(
    reader: &mut impl std::io::Read,
) -> std::io::Result<[u32; N]> {
    let mut buffer = [0; N];
    reader.read_exact(bytemuck::cast_slice_mut(buffer.as_mut_slice()))?;
    for i in buffer.iter_mut() {
        *i = u32::from_le(*i);
    }
    Ok(buffer)
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
        + std::ops::Add<Output = T>,
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
