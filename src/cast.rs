//! An internal module for casting between types.
//!
//! This serves as a thin wrapper around `zerocopy` to simplify usage and
//! provide panic safety. All methods in this module are guaranteed to
//! NEVER PANIC.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::Precision;

pub(crate) trait Castable: FromBytes + IntoBytes + Immutable {}
impl<T: FromBytes + IntoBytes + Immutable> Castable for T {}

pub(crate) fn from_bytes<T: Castable>(bytes: &[u8]) -> Option<&[T]> {
    FromBytes::ref_from_bytes(bytes).ok()
}
pub(crate) fn from_bytes_mut<T: Castable>(bytes: &mut [u8]) -> Option<&mut [T]> {
    FromBytes::mut_from_bytes(bytes).ok()
}

/// Casts a slice of `T` to a slice of `u8`.
pub(crate) fn as_bytes<T: Castable>(buffer: &[T]) -> &[u8] {
    buffer.as_bytes()
}
/// Casts a slice of `T` to a slice of `u8`.
pub(crate) fn as_bytes_mut<T: Castable>(buffer: &mut [T]) -> &mut [u8] {
    buffer.as_mut_bytes()
}

/// An implementation of `slice::as_flattened` for more Rust versions.
#[allow(unused)]
pub(crate) fn as_flattened<const N: usize, T>(buffer: &[[T; N]]) -> &[T]
where
    T: Castable,
    [T; N]: Castable,
{
    // PANIC SAFETY: This unwrap can never fail, because T isn't a ZST.
    from_bytes(as_bytes(buffer)).unwrap()
}
/// An implementation of `slice::as_flattened_mut` for more Rust versions.
pub(crate) fn as_flattened_mut<const N: usize, T>(buffer: &mut [[T; N]]) -> &mut [T]
where
    T: Castable,
    [T; N]: Castable,
{
    // PANIC SAFETY: This unwrap can never fail, because T isn't a ZST.
    from_bytes_mut(as_bytes_mut(buffer)).unwrap()
}

/// An implementation of something similar to `slice::array_chunks`.
pub(crate) fn as_array_chunks<const N: usize, T>(buffer: &[T]) -> Option<&[[T; N]]>
where
    T: Castable,
    [T; N]: Castable,
{
    from_bytes(as_bytes(buffer))
}
/// An implementation of something similar to `slice::array_chunks_mut`.
pub(crate) fn as_array_chunks_mut<const N: usize, T>(buffer: &mut [T]) -> Option<&mut [[T; N]]>
where
    T: Castable,
    [T; N]: Castable,
{
    from_bytes_mut(as_bytes_mut(buffer))
}

#[allow(unused)]
pub(crate) trait IsByteArray {}
impl<const N: usize> IsByteArray for [u8; N] {}
pub(crate) trait NonEmpty {}
macro_rules! non_empty {
    ($($n:literal),*) => {
        $(
            impl<T> NonEmpty for [T; $n] {}
        )*
    };
}
non_empty!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

/// Returns the native endian bytes of a value.
pub(crate) trait IntoNeBytes {
    type Bytes: IsByteArray + NonEmpty + Castable + Copy + Default;
    fn into_ne_bytes(self) -> Self::Bytes;
    fn from_ne_bytes(bytes: Self::Bytes) -> Self;
}

/// Creates a value from little-endian bytes.
pub(crate) trait FromLeBytes: IntoNeBytes {
    fn from_le_bytes(bytes: Self::Bytes) -> Self;
}

macro_rules! to_ne_bytes {
    ($($t:ty),*) => {
        $(
            impl IntoNeBytes for $t {
                type Bytes = [u8; std::mem::size_of::<Self>()];
                #[inline(always)]
                fn into_ne_bytes(self) -> Self::Bytes {
                    Self::to_ne_bytes(self)
                }
                #[inline(always)]
                fn from_ne_bytes(bytes: Self::Bytes) -> Self {
                    Self::from_ne_bytes(bytes)
                }
            }
            impl FromLeBytes for $t {
                #[inline(always)]
                fn from_le_bytes(bytes: Self::Bytes) -> Self {
                    Self::from_le_bytes(bytes)
                }
            }
        )*
    };
}
to_ne_bytes!(u8, u16, u32, f32);

impl<const N: usize> IntoNeBytes for [u8; N]
where
    [u8; N]: NonEmpty + Default,
{
    type Bytes = [u8; N];
    #[inline(always)]
    fn into_ne_bytes(self) -> Self::Bytes {
        self
    }
    #[inline(always)]
    fn from_ne_bytes(bytes: Self::Bytes) -> Self {
        bytes
    }
}
impl<const N: usize> FromLeBytes for [u8; N]
where
    [u8; N]: IntoNeBytes<Bytes = Self>,
{
    #[inline(always)]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        bytes
    }
}

macro_rules! transmute_array {
    ($([$t:ty; $n:literal]),*) => {
        $(
            impl IntoNeBytes for [$t; $n] {
                type Bytes = [u8; std::mem::size_of::<Self>()];
                #[inline(always)]
                fn into_ne_bytes(self) -> Self::Bytes {
                    zerocopy::transmute!(self)
                }
                #[inline(always)]
                fn from_ne_bytes(bytes: Self::Bytes) -> Self {
                    zerocopy::transmute!(bytes)
                }
            }
            impl FromLeBytes for [$t; $n] {
                #[inline(always)]
                fn from_le_bytes(bytes: Self::Bytes) -> Self {
                    const SIZE: usize = std::mem::size_of::<$t>();
                    let grouped: [[u8; SIZE]; $n] = zerocopy::transmute!(bytes);
                    grouped.map(|bytes| FromLeBytes::from_le_bytes(bytes))
                }
            }
        )*
    };
}
transmute_array!(
    [u16; 1], [u16; 2], [u16; 3], [u16; 4], [u32; 1], [u32; 2], [u32; 3], [u32; 4], [f32; 1],
    [f32; 2], [f32; 3], [f32; 4]
);

pub(crate) fn slice_le_to_ne_16(buf: &mut [u8]) {
    assert!(buf.len() % 2 == 0);

    if cfg!(target_endian = "big") {
        if let Some(aligned) = from_bytes_mut::<u16>(buf) {
            aligned.iter_mut().for_each(|x| *x = x.swap_bytes());
        } else {
            // we already checked the length, so this is safe to unwrap
            let unaligned: &mut [[u8; 2]] = as_array_chunks_mut(buf).unwrap();
            unaligned.iter_mut().for_each(|x| *x = [x[1], x[0]]);
        }
    }
}
pub(crate) fn slice_ne_to_le_16(buf: &mut [u8]) {
    slice_le_to_ne_16(buf);
}
pub(crate) fn slice_le_to_ne_32(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);

    if cfg!(target_endian = "big") {
        if let Some(aligned) = from_bytes_mut::<u32>(buf) {
            aligned.iter_mut().for_each(|x| *x = x.swap_bytes());
        } else {
            // we already checked the length, so this is safe to unwrap
            let unaligned: &mut [[u8; 4]] = as_array_chunks_mut(buf).unwrap();
            unaligned
                .iter_mut()
                .for_each(|x| *x = [x[3], x[2], x[1], x[0]]);
        }
    }
}
pub(crate) fn slice_ne_to_le_32(buf: &mut [u8]) {
    slice_le_to_ne_32(buf);
}

pub(crate) fn slice_ne_to_le(precision: Precision, buffer: &mut [u8]) {
    match precision {
        Precision::U8 => {}
        Precision::U16 => slice_ne_to_le_16(buffer),
        Precision::F32 => slice_ne_to_le_32(buffer),
    }
}

pub(crate) trait ToLe: Sized {
    fn to_le(buffer: &mut [Self]);
}
impl ToLe for u8 {
    fn to_le(_buffer: &mut [Self]) {}
}
impl ToLe for u16 {
    fn to_le(buffer: &mut [Self]) {
        slice_ne_to_le_16(as_bytes_mut(buffer));
    }
}
impl ToLe for u32 {
    fn to_le(buffer: &mut [Self]) {
        slice_ne_to_le_32(as_bytes_mut(buffer));
    }
}
impl ToLe for f32 {
    fn to_le(buffer: &mut [Self]) {
        slice_ne_to_le_32(as_bytes_mut(buffer));
    }
}
impl<const N: usize, T> ToLe for [T; N]
where
    T: ToLe + Castable,
{
    fn to_le(buffer: &mut [Self]) {
        let flat = as_flattened_mut(buffer);
        T::to_le(flat);
    }
}
