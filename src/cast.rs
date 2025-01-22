//! An internal module for casting between types.
//!
//! This serves as a thin wrapper around `zerocopy` to simplify usage and
//! provide panic safety. All methods in this module are guaranteed to
//! NEVER PANIC.

use zerocopy::{FromBytes, Immutable, IntoBytes, Ref};

pub(crate) trait Castable: FromBytes + IntoBytes + Immutable {}
impl<T: FromBytes + IntoBytes + Immutable> Castable for T {}

pub(crate) fn from_bytes<T: Castable>(bytes: &[u8]) -> Option<&[T]> {
    Ref::from_bytes(bytes).ok().map(Ref::into_ref)
}
pub(crate) fn from_bytes_mut<T: Castable>(bytes: &mut [u8]) -> Option<&mut [T]> {
    Ref::from_bytes(bytes).ok().map(Ref::into_mut)
}

/// Casts a slice of `T` to a slice of `u8`.
pub(crate) fn as_bytes_mut<T: Castable>(buffer: &mut [T]) -> &mut [u8] {
    buffer.as_mut_bytes()
}
/// Casts a slice of `T` to a slice of `u8`.
pub(crate) fn as_bytes<T: Castable>(buffer: &[T]) -> &[u8] {
    buffer.as_bytes()
}

/// Returns the native endian bytes of a value.
pub(crate) trait IntoNeBytes {
    type Bytes: Castable + Copy + Default;
    fn into_ne_bytes(self) -> Self::Bytes;
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

macro_rules! u8_array_to_ne_bytes {
    ($($n:literal),*) => {
        $(
            impl IntoNeBytes for [u8; $n] {
                type Bytes = [u8; $n];
                #[inline(always)]
                fn into_ne_bytes(self) -> Self::Bytes {
                    self
                }
            }
            impl FromLeBytes for [u8; $n] {
                #[inline(always)]
                fn from_le_bytes(bytes: Self::Bytes) -> Self {
                    bytes
                }
            }
        )*
    };
}
u8_array_to_ne_bytes!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

macro_rules! transmute_array {
    ($([$t:ty; $n:literal]),*) => {
        $(
            impl IntoNeBytes for [$t; $n] {
                type Bytes = [u8; std::mem::size_of::<Self>()];
                #[inline(always)]
                fn into_ne_bytes(self) -> Self::Bytes {
                    zerocopy::transmute!(self)
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
