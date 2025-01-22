//! An internal module for casting between types.
//!
//! This serves as a wrapper around `bytemuck` to provide panic safety. All
//! functions in this module are guaranteed to be safe and **NEVER** panic.

/// Marks a type as non-zero sized, meaning `size_of::<T>() > 0`.
pub(crate) trait NonZeroSized {}
impl NonZeroSized for u8 {}
impl NonZeroSized for u16 {}
impl NonZeroSized for u32 {}
impl NonZeroSized for u64 {}
impl NonZeroSized for f32 {}
impl NonZeroSized for f64 {}
impl<T: NonZeroSized> NonZeroSized for [T; 1] {}
impl<T: NonZeroSized> NonZeroSized for [T; 2] {}
impl<T: NonZeroSized> NonZeroSized for [T; 3] {}
impl<T: NonZeroSized> NonZeroSized for [T; 4] {}
impl<T: NonZeroSized> NonZeroSized for [T; 5] {}
impl<T: NonZeroSized> NonZeroSized for [T; 6] {}
impl<T: NonZeroSized> NonZeroSized for [T; 7] {}
impl<T: NonZeroSized> NonZeroSized for [T; 8] {}
impl<T: NonZeroSized> NonZeroSized for [T; 9] {}
impl<T: NonZeroSized> NonZeroSized for [T; 10] {}
impl<T: NonZeroSized> NonZeroSized for [T; 11] {}
impl<T: NonZeroSized> NonZeroSized for [T; 12] {}
impl<T: NonZeroSized> NonZeroSized for [T; 13] {}
impl<T: NonZeroSized> NonZeroSized for [T; 14] {}
impl<T: NonZeroSized> NonZeroSized for [T; 15] {}
impl<T: NonZeroSized> NonZeroSized for [T; 16] {}
impl<T: NonZeroSized> NonZeroSized for [T; 17] {}
impl<T: NonZeroSized> NonZeroSized for [T; 18] {}
impl<T: NonZeroSized> NonZeroSized for [T; 19] {}
impl<T: NonZeroSized> NonZeroSized for [T; 20] {}
impl<T: NonZeroSized> NonZeroSized for [T; 21] {}
impl<T: NonZeroSized> NonZeroSized for [T; 22] {}
impl<T: NonZeroSized> NonZeroSized for [T; 23] {}
impl<T: NonZeroSized> NonZeroSized for [T; 24] {}
impl<T: NonZeroSized> NonZeroSized for [T; 25] {}
impl<T: NonZeroSized> NonZeroSized for [T; 26] {}
impl<T: NonZeroSized> NonZeroSized for [T; 27] {}
impl<T: NonZeroSized> NonZeroSized for [T; 28] {}
impl<T: NonZeroSized> NonZeroSized for [T; 29] {}
impl<T: NonZeroSized> NonZeroSized for [T; 30] {}
impl<T: NonZeroSized> NonZeroSized for [T; 31] {}
impl<T: NonZeroSized> NonZeroSized for [T; 32] {}

pub(crate) trait Castable: bytemuck::Pod + NonZeroSized {}
impl<T: bytemuck::Pod + NonZeroSized> Castable for T {}

/// Casts a slice of `T` to a slice of `u8`.
pub(crate) fn as_bytes_mut<T: Castable>(buffer: &mut [T]) -> &mut [u8] {
    bytemuck::cast_slice_mut(buffer)
}

/// Casts a slice of `T` to a slice of `u8`.
pub(crate) fn as_bytes<T: Castable>(buffer: &[T]) -> &[u8] {
    bytemuck::cast_slice(buffer)
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
