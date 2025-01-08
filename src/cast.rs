//! An internal module for casting between types.
//!
//! This serves as a wrapper around `bytemuck` to provide panic safety. All
//! functions in this module are guaranteed to be safe and **NEVER** panic.

pub(crate) trait NonZeroSized {}
impl NonZeroSized for u8 {}
impl NonZeroSized for u16 {}
impl NonZeroSized for u32 {}
impl NonZeroSized for u64 {}
impl NonZeroSized for f32 {}
impl<const N: usize, T: NonZeroSized> NonZeroSized for [T; N] {}

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
