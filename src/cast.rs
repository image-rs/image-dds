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
