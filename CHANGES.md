# Changes

## Version 0.2.0 [Unreleased]

Breaking:

- Mipmap generation in `Encoder` is now controlled via the `mipmaps: MipmapOptions` field instead of a `WriteOptions` parameter. The parameter on `Encoder::write_surface_with` and `WriteOptions` have been removed.
- Renamed `Encoder::options` to `Encoder::encoding`.
- `decode_rect` and `Decoder::read_surface_rect` now take an `ImageViewMut` instead of a bare `&mut [u8]` + row pitch.
- Removed `DecodeError::RowPitchTooSmall` and `DecodeError::RectBufferTooSmall` errors.
- `ImageView` and `ImageViewMut` now support non-contiguous views. Consumers of those types need to adapt to.

Added:

- Added `DataLayout::is_{texture,texture_array,volume}()` methods to check the type of layout.
- Added `new_with`, `cropped`, and `is_contiguous` methods to `ImageView{,Mut}` for handling and using non-contiguous views.
- Added `DecodeError::UnexpectedRectSize` error for when the decoded rectangle size does not match the expected size.
- `R1_UNROM` now supports dithering. ([#63](https://github.com/image-rs/image-dds/pull/63))
- Fuzz untrusted inputs. ([#60](https://github.com/image-rs/image-dds/pull/60))

Fixed:

- Minimum supported version of `resize` now compiles with MSRV.

## Version 0.1.0 (2025-04-30)

Initial release of the project.
