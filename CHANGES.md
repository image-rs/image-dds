# Changes

## Version 0.2.0 (2025-11-10)

Breaking:

- Mipmap generation in `Encoder` is now controlled via the `mipmaps: MipmapOptions` field instead of a `WriteOptions` parameter. The parameter on `Encoder::write_surface_with` and `WriteOptions` have been removed.
- Renamed `Encoder::options` to `Encoder::encoding`.
- `decode_rect` and `Decoder::read_surface_rect` now take an `ImageViewMut` instead of a bare `&mut [u8]` + row pitch.
- Removed `DecodeError::RowPitchTooSmall` and `DecodeError::RectBufferTooSmall` errors.
- `ImageView` and `ImageViewMut` now support non-contiguous views. Consumers of those types need to adapt too.
- Renamed `SplitSurface` to `SplitView`. `SplitView` also lazily computes fragments on the fly instead of precomputing them. ([#69](https://github.com/image-rs/image-dds/pull/69))
- `Progress` tokens now support cancellation via the new `CancellationToken`s. ([#83](https://github.com/image-rs/image-dds/pull/83))

Added:

- Added BC7 encoding. ([#86](https://github.com/image-rs/image-dds/pull/86))
- Added `DataLayout::is_{texture,texture_array,volume}()` methods to check the kind of resource a layout holds.
- Added `new_with`, `cropped`, and `is_contiguous` methods to `ImageView{,Mut}` for handling and using non-contiguous views.
- Added `DecodeError::UnexpectedRectSize` error for when the decoded rectangle size does not match the expected size.
- `R1_UNROM` now supports dithering. ([#63](https://github.com/image-rs/image-dds/pull/63))
- All uncompressed formats (expect float32 formats) now support dithering. ([#63](https://github.com/image-rs/image-dds/pull/63))
- Fuzz untrusted inputs. ([#60](https://github.com/image-rs/image-dds/pull/60))
- Faster and higher quality BC1-5 encoding. ([#72](https://github.com/image-rs/image-dds/pull/72), [#73](https://github.com/image-rs/image-dds/pull/73), [#75](https://github.com/image-rs/image-dds/pull/75), [#76](https://github.com/image-rs/image-dds/pull/76), [#78](https://github.com/image-rs/image-dds/pull/78), [#80](https://github.com/image-rs/image-dds/pull/80), [#84](https://github.com/image-rs/image-dds/pull/84), [#90](https://github.com/image-rs/image-dds/pull/90))

Fixed:

- Minimum supported version of `resize` now compiles with MSRV.
- Handle allocation failures and fixed numeric overflow in decoders. ([#68](https://github.com/image-rs/image-dds/pull/68))

## Version 0.1.0 (2025-04-30)

Initial release of the project.
