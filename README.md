# DDS De/Encoder

[![crates.io](https://img.shields.io/crates/v/dds.svg)](https://crates.io/crates/dds)
[![Documentation](https://docs.rs/image/badge.svg)](https://docs.rs/dds)
[![Build Status](https://github.com/image-rs/image-dds/workflows/Rust%20CI/badge.svg)](https://github.com/image-rs/image-dds/actions)

A DDS decoder and encoder written in 100% safe Rust.

## Features

- Supports over 70 formats for decoding and most for encoding. See below for details.
- Both high-level and low-level APIs for decoding and encoding.
- Automatic multi-threading with rayon.
- Simple mipmap generation.

## Usage

See [crate documentation](https://docs.rs/dds/).

## Supported formats

This library supports a total of over 70 formats for decoding, including:

- All BCn/DXT formats. E.g. `BC1_UNORM`, `BC2_UNORM`, `BC3_UNORM`, `BC7_UNORM`.
- All LDR ASTC formats. E.g. `ASTC_6x6_UNORM`.
- Over 30 uncompressed formats. E.g. `R8G8B8A8_UNORM`, `R9G9B9E5_SHAREDEXP`, `R32G32B32_FLOAT`.
- Many YUV formats. E.g. `AYUV`, `Y416`, `YUY2`, `NV12`.

Most formats support encoding. Notable exceptions are the ASTC formats ([#23](https://github.com/image-rs/image-dds/issues/23)) and BC6 (currently not planned).

For a full list of all support formats and their capabilities, see [this document](./supported-formats.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
