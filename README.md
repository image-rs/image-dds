# DDS De/Encoder

[![crates.io](https://img.shields.io/crates/v/dds.svg)](https://crates.io/crates/dds)
[![Documentation](https://docs.rs/image/badge.svg)](https://docs.rs/dds)
[![Build Status](https://github.com/image-rs/image-dds/workflows/Rust%20CI/badge.svg)](https://github.com/image-rs/image-dds/actions)

A DDS decoder and encoder written in 100% safe Rust.

## Features

- Supports [over 70 formats](https://docs.rs/dds/latest/dds/enum.Format.html) for decoding and most for encoding. See below for details.
- Both high-level and low-level APIs for decoding and encoding.
- Automatic multi-threading with rayon.
- Easy automatic mipmap generation.

## Usage

See [crate documentation](https://docs.rs/dds/).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
