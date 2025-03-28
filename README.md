# DDS De/Encoder

A DDS decoder and encoder focused on speed and correctness written in 100% safe Rust.

## Supported formats

This library supports a total of over 70 formats for decoding, including:

- All BCn/DXT formats. E.g. `BC1_UNORM`, `BC2_UNORM`, `BC3_UNORM`, `BC7_UNORM`.
- All LDR ASTC formats. E.g. `ASTC_6x6_UNORM`.
- Over 30 uncompressed formats. E.g. `R8G8B8A8_UNORM`, `R9G9B9E5_SHAREDEXP`, `R32G32B32_FLOAT`.
- Many YUV formats. E.g. `AYUV`, `Y416`, `YUY2`, `NV12`.

Most formats support encoding. Notable exceptions are the ASTC formats ([#23](https://github.com/RunDevelopment/dds/issues/23)), BC7 ([#20](https://github.com/RunDevelopment/dds/issues/20)), and BC6 (currently not planned).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
