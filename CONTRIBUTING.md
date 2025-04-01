# Contributing

Maintainer: [@RunDevelopment](https://github.com/RunDevelopment)

Thank you so much for considering contributing to this project! We appreciate your help and support ❤️

For general information about contributing and the [code of conduct](https://github.com/image-rs/organization/blob/master/CODE_OF_CONDUCT.md), see the [image-rs contributing guide](https://www.rust-lang.org/policies/conduct.html). Below is everything specific to this repository.

## Testing

```bash
cargo test
```

This will run all tests and automatically update outdated snapshots.

If there are multiple outdated snapshots, `cargo test` might only update a few of them. You can fix this by running the command multiple times or using `cargo test --no-fail-fast`.

### Debugging tests

All tests are compiled with some optimizations enabled. This speeds up running tests from 1 minute to 5 seconds. However, it also prevents breakpoints from being hit in tests.

Comment out `opt-level = 1` under `[profile.test]` in `Cargo.toml` to disable optimizations for tests. This will make breakpoints work again.

### Snapshots and input data

This project heavily uses snapshots for testing. Both the snapshot files and input data files are located in the `test-data` directory.

- `test-data/images/`: A directory of DDS images used for testing.

  They are mainly used in two ways: 1) their headers are used to test the header parser, and 2) their data is used to test the decoders. Each image format has at least one image in this directory.

- `test-data/output/`: Contains the decoded images from `test-data/images/`. Decoded images are stored as 8-bit PNGs and hashes in `_hashes.yml` files.

  Note that some PNGs are git-ignored to avoid bloating the repository. Such images are only checked by hash.

- `test-data/output-rect/`: Contains the snapshots for rectangle decoding.

  Decoding whole images and rectangles uses different code paths where rectangle decoding is a lot more complex. As such, rectangle decoding is tested separately.

- `test-data/samples/`: Contains sample PNG images for encoding.

- `test-data/output-encode/`: Contains various DDS files created by the encoder.

There are also several text snapshot files in `test-data`, mostly containing headers and the results of various operations on them.

Lastly, there is `supported-formats.md`. This file contains a list of all supported formats and their properties. It doubles as a test for the metadata of all formats and as documentation for users.

## Benchmarks

There are 2 main benchmarks: one for decoding and one for encoding. The benchmarks' source code is located in the `benches` directory.

You can run them with the following commands:

- `cargo bench --bench decode`
- `cargo bench --bench encode`

## Linting and formatting

We use `clippy` and `rustfmt` for linting and formatting. You can run them with the following commands:

- `cargo clippy`
- `cargo fmt`

Both are checked by CI, so make sure to run them before submitting a PR.

## Coverage

If you want to check the code coverage of your changes, use [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov). [Install the tool](https://github.com/taiki-e/cargo-llvm-cov?tab=readme-ov-file#installation) and then run it with:

```bash
cargo llvm-cov --open
```

This will run all tests to collect coverage data and open a browser window with the coverage report.
