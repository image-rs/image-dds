use std::path::{Path, PathBuf};

use dds::{header::*, *};
use rand::{Rng, RngCore};
use util::{test_data_dir, Image, WithPrecision};

mod util;

fn get_sample(name: &str) -> PathBuf {
    util::test_data_dir().join("samples").join(name)
}

fn encode_image<T: WithPrecision + util::Castable, W: std::io::Write>(
    image: &Image<T>,
    format: Format,
    writer: &mut W,
    options: &EncodeOptions,
) -> Result<(), EncodingError> {
    encode(writer, image.view(), format, None, options)
}
fn write_image<T: WithPrecision + util::Castable, W: std::io::Write>(
    encoder: &mut Encoder<W>,
    image: &Image<T>,
) -> Result<(), EncodingError> {
    encoder.write_surface(image.view())
}
fn encode_decode(
    format: Format,
    options: &EncodeOptions,
    image: &Image<f32>,
) -> (Vec<u8>, Image<f32>) {
    // encode
    let mut encoded = Vec::new();
    let mut encoder = Encoder::new(
        &mut encoded,
        format,
        &Header::new_image(image.size.width, image.size.height, format),
    )
    .unwrap();
    encoder.options = options.clone();
    encoder.write_surface(image.view()).unwrap();
    encoder.finish().unwrap();

    // decode
    let mut decoder = Decoder::new(encoded.as_slice()).unwrap();
    let mut decoded = Image::new_empty(image.channels, image.size);
    decoder.read_surface(decoded.view_mut()).unwrap();

    (encoded, decoded)
}
fn create_random_color_blocks() -> Image<f32> {
    let mut rng = util::create_rng();

    let width = 256;
    let height = 256;
    let mut image = Image::new_empty(Channels::Rgb, Size::new(width as u32, height as u32));
    let block_stride = 4 * 3;
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let rgb: [f32; 3] = rng.gen();
            let block_line = [rgb; 4];
            let line_flat: &[f32] = util::cast_slice(&block_line);
            for j in 0..4 {
                let i = ((y + j) * width + x) * 3;
                image.data[i..i + block_stride].copy_from_slice(line_flat);
            }
        }
    }

    image
}

#[test]
fn encode_base() {
    let base_u8 = util::read_png_u8(&get_sample("base.png")).unwrap();
    assert!(base_u8.channels == Channels::Rgba);
    let base_u16 = base_u8.to_u16();
    let base_f32 = base_u8.to_f32();

    fn get_output_path(format: Format) -> PathBuf {
        let name = format!("{:?}.dds", format);
        test_data_dir().join("output-encode/base").join(&name)
    }
    let test = |format: Format, dds_path: &Path| -> Result<String, Box<dyn std::error::Error>> {
        let mut size = base_u8.size;
        if let Some(support) = format.encoding_support() {
            if let Some(size_multiple) = support.size_multiple() {
                // round down to the nearest multiple
                let w_mul = size_multiple.width;
                let h_mul = size_multiple.height;
                size = Size::new(
                    (size.width / w_mul) * w_mul,
                    (size.height / h_mul) * h_mul,
                );
            }
        };

        let mut output = Vec::new();
        let mut encoder = Encoder::new(
            &mut output,
            format,
            &Header::new_image(size.width, size.height, format),
        )?;
        encoder.options.quality = CompressionQuality::High;

        // and now the image data
        if format.precision() == Precision::U16 {
            write_image(&mut encoder, &base_u16.cropped(size))?;
        } else if format.precision() == Precision::F32 {
            write_image(&mut encoder, &base_f32.cropped(size))?;
        } else {
            write_image(&mut encoder, &base_u8.cropped(size))?;
        }
        encoder.finish()?;

        // write to disk
        std::fs::create_dir_all(dds_path.parent().unwrap())?;
        std::fs::write(dds_path, &output)?;

        let hex = util::hash_hex(&output);
        Ok(hex)
    };

    let mut summaries = util::OutputSummaries::new("_hashes");

    for format in util::ALL_FORMATS.iter().copied() {
        let dds_path = get_output_path(format);
        summaries.add_output_file_result(&dds_path, test(format, &dds_path));
    }

    summaries.snapshot_or_fail();
}

#[test]
fn encode_dither() {
    fn get_output_dds(format: Format, name: &str) -> PathBuf {
        let name = format!("{:?} {}.dds", format, name);
        test_data_dir().join("output-encode/dither").join(&name)
    }
    fn test(
        format: Format,
        image: &Image<f32>,
        dds_path: &Path,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut output = Vec::new();
        let mut encoder = Encoder::new(
            &mut output,
            format,
            &Header::new_image(image.size.width, image.size.height, format),
        )?;
        encoder.options.quality = CompressionQuality::High;
        encoder.options.dithering = Dithering::ColorAndAlpha;
        encoder.write_surface(image.view())?;
        encoder.finish()?;

        // write to disk
        std::fs::create_dir_all(dds_path.parent().unwrap())?;
        std::fs::write(dds_path, &output)?;

        let hex = util::hash_hex(&output);
        Ok(hex)
    }

    let base = util::read_png_u8(&get_sample("base.png")).unwrap().to_f32();
    let twirl = util::read_png_u8(&get_sample("color-twirl.png"))
        .unwrap()
        .to_f32();

    let ignore = [Format::BC4_SNORM, Format::BC5_UNORM, Format::BC5_SNORM];

    let mut summaries = util::OutputSummaries::new("_hashes");

    for (format, encoding) in util::ALL_FORMATS
        .iter()
        .copied()
        .filter(|f| !ignore.contains(f))
        .filter_map(|f| f.encoding_support().map(|e| (f, e)))
        .filter(|(_, e)| e.dithering() != Dithering::None)
    {
        let mut test_and_summarize = |image, name| {
            let output_path = get_output_dds(format, name);
            summaries.add_output_file_result(&output_path, test(format, image, &output_path));
        };

        test_and_summarize(&base, "base");

        if encoding.dithering().color() {
            test_and_summarize(&twirl, "twirl");
        }
    }

    summaries.snapshot_or_fail()
}

// Don't run this on big endian targets, it's just too slow
#[cfg(not(target_endian = "big"))]
// Don't run when doing code coverage, it's just too slow
#[cfg(not(coverage))]
#[test]
fn encode_measure_quality() {
    let base = &TestImage::from_file("base.png");
    let color_twirl = &TestImage::from_file("color-twirl.png");
    let bricks_d = &TestImage::from_file("bricks-d.png");
    let bricks_n = &TestImage::from_file("bricks-n.png");
    let clovers_d = &TestImage::from_file("clovers-d.png");
    let clovers_r = &TestImage::from_file("clovers-r.png");
    let stone_d = &TestImage::from_file("stone-d.png");
    let stone_h = &TestImage::from_file("stone-h.png");
    let grass = &TestImage::from_file("grass.png");
    let leaves = &TestImage::from_file("leaves.png");
    let random = &TestImage::new("random color", create_random_color_blocks());

    #[derive(Clone)]
    struct TestImage {
        name: String,
        image: Image<f32>,
    }
    impl TestImage {
        fn new(name: &str, image: Image<f32>) -> Self {
            Self {
                name: name.to_string(),
                image,
            }
        }
        fn from_file(name: &str) -> Self {
            let image = util::read_png_u8(&get_sample(name)).unwrap().to_f32();

            Self {
                name: name.to_string(),
                image,
            }
        }
    }
    struct TestCase<'a> {
        format: Format,
        options: Vec<(&'a str, EncodeOptions)>,
        images: &'a [&'a TestImage],
    }

    fn new_options(f: impl FnOnce(&mut EncodeOptions)) -> EncodeOptions {
        let mut options = EncodeOptions::default();
        f(&mut options);
        options
    }

    let cases = [
        TestCase {
            format: Format::BC1_UNORM,
            options: vec![
                (
                    "fast",
                    new_options(|options| options.quality = CompressionQuality::Fast),
                ),
                (
                    "normal",
                    new_options(|options| options.quality = CompressionQuality::Normal),
                ),
                (
                    "high",
                    new_options(|options| options.quality = CompressionQuality::High),
                ),
                (
                    "dither",
                    new_options(|options| {
                        options.dithering = Dithering::ColorAndAlpha;
                    }),
                ),
                (
                    "perc",
                    new_options(|options| {
                        options.quality = CompressionQuality::High;
                        options.error_metric = ErrorMetric::Perceptual;
                    }),
                ),
                (
                    "perc d",
                    new_options(|options| {
                        options.quality = CompressionQuality::High;
                        options.dithering = Dithering::Color;
                        options.error_metric = ErrorMetric::Perceptual;
                    }),
                ),
            ],
            images: &[
                base,
                color_twirl,
                bricks_d,
                bricks_n,
                clovers_d,
                clovers_r,
                stone_d,
                grass,
                leaves,
                random,
            ],
        },
        TestCase {
            format: Format::BC4_UNORM,
            options: vec![
                (
                    "fast",
                    new_options(|options| options.quality = CompressionQuality::Fast),
                ),
                (
                    "normal",
                    new_options(|options| options.quality = CompressionQuality::Normal),
                ),
                (
                    "high",
                    new_options(|options| options.quality = CompressionQuality::High),
                ),
                (
                    "dither",
                    new_options(|options| {
                        options.quality = CompressionQuality::High;
                        options.dithering = Dithering::ColorAndAlpha;
                    }),
                ),
            ],
            images: &[base, color_twirl, clovers_r, stone_h, random],
        },
        TestCase {
            format: Format::BC4_UNORM,
            options: vec![(
                "ref",
                new_options(|options| {
                    options.quality = CompressionQuality::Unreasonable;
                }),
            )],
            images: &[base],
        },
    ];

    let mut output_summaries = util::OutputSummaries::new("_hashes");
    struct ConfigMetrics(String, Vec<util::Metrics>);
    struct ImageInfo {
        has_alpha: bool,
    }
    #[allow(clippy::type_complexity)]
    let mut collect_metrics = |case: &TestCase| -> Result<
        Vec<(String, ImageInfo, Vec<ConfigMetrics>)>,
        Box<dyn std::error::Error>,
    > {
        let options = case.options.clone();

        let mut data: Vec<(String, ImageInfo, Vec<ConfigMetrics>)> = Vec::new();

        for image in case.images {
            let name = &image.name;
            let org_image = &image.image;

            let info = ImageInfo {
                has_alpha: org_image.channels == Channels::Rgba
                    || org_image.channels == Channels::Alpha,
            };

            let image = org_image.to_channels(case.format.channels());
            let mut metric_list: Vec<ConfigMetrics> = Vec::new();

            for (opt_name, options) in &options {
                let output_file = test_data_dir()
                    .join("output-encode/compression")
                    .join(format!(
                        "{:?} {} {}.dds",
                        case.format,
                        opt_name,
                        name.trim_end_matches(".png")
                    ));
                let (encoded_bytes, encoded_image) = encode_decode(case.format, options, &image);

                // write file
                std::fs::create_dir_all(output_file.parent().unwrap())?;
                std::fs::write(&output_file, &encoded_bytes)?;
                let hash = util::hash_hex(&encoded_bytes);
                output_summaries.add_output_file(&output_file, &hash);

                // get metrics
                let metrics = util::measure_compression_quality(&image, &encoded_image);

                metric_list.push(ConfigMetrics(opt_name.to_string(), metrics));
            }

            data.push((name.to_string(), info, metric_list));
        }

        // summary
        if data.len() > 1 {
            let mut summary: Vec<ConfigMetrics> = Vec::new();
            for (index, (opt_name, _)) in options.iter().enumerate() {
                let mut averages: Vec<util::Metrics> = Vec::new();
                let scale = 1.0 / data.len() as f64;
                for (_, _, metrics_for_config) in &data {
                    let metrics = &metrics_for_config[index].1;
                    for m in metrics {
                        let avg: &mut util::Metrics = if let Some(avg) =
                            averages.iter_mut().find(|avg| avg.channel == m.channel)
                        {
                            avg
                        } else {
                            averages.push(util::Metrics {
                                channel: m.channel,
                                mse: 0.0,
                                mse_blur: 0.0,
                                region_error: 0.0,
                            });
                            averages.last_mut().unwrap()
                        };

                        avg.mse += m.mse * scale;
                        avg.mse_blur += m.mse_blur * scale;
                        avg.region_error += m.region_error * scale;
                    }
                }

                summary.push(ConfigMetrics(opt_name.to_string(), averages));
            }
            data.insert(
                0,
                (
                    "Summary".to_string(),
                    ImageInfo { has_alpha: true },
                    summary,
                ),
            );
        }

        Ok(data)
    };
    let mut collect_info = |case: &TestCase| -> Result<String, Box<dyn std::error::Error>> {
        let mut output = String::new();

        let options = case.options.clone();
        for (name, option) in &options {
            output.push_str(&format!("- {}: {:?}\n", name, option));
        }
        output.push('\n');

        let mut table =
            util::PrettyTable::from_header(&["", "", "", "↑PSNR", "↑PSNR B", "↓Region err"]);

        for (name, info, metrics_for_config) in collect_metrics(case)? {
            table.add_empty_row();

            let mut name_mentioned = false;
            for ConfigMetrics(opt_name, metrics) in metrics_for_config {
                let mut opt_mentioned = false;
                let mut printed_metrics = 0;
                for m in metrics {
                    if m.channel == util::MetricChannel::A && !info.has_alpha {
                        continue;
                    }

                    table.add_row(&[
                        if name_mentioned {
                            String::new()
                        } else {
                            name.to_string()
                        },
                        if opt_mentioned {
                            String::new()
                        } else {
                            opt_name.to_string()
                        },
                        format!("{:?}", m.channel),
                        format!("{:.2}", m.psnr()),
                        format!("{:.2}", m.psnr_blur()),
                        format!("{:>5.2}", m.region_error * 255.),
                    ]);
                    name_mentioned = true;
                    opt_mentioned = true;
                    printed_metrics += 1;
                }
                if printed_metrics >= 2 {
                    table.add_empty_row();
                }
            }
        }

        table.print_markdown(&mut output);
        Ok(output)
    };

    let mut output = r"# Encode quality

<!-- This file is generated by `tests/encode.rs` -->

**Metrics:**
- **↑PSNR:** Peak Signal to Noise Ratio.
- **↑PSNR B:** Apply a small blur to the image and then measure PSNR. This is useful to measure the quality of dithering.
- **↓Region err:** This measures the absolute error after downscaling the image to 25%. The error is in 8 bit. So .e.g a region of 4 means that expected absolute error per pixel is +-4/255.

".to_string();
    for case in cases {
        output.push_str(&format!("## `{:?}`\n\n", case.format));

        match collect_info(&case) {
            Ok(info) => output.push_str(&info),
            Err(e) => output.push_str(&format!("Error: {}", e)),
        };

        output.push('\n');
        output.push('\n');
        output.push('\n');
    }

    _ = output_summaries.snapshot();
    util::compare_snapshot_text(&util::test_data_dir().join("encode_quality.md"), &output).unwrap();
}

#[test]
fn block_dither() {
    fn append_quantized(image: &mut Image<u8>) {
        assert!(image.channels == Channels::Grayscale);

        // use BC1 alpha for binary block dithering
        let mut options = EncodeOptions::default();
        options.dithering = Dithering::Alpha;
        let mut temp_image = image.to_f32();
        temp_image.channels = Channels::Alpha;
        let (_, encoded) = encode_decode(
            Format::BC1_UNORM,
            &options,
            &temp_image.to_channels(Channels::Rgba),
        );

        image.size.height *= 2;
        for pixel in encoded.data.chunks_exact(4) {
            let alpha = pixel[3];
            image.data.push((alpha * 255.) as u8);
        }
    }

    let size = Size::new(17 * 8, 8);
    let mut image = Image::new(
        (0..size.pixels() as usize)
            .map(|i| {
                let x = i % size.width as usize;
                // let y = i / size.width as usize;
                let x_quantized = x / 8;
                (x_quantized * 255 / 16) as u8
            })
            .collect(),
        Channels::Grayscale,
        size,
    );
    append_quantized(&mut image);

    let mut image_smooth = Image::new(
        (0..size.pixels() as usize)
            .map(|i| {
                let x = i % size.width as usize;
                (x * 255 / (size.width as usize - 1)) as u8
            })
            .collect(),
        Channels::Grayscale,
        size,
    );
    append_quantized(&mut image_smooth);

    image.size.height *= 2;
    image.data.extend_from_slice(&image_smooth.data);

    util::compare_snapshot_png_u8(
        &util::test_data_dir().join("output-encode/dither.png"),
        &image,
    )
    .unwrap();
}

/// Ensures that all color formats can be encoded (1) without error and (2)
/// get the same result.
///
/// Basically, if we encode a u8 image, we should get the same encoded result
/// as first converting the image to u16/f32 and then encoding that.
#[test]
fn encode_all_color_formats() {
    let base_u8 = util::read_png_u8(&get_sample("base.png")).unwrap();
    let base_u16 = base_u8.to_u16();
    let base_f32 = base_u8.to_f32();

    let mut options = EncodeOptions::default();
    options.quality = CompressionQuality::Fast; // quality isn't relevant here

    let mut failures = String::new();

    for &format in util::ALL_FORMATS {
        if let Some(support) = format.encoding_support() {
            if support.size_multiple().is_some() {
                continue;
            }
        } else {
            // encoding isn't supported
            continue;
        }

        let mut encoded_u8 = Vec::new();
        encode_image(&base_u8, format, &mut encoded_u8, &options).unwrap();

        let mut encoded_u16 = Vec::new();
        encode_image(&base_u16, format, &mut encoded_u16, &options).unwrap();

        let mut encoded_f32 = Vec::new();
        encode_image(&base_f32, format, &mut encoded_f32, &options).unwrap();

        if encoded_u8 != encoded_u16 {
            failures.push_str(&format!("{:?} u8 != u16\n", format));
        }
        if encoded_u8 != encoded_f32 {
            failures.push_str(&format!("{:?} u8 != f32\n", format));
        }
    }

    if !failures.is_empty() {
        panic!("Failed for formats:\n{}", failures);
    }
}

/// This tests mipmap generation.
#[test]
fn encode_mipmap() {
    fn create_mipmap_image(
        snap_path: &Path,
        format: Format,
        options: WriteOptions,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let base = util::read_png_u8(&get_sample("base.png"))?;
        let width = base.size.width;
        let height = base.size.height;

        let mut encoded = Vec::new();
        let mut encoder = Encoder::new(
            &mut encoded,
            format,
            &Header::new_image(width, height, format).with_mipmaps(),
        )?;
        encoder.write_surface_with(base.view(), None, &options)?;
        encoder.finish()?;

        let mut decoder = Decoder::new(std::io::Cursor::new(encoded.as_slice()))?;
        let mut decoded: Image<u8> =
            Image::new_empty(Channels::Rgba, Size::new(width * 3 / 2, height));
        let color = ColorFormat::RGBA_U8;
        let stride = decoded.size.width as usize * 4;
        decoder.read_surface_rect(
            decoded.as_bytes_mut(),
            stride,
            Rect::new(0, 0, width, height),
            color,
        )?;
        let mut offset_y = 0;
        while let Some(info) = decoder.surface_info() {
            let mip_size = info.size();

            decoder.read_surface_rect(
                &mut decoded.as_bytes_mut()[offset_y * stride + (width as usize * 4)..],
                stride,
                Rect::new(0, 0, mip_size.width, mip_size.height),
                color,
            )?;

            offset_y += mip_size.height as usize;
        }

        _ = util::update_snapshot_png_u8(snap_path, &decoded)?;

        let hex = util::hash_hex(&decoded.data);
        Ok(hex)
    }

    let options = [
        WriteOptions {
            resize_straight_alpha: true,
            resize_filter: ResizeFilter::Box,
            ..WriteOptions::default()
        },
        WriteOptions {
            resize_straight_alpha: false,
            resize_filter: ResizeFilter::Box,
            ..WriteOptions::default()
        },
        WriteOptions {
            resize_filter: ResizeFilter::Nearest,
            ..WriteOptions::default()
        },
        WriteOptions {
            resize_filter: ResizeFilter::Triangle,
            ..WriteOptions::default()
        },
        WriteOptions {
            resize_filter: ResizeFilter::Mitchell,
            ..WriteOptions::default()
        },
        WriteOptions {
            resize_filter: ResizeFilter::Lanczos3,
            ..WriteOptions::default()
        },
    ];

    let mut summaries = util::OutputSummaries::new("_hashes");
    for mut option in options {
        option.generate_mipmaps = true;

        let mut name = "base @ ".to_string();
        name.push_str(if option.resize_straight_alpha {
            "straight-alpha"
        } else {
            "no-straight-alpha"
        });
        name.push_str(&format!(" - {:?}", option.resize_filter));

        let snapshot_file = util::test_data_dir()
            .join("output-encode/mipmaps")
            .join(format!("{}.png", name));

        summaries.add_output_file_result(
            &snapshot_file,
            create_mipmap_image(&snapshot_file, Format::R8G8B8A8_UNORM, option),
        );
    }

    summaries.snapshot_or_fail();
}

#[test]
fn test_unaligned() {
    // aligned and unaligned buffers
    let mut buffer = vec![0_u32; 4096];
    let (first, second) = buffer.split_at_mut(2048);
    let aligned_buffer = util::as_bytes_mut(first);
    let unaligned_buffer = &mut util::as_bytes_mut(second)[1..];

    let mut rng = util::create_rng();
    rng.fill_bytes(aligned_buffer);
    unaligned_buffer.copy_from_slice(&aligned_buffer[..unaligned_buffer.len()]);

    let size = Size::new(7, 7);

    let mut aligned_encoded = Vec::new();
    let mut unaligned_encoded = Vec::new();

    for format in [
        Format::R8G8B8A8_UNORM,
        Format::R16G16_UNORM,
        Format::R32_FLOAT,
        Format::AYUV,
        Format::R1_UNORM,
        Format::R8G8_B8G8_UNORM,
        Format::BC1_UNORM,
    ] {
        for &color in util::ALL_COLORS {
            let stride = size.width as usize * color.bytes_per_pixel() as usize;
            let bytes = stride * size.height as usize;

            let aligned = &mut aligned_buffer[..bytes];
            let unaligned = &mut unaligned_buffer[..bytes];

            let aligned = ImageView::new(aligned, size, color).unwrap();
            let unaligned = ImageView::new(unaligned, size, color).unwrap();

            for options in [
                WriteOptions {
                    generate_mipmaps: false,
                    ..WriteOptions::default()
                },
                WriteOptions {
                    generate_mipmaps: true,
                    ..WriteOptions::default()
                },
            ] {
                aligned_encoded.clear();
                unaligned_encoded.clear();

                let mut header = Header::new_image(size.width, size.height, format);
                if options.generate_mipmaps {
                    header = header.with_mipmaps();
                }

                let mut aligned_encoder =
                    Encoder::new(&mut aligned_encoded, format, &header).unwrap();
                aligned_encoder
                    .write_surface_with(aligned, None, &options)
                    .unwrap();
                aligned_encoder.finish().unwrap();

                let mut unaligned_encoder =
                    Encoder::new(&mut unaligned_encoded, format, &header).unwrap();
                unaligned_encoder
                    .write_surface_with(unaligned, None, &options)
                    .unwrap();
                unaligned_encoder.finish().unwrap();

                assert_eq!(
                    aligned_encoded, unaligned_encoded,
                    "Failed for {:?} {:?} {:?}",
                    format, color, options
                );
            }
        }
    }
}
