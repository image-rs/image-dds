use std::path::{Path, PathBuf};

use ddsd::{header::*, *};
use rand::Rng;
use util::{as_bytes_mut, test_data_dir, Image, WithPrecision};

mod util;

fn get_sample(name: &str) -> PathBuf {
    util::test_data_dir().join("samples").join(name)
}

fn create_header(size: Size, format: Format) -> Header {
    if let Ok(dxgi_format) = format.try_into() {
        Header::new_image(size.width, size.height, dxgi_format)
    } else if let Ok(format) = format.try_into() {
        Dx9Header::new_image(size.width, size.height, format).into()
    } else {
        unreachable!("unsupported format: {:?}", format);
    }
}
fn write_dds_header(size: Size, format: Format) -> Vec<u8> {
    let header = create_header(size, format);

    let mut output = Vec::new();
    header.write(&mut output).unwrap();

    output
}
fn encode_image<T: WithPrecision + util::Castable, W: std::io::Write>(
    image: &Image<T>,
    format: Format,
    writer: &mut W,
    options: &EncodeOptions,
) -> Result<(), EncodeError> {
    encode(
        writer,
        format,
        image.size,
        image.color(),
        image.as_bytes(),
        options,
    )
}
fn encode_decode(
    format: Format,
    options: &EncodeOptions,
    image: &Image<f32>,
) -> (Vec<u8>, Image<f32>) {
    // encode
    let mut encoded = write_dds_header(image.size, format);
    encode_image(image, format, &mut encoded, options).unwrap();

    // decode
    let mut decoder = Decoder::new(encoded.as_slice()).unwrap();
    let mut output = vec![0_f32; image.size.pixels() as usize * image.channels.count() as usize];
    decoder
        .read_surface(as_bytes_mut(&mut output), image.color())
        .unwrap();

    let image = Image {
        size: image.size,
        channels: image.channels,
        data: output,
    };

    (encoded, image)
}
fn create_random_color_blocks() -> Image<f32> {
    let mut rng = util::create_rng();

    let width = 256;
    let height = 256;
    let mut data = vec![0_f32; width * height * 3];
    let block_stride = 4 * 3;
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let rgb: [f32; 3] = rng.gen();
            let block_line = [rgb; 4];
            let line_flat: &[f32] = util::cast_slice(&block_line);
            for j in 0..4 {
                let i = ((y + j) * width + x) * 3;
                data[i..i + block_stride].copy_from_slice(line_flat);
            }
        }
    }

    Image {
        size: Size::new(width as u32, height as u32),
        channels: Channels::Rgb,
        data,
    }
}
fn compression_ratio(data: &[u8]) -> f64 {
    let compressed = miniz_oxide::deflate::compress_to_vec(data, 6);
    compressed.len() as f64 / data.len() as f64
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
        if let Some(encoding) = format.encoding() {
            size = size.round_down_to_multiple(encoding.size_multiple);
        };

        let mut output = write_dds_header(size, format);

        let options = EncodeOptions::default();

        // and now the image data
        if format.precision() == Precision::U16 {
            encode_image(&base_u16.cropped(size), format, &mut output, &options)?;
        } else if format.precision() == Precision::F32 {
            encode_image(&base_f32.cropped(size), format, &mut output, &options)?;
        } else {
            encode_image(&base_u8.cropped(size), format, &mut output, &options)?;
        }

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
        let mut output = write_dds_header(image.size, format);

        let mut options = EncodeOptions::default();
        options.dithering = Dithering::ColorAndAlpha;
        encode_image(image, format, &mut output, &options)?;

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
        .filter_map(|f| f.encoding().map(|e| (f, e)))
        .filter(|(_, e)| e.dithering != Dithering::None)
    {
        let mut test_and_summarize = |image, name| {
            let output_path = get_output_dds(format, name);
            summaries.add_output_file_result(&output_path, test(format, image, &output_path));
        };

        test_and_summarize(&base, "base");

        if encoding.dithering.color() {
            test_and_summarize(&twirl, "twirl");
        }
    }

    summaries.snapshot_or_fail()
}

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
                ("default", EncodeOptions::default()),
                (
                    "dither",
                    new_options(|options| {
                        options.dithering = Dithering::ColorAndAlpha;
                    }),
                ),
                (
                    "perc",
                    new_options(|options| {
                        options.error_metric = ErrorMetric::Perceptual;
                    }),
                ),
                (
                    "perc d",
                    new_options(|options| {
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
                ("default", EncodeOptions::default()),
                (
                    "dither",
                    new_options(|options| {
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
    let mut collect_info = |case: &TestCase| -> Result<String, Box<dyn std::error::Error>> {
        let mut output = String::new();

        let mut options = case.options.clone();
        if options.is_empty() {
            options.push(("", EncodeOptions::default()));
        }
        for (name, option) in &options {
            if name.is_empty() {
                output.push_str("Default options");
            } else {
                output.push_str(name);
            }
            output.push_str(&format!(": {:?}\n", option));
        }
        output.push('\n');

        let mut table =
            util::PrettyTable::from_header(&["", "", "", "↑PSNR", "↑PSNR blur", "↓Region error"]);

        for image in case.images {
            let hash_alpha = matches!(image.image.channels, Channels::Rgba | Channels::Alpha);

            table.add_empty_row();

            let name = &image.name;
            let image = image.image.to_channels(case.format.channels());
            let mut name_mentioned = false;
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

                let compression = compression_ratio(&encoded_bytes);

                let metrics = util::measure_compression_quality(&image, &encoded_image);
                let mut opt_mentioned = false;
                for m in metrics {
                    if m.channel == util::MetricChannel::A && !hash_alpha {
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
                        format!("{:.4}", m.psnr),
                        format!("{:.4}", m.psnr_blur),
                        format!("{:.5}", m.region_error * 255.),
                    ]);
                    name_mentioned = true;
                    opt_mentioned = true;
                }

                table.add_row(&[
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    "compressed".to_string(),
                    format!("{:.2}%", compression * 100.),
                ]);
            }
        }

        table.print(&mut output);
        Ok(output)
    };

    let mut output = String::new();
    for case in cases {
        output.push_str(&format!("{:?}\n", case.format));

        let info = match collect_info(&case) {
            Ok(info) => info,
            Err(e) => format!("Error: {}", e),
        };

        for line in info.lines().map(|l| l.trim_end()) {
            if line.is_empty() {
                output.push('\n');
            } else {
                output.push_str(&format!("    {}\n", line));
            }
        }

        output.push('\n');
        output.push('\n');
        output.push('\n');
    }

    _ = output_summaries.snapshot();
    util::compare_snapshot_text(&util::test_data_dir().join("encode_quality.txt"), &output)
        .unwrap();
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
    let mut image = Image {
        size,
        channels: Channels::Grayscale,
        data: (0..size.pixels() as usize)
            .map(|i| {
                let x = i % size.width as usize;
                // let y = i / size.width as usize;
                let x_quantized = x / 8;
                (x_quantized * 255 / 16) as u8
            })
            .collect(),
    };
    append_quantized(&mut image);

    let mut image_smooth = Image {
        size,
        channels: Channels::Grayscale,
        data: (0..size.pixels() as usize)
            .map(|i| {
                let x = i % size.width as usize;
                (x * 255 / (size.width as usize - 1)) as u8
            })
            .collect(),
    };
    append_quantized(&mut image_smooth);

    image.size.height *= 2;
    image.data.extend_from_slice(&image_smooth.data);

    util::compare_snapshot_png_u8(
        &util::test_data_dir().join("output-encode/dither.png"),
        &image,
    )
    .unwrap();
}
