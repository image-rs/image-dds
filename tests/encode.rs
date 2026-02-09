use dds::*;
use rand::{Rng, RngCore};
use std::path::{Path, PathBuf};

use util::{test_data_dir, Image, Snapshot, WithPrecision, ALL_COLORS};

mod util;

trait SetConfiguration<T> {
    fn set(&mut self, value: T);
}
impl SetConfiguration<CompressionQuality> for EncodeOptions {
    fn set(&mut self, value: CompressionQuality) {
        self.quality = value;
    }
}
impl SetConfiguration<ErrorMetric> for EncodeOptions {
    fn set(&mut self, value: ErrorMetric) {
        self.error_metric = value;
    }
}
impl SetConfiguration<Dithering> for EncodeOptions {
    fn set(&mut self, value: Dithering) {
        self.dithering = value;
    }
}
macro_rules! new_options {
    ($($e:expr),*) => {{
        let mut options = EncodeOptions::default();
        $(
            options.set($e);
        )*
        options
    }};
}

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
    let mut encoder = Encoder::new_image(&mut encoded, image.size, format, false).unwrap();
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
        let name = format!("{format:?}.dds");
        test_data_dir().join("output-encode/base").join(&name)
    }
    let test = |format: Format, dds_path: &Path| -> Result<String, Box<dyn std::error::Error>> {
        let mut size = base_u8.size;
        if let Some(support) = format.encoding_support() {
            if let Some((w_mul, h_mul)) = support.size_multiple() {
                // round down to the nearest multiple
                let w_mul = w_mul.get();
                let h_mul = h_mul.get();
                size = Size::new((size.width / w_mul) * w_mul, (size.height / h_mul) * h_mul);
            }
        };

        let mut output = Vec::new();
        let mut encoder = Encoder::new_image(&mut output, size, format, false)?;
        encoder.options.quality = CompressionQuality::High;
        encoder.options.parallel = false;

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
        let name = format!("{format:?} {name}.dds");
        test_data_dir().join("output-encode/dither").join(&name)
    }
    fn test(
        format: Format,
        image: &Image<f32>,
        dds_path: &Path,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut output = Vec::new();
        let mut encoder = Encoder::new_image(&mut output, image.size, format, false)?;
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

    let base = util::read_png_f32(&get_sample("base.png")).unwrap();
    let twirl = util::read_png_f32(&get_sample("color-twirl.png")).unwrap();

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
    use crate::util::{MetricChannel, MetricChannelSet};

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
            let image = util::read_png_f32(&get_sample(name)).unwrap();

            Self {
                name: name.to_string(),
                image,
            }
        }
    }
    struct TestCase<'a> {
        format: Format,
        options: Vec<(&'a str, EncodeOptions, MetricChannelSet)>,
        get_overview_channel: Option<fn(&EncodeOptions, MetricChannelSet) -> MetricChannel>,
        images: &'a [&'a TestImage],
    }

    use CompressionQuality::*;
    use ErrorMetric::*;

    let bc1_metrics = MetricChannelSet::RGB | MetricChannel::L | MetricChannel::C;
    let bc7_metrics = MetricChannelSet::RGBA | MetricChannel::C;
    let cases = [
        TestCase {
            format: Format::BC1_UNORM,
            get_overview_channel: Some(|options, _| {
                if options.error_metric == Perceptual {
                    MetricChannel::L
                } else {
                    MetricChannel::C
                }
            }),
            options: vec![
                ("uni fast", new_options!(Fast, Uniform), bc1_metrics),
                ("uni norm", new_options!(Normal, Uniform), bc1_metrics),
                ("uni high", new_options!(High, Uniform), bc1_metrics),
                ("per fast", new_options!(Fast, Perceptual), bc1_metrics),
                ("per norm", new_options!(Normal, Perceptual), bc1_metrics),
                ("per high", new_options!(High, Perceptual), bc1_metrics),
                (
                    "dith uni",
                    new_options!(High, Uniform, Dithering::ColorAndAlpha),
                    bc1_metrics | MetricChannel::A,
                ),
                (
                    "dith per",
                    new_options!(High, Perceptual, Dithering::ColorAndAlpha),
                    bc1_metrics | MetricChannel::A,
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
            get_overview_channel: None,
            options: vec![
                ("fast", new_options!(Fast), MetricChannelSet::GRAY),
                ("normal", new_options!(Normal), MetricChannelSet::GRAY),
                ("high", new_options!(High), MetricChannelSet::GRAY),
                (
                    "dither",
                    new_options!(High, Dithering::ColorAndAlpha),
                    MetricChannelSet::GRAY,
                ),
            ],
            images: &[base, color_twirl, clovers_r, stone_h, random],
        },
        TestCase {
            format: Format::BC4_UNORM,
            get_overview_channel: None,
            options: vec![(
                "ref",
                new_options!(CompressionQuality::Unreasonable),
                MetricChannelSet::GRAY,
            )],
            images: &[base],
        },
        TestCase {
            format: Format::BC7_UNORM,
            get_overview_channel: Some(|_, _| MetricChannel::C),
            options: vec![
                ("fast", new_options!(Fast), bc7_metrics),
                ("norm", new_options!(Normal), bc7_metrics),
                ("high", new_options!(High), bc7_metrics),
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
    ];

    let mut output_summaries = util::OutputSummaries::new("_hashes");
    struct ConfigMetrics(String, Vec<util::Metrics>);
    struct ImageMeasurements {
        image_name: String,
        info: ImageInfo,
        metrics_for_config: Vec<ConfigMetrics>,
    }
    #[derive(Clone)]
    struct ImageInfo {
        has_alpha: bool,
    }
    #[allow(clippy::type_complexity)]
    let mut collect_metrics = |case: &TestCase| -> Result<_, Box<dyn std::error::Error>> {
        let options = case.options.clone();

        let mut data: Vec<ImageMeasurements> = Vec::new();

        for image in case.images {
            let name = &image.name;
            let org_image = &image.image;

            let info = ImageInfo {
                has_alpha: org_image.channels == Channels::Rgba
                    || org_image.channels == Channels::Alpha,
            };

            let image = org_image
                .to_channels(case.format.channels())
                .to_channels(Channels::Rgba);
            let mut metric_list: Vec<ConfigMetrics> = Vec::new();

            for (opt_name, options, metric_channels) in &options {
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
                let metrics =
                    util::measure_compression_quality(&image, &encoded_image, *metric_channels);

                metric_list.push(ConfigMetrics(opt_name.to_string(), metrics));
            }

            data.push(ImageMeasurements {
                image_name: name.to_string(),
                info,
                metrics_for_config: metric_list,
            });
        }

        // summary
        if data.len() > 1 {
            let summary = create_measurement_summary(&data);
            data.insert(0, summary);
        }

        Ok(data)
    };
    fn create_measurement_summary(data: &[ImageMeasurements]) -> ImageMeasurements {
        let mut option_names: Vec<String> = Vec::new();
        for m in &data[0].metrics_for_config {
            option_names.push(m.0.clone());
        }

        let mut summary: Vec<ConfigMetrics> = Vec::new();
        for (index, opt_name) in option_names.into_iter().enumerate() {
            let mut averages: Vec<util::Metrics> = Vec::new();
            let scale = 1.0 / data.len() as f64;
            for measurement in data {
                let metrics = &measurement.metrics_for_config[index].1;
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

        ImageMeasurements {
            image_name: "Summary".to_string(),
            info: ImageInfo { has_alpha: true },
            metrics_for_config: summary,
        }
    }
    fn create_overview(
        data: &[ImageMeasurements],
        channel_for_config: &[MetricChannel],
    ) -> Vec<ImageMeasurements> {
        let mut overview: Vec<ImageMeasurements> = Vec::new();

        for measurement in data {
            assert_eq!(
                measurement.metrics_for_config.len(),
                channel_for_config.len()
            );

            let mut metrics_for_config: Vec<ConfigMetrics> = Vec::new();

            for (metrics, &overview_channel) in measurement
                .metrics_for_config
                .iter()
                .zip(channel_for_config.iter())
            {
                let m = metrics
                    .1
                    .iter()
                    .find(|m| m.channel == overview_channel)
                    .unwrap_or(&metrics.1[0]);

                metrics_for_config.push(ConfigMetrics(metrics.0.clone(), vec![m.clone()]));
            }

            overview.push(ImageMeasurements {
                image_name: measurement.image_name.clone(),
                info: measurement.info.clone(),
                metrics_for_config,
            });
        }

        overview
    }
    fn create_table(data: &[ImageMeasurements]) -> util::PrettyTable {
        let mut table =
            util::PrettyTable::from_header(&["", "", "", "↑PSNR", "↑PSNR B", "↓Region err"]);

        for measurement in data {
            table.add_empty_row();

            let mut name_mentioned = false;
            for ConfigMetrics(opt_name, metrics) in &measurement.metrics_for_config {
                let mut opt_mentioned = false;
                let mut printed_metrics = 0;
                for m in metrics {
                    if m.channel == util::MetricChannel::A && !measurement.info.has_alpha {
                        continue;
                    }

                    table.add_row(&[
                        if name_mentioned {
                            String::new()
                        } else {
                            measurement.image_name.clone()
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

        table
    }
    let mut collect_info = |case: &TestCase| -> Result<String, Box<dyn std::error::Error>> {
        let mut output = String::new();

        let options = case.options.clone();
        for (name, option, _) in &options {
            output.push_str(&format!("- {name}: {option:?}\n"));
        }
        output.push('\n');

        let measurements = collect_metrics(case)?;

        if let Some(get_overview_channel) = case.get_overview_channel {
            let mut overview_channels = Vec::new();
            for option in &case.options {
                overview_channels.push(get_overview_channel(&option.1, option.2));
            }
            let overview = create_overview(&measurements, &overview_channels);
            create_table(&overview).print_markdown(&mut output);

            output.push_str("\n<details>\n<summary>Full details</summary>\n\n");
            create_table(&measurements).print_markdown(&mut output);
            output.push_str("\n</details>\n");
        } else {
            create_table(&measurements).print_markdown(&mut output);
        }

        Ok(output)
    };

    let mut output = r"# Encode quality

<!-- This file is generated by `tests/encode.rs` -->

**Channels:**
- **R/G/B/A:** Red, green, blue, and alpha channels.
- **C:** Average error of R+G+B.
- **L:** Oklab luminance channel.
- **Gray:** Grayscale. This is the same as (R+G+B)/3.

**Metrics:**
- **↑PSNR:** Peak Signal to Noise Ratio.
- **↑PSNR B:** Apply a small blur to the image and then measure PSNR. This is useful to measure the quality of dithering.
- **↓Region err:** This measures the absolute error after downscaling the image to 25%. The error is in 8 bit. So .e.g a region of 4 means that expected absolute error per pixel is +-4/255.

".to_string();
    for case in cases {
        output.push_str(&format!("## `{:?}`\n\n", case.format));

        match collect_info(&case) {
            Ok(info) => output.push_str(&info),
            Err(e) => output.push_str(&format!("Error: {e}")),
        };

        output.push('\n');
        output.push('\n');
        output.push('\n');
    }

    _ = output_summaries.snapshot();
    util::TextSnapshot.assert(&util::test_data_dir().join("encode_quality.md"), &output);
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

    util::PngSnapshot.assert(
        &util::test_data_dir().join("output-encode/dither.png"),
        &image,
    );
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
            failures.push_str(&format!("{format:?} u8 != u16\n"));
        }
        if encoded_u8 != encoded_f32 {
            failures.push_str(&format!("{format:?} u8 != f32\n"));
        }
    }

    if !failures.is_empty() {
        panic!("Failed for formats:\n{failures}");
    }
}

fn create_mipmap_chain_image(
    image: ImageView,
    mipmaps: MipmapOptions,
    format: Format,
) -> Image<u8> {
    let width = image.width();
    let height = image.height();

    let mut encoded = Vec::new();
    let mut encoder = Encoder::new_image(&mut encoded, image.size(), format, true).unwrap();
    encoder.mipmaps = mipmaps;
    encoder.mipmaps.generate = true;
    encoder.write_surface(image).unwrap();
    encoder.finish().unwrap();

    let mut decoder = Decoder::new(std::io::Cursor::new(encoded.as_slice())).unwrap();
    let mut decoded: Image<u8> = Image::new_empty(Channels::Rgba, Size::new(width * 3 / 2, height));
    decoder
        .read_surface(decoded.view_mut().cropped(Offset::ZERO, image.size()))
        .unwrap();
    let mut offset_y = 0;
    while let Some(info) = decoder.surface_info() {
        let mip_size = info.size();

        decoder
            .read_surface(
                decoded
                    .view_mut()
                    .cropped(Offset::new(width, offset_y), mip_size),
            )
            .unwrap();

        offset_y += mip_size.height;
    }

    decoded
}
// Don't run this on big endian targets, they have problems with f32 precision
#[cfg(not(target_endian = "big"))]
/// This tests mipmap generation.
#[test]
fn encode_mipmap_chain() {
    fn save_mipmap_chain_image(
        snap_path: &Path,
        format: Format,
        mipmaps: MipmapOptions,
        image: ImageView,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let chain = create_mipmap_chain_image(image, mipmaps, format);
        _ = util::PngSnapshot.write(snap_path, &chain)?;
        let hex = util::hash_hex(&chain.data);
        Ok(hex)
    }

    let image_names = ["base", "bricks-d"];

    let mut summaries = util::OutputSummaries::new("_hashes");
    for image_name in image_names {
        let image = util::read_png_u8(&get_sample(&format!("{image_name}.png"))).unwrap();
        let is_rgba =
            image.channels == Channels::Rgba && image.data.chunks_exact(4).any(|c| c[3] != 255);

        let image_u8 = image.to_channels(Channels::Rgba);
        let image_u16 = image_u8.to_u16();
        let image_f32 = image_u8.to_f32();

        let straight_alpha_options: &[bool] = if is_rgba { &[false, true] } else { &[true] };

        for image in [image_u8.view(), image_u16.view(), image_f32.view()] {
            for &filter in util::ALL_RESIZE_FILTERS {
                for &straight_alpha in straight_alpha_options {
                    let option = MipmapOptions {
                        resize_filter: filter,
                        resize_straight_alpha: straight_alpha,
                        ..Default::default()
                    };

                    let mut name = format!("{image_name} @ {filter:?} {}", image.color());

                    if image.color().channels == Channels::Rgba && is_rgba {
                        name += " ";
                        name += if option.resize_straight_alpha {
                            "alpha-straight"
                        } else {
                            "alpha-custom"
                        };
                    }

                    let snapshot_file = util::test_data_dir()
                        .join("output-encode/mipmaps")
                        .join(name + ".png");

                    summaries.add_output_file_result(
                        &snapshot_file,
                        save_mipmap_chain_image(
                            &snapshot_file,
                            Format::R8G8B8A8_UNORM,
                            option,
                            image,
                        ),
                    );
                }
            }
        }
    }

    summaries.snapshot_or_fail();
}
// Don't run this on big endian targets, it takes too long
#[cfg(not(target_endian = "big"))]
/// This test verifies the correctness of mipmap generation of all channel
/// variants by checking against RGBA. This means that other tests verifying
/// mipmap generation only need to consider RGBA.
#[test]
fn mipmap_channel_invariants() {
    fn check_invariants(
        filter: ResizeFilter,
        image_rgba: ImageView,
        image_rgb: ImageView,
        image_gray: ImageView,
        image_alpha: ImageView,
        info: &str,
    ) {
        let mut option = MipmapOptions {
            resize_filter: filter,
            ..Default::default()
        };
        let filter = Format::R8G8B8A8_UNORM;

        option.resize_straight_alpha = true;
        let chain_rgba_straight = create_mipmap_chain_image(image_rgba, option, filter);
        option.resize_straight_alpha = false;
        let chain_rgba_custom = create_mipmap_chain_image(image_rgba, option, filter);

        let chain_rgb = create_mipmap_chain_image(image_rgb, option, filter);
        let chain_gray = create_mipmap_chain_image(image_gray, option, filter);
        let chain_alpha = create_mipmap_chain_image(image_alpha, option, filter);

        assert_eq!(chain_rgba_straight.size, chain_rgba_custom.size);
        assert_eq!(chain_rgba_straight.size, chain_rgb.size);
        assert_eq!(chain_rgba_straight.size, chain_gray.size);
        assert_eq!(chain_rgba_straight.size, chain_alpha.size);

        assert_eq!(chain_rgba_straight.channels, Channels::Rgba);
        assert_eq!(chain_rgba_custom.channels, Channels::Rgba);
        assert_eq!(chain_rgb.channels, Channels::Rgba);
        assert_eq!(chain_gray.channels, Channels::Rgba);
        assert_eq!(chain_alpha.channels, Channels::Rgba);

        let rgba_straight: &[[u8; 4]] = util::from_bytes(&chain_rgba_straight.data).unwrap();
        let rgba_custom: &[[u8; 4]] = util::from_bytes(&chain_rgba_custom.data).unwrap();
        let rgb: &[[u8; 4]] = util::from_bytes(&chain_rgb.data).unwrap();
        let gray: &[[u8; 4]] = util::from_bytes(&chain_gray.data).unwrap();
        let alpha: &[[u8; 4]] = util::from_bytes(&chain_alpha.data).unwrap();

        for i in 0..rgba_straight.len() {
            let rgba_straight = rgba_straight[i];
            let rgba_custom = rgba_custom[i];
            let rgb = rgb[i];
            let gray = gray[i];
            let alpha = alpha[i];

            // the RGBAs
            assert_eq!(rgba_straight[3], rgba_custom[3], "Failed at {i} for {info}");

            // RGB
            assert_eq!(rgb[0], rgba_custom[0], "Failed at {i} for {info}");
            assert_eq!(rgb[1], rgba_custom[1], "Failed at {i} for {info}");
            assert_eq!(rgb[2], rgba_custom[2], "Failed at {i} for {info}");

            // Gray
            assert_eq!(gray[0], rgb[0], "Failed at {i} for {info}");

            // Alpha
            assert_eq!(alpha[3], rgba_custom[3], "Failed at {i} for {info}");
        }
    }

    let image_names = ["base", "bricks-d"];

    for image_name in image_names {
        let image_rgba = util::read_png_u8(&get_sample(&format!("{image_name}.png")))
            .unwrap()
            .to_channels(Channels::Rgba);

        let image_rgb = image_rgba.to_channels(Channels::Rgb);
        let image_gray = image_rgba.to_channels(Channels::Grayscale);
        let image_alpha = image_rgba.to_channels(Channels::Alpha);

        for &filter in util::ALL_RESIZE_FILTERS {
            let info = format!("{image_name} with {filter:?}");

            check_invariants(
                filter,
                image_rgba.view(),
                image_rgb.view(),
                image_gray.view(),
                image_alpha.view(),
                &info,
            );
            check_invariants(
                filter,
                image_rgba.to_u16().view(),
                image_rgb.to_u16().view(),
                image_gray.to_u16().view(),
                image_alpha.to_u16().view(),
                &info,
            );
            check_invariants(
                filter,
                image_rgba.to_f32().view(),
                image_rgb.to_f32().view(),
                image_gray.to_f32().view(),
                image_alpha.to_f32().view(),
                &info,
            );
        }
    }
}
/// This test verifies the correctness of mipmap generation of all precision
/// variants by checking against F32. This means that other tests verifying
/// mipmap generation only need to consider one precision.
#[test]
fn mipmap_precision_invariants() {
    fn image_diff(a: &Image<u8>, b: &Image<u8>) -> u8 {
        assert_eq!(a.channels, Channels::Rgba);
        assert_eq!(b.channels, Channels::Rgba);
        assert_eq!(a.size, b.size);

        let mut max_diff = 0;
        for (&a, &b) in a.data.iter().zip(&b.data) {
            max_diff = a.abs_diff(b).max(max_diff);
        }

        max_diff
    }

    let image_names = ["base", "bricks-d"];

    for image_name in image_names {
        let image_u8 = util::read_png_u8(&get_sample(&format!("{image_name}.png")))
            .unwrap()
            .to_channels(Channels::Rgba);

        let image_u16 = image_u8.to_u16();
        let image_f32 = image_u8.to_f32();

        for &filter in util::ALL_RESIZE_FILTERS {
            for straight_alpha in [false, true] {
                let info =
                    format!("{image_name} with {filter:?} and straight_alpha={straight_alpha}");
                let options = MipmapOptions {
                    resize_filter: filter,
                    resize_straight_alpha: straight_alpha,
                    ..Default::default()
                };
                let format = Format::R8G8B8A8_UNORM;

                let chain_u8 = create_mipmap_chain_image(image_u8.view(), options, format);
                let chain_u16 = create_mipmap_chain_image(image_u16.view(), options, format);
                let chain_f32 = create_mipmap_chain_image(image_f32.view(), options, format);

                let u8_diff = image_diff(&chain_u8, &chain_f32);
                let u16_diff = image_diff(&chain_u16, &chain_f32);

                assert!(
                    u8_diff <= 1,
                    "Expected U8 diff ({u8_diff}) to be smaller for {info}"
                );
                assert!(
                    u16_diff <= 1,
                    "Expected U16 diff ({u16_diff}) to be smaller for {info}"
                );
            }
        }
    }
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

            for mipmaps in [false, true] {
                aligned_encoded.clear();
                unaligned_encoded.clear();

                let mut aligned_encoder =
                    Encoder::new_image(&mut aligned_encoded, size, format, mipmaps).unwrap();
                aligned_encoder.write_surface(aligned).unwrap();
                aligned_encoder.finish().unwrap();

                let mut unaligned_encoder =
                    Encoder::new_image(&mut unaligned_encoded, size, format, mipmaps).unwrap();
                unaligned_encoder.write_surface(unaligned).unwrap();
                unaligned_encoder.finish().unwrap();

                assert_eq!(
                    aligned_encoded, unaligned_encoded,
                    "Failed for {format:?} {color:?} {mipmaps:?}"
                );
            }
        }
    }
}

// This test ensures that mipmap generation has the same results for unaligned
// and non-contiguous images as for simple aligned contiguous images.
#[test]
fn test_unaligned_mipmaps() {
    fn for_each_unaligned(image: ImageView, consume: impl FnOnce(ImageView, ImageView, ImageView)) {
        assert!(image.is_contiguous());
        let image_bytes = image.data();

        // a version of the image that is unaligned
        let mut offset_bytes = vec![0_u8; image_bytes.len() + 1];
        offset_bytes[1..].copy_from_slice(image_bytes);
        let offset_view = ImageView::new(&offset_bytes[1..], image.size(), image.color()).unwrap();

        // a version of the image that is unaligned and non-contiguous
        let row_pitch = image.row_pitch() + 5;
        let mut row_pitch_bytes = vec![0_u8; 1 + image.height() as usize * row_pitch];
        let row_pitch_bytes = &mut row_pitch_bytes[1..];
        for (y, row) in image.rows().enumerate() {
            let offset = y * row_pitch;
            let rrow = &mut row_pitch_bytes[offset..offset + image.row_pitch()];
            rrow.copy_from_slice(row);
        }
        let row_pitch_view =
            ImageView::new_with(row_pitch_bytes, row_pitch, image.size(), image.color()).unwrap();

        consume(image, offset_view, row_pitch_view);
    }

    fn encode_dds_bytes(image: ImageView, filter: ResizeFilter, straight_alpha: bool) -> Vec<u8> {
        let mut buffer = Vec::new();

        let mut encoder =
            Encoder::new_image(&mut buffer, image.size(), Format::R8G8B8A8_UNORM, true).unwrap();
        encoder.mipmaps.resize_filter = filter;
        encoder.mipmaps.resize_straight_alpha = straight_alpha;
        encoder.write_surface(image).unwrap();
        encoder.finish().unwrap();

        buffer
    }

    for &color in ALL_COLORS {
        for size in [Size::new(128, 128), Size::new(101, 127)] {
            let image_bytes = color.buffer_size(size).unwrap();
            let mut buffer = vec![0_u32; image_bytes.div_ceil(4)];
            let buffer = &mut util::as_bytes_mut(&mut buffer)[..image_bytes];
            let mut rng = util::create_rng();
            rng.fill_bytes(buffer);

            let image = ImageView::new(buffer, size, color).unwrap();

            for_each_unaligned(image, |image, image_unaligned, image_row_pitch| {
                for &filter in util::ALL_RESIZE_FILTERS {
                    for straight_alpha in [false, true] {
                        let config = format!("color={color} size={size:?} filter={filter:?} straight_alpha={straight_alpha}");

                        let dds = encode_dds_bytes(image, filter, straight_alpha);
                        let dds_unaligned =
                            encode_dds_bytes(image_unaligned, filter, straight_alpha);
                        let dds_row_pitch =
                            encode_dds_bytes(image_row_pitch, filter, straight_alpha);

                        assert!(
                            dds == dds_unaligned,
                            "encoding unaligned is different for config: {config}"
                        );
                        assert!(
                            dds == dds_row_pitch,
                            "encoding with row pitch is different for config: {config}"
                        );
                    }
                }
            });
        }
    }
}

#[test]
fn test_row_pitch() {
    for &color in util::ALL_COLORS {
        let bpp = color.bytes_per_pixel() as usize;

        let backing_size = Size::new(256, 256);
        let mut buffer = vec![0_u8; backing_size.pixels() as usize * bpp];
        util::create_rng().fill_bytes(&mut buffer);
        let backing = ImageView::new(&buffer, backing_size, color).unwrap();

        // I'm using prime numbers for the rect to make things as difficult as possible for the impl
        let image = backing.cropped(Offset::new(13, 29), Size::new(17, 51));

        // create a contiguous version of the image
        let mut cont = vec![0_u8; image.size().pixels() as usize * bpp];
        let row_pitch = image.row_pitch();
        let bytes_per_row = image.size().width as usize * bpp;
        let data = image.data();
        for y in 0..image.size().height as usize {
            let src_row = &data[y * row_pitch..][..bytes_per_row];
            let dst_row = &mut cont[y * bytes_per_row..][..bytes_per_row];
            dst_row.copy_from_slice(src_row);
        }
        let cont_image = ImageView::new(&cont, image.size(), color).unwrap();

        // check that the two images are the same
        for (row, cont_row) in image.rows().zip(cont_image.rows()) {
            assert_eq!(row, cont_row, "Row mismatch");
        }

        for format in util::ALL_FORMATS.iter().copied() {
            let Some(encoding) = format.encoding_support() else {
                // encoding isn't supported
                continue;
            };

            let mut size = image.size();
            if !encoding.supports_size(size) {
                let (w_mul, h_mul) = encoding.size_multiple().unwrap();
                // round down to the nearest multiple
                let w_mul = w_mul.get();
                let h_mul = h_mul.get();
                size = Size::new((size.width / w_mul) * w_mul, (size.height / h_mul) * h_mul);
            }

            let image = image.cropped(Offset::ZERO, size);
            let cont_image = cont_image.cropped(Offset::ZERO, size);

            // size multiple make mipmaps difficult, so only do them for
            // formats that support images of any size.
            let mipmaps = encoding.size_multiple().is_none();

            let mut cont_encoded = Vec::new();
            let mut cont_encoder =
                Encoder::new_image(&mut cont_encoded, size, format, mipmaps).unwrap();
            cont_encoder.write_surface(cont_image).unwrap();
            cont_encoder.finish().unwrap();

            let mut non_cont_encoded = Vec::new();
            let mut non_cont_encoder =
                Encoder::new_image(&mut non_cont_encoded, size, format, mipmaps).unwrap();
            non_cont_encoder.write_surface(image).unwrap();
            non_cont_encoder.finish().unwrap();

            assert_eq!(
                cont_encoded.len(),
                non_cont_encoded.len(),
                "Failed for {format:?}"
            );
            assert!(cont_encoded == non_cont_encoded, "Failed for {format:?}");
        }
    }
}

mod errors {
    use super::*;

    #[test]
    fn unsupported_format() {
        let result = Encoder::new_image(
            std::io::sink(),
            Size::new(1, 1),
            Format::ASTC_10X10_UNORM,
            false,
        );
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(
            err,
            EncodingError::UnsupportedFormat(Format::ASTC_10X10_UNORM)
        ));
        assert_eq!(
            err.to_string(),
            "Unsupported format for encoding: ASTC_10X10_UNORM"
        );
    }

    #[test]
    fn size_multiple() {
        // size is required to be multiple of a certain number
        // NV12 requires 2x2
        let mut encoder =
            Encoder::new_image(std::io::sink(), Size::new(5, 5), Format::NV12, false).unwrap();
        let image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(5, 5));

        let result = encoder.write_surface(image.view());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, EncodingError::InvalidSize(_, _)));
        assert_eq!(err.to_string(), "Size must be a multiple of 2x2");
    }

    #[test]
    fn wrong_surface_size() {
        let mut encoder =
            Encoder::new_image(std::io::sink(), Size::new(8, 8), Format::R8_UNORM, false).unwrap();
        let image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(8, 10));

        let result = encoder.write_surface(image.view());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, EncodingError::UnexpectedSurfaceSize));
        assert_eq!(err.to_string(), "Unexpected surface size");
    }

    #[test]
    fn too_many_surfaces() {
        let mut encoder =
            Encoder::new_image(std::io::sink(), Size::new(8, 8), Format::R8_UNORM, false).unwrap();
        let image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(8, 8));
        encoder.write_surface(image.view()).unwrap(); // write surface

        let result = encoder.write_surface(image.view());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, EncodingError::TooManySurfaces));
        assert_eq!(
            err.to_string(),
            "Too many surfaces are attempted to written"
        );
    }

    #[test]
    fn missing_surfaces() {
        let mut encoder =
            Encoder::new_image(std::io::sink(), Size::new(8, 8), Format::R8_UNORM, true).unwrap();
        let image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(8, 8));
        encoder.mipmaps.generate = false;
        encoder.write_surface(image.view()).unwrap(); // write surface

        let result = encoder.finish();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, EncodingError::MissingSurfaces));
        assert_eq!(err.to_string(), "Not enough surfaces have been written");
    }
}
