use dds::{header::*, *};
use rand::Rng;

mod util;

/// Whether the test will include a timestamp for each progress report.
///
/// This is useful for debugging and improving the frequency of progress
/// reports.
///
/// This has to be `false` for CI, as timestamps are not deterministic.
const LOG_TIME: bool = false;

// Don't run this on big endian targets, it's just too slow
#[cfg(not(target_endian = "big"))]
// Don't run when doing code coverage, it's just too slow
#[cfg(not(coverage))]
#[test]
fn track_progress() {
    use util::Image;
    use Channels::*;

    let mut rng = util::create_rng();

    let mut image_rgba_u8: Image<u8> = Image::new_empty(Rgba, Size::new(4096, 2048));
    rng.fill(image_rgba_u8.data.as_mut_slice());
    let mut image_gray_f32: Image<f32> = Image::new_empty(Grayscale, Size::new(1024, 1024));
    rng.fill(image_gray_f32.data.as_mut_slice());

    let mut options = EncodeOptions::default();
    options.quality = CompressionQuality::Fast;
    options.parallel = false; // progress with multiple threads is not deterministic

    let formats = [
        Format::R8_UNORM,
        Format::B5G6R5_UNORM,
        Format::R8G8B8A8_UNORM,
        Format::R16G16B16A16_SNORM,
        Format::R16G16B16A16_FLOAT,
        Format::R9G9B9E5_SHAREDEXP,
        Format::BC1_UNORM,
        Format::BC4_UNORM,
        Format::R1_UNORM,
        Format::R8G8_B8G8_UNORM,
        Format::Y210,
        Format::AYUV,
        Format::NV12,
    ];

    fn test(
        format: Format,
        options: &EncodeOptions,
        image: ImageView,
        mipmaps: bool,
    ) -> Result<String, EncodeError> {
        let mut progress_report = String::new();
        let start_time = std::time::Instant::now();
        let log_time = |progress_report: &mut String| {
            if LOG_TIME {
                let ms = start_time.elapsed().as_secs_f64() * 1000.0;
                progress_report.push_str(&format!("{ms:.2}ms: "));
            }
        };

        let mut consume_progress = |progress| {
            log_time(&mut progress_report);
            progress_report.push_str(&format!("{:.2}%\n", progress * 100.0));
        };
        let mut progress = Progress::new(&mut consume_progress);

        let mut header = Header::new_image(image.width(), image.height(), format);
        if mipmaps && format.encoding_support().unwrap().size_multiple() == SizeMultiple::ONE {
            header = header.with_mipmaps();
        }

        let mut encoder = Encoder::new(std::io::sink(), format, &header)?;
        encoder.options = options.clone();

        encoder.write_surface_with(
            image,
            Some(&mut progress),
            &WriteOptions {
                generate_mipmaps: true,
                ..Default::default()
            },
        )?;
        encoder.finish()?;

        log_time(&mut progress_report);
        progress_report.push_str("Done.\n");

        Ok(progress_report)
    }
    fn add_to_output(
        output: &mut String,
        format: Format,
        options: &EncodeOptions,
        image: ImageView,
        mipmaps: bool,
    ) {
        let info = match test(format, options, image, mipmaps) {
            Ok(info) => info,
            Err(e) => format!(
                "Failed to encode {format:?} with dither {0:?}: {e}",
                options.dithering
            ),
        };

        let width = image.width();
        let height = image.height();
        let color = image.color();
        output.push_str(&format!(
            "  \"Dither {0:?}: {color} Image {width}x{height}\": |\n",
            options.dithering
        ));
        output.push_str(&util::indent("    ", &info));
        output.push('\n');
    }

    let output = &mut String::new();

    output.push_str("With mipmaps:\n");
    add_to_output(
        output,
        Format::R8G8B8A8_UNORM,
        &options,
        image_rgba_u8.view(),
        true,
    );
    output.push_str("\n\n");

    for format in formats {
        output.push_str(&format!("{format:?}:\n"));

        let supports_dither = format
            .encoding_support()
            .map_or(false, |e| e.dithering() != Dithering::None);
        let is_slow_format = format!("{:?}", format).contains("BC");

        for dither in [Dithering::None, Dithering::ColorAndAlpha] {
            if dither != Dithering::None && !supports_dither {
                continue;
            }

            options.dithering = dither;
            for image in [image_rgba_u8.view(), image_gray_f32.view()] {
                if is_slow_format && image.width() > 1024 {
                    continue;
                }

                add_to_output(output, format, &options, image, false);
            }
        }
        output.push_str("\n\n");
    }

    let snapshot_file = util::test_data_dir().join("progress_snapshot.yml");
    util::compare_snapshot_text(&snapshot_file, output).unwrap();
}

// Don't run this on big endian targets, it's just too slow
#[cfg(not(target_endian = "big"))]
// Don't run when doing code coverage, it's just too slow
#[cfg(not(coverage))]
// This tests that progress only ever increases
#[test]
fn forward_progress() {
    use util::Image;
    use Channels::*;

    let mut rng = util::create_rng();

    let mut image_rgba_u8: Image<u8> = Image::new_empty(Rgba, Size::new(1024, 1024));
    rng.fill(image_rgba_u8.data.as_mut_slice());
    let mut image_gray_f32: Image<f32> = Image::new_empty(Grayscale, Size::new(1024, 1024));
    rng.fill(image_gray_f32.data.as_mut_slice());

    let mut options = EncodeOptions::default();
    options.quality = CompressionQuality::Fast;
    options.parallel = true;

    let formats = [Format::BC1_UNORM, Format::BC4_UNORM];

    fn test(format: Format, options: &EncodeOptions, image: ImageView) -> Result<(), EncodeError> {
        let mut last_progress = 0.0;
        let mut consume_progress = |progress| {
            assert!(progress >= last_progress);
            last_progress = progress;
        };
        let mut progress = Progress::new(&mut consume_progress);

        let mut header = Header::new_image(image.width(), image.height(), format);
        if format.encoding_support().unwrap().size_multiple() == SizeMultiple::ONE {
            header = header.with_mipmaps();
        }

        let mut encoder = Encoder::new(std::io::sink(), format, &header)?;
        encoder.options = options.clone();

        encoder.write_surface_with(
            image,
            Some(&mut progress),
            &WriteOptions {
                generate_mipmaps: true,
                ..Default::default()
            },
        )?;
        encoder.finish()?;

        Ok(())
    }

    for format in formats {
        let supports_dither = format
            .encoding_support()
            .map_or(false, |e| e.dithering() != Dithering::None);
        for dither in [Dithering::None, Dithering::ColorAndAlpha] {
            if dither != Dithering::None && !supports_dither {
                continue;
            }

            options.dithering = dither;
            for image in [image_rgba_u8.view(), image_gray_f32.view()] {
                if let Err(e) = test(format, &options, image) {
                    panic!("Failed to encode {format:?} with dither {dither:?}: {e}");
                }
            }
        }
    }
}
