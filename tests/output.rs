use std::{
    fs::File,
    path::{Path, PathBuf},
};

use ddsd::*;

fn is_ci() -> bool {
    std::env::var("CI").is_ok()
}

fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("test-data");
    path
}

fn get_test_images() -> Vec<TestImage> {
    glob::glob(test_data_dir().join("images/**/*.dds").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .map(|x| x.unwrap())
        // ignore files starting with "_"
        .filter(|x| !x.file_name().unwrap().to_str().unwrap().starts_with('_'))
        .map(|x| TestImage { path: x })
        .collect()
}

struct TestImage {
    path: PathBuf,
}

#[test]
fn file_data_layout() {
    for test_image in get_test_images() {
        let mut file = File::open(&test_image.path).expect("Failed to open file");
        let file_len = file.metadata().unwrap().len();

        let decoder_result = DdsDecoder::new(&mut file);
        let decoder = match decoder_result {
            Ok(decoder) => decoder,
            Err(e) => panic!("Failed to decode {}\nFile: {:?}", e, file),
        };

        let header = decoder.header();
        let header_len = 4 + 124 + if header.dxt10.is_some() { 20 } else { 0 };
        let data_len = file_len - header_len;
        let expected_len = decoder.layout().data_len();
        assert_eq!(data_len, expected_len, "File: {:?}", &test_image.path);
    }
}

#[test]
fn output() {
    fn get_png_path(dds_path: &Path) -> PathBuf {
        test_data_dir()
            .join("output")
            .join(dds_path.parent().unwrap().file_name().unwrap())
            .join(dds_path.file_name().unwrap())
            .with_extension("png")
    }
    fn dds_to_png_8bit(
        dds_path: &PathBuf,
        png_path: &PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // read dds
        let mut file = File::open(dds_path)?;

        let decoder = DdsDecoder::new(&mut file)?;
        let size = decoder.header().size();
        let format = decoder.format();
        if !format.supported_precisions().contains(Precision::U8) {
            // don't care after DDS files we can't read as u8
            return Ok(());
        }

        let (channels, color) = match format.channels() {
            Channels::Grayscale => (Channels::Grayscale, png::ColorType::Grayscale),
            Channels::Alpha => (Channels::Rgba, png::ColorType::Rgba),
            Channels::Rgb => (Channels::Rgb, png::ColorType::Rgb),
            Channels::Rgba => (Channels::Rgba, png::ColorType::Rgba),
        };
        if !format.supported_channels().contains(channels) {
            // can't read in a way PNG likes
            return Err("Unsupported channels".into());
        }

        let mut image_data = vec![0_u8; size.pixels() as usize * channels.count() as usize];
        format.decode_u8(&mut file, size, channels, &mut image_data)?;

        // compare to PNG
        let png_exists = png_path.exists();
        if png_exists {
            let png_decoder = png::Decoder::new(File::open(png_path)?);
            let mut png_reader = png_decoder.read_info()?;
            let (png_color, png_bits) = png_reader.output_color_type();
            if png_bits != png::BitDepth::Eight {
                return Err("Output PNG is not 8-bit, which shouldn't happen.".into());
            }
            if png_color != color {
                eprintln!("Color mismatch: {:?} != {:?}", png_color, color);
            } else {
                assert!(png_reader.output_buffer_size() == image_data.len());
                let mut png_image_data = vec![0; image_data.len()];
                png_reader.next_frame(&mut png_image_data)?;
                png_reader.finish()?;

                if png_image_data == image_data {
                    // all good
                    return Ok(());
                }
            }
        }

        // write output PNG
        if !is_ci() {
            println!("Writing PNG: {:?}", png_path);
            let mut output = Vec::new();
            let mut encoder = png::Encoder::new(&mut output, size.width, size.height);
            encoder.set_color(color);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(&image_data)?;
            writer.finish()?;

            std::fs::create_dir_all(png_path.parent().unwrap())?;
            std::fs::write(png_path, output)?;
        }

        if !png_exists {
            return Err("Output PNG didn't exist".into());
        }
        Err("Output PNG didn't match".into())
    }

    let mut failed_count = 0;
    for test_image in get_test_images() {
        if let Err(e) = dds_to_png_8bit(&test_image.path, &get_png_path(&test_image.path)) {
            let path = test_image.path.strip_prefix(test_data_dir()).unwrap();
            eprintln!("Failed to convert {:?}: {}", path, e);
            failed_count += 1;
        }
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
}
