use ddsd::*;
use Precision::*;

use std::{fs::File, path::PathBuf};

pub fn is_ci() -> bool {
    std::env::var("CI").is_ok()
}

pub fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("test-data");
    path
}

pub struct Image {
    pub data: Vec<u8>,
    pub channels: Channels,
    pub size: Size,
}

pub fn read_dds_as_u8(
    dds_path: &PathBuf,
) -> Result<(Image, DdsDecoder), Box<dyn std::error::Error>> {
    // read dds
    let mut file = File::open(dds_path)?;

    let decoder = DdsDecoder::new(&mut file)?;
    let size = decoder.header().size();
    let format = decoder.format();
    if !format.supported_precisions().contains(Precision::U8) {
        return Err("Format does not support decoding as U8".into());
    }

    let channels = to_png_compatible_channels(format.channels()).0;
    if !format.supported_channels().contains(channels) {
        // can't read in a way PNG likes
        return Err("Unsupported channels".into());
    }

    let mut image_data = vec![0_u8; size.pixels() as usize * channels.count() as usize];
    format.decode_u8(&mut file, size, channels, &mut image_data)?;

    let image = Image {
        data: image_data,
        channels,
        size,
    };
    Ok((image, decoder))
}

pub fn read_dds_rect_as_u8(
    dds_path: &PathBuf,
    rect: Rect,
) -> Result<(Image, DdsDecoder), Box<dyn std::error::Error>> {
    // read dds
    let mut file = File::open(dds_path)?;

    let decoder = DdsDecoder::new(&mut file)?;
    let size = decoder.header().size();
    let format = decoder.format();
    if !format.supported_precisions().contains(Precision::U8) {
        return Err("Format does not support decoding as U8".into());
    }

    let channels = to_png_compatible_channels(format.channels()).0;
    if !format.supported_channels().contains(channels) {
        // can't read in a way PNG likes
        return Err("Unsupported channels".into());
    }

    let color = ColorFormat::new(channels, U8);
    let bpp = color.bytes_per_pixel() as usize;
    let mut image_data = vec![0_u8; rect.size().pixels() as usize * bpp];
    format.decode_rect(
        &mut file,
        size,
        rect,
        color,
        &mut image_data,
        rect.width as usize * bpp,
    )?;

    let image = Image {
        data: image_data,
        channels,
        size: rect.size(),
    };
    Ok((image, decoder))
}

pub fn to_png_compatible_channels(channels: Channels) -> (Channels, png::ColorType) {
    match channels {
        Channels::Grayscale => (Channels::Grayscale, png::ColorType::Grayscale),
        Channels::Alpha => (Channels::Rgba, png::ColorType::Rgba),
        Channels::Rgb => (Channels::Rgb, png::ColorType::Rgb),
        Channels::Rgba => (Channels::Rgba, png::ColorType::Rgba),
    }
}

pub fn compare_snapshot_png_u8(
    png_path: &PathBuf,
    image: &Image,
) -> Result<(), Box<dyn std::error::Error>> {
    let (channels, color) = to_png_compatible_channels(image.channels);
    assert!(channels == image.channels);

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
            assert!(png_reader.output_buffer_size() == image.data.len());
            let mut png_image_data = vec![0; image.data.len()];
            png_reader.next_frame(&mut png_image_data)?;
            png_reader.finish()?;

            if png_image_data == image.data {
                // all good
                return Ok(());
            }
        }
    }

    // write output PNG
    if !is_ci() {
        println!("Writing PNG: {:?}", png_path);
        let mut output = Vec::new();
        let mut encoder = png::Encoder::new(&mut output, image.size.width, image.size.height);
        encoder.set_color(color);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&image.data)?;
        writer.finish()?;

        std::fs::create_dir_all(png_path.parent().unwrap())?;
        std::fs::write(png_path, output)?;
    }

    if !png_exists {
        return Err("Output PNG didn't exist".into());
    }
    Err("Output PNG didn't match".into())
}
