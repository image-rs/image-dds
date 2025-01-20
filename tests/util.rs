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

pub struct Image<T> {
    pub data: Vec<T>,
    pub channels: Channels,
    pub size: Size,
}
impl<T: WithPrecision> Image<T> {
    pub fn precision(&self) -> Precision {
        T::PRECISION
    }
    pub fn color(&self) -> ColorFormat {
        ColorFormat::new(self.channels, T::PRECISION)
    }
}

pub trait WithPrecision {
    const PRECISION: Precision;
}
impl WithPrecision for u8 {
    const PRECISION: Precision = U8;
}
impl WithPrecision for u16 {
    const PRECISION: Precision = U16;
}
impl WithPrecision for f32 {
    const PRECISION: Precision = F32;
}

pub fn read_dds<T: WithPrecision + Default + Copy + bytemuck::Pod>(
    dds_path: &PathBuf,
) -> Result<(Image<T>, DdsDecoder), Box<dyn std::error::Error>> {
    read_dds_with_channels_select(dds_path, |f| f.channels())
}
pub fn read_dds_with_channels<T: WithPrecision + Default + Copy + bytemuck::Pod>(
    dds_path: &PathBuf,
    channels: Channels,
) -> Result<(Image<T>, DdsDecoder), Box<dyn std::error::Error>> {
    read_dds_with_channels_select(dds_path, |_| channels)
}
pub fn read_dds_with_channels_select<T: WithPrecision + Default + Copy + bytemuck::Pod>(
    dds_path: &PathBuf,
    select_channels: impl FnOnce(SupportedFormat) -> Channels,
) -> Result<(Image<T>, DdsDecoder), Box<dyn std::error::Error>> {
    let mut file = File::open(dds_path)?;

    let decoder = DdsDecoder::new(&mut file)?;
    let size = decoder.header().size();
    let format = decoder.format();
    if !format.supported_precisions().contains(T::PRECISION) {
        return Err(format!("Format does not support decoding as {:?}", T::PRECISION).into());
    }

    let channels = select_channels(format);
    if !format.supported_channels().contains(channels) {
        // can't read in a way PNG likes
        return Err("Unsupported channels".into());
    }

    let mut image_data = vec![T::default(); size.pixels() as usize * channels.count() as usize];
    let image_data_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut image_data);
    format.decode(
        &mut file,
        size,
        ColorFormat::new(channels, T::PRECISION),
        image_data_bytes,
    )?;

    let image = Image {
        data: image_data,
        channels,
        size,
    };
    Ok((image, decoder))
}

pub fn read_dds_png_compatible(
    dds_path: &PathBuf,
) -> Result<(Image<u8>, DdsDecoder), Box<dyn std::error::Error>> {
    read_dds_with_channels_select(dds_path, |f| to_png_compatible_channels(f.channels()).0)
}

pub fn read_dds_rect_as_u8(
    dds_path: &PathBuf,
    rect: Rect,
) -> Result<(Image<u8>, DdsDecoder), Box<dyn std::error::Error>> {
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
    image: &Image<u8>,
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
