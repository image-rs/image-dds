use dds::*;
use std::{fs::File, path::Path};

use crate::util::{
    convert_channels, is_ci, read_png_u8, to_png_compatible_channels, write_simple_dds_header,
    Image,
};

#[must_use]
pub enum UpdateResult {
    Unchanged,
    Created,
    Mismatch,
}
impl UpdateResult {
    pub fn result(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            UpdateResult::Unchanged => Ok(()),
            UpdateResult::Created => Err("Snapshot didn't exist".into()),
            UpdateResult::Mismatch => Err("Snapshot mismatch".into()),
        }
    }
}

pub trait Snapshot<T>
where
    T: ?Sized,
{
    /// Writes the snapshot data to `path` and returns whether it was unchanged, created or mismatched.
    ///
    /// This will returns an error if an IO error or encoding error occurs.
    fn write(&self, path: &Path, data: &T) -> Result<UpdateResult, Box<dyn std::error::Error>>;

    /// Compares the given data with the snapshot file.
    ///
    /// Returns `Ok(())` if the snapshot matches, otherwise an error.
    fn result(&self, path: &Path, data: &T) -> Result<(), Box<dyn std::error::Error>> {
        match self.write(path, data)? {
            UpdateResult::Unchanged => Ok(()),
            UpdateResult::Created => {
                Err(format!("Snapshot didn't exist: {}", path.display()).into())
            }
            UpdateResult::Mismatch => Err(format!("Snapshot mismatch: {}", path.display()).into()),
        }
    }

    /// Returns normally if the snapshot matches, otherwise panics.
    #[track_caller]
    fn assert(&self, path: &Path, data: &T) {
        match self.write(path, data) {
            Ok(UpdateResult::Unchanged) => {}
            Ok(UpdateResult::Created) => {
                panic!("Snapshot didn't exist: {}", path.display())
            }
            Ok(UpdateResult::Mismatch) => panic!("Snapshot mismatch: {}", path.display()),
            Err(e) => panic!(
                "Failed to compare snapshot:\nFile:{}\nError:{:?}",
                path.display(),
                e
            ),
        }
    }
}

pub struct TextSnapshot;
impl Snapshot<str> for TextSnapshot {
    fn write(&self, path: &Path, data: &str) -> Result<UpdateResult, Box<dyn std::error::Error>> {
        // normalize line endings to LF
        let text = data.replace("\r\n", "\n");

        // compare to snapshot
        let snapshot = std::fs::read_to_string(path);
        let file_exists = snapshot.is_ok();
        let mut native_line_ends = "\n";

        if let Ok(mut snapshot) = snapshot {
            if snapshot.contains("\r\n") {
                native_line_ends = "\r\n";
                snapshot = snapshot.replace("\r\n", "\n");
            }

            if text.trim() == snapshot.trim() {
                // all ok
                return Ok(UpdateResult::Unchanged);
            }
        }

        // write snapshot
        if !is_ci() {
            println!("Writing snapshot: {path:?}");

            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(path, text.replace("\n", native_line_ends))?;
        }

        if !file_exists {
            Ok(UpdateResult::Created)
        } else {
            Ok(UpdateResult::Mismatch)
        }
    }
}

pub struct PngSnapshot;
impl Snapshot<Image<u8>> for PngSnapshot {
    fn write(
        &self,
        png_path: &Path,
        image: &Image<u8>,
    ) -> Result<UpdateResult, Box<dyn std::error::Error>> {
        let (channels, color) = to_png_compatible_channels(image.channels);
        assert!(channels == image.channels);

        // compare to PNG
        let png_exists = png_path.exists();
        if png_exists {
            let mut png = read_png_u8(png_path)?;

            if image.channels == Channels::Rgba && png.channels == Channels::Rgb {
                // convert to RGBA
                png.data = convert_channels(&png.data, Channels::Rgb, Channels::Rgba);
                png.channels = Channels::Rgba;
            }

            if png.data == image.data {
                // all good
                return Ok(UpdateResult::Unchanged);
            }

            if image.channels != png.channels {
                eprintln!("Color mismatch: {:?} != {:?}", png.channels, image.channels);
            }
        }

        // write output PNG
        if !is_ci() {
            println!("Writing PNG: {png_path:?}");
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
            Ok(UpdateResult::Created)
        } else {
            Ok(UpdateResult::Mismatch)
        }
    }
}

pub struct DdsF32Snapshot;
impl Snapshot<Image<f32>> for DdsF32Snapshot {
    fn write(
        &self,
        path: &Path,
        image: &Image<f32>,
    ) -> Result<UpdateResult, Box<dyn std::error::Error>> {
        // compare to DDS
        let dds_exists = path.exists();
        if dds_exists {
            let mut file = File::open(path)?;
            let mut decoder = Decoder::new(file)?;
            let size = decoder.main_size();

            let mut dds_image: Image<f32> = Image::new_empty(image.channels, size);
            decoder.read_surface(dds_image.view_mut())?;

            assert!(dds_image.data.len() == image.data.len());
            if dds_image.data == image.data {
                // all good
                return Ok(UpdateResult::Unchanged);
            }
        }

        // write output DDS
        if !is_ci() {
            println!("Writing DDS: {path:?}");

            let format = match image.channels {
                Channels::Grayscale => Format::R32_FLOAT,
                Channels::Alpha => Format::R32_FLOAT,
                Channels::Rgb => Format::R32G32B32_FLOAT,
                Channels::Rgba => Format::R32G32B32A32_FLOAT,
            };
            let mut output = Vec::new();
            write_simple_dds_header(&mut output, image.size, format.try_into().unwrap())?;

            // convert to LE
            encode(
                &mut output,
                image.view(),
                format,
                None,
                &EncodeOptions::default(),
            )?;

            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(path, output)?;
        }

        if !dds_exists {
            Ok(UpdateResult::Created)
        } else {
            Ok(UpdateResult::Mismatch)
        }
    }
}
