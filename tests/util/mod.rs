#![allow(unused)]

use bitflags::bitflags;
use dds::{header::*, *};
use rand::SeedableRng;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    io::Seek,
    path::{Path, PathBuf},
};
use zerocopy::{FromBytes, Immutable, IntoBytes};
use Precision::*;

mod data;
mod encode_quality;
mod image;
mod pretty_print;
mod snapshot;
mod table;

pub use data::*;
pub use encode_quality::*;
pub use image::*;
pub use pretty_print::*;
pub use snapshot::*;
pub use table::*;

pub fn create_rng() -> impl rand::Rng {
    rand_chacha::ChaChaRng::seed_from_u64(123456789)
}

pub fn hash_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let bytes: [u8; 32] = result.into();

    let mut hex = String::new();
    for byte in bytes.iter() {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}
pub fn hash_hex_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for f in data {
        let bytes: [u8; 4] = f.to_le_bytes();
        hasher.update(bytes);
    }
    let result = hasher.finalize();
    let bytes: [u8; 32] = result.into();

    let mut hex = String::new();
    for byte in bytes.iter() {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

pub trait Castable: FromBytes + IntoBytes + Immutable {}
impl<T: FromBytes + IntoBytes + Immutable> Castable for T {}
pub fn from_bytes<T: Castable>(bytes: &[u8]) -> Option<&[T]> {
    FromBytes::ref_from_bytes(bytes).ok()
}
pub fn from_bytes_mut<T: Castable>(bytes: &mut [u8]) -> Option<&mut [T]> {
    FromBytes::mut_from_bytes(bytes).ok()
}
pub fn as_bytes_mut<T: Castable>(buffer: &mut [T]) -> &mut [u8] {
    buffer.as_mut_bytes()
}
pub fn as_bytes<T: Castable>(buffer: &[T]) -> &[u8] {
    buffer.as_bytes()
}
pub fn cast_slice<T: Castable, U: Castable>(data: &[T]) -> &[U] {
    let data_bytes = as_bytes(data);
    from_bytes(data_bytes).unwrap()
}
pub fn cast_slice_mut<T: Castable, U: Castable>(data: &mut [T]) -> &mut [U] {
    let data_bytes = as_bytes_mut(data);
    from_bytes_mut(data_bytes).unwrap()
}

pub fn is_ci() -> bool {
    std::env::var("CI").is_ok()
}

pub fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("test-data");
    path
}

pub fn example_dds_files() -> Vec<PathBuf> {
    example_dds_files_in("**")
}
pub fn example_dds_files_in(parent_dir: &str) -> Vec<PathBuf> {
    glob::glob(
        test_data_dir()
            .join(format!("images/{parent_dir}/*.dds"))
            .to_str()
            .unwrap(),
    )
    .expect("Failed to read glob pattern")
    .map(|x| x.unwrap())
    // ignore files starting with "_"
    .filter(|x| !x.file_name().unwrap().to_str().unwrap().starts_with('_'))
    .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
impl Rect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn offset(&self) -> Offset {
        Offset::new(self.x, self.y)
    }
    pub const fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    /// Returns `true` if this rectangle is completely within the bounds of the
    /// given size.
    ///
    /// This means that `self.x + self.width <= size.width` and
    /// `self.y + self.height <= size.height`.
    pub(crate) fn is_within_bounds(&self, size: Size) -> bool {
        // use u64 to prevent overflow
        let end_x = self.x as u64 + self.width as u64;
        let end_y = self.y as u64 + self.height as u64;
        end_x <= size.width as u64 && end_y <= size.height as u64
    }
}

#[derive(Clone, Copy)]
pub struct ReadSettings {
    pub complete_cube_map: bool,
    pub is_bc3n: bool,
    pub force_channels: Option<Channels>,
    pub select_channels: Option<fn(Format) -> Channels>,
}
impl ReadSettings {
    pub fn any() -> Self {
        Self {
            complete_cube_map: true,
            is_bc3n: false,
            force_channels: None,
            select_channels: None,
        }
    }
    pub fn no_cube_map() -> Self {
        Self {
            complete_cube_map: false,
            is_bc3n: false,
            force_channels: None,
            select_channels: None,
        }
    }
    pub fn force(channels: Channels) -> Self {
        Self {
            complete_cube_map: false,
            is_bc3n: false,
            force_channels: Some(channels),
            select_channels: None,
        }
    }

    pub fn allow_cube_map(&self) -> bool {
        self.complete_cube_map && self.force_channels.is_none()
    }
    pub fn pick_channels(&self, format: Format) -> Channels {
        if let Some(channels) = self.force_channels {
            return channels;
        }
        if let Some(f) = self.select_channels {
            return f(format);
        }
        format.channels()
    }
}

#[derive(Debug, Clone)]
pub struct DdsInfo {
    pub header: Header,
    pub format: Format,
    pub layout: DataLayout,
}
impl DdsInfo {
    pub fn from_decoder<R>(decoder: &Decoder<R>) -> Self {
        Self {
            header: decoder.header().clone(),
            format: decoder.format(),
            layout: decoder.layout(),
        }
    }
}

pub fn read_dds<T: WithPrecision + Default + Copy + Castable>(
    dds_path: &Path,
) -> Result<(Image<T>, DdsInfo), Box<dyn std::error::Error>> {
    read_dds_with_settings(dds_path, ReadSettings::any())
}
pub fn read_dds_with_channels<T: WithPrecision + Default + Copy + Castable>(
    dds_path: &Path,
    channels: Channels,
) -> Result<(Image<T>, DdsInfo), Box<dyn std::error::Error>> {
    read_dds_with_settings(dds_path, ReadSettings::force(channels))
}
pub fn read_dds_with_settings<T: WithPrecision + Default + Copy + Castable>(
    dds_path: &Path,
    settings: ReadSettings,
) -> Result<(Image<T>, DdsInfo), Box<dyn std::error::Error>> {
    let mut file = File::open(dds_path)?;

    let options = ParseOptions::new_permissive(Some(file.metadata()?.len()));
    decode_dds_with_settings(&options, &mut file, settings)
}

pub fn decode_dds_with_channels<T: WithPrecision + Default + Copy + Castable>(
    options: &ParseOptions,
    reader: impl std::io::Read + Seek,
    channels: Channels,
) -> Result<(Image<T>, DdsInfo), Box<dyn std::error::Error>> {
    decode_dds_with_settings(options, reader, ReadSettings::force(channels))
}
pub fn decode_dds_with_settings<T: WithPrecision + Default + Copy + Castable>(
    options: &ParseOptions,
    mut reader: impl std::io::Read + Seek,
    settings: ReadSettings,
) -> Result<(Image<T>, DdsInfo), Box<dyn std::error::Error>> {
    let header = Header::read(&mut reader, options)?;
    let mut format = Format::from_header(&header)?;
    if format == Format::BC3_UNORM && settings.is_bc3n {
        format = Format::BC3_UNORM_NORMAL;
    }
    let mut decoder = Decoder::from_header_with(reader, header, format)?;
    let size = decoder.main_size();

    if let Some(array) = decoder.layout().texture_array() {
        if array.kind() == TextureArrayKind::CubeMaps && settings.allow_cube_map() {
            let out_size = Size::new(size.width * 4, size.height * 3);
            let mut image = Image::new_empty(Channels::Rgba, out_size);
            decoder.read_cube_map(image.view_mut())?;

            return Ok((image, DdsInfo::from_decoder(&decoder)));
        }
    };

    let channels = settings.pick_channels(format);
    let mut image = Image::new_empty(channels, size);
    decoder.read_surface(image.view_mut())?;

    Ok((image, DdsInfo::from_decoder(&decoder)))
}

pub fn read_dds_png_compatible(
    dds_path: &Path,
) -> Result<(Image<u8>, DdsInfo), Box<dyn std::error::Error>> {
    let is_bc3n = dds_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_ascii_lowercase()
        .contains("bc3n");
    read_dds_with_settings(
        dds_path,
        ReadSettings {
            complete_cube_map: true,
            is_bc3n,
            force_channels: None,
            select_channels: Some(|f| to_png_compatible_channels(f.channels()).0),
        },
    )
}

pub fn read_dds_rect_as_u8(
    dds_path: &Path,
    rect: Rect,
) -> Result<(Image<u8>, DdsInfo), Box<dyn std::error::Error>> {
    // read dds
    let mut file = File::open(dds_path)?;
    let mut decoder = Decoder::new(file)?;

    let channels = to_png_compatible_channels(decoder.format().channels()).0;

    let mut image = Image::new_empty(channels, rect.size());
    decoder.read_surface_rect(image.view_mut(), rect.offset())?;

    Ok((image, DdsInfo::from_decoder(&decoder)))
}

pub fn to_png_compatible_channels(channels: Channels) -> (Channels, png::ColorType) {
    match channels {
        Channels::Grayscale => (Channels::Grayscale, png::ColorType::Grayscale),
        Channels::Alpha => (Channels::Rgba, png::ColorType::Rgba),
        Channels::Rgb => (Channels::Rgb, png::ColorType::Rgb),
        Channels::Rgba => (Channels::Rgba, png::ColorType::Rgba),
    }
}

pub fn read_png_u16(png_path: &Path) -> Result<Image<u16>, Box<dyn std::error::Error>> {
    let png_decoder = png::Decoder::new(File::open(png_path)?);
    let mut png_reader = png_decoder.read_info()?;
    let (color, bits) = png_reader.output_color_type();

    let channels = match color {
        png::ColorType::Grayscale => Channels::Grayscale,
        png::ColorType::Rgb => Channels::Rgb,
        png::ColorType::Rgba => Channels::Rgba,
        _ => return Err("Unsupported PNG color type".into()),
    };

    match bits {
        png::BitDepth::Sixteen => {
            let mut png_image_data: Vec<u16> = vec![0; png_reader.output_buffer_size() / 2];
            png_reader.next_frame(cast_slice_mut(&mut png_image_data))?;
            png_reader.finish()?;

            png_image_data.iter_mut().for_each(|v| *v = v.to_be());

            Ok(Image::new(
                png_image_data,
                channels,
                Size::new(png_reader.info().width, png_reader.info().height),
            ))
        }
        png::BitDepth::Eight => {
            let mut png_image_data = vec![0; png_reader.output_buffer_size()];
            png_reader.next_frame(&mut png_image_data)?;
            png_reader.finish()?;

            let image_u8 = Image::new(
                png_image_data,
                channels,
                Size::new(png_reader.info().width, png_reader.info().height),
            );

            Ok(image_u8.to_u16())
        }
        _ => Err("Output PNG is not 8/16-bit, which shouldn't happen.".into()),
    }
}
pub fn read_png_u8(png_path: &Path) -> Result<Image<u8>, Box<dyn std::error::Error>> {
    Ok(read_png_u16(png_path)?.to_u8())
}
pub fn read_png_f32(png_path: &Path) -> Result<Image<f32>, Box<dyn std::error::Error>> {
    Ok(read_png_u16(png_path)?.to_f32())
}

pub fn write_simple_dds_header(
    w: &mut impl std::io::Write,
    size: Size,
    format: DxgiFormat,
) -> std::io::Result<()> {
    let mut header = Dx10Header::new_image(size.width, size.height, format);
    header.alpha_mode = AlphaMode::Unknown;

    Header::from(header).write(w)?;

    Ok(())
}

pub struct OutputSummaries {
    name: String,
    by_folder: HashMap<PathBuf, String>,
}
impl OutputSummaries {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            by_folder: HashMap::new(),
        }
    }
    pub fn add_output_file(&mut self, file_path: &Path, info: &str) {
        let folder = file_path.parent().unwrap().to_path_buf();
        let name = file_path.file_name().unwrap().to_string_lossy();

        let mut lines = String::new();
        lines.push_str(&format!("{name}: >\n"));
        for l in info.lines().map(|l| l.trim_end()) {
            if l.is_empty() {
                lines.push('\n');
            } else {
                lines.push_str(&format!("    {l}\n"));
            }
        }
        lines.push('\n');

        self.by_folder.entry(folder).or_default().push_str(&lines);
    }
    pub fn add_output_file_error<E: std::error::Error + ?Sized>(
        &mut self,
        file_path: &Path,
        error: &E,
    ) {
        eprintln!("Failed for: {file_path:?}");
        eprintln!("Error: {error}");

        self.add_output_file(file_path, &format!("Error: {error}"));
    }
    pub fn add_output_file_result(
        &mut self,
        file_path: &Path,
        result: Result<String, Box<dyn std::error::Error>>,
    ) {
        match result {
            Ok(info) => self.add_output_file(file_path, &info),
            Err(e) => self.add_output_file_error(file_path, &*e),
        }
    }

    pub fn snapshot(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut result = Ok(());
        for (folder, text) in &self.by_folder {
            let path = folder.join(format!("{}.yml", self.name));
            let r = TextSnapshot.result(&path, text);
            if r.is_err() {
                result = r;
            }
        }
        result
    }
    #[track_caller]
    pub fn snapshot_or_fail(&self) {
        if let Err(e) = self.snapshot() {
            panic!("Some tests failed: {e}");
        }
    }
}

pub fn indent(indent: &str, text: &str) -> String {
    let mut result = String::new();
    for line in text.lines() {
        if line.is_empty() {
            result.push('\n');
        } else {
            result.push_str(indent);
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}
