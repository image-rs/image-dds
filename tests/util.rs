#![allow(unused)]

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

pub const ALL_FORMATS: &[Format] = &[
    // uncompressed formats
    Format::R8G8B8_UNORM,
    Format::B8G8R8_UNORM,
    Format::R8G8B8A8_UNORM,
    Format::R8G8B8A8_SNORM,
    Format::B8G8R8A8_UNORM,
    Format::B8G8R8X8_UNORM,
    Format::B5G6R5_UNORM,
    Format::B5G5R5A1_UNORM,
    Format::B4G4R4A4_UNORM,
    Format::A4B4G4R4_UNORM,
    Format::R8_SNORM,
    Format::R8_UNORM,
    Format::R8G8_UNORM,
    Format::R8G8_SNORM,
    Format::A8_UNORM,
    Format::R16_UNORM,
    Format::R16_SNORM,
    Format::R16G16_UNORM,
    Format::R16G16_SNORM,
    Format::R16G16B16A16_UNORM,
    Format::R16G16B16A16_SNORM,
    Format::R10G10B10A2_UNORM,
    Format::R11G11B10_FLOAT,
    Format::R9G9B9E5_SHAREDEXP,
    Format::R16_FLOAT,
    Format::R16G16_FLOAT,
    Format::R16G16B16A16_FLOAT,
    Format::R32_FLOAT,
    Format::R32G32_FLOAT,
    Format::R32G32B32_FLOAT,
    Format::R32G32B32A32_FLOAT,
    Format::R10G10B10_XR_BIAS_A2_UNORM,
    Format::AYUV,
    Format::Y410,
    Format::Y416,
    // sub-sampled formats
    Format::R1_UNORM,
    Format::R8G8_B8G8_UNORM,
    Format::G8R8_G8B8_UNORM,
    Format::UYVY,
    Format::YUY2,
    Format::Y210,
    Format::Y216,
    // bi-planar formats
    Format::NV12,
    Format::P010,
    Format::P016,
    // block compression formats
    Format::BC1_UNORM,
    Format::BC2_UNORM,
    Format::BC2_UNORM_PREMULTIPLIED_ALPHA,
    Format::BC3_UNORM,
    Format::BC3_UNORM_PREMULTIPLIED_ALPHA,
    Format::BC4_UNORM,
    Format::BC4_SNORM,
    Format::BC5_UNORM,
    Format::BC5_SNORM,
    Format::BC6H_UF16,
    Format::BC6H_SF16,
    Format::BC7_UNORM,
    // ASTC formats
    Format::ASTC_4X4_UNORM,
    Format::ASTC_5X4_UNORM,
    Format::ASTC_5X5_UNORM,
    Format::ASTC_6X5_UNORM,
    Format::ASTC_6X6_UNORM,
    Format::ASTC_8X5_UNORM,
    Format::ASTC_8X6_UNORM,
    Format::ASTC_8X8_UNORM,
    Format::ASTC_10X5_UNORM,
    Format::ASTC_10X6_UNORM,
    Format::ASTC_10X8_UNORM,
    Format::ASTC_10X10_UNORM,
    Format::ASTC_12X10_UNORM,
    Format::ASTC_12X12_UNORM,
    // non-standard formats
    Format::BC3_UNORM_RXGB,
    Format::BC3_UNORM_NORMAL,
];

pub const ALL_COLORS: &[ColorFormat] = &[
    ColorFormat::ALPHA_U8,
    ColorFormat::GRAYSCALE_U8,
    ColorFormat::RGB_U8,
    ColorFormat::RGBA_U8,
    ColorFormat::ALPHA_U16,
    ColorFormat::GRAYSCALE_U16,
    ColorFormat::RGB_U16,
    ColorFormat::RGBA_U16,
    ColorFormat::ALPHA_F32,
    ColorFormat::GRAYSCALE_F32,
    ColorFormat::RGB_F32,
    ColorFormat::RGBA_F32,
];

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Image<T> {
    pub data: Vec<T>,
    pub channels: Channels,
    pub size: Size,
}
impl<T> Image<T> {
    #[track_caller]
    pub fn new(data: Vec<T>, channels: Channels, size: Size) -> Self {
        assert_eq!(
            data.len(),
            size.pixels() as usize * channels.count() as usize,
            "Data size doesn't match image size"
        );

        Self {
            data,
            channels,
            size,
        }
    }
    pub fn new_empty(channels: Channels, size: Size) -> Self
    where
        T: Default + Copy,
    {
        let data = vec![T::default(); size.pixels() as usize * channels.count() as usize];
        Self::new(data, channels, size)
    }

    pub fn stride(&self) -> usize {
        self.size.width as usize * self.channels.count() as usize * std::mem::size_of::<T>()
    }

    pub fn as_bytes(&self) -> &[u8]
    where
        T: Castable,
    {
        as_bytes(&self.data)
    }
    pub fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        T: Castable,
    {
        as_bytes_mut(&mut self.data)
    }

    pub fn precision(&self) -> Precision
    where
        T: WithPrecision,
    {
        T::PRECISION
    }
    pub fn color(&self) -> ColorFormat
    where
        T: WithPrecision,
    {
        ColorFormat::new(self.channels, T::PRECISION)
    }

    pub fn view(&self) -> ImageView
    where
        T: Castable + WithPrecision,
    {
        ImageView::new(self.as_bytes(), self.size, self.color()).unwrap()
    }
    pub fn view_mut(&mut self) -> ImageViewMut
    where
        T: Castable + WithPrecision,
    {
        let size = self.size;
        let color = self.color();
        ImageViewMut::new(self.as_bytes_mut(), size, color).unwrap()
    }

    pub fn to_channels(&self, channels: Channels) -> Image<T>
    where
        T: Copy + Default + Castable + Norm,
    {
        Image::new(
            convert_channels(&self.data, self.channels, channels),
            channels,
            self.size,
        )
    }

    pub fn cropped(&self, new_size: Size) -> Image<T>
    where
        T: Copy,
    {
        if new_size == self.size {
            return self.clone();
        }
        assert!(new_size.width <= self.size.width);
        assert!(new_size.height <= self.size.height);

        let new_width = new_size.width as usize;
        let new_height = new_size.height as usize;
        let new_stride = new_width * self.channels.count() as usize;

        let mut new_data = Vec::with_capacity(new_stride * new_height);
        for y in 0..new_height {
            let src_offset = y * self.size.width as usize * self.channels.count() as usize;
            let dst_offset = y * new_stride;
            new_data.extend_from_slice(&self.data[src_offset..src_offset + new_stride]);
        }

        Image::new(new_data, self.channels, new_size)
    }
}
impl Image<u8> {
    pub fn to_u16(&self) -> Image<u16> {
        Image::new(
            self.data.iter().map(|&x| x as u16 * 257).collect(),
            self.channels,
            self.size,
        )
    }
    pub fn to_f32(&self) -> Image<f32> {
        Image::new(
            self.data.iter().map(|&x| x as f32 / 255.0).collect(),
            self.channels,
            self.size,
        )
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
    let color = ColorFormat::new(channels, U8);
    let bpp = color.bytes_per_pixel() as usize;

    let mut image = Image::new_empty(channels, rect.size());
    let row_pitch = rect.width as usize * bpp;
    decoder.read_surface_rect(image.as_bytes_mut(), row_pitch, rect, color)?;

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

pub fn read_png_u8(png_path: &Path) -> Result<Image<u8>, Box<dyn std::error::Error>> {
    let png_decoder = png::Decoder::new(File::open(png_path)?);
    let mut png_reader = png_decoder.read_info()?;
    let (color, bits) = png_reader.output_color_type();

    if bits != png::BitDepth::Eight {
        return Err("Output PNG is not 8-bit, which shouldn't happen.".into());
    }
    let channels = match color {
        png::ColorType::Grayscale => Channels::Grayscale,
        png::ColorType::Rgb => Channels::Rgb,
        png::ColorType::Rgba => Channels::Rgba,
        _ => return Err("Unsupported PNG color type".into()),
    };

    let mut png_image_data = vec![0; png_reader.output_buffer_size()];
    png_reader.next_frame(&mut png_image_data)?;
    png_reader.finish()?;

    Ok(Image::new(
        png_image_data,
        channels,
        Size::new(png_reader.info().width, png_reader.info().height),
    ))
}

pub fn compare_snapshot_png_u8(
    png_path: &Path,
    image: &Image<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    match update_snapshot_png_u8(png_path, image)? {
        UpdateResult::Unchanged => Ok(()),
        UpdateResult::Created => Err("Output PNG didn't exist".into()),
        UpdateResult::Updated => Err("Output PNG didn't match".into()),
    }
}
pub enum UpdateResult {
    Unchanged,
    Created,
    Updated,
}
pub fn update_snapshot_png_u8(
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
        return Ok(UpdateResult::Created);
    }
    Ok(UpdateResult::Updated)
}

pub fn compare_snapshot_dds_f32(
    dds_path: &Path,
    image: &Image<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // compare to DDS
    let dds_exists = dds_path.exists();
    if dds_exists {
        let mut file = File::open(dds_path)?;
        let mut decoder = Decoder::new(file)?;
        let size = decoder.main_size();

        let mut dds_image: Image<f32> = Image::new_empty(image.channels, size);
        decoder.read_surface(dds_image.view_mut())?;

        assert!(dds_image.data.len() == image.data.len());
        if dds_image.data == image.data {
            // all good
            return Ok(());
        }
    }

    // write output DDS
    if !is_ci() {
        println!("Writing DDS: {dds_path:?}");

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

        std::fs::create_dir_all(dds_path.parent().unwrap())?;
        std::fs::write(dds_path, output)?;
    }

    if !dds_exists {
        return Err("Output DDS didn't exist".into());
    }
    Err("Output DDS didn't match".into())
}

pub trait Norm {
    const NORM_ONE: Self;
    const NORM_ZERO: Self;
}
impl Norm for u8 {
    const NORM_ONE: Self = u8::MAX;
    const NORM_ZERO: Self = 0;
}
impl Norm for u16 {
    const NORM_ONE: Self = u16::MAX;
    const NORM_ZERO: Self = 0;
}
impl Norm for f32 {
    const NORM_ONE: Self = 1.0;
    const NORM_ZERO: Self = 0.0;
}

pub fn convert_channels<T>(data: &[T], from: Channels, to: Channels) -> Vec<T>
where
    T: Copy + Default + Castable + Norm,
{
    if from == to {
        return data.to_vec();
    }

    fn convert<const N: usize, const M: usize, T>(
        data: &[T],
        f: impl Fn([T; N]) -> [T; M],
    ) -> Vec<T>
    where
        T: Copy + Default + Castable,
    {
        let pixels = data.len() / N;
        let mut result: Vec<T> = vec![Default::default(); pixels * M];

        let data_n: &[[T; N]] = cast_slice(data);
        let result_m: &mut [[T; M]] = cast_slice_mut(&mut result);

        for (i, o) in data_n.iter().zip(result_m.iter_mut()) {
            *o = f(*i);
        }

        result
    }

    match (from, to) {
        // already handled
        (Channels::Grayscale, Channels::Grayscale)
        | (Channels::Alpha, Channels::Alpha)
        | (Channels::Rgb, Channels::Rgb)
        | (Channels::Rgba, Channels::Rgba) => unreachable!(),

        (Channels::Grayscale, Channels::Alpha) => convert(data, |[_]| [T::NORM_ONE]),
        (Channels::Grayscale, Channels::Rgb) => convert(data, |[g]| [g, g, g]),
        (Channels::Grayscale, Channels::Rgba) => convert(data, |[g]| [g, g, g, T::NORM_ONE]),
        (Channels::Alpha, Channels::Grayscale) => convert(data, |[_]| [T::NORM_ZERO]),
        (Channels::Alpha, Channels::Rgb) => {
            convert(data, |[_]| [T::NORM_ZERO, T::NORM_ZERO, T::NORM_ZERO])
        }
        (Channels::Alpha, Channels::Rgba) => {
            convert(data, |[a]| [T::NORM_ZERO, T::NORM_ZERO, T::NORM_ZERO, a])
        }
        (Channels::Rgb, Channels::Grayscale) => convert(data, |[r, _, _]| [r]),
        (Channels::Rgb, Channels::Alpha) => convert(data, |[_, _, _]| [T::NORM_ONE]),
        (Channels::Rgb, Channels::Rgba) => convert(data, |[r, g, b]| [r, g, b, T::NORM_ONE]),
        (Channels::Rgba, Channels::Grayscale) => convert(data, |[r, _, _, _]| [r]),
        (Channels::Rgba, Channels::Alpha) => convert(data, |[_, _, _, a]| [a]),
        (Channels::Rgba, Channels::Rgb) => convert(data, |[r, g, b, _]| [r, g, b]),
    }
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

pub fn compare_snapshot_text(
    snapshot_file: &Path,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = text.replace("\r\n", "\n");

    // compare to snapshot
    let file_exists = snapshot_file.exists();
    let mut native_line_ends = "\n";

    if file_exists {
        let mut snapshot = std::fs::read_to_string(snapshot_file).unwrap();
        if snapshot.contains("\r\n") {
            native_line_ends = "\r\n";
            snapshot = snapshot.replace("\r\n", "\n");
        }

        if text.trim() == snapshot.trim() {
            // all ok
            return Ok(());
        }
    }

    // write snapshot
    if !is_ci() {
        println!("Writing snapshot: {snapshot_file:?}");

        std::fs::create_dir_all(snapshot_file.parent().unwrap()).unwrap();
        std::fs::write(snapshot_file, text.replace("\n", native_line_ends)).unwrap();
    }

    if !file_exists {
        Err(format!("Snapshot file not found: {snapshot_file:?}").into())
    } else {
        Err("Snapshot differs from expected.".into())
    }
}

pub fn pretty_print_header(out: &mut String, header: &Header) {
    out.push_str("Header:\n");
    if let Some(d) = header.depth() {
        out.push_str(&format!(
            "    w/h/d: {:?} x {:?} x {:?}\n",
            header.width(),
            header.height(),
            d
        ));
    } else {
        out.push_str(&format!(
            "    w/h: {:?} x {:?}\n",
            header.width(),
            header.height()
        ));
    }
    out.push_str(&format!("    mipmap_count: {:?}\n", header.mipmap_count()));
    match header {
        Header::Dx9(dx9) => {
            if !dx9.caps2.is_empty() {
                out.push_str(&format!("    caps2: {:?}\n", dx9.caps2));
            }

            match &dx9.pixel_format {
                Dx9PixelFormat::FourCC(four_cc) => {
                    out.push_str(&format!("    format: {four_cc:?}\n"));
                }
                Dx9PixelFormat::Mask(pixel_format) => {
                    out.push_str("    format: masked\n");
                    out.push_str(&format!("        flags: {:?}\n", pixel_format.flags));
                    out.push_str(&format!(
                        "        rgb_bit_count: {:?}\n",
                        pixel_format.rgb_bit_count as u32
                    ));
                    out.push_str(&format!(
                        "        bit_mask: r:0x{:x} g:0x{:x} b:0x{:x} a:0x{:x}\n",
                        pixel_format.r_bit_mask,
                        pixel_format.g_bit_mask,
                        pixel_format.b_bit_mask,
                        pixel_format.a_bit_mask
                    ));
                }
            }
        }
        Header::Dx10(dx10) => {
            out.push_str(&format!("    DX10: {:?}\n", dx10.resource_dimension));
            out.push_str(&format!("        dxgi_format: {:?}\n", dx10.dxgi_format));
            if !dx10.misc_flag.is_empty() {
                out.push_str(&format!("        misc_flag: {:?}\n", dx10.misc_flag));
            }
            if dx10.array_size != 1 {
                out.push_str(&format!("        array_size: {:?}\n", dx10.array_size));
            }
            if dx10.alpha_mode != AlphaMode::Unknown {
                out.push_str(&format!("        alpha_mode: {:?}\n", dx10.alpha_mode));
            }
        }
    };
}

pub fn pretty_print_raw_header(out: &mut String, raw: &RawHeader) {
    out.push_str("Raw Header:\n");

    if raw.size != 124 {
        out.push_str(&format!("    size: {:?}\n", raw.size));
    }
    out.push_str(&format!("    flags: {:?}\n", raw.flags));

    if raw.flags.contains(DdsFlags::DEPTH) {
        out.push_str(&format!(
            "    w/h/d: {:?} x {:?} x {:?}\n",
            raw.width, raw.height, raw.depth
        ));
    } else {
        out.push_str(&format!(
            "    w/h: {:?} x {:?} (x {:?})\n",
            raw.width, raw.height, raw.depth
        ));
    }

    let size = raw.pitch_or_linear_size;
    if raw.flags.contains(DdsFlags::PITCH) && !raw.flags.contains(DdsFlags::LINEAR_SIZE) {
        out.push_str(&format!("    pitch: {size:?}\n"));
    } else if !raw.flags.contains(DdsFlags::PITCH) && raw.flags.contains(DdsFlags::LINEAR_SIZE) {
        out.push_str(&format!("    linear_size: {size:?}\n"));
    } else {
        out.push_str(&format!("    pitch_or_linear_size: {size:?}\n"));
    }

    out.push_str(&format!("    mipmap_count: {:?}", raw.mipmap_count));
    if !raw.flags.contains(DdsFlags::MIPMAP_COUNT) {
        out.push_str("  (not specified)");
    }
    out.push('\n');

    if raw.reserved1.iter().any(|&x| x != 0) {
        out.push_str("    reserved1:\n");
        let zero_prefix = raw.reserved1.iter().take_while(|&&x| x == 0).count();
        if zero_prefix > 0 {
            out.push_str(&format!("        0..={}: 0\n", zero_prefix - 1));
        }
        for i in zero_prefix..raw.reserved1.len() {
            out.push_str(&format!("           {i:>2}: "));

            let n = raw.reserved1[i];
            let bytes = n.to_le_bytes();

            if bytes.iter().all(|x| x.is_ascii_alphanumeric()) {
                for byte in bytes {
                    out.push(byte as char);
                }
                out.push_str(" (ASCII)");
            } else {
                out.push_str(&format!("{n:#010X} {n}"));
            }

            out.push('\n');
        }
    }

    if raw.pixel_format.flags == PixelFormatFlags::FOURCC
        && raw.pixel_format.rgb_bit_count == 0
        && raw.pixel_format.r_bit_mask == 0
        && raw.pixel_format.g_bit_mask == 0
        && raw.pixel_format.b_bit_mask == 0
        && raw.pixel_format.a_bit_mask == 0
    {
        out.push_str(&format!(
            "    pixel_format: {:?}\n",
            raw.pixel_format.four_cc
        ));
    } else {
        out.push_str("    pixel_format:\n");
        out.push_str(&format!("        flags: {:?}\n", raw.pixel_format.flags));
        if raw.pixel_format.four_cc != FourCC::NONE {
            out.push_str(&format!(
                "        four_cc: {:?}\n",
                raw.pixel_format.four_cc
            ));
        }
        out.push_str(&format!(
            "        rgb_bit_count: {:?}\n",
            raw.pixel_format.rgb_bit_count
        ));
        out.push_str(&format!(
            "        bit_mask: r:0x{:x} g:0x{:x} b:0x{:x} a:0x{:x}\n",
            raw.pixel_format.r_bit_mask,
            raw.pixel_format.g_bit_mask,
            raw.pixel_format.b_bit_mask,
            raw.pixel_format.a_bit_mask
        ));
    }

    out.push_str(&format!("    caps: {:?}", raw.caps));
    if !raw.flags.contains(DdsFlags::CAPS) {
        out.push_str("  (not specified)");
    }
    out.push('\n');

    out.push_str(&format!("    caps2: {:?}\n", raw.caps2));
    if raw.caps3 != 0 || raw.caps4 != 0 || raw.reserved2 != 0 {
        out.push_str(&format!("    caps3: {:?}\n", raw.caps3));
        out.push_str(&format!("    caps4: {:?}\n", raw.caps4));
        out.push_str(&format!("    reserved2: {:?}\n", raw.reserved2));
    }

    if let Some(dx10) = &raw.dx10 {
        out.push_str("    DX10:\n");

        out.push_str("        dxgi_format: ");
        if let Ok(dxgi) = DxgiFormat::try_from(dx10.dxgi_format) {
            out.push_str(&format!("{dxgi:?}"));
        } else {
            out.push_str(&format!("{:?}", dx10.dxgi_format));
        }
        out.push('\n');

        out.push_str("        resource_dimension: ");
        if let Ok(dim) = ResourceDimension::try_from(dx10.resource_dimension) {
            out.push_str(&format!("{dim:?}"));
        } else {
            out.push_str(&format!("{:?}", dx10.resource_dimension));
        }
        out.push('\n');

        out.push_str(&format!("        misc_flag: {:?}\n", dx10.misc_flag));
        out.push_str(&format!("        array_size: {:?}\n", dx10.array_size));
        out.push_str(&format!("        misc_flags2: {:?}\n", dx10.misc_flags2));
    }
}

pub fn pretty_print_data_layout(out: &mut String, layout: &DataLayout) {
    out.push_str("Layout: ");
    let pixels = layout.pixel_info();
    match layout {
        DataLayout::Texture(texture) => {
            out.push_str(&format!(
                "Texture ({} bytes @ {:?})\n",
                texture.data_len(),
                pixels
            ));
            for (i, surface) in texture.iter_mips().enumerate() {
                out.push_str(&format!(
                    "    Surface[{i}] {}x{} ({} bytes)\n",
                    surface.width(),
                    surface.height(),
                    surface.data_len()
                ));
            }
        }
        DataLayout::Volume(volume) => {
            out.push_str(&format!(
                "Volume ({} bytes @ {:?})\n",
                volume.data_len(),
                pixels
            ));
            for (i, volume) in volume.iter_mips().enumerate() {
                out.push_str(&format!(
                    "    Volume[{i}] {}x{}x{} ({} bytes)\n",
                    volume.width(),
                    volume.height(),
                    volume.depth(),
                    volume.data_len()
                ));
                for (i, surface) in volume.iter_depth_slices().enumerate() {
                    out.push_str(&format!(
                        "        Surface[{i}] {}x{} ({} bytes)\n",
                        surface.width(),
                        surface.height(),
                        surface.data_len()
                    ));
                }
            }
        }
        DataLayout::TextureArray(texture_array) => {
            out.push_str(&format!(
                "TextureArray len:{} kind:{:?} ({} bytes @ {:?})\n",
                texture_array.len(),
                texture_array.kind(),
                texture_array.data_len(),
                pixels
            ));
            for (i, texture) in texture_array.iter().enumerate() {
                out.push_str(&format!(
                    "    Texture[{i}] ({} bytes)\n",
                    texture.data_len()
                ));
                for (i, surface) in texture.iter_mips().enumerate() {
                    out.push_str(&format!(
                        "        Surface[{i}] {}x{} ({} bytes)\n",
                        surface.width(),
                        surface.height(),
                        surface.data_len()
                    ));
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricChannel {
    L,
    R,
    G,
    B,
    A,
}
pub struct Metrics {
    pub channel: MetricChannel,
    pub mse: f64,
    /// This is the MSE of the image after a small blur
    pub mse_blur: f64,
    pub region_error: f64,
}
impl Metrics {
    pub fn psnr(&self) -> f64 {
        -10.0 * self.mse.log(10.0)
    }
    pub fn psnr_blur(&self) -> f64 {
        -10.0 * self.mse_blur.log(10.0)
    }
}
pub fn measure_compression_quality(org: &Image<f32>, compressed: &Image<f32>) -> Vec<Metrics> {
    assert!(org.size == compressed.size);
    assert!(org.channels == compressed.channels);
    assert!(org.data.len() == compressed.data.len());
    let width = org.size.width as usize;
    let height = org.size.height as usize;

    fn calculate_mse<T, F>(org: &[T], compressed: &[T], get_value: F) -> f64
    where
        T: Copy,
        F: Fn(T) -> f64,
    {
        let mut mse = 0.0;
        for (&o, &c) in org.iter().zip(compressed.iter()) {
            let diff = get_value(o) - get_value(c);
            mse += diff * diff;
        }
        mse /= org.len() as f64;
        mse
    }
    fn box_blur<T, F>(image: &[T], width: usize, height: usize, get_value: F) -> Vec<f64>
    where
        T: Copy,
        F: Fn(T) -> f64,
    {
        let mut blurred: Vec<f64> = image.iter().map(|&x| get_value(x)).collect();

        const GAUSS_WEIGHTS: [f64; 5] = {
            let raw = [1.0, 4.0, 6.0, 4.0, 1.0];
            let sum = raw[0] + raw[1] + raw[2] + raw[3] + raw[4];
            [
                raw[0] / sum,
                raw[1] / sum,
                raw[2] / sum,
                raw[3] / sum,
                raw[4] / sum,
            ]
        };
        fn weigh(values: [f64; 5]) -> f64 {
            values[0] * GAUSS_WEIGHTS[0]
                + values[1] * GAUSS_WEIGHTS[1]
                + values[2] * GAUSS_WEIGHTS[2]
                + values[3] * GAUSS_WEIGHTS[3]
                + values[4] * GAUSS_WEIGHTS[4]
        }

        // Pass 1: horizontal
        for y in 0..height {
            let index_base = y * width;
            let mut prev_prev = blurred[index_base];
            let mut prev = blurred[index_base];
            let last_index = index_base + width - 1;
            for x in 0..width {
                let current = blurred[index_base + x];
                let next = blurred[last_index.min(index_base + x + 1)];
                let next_next = blurred[last_index.min(index_base + x + 2)];
                let sum = weigh([prev_prev, prev, current, next, next_next]);
                prev_prev = prev;
                prev = current;
                blurred[index_base + x] = sum;
            }
        }

        // Pass 2: vertical
        for x in 0..width {
            let mut prev_prev = blurred[x];
            let mut prev = blurred[x];
            let last_index = (height - 1) * width + x;
            for y in 0..height {
                let index = y * width + x;
                let current = blurred[index];
                let next = blurred[last_index.min(index + width)];
                let next_next = blurred[last_index.min(index + 2 * width)];
                let sum = weigh([prev_prev, prev, current, next, next_next]);
                prev_prev = prev;
                prev = current;
                blurred[index] = sum;
            }
        }

        blurred
    }
    fn calculate_metrics<T, F>(
        org: &[T],
        compressed: &[T],
        width: usize,
        height: usize,
        channel: MetricChannel,
        get_value: F,
    ) -> Metrics
    where
        T: Copy,
        F: Copy + Fn(T) -> f64,
    {
        // PSNR
        let mse = calculate_mse(org, compressed, get_value);

        // blurred PSNR
        let blurred_org = box_blur(org, width, height, get_value);
        let blurred_compressed = box_blur(compressed, width, height, get_value);
        let mse_blur = calculate_mse(&blurred_org, &blurred_compressed, |x| x);

        // region error is just the absolute average error per 4x4 region
        const REGION_SIZE: usize = 4;
        let mut region_error = 0.0;
        for region_y in 0..height / REGION_SIZE {
            for region_x in 0..width / REGION_SIZE {
                let mut region = 0.0;
                for y in 0..REGION_SIZE {
                    for x in 0..REGION_SIZE {
                        let i = (region_y * REGION_SIZE + y) * width + region_x * REGION_SIZE + x;
                        let diff = get_value(org[i]) - get_value(compressed[i]);
                        region += diff;
                    }
                }
                region_error += region.abs() / (REGION_SIZE * REGION_SIZE) as f64;
            }
        }
        region_error /= (width / REGION_SIZE * height / REGION_SIZE) as f64;

        Metrics {
            channel,
            mse,
            mse_blur,
            region_error,
        }
    }

    #[allow(clippy::excessive_precision)]
    fn rgb_to_l(r: f32, g: f32, b: f32) -> f32 {
        // OKLab
        fn srgb_to_linear(c: f32) -> f32 {
            if c >= 0.04045 {
                ((c + 0.055) / 1.055).powf(2.4)
            } else {
                c / 12.92
            }
        }
        fn cbrt(x: f32) -> f32 {
            // This is the fast cbrt approximation from the oklab crate.
            // Source: https://gitlab.com/kornelski/oklab/-/blob/d3c074f154187dd5c0642119a6402a6c0753d70c/oklab/src/lib.rs#L61
            // Author: Kornel (https://gitlab.com/kornelski/)
            const B: u32 = 709957561;
            const C: f32 = 5.4285717010e-1;
            const D: f32 = -7.0530611277e-1;
            const E: f32 = 1.4142856598e+0;
            const F: f32 = 1.6071428061e+0;
            const G: f32 = 3.5714286566e-1;

            let mut t = f32::from_bits((x.to_bits() / 3).wrapping_add(B));
            let s = C + (t * t) * (t / x);
            t *= G + F / (s + E + D / s);
            t
        }

        let [r, g, b] = [r, g, b].map(srgb_to_linear);

        let mut l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
        let mut m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
        let mut s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

        l = cbrt(l);
        m = cbrt(m);
        s = cbrt(s);

        l * 0.2104542553 + m * 0.7936177850 + s * -0.0040720468
    }

    match org.channels {
        Channels::Grayscale => {
            let l = calculate_metrics(
                &org.data,
                &compressed.data,
                width,
                height,
                MetricChannel::L,
                |x| x as f64,
            );

            vec![l]
        }
        Channels::Alpha => todo!(),
        Channels::Rgb => {
            let org: &[[f32; 3]] = cast_slice(&org.data);
            let compressed: &[[f32; 3]] = cast_slice(&compressed.data);

            let l = calculate_metrics(
                org,
                compressed,
                width,
                height,
                MetricChannel::L,
                |[r, g, b]| rgb_to_l(r, g, b) as f64,
            );
            let r = calculate_metrics(org, compressed, width, height, MetricChannel::R, |x| {
                x[0] as f64
            });
            let g = calculate_metrics(org, compressed, width, height, MetricChannel::G, |x| {
                x[1] as f64
            });
            let b = calculate_metrics(org, compressed, width, height, MetricChannel::B, |x| {
                x[2] as f64
            });

            vec![l, r, g, b]
        }
        Channels::Rgba => {
            let org: &[[f32; 4]] = cast_slice(&org.data);
            let compressed: &[[f32; 4]] = cast_slice(&compressed.data);

            let l = calculate_metrics(
                org,
                compressed,
                width,
                height,
                MetricChannel::L,
                |[r, g, b, a]| (rgb_to_l(r, g, b) * a) as f64,
            );
            let r = calculate_metrics(org, compressed, width, height, MetricChannel::R, |x| {
                x[0] as f64
            });
            let g = calculate_metrics(org, compressed, width, height, MetricChannel::G, |x| {
                x[1] as f64
            });
            let b = calculate_metrics(org, compressed, width, height, MetricChannel::B, |x| {
                x[2] as f64
            });
            let a = calculate_metrics(org, compressed, width, height, MetricChannel::A, |x| {
                x[3] as f64
            });

            vec![l, r, g, b, a]
        }
    }
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
            let r = compare_snapshot_text(&path, text);
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

pub struct PrettyTable {
    cells: Vec<String>,
    width: usize,
    height: usize,
}
impl PrettyTable {
    pub fn new_empty(width: usize, height: usize) -> Self {
        Self {
            cells: vec![String::new(); width * height],
            width,
            height,
        }
    }
    pub fn from_header<S: AsRef<str>>(header: &[S]) -> Self {
        let mut table = Self::new_empty(header.len(), 0);
        table.add_row(header);
        table
    }
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get(&self, x: usize, y: usize) -> &str {
        &self.cells[y * self.width + x]
    }
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut String {
        &mut self.cells[y * self.width + x]
    }

    #[allow(unused)]
    pub fn set(&mut self, x: usize, y: usize, value: impl Into<String>) {
        *self.get_mut(x, y) = value.into();
    }

    #[track_caller]
    pub fn add_row<S: AsRef<str>>(&mut self, row: &[S]) {
        assert!(row.len() == self.width);
        self.height += 1;
        for cell in row {
            self.cells.push(cell.as_ref().to_string());
        }
    }
    pub fn add_empty_row(&mut self) {
        self.height += 1;
        for _ in 0..self.width {
            self.cells.push(String::new());
        }
    }

    pub fn print(&self, out: &mut String) {
        let column_width: Vec<usize> = (0..self.width)
            .map(|x| {
                (0..self.height)
                    .map(|y| self.get(x, y).chars().count())
                    .max()
                    .unwrap()
            })
            .collect();

        let mut line = String::new();
        for y in 0..self.height {
            #[allow(clippy::needless_range_loop)]
            for x in 0..self.width {
                let cell = self.get(x, y);
                line.push_str(cell);
                for _ in 0..column_width[x] - cell.chars().count() {
                    line.push(' ');
                }
                line.push_str("  ");
            }
            out.push_str(line.trim_end());
            out.push('\n');
            line.clear();
        }
    }

    pub fn print_markdown(&self, out: &mut String) {
        let column_width: Vec<usize> = (0..self.width)
            .map(|x| {
                (0..self.height)
                    .map(|y| self.get(x, y).chars().count())
                    .max()
                    .unwrap()
            })
            .collect();

        for y in 0..self.height {
            #[allow(clippy::needless_range_loop)]
            for x in 0..self.width {
                let cell = self.get(x, y);
                out.push_str("| ");
                out.push_str(cell);
                for _ in 0..column_width[x] - cell.chars().count() {
                    out.push(' ');
                }
                out.push(' ');
            }

            // poor man's trim
            while let Some(last) = out.chars().last() {
                if last == ' ' {
                    out.pop();
                } else {
                    break;
                }
            }

            out.push('\n');

            if y == 0 {
                #[allow(clippy::needless_range_loop)]
                for x in 0..self.width {
                    out.push_str("| ");
                    for _ in 0..column_width[x] {
                        out.push('-');
                    }
                    if x != self.width - 1 {
                        out.push(' ');
                    }
                }
                out.push('\n');
            }
        }
    }
}
impl std::fmt::Display for PrettyTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        self.print(&mut out);
        write!(f, "{out}")
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
