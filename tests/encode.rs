use ddsd::*;
use rand::Rng;
use util::{test_data_dir, Image, WithPrecision};

mod util;

const FORMATS: &[EncodeFormat] = &[
    // uncompressed formats
    EncodeFormat::R8G8B8_UNORM,
    EncodeFormat::B8G8R8_UNORM,
    EncodeFormat::R8G8B8A8_UNORM,
    EncodeFormat::R8G8B8A8_SNORM,
    EncodeFormat::B8G8R8A8_UNORM,
    EncodeFormat::B8G8R8X8_UNORM,
    EncodeFormat::B5G6R5_UNORM,
    EncodeFormat::B5G5R5A1_UNORM,
    EncodeFormat::B4G4R4A4_UNORM,
    EncodeFormat::A4B4G4R4_UNORM,
    EncodeFormat::R8_SNORM,
    EncodeFormat::R8_UNORM,
    EncodeFormat::R8G8_UNORM,
    EncodeFormat::R8G8_SNORM,
    EncodeFormat::A8_UNORM,
    EncodeFormat::R16_UNORM,
    EncodeFormat::R16_SNORM,
    EncodeFormat::R16G16_UNORM,
    EncodeFormat::R16G16_SNORM,
    EncodeFormat::R16G16B16A16_UNORM,
    EncodeFormat::R16G16B16A16_SNORM,
    EncodeFormat::R10G10B10A2_UNORM,
    EncodeFormat::R11G11B10_FLOAT,
    EncodeFormat::R9G9B9E5_SHAREDEXP,
    EncodeFormat::R16_FLOAT,
    EncodeFormat::R16G16_FLOAT,
    EncodeFormat::R16G16B16A16_FLOAT,
    EncodeFormat::R32_FLOAT,
    EncodeFormat::R32G32_FLOAT,
    EncodeFormat::R32G32B32_FLOAT,
    EncodeFormat::R32G32B32A32_FLOAT,
    EncodeFormat::R10G10B10_XR_BIAS_A2_UNORM,
    EncodeFormat::AYUV,
    EncodeFormat::Y410,
    EncodeFormat::Y416,
    // sub-sampled formats
    EncodeFormat::R1_UNORM,
    EncodeFormat::R8G8_B8G8_UNORM,
    EncodeFormat::G8R8_G8B8_UNORM,
    EncodeFormat::UYVY,
    EncodeFormat::YUY2,
    EncodeFormat::Y210,
    EncodeFormat::Y216,
    // block compression formats
    EncodeFormat::BC1_UNORM,
    EncodeFormat::BC2_UNORM,
    EncodeFormat::BC2_UNORM_PREMULTIPLIED_ALPHA,
    EncodeFormat::BC3_UNORM,
    EncodeFormat::BC3_UNORM_PREMULTIPLIED_ALPHA,
    EncodeFormat::BC4_UNORM,
    EncodeFormat::BC4_SNORM,
    EncodeFormat::BC5_UNORM,
    EncodeFormat::BC5_SNORM,
    EncodeFormat::BC6H_UF16,
    EncodeFormat::BC6H_SF16,
    EncodeFormat::BC7_UNORM,
];

fn create_header(size: Size, format: EncodeFormat) -> Header {
    if let Ok(dxgi_format) = format.try_into() {
        Header::new_image(size.width, size.height, dxgi_format)
    } else if let Ok(format) = format.try_into() {
        Dx9Header::new_image(size.width, size.height, format).into()
    } else {
        unreachable!("unsupported format: {:?}", format);
    }
}
fn write_dds_header(size: Size, format: EncodeFormat) -> Vec<u8> {
    let header = create_header(size, format);

    let mut output = Vec::new();
    output.extend_from_slice(&Header::MAGIC);
    header.to_raw().write(&mut output).unwrap();

    output
}
fn encode_image<T: WithPrecision + util::Castable, W: std::io::Write>(
    image: &Image<T>,
    format: EncodeFormat,
    writer: &mut W,
    options: &EncodeOptions,
) -> Result<(), EncodeError> {
    format.encode(writer, image.size, image.color(), image.as_bytes(), options)
}
fn encode_decode(format: EncodeFormat, options: &EncodeOptions, image: &Image<f32>) -> Image<f32> {
    // encode
    let mut encoded = Vec::new();
    encode_image(image, format, &mut encoded, options).unwrap();

    // decode
    let header = create_header(image.size, format);
    let decode_format = DecodeFormat::from_header(&header).unwrap();
    let mut output = vec![0_f32; image.size.pixels() as usize * image.channels.count() as usize];
    decode_format
        .decode_f32(
            &mut encoded.as_slice(),
            image.size,
            image.channels,
            &mut output,
        )
        .unwrap();

    Image {
        size: image.size,
        channels: image.channels,
        data: output,
    }
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

#[test]
fn encode_base() {
    let base_u8 = util::read_png_u8(&util::test_data_dir().join("base.png")).unwrap();
    assert!(base_u8.channels == Channels::Rgba);
    let base_u16 = base_u8.to_u16();
    let base_f32 = base_u8.to_f32();

    let test = |format: EncodeFormat| -> Result<(), Box<dyn std::error::Error>> {
        let mut output = write_dds_header(base_u8.size, format);

        let options = EncodeOptions::default();

        // and now the image data
        if format.precision() == Precision::U16 {
            encode_image(&base_u16, format, &mut output, &options)?;
        } else if format.precision() == Precision::F32 {
            encode_image(&base_f32, format, &mut output, &options)?;
        } else {
            encode_image(&base_u8, format, &mut output, &options)?;
        }

        // write to disk
        let name = format!("{:?}.dds", format);
        let path = test_data_dir().join("output-encode/base").join(&name);
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, &output)?;

        Ok(())
    };

    let mut failed_count = 0;
    for format in FORMATS.iter().copied() {
        if let Err(e) = test(format) {
            eprintln!("Failed to encode {:?}: {}", format, e);
            failed_count += 1;
        }
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
}

#[test]
fn encode_dither() {
    let test = |format: EncodeFormat,
                image: &Image<f32>,
                name: &str|
     -> Result<(), Box<dyn std::error::Error>> {
        let mut output = write_dds_header(image.size, format);

        let mut options = EncodeOptions::default();
        options.dither = DitheredChannels::All;
        encode_image(image, format, &mut output, &options)?;

        // write to disk
        let name = format!("{:?} {}.dds", format, name);
        let path = test_data_dir().join("output-encode/dither").join(&name);
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, &output)?;

        Ok(())
    };

    let base = util::read_png_u8(&util::test_data_dir().join("base.png"))
        .unwrap()
        .to_f32();
    let twirl = util::read_png_u8(&util::test_data_dir().join("color-twirl.png"))
        .unwrap()
        .to_f32();

    let ignore = [
        EncodeFormat::BC4_SNORM,
        EncodeFormat::BC5_UNORM,
        EncodeFormat::BC5_SNORM,
    ];

    let mut failed_count = 0;
    for format in FORMATS
        .iter()
        .copied()
        .filter(|f| f.supports_dither() != DitheredChannels::None)
        .filter(|f| !ignore.contains(f))
    {
        let dither = format.supports_dither();

        if let Err(e) = test(format, &base, "base") {
            eprintln!("Failed to encode {:?}: {}", format, e);
            failed_count += 1;
        }

        if dither != DitheredChannels::AlphaOnly {
            if let Err(e) = test(format, &twirl, "twirl") {
                eprintln!("Failed to encode {:?}: {}", format, e);
                failed_count += 1;
            }
        }
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
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
            let image = util::read_png_u8(&util::test_data_dir().join(name))
                .unwrap()
                .to_f32();

            Self {
                name: name.to_string(),
                image,
            }
        }
    }
    struct TestCase<'a> {
        format: EncodeFormat,
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
            format: EncodeFormat::BC1_UNORM,
            options: vec![
                ("default", EncodeOptions::default()),
                (
                    "dither",
                    new_options(|options| {
                        options.dither = DitheredChannels::All;
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
                random,
            ],
        },
        TestCase {
            format: EncodeFormat::BC4_UNORM,
            options: vec![
                ("default", EncodeOptions::default()),
                (
                    "dither",
                    new_options(|options| {
                        options.dither = DitheredChannels::All;
                    }),
                ),
            ],
            images: &[base, color_twirl, clovers_r, stone_h, random],
        },
    ];

    let collect_info = |case: &TestCase| -> Result<String, Box<dyn std::error::Error>> {
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
            PrettyTable::from_header(&["", "", "", "↑PSNR", "↑PSNR blur", "↓Region error"]);

        for image in case.images {
            let hash_alpha = matches!(image.image.channels, Channels::Rgba | Channels::Alpha);

            table.add_empty_row();

            let name = &image.name;
            let image = image.image.to_channels(case.format.channels());
            let mut name_mentioned = false;
            for (opt_name, options) in &options {
                let encoded = encode_decode(case.format, options, &image);
                let metrics = util::measure_compression_quality(&image, &encoded);
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

    util::compare_snapshot_text(&util::test_data_dir().join("encode_quality.txt"), &output)
        .unwrap();
}

struct PrettyTable {
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

        for y in 0..self.height {
            #[allow(clippy::needless_range_loop)]
            for x in 0..self.width {
                let cell = self.get(x, y);
                out.push_str(cell);
                for _ in 0..column_width[x] - cell.chars().count() {
                    out.push(' ');
                }
                out.push_str("  ");
            }
            out.push('\n');
        }
    }
}
