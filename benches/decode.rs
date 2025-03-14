use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ddsd::*;
use rand::{seq::SliceRandom, Rng, RngCore};

fn random_bytes(len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    let mut rng = rand::thread_rng();
    rng.fill_bytes(&mut out);
    out
}

type DataModifier = Box<dyn FnMut(&mut [u8])>;
struct BenchConfig {
    data_modifier: DataModifier,
    size: Size,
    name: String,
}
impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            data_modifier: Box::new(|_| {}),
            size: (4096, 4096).into(),
            name: String::new(),
        }
    }
}
fn bench_decoder(c: &mut Criterion, format: DxgiFormat, channels: Channels, precision: Precision) {
    bench_decoder_with(c, format, channels, precision, |_| {});
}
fn bench_decoder_with(
    c: &mut Criterion,
    format: DxgiFormat,
    channels: Channels,
    precision: Precision,
    create_config: impl FnOnce(&mut BenchConfig),
) {
    let mut config = BenchConfig::default();
    create_config(&mut config);

    let color = ColorFormat::new(channels, precision);
    let mut name = format!("{:?} -> {}", format, color);
    if !config.name.is_empty() {
        name += " - ";
        name += &config.name;
    }

    c.bench_function(&name, |b| {
        let header = Header::new_image(config.size.width, config.size.height, format);

        let reader = DdsDecoder::from_header(header).unwrap();
        let format = reader.format();

        let surface = reader.layout().texture().unwrap().main();
        let mut bytes = random_bytes(surface.data_len() as usize).into_boxed_slice();
        (config.data_modifier)(&mut bytes);
        let mut output =
            vec![0; surface.size().pixels() as usize * color.bytes_per_pixel() as usize];
        b.iter(|| {
            let result = format.decode(
                black_box(&mut bytes.as_ref()),
                surface.size(),
                color,
                black_box(&mut output),
            );
            black_box(result).unwrap();
        });
    });
}

/// This sets the BC7 block modes such that each mode is equally likely.
///
/// This is necessary, because the block mode is decided by the number of
/// leading zeros, meaning that for random bytes, 50% of the blocks will be
/// mode 0. This does NOT represent real-world data at all, hence this function.
///
/// Note that this is not a perfect solution either, but it should be good
/// enough.
fn random_bc7_modes(data: &mut [u8]) {
    let mut rng = rand::thread_rng();
    for i in (0..data.len()).step_by(16) {
        let mode: u8 = rng.gen_range(0..8);
        let mut byte = data[i];
        byte |= 1;
        byte <<= mode;
        data[i] = byte;
    }
}
fn set_bc7_modes(data: &mut [u8], mode: u8) {
    for i in (0..data.len()).step_by(16) {
        let mut byte = data[i];
        byte |= 1;
        byte <<= mode;
        data[i] = byte;
    }
}

const BC6_MODES: [(u8, u8); 15] = [
    (0b00, 2),
    (0b01, 2),
    (0b00010, 5),
    (0b00110, 5),
    (0b01010, 5),
    (0b01110, 5),
    (0b10010, 5),
    (0b10110, 5),
    (0b11010, 5),
    (0b11110, 5),
    (0b00011, 5),
    (0b00111, 5),
    (0b01011, 5),
    (0b01111, 5),
    (0b11111, 5), // reserved
];
/// This sets the BC6 block modes such that each mode is equally likely.
fn random_bc6_modes(data: &mut [u8]) {
    let mut rng = rand::thread_rng();
    for i in (0..data.len()).step_by(16) {
        let (mode, mode_bits): (u8, u8) = *BC6_MODES.choose(&mut rng).unwrap();
        let mut byte = data[i];
        byte <<= mode_bits;
        byte |= mode;
        data[i] = byte;
    }
}

pub fn uncompressed(c: &mut Criterion) {
    use Channels::*;
    use Precision::*;

    // uncompressed formats
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgba, U16);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgba, F32);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgb, U8);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgb, U16);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgb, F32);

    bench_decoder(c, DxgiFormat::R8G8B8A8_SNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::R8G8B8A8_SNORM, Rgba, U16);
    bench_decoder(c, DxgiFormat::R8G8B8A8_SNORM, Rgba, F32);
    bench_decoder(c, DxgiFormat::R8G8B8A8_SNORM, Rgb, U8);
    bench_decoder(c, DxgiFormat::R8G8B8A8_SNORM, Rgb, U16);
    bench_decoder(c, DxgiFormat::R8G8B8A8_SNORM, Rgb, F32);

    bench_decoder(c, DxgiFormat::R16G16_SNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::B8G8R8X8_UNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::R9G9B9E5_SHAREDEXP, Rgb, U8);

    bench_decoder(c, DxgiFormat::R16G16B16A16_FLOAT, Rgba, U8);
    bench_decoder(c, DxgiFormat::R16G16B16A16_FLOAT, Rgba, U16);
    bench_decoder(c, DxgiFormat::R16G16B16A16_FLOAT, Rgba, F32);

    bench_decoder(c, DxgiFormat::R32G32B32A32_FLOAT, Rgba, U8);
    bench_decoder(c, DxgiFormat::R32G32B32A32_FLOAT, Rgba, U16);
    bench_decoder(c, DxgiFormat::R32G32B32A32_FLOAT, Rgba, F32);

    bench_decoder(c, DxgiFormat::R11G11B10_FLOAT, Rgba, U8);
    bench_decoder(c, DxgiFormat::R11G11B10_FLOAT, Rgba, U16);
    bench_decoder(c, DxgiFormat::R11G11B10_FLOAT, Rgba, F32);

    // sub-sampled formats
    bench_decoder(c, DxgiFormat::R8G8_B8G8_UNORM, Rgb, U8);

    // block-compressed formats
    bench_decoder(c, DxgiFormat::BC1_UNORM, Rgba, U8);
    bench_decoder_with(c, DxgiFormat::BC1_UNORM, Rgba, U8, |c| {
        c.size = (4095, 4095).into();
    });
    bench_decoder(c, DxgiFormat::BC1_UNORM, Rgb, U8);
    bench_decoder_with(c, DxgiFormat::BC1_UNORM, Rgb, U8, |c| {
        c.size = (4095, 4095).into();
    });
    bench_decoder(c, DxgiFormat::BC4_UNORM, Grayscale, U8);
    bench_decoder(c, DxgiFormat::BC4_SNORM, Grayscale, U8);
    bench_decoder_with(c, DxgiFormat::BC7_UNORM, Rgba, U8, |c| {
        c.data_modifier = Box::new(random_bc7_modes);
    });
    bench_decoder_with(c, DxgiFormat::BC6H_SF16, Rgb, U8, |c| {
        c.data_modifier = Box::new(random_bc6_modes);
        c.size = (1024, 1024).into();
    });
    bench_decoder_with(c, DxgiFormat::BC6H_SF16, Rgb, U16, |c| {
        c.data_modifier = Box::new(random_bc6_modes);
        c.size = (1024, 1024).into();
    });
    bench_decoder_with(c, DxgiFormat::BC6H_SF16, Rgb, F32, |c| {
        c.data_modifier = Box::new(random_bc6_modes);
        c.size = (1024, 1024).into();
    });
    bench_decoder_with(c, DxgiFormat::BC6H_UF16, Rgb, U8, |c| {
        c.data_modifier = Box::new(random_bc6_modes);
        c.size = (1024, 1024).into();
    });
    bench_decoder_with(c, DxgiFormat::BC6H_UF16, Rgb, U16, |c| {
        c.data_modifier = Box::new(random_bc6_modes);
        c.size = (1024, 1024).into();
    });
    bench_decoder_with(c, DxgiFormat::BC6H_UF16, Rgb, F32, |c| {
        c.data_modifier = Box::new(random_bc6_modes);
        c.size = (1024, 1024).into();
    });
}

pub fn bc7_modes(c: &mut Criterion) {
    use Channels::*;
    use Precision::*;

    let modes = [0, 1, 2, 3, 4, 5, 6, 7];
    for mode in modes {
        bench_decoder_with(c, DxgiFormat::BC7_UNORM, Rgba, U8, |c| {
            c.data_modifier = Box::new(move |data| {
                set_bc7_modes(data, mode);
            });
            c.name = format!("mode {mode}");
            c.size = (1024, 1024).into();
        });
    }
}

criterion_group!(benches, uncompressed, bc7_modes);
criterion_main!(benches);
