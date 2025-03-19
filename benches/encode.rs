use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dds::*;
use rand::{Rng, RngCore};

fn random_bytes(len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    let mut rng = rand::thread_rng();
    rng.fill_bytes(&mut out);
    out
}

struct Image<T> {
    data: Vec<T>,
    size: Size,
    channels: Channels,
    name: String,
}
impl<T: 'static> Image<T> {
    fn new(data: Vec<T>, size: Size, channels: Channels, name: impl Into<String>) -> Self {
        Self {
            data,
            size,
            channels,
            name: name.into(),
        }
    }

    fn precision(&self) -> Precision {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
            Precision::U8
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>() {
            Precision::U16
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            Precision::F32
        } else {
            panic!("Unsupported type");
        }
    }

    fn random(size: Size, channels: Channels) -> Image<T>
    where
        T: Default + Copy,
        [T]: rand::Fill,
    {
        let mut data = vec![T::default(); size.pixels() as usize * channels.count() as usize];
        let mut rng = rand::thread_rng();
        rng.fill(data.as_mut_slice());
        Image::new(data, size, channels, "random_u8")
    }
}
trait ImageAsBytes {
    fn color(&self) -> ColorFormat;
    fn as_bytes(&self) -> &[u8];
}
impl ImageAsBytes for Image<u8> {
    fn color(&self) -> ColorFormat {
        ColorFormat::new(self.channels, self.precision())
    }
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}
impl ImageAsBytes for Image<u16> {
    fn color(&self) -> ColorFormat {
        ColorFormat::new(self.channels, self.precision())
    }
    fn as_bytes(&self) -> &[u8] {
        zerocopy::IntoBytes::as_bytes(self.data.as_slice())
    }
}
impl ImageAsBytes for Image<f32> {
    fn color(&self) -> ColorFormat {
        ColorFormat::new(self.channels, self.precision())
    }
    fn as_bytes(&self) -> &[u8] {
        zerocopy::IntoBytes::as_bytes(self.data.as_slice())
    }
}

fn bench_encoder<T>(c: &mut Criterion, format: Format, options: &EncodeOptions, image: &Image<T>)
where
    Image<T>: ImageAsBytes,
{
    let color = image.color();
    let mut name = format!("{:?}: {}", format, color);
    if options != &EncodeOptions::default() {
        name += &format!(" - {:?}", options);
    }

    c.bench_function(&name, |b| {
        let mut output: Vec<u8> = Vec::with_capacity(image.size.pixels() as usize * 16);

        b.iter(|| {
            output.truncate(0);

            let image = black_box(image);

            let result = encode(
                black_box(&mut output),
                format,
                image.size,
                image.color(),
                image.as_bytes(),
                black_box(options),
            );
            black_box(result).unwrap();
            assert!(!black_box(&output).is_empty());
        });
    });
}

pub fn encode_uncompressed(c: &mut Criterion) {
    use Channels::*;

    // images
    let random_gray_u8: Image<u8> = Image::random(Size::new(1024, 1024), Grayscale);
    // let random_gray_u16: Image<u16> = Image::random(Size::new(1024, 1024), Grayscale);
    let random_gray_f32: Image<f32> = Image::random(Size::new(1024, 1024), Grayscale);
    let random_rgba_u8: Image<u8> = Image::random(Size::new(1024, 1024), Rgba);
    let random_rgba_u16: Image<u16> = Image::random(Size::new(1024, 1024), Rgba);
    let random_rgba_f32: Image<f32> = Image::random(Size::new(1024, 1024), Rgba);

    // options
    let def = &EncodeOptions::default();
    let mut dither = EncodeOptions::default();
    dither.dithering = Dithering::ColorAndAlpha;
    let dither = &dither;

    // uncompressed formats
    bench_encoder(c, Format::R8G8B8_UNORM, def, &random_gray_u8);
    bench_encoder(c, Format::R8G8B8_UNORM, def, &random_rgba_u8);
    bench_encoder(c, Format::R8G8B8_UNORM, def, &random_rgba_u16);
    bench_encoder(c, Format::R8G8B8_UNORM, def, &random_rgba_f32);

    bench_encoder(c, Format::R8G8B8A8_UNORM, def, &random_gray_u8);
    bench_encoder(c, Format::R8G8B8A8_UNORM, def, &random_rgba_u8);
    bench_encoder(c, Format::R8G8B8A8_UNORM, def, &random_rgba_u16);
    bench_encoder(c, Format::R8G8B8A8_UNORM, def, &random_rgba_f32);

    bench_encoder(c, Format::R8G8B8A8_SNORM, def, &random_gray_u8);
    bench_encoder(c, Format::R8G8B8A8_SNORM, def, &random_rgba_u8);
    bench_encoder(c, Format::R8G8B8A8_SNORM, def, &random_rgba_u16);
    bench_encoder(c, Format::R8G8B8A8_SNORM, def, &random_rgba_f32);

    bench_encoder(c, Format::B4G4R4A4_UNORM, def, &random_rgba_u8);
    bench_encoder(c, Format::B4G4R4A4_UNORM, def, &random_rgba_f32);
    bench_encoder(c, Format::B4G4R4A4_UNORM, dither, &random_rgba_u8);
    bench_encoder(c, Format::B4G4R4A4_UNORM, dither, &random_rgba_f32);

    // sub-sampled formats
    bench_encoder(c, Format::R8G8_B8G8_UNORM, def, &random_rgba_u8);
    bench_encoder(c, Format::R8G8_B8G8_UNORM, def, &random_rgba_f32);

    bench_encoder(c, Format::YUY2, def, &random_rgba_u8);
    bench_encoder(c, Format::YUY2, def, &random_rgba_f32);

    bench_encoder(c, Format::Y216, def, &random_rgba_u8);
    bench_encoder(c, Format::Y216, def, &random_rgba_f32);

    bench_encoder(c, Format::R1_UNORM, def, &random_gray_u8);
    bench_encoder(c, Format::R1_UNORM, def, &random_gray_f32);
}

pub fn encode_compressed(c: &mut Criterion) {
    use Channels::*;

    // images
    let random: Image<f32> = Image::random(Size::new(128, 128), Rgba);
    let random_rgb: Image<f32> = Image::random(Size::new(128, 128), Rgb);
    let random_tiny: Image<f32> = Image::random(Size::new(16, 16), Rgba);

    // options
    let def = EncodeOptions::default();
    let mut dither = EncodeOptions::default();
    dither.dithering = Dithering::ColorAndAlpha;
    let mut perceptual = EncodeOptions::default();
    perceptual.error_metric = ErrorMetric::Perceptual;
    let mut unreasonable = EncodeOptions::default();
    unreasonable.quality = CompressionQuality::Unreasonable;

    let def = &def;
    let dither = &dither;
    let perceptual = &perceptual;
    let unreasonable = &unreasonable;

    // uncompressed formats
    bench_encoder(c, Format::BC1_UNORM, def, &random_rgb);
    bench_encoder(c, Format::BC1_UNORM, dither, &random_rgb);
    bench_encoder(c, Format::BC1_UNORM, perceptual, &random_rgb);

    bench_encoder(c, Format::BC4_UNORM, def, &random_rgb);
    bench_encoder(c, Format::BC4_UNORM, dither, &random_rgb);
    bench_encoder(c, Format::BC4_UNORM, unreasonable, &random_tiny);
}

criterion_group!(benches, encode_uncompressed, encode_compressed);
criterion_main!(benches);
