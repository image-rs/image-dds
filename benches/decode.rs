use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ddsd::*;

fn simple_texture_header(size: Size, format: DxgiFormat) -> Header {
    Header {
        flags: DdsFlags::REQUIRED,
        height: size.height,
        width: size.width,
        depth: None,
        mipmap_count: None,
        pixel_format: PixelFormat {
            flags: PixelFormatFlags::FOURCC,
            four_cc: Some(FourCC::DX10),
            rgb_bit_count: 0,
            r_bit_mask: 0,
            g_bit_mask: 0,
            b_bit_mask: 0,
            a_bit_mask: 0,
        },
        caps: DdsCaps::REQUIRED,
        caps2: DdsCaps2::empty(),
        dxt10: Some(HeaderDxt10 {
            dxgi_format: format,
            resource_dimension: ResourceDimension::Texture2D,
            misc_flag: MiscFlags::empty(),
            array_size: 1,
            misc_flags2: MiscFlags2::empty(),
        }),
    }
}

fn random_bytes(len: usize) -> Vec<u8> {
    let mut out = vec![0; len];

    let mut state = 0x0123456789ABCDEFu64;
    for i in 0..(len / 4) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = (state >> 31) as u32;
        out[i * 4] = x as u8;
        out[i * 4 + 1] = (x >> 8) as u8;
        out[i * 4 + 2] = (x >> 16) as u8;
        out[i * 4 + 3] = (x >> 24) as u8;
    }

    out
}

fn bench_decoder(c: &mut Criterion, format: DxgiFormat, channels: Channels, precision: Precision) {
    let color = ColorFormat::new(channels, precision);
    let name = format!("{:?} -> {}", format, color);

    c.bench_function(&name, |b| {
        let header = simple_texture_header((4096, 4096).into(), format);

        let reader = DdsDecoder::from_header(header).unwrap();
        let format = reader.format();
        assert!(format.supports(color));

        let surface = reader.layout().texture().unwrap().main();
        let bytes = random_bytes(surface.data_len() as usize).into_boxed_slice();
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

pub fn uncompressed(c: &mut Criterion) {
    use Channels::*;
    use Precision::*;

    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::R8G8B8A8_SNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgba, U16);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgba, F32);
    bench_decoder(c, DxgiFormat::R8G8B8A8_UNORM, Rgb, U8);
    bench_decoder(c, DxgiFormat::R16G16_SNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::B8G8R8X8_UNORM, Rgba, U8);
    bench_decoder(c, DxgiFormat::BC1_UNORM, Rgba, U8);
}

criterion_group!(benches, uncompressed);
criterion_main!(benches);
