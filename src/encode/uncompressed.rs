use crate::{
    cast, ch, convert_channels_untyped, convert_to_rgba_f32, encode::write::ToLe, fp10, fp11, fp16,
    n1, n10, n16, n2, n4, n5, n6, n8, rgb9995f, s16, s8, util, xr10, yuv10, yuv16, yuv8, Channels,
    ColorFormat, ColorFormatSet, Precision,
};

use super::{
    write::{convert_to_le, BaseEncoder, Flags},
    Args, DecodedArgs, DitheredChannels, EncodeError,
};

// helpers

fn f4_add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
fn f4_sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}
fn f4_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}
fn f4_fma(a: &mut [f32; 4], b: [f32; 4], scale: f32) {
    a[0] += b[0] * scale;
    a[1] += b[1] * scale;
    a[2] += b[2] * scale;
    a[3] += b[3] * scale;
}

fn uncompressed_universal<EncodedPixel>(
    args: Args,
    process: fn(&[[f32; 4]], &mut [EncodedPixel]),
) -> Result<(), EncodeError>
where
    EncodedPixel: Default + Copy + ToLe + cast::Castable,
{
    let DecodedArgs {
        data,
        color,
        writer,
        ..
    } = DecodedArgs::from(args)?;
    let bytes_per_pixel = color.bytes_per_pixel() as usize;

    const BUFFER_PIXELS: usize = 512;
    let mut intermediate_buffer = [[0_f32; 4]; BUFFER_PIXELS];
    let mut encoded_buffer = [EncodedPixel::default(); BUFFER_PIXELS];

    let chunk_size = BUFFER_PIXELS * bytes_per_pixel;
    for line in data.chunks(chunk_size) {
        debug_assert!(line.len() % bytes_per_pixel == 0);
        let pixels = line.len() / bytes_per_pixel;

        let intermediate = &mut intermediate_buffer[..pixels];
        let encoded = &mut encoded_buffer[..pixels];

        convert_to_rgba_f32(color, line, intermediate);

        process(intermediate, encoded);

        ToLe::to_le(encoded);

        writer.write_all(cast::as_bytes(encoded))?;
    }

    Ok(())
}

fn uncompressed_universal_dither<EncodedPixel, F>(args: Args, f: F) -> Result<(), EncodeError>
where
    EncodedPixel: Default + Copy + ToLe + cast::Castable,
    F: Fn([f32; 4]) -> (EncodedPixel, [f32; 4]),
{
    let DecodedArgs {
        data,
        color,
        writer,
        width,
        options,
        ..
    } = DecodedArgs::from(args)?;
    let bytes_per_pixel = color.bytes_per_pixel() as usize;

    let error_padding = 2;
    let mut error_buffer = vec![[0_f32; 4]; 2 * (width + error_padding * 2)];
    let (mut current_error, mut next_error) = error_buffer.split_at_mut(width + error_padding * 2);

    let error_mask = match options.dither {
        DitheredChannels::None => [0.0; 4],
        DitheredChannels::All => [1.0; 4],
        DitheredChannels::ColorOnly => [1.0, 1.0, 1.0, 0.0],
        DitheredChannels::AlphaOnly => [0.0, 0.0, 0.0, 1.0],
    };

    const BUFFER_PIXELS: usize = 512;
    let mut intermediate_buffer = [[0_f32; 4]; BUFFER_PIXELS];
    let mut encoded_buffer = [EncodedPixel::default(); BUFFER_PIXELS];

    for row in data.chunks(width * bytes_per_pixel) {
        debug_assert!(row.len() == width * bytes_per_pixel);

        // prepare error buffers
        std::mem::swap(&mut current_error, &mut next_error);
        next_error.fill([0.0; 4]);
        let mut error_offset = error_padding;

        let chunk_size = BUFFER_PIXELS * bytes_per_pixel;
        for line in row.chunks(chunk_size) {
            debug_assert!(line.len() % bytes_per_pixel == 0);
            let pixels = line.len() / bytes_per_pixel;

            let intermediate = &mut intermediate_buffer[..pixels];
            let encoded = &mut encoded_buffer[..pixels];
            convert_to_rgba_f32(color, line, intermediate);

            for (i, out) in intermediate.iter().zip(encoded.iter_mut()) {
                let error = current_error[error_offset];
                let (encoded_pixel, mut error) = f(f4_add(*i, error));
                error = f4_mul(error, error_mask);

                // diffuse error with Floyd-Steinberg weights
                f4_fma(&mut current_error[error_offset + 1], error, 7.0 / 16.0);
                f4_fma(&mut next_error[error_offset - 1], error, 3.0 / 16.0);
                f4_fma(&mut next_error[error_offset], error, 5.0 / 16.0);
                f4_fma(&mut next_error[error_offset + 1], error, 1.0 / 16.0);

                *out = encoded_pixel;
                error_offset += 1;
            }

            ToLe::to_le(encoded);
            writer.write_all(cast::as_bytes(encoded))?;
        }
    }

    Ok(())
}

fn uncompressed_untyped(
    args: Args,
    bytes_per_encoded_pixel: usize,
    f: impl Fn(&[u8], ColorFormat, &mut [u8]),
) -> Result<(), EncodeError> {
    let DecodedArgs {
        data,
        color,
        writer,
        ..
    } = DecodedArgs::from(args)?;
    let bytes_per_pixel = color.bytes_per_pixel() as usize;

    let mut raw_buffer = [0_u32; 1024];
    let encoded_buffer = cast::as_bytes_mut(&mut raw_buffer);

    let chuck_size = encoded_buffer.len() / bytes_per_pixel * bytes_per_pixel;
    for line in data.chunks(chuck_size) {
        debug_assert!(line.len() % bytes_per_pixel == 0);
        let pixels = line.len() / bytes_per_pixel;
        let encoded = &mut encoded_buffer[..pixels * bytes_per_encoded_pixel];

        f(line, color, encoded);

        writer.write_all(encoded)?;
    }

    Ok(())
}
fn simple_color_convert(
    target: ColorFormat,
    snorm: bool,
) -> impl Fn(&[u8], ColorFormat, &mut [u8]) {
    if snorm {
        assert!(matches!(target.precision, Precision::U8 | Precision::U16));
    }

    move |line, color, out| {
        assert!(color.precision == target.precision);

        let from = color.channels;
        let to = target.channels;
        match target.precision {
            Precision::U8 => convert_channels_untyped::<u8>(from, to, line, out),
            Precision::U16 => convert_channels_untyped::<u16>(from, to, line, out),
            Precision::F32 => convert_channels_untyped::<f32>(from, to, line, out),
        }

        if snorm {
            match target.precision {
                Precision::U8 => {
                    out.iter_mut().for_each(|o| *o = s8::from_n8(*o));
                }
                Precision::U16 => {
                    let chunked: &mut [[u8; 2]] =
                        cast::as_array_chunks_mut(out).expect("invalid buffer size");
                    chunked.iter_mut().for_each(|o| {
                        *o = s16::from_n16(u16::from_ne_bytes(*o)).to_ne_bytes();
                    });
                }
                Precision::F32 => unreachable!(),
            }
        }

        convert_to_le(target.precision, out);
    }
}

macro_rules! color_convert {
    ($target:expr) => {
        BaseEncoder {
            color_formats: ColorFormatSet::from_precision($target.precision),
            flags: Flags::exact_for($target.precision),
            encode: |args| {
                uncompressed_untyped(
                    args,
                    $target.bytes_per_pixel() as usize,
                    simple_color_convert($target, false),
                )
            },
        }
    };
    ($target:expr, SNORM) => {
        BaseEncoder {
            color_formats: ColorFormatSet::from_precision($target.precision),
            flags: Flags::exact_for($target.precision),
            encode: |args| {
                uncompressed_untyped(
                    args,
                    $target.bytes_per_pixel() as usize,
                    simple_color_convert($target, true),
                )
            },
        }
    };
}

macro_rules! universal {
    ($out:ty, $f:expr) => {{
        fn process_line(line: &[[f32; 4]], out: &mut [$out]) {
            assert!(line.len() == out.len());
            let f = util::closure_types::<[f32; 4], $out, _>($f);
            for (i, o) in line.iter().zip(out.iter_mut()) {
                *o = f(*i);
            }
        }
        BaseEncoder {
            color_formats: ColorFormatSet::ALL,
            flags: Flags::empty(),
            encode: |args| uncompressed_universal(args, process_line),
        }
    }};
}
macro_rules! universal_grayscale {
    ($out:ty, $f:expr) => {
        universal!($out, |rgba| ($f)(ch::rgba_to_grayscale(rgba)[0]))
    };
}
macro_rules! universal_dither {
    ($out:ty, $f:expr) => {
        BaseEncoder {
            color_formats: ColorFormatSet::ALL,
            flags: Flags::empty(),
            encode: |args| uncompressed_universal_dither::<$out, _>(args, $f),
        }
    };
}

// encoders

pub const R8G8B8_UNORM: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::RGB_U8),
    color_convert!(ColorFormat::RGB_U8),
    universal!([u8; 3], |[r, g, b, _]| [r, g, b].map(n8::from_f32)),
];

pub const B8G8R8_UNORM: &[BaseEncoder] = &[
    BaseEncoder {
        color_formats: ColorFormatSet::U8,
        flags: Flags::EXACT_U8,
        encode: |args| {
            fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
                assert!(color.precision == Precision::U8);
                convert_channels_untyped::<u8>(color.channels, Channels::Rgb, line, out);

                // swap R and B
                let chunked: &mut [[u8; 3]] =
                    cast::as_array_chunks_mut(out).expect("invalid buffer size");
                chunked.iter_mut().for_each(|p| p.swap(0, 2));
            }

            uncompressed_untyped(args, 3, process_line)
        },
    },
    universal!([u8; 3], |[r, g, b, _]| [b, g, r].map(n8::from_f32)),
];

pub const R8G8B8A8_UNORM: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::RGBA_U8),
    color_convert!(ColorFormat::RGBA_U8),
    universal!([u8; 4], |rgba| rgba.map(n8::from_f32)),
];

pub const R8G8B8A8_SNORM: &[BaseEncoder] = &[
    color_convert!(ColorFormat::RGBA_U8, SNORM),
    universal!([u8; 4], |rgba| rgba.map(s8::from_uf32)),
];

pub const B8G8R8A8_UNORM: &[BaseEncoder] = &[
    BaseEncoder {
        color_formats: ColorFormatSet::U8,
        flags: Flags::EXACT_U8,
        encode: |args| {
            fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
                assert!(color.precision == Precision::U8);
                convert_channels_untyped::<u8>(color.channels, Channels::Rgba, line, out);

                // swap R and B
                let chunked: &mut [[u8; 4]] =
                    cast::as_array_chunks_mut(out).expect("invalid buffer size");
                chunked.iter_mut().for_each(|p| p.swap(0, 2));
            }

            uncompressed_untyped(args, 4, process_line)
        },
    },
    universal!([u8; 4], |[r, g, b, a]| [b, g, r, a].map(n8::from_f32)),
];

pub const B8G8R8X8_UNORM: &[BaseEncoder] = &[
    BaseEncoder {
        color_formats: ColorFormatSet::U8,
        flags: Flags::EXACT_U8,
        encode: |args| {
            fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
                assert!(color.precision == Precision::U8);
                convert_channels_untyped::<u8>(color.channels, Channels::Rgba, line, out);

                // swap R and B and set X to 0xFF
                let chunked: &mut [[u8; 4]] =
                    cast::as_array_chunks_mut(out).expect("invalid buffer size");
                chunked.iter_mut().for_each(|p| {
                    p.swap(0, 2);
                    p[3] = 0xFF;
                });
            }

            uncompressed_untyped(args, 4, process_line)
        },
    },
    universal!([u8; 4], |[r, g, b, _]| [
        n8::from_f32(b),
        n8::from_f32(g),
        n8::from_f32(r),
        0xFF
    ]),
];

pub const B5G6R5_UNORM: &[BaseEncoder] = &[
    universal!(u16, |[r, g, b, _]| {
        let r = n5::from_f32(r) as u16;
        let g = n6::from_f32(g) as u16;
        let b = n5::from_f32(b) as u16;
        b | (g << 5) | (r << 11)
    }),
    universal_dither!(u16, |pixel| {
        let r = n5::from_f32(pixel[0]) as u16;
        let g = n6::from_f32(pixel[1]) as u16;
        let b = n5::from_f32(pixel[2]) as u16;

        let back = [n5::f32(r as u8), n6::f32(g as u8), n5::f32(b as u8), 1.0];
        let error = f4_sub(pixel, back);

        (b | (g << 5) | (r << 11), error)
    })
    .add_flags(Flags::DITHER_COLOR),
];

pub const B5G5R5A1_UNORM: &[BaseEncoder] = &[
    universal!(u16, |[r, g, b, a]| {
        let r = n5::from_f32(r) as u16;
        let g = n5::from_f32(g) as u16;
        let b = n5::from_f32(b) as u16;
        let a = n1::from_f32(a) as u16;
        b | (g << 5) | (r << 10) | (a << 15)
    }),
    universal_dither!(u16, |pixel| {
        let r = n5::from_f32(pixel[0]) as u16;
        let g = n5::from_f32(pixel[1]) as u16;
        let b = n5::from_f32(pixel[2]) as u16;
        let a = n1::from_f32(pixel[3]) as u16;

        let back = [
            n5::f32(r as u8),
            n5::f32(g as u8),
            n5::f32(b as u8),
            n1::f32(a as u8),
        ];
        let error = f4_sub(pixel, back);

        (b | (g << 5) | (r << 10) | (a << 15), error)
    })
    .add_flags(Flags::DITHER_ALL),
];

fn rgba4_encode_with_error(pixel: [f32; 4]) -> ([u8; 4], [f32; 4]) {
    let encoded = pixel.map(n4::from_f32);
    let back = encoded.map(n4::f32);
    let error = f4_sub(pixel, back);
    (encoded, error)
}

pub const B4G4R4A4_UNORM: &[BaseEncoder] = &[
    universal!(u16, |[r, g, b, a]| {
        let r = n4::from_f32(r) as u16;
        let g = n4::from_f32(g) as u16;
        let b = n4::from_f32(b) as u16;
        let a = n4::from_f32(a) as u16;
        b | (g << 4) | (r << 8) | (a << 12)
    }),
    universal_dither!(u16, |pixel| {
        let (encoded, error) = rgba4_encode_with_error(pixel);
        let [r, g, b, a] = encoded.map(|c| c as u16);
        (b | (g << 4) | (r << 8) | (a << 12), error)
    })
    .add_flags(Flags::DITHER_ALL),
];

pub const A4B4G4R4_UNORM: &[BaseEncoder] = &[
    universal!(u16, |[r, g, b, a]| {
        let r = n4::from_f32(r) as u16;
        let g = n4::from_f32(g) as u16;
        let b = n4::from_f32(b) as u16;
        let a = n4::from_f32(a) as u16;
        a | (b << 4) | (g << 8) | (r << 12)
    }),
    universal_dither!(u16, |pixel| {
        let (encoded, error) = rgba4_encode_with_error(pixel);
        let [r, g, b, a] = encoded.map(|c| c as u16);
        (a | (b << 4) | (g << 8) | (r << 12), error)
    })
    .add_flags(Flags::DITHER_ALL),
];

pub const R8_UNORM: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::GRAYSCALE_U8),
    color_convert!(ColorFormat::GRAYSCALE_U8),
    universal_grayscale!(u8, n8::from_f32),
];

pub const R8_SNORM: &[BaseEncoder] = &[
    color_convert!(ColorFormat::GRAYSCALE_U8, SNORM),
    universal_grayscale!(u8, s8::from_uf32),
];

pub const R8G8_UNORM: &[BaseEncoder] =
    &[universal!([u8; 2], |[r, g, _, _]| [r, g].map(n8::from_f32)).add_flags(Flags::EXACT_U8)];

pub const R8G8_SNORM: &[BaseEncoder] =
    &[universal!([u8; 2], |[r, g, _, _]| [r, g].map(s8::from_uf32)).add_flags(Flags::EXACT_U8)];

pub const A8_UNORM: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::ALPHA_U8),
    color_convert!(ColorFormat::ALPHA_U8),
    universal!(u8, |[_, _, _, a]| n8::from_f32(a)),
];

pub const R16_UNORM: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::GRAYSCALE_U16),
    color_convert!(ColorFormat::GRAYSCALE_U16),
    universal_grayscale!(u16, n16::from_f32),
];

pub const R16_SNORM: &[BaseEncoder] = &[
    color_convert!(ColorFormat::GRAYSCALE_U16, SNORM),
    universal_grayscale!(u16, s16::from_uf32),
];

pub const R16G16_UNORM: &[BaseEncoder] =
    &[universal!([u16; 2], |[r, g, _, _]| [r, g].map(n16::from_f32)).add_flags(Flags::EXACT_U16)];

pub const R16G16_SNORM: &[BaseEncoder] =
    &[universal!([u16; 2], |[r, g, _, _]| [r, g].map(s16::from_uf32)).add_flags(Flags::EXACT_U16)];

pub const R16G16B16A16_UNORM: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::RGBA_U16),
    color_convert!(ColorFormat::RGBA_U16),
    universal!([u16; 4], |rgba| rgba.map(n16::from_f32)),
];

pub const R16G16B16A16_SNORM: &[BaseEncoder] = &[
    color_convert!(ColorFormat::RGBA_U16, SNORM),
    universal!([u16; 4], |rgba| rgba.map(s16::from_uf32)),
];

pub const R10G10B10A2_UNORM: &[BaseEncoder] = &[
    universal!(u32, |[r, g, b, a]| {
        let r = n10::from_f32(r) as u32;
        let g = n10::from_f32(g) as u32;
        let b = n10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;
        (a << 30) | (b << 20) | (g << 10) | r
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a] = pixel;
        let r = n10::from_f32(r) as u32;
        let g = n10::from_f32(g) as u32;
        let b = n10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;

        let back = [
            n10::f32(r as u16),
            n10::f32(g as u16),
            n10::f32(b as u16),
            n2::f32(a as u8),
        ];
        let error = f4_sub(pixel, back);

        ((a << 30) | (b << 20) | (g << 10) | r, error)
    })
    .add_flags(Flags::DITHER_ALL),
];

pub const R11G11B10_FLOAT: &[BaseEncoder] = &[
    universal!(u32, |[r, g, b, _]| {
        let r11 = fp11::from_f32(r) as u32;
        let g11 = fp11::from_f32(g) as u32;
        let b10 = fp10::from_f32(b) as u32;
        (b10 << 22) | (g11 << 11) | r11
    }),
    universal_dither!(u32, |pixel| {
        let r11 = fp11::from_f32(pixel[0]) as u32;
        let g11 = fp11::from_f32(pixel[1]) as u32;
        let b10 = fp10::from_f32(pixel[2]) as u32;

        let back = [
            fp11::f32(r11 as u16),
            fp11::f32(g11 as u16),
            fp10::f32(b10 as u16),
            1.0,
        ];
        let error = f4_sub(pixel, back);

        ((b10 << 22) | (g11 << 11) | r11, error)
    })
    .add_flags(Flags::DITHER_COLOR),
];

pub const R9G9B9E5_SHAREDEXP: &[BaseEncoder] =
    &[
        universal!(u32, |[r, g, b, _]| { rgb9995f::from_f32([r, g, b]) })
            .add_flags(Flags::EXACT_U8),
    ];

pub const R16_FLOAT: &[BaseEncoder] =
    &[universal_grayscale!(u16, fp16::from_f32).add_flags(Flags::EXACT_U8)];

pub const R16G16_FLOAT: &[BaseEncoder] =
    &[universal!([u16; 2], |[r, g, _, _]| [r, g].map(fp16::from_f32)).add_flags(Flags::EXACT_U8)];

pub const R16G16B16A16_FLOAT: &[BaseEncoder] =
    &[universal!([u16; 4], |rgba| rgba.map(fp16::from_f32)).add_flags(Flags::EXACT_U8)];

pub const R32_FLOAT: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::GRAYSCALE_F32),
    color_convert!(ColorFormat::GRAYSCALE_F32),
    universal_grayscale!(f32, |r| r),
];

pub const R32G32_FLOAT: &[BaseEncoder] =
    &[universal!([f32; 2], |[r, g, _, _]| [r, g]).add_flags(Flags::EXACT_F32)];

pub const R32G32B32_FLOAT: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::RGB_F32),
    color_convert!(ColorFormat::RGB_F32),
    universal!([f32; 3], |[r, g, b, _]| [r, g, b]),
];

pub const R32G32B32A32_FLOAT: &[BaseEncoder] = &[
    BaseEncoder::copy(ColorFormat::RGBA_F32),
    color_convert!(ColorFormat::RGBA_F32),
    universal!([f32; 4], |[r, g, b, a]| [r, g, b, a]),
];

pub const R10G10B10_XR_BIAS_A2_UNORM: &[BaseEncoder] = &[
    universal!(u32, |[r, g, b, a]| {
        let r = xr10::from_f32(r) as u32;
        let g = xr10::from_f32(g) as u32;
        let b = xr10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;
        (a << 30) | (b << 20) | (g << 10) | r
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a] = pixel;
        let r = xr10::from_f32(r) as u32;
        let g = xr10::from_f32(g) as u32;
        let b = xr10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;

        let back = [
            xr10::f32(r as u16),
            xr10::f32(g as u16),
            xr10::f32(b as u16),
            n2::f32(a as u8),
        ];
        let error = f4_sub(pixel, back);

        ((a << 30) | (b << 20) | (g << 10) | r, error)
    })
    .add_flags(Flags::DITHER_ALL),
];

pub const AYUV: &[BaseEncoder] = &[universal!([u8; 4], |[r, g, b, a]| {
    let [y, u, v] = yuv8::from_rgb_f32([r, g, b]);
    let a = n8::from_f32(a);
    [v, u, y, a]
})];

pub const Y410: &[BaseEncoder] = &[
    universal!(u32, |[r, g, b, a]| {
        let [y, u, v] = yuv10::from_rgb_f32([r, g, b]);
        let a = n2::from_f32(a) as u32;
        (a << 30) | ((v as u32) << 20) | ((y as u32) << 10) | (u as u32)
    }),
    universal_dither!(u32, |[r, g, b, a_f32]| {
        let [y, u, v] = yuv10::from_rgb_f32([r, g, b]);
        let a = n2::from_f32(a_f32) as u32;

        let a_back = n2::f32(a as u8);
        let error = [0.0, 0.0, 0.0, a_f32 - a_back];

        (
            (a << 30) | ((v as u32) << 20) | ((y as u32) << 10) | (u as u32),
            error,
        )
    })
    .add_flags(Flags::DITHER_ALPHA),
];

pub const Y416: &[BaseEncoder] = &[universal!([u16; 4], |[r, g, b, a]| {
    let [y, u, v] = yuv16::from_rgb_f32([r, g, b]);
    let a = n16::from_f32(a);
    [u, y, v, a]
})
.add_flags(Flags::EXACT_U8)];
