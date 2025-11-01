use glam::Vec4;

use crate::{
    as_rgba_f32, cast, ch, convert_channels, convert_channels_for,
    encode::write_util::for_each_chunk, fp10, fp11, fp16, n1, n10, n16, n2, n4, n5, n6, n8,
    rgb9995f, s16, s8, util, xr10, yuv10, yuv16, yuv8, Channels, ColorFormat, ColorFormatSet,
    EncodingError, Precision, Report,
};

use super::{
    encoder::{Args, Encoder, EncoderSet, Flags},
    Dithering,
};

// helpers

const REPORT_FREQUENCY: usize = 2048;

fn uncompressed_universal<EncodedPixel>(
    args: Args,
    process: fn(&[[f32; 4]], &mut [EncodedPixel]),
) -> Result<(), EncodingError>
where
    EncodedPixel: Default + Copy + cast::ToLe + cast::Castable,
{
    let Args {
        image,
        writer,
        mut progress,
        ..
    } = args;
    let color = image.color();

    const BUFFER_PIXELS: usize = 512;
    let mut intermediate_buffer = [[0_f32; 4]; BUFFER_PIXELS];
    let mut encoded_buffer = [EncodedPixel::default(); BUFFER_PIXELS];

    let chunk_count = util::div_ceil(image.size().pixels() as usize, BUFFER_PIXELS);
    let mut chunk_index = 0;
    for_each_chunk(
        image,
        &mut encoded_buffer,
        1,
        |partial, encoded| {
            let intermediate = &mut intermediate_buffer[..encoded.len()];
            process(as_rgba_f32(color, partial, intermediate), encoded);
        },
        |encoded| {
            // occasionally report progress
            if chunk_index % REPORT_FREQUENCY == 0 {
                progress.report(chunk_index as f32 / chunk_count as f32);
            }
            chunk_index += 1;

            cast::ToLe::to_le(encoded);
            writer.write_all(cast::as_bytes(encoded))
        },
    )?;

    Ok(())
}

type DitherProcessChunkFn = fn(
    chunk: &[[f32; 4]],
    encoded: &mut [u8],
    error_mask: Vec4,
    next_error_add: Vec4,
    current_line_error: &[Vec4],
    next_line_error: &mut [Vec4],
) -> Vec4;
fn uncompressed_universal_dither(
    args: Args,
    encoded_pixel_size: usize,
    encoded_pixel_align: usize,
    process_chunk: DitherProcessChunkFn,
) -> Result<(), EncodingError> {
    let Args {
        image,
        writer,
        options,
        mut progress,
        ..
    } = args;
    let color = image.color();
    let bytes_per_pixel = color.bytes_per_pixel() as usize;
    let width = image.width() as usize;
    let height = image.height() as usize;

    type EncodedBufferType = u64;
    assert!(encoded_pixel_align <= std::mem::align_of::<EncodedBufferType>());

    let error_padding = 2;
    let mut error_buffer = vec![Vec4::ZERO; 2 * (width + error_padding * 2)];
    let (mut current_line_error, mut next_line_error) =
        error_buffer.split_at_mut(width + error_padding * 2);

    let error_mask = match options.dithering {
        Dithering::None => Vec4::ZERO,
        Dithering::ColorAndAlpha => Vec4::ONE,
        Dithering::Color => Vec4::new(1.0, 1.0, 1.0, 0.0),
        Dithering::Alpha => Vec4::new(0.0, 0.0, 0.0, 1.0),
    };

    const BUFFER_PIXELS: usize = 512;
    let mut intermediate_buffer = [[0_f32; 4]; BUFFER_PIXELS];
    let mut encoded_buffer = [EncodedBufferType::default(); BUFFER_PIXELS];
    let encoded_buffer: &mut [u8] = cast::as_bytes_mut(&mut encoded_buffer[..]);

    let chunk_pixels = usize::min(BUFFER_PIXELS, encoded_buffer.len() / encoded_pixel_size);
    let chunk_size = chunk_pixels * bytes_per_pixel;
    let chunk_count = height * util::div_ceil(width * bytes_per_pixel, chunk_size);
    let mut chunk_index: usize = 0;
    for row in image.rows() {
        debug_assert!(row.len() == width * bytes_per_pixel);

        // prepare error buffers
        std::mem::swap(&mut current_line_error, &mut next_line_error);
        next_line_error.fill(Vec4::ZERO);
        let mut error_offset = error_padding;
        let mut next_error_add = Vec4::ZERO;

        for line in row.chunks(chunk_size) {
            // occasionally report progress
            if chunk_index % REPORT_FREQUENCY == 0 {
                progress.report(chunk_index as f32 / chunk_count as f32);
            }
            chunk_index += 1;

            debug_assert!(line.len() % bytes_per_pixel == 0);
            let pixels = line.len() / bytes_per_pixel;

            let intermediate = &mut intermediate_buffer[..pixels];
            let encoded = &mut encoded_buffer[..pixels * encoded_pixel_size];
            let intermediate = as_rgba_f32(color, line, intermediate);

            next_error_add = process_chunk(
                intermediate,
                encoded,
                error_mask,
                next_error_add,
                &current_line_error[error_offset..(error_offset + pixels)],
                &mut next_line_error[error_offset - 1..(error_offset + pixels + 1)],
            );
            error_offset += pixels;

            cast::ToLe::to_le(encoded);
            writer.write_all(cast::as_bytes(encoded))?;
        }
    }

    Ok(())
}

fn uncompressed_untyped(
    args: Args,
    bytes_per_encoded_pixel: usize,
    f: fn(&[u8], ColorFormat, &mut [u8]),
) -> Result<(), EncodingError> {
    let Args {
        image,
        writer,
        mut progress,
        ..
    } = args;
    let color = image.color();

    let mut raw_buffer = [0_u32; 1024];
    let encoded_buffer = cast::as_bytes_mut(&mut raw_buffer);

    let chunk_count = util::div_ceil(
        image.size().pixels() as usize,
        encoded_buffer.len() / bytes_per_encoded_pixel,
    );
    let mut chunk_index = 0;

    for_each_chunk(
        image,
        encoded_buffer,
        bytes_per_encoded_pixel,
        |partial, encoded| f(partial, color, encoded),
        |encoded| {
            // occasionally report progress
            if chunk_index % REPORT_FREQUENCY == 0 {
                progress.report(chunk_index as f32 / chunk_count as f32);
            }
            chunk_index += 1;

            writer.write_all(encoded)
        },
    )?;

    Ok(())
}
fn simple_color_convert(
    line: &[u8],
    color: ColorFormat,
    out: &mut [u8],
    target: ColorFormat,
    snorm: bool,
) {
    assert!(color.precision == target.precision);

    convert_channels_for(color, target.channels, line, out);

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

    cast::slice_ne_to_le(target.precision, out);
}

macro_rules! color_convert {
    ($target:expr, snorm = $snorm:literal) => {
        Encoder::new(
            ColorFormatSet::from_precision($target.precision),
            Flags::exact_for($target.precision),
            |args| {
                uncompressed_untyped(
                    args,
                    $target.bytes_per_pixel() as usize,
                    |line, color, out| simple_color_convert(line, color, out, $target, $snorm),
                )
            },
        )
    };
    ($target:expr) => {
        color_convert!($target, snorm = false)
    };
}

macro_rules! universal {
    ($out:ty, gray = $f:expr) => {{
        universal!($out, adopt_gray($f))
    }};
    ($out:ty, rg = $f:expr) => {{
        universal!($out, adopt_rg($f))
    }};
    ($out:ty, rgba = $f:expr) => {{
        universal!($out, adopt_rgba($f))
    }};
    ($out:ty, $f:expr) => {{
        type Out = $out;
        fn process_line(line: &[[f32; 4]], out: &mut [Out]) {
            debug_assert!(line.len() == out.len());
            let f = util::closure_types::<[f32; 4], Out, _>($f);

            for (i, o) in line.iter().zip(out.iter_mut()) {
                *o = f(*i);
            }
        }
        Encoder::new_universal(|args| uncompressed_universal(args, process_line))
    }};
}
macro_rules! universal_dither {
    ($out:ty, gray = $f:expr, $g: expr) => {
        universal_dither!($out, adopt_dither_gray($f, $g)).add_flags(Flags::DITHER_COLOR)
    };
    ($out:ty, rg = $f:expr, $g: expr) => {
        universal_dither!($out, adopt_dither_rg($f, $g)).add_flags(Flags::DITHER_COLOR)
    };
    ($out:ty, rgba = $f:expr, $g: expr) => {
        universal_dither!($out, adopt_dither_rgba($f, $g)).add_flags(Flags::DITHER_ALL)
    };
    ($out:ty, $f:expr) => {{
        type Out = $out;
        fn process_chunk(
            chunk: &[[f32; 4]],
            encoded: &mut [u8],
            error_mask: Vec4,
            mut next_error_add: Vec4,
            current_line_error: &[Vec4],
            next_line_error: &mut [Vec4],
        ) -> Vec4 {
            let encoded: &mut [Out] =
                cast::from_bytes_mut(encoded).expect("invalid encoded buffer size");

            debug_assert_eq!(chunk.len(), encoded.len());
            debug_assert_eq!(chunk.len(), current_line_error.len());
            debug_assert_eq!(chunk.len() + 2, next_line_error.len());

            let f = util::closure_types::<Vec4, (Out, Vec4), _>($f);

            let mut error_offset: usize = 1;
            for ((&pixel, out), &current_error) in chunk
                .iter()
                .zip(encoded.iter_mut())
                .zip(current_line_error.iter())
            {
                let error = current_error + next_error_add;
                let (encoded_pixel, mut error) = f(Vec4::from(pixel) + error);

                // diffuse error with Floyd-Steinberg weights
                error *= error_mask;
                next_error_add = error * (7.0 / 16.0);
                next_line_error[error_offset - 1] += error * (3.0 / 16.0);
                next_line_error[error_offset] += error * (5.0 / 16.0);
                next_line_error[error_offset + 1] += error * (1.0 / 16.0);

                *out = encoded_pixel;
                error_offset += 1;
            }

            cast::ToLe::to_le(encoded);

            next_error_add
        }

        Encoder::new_universal(|args| {
            uncompressed_universal_dither(
                args,
                std::mem::size_of::<Out>(),
                std::mem::align_of::<Out>(),
                process_chunk,
            )
        })
    }};
}

fn adopt_gray<Out>(f: impl Fn(f32) -> Out) -> impl Fn([f32; 4]) -> Out {
    move |rgba: [f32; 4]| f(ch::rgba_to_grayscale(rgba)[0])
}
fn adopt_rg<Out>(f: impl Fn(f32) -> Out) -> impl Fn([f32; 4]) -> [Out; 2] {
    move |rgba: [f32; 4]| [f(rgba[0]), f(rgba[1])]
}
fn adopt_rgba<Out>(f: impl Fn(f32) -> Out) -> impl Fn([f32; 4]) -> [Out; 4] {
    move |rgba: [f32; 4]| [f(rgba[0]), f(rgba[1]), f(rgba[2]), f(rgba[3])]
}

fn adopt_dither_gray<Out: Copy>(
    f: impl Fn(f32) -> Out,
    g: impl Fn(Out) -> f32,
) -> impl Fn(Vec4) -> (Out, Vec4) {
    move |pixel: Vec4| {
        let out = f(pixel.x);
        let back = Vec4::new(g(out), 1.0, 1.0, 1.0);
        let error = pixel - back;
        (out, error)
    }
}
fn adopt_dither_rg<Out: Copy>(
    f: impl Fn(f32) -> Out,
    g: impl Fn(Out) -> f32,
) -> impl Fn(Vec4) -> ([Out; 2], Vec4) {
    move |pixel: Vec4| {
        let out = [f(pixel.x), f(pixel.y)];
        let back = Vec4::new(g(out[0]), g(out[1]), 1.0, 1.0);
        let error = pixel - back;
        (out, error)
    }
}
fn adopt_dither_rgba<Out: Copy>(
    f: impl Fn(f32) -> Out,
    g: impl Fn(Out) -> f32,
) -> impl Fn(Vec4) -> ([Out; 4], Vec4) {
    move |pixel: Vec4| {
        let out = [f(pixel.x), f(pixel.y), f(pixel.z), f(pixel.w)];
        let back = Vec4::new(g(out[0]), g(out[1]), g(out[2]), g(out[3]));
        let error = pixel - back;
        (out, error)
    }
}

// encoders

pub(crate) const R8G8B8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGB_U8),
    color_convert!(ColorFormat::RGB_U8),
    universal!([u8; 3], |[r, g, b, _]| [r, g, b].map(n8::from_f32)),
    universal_dither!([u8; 3], |pixel| {
        let r = n8::from_f32(pixel[0]);
        let g = n8::from_f32(pixel[1]);
        let b = n8::from_f32(pixel[2]);

        let back = Vec4::new(n8::f32(r), n8::f32(g), n8::f32(b), 1.0);
        let error = pixel - back;

        ([r, g, b], error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const B8G8R8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::new(ColorFormatSet::U8, Flags::EXACT_U8, |args| {
        fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
            assert!(color.precision == Precision::U8);
            convert_channels::<u8>(color.channels, Channels::Rgb, line, out);

            // swap R and B
            let chunked: &mut [[u8; 3]] =
                cast::as_array_chunks_mut(out).expect("invalid buffer size");
            chunked.iter_mut().for_each(|p| p.swap(0, 2));
        }

        uncompressed_untyped(args, 3, process_line)
    }),
    universal!([u8; 3], |[r, g, b, _]| [b, g, r].map(n8::from_f32)),
    universal_dither!([u8; 3], |pixel| {
        let r = n8::from_f32(pixel[0]);
        let g = n8::from_f32(pixel[1]);
        let b = n8::from_f32(pixel[2]);

        let back = Vec4::new(n8::f32(r), n8::f32(g), n8::f32(b), 1.0);
        let error = pixel - back;

        ([b, g, r], error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const R8G8B8A8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGBA_U8),
    color_convert!(ColorFormat::RGBA_U8),
    universal!([u8; 4], rgba = n8::from_f32),
    universal_dither!([u8; 4], rgba = n8::from_f32, n8::f32),
]);

pub(crate) const R8G8B8A8_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::RGBA_U8, snorm = true),
    universal!([u8; 4], rgba = s8::from_uf32),
    universal_dither!([u8; 4], rgba = s8::from_uf32, s8::uf32),
]);

pub(crate) const B8G8R8A8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::new(ColorFormatSet::U8, Flags::EXACT_U8, |args| {
        fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
            assert!(color.precision == Precision::U8);
            convert_channels::<u8>(color.channels, Channels::Rgba, line, out);

            // swap R and B
            let chunked: &mut [[u8; 4]] =
                cast::as_array_chunks_mut(out).expect("invalid buffer size");
            chunked.iter_mut().for_each(|p| p.swap(0, 2));
        }

        uncompressed_untyped(args, 4, process_line)
    }),
    universal!([u8; 4], |[r, g, b, a]| [b, g, r, a].map(n8::from_f32)),
    universal_dither!([u8; 4], |pixel| {
        let r = n8::from_f32(pixel[0]);
        let g = n8::from_f32(pixel[1]);
        let b = n8::from_f32(pixel[2]);
        let a = n8::from_f32(pixel[3]);

        let back = Vec4::new(n8::f32(r), n8::f32(g), n8::f32(b), n8::f32(a));
        let error = pixel - back;

        ([b, g, r, a], error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const B8G8R8X8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::new(ColorFormatSet::U8, Flags::EXACT_U8, |args| {
        fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
            assert!(color.precision == Precision::U8);
            convert_channels::<u8>(color.channels, Channels::Rgba, line, out);

            // swap R and B and set X to 0xFF
            let chunked: &mut [[u8; 4]] =
                cast::as_array_chunks_mut(out).expect("invalid buffer size");
            chunked.iter_mut().for_each(|p| {
                p.swap(0, 2);
                p[3] = 0xFF;
            });
        }

        uncompressed_untyped(args, 4, process_line)
    }),
    universal!([u8; 4], |[r, g, b, _]| [
        n8::from_f32(b),
        n8::from_f32(g),
        n8::from_f32(r),
        0xFF
    ]),
    universal_dither!([u8; 4], |pixel| {
        let r = n8::from_f32(pixel[0]);
        let g = n8::from_f32(pixel[1]);
        let b = n8::from_f32(pixel[2]);

        let back = Vec4::new(n8::f32(r), n8::f32(g), n8::f32(b), 1.0);
        let error = pixel - back;

        ([b, g, r, 0xFF], error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const B5G6R5_UNORM: EncoderSet = EncoderSet::new(&[
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

        let back = Vec4::new(n5::f32(r as u8), n6::f32(g as u8), n5::f32(b as u8), 1.0);
        let error = pixel - back;

        (b | (g << 5) | (r << 11), error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const B5G5R5A1_UNORM: EncoderSet = EncoderSet::new(&[
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

        let back = Vec4::new(
            n5::f32(r as u8),
            n5::f32(g as u8),
            n5::f32(b as u8),
            n1::f32(a as u8),
        );
        let error = pixel - back;

        (b | (g << 5) | (r << 10) | (a << 15), error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

fn rgba4_encode_with_error(pixel: Vec4) -> ([u8; 4], Vec4) {
    let encoded = pixel.to_array().map(n4::from_f32);
    let back = Vec4::from(encoded.map(n4::f32));
    let error = pixel - back;
    (encoded, error)
}

pub(crate) const B4G4R4A4_UNORM: EncoderSet = EncoderSet::new(&[
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
]);

pub(crate) const A4B4G4R4_UNORM: EncoderSet = EncoderSet::new(&[
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
]);

pub(crate) const R8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::GRAYSCALE_U8),
    color_convert!(ColorFormat::GRAYSCALE_U8),
    universal!(u8, gray = n8::from_f32),
    universal_dither!(u8, gray = n8::from_f32, n8::f32),
]);

pub(crate) const R8_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::GRAYSCALE_U8, snorm = true),
    universal!(u8, gray = s8::from_uf32),
    universal_dither!(u8, gray = s8::from_uf32, s8::uf32),
]);

pub(crate) const R8G8_UNORM: EncoderSet = EncoderSet::new(&[
    universal!([u8; 2], rg = n8::from_f32).add_flags(Flags::EXACT_U8),
    universal_dither!([u8; 2], rg = n8::from_f32, n8::f32),
]);

pub(crate) const R8G8_SNORM: EncoderSet = EncoderSet::new(&[
    universal!([u8; 2], rg = s8::from_uf32).add_flags(Flags::EXACT_U8),
    universal_dither!([u8; 2], rg = s8::from_uf32, s8::uf32),
]);

pub(crate) const A8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::ALPHA_U8),
    color_convert!(ColorFormat::ALPHA_U8),
    universal!(u8, |[_, _, _, a]| n8::from_f32(a)),
    universal_dither!(u8, |pixel| {
        let a = n8::from_f32(pixel[3]);

        let back = Vec4::new(1.0, 1.0, 1.0, n8::f32(a));
        let error = pixel - back;

        (a, error)
    })
    .add_flags(Flags::DITHER_ALPHA),
]);

pub(crate) const R16_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::GRAYSCALE_U16),
    color_convert!(ColorFormat::GRAYSCALE_U16),
    universal!(u16, gray = n16::from_f32),
    universal_dither!(u16, gray = n16::from_f32, n16::f32),
]);

pub(crate) const R16_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::GRAYSCALE_U16, snorm = true),
    universal!(u16, gray = s16::from_uf32),
    universal_dither!(u16, gray = s16::from_uf32, s16::uf32),
]);

pub(crate) const R16G16_UNORM: EncoderSet = EncoderSet::new(&[
    universal!([u16; 2], rg = n16::from_f32).add_flags(Flags::EXACT_U16),
    universal_dither!([u16; 2], rg = n16::from_f32, n16::f32),
]);

pub(crate) const R16G16_SNORM: EncoderSet = EncoderSet::new(&[
    universal!([u16; 2], rg = s16::from_uf32).add_flags(Flags::EXACT_U16),
    universal_dither!([u16; 2], rg = s16::from_uf32, s16::uf32),
]);

pub(crate) const R16G16B16A16_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGBA_U16),
    color_convert!(ColorFormat::RGBA_U16),
    universal!([u16; 4], rgba = n16::from_f32),
    universal_dither!([u16; 4], rgba = n16::from_f32, n16::f32),
]);

pub(crate) const R16G16B16A16_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::RGBA_U16, snorm = true),
    universal!([u16; 4], rgba = s16::from_uf32),
    universal_dither!([u16; 4], rgba = s16::from_uf32, s16::uf32),
]);

pub(crate) const R10G10B10A2_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, a]| {
        let r = n10::from_f32(r) as u32;
        let g = n10::from_f32(g) as u32;
        let b = n10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;
        (a << 30) | (b << 20) | (g << 10) | r
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a] = pixel.to_array();
        let r = n10::from_f32(r) as u32;
        let g = n10::from_f32(g) as u32;
        let b = n10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;

        let back = Vec4::new(
            n10::f32(r as u16),
            n10::f32(g as u16),
            n10::f32(b as u16),
            n2::f32(a as u8),
        );
        let error = pixel - back;

        ((a << 30) | (b << 20) | (g << 10) | r, error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const R11G11B10_FLOAT: EncoderSet = EncoderSet::new(&[
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

        let back = Vec4::new(
            fp11::f32(r11 as u16),
            fp11::f32(g11 as u16),
            fp10::f32(b10 as u16),
            1.0,
        );
        let error = pixel - back;

        ((b10 << 22) | (g11 << 11) | r11, error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const R9G9B9E5_SHAREDEXP: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, _]| { rgb9995f::from_f32([r, g, b]) }).add_flags(Flags::EXACT_U8),
    universal_dither!(u32, |pixel| {
        let rgb = rgb9995f::from_f32([pixel[0], pixel[1], pixel[2]]);
        let back = rgb9995f::f32(rgb);

        let back = Vec4::new(back[0], back[1], back[2], 1.0);
        let error = pixel - back;

        (rgb, error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const R16_FLOAT: EncoderSet = EncoderSet::new(&[
    universal!(u16, gray = fp16::from_f32).add_flags(Flags::EXACT_U8),
    universal_dither!(u16, gray = fp16::from_f32, fp16::f32),
]);

pub(crate) const R16G16_FLOAT: EncoderSet = EncoderSet::new(&[
    universal!([u16; 2], rg = fp16::from_f32).add_flags(Flags::EXACT_U8),
    universal_dither!([u16; 2], rg = fp16::from_f32, fp16::f32),
]);

pub(crate) const R16G16B16A16_FLOAT: EncoderSet = EncoderSet::new(&[
    universal!([u16; 4], rgba = fp16::from_f32).add_flags(Flags::EXACT_U8),
    universal_dither!([u16; 4], rgba = fp16::from_f32, fp16::f32),
]);

pub(crate) const R32_FLOAT: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::GRAYSCALE_F32),
    color_convert!(ColorFormat::GRAYSCALE_F32),
    universal!(f32, gray = |r| r),
]);

pub(crate) const R32G32_FLOAT: EncoderSet =
    EncoderSet::new(&[universal!([f32; 2], rg = |rg| rg).add_flags(Flags::EXACT_F32)]);

pub(crate) const R32G32B32_FLOAT: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGB_F32),
    color_convert!(ColorFormat::RGB_F32),
    universal!([f32; 3], |[r, g, b, _]| [r, g, b]),
]);

pub(crate) const R32G32B32A32_FLOAT: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGBA_F32),
    color_convert!(ColorFormat::RGBA_F32),
    universal!([f32; 4], |rgba| rgba),
]);

pub(crate) const R10G10B10_XR_BIAS_A2_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, a]| {
        let r = xr10::from_f32(r) as u32;
        let g = xr10::from_f32(g) as u32;
        let b = xr10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;
        (a << 30) | (b << 20) | (g << 10) | r
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a] = pixel.to_array();
        let r = xr10::from_f32(r) as u32;
        let g = xr10::from_f32(g) as u32;
        let b = xr10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;

        let back = Vec4::new(
            xr10::f32(r as u16),
            xr10::f32(g as u16),
            xr10::f32(b as u16),
            n2::f32(a as u8),
        );
        let error = pixel - back;

        ((a << 30) | (b << 20) | (g << 10) | r, error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const AYUV: EncoderSet = EncoderSet::new(&[
    universal!([u8; 4], |[r, g, b, a]| {
        let [y, u, v] = yuv8::from_rgb_f32([r, g, b]);
        let a = n8::from_f32(a);
        [v, u, y, a]
    }),
    universal_dither!([u8; 4], |pixel| {
        let [y, u, v] = yuv8::from_rgb_f32([pixel[0], pixel[1], pixel[2]]);
        let a = n8::from_f32(pixel[3]);

        let [r, g, b] = yuv8::f32([y, u, v]);
        let back = Vec4::new(r, g, b, n8::f32(a));
        let error = pixel - back;

        ([v, u, y, a], error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const Y410: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, a]| {
        let [y, u, v] = yuv10::from_rgb_f32([r, g, b]);
        let a = n2::from_f32(a) as u32;
        (a << 30) | ((v as u32) << 20) | ((y as u32) << 10) | (u as u32)
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a] = pixel.to_array();
        let [y, u, v] = yuv10::from_rgb_f32([r, g, b]);
        let a = n2::from_f32(a) as u32;

        let [r, g, b] = yuv10::f32([y, u, v]);
        let back = Vec4::new(r, g, b, n2::f32(a as u8));
        let error = pixel - back;

        (
            (a << 30) | ((v as u32) << 20) | ((y as u32) << 10) | (u as u32),
            error,
        )
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const Y416: EncoderSet = EncoderSet::new(&[
    universal!([u16; 4], |[r, g, b, a]| {
        let [y, u, v] = yuv16::from_rgb_f32([r, g, b]);
        let a = n16::from_f32(a);
        [u, y, v, a]
    })
    .add_flags(Flags::EXACT_U8),
    universal_dither!([u16; 4], |pixel| {
        let [y, u, v] = yuv16::from_rgb_f32([pixel[0], pixel[1], pixel[2]]);
        let a = n16::from_f32(pixel[3]);

        let [r, g, b] = yuv16::f32([y, u, v]);
        let back = Vec4::new(r, g, b, n16::f32(a));
        let error = pixel - back;

        ([u, y, v, a], error)
    })
    .add_flags(Flags::DITHER_ALL),
]);
