use crate::{as_rgba_f32, cast, ch, n1, n8, util, yuv16, yuv8, ColorFormatSet, EncodeError};

use super::encoder::{Args, Encoder, EncoderSet, Flags};

// helpers

fn process_subsample<const BLOCK_WIDTH: usize, EncodedBlock, F>(
    data: &[[f32; 4]],
    out: &mut [EncodedBlock],
    f: F,
) where
    F: Fn(&[[f32; 4]; BLOCK_WIDTH]) -> EncodedBlock,
{
    let full_blocks_len = data.len() / BLOCK_WIDTH * BLOCK_WIDTH;
    let rest = data.len() - full_blocks_len;

    // process full blocks
    let full = cast::as_array_chunks(&data[..full_blocks_len]).unwrap();
    for (i, o) in full.iter().zip(out.iter_mut()) {
        *o = f(i);
    }

    // process the last partial block (if any)
    if rest > 0 {
        let mut last_block = [[0_f32; 4]; BLOCK_WIDTH];
        last_block[..rest].copy_from_slice(&data[full_blocks_len..]);
        last_block[rest..].fill(data[data.len() - 1]);

        out[full.len()] = f(&last_block);
    }
}
fn uncompressed_universal_subsample<EncodedBlock>(
    args: Args,
    block_width: usize,
    process: fn(&[[f32; 4]], &mut [EncodedBlock]),
) -> Result<(), EncodeError>
where
    EncodedBlock: Default + Copy + cast::ToLe + cast::Castable,
{
    let Args {
        data,
        color,
        writer,
        width,
        ..
    } = args;
    let bytes_per_pixel = color.bytes_per_pixel() as usize;

    assert!(block_width >= 2);

    const BUFFER_PIXELS: usize = 512;
    let mut intermediate_buffer = [[0_f32; 4]; BUFFER_PIXELS];
    let mut encoded_buffer = [EncodedBlock::default(); BUFFER_PIXELS / 2];

    for y_line in data.chunks(width * bytes_per_pixel) {
        debug_assert!(y_line.len() == width * bytes_per_pixel);

        let chunk_pixels = BUFFER_PIXELS / block_width * block_width;
        let chunk_size = chunk_pixels * bytes_per_pixel;
        for chunk in y_line.chunks(chunk_size) {
            let pixels = chunk.len() / bytes_per_pixel;

            let intermediate = &mut intermediate_buffer[..pixels];
            let encoded = &mut encoded_buffer[..util::div_ceil(pixels, block_width)];

            process(as_rgba_f32(color, chunk, intermediate), encoded);

            cast::ToLe::to_le(encoded);

            writer.write_all(cast::as_bytes(encoded))?;
        }
    }

    Ok(())
}

macro_rules! universal_subsample {
    ($block_width:literal, $out:ty, $f:expr) => {{
        fn process_blocks(block: &[[f32; 4]], out: &mut [$out]) {
            process_subsample::<$block_width, $out, _>(block, out, $f);
        }
        Encoder {
            color_formats: ColorFormatSet::ALL,
            flags: Flags::empty(),
            encode: |args| uncompressed_universal_subsample(args, $block_width, process_blocks),
        }
    }};
}

// encoders

fn to_rgbg([p0, p1]: &[[f32; 4]; 2]) -> [u8; 4] {
    let g0 = n8::from_f32(p0[1]);
    let g1 = n8::from_f32(p1[1]);
    let r = n8::from_f32((p0[0] + p1[0]) * 0.5);
    let b = n8::from_f32((p0[2] + p1[2]) * 0.5);
    [r, g0, b, g1]
}

pub(crate) const R8G8_B8G8_UNORM: EncoderSet =
    EncoderSet::new(&[universal_subsample!(2, [u8; 4], to_rgbg).add_flags(Flags::EXACT_U8)]);

pub(crate) const G8R8_G8B8_UNORM: EncoderSet =
    EncoderSet::new(&[universal_subsample!(2, [u8; 4], |pair| {
        let [r, g0, b, g1] = to_rgbg(pair);
        [g0, r, g1, b]
    })
    .add_flags(Flags::EXACT_U8)]);

fn to_yuy2([p0, p1]: &[[f32; 4]; 2]) -> [u8; 4] {
    let yuv1 = yuv8::from_rgb_f32([p0[0], p0[1], p0[2]]);
    let yuv2 = yuv8::from_rgb_f32([p1[0], p1[1], p1[2]]);
    let y0 = yuv1[0];
    let y1 = yuv2[0];
    fn pick_mid(a: u8, b: u8) -> u8 {
        let a = a as u16;
        let b = b as u16;
        ((a + b) / 2) as u8
    }
    let u = pick_mid(yuv1[1], yuv2[1]);
    let v = pick_mid(yuv1[2], yuv2[2]);
    [y0, u, y1, v]
}

pub(crate) const YUY2: EncoderSet = EncoderSet::new(&[universal_subsample!(2, [u8; 4], to_yuy2)]);

pub(crate) const UYVY: EncoderSet = EncoderSet::new(&[universal_subsample!(2, [u8; 4], |pair| {
    let [y0, u, y1, v] = to_yuy2(pair);
    [u, y0, v, y1]
})]);

fn to_y216([p0, p1]: &[[f32; 4]; 2]) -> [u16; 4] {
    let yuv1 = yuv16::from_rgb_f32([p0[0], p0[1], p0[2]]);
    let yuv2 = yuv16::from_rgb_f32([p1[0], p1[1], p1[2]]);
    let y0 = yuv1[0];
    let y1 = yuv2[0];
    fn pick_mid(a: u16, b: u16) -> u16 {
        let a = a as u32;
        let b = b as u32;
        ((a + b) / 2) as u16
    }
    let u = pick_mid(yuv1[1], yuv2[1]);
    let v = pick_mid(yuv1[2], yuv2[2]);
    [y0, u, y1, v]
}

pub(crate) const Y210: EncoderSet =
    EncoderSet::new(&[
        universal_subsample!(2, [u16; 4], |pair| to_y216(pair).map(|c| c & 0xFFC0))
            .add_flags(Flags::EXACT_U8),
    ]);

pub(crate) const Y216: EncoderSet =
    EncoderSet::new(&[universal_subsample!(2, [u16; 4], to_y216).add_flags(Flags::EXACT_U8)]);

pub(crate) const R1_UNORM: EncoderSet = EncoderSet::new(&[universal_subsample!(8, u8, |block| {
    let mut out = 0_u8;
    for (i, &p) in block.iter().enumerate() {
        out |= n1::from_f32(ch::rgba_to_grayscale(p)[0]) << (7 - i);
    }
    out
})]);
