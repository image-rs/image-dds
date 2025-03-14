//! An internal module with helper methods for reading bytes from a reader, and
//! writing decoded pixels to the output buffer.

use std::io::{Read, SeekFrom};
use std::mem::size_of;

use crate::{cast, util::div_ceil, DecodeError, Rect, Size};
use crate::{convert_channels_untyped_for, util, Channels, ColorFormat};

use super::ReadSeek;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PixelSize {
    /// The size of an encoded pixel in bytes.
    pub encoded_size: u8,
    /// The size of a decoded pixel in bytes.
    pub decoded_size: u8,
}

/// A function that processes a row of pixels.
///
/// The first argument is a byte slice of encoded pixels. The slice is
/// guaranteed te have a length that is a multiple of `size_of::<InPixel>()`.
///
/// The second argument is a byte slice of decoded pixels. The slice is
/// guaranteed te have a length that is a multiple of `size_of::<OutputPixel>()`.
///
/// Both slices are guaranteed to have the same number of pixels.
pub(crate) type ProcessPixelsFn = fn(encoded_decoded: PixelArgs);
// Another hack to work around that mutable references aren't allowed in const
// environments on MSRV.
pub(crate) struct PixelArgs<'a, 'b>(pub &'a [u8], pub &'b mut [u8]);

/// A helper function for implementing [`ProcessPixelsFn`]s.
#[inline]
pub(crate) fn process_pixels_helper<InPixel: cast::FromLeBytes, OutPixel: cast::IntoNeBytes>(
    encoded: &[u8],
    decoded: &mut [u8],
    f: impl Fn(InPixel) -> OutPixel,
) {
    // group bytes into chunks
    let encoded: &[InPixel::Bytes] = cast::from_bytes(encoded).expect("Invalid input buffer");
    let decoded: &mut [OutPixel::Bytes] =
        cast::from_bytes_mut(decoded).expect("Invalid output buffer");

    for (encoded, decoded) in encoded.iter().zip(decoded.iter_mut()) {
        let input: InPixel = cast::FromLeBytes::from_le_bytes(*encoded);
        *decoded = cast::IntoNeBytes::into_ne_bytes(f(input));
    }
}
#[inline]
pub(crate) fn process_pixels_helper_unroll<const UNROLL: usize, InPixel, OutPixel, F>(
    encoded: &[u8],
    decoded: &mut [u8],
    f: F,
) where
    InPixel: cast::FromLeBytes,
    OutPixel: cast::IntoNeBytes,
    [InPixel; UNROLL]: cast::FromLeBytes,
    [OutPixel; UNROLL]: cast::IntoNeBytes,
    F: Copy + Fn(InPixel) -> OutPixel,
{
    let pixels = encoded.len() / size_of::<InPixel>();
    let rolled_chunks = pixels / UNROLL;

    let encoded_chunks_bytes = rolled_chunks * size_of::<[InPixel; UNROLL]>();
    let decoded_chunks_bytes = rolled_chunks * size_of::<[OutPixel; UNROLL]>();

    // process unrolled chunks
    process_pixels_helper(
        &encoded[..encoded_chunks_bytes],
        &mut decoded[..decoded_chunks_bytes],
        move |input: [InPixel; UNROLL]| input.map(f),
    );

    // process the rest
    let encoded: &[InPixel::Bytes] =
        cast::from_bytes(&encoded[encoded_chunks_bytes..]).expect("Invalid input buffer");
    let decoded: &mut [OutPixel::Bytes] =
        cast::from_bytes_mut(&mut decoded[decoded_chunks_bytes..]).expect("Invalid output buffer");
    debug_assert!(encoded.len() == decoded.len());

    for (encoded, decoded) in encoded.iter().zip(decoded.iter_mut()) {
        let input: InPixel = cast::FromLeBytes::from_le_bytes(*encoded);
        *decoded = cast::IntoNeBytes::into_ne_bytes(f(input));
    }
}

/// Helper method for decoding UNCOMPRESSED formats.
pub(crate) fn for_each_pixel_untyped(
    r: &mut dyn Read,
    buf: &mut [u8],
    buf_channels: Channels,
    native_color: ColorFormat,
    pixel_size: PixelSize,
    process_pixels: ProcessPixelsFn,
) -> Result<(), DecodeError> {
    fn inner(
        r: &mut dyn Read,
        buf: &mut [u8],
        buf_color: ColorFormat,
        native_color: ColorFormat,
        size_of_in: usize,
        process_pixels: ProcessPixelsFn,
    ) -> Result<(), DecodeError> {
        let buf_bytes_per_pixel = buf_color.bytes_per_pixel() as usize;
        assert!(buf.len() % buf_bytes_per_pixel == 0);
        let pixels = buf.len() / buf_bytes_per_pixel;

        let mut read_buffer = UntypedPixelBuffer::new(pixels, size_of_in);
        let mut conversion_buffer = ChannelConversionBuffer::new(native_color, buf_color.channels);
        for buf in buf.chunks_mut(read_buffer.buffered_pixels() * buf_bytes_per_pixel) {
            let row = read_buffer.read(r)?;
            debug_assert!(row.len() % size_of_in == 0);
            debug_assert!(buf.len() % buf_bytes_per_pixel == 0);
            let pixels = row.len() / size_of_in;
            debug_assert_eq!(pixels, buf.len() / buf_bytes_per_pixel);

            conversion_buffer.process_pixels(row, buf, process_pixels);
        }
        Ok(())
    }

    let buf_color = ColorFormat::new(buf_channels, native_color.precision);
    debug_assert_eq!(native_color.bytes_per_pixel(), pixel_size.decoded_size);

    inner(
        r,
        buf,
        buf_color,
        native_color,
        pixel_size.encoded_size as usize,
        process_pixels,
    )
}

/// Helper method for decoding UNCOMPRESSED formats.
///
/// `process_pixels` has the same semantics as in `for_each_pixel_untyped`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn for_each_pixel_rect_untyped(
    r: &mut dyn ReadSeek,
    buf: &mut [u8],
    row_pitch: usize,
    size: Size,
    rect: Rect,
    buf_channels: Channels,
    native_color: ColorFormat,
    pixel_size: PixelSize,
    process_pixels: ProcessPixelsFn,
) -> Result<(), DecodeError> {
    #[allow(clippy::too_many_arguments)]
    fn inner(
        r: &mut dyn ReadSeek,
        buf: &mut [u8],
        row_pitch: usize,
        size: Size,
        rect: Rect,
        buf_color: ColorFormat,
        native_color: ColorFormat,
        size_of_in: usize,
        process_pixels: ProcessPixelsFn,
    ) -> Result<(), DecodeError> {
        // assert that no overflow will occur for byte positions in the encoded image/reader
        assert!(size
            .pixels()
            .checked_mul(size_of_in as u64)
            .map(|bytes| bytes <= i64::MAX as u64)
            .unwrap_or(false));

        let encoded_bytes_per_row = size.width as i64 * size_of_in as i64;
        let encoded_bytes_before_rect = rect.x as i64 * size_of_in as i64;
        let encoded_bytes_after_rect =
            (size.width - rect.x - rect.width) as i64 * size_of_in as i64;

        let buffer_bytes_per_pixel = buf_color.bytes_per_pixel() as usize;

        // jump to the first pixel
        seek_relative(
            r,
            encoded_bytes_per_row * rect.y as i64 + encoded_bytes_before_rect,
        )?;

        let pixels_per_line = rect.width as usize;
        let mut row: Box<[u8]> =
            vec![Default::default(); pixels_per_line * size_of_in].into_boxed_slice();
        let mut conversion_buffer = ChannelConversionBuffer::new(native_color, buf_color.channels);
        for y in 0..rect.height {
            if y > 0 {
                // jump to the first pixel in the next row
                // (this has already been done for the first row; see above)
                seek_relative(r, encoded_bytes_before_rect + encoded_bytes_after_rect)?;
            }

            // read next line
            r.read_exact(&mut row)?;

            let buf_start = y as usize * row_pitch;
            let buf_len = pixels_per_line * buffer_bytes_per_pixel;
            let buf = &mut buf[buf_start..(buf_start + buf_len)];
            debug_assert_eq!(row.len() / size_of_in, buf.len() / buffer_bytes_per_pixel);

            conversion_buffer.process_pixels(&row, buf, process_pixels);
        }

        // jump to the end of the surface to put the reader into a known position
        seek_relative(
            r,
            encoded_bytes_after_rect
                + (size.height - rect.y - rect.height) as i64 * encoded_bytes_per_row,
        )?;

        Ok(())
    }

    let buf_color = ColorFormat::new(buf_channels, native_color.precision);
    debug_assert_eq!(native_color.bytes_per_pixel(), pixel_size.decoded_size);

    inner(
        r,
        buf,
        row_pitch,
        size,
        rect,
        buf_color,
        native_color,
        pixel_size.encoded_size as usize,
        process_pixels,
    )
}

fn seek_relative(r: &mut dyn ReadSeek, offset: i64) -> std::io::Result<()> {
    if offset != 0 {
        r.seek(SeekFrom::Current(offset))?;
    }
    Ok(())
}

/// A function that processes a row of blocks.
///
/// Arguments:
///
/// `encoded_blocks` is a byte slice of blocks. The slice is
/// guaranteed te have a length that is a multiple of `BYTES_PER_BLOCK`.
///
/// `decoded` is a byte slice of decoded pixels.
///
/// `stride` is the number of bytes between the start of two consecutive rows
/// in `decoded`.
pub(crate) type ProcessBlocksFn =
    fn(encoded_blocks: &[u8], decoded: &mut [u8], stride: usize, range: PixelRange);
#[derive(Debug, Clone)]
pub(crate) struct PixelRange {
    /// The number of pixels in a row. This might *not* be a multiple of `BLOCK_SIZE_X`
    pub width: u32,
    /// The number of pixels in the first block that should be skipped.
    ///
    /// This is at most `BLOCK_SIZE_X - 1`.
    pub width_offset: u8,
    /// A non-empty range of the rows to decode. `rows.end` is at most `BLOCK_SIZE_Y`.
    pub rows: core::ops::Range<u8>,
}

/// A helper function for implementing [`ProcessBlocksFn`]s.
#[inline]
pub(crate) fn process_2x1_blocks_helper<
    const BYTES_PER_BLOCK: usize,
    OutPixel: cast::IntoNeBytes,
>(
    encoded_blocks: &[u8],
    decoded: &mut [u8],
    range: PixelRange,
    process_block: impl Fn([u8; BYTES_PER_BLOCK]) -> [OutPixel; 2],
) {
    // group bytes into chunks
    let mut encoded_blocks: &[[u8; BYTES_PER_BLOCK]] =
        cast::from_bytes(encoded_blocks).expect("Invalid block buffer");

    let mut width = range.width as usize;
    let mut decoded: &mut [OutPixel::Bytes] =
        cast::from_bytes_mut(&mut decoded[..width * size_of::<OutPixel>()])
            .expect("Invalid output buffer");
    debug_assert!(decoded.len() == width);

    let width_offset = range.width_offset;
    if width_offset == 1 {
        // skip the first pixel in the first block
        debug_assert!(width > 0);

        let [_, p1] = process_block(encoded_blocks[0]);
        decoded[0] = cast::IntoNeBytes::into_ne_bytes(p1);

        // adjust the width and blocks
        width -= 1;
        encoded_blocks = &encoded_blocks[1..];
        decoded = &mut decoded[1..];
    }

    let width_half = width / 2;

    // do full pairs first
    let decoded_pairs: &mut [[OutPixel::Bytes; 2]] =
        cast::as_array_chunks_mut(&mut decoded[..(width_half * 2)]).unwrap();
    for (encoded, decoded) in encoded_blocks.iter().zip(decoded_pairs.iter_mut()) {
        let [p0, p1] = process_block(*encoded);
        decoded[0] = cast::IntoNeBytes::into_ne_bytes(p0);
        decoded[1] = cast::IntoNeBytes::into_ne_bytes(p1);
    }

    // last lone pixel (if any)
    if width % 2 == 1 {
        let encoded = encoded_blocks.last().unwrap();
        let [p0, _] = process_block(*encoded);
        decoded[width - 1] = cast::IntoNeBytes::into_ne_bytes(p0);
    }
}

/// A helper function for implementing [`ProcessBlocksFn`]s.
#[inline]
pub(crate) fn process_8x1_blocks_helper<
    OutPixel: cast::IntoNeBytes + Copy,
    F: Fn(u8) -> [OutPixel; 8],
>(
    encoded_blocks: &[u8],
    decoded: &mut [u8],
    stride: usize,
    range: PixelRange,
    process_block: F,
) {
    general_process_blocks::<8, 1, 8, 1, OutPixel, _>(
        encoded_blocks,
        decoded,
        stride,
        range,
        |block| process_block(block[0]),
    );
}
/// A helper function for implementing [`ProcessBlocksFn`]s.
#[inline]
pub(crate) fn process_4x4_blocks_helper<
    const BYTES_PER_BLOCK: usize,
    OutPixel: cast::IntoNeBytes + cast::Castable + Copy,
>(
    mut encoded_blocks: &[u8],
    mut decoded: &mut [u8],
    stride: usize,
    mut range: PixelRange,
    process_block: impl Copy + Fn([u8; BYTES_PER_BLOCK]) -> [OutPixel; 16],
) {
    debug_assert!(range.rows.len() <= 4);
    debug_assert!(encoded_blocks.len() % BYTES_PER_BLOCK == 0);
    debug_assert_eq!(
        encoded_blocks.len() / BYTES_PER_BLOCK,
        div_ceil(range.width_offset as u32 + range.width, 4) as usize
    );
    if decoded.len()
        < stride * (range.rows.len() - 1) + range.width as usize * size_of::<OutPixel>()
    {
        println!("debugger");
    }
    debug_assert!(
        decoded.len()
            >= stride * (range.rows.len() - 1) + range.width as usize * size_of::<OutPixel>(),
        "decoded.len() = {}, stride = {}, range = {:?}",
        decoded.len(),
        stride,
        range
    );

    if range.width_offset != 0 {
        // handle offset separately
        let skip = handle_width_offset::<4, 4, 16, BYTES_PER_BLOCK, OutPixel, _>(
            &mut encoded_blocks,
            decoded,
            stride,
            &mut range,
            process_block,
        );
        decoded = &mut decoded[skip..];
    }

    // optimized code path for 4x4 blocks
    if range.rows.len() == 4 && stride % size_of::<OutPixel>() == 0 {
        if let Some(decoded) = cast::from_bytes_mut::<OutPixel>(decoded) {
            let encoded_blocks: &[[u8; BYTES_PER_BLOCK]] =
                cast::from_bytes(encoded_blocks).expect("Invalid block buffer");

            let stride = stride / size_of::<OutPixel>();
            let full_blocks = range.width as usize / 4;

            for (block_index, block) in encoded_blocks[..full_blocks].iter().enumerate() {
                let pixel_index = block_index * 4;

                let block = process_block(*block);

                for y in 0..4 {
                    let row_start = stride * y + pixel_index;
                    let row = &mut decoded[row_start..row_start + 4];
                    for x in 0..4 {
                        row[x] = block[y * 4 + x];
                    }
                }
            }

            if range.width % 4 != 0 {
                let block = encoded_blocks[full_blocks];
                let pixel_index = full_blocks * 4;
                let block_w = range.width as usize - pixel_index;

                let block = process_block(block);

                for y in 0..4 {
                    let row_start = stride * y + pixel_index;
                    let row = &mut decoded[row_start..row_start + block_w];
                    for x in 0..block_w {
                        row[x] = block[y * 4 + x];
                    }
                }
            }

            return;
        }
    }

    // General implementation. Slower.
    general_process_blocks::<4, 4, 16, BYTES_PER_BLOCK, OutPixel, _>(
        encoded_blocks,
        decoded,
        stride,
        range,
        process_block,
    );
}
/// Handles `range.width_offset` and returns how many bytes in `decoded` need to
/// be skipped after this method.
///
/// After this method returns, `range.width_offset` is guaranteed to be 0.
fn handle_width_offset<
    const BLOCK_SIZE_X: usize,
    const BLOCK_SIZE_Y: usize,
    const BLOCK_PIXELS: usize,
    const BYTES_PER_BLOCK: usize,
    OutPixel: cast::IntoNeBytes + Copy,
    F: Fn([u8; BYTES_PER_BLOCK]) -> [OutPixel; BLOCK_PIXELS],
>(
    encoded_blocks: &mut &[u8],
    decoded: &mut [u8],
    stride: usize,
    range: &mut PixelRange,
    process_block: F,
) -> usize {
    let offset = range.width_offset as u32;
    debug_assert!(offset < BLOCK_SIZE_X as u32);
    let pixel_w = u32::min(BLOCK_SIZE_X as u32 - offset, range.width);
    if pixel_w == 0 {
        return 0;
    }

    general_process_blocks::<BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_PIXELS, BYTES_PER_BLOCK, OutPixel, _>(
        &encoded_blocks[..BYTES_PER_BLOCK],
        decoded,
        stride,
        PixelRange {
            width: pixel_w,
            width_offset: range.width_offset,
            rows: range.rows.clone(),
        },
        process_block,
    );

    // adjust range
    range.width -= pixel_w;
    range.width_offset = 0;
    // skip first block that has now been processed
    *encoded_blocks = &encoded_blocks[BYTES_PER_BLOCK..];
    // skip first offset pixels
    pixel_w as usize * size_of::<OutPixel>()
}
/// A helper function for implementing [`ProcessBlocksFn`]s.
///
/// This is a *general* implementation. It will work with any block size, but
/// it's a lot slower than the specialized versions. Don't use this directly.
/// Instead, use it as the starting point for a specialized implementation.
pub(crate) fn general_process_blocks<
    const BLOCK_SIZE_X: usize,
    const BLOCK_SIZE_Y: usize,
    const BLOCK_PIXELS: usize,
    const BYTES_PER_BLOCK: usize,
    OutPixel: cast::IntoNeBytes + Copy,
    F: Fn([u8; BYTES_PER_BLOCK]) -> [OutPixel; BLOCK_PIXELS],
>(
    encoded_blocks: &[u8],
    decoded: &mut [u8],
    stride: usize,
    range: PixelRange,
    process_block: F,
) {
    debug_assert_eq!(BLOCK_SIZE_X * BLOCK_SIZE_Y, BLOCK_PIXELS);
    debug_assert!((range.width_offset as usize) < BLOCK_SIZE_X);

    // group bytes into chunks
    let encoded_blocks: &[[u8; BYTES_PER_BLOCK]] =
        cast::from_bytes(encoded_blocks).expect("Invalid block buffer");

    let mut pixel_x = 0;
    for (block_index, block) in encoded_blocks.iter().enumerate() {
        let pixel_offset_x = if block_index == 0 {
            range.width_offset as usize
        } else {
            0
        };
        let block_w = (BLOCK_SIZE_X - pixel_offset_x)
            .min(range.width as usize)
            .min(range.width as usize + range.width_offset as usize - block_index * BLOCK_SIZE_X);

        // This whole method is structured to call this function exactly once.
        // This is done to reduce code size.
        let block = process_block(*block);

        for y in range.rows.clone() {
            let row_start =
                (y - range.rows.start) as usize * stride + pixel_x * size_of::<OutPixel>();
            let row = &mut decoded[row_start..(row_start + block_w * size_of::<OutPixel>())];
            let row: &mut [OutPixel::Bytes] =
                cast::from_bytes_mut(row).expect("Invalid output buffer");
            debug_assert!(row.len() == block_w);

            for x in 0..block_w {
                row[x] = cast::IntoNeBytes::into_ne_bytes(
                    block[y as usize * BLOCK_SIZE_X + x + pixel_offset_x],
                );
            }
        }

        pixel_x += block_w;
    }
}

pub(crate) fn for_each_block_untyped<
    const BLOCK_SIZE_X: usize,
    const BLOCK_SIZE_Y: usize,
    const BYTES_PER_BLOCK: usize,
    OutPixel,
>(
    r: &mut dyn Read,
    buf: &mut [u8],
    size: Size,
    buf_channels: Channels,
    native_color: ColorFormat,
    process_pixels: ProcessBlocksFn,
) -> Result<(), DecodeError> {
    #[allow(clippy::too_many_arguments)]
    fn inner(
        r: &mut dyn Read,
        buf: &mut [u8],
        size: Size,
        block_size: (usize, usize),
        bytes_per_block: usize,
        buf_color: ColorFormat,
        native_color: ColorFormat,
        process_blocks: ProcessBlocksFn,
    ) -> Result<(), DecodeError> {
        // The basic idea here is to decode the image line by line. A line is a
        // sequence of encoded blocks that together describe BLOCK_SIZE_Y rows of
        // pixels in the final image.
        //
        // Since reading a bunch of small lines from disk is slow, we allocate one
        // large buffer to hold N lines at a time. The we process the lines in the
        // buffer and refill as needed.

        assert!(!size.is_empty());

        let (block_size_x, block_size_y) = block_size;
        let width_blocks = div_ceil(size.width, block_size_x as u32) as usize;
        let height_blocks = div_ceil(size.height, block_size_y as u32) as usize;

        let mut line_buffer = UntypedLineBuffer::new(width_blocks * bytes_per_block, height_blocks);
        let mut conversion_buffer = ChannelConversionBuffer::new(native_color, buf_color.channels);

        let mut block_y = 0;
        while let Some(block_line) = line_buffer.next_line(r)? {
            // how many rows of pixels we'll decode
            // this is usually BLOCK_SIZE_Y, but might be less for the last block
            let pixel_rows = block_size_y.min(size.height as usize - block_y * block_size_y);
            let pixel_row_bytes = size.width as usize * buf_color.bytes_per_pixel() as usize;
            debug_assert!(buf.len() % pixel_row_bytes == 0);

            let start_pixel_y = block_y * block_size_y;
            let buf = &mut buf
                [start_pixel_y * pixel_row_bytes..(start_pixel_y + pixel_rows) * pixel_row_bytes];

            let range = PixelRange {
                width: size.width,
                width_offset: 0,
                rows: 0..pixel_rows as u8,
            };

            conversion_buffer.process_blocks(
                bytes_per_block,
                block_size_x as u32,
                block_line,
                buf,
                pixel_row_bytes,
                range,
                process_blocks,
            );

            block_y += 1;
        }
        Ok(())
    }

    let buf_color = ColorFormat::new(buf_channels, native_color.precision);
    debug_assert_eq!(native_color.bytes_per_pixel(), size_of::<OutPixel>() as u8);

    inner(
        r,
        buf,
        size,
        (BLOCK_SIZE_X, BLOCK_SIZE_Y),
        BYTES_PER_BLOCK,
        buf_color,
        native_color,
        process_pixels,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn for_each_block_rect_untyped<
    const BLOCK_SIZE_X: usize,
    const BLOCK_SIZE_Y: usize,
    const BYTES_PER_BLOCK: usize,
>(
    r: &mut dyn ReadSeek,
    buf: &mut [u8],
    row_pitch: usize,
    size: Size,
    rect: Rect,
    buf_channels: Channels,
    native_color: ColorFormat,
    process_pixels: ProcessBlocksFn,
) -> Result<(), DecodeError> {
    #[allow(clippy::too_many_arguments)]
    fn inner(
        r: &mut dyn ReadSeek,
        buf: &mut [u8],
        row_pitch: usize,
        size: Size,
        rect: Rect,
        block_size: (usize, usize),
        bytes_per_block: usize,
        buf_color: ColorFormat,
        native_color: ColorFormat,
        process_blocks: ProcessBlocksFn,
    ) -> Result<(), DecodeError> {
        // To make this algorithm easier to implement, we'll always read full
        // lines of blocks.

        let (block_size_x, block_size_y) = block_size;
        let blocks_per_line = div_ceil(size.width, block_size_x as u32) as usize;

        // blocks before the block lines we want to read.
        let skip_block_lines_before = rect.y as usize / block_size_y;
        // blocks of the lines we want to read
        let block_lines_to_read =
            div_ceil(rect.height + rect.y, block_size_y as u32) as usize - skip_block_lines_before;
        // blocks after the block lines we want to read
        let skip_block_lines_after = div_ceil(size.height, block_size_y as u32) as usize
            - skip_block_lines_before
            - block_lines_to_read;

        // jump to the first line of blocks
        seek_relative(
            r,
            (blocks_per_line * skip_block_lines_before * bytes_per_block) as i64,
        )?;

        let mut line_buffer =
            UntypedLineBuffer::new(blocks_per_line * bytes_per_block, block_lines_to_read);
        let mut conversion_buffer = ChannelConversionBuffer::new(native_color, buf_color.channels);

        // the range of blocks within a block line
        let block_range_start = rect.x as usize / block_size_x;
        let block_range_end = div_ceil(rect.x + rect.width, block_size_x as u32) as usize;
        let block_range =
            (block_range_start * bytes_per_block)..(block_range_end * bytes_per_block);

        // re-calculated parts of the pixel range
        let width = rect.width;
        let width_offset = (rect.x % block_size_x as u32) as u8;

        let mut block_line_y = skip_block_lines_before;
        let mut pixel_row = 0;
        while let Some(block_line) = line_buffer.next_line(r)? {
            // ignore blocks not part of the rect
            let block_line = &block_line[block_range.clone()];

            let rel_row_start = rect.y as isize - (block_line_y * block_size_y) as isize;
            let rel_row_end =
                (rect.y + rect.height) as isize - (block_line_y * block_size_y) as isize;
            debug_assert!(rel_row_start < block_size_y as isize);
            debug_assert!(rel_row_end > 0);

            let row_start = rel_row_start.clamp(0, block_size_y as isize) as u8;
            let row_end = rel_row_end.clamp(0, block_size_y as isize) as u8;
            let rows = row_start..row_end;

            let range = PixelRange {
                width,
                width_offset,
                rows: rows.clone(),
            };

            conversion_buffer.process_blocks(
                bytes_per_block,
                block_size_x as u32,
                block_line,
                &mut buf[pixel_row * row_pitch..],
                row_pitch,
                range,
                process_blocks,
            );

            block_line_y += 1;
            pixel_row += rows.len();
        }

        // jump to the end of the surface to put the reader into a known position
        seek_relative(
            r,
            (blocks_per_line * skip_block_lines_after * bytes_per_block) as i64,
        )?;

        Ok(())
    }

    let buf_color = ColorFormat::new(buf_channels, native_color.precision);

    inner(
        r,
        buf,
        row_pitch,
        size,
        rect,
        (BLOCK_SIZE_X, BLOCK_SIZE_Y),
        BYTES_PER_BLOCK,
        buf_color,
        native_color,
        process_pixels,
    )
}

struct ChannelConversionBuffer {
    buffer: [u32; Self::BUFFER_BYTES / 4],
    native_color: ColorFormat,
    target: Channels,
}
impl ChannelConversionBuffer {
    const BUFFER_BYTES: usize = 1024;
    fn new(native_color: ColorFormat, target: Channels) -> Self {
        Self {
            buffer: [0_u32; Self::BUFFER_BYTES / 4],
            native_color,
            target,
        }
    }

    fn process_pixels(&mut self, encoded: &[u8], out: &mut [u8], f: ProcessPixelsFn) {
        // fast path: no conversion needed
        if self.native_color.channels == self.target {
            f(PixelArgs(encoded, out));
            return;
        }

        let out_bytes_per_pixel =
            ColorFormat::new(self.target, self.native_color.precision).bytes_per_pixel() as usize;
        let pixels = out.len() / out_bytes_per_pixel;
        let encoded_bytes_per_pixel = encoded.len() / pixels;
        debug_assert!(out_bytes_per_pixel % self.target.count() as usize == 0);
        let buffer_bytes_per_pixel = self.native_color.bytes_per_pixel() as usize;
        let buffer_pixels = Self::BUFFER_BYTES / buffer_bytes_per_pixel;

        let buffer = cast::as_bytes_mut(&mut self.buffer);

        for chunk_start in (0..pixels).step_by(buffer_pixels) {
            let chunk_end = (chunk_start + buffer_pixels).min(pixels);
            let chunk_size = chunk_end - chunk_start;

            let encoded_chunk = &encoded
                [chunk_start * encoded_bytes_per_pixel..chunk_end * encoded_bytes_per_pixel];
            let out_chunk =
                &mut out[chunk_start * out_bytes_per_pixel..chunk_end * out_bytes_per_pixel];
            let buffer_chunk = &mut buffer[..chunk_size * buffer_bytes_per_pixel];

            // decode into the temporary buffer
            f(PixelArgs(encoded_chunk, buffer_chunk));

            // convert the channels into the output buffer
            convert_channels_untyped_for(self.native_color, self.target, buffer_chunk, out_chunk);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_blocks(
        &mut self,
        block_bytes: usize,
        block_width: u32,
        mut encoded_blocks: &[u8],
        mut out: &mut [u8],
        stride: usize,
        mut range: PixelRange,
        f: ProcessBlocksFn,
    ) {
        // fast path: no conversion needed
        if self.native_color.channels == self.target {
            f(encoded_blocks, out, stride, range);
            return;
        }

        fn nice_multiple(width: u32) -> u32 {
            // round down to a multiple of 4
            width & !3
        }

        let size = Size::new(range.width, range.rows.len() as u32);
        debug_assert!(size.height > 0);
        let buffer_bytes_per_pixel = self.native_color.bytes_per_pixel() as usize;
        let buffer_size = Size::new(
            nice_multiple(
                (Self::BUFFER_BYTES / (buffer_bytes_per_pixel * size.height as usize)) as u32,
            ),
            size.height,
        );
        debug_assert!(buffer_size.width >= block_width);
        let out_bytes_per_pixel =
            ColorFormat::new(self.target, self.native_color.precision).bytes_per_pixel() as usize;

        let buffer = cast::as_bytes_mut(&mut self.buffer);

        // If the entire decoded line fits into the buffer, just do it in one go.
        if size.width <= buffer_size.width {
            let buffer_stride = size.width as usize * buffer_bytes_per_pixel;
            let buffer = &mut buffer[..buffer_stride * size.height as usize];

            // decode into the temporary buffer
            f(encoded_blocks, buffer, buffer_stride, range);

            // convert the channels into the output buffer
            for y in 0..size.height as usize {
                let buffer_row = &buffer[y * buffer_stride..(y + 1) * buffer_stride];
                let out_row =
                    &mut out[y * stride..y * stride + size.width as usize * out_bytes_per_pixel];
                convert_channels_untyped_for(self.native_color, self.target, buffer_row, out_row);
            }
            return;
        }

        // To simplify the following code, start by handling the width offset, so that
        // the general case can assume `range.width_offset == 0`.
        if range.width_offset != 0 {
            let offset_width = block_width - range.width_offset as u32;

            let buffer_stride = offset_width as usize * buffer_bytes_per_pixel;
            let buffer = &mut buffer[..buffer_stride * size.height as usize];

            // decode into the temporary buffer
            f(
                &encoded_blocks[..block_bytes],
                buffer,
                buffer_stride,
                PixelRange {
                    width: offset_width,
                    width_offset: range.width_offset,
                    rows: range.rows.clone(),
                },
            );

            // convert the channels into the output buffer
            for y in 0..size.height as usize {
                let buffer_row = &buffer[y * buffer_stride..(y + 1) * buffer_stride];
                let out_row =
                    &mut out[y * stride..y * stride + offset_width as usize * out_bytes_per_pixel];
                convert_channels_untyped_for(self.native_color, self.target, buffer_row, out_row);
            }

            // adjust inputs
            range.width_offset = 0;
            range.width -= offset_width;
            encoded_blocks = &encoded_blocks[block_bytes..];
            out = &mut out[offset_width as usize * out_bytes_per_pixel..];
        }

        debug_assert!(range.width_offset == 0);
        for chunk_start in
            (0..size.width).step_by((buffer_size.width / block_width * block_width) as usize)
        {
            let chunk_end = (chunk_start + buffer_size.width).min(size.width);
            let chunk_size = chunk_end - chunk_start;

            let block_offset = (chunk_start / block_width) as usize;
            let block_count = div_ceil(chunk_size, block_width) as usize;

            let encoded_chunk = &encoded_blocks
                [block_offset * block_bytes..(block_offset + block_count) * block_bytes];
            let out_chunk = &mut out[chunk_start as usize * out_bytes_per_pixel..];

            let buffer_stride = chunk_size as usize * buffer_bytes_per_pixel;
            let buffer_chunk = &mut buffer[..buffer_stride * size.height as usize];

            // decode into the temporary buffer
            f(
                encoded_chunk,
                buffer_chunk,
                buffer_stride,
                PixelRange {
                    width: chunk_size,
                    width_offset: 0,
                    rows: range.rows.clone(),
                },
            );

            // convert the channels into the output buffer
            for y in 0..size.height as usize {
                let buffer_row = &buffer_chunk[y * buffer_stride..(y + 1) * buffer_stride];
                let out_row = &mut out_chunk
                    [y * stride..y * stride + chunk_size as usize * out_bytes_per_pixel];
                convert_channels_untyped_for(self.native_color, self.target, buffer_row, out_row);
            }
        }
    }
}

/// A buffer holding raw encoded pixels straight from the reader.
struct UntypedPixelBuffer {
    buf: Vec<u8>,
    bytes_per_pixel: usize,
    bytes_left: usize,
}
impl UntypedPixelBuffer {
    /// The target buffer size is in bytes. Currently 64 KiB.
    const TARGET: usize = 64 * 1024;

    fn new(pixels: usize, bytes_per_pixel: usize) -> Self {
        let buf_bytes = (Self::TARGET / bytes_per_pixel).min(pixels) * bytes_per_pixel;
        let buf = vec![0; buf_bytes];
        Self {
            buf,
            bytes_per_pixel,
            bytes_left: pixels * bytes_per_pixel,
        }
    }

    fn buffered_pixels(&self) -> usize {
        self.buf.len() / self.bytes_per_pixel
    }

    fn read<R: Read + ?Sized>(&mut self, r: &mut R) -> Result<&[u8], DecodeError> {
        let bytes_to_read = self.buf.len().min(self.bytes_left);
        assert!(bytes_to_read > 0);
        self.bytes_left -= bytes_to_read;

        let buf = &mut self.buf[..bytes_to_read];
        r.read_exact(buf)?;
        Ok(buf)
    }
}

/// A buffer holding raw encoded lines of pixels straight from the reader.
struct UntypedLineBuffer {
    buf: Vec<u8>,
    bytes_per_line: usize,
    /// How many lines are still left to read from disk
    lines_on_disk: usize,
    /// The index at which the current line starts in the buffer.
    ///
    /// If `>= buffer.len()`, the buffer is empty and needs to be refilled.
    current_line_start: usize,
}
impl UntypedLineBuffer {
    fn new(bytes_per_line: usize, height: usize) -> Self {
        const TARGET_BUFFER_SIZE: usize = 64 * 1024; // 64 KB

        let lines_in_buffer = (TARGET_BUFFER_SIZE / bytes_per_line).clamp(1, height);
        let buf_len = lines_in_buffer * bytes_per_line;
        // TODO: protect against allocating very large buffers (> 1 MB)
        let buf = vec![0_u8; buf_len];

        Self {
            buf,
            bytes_per_line,
            lines_on_disk: height,
            current_line_start: buf_len,
        }
    }

    // CURSE YOU, lack of trait up-casting
    fn next_line<R: Read + ?Sized>(&mut self, r: &mut R) -> Result<Option<&[u8]>, DecodeError> {
        if self.current_line_start >= self.buf.len() {
            if self.lines_on_disk == 0 {
                // all lines have been read
                return Ok(None);
            }

            // refill the buffer
            let lines_to_read = (self.buf.len() / self.bytes_per_line).min(self.lines_on_disk);
            self.lines_on_disk -= lines_to_read;
            self.buf.truncate(lines_to_read * self.bytes_per_line);
            r.read_exact(&mut self.buf)?;
            self.current_line_start = 0;
        }

        // get a line from the buffer
        let line_end = self.current_line_start + self.bytes_per_line;
        let line = &self.buf[self.current_line_start..line_end];
        self.current_line_start = line_end;
        Ok(Some(line))
    }
}
