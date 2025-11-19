//! An internal module with helper methods for reading bytes from a reader, and
//! writing decoded pixels to the output buffer.

use std::io::Read;
use std::mem::size_of;

use crate::{cast, util::div_ceil, DecodingError, Offset, Size};
use crate::{convert_channels_for, util, Channels, ColorFormat, ImageViewMut};

use super::{DecodeContext, ReadSeek};

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
/// guaranteed to have a length that is a multiple of `size_of::<InPixel>()`.
///
/// The second argument is a byte slice of decoded pixels. The slice is
/// guaranteed to have a length that is a multiple of `size_of::<OutputPixel>()`.
///
/// Both slices are guaranteed to have the same number of pixels.
pub(crate) type ProcessPixelsFn = fn(&[u8], &mut [u8]);

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
    image: &mut ImageViewMut,
    mut context: DecodeContext,
    native_color: ColorFormat,
    pixel_size: PixelSize,
    process_pixels: ProcessPixelsFn,
) -> Result<(), DecodingError> {
    debug_assert_eq!(image.color().precision, native_color.precision);
    debug_assert_eq!(native_color.bytes_per_pixel(), pixel_size.decoded_size);

    let size_of_in = pixel_size.encoded_size as usize;

    let mut line_buffer = UntypedLineBuffer::new(
        image.width() as usize * size_of_in,
        image.height(),
        &mut context,
    )?;
    let mut conversion_buffer = ChannelConversionBuffer::new(native_color, image.color().channels);
    for buf in image.rows_mut() {
        let line = line_buffer
            .next_line(r)?
            .expect("height of image and line buffer must match");
        debug_assert!(line.len() % size_of_in == 0);

        conversion_buffer.process_pixels(line, buf, process_pixels);
    }
    Ok(())
}

/// Helper method for decoding UNCOMPRESSED formats.
///
/// `process_pixels` has the same semantics as in `for_each_pixel_untyped`.
pub(crate) fn for_each_pixel_rect_untyped(
    r: &mut dyn ReadSeek,
    image: &mut ImageViewMut,
    offset: Offset,
    mut context: DecodeContext,
    native_color: ColorFormat,
    pixel_size: PixelSize,
    process_pixels: ProcessPixelsFn,
) -> Result<(), DecodingError> {
    debug_assert_eq!(image.color.precision, native_color.precision);
    debug_assert_eq!(native_color.bytes_per_pixel(), pixel_size.decoded_size);

    let size_of_in = pixel_size.encoded_size as usize;

    let surface_size = context.surface_size;

    // assert that no overflow will occur for byte positions in the encoded image/reader
    assert!(surface_size
        .pixels()
        .checked_mul(size_of_in as u64)
        .map(|bytes| bytes <= i64::MAX as u64)
        .unwrap_or(false));

    let encoded_bytes_per_row = surface_size.width as u64 * size_of_in as u64;
    let encoded_bytes_before_rect = offset.x as u64 * size_of_in as u64;
    let encoded_bytes_after_rect =
        (surface_size.width - offset.x - image.width()) as u64 * size_of_in as u64;

    let image_bytes_per_pixel = image.color().bytes_per_pixel() as usize;

    // jump to the first pixel
    util::io_skip_exact(
        r,
        encoded_bytes_per_row * offset.y as u64 + encoded_bytes_before_rect,
    )?;

    let mut row: Box<[u8]> = context.alloc(image.width() as usize * size_of_in)?;
    let mut conversion_buffer = ChannelConversionBuffer::new(native_color, image.color().channels);
    for y in 0..image.height() {
        if y > 0 {
            // jump to the first pixel in the next row
            // (this has already been done for the first row; see above)
            util::io_skip_exact(r, encoded_bytes_before_rect + encoded_bytes_after_rect)?;
        }

        // read next line
        r.read_exact(&mut row)?;

        let buf = image.get_row(y as usize);
        debug_assert_eq!(row.len() / size_of_in, buf.len() / image_bytes_per_pixel);

        conversion_buffer.process_pixels(&row, buf, process_pixels);
    }

    // jump to the end of the surface to put the reader into a known position
    util::io_skip_exact(
        r,
        encoded_bytes_after_rect
            + (surface_size.height - offset.y - image.height()) as u64 * encoded_bytes_per_row,
    )?;

    Ok(())
}

/// A function that processes a row of blocks.
///
/// Arguments:
///
/// `encoded_blocks` is a byte slice of blocks. The slice is
/// guaranteed to have a length that is a multiple of `BYTES_PER_BLOCK`.
///
/// `decoded` is a byte slice of decoded pixels.
///
/// `row_pitch` is the number of bytes between the start of two consecutive rows
/// in `decoded`.
pub(crate) type ProcessBlocksFn =
    fn(encoded_blocks: &[u8], decoded: &mut [u8], row_pitch: usize, range: PixelRange);
#[derive(Debug, Clone)]
pub(crate) struct PixelRange {
    /// The number of pixels in a row. This might *not* be a multiple of `BLOCK_SIZE_X`
    pub width: u32,
    /// The number of pixels in the first block that should be skipped.
    ///
    /// This is at most `BLOCK_SIZE_X - 1`.
    pub width_offset: u8,
    /// A non-empty range of the rows to decode. `rows.end` is at most `BLOCK_SIZE_Y`.
    pub rows: RowRange,
}
/// A non-empty range of rows of pixels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowRange {
    /// The start of the row range.
    pub start: u8,
    /// The end of the row range (exclusive).
    pub end: u8,
}
impl RowRange {
    pub fn new(start: u8, end: u8) -> Self {
        debug_assert!(start < end);
        Self { start, end }
    }
    /// Returns the number of rows in this range.
    ///
    /// This guaranteed to be at least 1.
    pub fn len(self) -> u8 {
        self.end - self.start
    }
    pub fn iter(self) -> core::ops::Range<u8> {
        self.start..self.end
    }
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
    debug_assert_eq!(decoded.len(), width);

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

    debug_assert_eq!(encoded_blocks.len(), util::div_ceil(width, 2));
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
        let encoded = encoded_blocks.last().expect("invalid block buffer");
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
    general_process_blocks::<8, 1, 8, 1, OutPixel>(
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
    debug_assert!(
        decoded.len()
            >= stride * (range.rows.len() as usize - 1)
                + range.width as usize * size_of::<OutPixel>(),
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
    general_process_blocks::<4, 4, 16, BYTES_PER_BLOCK, OutPixel>(
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
    const BLOCK_SIZE_X: u8,
    const BLOCK_SIZE_Y: u8,
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
    let offset = range.width_offset;
    debug_assert!(offset < BLOCK_SIZE_X);
    let pixel_w = u32::min((BLOCK_SIZE_X - offset) as u32, range.width);
    if pixel_w == 0 {
        return 0;
    }

    general_process_blocks::<BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_PIXELS, BYTES_PER_BLOCK, OutPixel>(
        &encoded_blocks[..BYTES_PER_BLOCK],
        decoded,
        stride,
        PixelRange {
            width: pixel_w,
            width_offset: range.width_offset,
            rows: range.rows,
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
    const BLOCK_SIZE_X: u8,
    const BLOCK_SIZE_Y: u8,
    const BLOCK_PIXELS: usize,
    const BYTES_PER_BLOCK: usize,
    OutPixel: cast::IntoNeBytes + Copy,
>(
    encoded_blocks: &[u8],
    decoded: &mut [u8],
    stride: usize,
    range: PixelRange,
    process_block: impl Fn([u8; BYTES_PER_BLOCK]) -> [OutPixel; BLOCK_PIXELS],
) {
    debug_assert_eq!(BLOCK_SIZE_X as usize * BLOCK_SIZE_Y as usize, BLOCK_PIXELS);
    debug_assert!(range.width_offset < BLOCK_SIZE_X);

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
        let block_w = (BLOCK_SIZE_X as usize - pixel_offset_x)
            .min(range.width as usize)
            .min(
                range.width as usize + range.width_offset as usize
                    - block_index * BLOCK_SIZE_X as usize,
            );

        // This whole method is structured to call this function exactly once.
        // This is done to reduce code size.
        let block = process_block(*block);

        for y in range.rows.iter() {
            let row_start =
                (y - range.rows.start) as usize * stride + pixel_x * size_of::<OutPixel>();
            let row = &mut decoded[row_start..(row_start + block_w * size_of::<OutPixel>())];
            let row: &mut [OutPixel::Bytes] =
                cast::from_bytes_mut(row).expect("Invalid output buffer");
            debug_assert!(row.len() == block_w);

            for x in 0..block_w {
                row[x] = cast::IntoNeBytes::into_ne_bytes(
                    block[y as usize * BLOCK_SIZE_X as usize + x + pixel_offset_x],
                );
            }
        }

        pixel_x += block_w;
    }
}

pub(crate) fn for_each_block_untyped<
    const BLOCK_SIZE_X: u8,
    const BLOCK_SIZE_Y: u8,
    const BYTES_PER_BLOCK: usize,
    OutPixel,
>(
    r: &mut dyn Read,
    image: &mut ImageViewMut,
    context: DecodeContext,
    native_color: ColorFormat,
    process_pixels: ProcessBlocksFn,
) -> Result<(), DecodingError> {
    fn inner(
        r: &mut dyn Read,
        image: &mut ImageViewMut,
        mut context: DecodeContext,
        block_size: (u8, u8),
        bytes_per_block: usize,
        native_color: ColorFormat,
        process_blocks: ProcessBlocksFn,
    ) -> Result<(), DecodingError> {
        let size = context.surface_size;

        // The basic idea here is to decode the image line by line. A line is a
        // sequence of encoded blocks that together describe BLOCK_SIZE_Y rows of
        // pixels in the final image.
        //
        // Since reading a bunch of small lines from disk is slow, we allocate one
        // large buffer to hold N lines at a time. The we process the lines in the
        // buffer and refill as needed.

        assert!(!size.is_empty());

        let block_size_x = block_size.0 as u32;
        let block_size_y = block_size.1 as u32;
        let width_blocks = div_ceil(size.width, block_size_x);
        let height_blocks = div_ceil(size.height, block_size_y);

        let mut line_buffer = UntypedLineBuffer::new(
            width_blocks as usize * bytes_per_block,
            height_blocks,
            &mut context,
        )?;
        let mut conversion_buffer =
            ChannelConversionBuffer::new(native_color, image.color().channels);

        let row_pitch = image.row_pitch();

        let mut block_y = 0;
        while let Some(block_line) = line_buffer.next_line(r)? {
            // how many rows of pixels we'll decode
            // this is usually BLOCK_SIZE_Y, but might be less for the last block
            let pixel_rows = block_size_y.min(size.height - block_y * block_size_y);

            let buf = image.get_row_range((block_y * block_size_y) as usize, pixel_rows as usize);

            let range = PixelRange {
                width: size.width,
                width_offset: 0,
                rows: RowRange::new(0, pixel_rows as u8),
            };

            conversion_buffer.process_blocks(
                bytes_per_block,
                block_size_x,
                block_line,
                buf,
                row_pitch,
                range,
                process_blocks,
            );

            block_y += 1;
        }
        Ok(())
    }

    debug_assert_eq!(image.color().precision, native_color.precision);
    debug_assert_eq!(
        native_color.bytes_per_pixel() as usize,
        size_of::<OutPixel>()
    );

    inner(
        r,
        image,
        context,
        (BLOCK_SIZE_X, BLOCK_SIZE_Y),
        BYTES_PER_BLOCK,
        native_color,
        process_pixels,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn for_each_block_rect_untyped<
    const BLOCK_SIZE_X: u8,
    const BLOCK_SIZE_Y: u8,
    const BYTES_PER_BLOCK: usize,
>(
    r: &mut dyn ReadSeek,
    image: &mut ImageViewMut,
    offset: Offset,
    context: DecodeContext,
    native_color: ColorFormat,
    process_pixels: ProcessBlocksFn,
) -> Result<(), DecodingError> {
    #[allow(clippy::too_many_arguments)]
    fn inner(
        r: &mut dyn ReadSeek,
        image: &mut ImageViewMut,
        offset: Offset,
        mut context: DecodeContext,
        block_size: (u8, u8),
        bytes_per_block: usize,
        native_color: ColorFormat,
        process_blocks: ProcessBlocksFn,
    ) -> Result<(), DecodingError> {
        let surface_size = context.surface_size;
        let image_width = image.width();
        let image_height = image.height();

        // To make this algorithm easier to implement, we'll always read full
        // lines of blocks.

        let block_size_x = block_size.0 as u32;
        let block_size_y = block_size.1 as u32;
        let blocks_per_line = div_ceil(surface_size.width, block_size_x);

        // blocks before the block lines we want to read.
        let skip_block_lines_before = offset.y / block_size_y;
        // blocks of the lines we want to read
        let block_lines_to_read =
            div_ceil(image_height + offset.y, block_size_y) - skip_block_lines_before;
        // blocks after the block lines we want to read
        let skip_block_lines_after = div_ceil(surface_size.height, block_size_y)
            - skip_block_lines_before
            - block_lines_to_read;

        // jump to the first line of blocks
        util::io_skip_exact(
            r,
            blocks_per_line as u64 * skip_block_lines_before as u64 * bytes_per_block as u64,
        )?;

        let mut line_buffer = UntypedLineBuffer::new(
            blocks_per_line as usize * bytes_per_block,
            block_lines_to_read,
            &mut context,
        )?;
        let mut conversion_buffer =
            ChannelConversionBuffer::new(native_color, image.color.channels);

        // the range of blocks within a block line
        let block_range_start = offset.x / block_size_x;
        let block_range_end = div_ceil(offset.x + image_width, block_size_x);
        let block_range = (block_range_start as usize * bytes_per_block)
            ..(block_range_end as usize * bytes_per_block);

        // re-calculated parts of the pixel range
        let width_offset = (offset.x % block_size_x) as u8;

        let mut block_line_y = skip_block_lines_before;
        let mut pixel_row = 0;
        while let Some(block_line) = line_buffer.next_line(r)? {
            // ignore blocks not part of the rect
            let block_line = &block_line[block_range.clone()];

            let rel_row_start = offset.y.saturating_sub(block_line_y * block_size_y);
            let rel_row_end = offset.y + image_height - block_line_y * block_size_y;
            debug_assert!(rel_row_start < block_size_y);
            debug_assert!(rel_row_end > 0);

            let row_start = rel_row_start as u8;
            let row_end = rel_row_end.min(block_size_y) as u8;
            let rows = RowRange::new(row_start, row_end);

            let range = PixelRange {
                width: image_width,
                width_offset,
                rows,
            };

            let row_pitch = image.row_pitch();
            let out = &mut image.data()[pixel_row * row_pitch..];

            conversion_buffer.process_blocks(
                bytes_per_block,
                block_size_x,
                block_line,
                out,
                row_pitch,
                range,
                process_blocks,
            );

            block_line_y += 1;
            pixel_row += rows.len() as usize;
        }

        // jump to the end of the surface to put the reader into a known position
        util::io_skip_exact(
            r,
            blocks_per_line as u64 * skip_block_lines_after as u64 * bytes_per_block as u64,
        )?;

        Ok(())
    }

    debug_assert_eq!(image.color.precision, native_color.precision);

    inner(
        r,
        image,
        offset,
        context,
        (BLOCK_SIZE_X, BLOCK_SIZE_Y),
        BYTES_PER_BLOCK,
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
    const BUFFER_BYTES: usize = 3072;
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
            f(encoded, out);
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
            f(encoded_chunk, buffer_chunk);

            // convert the channels into the output buffer
            convert_channels_for(self.native_color, self.target, buffer_chunk, out_chunk);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_blocks(
        &mut self,
        block_bytes: usize,
        block_width: u32,
        mut encoded_blocks: &[u8],
        mut out: &mut [u8],
        row_pitch: usize,
        mut range: PixelRange,
        f: ProcessBlocksFn,
    ) {
        // fast path: no conversion needed
        if self.native_color.channels == self.target {
            f(encoded_blocks, out, row_pitch, range);
            return;
        }

        let height = range.rows.len() as usize;
        debug_assert!(height > 0);
        let buffer_bytes_per_pixel = self.native_color.bytes_per_pixel() as usize;
        let buffer_size = Size::new(
            (Self::BUFFER_BYTES / (buffer_bytes_per_pixel * height)) as u32,
            height as u32,
        );
        debug_assert!(buffer_size.width >= block_width);
        let out_bytes_per_pixel =
            ColorFormat::new(self.target, self.native_color.precision).bytes_per_pixel() as usize;

        let buffer = cast::as_bytes_mut(&mut self.buffer);

        // To simplify the following code, start by handling the width offset, so that
        // the general case can assume `range.width_offset == 0`.
        if range.width_offset != 0 {
            let offset_width = (block_width - range.width_offset as u32).min(range.width);

            let buffer_stride = offset_width as usize * buffer_bytes_per_pixel;
            let buffer = &mut buffer[..buffer_stride * height];

            // decode into the temporary buffer
            f(
                &encoded_blocks[..block_bytes],
                buffer,
                buffer_stride,
                PixelRange {
                    width: offset_width,
                    width_offset: range.width_offset,
                    rows: range.rows,
                },
            );

            // convert the channels into the output buffer
            for y in 0..height {
                let buffer_row = &buffer[y * buffer_stride..(y + 1) * buffer_stride];
                let out_row = &mut out
                    [y * row_pitch..y * row_pitch + offset_width as usize * out_bytes_per_pixel];
                convert_channels_for(self.native_color, self.target, buffer_row, out_row);
            }

            // adjust inputs
            range.width_offset = 0;
            range.width -= offset_width;
            encoded_blocks = &encoded_blocks[block_bytes..];
            out = &mut out[offset_width as usize * out_bytes_per_pixel..];
        }

        debug_assert!(range.width_offset == 0);
        let preferred_chunk_size = util::round_down_to_multiple(buffer_size.width, block_width);
        for chunk_start in (0..range.width).step_by(preferred_chunk_size as usize) {
            let chunk_end = (chunk_start + preferred_chunk_size).min(range.width);
            let chunk_size = chunk_end - chunk_start;

            let block_offset = (chunk_start / block_width) as usize;
            let block_count = div_ceil(chunk_size, block_width) as usize;

            let encoded_chunk = &encoded_blocks
                [block_offset * block_bytes..(block_offset + block_count) * block_bytes];
            let out_chunk = &mut out[chunk_start as usize * out_bytes_per_pixel..];

            let buffer_stride = chunk_size as usize * buffer_bytes_per_pixel;
            let buffer_chunk = &mut buffer[..buffer_stride * height];

            // decode into the temporary buffer
            f(
                encoded_chunk,
                buffer_chunk,
                buffer_stride,
                PixelRange {
                    width: chunk_size,
                    width_offset: 0,
                    rows: range.rows,
                },
            );

            // convert the channels into the output buffer
            for y in 0..height {
                let buffer_row = &buffer_chunk[y * buffer_stride..(y + 1) * buffer_stride];
                let out_row = &mut out_chunk
                    [y * row_pitch..y * row_pitch + chunk_size as usize * out_bytes_per_pixel];
                convert_channels_for(self.native_color, self.target, buffer_row, out_row);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_bi_planar(
        &mut self,
        info: BiPlaneInfo,
        mut plane1: &[u8],
        mut plane2: &[u8],
        mut out: &mut [u8],
        mut range: PlaneRange,
        f: ProcessBiPlanarFn,
    ) {
        // fast path: no conversion needed
        if self.native_color.channels == self.target {
            f(plane1, plane2, out, range);
            return;
        }

        let out_bytes_per_pixel =
            ColorFormat::new(self.target, self.native_color.precision).bytes_per_pixel() as usize;
        let plane1_bytes_per_pixel = info.plane1_element_size as usize;

        debug_assert_eq!(range.width as usize * out_bytes_per_pixel, out.len());
        debug_assert_eq!(plane1.len(), range.width as usize * plane1_bytes_per_pixel);
        debug_assert_eq!(
            plane2.len(),
            div_ceil(range.offset + range.width, info.sub_sampling.0 as u32) as usize
                * info.plane2_element_size as usize
        );

        let buffer_bytes_per_pixel = self.native_color.bytes_per_pixel() as usize;
        let buffer_pixels = Self::BUFFER_BYTES / buffer_bytes_per_pixel;
        let buffer = cast::as_bytes_mut(&mut self.buffer);

        // To simplify the following code, start by handling the width offset, so that
        // the general case can assume `range.offset == 0`.
        if range.offset != 0 {
            let offset_width = (info.sub_sampling.0 as u32 - range.offset).min(range.width);

            let plane1_chunk = &plane1[..offset_width as usize * plane1_bytes_per_pixel];
            let plane2_chunk = &plane2[..info.plane2_element_size as usize];
            let buffer_chunk = &mut buffer[..offset_width as usize * buffer_bytes_per_pixel];
            let out_chunk = &mut out[..offset_width as usize * out_bytes_per_pixel];

            // decode into the temporary buffer
            f(
                plane1_chunk,
                plane2_chunk,
                buffer_chunk,
                PlaneRange {
                    offset: range.offset,
                    width: offset_width,
                    y: range.y,
                },
            );

            // convert the channels into the output buffer
            convert_channels_for(self.native_color, self.target, buffer_chunk, out_chunk);

            // adjust inputs
            range.offset = 0;
            range.width -= offset_width;
            plane1 = &plane1[offset_width as usize * plane1_bytes_per_pixel..];
            plane2 = &plane2[info.plane2_element_size as usize..];
            out = &mut out[offset_width as usize * out_bytes_per_pixel..];
        }

        debug_assert!(range.offset == 0);
        let preferred_chunk_size =
            util::round_down_to_multiple(buffer_pixels, info.sub_sampling.0 as usize);
        for chunk_start in (0..range.width as usize).step_by(preferred_chunk_size) {
            let chunk_end = (chunk_start + preferred_chunk_size).min(range.width as usize);
            let chunk_size = chunk_end - chunk_start;

            let plane2_start = chunk_start / info.sub_sampling.0 as usize;
            let plane2_end = div_ceil(chunk_end, info.sub_sampling.0 as usize);

            let plane1_chunk =
                &plane1[chunk_start * plane1_bytes_per_pixel..chunk_end * plane1_bytes_per_pixel];
            let plane2_chunk = &plane2[plane2_start * info.plane2_element_size as usize
                ..plane2_end * info.plane2_element_size as usize];
            let buffer_chunk = &mut buffer[..chunk_size * buffer_bytes_per_pixel];
            let out_chunk =
                &mut out[chunk_start * out_bytes_per_pixel..chunk_end * out_bytes_per_pixel];

            // decode into the temporary buffer
            f(
                plane1_chunk,
                plane2_chunk,
                buffer_chunk,
                PlaneRange {
                    offset: 0,
                    width: chunk_size as u32,
                    y: range.y,
                },
            );

            // convert the channels into the output buffer
            convert_channels_for(self.native_color, self.target, buffer_chunk, out_chunk);
        }
    }
}

/// A buffer holding raw encoded lines of pixels straight from the reader.
struct UntypedLineBuffer {
    buf: Box<[u8]>,
    buf_filled: usize,
    bytes_per_line: usize,
    /// How many lines are still left to read from disk
    lines_on_disk: usize,
    /// The index at which the current line starts in the buffer.
    ///
    /// If `>= buffer.len()`, the buffer is empty and needs to be refilled.
    current_line_start: usize,
}
impl UntypedLineBuffer {
    fn new(
        bytes_per_line: usize,
        height: u32,
        context: &mut DecodeContext,
    ) -> Result<Self, DecodingError> {
        const TARGET_BUFFER_SIZE: usize = 64 * 1024; // 64 KB

        let lines_in_buffer = (TARGET_BUFFER_SIZE / bytes_per_line).clamp(1, height as usize);
        let buf_len = lines_in_buffer * bytes_per_line;
        let buf = context.alloc(buf_len)?;

        Ok(Self {
            buf,
            buf_filled: 0,
            bytes_per_line,
            lines_on_disk: height as usize,
            current_line_start: buf_len,
        })
    }

    // CURSE YOU, lack of trait up-casting
    fn next_line<R: Read + ?Sized>(&mut self, r: &mut R) -> Result<Option<&[u8]>, DecodingError> {
        if self.current_line_start >= self.buf_filled {
            if self.lines_on_disk == 0 {
                // all lines have been read
                return Ok(None);
            }

            // refill the buffer
            let lines_to_read = (self.buf.len() / self.bytes_per_line).min(self.lines_on_disk);
            self.lines_on_disk -= lines_to_read;
            self.buf_filled = lines_to_read * self.bytes_per_line;
            r.read_exact(&mut self.buf[..self.buf_filled])?;
            self.current_line_start = 0;
        }

        // get a line from the buffer
        let line_end = self.current_line_start + self.bytes_per_line;
        let line = &self.buf[self.current_line_start..line_end];
        self.current_line_start = line_end;
        Ok(Some(line))
    }
}

pub(crate) struct PlaneRange {
    pub offset: u32,
    pub width: u32,
    pub y: u8,
}
pub(crate) type ProcessBiPlanarFn =
    fn(plane1: &[u8], plane2: &[u8], decoded: &mut [u8], range: PlaneRange);

/// A helper function for implementing [`ProcessPixelsFn`]s.
#[inline]
pub(crate) fn process_bi_planar_helper<
    const SUB_SAMPLING_X: usize,
    Plane1: cast::FromLeBytes + Copy + Default,
    Plane2: cast::FromLeBytes,
    OutPixel: cast::IntoNeBytes + Copy,
>(
    plane1: &[u8],
    plane2: &[u8],
    decoded: &mut [u8],
    mut range: PlaneRange,
    f: impl Fn([Plane1; SUB_SAMPLING_X], Plane2, u8) -> [OutPixel; SUB_SAMPLING_X],
) {
    // group bytes into chunks
    let mut plane1: &[Plane1::Bytes] = cast::from_bytes(plane1).expect("Invalid plane1 buffer");
    let mut plane2: &[Plane2::Bytes] = cast::from_bytes(plane2).expect("Invalid plane2 buffer");
    let mut decoded: &mut [OutPixel::Bytes] =
        cast::from_bytes_mut(decoded).expect("Invalid output buffer");

    // handle offset
    if range.offset > 0 {
        debug_assert!(range.offset < SUB_SAMPLING_X as u32);
        let w = (SUB_SAMPLING_X - range.offset as usize).min(range.width as usize);

        let mut plane1_items = [Plane1::default(); SUB_SAMPLING_X];
        for x in 0..w {
            plane1_items[x] = Plane1::from_le_bytes(plane1[x]);
        }
        let plane2_item = Plane2::from_le_bytes(plane2[0]);

        let out = f(plane1_items, plane2_item, range.y);
        for x in 0..w {
            decoded[x] = cast::IntoNeBytes::into_ne_bytes(out[x]);
        }

        // adjust inputs
        range.offset = 0;
        range.width -= w as u32;
        plane1 = &plane1[w..];
        plane2 = &plane2[1..];
        decoded = &mut decoded[w..];
    }

    // full macro pixels
    let full = range.width as usize / SUB_SAMPLING_X;
    let full_w = full * SUB_SAMPLING_X;
    let plane1_full: &[[Plane1::Bytes; SUB_SAMPLING_X]] =
        cast::as_array_chunks(&plane1[..full_w]).expect("Invalid plane1 buffer");
    let plane2_full: &[Plane2::Bytes] = &plane2[..full];
    let decoded_full: &mut [[OutPixel::Bytes; SUB_SAMPLING_X]] =
        cast::as_array_chunks_mut(&mut decoded[..full * SUB_SAMPLING_X])
            .expect("Invalid output buffer");

    for x in 0..full {
        let plane1_items = plane1_full[x].map(Plane1::from_le_bytes);
        let plane2_item = Plane2::from_le_bytes(plane2_full[x]);
        let out = f(plane1_items, plane2_item, range.y);
        decoded_full[x] = out.map(cast::IntoNeBytes::into_ne_bytes);
    }

    // rest
    let rest_w = range.width as usize - full * SUB_SAMPLING_X;
    if rest_w > 0 {
        let mut plane1_items = [Plane1::default(); SUB_SAMPLING_X];
        for x in 0..rest_w {
            plane1_items[x] = Plane1::from_le_bytes(plane1[full_w + x]);
        }
        let plane2_item = Plane2::from_le_bytes(plane2[full]);

        let out = f(plane1_items, plane2_item, range.y);
        for x in 0..rest_w {
            decoded[full_w + x] = cast::IntoNeBytes::into_ne_bytes(out[x]);
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct BiPlaneInfo {
    pub plane1_element_size: u8,
    pub plane2_element_size: u8,
    /// The sub-sampling of plane2.
    pub sub_sampling: (u8, u8),
}
pub(crate) fn for_each_bi_planar(
    r: &mut dyn Read,
    image: &mut ImageViewMut,
    mut context: DecodeContext,
    native_color: ColorFormat,
    info: BiPlaneInfo,
    process_bi_planar: ProcessBiPlanarFn,
) -> Result<(), DecodingError> {
    let size = context.surface_size;
    debug_assert_eq!(image.color().precision, native_color.precision);

    // Step 1: Read the entirety of plane 1
    let plain1_bytes_per_line = size.width as usize * info.plane1_element_size as usize;
    let plane1 = context.alloc_read(plain1_bytes_per_line as u64 * size.height as u64, r)?;

    // Step 2: Go through plane 2
    let sub_sampling_x = info.sub_sampling.0 as u32;
    let sub_sampling_y = info.sub_sampling.1 as u32;

    let uv_width = div_ceil(size.width, sub_sampling_x);
    let uv_lines = div_ceil(size.height, sub_sampling_y);
    let uv_bytes_per_line = uv_width as usize * info.plane2_element_size as usize;

    let mut line_buffer = UntypedLineBuffer::new(uv_bytes_per_line, uv_lines, &mut context)?;
    let mut conversion_buffer = ChannelConversionBuffer::new(native_color, image.color().channels);

    let mut y: usize = 0;
    while let Some(uv_line) = line_buffer.next_line(r)? {
        debug_assert!(y < size.height as usize);

        for y_offset in 0..sub_sampling_y as u8 {
            if y >= size.height as usize {
                break;
            }

            let plane1_line = &plane1[y * plain1_bytes_per_line..(y + 1) * plain1_bytes_per_line];
            let out_line = image.get_row(y);

            conversion_buffer.process_bi_planar(
                info,
                plane1_line,
                uv_line,
                out_line,
                PlaneRange {
                    offset: 0,
                    width: size.width,
                    y: y_offset,
                },
                process_bi_planar,
            );

            y += 1;
        }
    }

    Ok(())
}
pub(crate) fn for_each_bi_planar_rect(
    r: &mut dyn ReadSeek,
    image: &mut ImageViewMut,
    offset: Offset,
    mut context: DecodeContext,
    native_color: ColorFormat,
    info: BiPlaneInfo,
    process_bi_planar: ProcessBiPlanarFn,
) -> Result<(), DecodingError> {
    let surface_size = context.surface_size;
    let image_width = image.width();
    let image_height = image.height();

    debug_assert_eq!(image.color().precision, native_color.precision);

    // Step 1: Read the entirety of plane 1
    let plain1_bytes_per_line = surface_size.width as usize * info.plane1_element_size as usize;
    util::io_skip_exact(r, plain1_bytes_per_line as u64 * offset.y as u64)?;
    let plane1 = context.alloc_read(plain1_bytes_per_line as u64 * image_height as u64, r)?;
    util::io_skip_exact(
        r,
        plain1_bytes_per_line as u64 * (surface_size.height - offset.y - image_height) as u64,
    )?;

    // Step 2: Go through plane 2
    let sub_sampling_x = info.sub_sampling.0 as u32;
    let sub_sampling_y = info.sub_sampling.1 as u32;

    let uv_before = offset.y / sub_sampling_y;
    let uv_after = div_ceil(surface_size.height, sub_sampling_y)
        - div_ceil(offset.y + image_height, sub_sampling_y);
    let uv_width = div_ceil(surface_size.width, sub_sampling_x);
    let uv_lines = div_ceil(surface_size.height, sub_sampling_y) - uv_before - uv_after;
    let uv_bytes_per_line = uv_width as usize * info.plane2_element_size as usize;

    util::io_skip_exact(r, uv_before as u64 * uv_bytes_per_line as u64)?;

    let mut line_buffer = UntypedLineBuffer::new(uv_bytes_per_line, uv_lines, &mut context)?;
    let mut conversion_buffer = ChannelConversionBuffer::new(native_color, image.color().channels);

    let mut y: usize = uv_before as usize * sub_sampling_y as usize;
    while let Some(uv_line) = line_buffer.next_line(r)? {
        debug_assert!(y < (offset.y + image_height) as usize);

        for y_offset in 0..sub_sampling_y as u8 {
            if y < offset.y as usize {
                y += 1;
                continue;
            }
            if y >= (offset.y + image_height) as usize {
                break;
            }

            let plane1_start = (y - offset.y as usize) * plain1_bytes_per_line
                + offset.x as usize * info.plane1_element_size as usize;
            let plane1_line = &plane1[plane1_start
                ..plane1_start + image_width as usize * info.plane1_element_size as usize];

            let out_line = image.get_row(y - offset.y as usize);

            let uv_start = (offset.x / sub_sampling_x) as usize * info.plane2_element_size as usize;
            let uv_end = div_ceil(offset.x + image_width, sub_sampling_x) as usize
                * info.plane2_element_size as usize;
            let uv_line = &uv_line[uv_start..uv_end];

            let offset = offset.x % sub_sampling_x;

            conversion_buffer.process_bi_planar(
                info,
                plane1_line,
                uv_line,
                out_line,
                PlaneRange {
                    offset,
                    width: image_width,
                    y: y_offset,
                },
                process_bi_planar,
            );

            y += 1;
        }
    }

    util::io_skip_exact(r, uv_after as u64 * uv_bytes_per_line as u64)?;

    Ok(())
}

pub(crate) fn read_exact_image<R: Read + ?Sized>(
    r: &mut R,
    image: &mut ImageViewMut,
) -> Result<(), std::io::Error> {
    if image.is_contiguous() {
        // we can read everything in one go
        r.read_exact(image.data())
    } else {
        // read row by row
        for row in image.rows_mut() {
            r.read_exact(row)?;
        }
        Ok(())
    }
}

pub(crate) fn for_each_slice(image: &mut ImageViewMut, mut f: impl FnMut(&mut [u8])) {
    if image.is_contiguous() {
        f(image.data())
    } else {
        image.rows_mut().for_each(f);
    }
}
