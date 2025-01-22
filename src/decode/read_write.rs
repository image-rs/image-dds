//! An internal module with helper methods for reading bytes from a reader, and
//! writing decoded pixels to the output buffer.

use std::io::{Read, SeekFrom};
use std::mem::size_of;

use crate::{cast, util::div_ceil, DecodeError, Rect, Size};

use super::ReadSeek;

/// Helper method for decoding UNCOMPRESSED formats.
pub(crate) fn for_each_pixel_untyped<InPixel, OutPixel>(
    r: &mut dyn Read,
    buf: &mut [u8],
    process_pixels: fn(&[u8], &mut [u8]),
) -> Result<(), DecodeError> {
    fn inner(
        r: &mut dyn Read,
        buf: &mut [u8],
        size_of_in: usize,
        size_of_out: usize,
        process_pixels: fn(&[u8], &mut [u8]),
    ) -> Result<(), DecodeError> {
        assert!(buf.len() % size_of_out == 0);
        let pixels = buf.len() / size_of_out;

        let mut read_buffer = UntypedPixelBuffer::new(pixels, size_of_in);
        for buf in buf.chunks_mut(read_buffer.buffered_pixels() * size_of_out) {
            let row = read_buffer.read(r)?;
            debug_assert!(row.len() % size_of_in == 0);
            debug_assert!(buf.len() % size_of_out == 0);
            debug_assert_eq!(row.len() / size_of_in, buf.len() / size_of_out);
            process_pixels(row, buf);
        }
        Ok(())
    }

    inner(
        r,
        buf,
        size_of::<InPixel>(),
        size_of::<OutPixel>(),
        process_pixels,
    )
}

/// Helper method for decoding UNCOMPRESSED formats.
pub(crate) fn for_each_pixel_rect_untyped<InPixel, OutPixel>(
    r: &mut dyn ReadSeek,
    buf: &mut [u8],
    row_pitch: usize,
    size: Size,
    rect: Rect,
    process_pixels: fn(&[u8], &mut [u8]),
) -> Result<(), DecodeError> {
    #[allow(clippy::too_many_arguments)]
    fn inner(
        r: &mut dyn ReadSeek,
        buf: &mut [u8],
        row_pitch: usize,
        size: Size,
        rect: Rect,
        size_of_in: usize,
        size_of_out: usize,
        process_pixels: fn(&[u8], &mut [u8]),
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

        // jump to the first pixel
        seek_relative(
            r,
            encoded_bytes_per_row * rect.y as i64 + encoded_bytes_before_rect,
        )?;

        let pixels_per_line = rect.width as usize;
        let mut row: Box<[u8]> =
            vec![Default::default(); pixels_per_line * size_of_in].into_boxed_slice();
        for y in 0..rect.height {
            if y > 0 {
                // jump to the first pixel in the next row
                // (this has already been done for the first row; see above)
                seek_relative(r, encoded_bytes_before_rect + encoded_bytes_after_rect)?;
            }

            // read next line
            r.read_exact(&mut row)?;

            let buf_start = y as usize * row_pitch;
            let buf_len = pixels_per_line * size_of_out;
            let buf = &mut buf[buf_start..(buf_start + buf_len)];
            debug_assert_eq!(row.len() / size_of_in, buf.len() / size_of_out);
            process_pixels(&row, buf);
        }

        // jump to the end of the surface to put the reader into a known position
        seek_relative(
            r,
            encoded_bytes_after_rect
                + (size.height - rect.y - rect.height) as i64 * encoded_bytes_per_row,
        )?;

        Ok(())
    }

    inner(
        r,
        buf,
        row_pitch,
        size,
        rect,
        size_of::<InPixel>(),
        size_of::<OutPixel>(),
        process_pixels,
    )
}

fn seek_relative(r: &mut dyn ReadSeek, offset: i64) -> std::io::Result<()> {
    if offset != 0 {
        r.seek(SeekFrom::Current(offset))?;
    }
    Ok(())
}

/// Utility method for sub-sampled formats.
///
/// Since there currently are only 2 sub-sampled format, this method isn't any
/// more generic than it has to be. Both formats:
///
/// 1. Encode pairs of pixels as 4 bytes.
/// 2. Ignore the last encoded pixel if the width is odd.
pub(crate) fn for_each_pair<OutPixel>(
    r: &mut dyn Read,
    out: &mut [u8],
    size: Size,
    process_pixel1: impl Fn([u8; 4]) -> OutPixel,
    process_pixel2: impl Fn([u8; 4]) -> OutPixel,
) -> Result<(), DecodeError>
where
    OutPixel: cast::Castable + Default + Copy,
{
    // The basic idea here is to decode the image line by line. A line is a
    // sequence of encoded pixels pairs that together describe a single row of
    // pixels in the final image.
    //
    // Since reading a bunch of small lines from disk is slow, we allocate one
    // large buffer to hold N lines at a time. The we process the lines in the
    // buffer and refill as needed.

    assert!(!size.is_empty());

    let line_len = div_ceil(size.width, 2) as usize;
    let mut line_buffer: LineBuffer<[u8; 4]> = LineBuffer::new(line_len, size.height as usize);
    let mut write_aligned: AlignedWriter<[OutPixel; 2]> = AlignedWriter::new();

    let out_line_len = size.width as usize * size_of::<OutPixel>();
    let mut y = 0;
    while let Some(line) = line_buffer.next_line(r)? {
        let out_line = out[y * out_line_len..(y + 1) * out_line_len].as_mut();

        // write all whole pairs of pixels
        let whole_pairs = size.width as usize / 2;
        let whole_pairs_len = whole_pairs * 2 * size_of::<OutPixel>();
        write_aligned.write(&mut out_line[..whole_pairs_len], |out_line| {
            for (out, pixel) in out_line.iter_mut().zip(line) {
                out[0] = process_pixel1(*pixel);
                out[1] = process_pixel2(*pixel);
            }
        });

        // write the sole last pixel if the width is odd
        if size.width % 2 == 1 {
            let pixel = process_pixel1(line[line.len() - 1]);
            let pixel_array = [pixel];
            // Using memcpy here is pretty inefficient, but idc
            out_line[whole_pairs_len..].copy_from_slice(cast::as_bytes(&pixel_array));
        }

        y += 1;
    }
    Ok(())
}

/// Utility method for 4x4 block-compressed formats.
pub(crate) fn for_each_block_4x4<const BYTES_PER_BLOCK: usize, OutPixel>(
    r: &mut dyn Read,
    out: &mut [u8],
    size: Size,
    process_block: impl Fn([u8; BYTES_PER_BLOCK]) -> [OutPixel; 16],
) -> Result<(), DecodeError>
where
    [u8; BYTES_PER_BLOCK]: cast::Castable + Default,
    OutPixel: cast::Castable + Default + Copy,
{
    // The basic idea here is to decode the image line by line. A line is a
    // sequence of encoded pixels pairs that together describe a single row of
    // pixels in the final image.
    //
    // Since reading a bunch of small lines from disk is slow, we allocate one
    // large buffer to hold N lines at a time. The we process the lines in the
    // buffer and refill as needed.

    assert!(!size.is_empty());

    let width_blocks = div_ceil(size.width, 4) as usize;
    let height_blocks = div_ceil(size.height, 4) as usize;

    let mut line_buffer: LineBuffer<[u8; BYTES_PER_BLOCK]> =
        LineBuffer::new(width_blocks, height_blocks);
    let mut write_aligned: AlignedWriter<OutPixel> = AlignedWriter::new();

    let mut block_y = 0;
    while let Some(block_line) = line_buffer.next_line(r)? {
        // how many rows of pixels we'll decode
        // this is usually 4, but might be less for the last block
        let pixel_rows = 4.min(size.height - block_y * 4) as usize;
        let pixel_row_bytes = size.width as usize * size_of::<OutPixel>();
        let start_pixel_y = block_y as usize * 4;
        let out_pixel_rows = out
            [(start_pixel_y * pixel_row_bytes)..(start_pixel_y + pixel_rows) * pixel_row_bytes]
            .as_mut();

        write_aligned.write(out_pixel_rows, |out| {
            if size.width % 4 == 0 && pixel_rows == 4 {
                // This is the easy case where we can always write the entire
                // 4x4 block out without needing to do bounds checking.
                assert!(out.len() == 4 * size.width as usize);

                for (block_index, block) in block_line.iter().enumerate() {
                    let pixels = process_block(*block);
                    for y in 0..4 {
                        let row_start = block_index * 4 + y * size.width as usize;
                        let row = &mut out[row_start..row_start + 4];
                        for x in 0..4 {
                            row[x] = pixels[y * 4 + x];
                        }
                    }
                }
            } else {
                let block_h = pixel_rows;
                for (block_index, block) in block_line.iter().enumerate() {
                    let block_w = 4.min(size.width as usize - block_index * 4);
                    let pixels = process_block(*block);
                    for y in 0..block_h {
                        let row_start = block_index * 4 + y * size.width as usize;
                        let row = &mut out[row_start..row_start + block_w];
                        for x in 0..block_w {
                            row[x] = pixels[y * 4 + x];
                        }
                    }
                }
            }
        });

        block_y += 1;
    }
    Ok(())
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
struct LineBuffer<T> {
    /// Number of `T`s in a line.
    width: usize,
    /// How many lines are still to read from disk
    lines_left_to_read: usize,
    /// The index at which the next line starts in the buffer.
    ///
    /// If >= buffer.len(), the buffer is empty and needs to be refilled.
    next_line_start: usize,
    buffer: Vec<T>,
}
impl<T> LineBuffer<T> {
    fn new(width: usize, height: usize) -> Self
    where
        T: Default + Clone,
    {
        const TARGET_BUFFER_SIZE: usize = 64 * 1024; // 64 KB

        let lines_in_buffer = (TARGET_BUFFER_SIZE / (width * size_of::<T>())).clamp(1, height);
        // TODO: protect against allocating very large buffers (> 1 MB)
        let buffer = vec![T::default(); width * lines_in_buffer];

        Self {
            width,
            lines_left_to_read: height,
            next_line_start: buffer.len(),
            buffer,
        }
    }

    fn next_line(&mut self, r: &mut dyn Read) -> Result<Option<&[T]>, DecodeError>
    where
        T: cast::Castable,
    {
        if self.next_line_start >= self.buffer.len() {
            if self.lines_left_to_read == 0 {
                // all lines have been read
                return Ok(None);
            }

            // refill the buffer
            let lines_to_read = (self.buffer.len() / self.width).min(self.lines_left_to_read);
            self.lines_left_to_read -= lines_to_read;
            self.buffer.truncate(lines_to_read * self.width);
            r.read_exact(cast::as_bytes_mut(&mut self.buffer))?;
            self.next_line_start = 0;
        }

        // get a line from the buffer
        let line_end = self.next_line_start + self.width;
        let line = &self.buffer[self.next_line_start..line_end];
        self.next_line_start = line_end;
        Ok(Some(line))
    }
}

/// A helper to write `T` into a `[u8]` buffer.
struct AlignedWriter<T> {
    temp_buf: Option<Vec<T>>,
}
impl<T> AlignedWriter<T> {
    fn new() -> Self {
        Self { temp_buf: None }
    }

    fn element_size(&self) -> usize {
        size_of::<T>()
    }

    // Don't inline this method for code size reasons.
    #[inline(never)]
    fn get_temp(&mut self, buf_len: usize) -> &mut [T]
    where
        T: Default + Copy,
    {
        let size = self.element_size();
        assert_eq!(buf_len % size, 0);
        let len = buf_len / size;

        // create a buffer if it doesn't exist
        if self.temp_buf.is_none() {
            self.temp_buf = Some(vec![T::default(); len]);
        }

        // NOTE: This unwrap cannot fail
        let temp = self.temp_buf.as_deref_mut().unwrap();
        assert!(temp.len() >= len);
        &mut temp[..len]
    }

    fn write(&mut self, buf: &mut [u8], write: impl FnOnce(&mut [T]))
    where
        T: cast::Castable + Default + Copy,
    {
        // The whole method is structured around calling `write` at one place.
        // This is done to promote inlining the `write` closure, while also
        // reducing code size.

        enum Aligned<'a, 'b, T> {
            Slice(&'a mut [T]),
            Temp(&'b mut [T]),
        }
        impl<T> Aligned<'_, '_, T> {
            fn as_mut(&mut self) -> &mut [T] {
                match self {
                    Self::Slice(buf) => buf,
                    Self::Temp(buf) => buf,
                }
            }
        }

        let mut write_to: Aligned<T> = if let Some(buf) = cast::from_bytes_mut(buf) {
            // the buffer is aligned already
            Aligned::Slice(buf)
        } else {
            // use a temporary buffer and copy over the result later
            Aligned::Temp(self.get_temp(buf.len()))
        };

        write(write_to.as_mut());

        if let Aligned::Temp(temp) = write_to {
            // copy the result to the output buffer
            buf.copy_from_slice(cast::as_bytes(temp));
        }
    }
}
