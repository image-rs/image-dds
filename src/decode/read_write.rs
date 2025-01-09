//! An internal module with helper methods for reading bytes from a reader, and
//! writing decoded pixels to the output buffer.

use std::io::Read;
use std::mem::size_of;

use crate::{cast, util::div_ceil, DecodeError, Size};

/// Helper method for decoding UNCOMPRESSED formats.
#[inline(always)]
pub(crate) fn for_each_pixel<const N: usize, InChannel, OutPixel>(
    r: &mut dyn Read,
    buf: &mut [u8],
    process_pixel: impl Fn([InChannel; N]) -> OutPixel,
) -> Result<(), DecodeError>
where
    InChannel: FromLe,
    [InChannel::Raw; N]: cast::Castable + Default,
    OutPixel: cast::Castable + Default,
{
    let out_pixel_size = size_of::<OutPixel>();
    assert!(buf.len() % out_pixel_size == 0);
    let pixels = buf.len() / out_pixel_size;

    let mut read_buffer: PixelBuffer<[InChannel::Raw; N]> = PixelBuffer::new(pixels);
    let mut write_aligned: AlignedWriter<OutPixel> = AlignedWriter::new();
    for buf in buf.chunks_mut(read_buffer.buffered_pixels * write_aligned.element_size()) {
        let row = read_buffer.read(r)?;

        write_aligned.write(buf, |buf| {
            for (pixel, out) in row.iter().zip(buf) {
                *out = process_pixel(pixel.map(FromLe::from_le));
            }
        });
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
    OutPixel: cast::Castable + Default,
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

pub(crate) trait FromLe {
    type Raw;

    fn from_le(raw: Self::Raw) -> Self;
}
impl FromLe for u8 {
    type Raw = u8;

    fn from_le(raw: Self::Raw) -> Self {
        raw
    }
}
impl FromLe for u16 {
    type Raw = u16;

    fn from_le(raw: Self::Raw) -> Self {
        u16::from_le(raw)
    }
}
impl FromLe for u32 {
    type Raw = u32;

    fn from_le(raw: Self::Raw) -> Self {
        u32::from_le(raw)
    }
}
impl FromLe for f32 {
    type Raw = u32;

    fn from_le(raw: Self::Raw) -> Self {
        f32::from_bits(u32::from_le(raw))
    }
}

/// A buffer holding raw encoded pixels straight from the reader.
struct PixelBuffer<T> {
    buf: Vec<T>,
    buffered_pixels: usize,
    pixels_left: usize,
}
impl<T> PixelBuffer<T> {
    /// The target buffer size is in bytes. Currently 64 KiB.
    const TARGET: usize = 64 * 1024;

    fn new(pixels: usize) -> Self
    where
        T: Default + Clone,
    {
        let bytes_per_pixel = size_of::<T>();
        let buf_pixels = (Self::TARGET / bytes_per_pixel).min(pixels);
        let buf = vec![T::default(); buf_pixels];
        Self {
            buf,
            buffered_pixels: buf_pixels,
            pixels_left: pixels,
        }
    }

    fn read(&mut self, r: &mut dyn Read) -> Result<&[T], DecodeError>
    where
        T: cast::Castable,
    {
        let pixels_to_read = self.buffered_pixels.min(self.pixels_left);
        assert!(pixels_to_read > 0);
        self.pixels_left -= pixels_to_read;

        let buf = &mut self.buf[..pixels_to_read];
        r.read_exact(cast::as_bytes_mut(buf))?;
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

    fn write(&mut self, buf: &mut [u8], write: impl FnOnce(&mut [T]))
    where
        T: cast::Castable + Default + Copy,
    {
        // TODO: Restructure this method to call `write` exactly once to reduce
        //       code size. The borrow checker is going to hate this though.

        if let Ok(buf) = bytemuck::try_cast_slice_mut(buf) {
            // the buffer is aligned already
            write(buf);
        } else {
            // use a temporary buffer and copy over
            let size = size_of::<T>();
            assert_eq!(buf.len() % size, 0);
            let len = buf.len() / size;

            let temp = if let Some(temp) = self.temp_buf.as_mut() {
                assert!(temp.len() >= len);
                &mut temp[..len]
            } else {
                let temp = vec![T::default(); len];
                self.temp_buf = Some(temp);
                // we just assigned a value, so unwrap is okay
                self.temp_buf.as_mut().unwrap()
            };

            write(temp);

            buf.copy_from_slice(cast::as_bytes(temp));
        }
    }
}
