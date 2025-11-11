use crate::{convert_to_rgba_f32, ImageView};

/// Converts the given images to f32 RGBA, `block_height` rows at a time. The
/// closure `f` is called with a mutable slice of `[[f32; 4]]` that contains
/// `block_height` rows of the image, each row containing `width` pixels.
///
/// If the image height is not a multiple of `block_height`, the implementation
/// will fill the missing rows by duplicating previous rows.
pub(crate) fn for_each_f32_rgba_rows<E>(
    image: ImageView,
    block_height: usize,
    mut f: impl FnMut(&mut [[f32; 4]]) -> Result<(), E>,
) -> Result<(), E> {
    debug_assert!(block_height != 0);

    let color = image.color();
    let width = image.width() as usize;
    let height = image.height() as usize;

    // this is the one and only buffer we need
    let mut intermediate_buffer = vec![[0_f32; 4]; width * block_height].into_boxed_slice();

    // go through the image row by row, convert it to f32 RGBA, and then
    // pass it to the closure
    let mut rows = image.rows();

    let full_blocks = height / block_height;
    for _ in 0..full_blocks {
        // fill the intermediate buffer
        for i in 0..block_height {
            convert_to_rgba_f32(
                color,
                rows.next().expect("Image has too few rows"),
                &mut intermediate_buffer[i * width..(i + 1) * width],
            );
        }

        f(&mut intermediate_buffer)?;
    }

    let rest_blocks = height % block_height;
    if rest_blocks > 0 {
        // fill intermediate buffer with the remaining rows
        for i in 0..rest_blocks {
            convert_to_rgba_f32(
                color,
                rows.next().expect("Image has too few rows"),
                &mut intermediate_buffer[i * width..(i + 1) * width],
            );
        }
        debug_assert!(rows.next().is_none());

        // fill missing rows
        for i in rest_blocks..block_height {
            // copy the first line to fill the rest
            // TODO: maybe change this to fill with the last line?
            intermediate_buffer.copy_within(..width, i * width);
        }

        f(&mut intermediate_buffer)?;
    }

    Ok(())
}

pub(crate) fn for_each_chunk<T, E>(
    image: ImageView,
    buffer: &mut [T],
    buffer_elements_per_pixel: usize,
    mut copy_to_buffer: impl FnMut(&[u8], &mut [T]),
    mut process_chunk: impl FnMut(&mut [T]) -> Result<(), E>,
) -> Result<(), E> {
    let buffer_pixels = buffer.len() / buffer_elements_per_pixel;
    let buffer = &mut buffer[..buffer_pixels * buffer_elements_per_pixel];
    let bytes_per_pixel = image.color().bytes_per_pixel() as usize;

    if image.is_contiguous() {
        // Since the image is contiguous, we can process it in chunks directly
        for chunk in image.data().chunks(buffer_pixels * bytes_per_pixel) {
            let pixels = chunk.len() / bytes_per_pixel;
            let chunk_buffer = &mut buffer[..pixels * buffer_elements_per_pixel];
            copy_to_buffer(chunk, chunk_buffer);
            process_chunk(chunk_buffer)?;
        }
    } else {
        // With non-contiguous images, we need to build up the buffer from
        // multiple rows
        let mut fill_pixels = 0;
        for mut row in image.rows() {
            while !row.is_empty() {
                if fill_pixels == buffer_pixels {
                    // Buffer is full, flush it
                    process_chunk(buffer)?;
                    fill_pixels = 0;
                }

                debug_assert!(row.len() % bytes_per_pixel == 0);
                let row_pixels = row.len() / bytes_per_pixel;
                let write_pixels = row_pixels.min(buffer_pixels - fill_pixels);
                copy_to_buffer(
                    &row[..write_pixels * bytes_per_pixel],
                    &mut buffer[fill_pixels * buffer_elements_per_pixel
                        ..(fill_pixels + write_pixels) * buffer_elements_per_pixel],
                );
                fill_pixels += write_pixels;
                row = &row[write_pixels * bytes_per_pixel..];
            }
        }
        if fill_pixels > 0 {
            // Flush the remaining data in the buffer
            process_chunk(&mut buffer[..fill_pixels * buffer_elements_per_pixel])?;
        }
    }

    Ok(())
}
