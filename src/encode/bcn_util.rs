/// This will dither within a 4x4 block.
///
/// The input function `f` is called with:
/// - the index of the pixel
/// - the pixel value (this includes any diffused error)
///
/// and returns:
/// - the closest value in the palette
pub(crate) fn block_dither<T>(block: impl Block4x4<T>, mut get_closest: impl FnMut(usize, T) -> T)
where
    T: Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + std::ops::Sub<Output = T>
        + std::ops::Mul<f32, Output = T>,
{
    // This implements a modified version of the Floyd-Steinberg dithering
    let mut error_map: [T; 16] = Default::default();
    for y in 0..4 {
        for x in 0..4 {
            let pixel_index = y * 4 + x;
            let pixel = block.get_pixel_at(pixel_index) + error_map[pixel_index];
            let closest = get_closest(pixel_index, pixel);
            let error = pixel - closest;

            // diffuse the error
            let mut weight_right = 7. / 16.;
            let mut weight_next_left = 3. / 16.;
            let mut weight_next_middle = 5. / 16.;
            let mut weight_next_right = 1. / 16.;
            // adjust the weights, so we lose as little of the error as possible
            if x == 0 {
                weight_next_left = 0.0;
                weight_next_middle = 6. / 16.;
                weight_next_right = 2. / 16.;
            }
            if x == 3 {
                // we lose 25% of the error, per pixel in the last column
                weight_right = 0.0;
                weight_next_left = 5. / 16.;
                weight_next_middle = 7. / 16.;
                weight_next_right = 0.0;
            }
            if y == 3 {
                // we lose 50% of the error, per pixel in the last row
                weight_right = 8. / 16.;
                weight_next_left = 0.0;
                weight_next_middle = 0.0;
                weight_next_right = 0.0;
            }

            if x < 3 {
                error_map[pixel_index + 1] += error * weight_right;
            }
            if y < 3 {
                if x > 0 {
                    error_map[pixel_index + 4 - 1] += error * weight_next_left;
                }
                error_map[pixel_index + 4] += error * weight_next_middle;
                if x < 3 {
                    error_map[pixel_index + 4 + 1] += error * weight_next_right;
                }
            }
        }
    }
}

pub(crate) trait Block4x4<T> {
    /// Returns the pixel at the given index.
    ///
    /// The index must be in the range `0..16`.
    fn get_pixel_at(&self, index: usize) -> T;
}
impl<T: Copy> Block4x4<T> for &[T; 16] {
    #[inline(always)]
    fn get_pixel_at(&self, index: usize) -> T {
        self[index]
    }
}
impl<T: Copy> Block4x4<T> for T {
    #[inline(always)]
    fn get_pixel_at(&self, _index: usize) -> T {
        *self
    }
}
