use glam::Vec3A;

/// Indicates the color is not in RGB/sRGB, but a different color space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub(crate) struct ColorSpace(pub Vec3A);
impl std::ops::Add for ColorSpace {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ColorSpace(self.0 + rhs.0)
    }
}
impl std::ops::AddAssign for ColorSpace {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl std::ops::Sub for ColorSpace {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ColorSpace(self.0 - rhs.0)
    }
}
impl std::ops::Mul<f32> for ColorSpace {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        ColorSpace(self.0 * rhs)
    }
}

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

#[derive(Debug, Clone)]
pub(crate) struct RefinementOptions {
    /// The initial step size,
    pub step_initial: f32,
    /// The step size will be multiplied by this value after each iteration.
    pub step_decay: f32,
    /// The minimum step size. The process is over when the step size is
    /// smaller than this value.
    pub step_min: f32,
    /// The maximum number of iterations.
    pub max_iter: u32,
}
impl RefinementOptions {
    pub fn new_bc4(min: f32, max: f32) -> Self {
        Self {
            step_initial: 0.15 * (max - min),
            step_decay: 0.5,
            step_min: 1. / 255. / 2.,
            max_iter: 10,
        }
    }
    pub fn new_bc4_fast(min: f32, max: f32) -> Self {
        Self {
            step_initial: 0.1 * (max - min),
            step_decay: 0.5,
            step_min: 1. / 255.,
            max_iter: 2,
        }
    }
    pub fn new_bc1(dist: f32, max_iter: u32) -> Self {
        Self {
            step_initial: 0.5 * dist,
            step_decay: 0.5,
            step_min: 1. / 64.,
            max_iter,
        }
    }
}
pub(crate) fn refine_endpoints<T: RefinementSteps>(
    min: T,
    max: T,
    options: RefinementOptions,
    mut compute_error: impl FnMut((T, T)) -> f32,
) -> (T, T) {
    let mut step = options.step_initial;
    let mut best = (min, max);
    let mut iters = 0;
    if !(step > options.step_min && iters < options.max_iter) {
        return best;
    }

    let mut error = compute_error((min, max));
    while step > options.step_min && iters < options.max_iter {
        RefinementSteps::for_each_endpoint(best, step, |current| {
            let new_error = compute_error(current);
            if new_error < error {
                error = new_error;
                best = current;
            }
        });
        step *= options.step_decay;
        iters += 1;
    }

    best
}
pub(crate) trait RefinementSteps
where
    Self: Copy + Sized,
{
    fn for_each_endpoint(start: (Self, Self), step: f32, f: impl FnMut((Self, Self)));
}
impl RefinementSteps for f32 {
    fn for_each_endpoint((min, max): (f32, f32), step: f32, mut f: impl FnMut((f32, f32))) {
        for (delta_min, delta_max) in [(step, 0.0), (0.0, step), (-step, 0.0), (0.0, -step)] {
            let new_min = (min + delta_min).clamp(0.0, 1.0);
            let new_max = (max + delta_max).clamp(0.0, 1.0);
            if new_min < new_max {
                f((new_min, new_max));
            }
        }
    }
}
impl RefinementSteps for Vec3A {
    fn for_each_endpoint((min, max): (Vec3A, Vec3A), step: f32, mut f: impl FnMut((Vec3A, Vec3A))) {
        let main_dir_1 = (min - max).try_normalize().unwrap_or(Vec3A::X);
        let (main_dir_2, main_dir_3) = main_dir_1.any_orthonormal_pair();

        let directions = [
            main_dir_1 * step,
            main_dir_2 * step,
            main_dir_3 * step,
            main_dir_1 * -step,
            main_dir_2 * -step,
            main_dir_3 * -step,
        ];
        for &dir in &directions {
            let new_min = (min + dir).clamp(Vec3A::ZERO, Vec3A::ONE);
            f((new_min, max));
        }
        for &dir in &directions {
            let new_max = (max + dir).clamp(Vec3A::ZERO, Vec3A::ONE);
            f((min, new_max));
        }
    }
}
impl RefinementSteps for ColorSpace {
    fn for_each_endpoint(
        (min, max): (ColorSpace, ColorSpace),
        step: f32,
        mut f: impl FnMut((ColorSpace, ColorSpace)),
    ) {
        Vec3A::for_each_endpoint((min.0, max.0), step, move |(min, max)| {
            f((ColorSpace(min), ColorSpace(max)));
        });
    }
}
