use glam::{Vec3A, Vec4};

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
impl From<ColorSpace> for Vec3A {
    #[inline(always)]
    fn from(value: ColorSpace) -> Self {
        value.0
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
pub(crate) fn refine_endpoints<T: RefinementSteps, E: PartialOrd>(
    min: T,
    max: T,
    options: RefinementOptions,
    mut compute_error: impl FnMut((T, T)) -> E,
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

/// This stores the result of `E = inv(A^T * A)`, where `A` is the n-by-2
/// weight matrix. Each row in `A` has the form `[w_i, 1 - w_i]`.
///
/// `E` is stored in a form optimized for multiplication with `A^T` such that
/// `E * A^T` can be computed efficiently. Let `G = E * A^T` be a 2-by-n matrix
/// where each column is:
///
/// ```txt
/// ( g_0i ) = ( e00 * w_i + e01 * (1 - w_i) ) = ( e01 + (e00 - e01) * w_i )
/// ( g_1i ) = ( e10 * w_i + e11 * (1 - w_i) ) = ( e11 + (e10 - e11) * w_i )
/// ```
///
/// Only `e01`, `e11`, `e00 - e01`, and `e10 - e11` are stored.
///
/// (Note that `E` is symmetric, so `e10 == e01`.)
struct LeastSquaresWeightMatrix {
    pub e01: f32,
    pub e11: f32,
    /// e00 - e01
    pub e00_01: f32,
    /// e10 - e11
    pub e10_11: f32,
}
impl LeastSquaresWeightMatrix {
    /// Returns a matrix that will set both endpoints to the mean of all input
    /// colors regardless of weights.
    fn mean(color_count: usize) -> Self {
        let r = 1.0 / (color_count as f32);
        Self {
            e01: r,
            e11: r,
            e00_01: 0.0,
            e10_11: 0.0,
        }
    }

    /// ```txt
    /// D = A^T*A = (a b)
    ///             (b c)
    /// ```
    fn from_d(a: f32, b: f32, c: f32) -> Option<Self> {
        // Find D^-1
        let d_det = a * c - b * b;
        if d_det.abs() < f32::EPSILON {
            // All weights are the same, which is makes inversion impossible
            return None;
        }
        // E = D^-1 = ( c/det  -b/det)
        //            (-b/det   a/det)
        let d_det_rep = 1.0 / d_det;
        let (e00, e01, e11) = (c * d_det_rep, -b * d_det_rep, a * d_det_rep);

        Some(Self {
            e01,
            e11,
            e00_01: e00 - e01,
            e10_11: e01 - e11, // e10 == e01
        })
    }
}

/// Least squares fits 2 endpoints to the given colors and weights.
///
/// If all weights are the same, the endpoints will be set to the mean of all colors.
///
/// https://fgiesen.wordpress.com/2024/08/29/when-is-a-bcn-astc-endpoints-from-indices-solve-singular/
pub(crate) fn least_squares_weights<
    R: Copy + Default + std::ops::Mul<f32, Output = R> + std::ops::AddAssign,
    C: Copy + Into<R>,
>(
    colors: &[C],
    weights: &[f32],
) -> (R, R) {
    assert_eq!(weights.len(), colors.len());

    // Let A be a n-by-2 matrix where each row is [w_i, 1 - w_i].
    // First, compute D = A^T*A = (a b)
    //                            (b c)
    let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
    for &w in weights {
        let w_inv = 1.0 - w;
        a += w * w;
        b += w * w_inv;
        c += w_inv * w_inv;
    }

    // Second, find E = D^-1
    let LeastSquaresWeightMatrix {
        e01,
        e11,
        e00_01,
        e10_11,
    } = LeastSquaresWeightMatrix::from_d(a, b, c)
        .unwrap_or(LeastSquaresWeightMatrix::mean(weights.len()));

    // Let B be an n-by-3 matrix where each row is the color vector.
    // Let X be the 2-by-3 matrix of the two endpoints we want to find.
    // Third, compute X = (E * A^T) * B
    let (mut x0, mut x1) = (R::default(), R::default());
    for (&color, &w) in colors.iter().zip(weights) {
        let color: R = color.into();
        // Let G = E * A^T be a 2-by-n matrix where each column is:
        //   ( g_0i ) = ( e00 * w_i + e01 * (1 - w_i) ) = ( e01 + (e00 - e01) * w )
        //   ( g_1i ) = ( e01 * w_i + e11 * (1 - w_i) ) = ( e11 + (e01 - e11) * w )
        // TODO: This can be a single FMA operation
        let g0 = e01 + (e00_01) * w;
        let g1 = e11 + (e10_11) * w;

        x0 += color * g0;
        x1 += color * g1;
    }

    (x0, x1)
}

/// Least squares fits 2 endpoints to the given colors and weights.
///
/// This version is optimized for vectorized 4x4 f32 blocks.
pub(crate) fn least_squares_weights_f32_vec4(
    colors: &[Vec4; 4],
    weights: &[Vec4; 4],
) -> (f32, f32) {
    let [w0, w1, w2, w3] = *weights;

    // Let A be a n-by-2 matrix where each row is [w_i, 1 - w_i].
    // First, compute D = A^T*A = (a b)
    //                            (b c)
    let [w0_, w1_, w2_, w3_] = [1.0 - w0, 1.0 - w1, 1.0 - w2, 1.0 - w3];
    let a = w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3;
    let b = w0 * w0_ + w1 * w1_ + w2 * w2_ + w3 * w3_;
    let c = w0_ * w0_ + w1_ * w1_ + w2_ * w2_ + w3_ * w3_;
    let a = (a.x + a.y) + (a.z + a.w);
    let b = (b.x + b.y) + (b.z + b.w);
    let c = (c.x + c.y) + (c.z + c.w);

    // Second, find E = D^-1
    let LeastSquaresWeightMatrix {
        e01,
        e11,
        e00_01,
        e10_11,
    } = LeastSquaresWeightMatrix::from_d(a, b, c).unwrap_or(LeastSquaresWeightMatrix::mean(16));

    // Let B be an n-by-1 matrix where each row is the color vector.
    // Let X be the 2-by-1 matrix of the two endpoints we want to find.
    // Third, compute X = (E * A^T) * B
    // Let G = E * A^T be a 2-by-n matrix where each column is:
    //   ( g_0i ) = ( e00 * w_i + e01 * (1 - w_i) ) = ( e01 + (e00 - e01) * w )
    //   ( g_1i ) = ( e01 * w_i + e11 * (1 - w_i) ) = ( e11 + (e01 - e11) * w )
    // TODO: This can be a single FMA operation
    let [c0, c1, c2, c3] = *colors;
    let x0 = (c0 * (e01 + e00_01 * w0))
        + (c1 * (e01 + e00_01 * w1))
        + (c2 * (e01 + e00_01 * w2))
        + (c3 * (e01 + e00_01 * w3));
    let x1 = (c0 * (e11 + e10_11 * w0))
        + (c1 * (e11 + e10_11 * w1))
        + (c2 * (e11 + e10_11 * w2))
        + (c3 * (e11 + e10_11 * w3));
    let x0 = (x0.x + x0.y) + (x0.z + x0.w);
    let x1 = (x1.x + x1.y) + (x1.z + x1.w);

    (x0, x1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that least_square_weights supports the case where all weights are the same.
    #[test]
    fn test_least_square_weights_all_the_same() {
        let colors: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let weights: &[f32] = &[0.5, 0.5, 0.5, 0.5];
        let (min, max): (f32, f32) = least_squares_weights(colors, weights);
        assert!((min - max).abs() < 1e-6);
        assert!((max - 2.5).abs() < 1e-6);
    }
}
