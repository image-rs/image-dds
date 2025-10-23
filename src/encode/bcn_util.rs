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

/// Fits a line through the given colors and returns two endpoints along the
/// line.
pub(crate) fn line3_fit_endpoints<C: Copy + Into<Vec3A>>(
    colors: &[C],
    nudge_factor: f32,
) -> (Vec3A, Vec3A) {
    debug_assert!(!colors.is_empty());

    // find the best line through the colors
    let line = ColorLine3::new(colors);

    // sort all colors along the line and find the min/max projection
    let mut min_t = f32::INFINITY;
    let mut max_t = f32::NEG_INFINITY;
    for &color in colors.iter() {
        let color: Vec3A = color.into();
        let t = line.project(color);
        min_t = min_t.min(t);
        max_t = max_t.max(t);
    }

    // Instead of using min_t and max_t directly, it's better to slightly nudge
    // them towards the midpoint. This prevent extreme endpoints and makes the
    // refinement converge faster.
    let mid_t = (min_t + max_t) * 0.5;
    min_t = mid_t + (min_t - mid_t) * nudge_factor;
    max_t = mid_t + (max_t - mid_t) * nudge_factor;

    // select initial points along the line
    (line.at(min_t), line.at(max_t))
}
/// Fits a line through the given colors and returns two endpoints along the
/// line.
pub(crate) fn line4_fit_endpoints<C: Copy + Into<Vec4>>(
    colors: &[C],
    nudge_factor: f32,
) -> (Vec4, Vec4) {
    debug_assert!(!colors.is_empty());

    // find the best line through the colors
    let line = ColorLine4::new(colors);

    // sort all colors along the line and find the min/max projection
    let mut min_t = f32::INFINITY;
    let mut max_t = f32::NEG_INFINITY;
    for &color in colors.iter() {
        let color: Vec4 = color.into();
        let t = line.project(color);
        min_t = min_t.min(t);
        max_t = max_t.max(t);
    }

    // Instead of using min_t and max_t directly, it's better to slightly nudge
    // them towards the midpoint. This prevent extreme endpoints and makes the
    // refinement converge faster.
    let mid_t = (min_t + max_t) * 0.5;
    min_t = mid_t + (min_t - mid_t) * nudge_factor;
    max_t = mid_t + (max_t - mid_t) * nudge_factor;

    // select initial points along the line
    (line.at(min_t), line.at(max_t))
}
pub(crate) struct ColorLine3 {
    /// The centroid of the colors
    centroid: Vec3A,
    /// The normalized direction of the line
    d: Vec3A,
}
impl ColorLine3 {
    pub fn new<C: Copy + Into<Vec3A>>(colors: &[C]) -> Self {
        fn mean<C: Copy + Into<Vec3A>>(colors: &[C]) -> Vec3A {
            let mut mean = Vec3A::ZERO;
            for &color in colors {
                let color: Vec3A = color.into();
                mean += color;
            }
            mean * (1. / colors.len() as f32)
        }
        fn covariance_matrix<C: Copy + Into<Vec3A>>(colors: &[C], centroid: Vec3A) -> [Vec3A; 3] {
            let mut cov = [Vec3A::ZERO; 3];

            for &p in colors {
                let p: Vec3A = p.into();
                let d = p - centroid;
                cov[0] += d * d.x;
                cov[1] += d * d.y;
                cov[2] += d * d.z;
            }

            let n_r = 1.0 / colors.len() as f32;
            cov[0] *= n_r;
            cov[1] *= n_r;
            cov[2] *= n_r;

            cov
        }
        fn largest_eigenvector(matrix: [Vec3A; 3]) -> Vec3A {
            // A simple power iteration method to approximate the dominant eigenvector
            let mut v = Vec3A::ONE;
            for _ in 0..2 {
                let r = matrix[0].dot(v);
                let g = matrix[1].dot(v);
                let b = matrix[2].dot(v);
                v = Vec3A::new(r, g, b).normalize_or_zero();
            }
            v
        }

        debug_assert!(!colors.is_empty());

        let centroid = mean(colors);
        let covariance = covariance_matrix(colors, centroid);
        let eigenvector = largest_eigenvector(covariance);

        Self {
            centroid,
            d: eigenvector,
        }
    }

    /// Returns the point along the line at parameter `t`.
    pub fn at(&self, t: f32) -> Vec3A {
        self.centroid + self.d * t
    }
    /// Projects the points onto the line and returns the parameter `t`.
    pub fn project(&self, color: Vec3A) -> f32 {
        let diff = color - self.centroid;
        diff.dot(self.d)
    }
    /// Returns the squared distance from the color to the line.
    pub fn dist_sq(&self, color: Vec3A) -> f32 {
        let diff = color - self.centroid;
        let t = self.d.dot(diff);
        (diff - self.d * t).length_squared()
    }
    /// Returns the sum of squared distance from the colors to the line.
    pub fn sum_dist_sq(&self, colors: &[Vec3A]) -> f32 {
        let mut sum = Vec3A::ZERO;
        for &color in colors {
            let diff = color - self.centroid;
            let t = self.d.dot(diff);
            let dist = diff - self.d * t;
            sum += dist * dist;
        }
        sum.x + sum.y + sum.z
    }
}
pub(crate) struct ColorLine4 {
    /// The centroid of the colors
    centroid: Vec4,
    /// The normalized direction of the line
    d: Vec4,
}
impl ColorLine4 {
    pub fn new<C: Copy + Into<Vec4>>(colors: &[C]) -> Self {
        fn mean<C: Copy + Into<Vec4>>(colors: &[C]) -> Vec4 {
            let mut mean = Vec4::ZERO;
            for &color in colors {
                let color: Vec4 = color.into();
                mean += color;
            }
            mean * (1. / colors.len() as f32)
        }
        fn covariance_matrix<C: Copy + Into<Vec4>>(colors: &[C], centroid: Vec4) -> [Vec4; 4] {
            let mut cov = [Vec4::ZERO; 4];

            for &p in colors {
                let p: Vec4 = p.into();
                let d = p - centroid;
                cov[0] += d * d.x;
                cov[1] += d * d.y;
                cov[2] += d * d.z;
                cov[3] += d * d.w;
            }

            let n_r = 1.0 / colors.len() as f32;
            cov[0] *= n_r;
            cov[1] *= n_r;
            cov[2] *= n_r;
            cov[3] *= n_r;

            cov
        }
        fn largest_eigenvector(matrix: [Vec4; 4]) -> Vec4 {
            // A simple power iteration method to approximate the dominant eigenvector
            let mut v = Vec4::ONE;
            for _ in 0..2 {
                let r = matrix[0].dot(v);
                let g = matrix[1].dot(v);
                let b = matrix[2].dot(v);
                let a = matrix[3].dot(v);
                v = Vec4::new(r, g, b, a).normalize_or_zero();
            }
            v
        }

        debug_assert!(!colors.is_empty());

        let centroid = mean(colors);
        let covariance = covariance_matrix(colors, centroid);
        let eigenvector = largest_eigenvector(covariance);

        Self {
            centroid,
            d: eigenvector,
        }
    }

    /// Returns the point along the line at parameter `t`.
    pub fn at(&self, t: f32) -> Vec4 {
        self.centroid + self.d * t
    }
    /// Projects the points onto the line and returns the parameter `t`.
    pub fn project(&self, color: Vec4) -> f32 {
        let diff = color - self.centroid;
        diff.dot(self.d)
    }
    /// Returns the squared distance from the color to the line.
    pub fn dist_sq(&self, color: Vec4) -> f32 {
        let diff = color - self.centroid;
        let t = self.d.dot(diff);
        (diff - self.d * t).length_squared()
    }
    /// Returns the sum of squared distance from the colors to the line.
    pub fn sum_dist_sq(&self, colors: &[Vec4]) -> f32 {
        let mut sum = Vec4::ZERO;
        for &color in colors {
            let diff = color - self.centroid;
            let t = self.d.dot(diff);
            let dist = diff - self.d * t;
            sum += dist * dist;
        }
        (sum.x + sum.y) + (sum.z + sum.w)
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

pub(crate) trait Quantized: Copy + Sized + WithChannels<E = u8> {
    /// The unquantized f32 vector type.
    type V: VectorType + WithChannels<E = f32>;

    fn round(v: Self::V) -> Self;
    fn floor(v: Self::V) -> Self;
    fn ceil(v: Self::V) -> Self;
    fn to_vec(self) -> Self::V;
}
pub(crate) trait VectorType: Copy + Sized + std::ops::Sub<Output = Self> {}
impl VectorType for f32 {}
impl VectorType for glam::Vec3A {}
impl VectorType for glam::Vec4 {}
pub(crate) trait WithChannels {
    type E;
    const CHANNELS: usize;
    fn get(&self, channel: usize) -> Self::E;
    fn set(&mut self, channel: usize, value: Self::E);
}
impl WithChannels for f32 {
    type E = f32;
    const CHANNELS: usize = 1;

    #[inline(always)]
    fn get(&self, channel: usize) -> Self::E {
        debug_assert!(channel == 0);
        *self
    }
    #[inline(always)]
    fn set(&mut self, channel: usize, value: Self::E) {
        debug_assert!(channel == 0);
        *self = value;
    }
}
impl WithChannels for glam::Vec3A {
    type E = f32;
    const CHANNELS: usize = 3;

    #[inline(always)]
    fn get(&self, channel: usize) -> Self::E {
        self[channel]
    }
    #[inline(always)]
    fn set(&mut self, channel: usize, value: Self::E) {
        self[channel] = value;
    }
}
impl WithChannels for glam::Vec4 {
    type E = f32;
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn get(&self, channel: usize) -> Self::E {
        self[channel]
    }
    #[inline(always)]
    fn set(&mut self, channel: usize, value: Self::E) {
        self[channel] = value;
    }
}

/// BCn encoding is a discrete optimization problem. However, we treat it as a
/// continuous problem and then quantize the results to get the final discrete
/// endpoints.
///
/// This enum determines how the quantization is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quantization {
    /// Continuous endpoints are rounded to the nearest discrete color.
    ///
    /// This is the fastest possible quantization, but also the one with the
    /// worst quality.
    Round,
    /// Each channel is optimized independently. Optimization is performed
    /// using all 4 possible combinations of floor/ceil for each endpoint (per
    /// channel).
    ///
    /// This option is rather slow, but provides good quality.
    ChannelWise,
    /// This is a mix between `Round` and `ChannelWise`.
    ///
    /// This is based on 3 ideas:
    ///
    /// 1. The main performance cost comes from calling the error metric, so
    ///    we want to minimize the number of calls to it.
    /// 2. If the min and max (floor and ceil) of one endpoint is the same,
    ///    then the number of calls to the error metric shrinks from 3 to 1.
    /// 3. If the non-quantized endpoint is very close to the quantized min or
    ///    max, then it is likely that the optimal quantized value is the min or
    ///    max. So we can skip the other value.
    ///    E.g. if the non-quantized value of a channel is 0.99, then the
    ///    optimal quantized value for that channel is likely 1 and not 0.
    ///
    /// Using these ideas, a threshold can be defined for when a non-quantized
    /// value is considered "close enough" to the min or max to skip checking
    /// the other. This threshold can also be thought of a radius around the min
    /// and max. As such, a threshold of 0 will have the same behavior as
    /// `ChannelWise`, while a threshold of 0.5 will have the same behavior as
    /// `Round`.
    ///
    /// Using this threshold, we can smoothly trade-off between quality and
    /// performance.
    ///
    /// Right now, the threshold is hardcoded to 0.25.
    ChannelWiseOptimized,
}
impl Quantization {
    /// Rounds to the nearest R5G6B5 colors.
    pub fn round<Q: Quantized>(c0: Q::V, c1: Q::V) -> (Q, Q) {
        (Q::round(c0), Q::round(c1))
    }
    /// Uses floor/ceil for each channel depending to maximize the range. The
    /// returned colors will me maximally apart within the constraints of rounding.
    pub fn wide<Q: Quantized>(c0: Q::V, c1: Q::V) -> (Q, Q) {
        let c0_floor = Q::floor(c0);
        let c0_ceil = Q::ceil(c0);
        let c1_floor = Q::floor(c1);
        let c1_ceil = Q::ceil(c1);

        // start by assuming that c0 < c1
        let mut q0 = c0_floor;
        let mut q1 = c1_ceil;
        for c in 0..Q::V::CHANNELS {
            if c0.get(c) > c1.get(c) {
                q0.set(c, c0_ceil.get(c));
                q1.set(c, c1_floor.get(c));
            }
        }

        (q0, q1)
    }

    pub fn pick_best<Q: Quantized, E: PartialOrd>(
        self,
        c0: Q::V,
        c1: Q::V,
        mut error_metric: impl FnMut(Q, Q) -> E,
    ) -> (Q, Q) {
        let mut best = Self::round(c0, c1);

        if self == Quantization::Round {
            // For simple rounding, we don't need to optimize at all
            return best;
        }

        let mut best_error = error_metric(best.0, best.1);

        let get_range = match self {
            Quantization::ChannelWiseOptimized => Self::optimized_range::<Q>,
            _ => Self::full_range::<Q>,
        };

        let (c0_min, c0_max) = get_range(c0);
        let (c1_min, c1_max) = get_range(c1);

        // Channel-wise optimization
        for c in 0..Q::CHANNELS {
            let skip0 = best.0.get(c);
            let skip1 = best.1.get(c);
            for channel0 in c0_min.get(c)..=c0_max.get(c) {
                for channel1 in c1_min.get(c)..=c1_max.get(c) {
                    if channel0 == skip0 && channel1 == skip1 {
                        continue;
                    }
                    let (mut c0, mut c1) = best;

                    c0.set(c, channel0);
                    c1.set(c, channel1);
                    let error = error_metric(c0, c1);
                    if error < best_error {
                        best = (c0, c1);
                        best_error = error;
                    }
                }
            }
        }
        best
    }

    /// Returns the floor and ceil of the given color.
    fn full_range<Q: Quantized>(c: Q::V) -> (Q, Q) {
        (Q::floor(c), Q::ceil(c))
    }

    const CULL_THRESHOLD: f32 = 0.25;
    /// Returns the floor and ceil of the given color. But if the color value
    /// is very close to the floor or ceil, then it will only return one of the
    /// two (returning the same value floor and ceil).
    ///
    /// The threshold for "very close" is defined by `CULL_THRESHOLD`. A
    /// threshold of 0 will behave the same as `full_range`, while a threshold
    /// of 0.5 will behave the same as `Quantization::Round`.
    fn optimized_range<Q: Quantized>(c: Q::V) -> (Q, Q) {
        let mut floor = Q::floor(c);
        let mut ceil = Q::ceil(c);

        let v_floor = floor.to_vec();
        let v_ceil = ceil.to_vec();
        let floor_dist = c - v_floor;
        let dist = v_ceil - v_floor;

        const FLOOR_THRESHOLD: f32 = Quantization::CULL_THRESHOLD;
        const CEIL_THRESHOLD: f32 = 1.0 - Quantization::CULL_THRESHOLD;

        for c in 0..Q::V::CHANNELS {
            let floor_dist: f32 = floor_dist.get(c);
            let dist: f32 = dist.get(c);
            if floor_dist < FLOOR_THRESHOLD * dist {
                // close to floor, so we can skip ceil
                ceil.set(c, floor.get(c));
            } else if floor_dist > CEIL_THRESHOLD * dist {
                // close to ceil, so we can skip floor
                floor.set(c, ceil.get(c));
            }
        }

        (floor, ceil)
    }
}
