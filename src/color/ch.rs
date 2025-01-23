use super::Norm;

pub fn alpha_to_rgba<Precision: Norm>(pixel: [Precision; 1]) -> [Precision; 4] {
    [Norm::ZERO, Norm::ZERO, Norm::ZERO, pixel[0]]
}
pub fn grayscale_to_rgb<Precision: Norm>(pixel: [Precision; 1]) -> [Precision; 3] {
    [pixel[0], pixel[0], pixel[0]]
}
pub fn grayscale_to_rgba<Precision: Norm>(pixel: [Precision; 1]) -> [Precision; 4] {
    [pixel[0], pixel[0], pixel[0], Norm::ONE]
}
pub fn rgb_to_grayscale<Precision: Norm>(pixel: [Precision; 3]) -> [Precision; 1] {
    [pixel[0]]
}
pub fn rgb_to_rgba<Precision: Norm>(pixel: [Precision; 3]) -> [Precision; 4] {
    [pixel[0], pixel[1], pixel[2], Norm::ONE]
}
pub fn rgba_to_grayscale<Precision: Norm>(pixel: [Precision; 4]) -> [Precision; 1] {
    [pixel[0]]
}
pub fn rgba_to_alpha<Precision: Norm>(pixel: [Precision; 4]) -> [Precision; 1] {
    [pixel[3]]
}
pub fn rgba_to_rgb<Precision: Norm>(pixel: [Precision; 4]) -> [Precision; 3] {
    [pixel[0], pixel[1], pixel[2]]
}
