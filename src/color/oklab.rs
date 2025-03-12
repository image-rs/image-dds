use glam::Vec3A;

trait Operations {
    fn srgb_to_linear(c: f32) -> f32;
    fn linear_to_srgb(c: f32) -> f32;
    fn cbrt(x: f32) -> f32;
}

struct Reference;
impl Operations for Reference {
    fn srgb_to_linear(c: f32) -> f32 {
        if c >= 0.04045 {
            ((c + 0.055) / 1.055).powf(2.4)
        } else {
            c / 12.92
        }
    }
    fn linear_to_srgb(c: f32) -> f32 {
        if c > 0.0031308 {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * c
        }
    }
    fn cbrt(x: f32) -> f32 {
        f32::cbrt(x)
    }
}

struct Fast;
impl Operations for Fast {
    fn srgb_to_linear(c: f32) -> f32 {
        if c >= 0.04045 {
            // This uses a Padé approximant for ((c + 0.055) / 1.055) ^ 2.4:
            // (0.000857709 +0.0359438 x+0.524293 x^2+1.31193 x^3)/(1+0.992498 x-0.119725 x^2)
            let c2 = c * c;
            let c3 = c2 * c;
            f32::min(
                1.0,
                (0.000857709 + 0.0359438 * c + 0.524293 * c2 + 1.31193 * c3)
                    / (1.0 + 0.992498 * c - 0.119725 * c2),
            )
        } else {
            c * (1.0 / 12.92)
        }
    }
    fn linear_to_srgb(c: f32) -> f32 {
        if c > 0.0031308 {
            // This uses a Padé approximant for 1.055 c^(1/2.4) - 0.055:
            // (-0.0117264+21.0897 x+949.46 x^2+2225.62 x^3)/(1+176.398 x+1983.15 x^2+1035.65 x^3)
            let c2 = c * c;
            let c3 = c2 * c;
            (-0.0117264 + 21.0897 * c + 949.46 * c2 + 2225.62 * c3)
                / (1.0 + 176.398 * c + 1983.15 * c2 + 1035.65 * c3)
        } else {
            12.92 * c
        }
    }
    #[allow(clippy::excessive_precision)]
    fn cbrt(x: f32) -> f32 {
        // This is the fast cbrt approximation from the oklab crate.
        // Source: https://gitlab.com/kornelski/oklab/-/blob/d3c074f154187dd5c0642119a6402a6c0753d70c/oklab/src/lib.rs#L61
        // Author: Kornel (https://gitlab.com/kornelski/)
        const B: u32 = 709957561;
        const C: f32 = 5.4285717010e-1;
        const D: f32 = -7.0530611277e-1;
        const E: f32 = 1.4142856598e+0;
        const F: f32 = 1.6071428061e+0;
        const G: f32 = 3.5714286566e-1;

        let mut t = f32::from_bits((x.to_bits() / 3).wrapping_add(B));
        let s = C + (t * t) * (t / x);
        t *= G + F / (s + E + D / s);
        t
    }
}

#[allow(clippy::excessive_precision)]
fn srgb_to_oklab_impl<O: Operations>(rgb: Vec3A) -> Vec3A {
    let [r, g, b] = rgb.to_array().map(O::srgb_to_linear);

    let mut l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let mut m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let mut s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    l = O::cbrt(l);
    m = O::cbrt(m);
    s = O::cbrt(s);

    let l_final = l * 0.2104542553 + m * 0.7936177850 + s * -0.0040720468;
    let a = l * 1.9779984951 + m * -2.4285922050 + s * 0.4505937099;
    let b = l * 0.0259040371 + m * 0.7827717662 + s * -0.8086757660;

    // normalize everything to the 0..1 range
    Vec3A::new(l_final, a + 0.5, b + 0.5)
}
#[allow(clippy::excessive_precision)]
fn oklab_to_srgb_impl<O: Operations>(lab: Vec3A) -> Vec3A {
    let l_org = lab.x;
    let a = lab.y - 0.5;
    let b = lab.z - 0.5;

    let mut l = l_org + a * 0.3963377774 + b * 0.2158037573;
    let mut m = l_org + a * -0.1055613458 + b * -0.0638541728;
    let mut s = l_org + a * -0.0894841775 + b * -1.2914855480;

    l = l * l * l;
    m = m * m * m;
    s = s * s * s;

    let r = l * 4.0767416621 + m * -3.3077115913 + s * 0.2309699292;
    let g = l * -1.2684380046 + m * 2.6097574011 + s * -0.3413193965;
    let b = l * -0.0041960863 + m * -0.7034186147 + s * 1.7076147010;

    Vec3A::new(
        O::linear_to_srgb(r),
        O::linear_to_srgb(g),
        O::linear_to_srgb(b),
    )
}

#[allow(unused)]
pub(crate) fn srgb_to_oklab(rgb: Vec3A) -> Vec3A {
    srgb_to_oklab_impl::<Reference>(rgb)
}
#[allow(unused)]
pub(crate) fn oklab_to_srgb(lab: Vec3A) -> Vec3A {
    oklab_to_srgb_impl::<Reference>(lab)
}
pub(crate) fn fast_srgb_to_oklab(rgb: Vec3A) -> Vec3A {
    srgb_to_oklab_impl::<Fast>(rgb)
}
pub(crate) fn fast_oklab_to_srgb(lab: Vec3A) -> Vec3A {
    oklab_to_srgb_impl::<Fast>(lab)
}

// tests for OKLab
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oklab_srgb() {
        let max_i = 20;
        let max_f = max_i as f32;
        for r in 0..=max_i {
            for g in 0..=max_i {
                for b in 0..=max_i {
                    let color = Vec3A::new(r as f32 / max_f, g as f32 / max_f, b as f32 / max_f);

                    let oklab = srgb_to_oklab(color);
                    let srgb = oklab_to_srgb(oklab);

                    assert!(
                        (color - srgb).abs().max_element() < 1e-4,
                        "{:?} -> {:?}",
                        color,
                        srgb
                    );

                    assert!(oklab.max_element() <= 1.0, "{:?} -> {:?}", color, oklab);
                    assert!(oklab.min_element() >= 0.0, "{:?} -> {:?}", color, oklab);
                }
            }
        }
    }

    #[test]
    fn test_fast_oklab_srgb() {
        let max_i = 20;
        let max_f = max_i as f32;
        for r in 0..=max_i {
            for g in 0..=max_i {
                for b in 0..=max_i {
                    let color = Vec3A::new(r as f32 / max_f, g as f32 / max_f, b as f32 / max_f);

                    let fast_oklab = fast_srgb_to_oklab(color);
                    let ref_oklab = srgb_to_oklab(color);

                    assert!(
                        (fast_oklab - ref_oklab).abs().max_element() < 1e-3,
                        "{:?} -> fast: {:?} vs ref: {:?}",
                        color,
                        fast_oklab,
                        ref_oklab
                    );

                    let srgb = fast_oklab_to_srgb(fast_oklab);

                    assert!(
                        (color - srgb).abs().max_element() < 2.5e-3,
                        "{:?} -> {:?}",
                        color,
                        srgb
                    );

                    assert!(
                        fast_oklab.max_element() <= 1.0,
                        "{:?} -> {:?}",
                        color,
                        fast_oklab
                    );
                    assert!(
                        fast_oklab.min_element() >= 0.0,
                        "{:?} -> {:?}",
                        color,
                        fast_oklab
                    );
                }
            }
        }
    }

    #[test]
    fn test_linear_srgb() {
        for c in 0..=255 {
            let c = c as f32 / 255.0;
            let l = Reference::srgb_to_linear(c);
            let c2 = Reference::linear_to_srgb(l);

            assert!((c - c2).abs() < 1e-6, "{} -> {}", c, c2);
        }

        for c in 0..=255 {
            let c = c as f32 / 255.0;
            let l = Fast::srgb_to_linear(c);
            let c2 = Fast::linear_to_srgb(l);

            assert!((c - c2).abs() < 2.5e-3, "{} -> {}", c, c2);
            assert!((0.0..=1.0).contains(&l), "{} -> {}", c, l);
            assert!((0.0..=1.0).contains(&c2), "{} -> {}", c, l);
        }

        assert_eq!(Reference::srgb_to_linear(0.0), 0.0);
        assert!((Reference::srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert_eq!(Fast::linear_to_srgb(0.0), 0.0);
        assert!((Fast::srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_error_fast_srgb_to_linear() {
        assert_eq!(
            get_error_stats(Reference::srgb_to_linear, Fast::srgb_to_linear),
            "Error: avg=0.00002514 max=0.00013047 for 0.999"
        );
    }
    #[test]
    fn test_error_fast_linear_to_srgb() {
        assert_eq!(
            get_error_stats(Reference::linear_to_srgb, Fast::linear_to_srgb),
            "Error: avg=0.00105457 max=0.00236702 for 0.732"
        );
    }

    fn get_error_stats(f1: impl Fn(f32) -> f32, f2: impl Fn(f32) -> f32) -> String {
        let count = 1000;

        let mut avg: f64 = 0.0;
        let mut max: f64 = 0.0;
        let mut max_input = 0.0;
        for c in 0..=count {
            let c = c as f32 / count as f32;
            let expected = f1(c) as f64;
            let actual = f2(c) as f64;

            let diff = (expected - actual).abs();
            avg += diff;
            if diff > max {
                max_input = c;
                max = diff;
            }
        }
        avg /= (count + 1) as f64;

        format!("Error: avg={avg:.8} max={max:.8} for {max_input:.3}")
    }
}
