use glam::Vec3A;

trait Operations {
    fn srgb_to_linear(c: f32) -> f32;
    fn linear_to_srgb(c: f32) -> f32;
    fn cbrt(x: f32) -> f32;

    fn srgb_to_linear_vec(x: Vec3A) -> Vec3A {
        Vec3A::new(
            Self::srgb_to_linear(x.x),
            Self::srgb_to_linear(x.y),
            Self::srgb_to_linear(x.z),
        )
    }
    fn linear_to_srgb_vec(x: Vec3A) -> Vec3A {
        Vec3A::new(
            Self::linear_to_srgb(x.x),
            Self::linear_to_srgb(x.y),
            Self::linear_to_srgb(x.z),
        )
    }
    fn cbrt_vec(x: Vec3A) -> Vec3A {
        Vec3A::new(Self::cbrt(x.x), Self::cbrt(x.y), Self::cbrt(x.z))
    }
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

    fn srgb_to_linear_vec(c: Vec3A) -> Vec3A {
        Vec3A::select(
            c.cmpge(Vec3A::splat(0.04045)),
            {
                // This uses a Padé approximant for ((c + 0.055) / 1.055) ^ 2.4:
                // (0.000857709 +0.0359438 x+0.524293 x^2+1.31193 x^3)/(1+0.992498 x-0.119725 x^2)
                let c2 = c * c;
                let c3 = c2 * c;
                Vec3A::min(
                    Vec3A::ONE,
                    (0.000857709 + 0.0359438 * c + 0.524293 * c2 + 1.31193 * c3)
                        / (Vec3A::ONE + 0.992498 * c - 0.119725 * c2),
                )
            },
            c * (1.0 / 12.92),
        )
    }
    fn linear_to_srgb_vec(c: Vec3A) -> Vec3A {
        Vec3A::select(
            c.cmpgt(Vec3A::splat(0.0031308)),
            {
                // This uses a Padé approximant for 1.055 c^(1/2.4) - 0.055:
                // (-0.0117264+21.0897 x+949.46 x^2+2225.62 x^3)/(1+176.398 x+1983.15 x^2+1035.65 x^3)
                let c2 = c * c;
                let c3 = c2 * c;
                (-0.0117264 + 21.0897 * c + 949.46 * c2 + 2225.62 * c3)
                    / (1.0 + 176.398 * c + 1983.15 * c2 + 1035.65 * c3)
            },
            c * 12.92,
        )
    }
    #[allow(clippy::excessive_precision)]
    fn cbrt_vec(x: Vec3A) -> Vec3A {
        // This is the fast cbrt approximation from the oklab crate.
        // Source: https://gitlab.com/kornelski/oklab/-/blob/d3c074f154187dd5c0642119a6402a6c0753d70c/oklab/src/lib.rs#L61
        // Author: Kornel (https://gitlab.com/kornelski/)
        const B: u32 = 709957561;
        const C: f32 = 5.4285717010e-1;
        const D: f32 = -7.0530611277e-1;
        const E: f32 = 1.4142856598e+0;
        const F: f32 = 1.6071428061e+0;
        const G: f32 = 3.5714286566e-1;

        let mut t = Vec3A::from_array(
            x.to_array()
                .map(|x| f32::from_bits((x.to_bits() / 3).wrapping_add(B))),
        );
        let s = C + (t * t) * (t / x);
        t *= G + F / (s + E + D / s);
        t
    }
}

#[allow(clippy::excessive_precision)]
fn srgb_to_oklab_impl<O: Operations>(srgb: Vec3A) -> Vec3A {
    let rgb = O::srgb_to_linear_vec(srgb);

    let lms = Vec3A::new(
        rgb.dot(Vec3A::new(0.4122214708, 0.5363325363, 0.0514459929)),
        rgb.dot(Vec3A::new(0.2119034982, 0.6806995451, 0.1073969566)),
        rgb.dot(Vec3A::new(0.0883024619, 0.2817188376, 0.6299787005)),
    );
    let lms = O::cbrt_vec(lms);

    let lab = Vec3A::new(
        lms.dot(Vec3A::new(0.2104542553, 0.7936177850, -0.0040720468)),
        lms.dot(Vec3A::new(1.9779984951, -2.4285922050, 0.4505937099)),
        lms.dot(Vec3A::new(0.0259040371, 0.7827717662, -0.8086757660)),
    );

    // normalize everything to the 0..1 range
    lab + Vec3A::new(0.0, 0.5, 0.5)
}
#[allow(clippy::excessive_precision)]
fn oklab_to_srgb_impl<O: Operations>(lab: Vec3A) -> Vec3A {
    let lab_norm = lab - Vec3A::new(0.0, 0.5, 0.5);
    let lms = Vec3A::new(
        lab_norm.dot(Vec3A::new(1.0, 0.3963377774, 0.2158037573)),
        lab_norm.dot(Vec3A::new(1.0, -0.1055613458, -0.0638541728)),
        lab_norm.dot(Vec3A::new(1.0, -0.0894841775, -1.2914855480)),
    );
    let lms = lms * lms * lms; // lms^3
    let rgb = Vec3A::new(
        lms.dot(Vec3A::new(4.0767416621, -3.3077115913, 0.2309699292)),
        lms.dot(Vec3A::new(-1.2684380046, 2.6097574011, -0.3413193965)),
        lms.dot(Vec3A::new(-0.0041960863, -0.7034186147, 1.7076147010)),
    );

    O::linear_to_srgb_vec(rgb)
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
                        "{color:?} -> {srgb:?}"
                    );

                    assert!(oklab.max_element() <= 1.0, "{color:?} -> {oklab:?}");
                    assert!(oklab.min_element() >= 0.0, "{color:?} -> {oklab:?}");
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
                        "{color:?} -> fast: {fast_oklab:?} vs ref: {ref_oklab:?}"
                    );

                    let srgb = fast_oklab_to_srgb(fast_oklab);

                    assert!(
                        (color - srgb).abs().max_element() < 2.5e-3,
                        "{color:?} -> {srgb:?}"
                    );

                    assert!(
                        fast_oklab.max_element() <= 1.0,
                        "{color:?} -> {fast_oklab:?}"
                    );
                    assert!(
                        fast_oklab.min_element() >= 0.0,
                        "{color:?} -> {fast_oklab:?}"
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

            assert!((c - c2).abs() < 1e-6, "{c} -> {c2}");
        }

        for c in 0..=255 {
            let c = c as f32 / 255.0;
            let l = Fast::srgb_to_linear(c);
            let c2 = Fast::linear_to_srgb(l);

            assert!((c - c2).abs() < 2.5e-3, "{c} -> {c2}");
            assert!((0.0..=1.0).contains(&l), "{c} -> {l}");
            assert!((0.0..=1.0).contains(&c2), "{c} -> {l}");
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
