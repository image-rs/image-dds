use glam::Vec3A;

trait Operations {
    fn srgb_to_linear(c: Vec3A) -> Vec3A;
    fn linear_to_srgb(c: Vec3A) -> Vec3A;
    fn cbrt(x: Vec3A) -> Vec3A;
}

struct Reference;
impl Operations for Reference {
    fn srgb_to_linear(c: Vec3A) -> Vec3A {
        fn srgb_to_linear(c: f32) -> f32 {
            if c >= 0.04045 {
                ((c + 0.055) / 1.055).powf(2.4)
            } else {
                c / 12.92
            }
        }

        Vec3A::new(
            srgb_to_linear(c.x),
            srgb_to_linear(c.y),
            srgb_to_linear(c.z),
        )
    }
    fn linear_to_srgb(c: Vec3A) -> Vec3A {
        fn linear_to_srgb(c: f32) -> f32 {
            if c > 0.0031308 {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * c
            }
        }

        Vec3A::new(
            linear_to_srgb(c.x),
            linear_to_srgb(c.y),
            linear_to_srgb(c.z),
        )
    }
    fn cbrt(x: Vec3A) -> Vec3A {
        Vec3A::new(x.x.cbrt(), x.y.cbrt(), x.z.cbrt())
    }
}

/// A fast fused multiply-add operation that uses hardware FMA if available.
/// If hardware FMA is not available, it falls back to a regular multiply-add.
#[inline(always)]
fn fma(a: Vec3A, b: Vec3A, c: Vec3A) -> Vec3A {
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    ))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    )))]
    {
        a * b + c
    }
}

struct Fast;
#[allow(clippy::excessive_precision)]
impl Operations for Fast {
    fn srgb_to_linear(c: Vec3A) -> Vec3A {
        Vec3A::select(
            c.cmpge(Vec3A::splat(0.04045)),
            {
                // Polynomial approximation for ((c + 0.055) / 1.055) ^ 2.4
                // This has a max error of 0.0001228 and is exact at c=0.04045 and c=1
                const A0: f32 = 0.00117465;
                const A1: f32 = 0.02381997;
                const A2: f32 = 0.58750746;
                const A3: f32 = 0.47736490;
                const A4: f32 = -0.08986699;
                let c2 = c * c;
                let p01 = fma(c, Vec3A::splat(A1), Vec3A::splat(A0));
                let p23 = fma(c, Vec3A::splat(A3), Vec3A::splat(A2));
                let t = fma(c2, Vec3A::splat(A4), p23);
                fma(c2, t, p01)
            },
            c * (1.0 / 12.92),
        )
    }
    fn linear_to_srgb(c: Vec3A) -> Vec3A {
        Vec3A::select(
            c.cmpgt(Vec3A::splat(0.0031308)),
            {
                // This uses a Padé approximant for 1.055 c^(1/2.4) - 0.055:
                // (-0.0117264+21.0897 x+949.46 x^2+2225.62 x^3)/(1+176.398 x+1983.15 x^2+1035.65 x^3)
                const P0: f32 = -0.0117264;
                const P1: f32 = 21.0897;
                const P2: f32 = 949.46;
                const P3: f32 = 2225.62;
                const Q1: f32 = 176.398;
                const Q2: f32 = 1983.15;
                const Q3: f32 = 1035.65;
                let c2 = c * c;
                let p01 = fma(c, Vec3A::splat(P1), Vec3A::splat(P0));
                let p23 = fma(c, Vec3A::splat(P3), Vec3A::splat(P2));
                let p = fma(c2, p23, p01);
                let q01 = fma(c, Vec3A::splat(Q1), Vec3A::ONE);
                let q23 = fma(c, Vec3A::splat(Q3), Vec3A::splat(Q2));
                let q = fma(c2, q23, q01);
                p / q
            },
            c * 12.92,
        )
    }
    fn cbrt(x: Vec3A) -> Vec3A {
        // This is the fast cbrt approximation inspired by the non-std cbrt
        // implementation (https://gitlab.com/kornelski/oklab/-/blob/d3c074f154187dd5c0642119a6402a6c0753d70c/oklab/src/lib.rs#L61)
        // in the oklab crate by Kornel (https://gitlab.com/kornelski/), which
        // in turn seems to be based on the libm implementation.
        // In this version, I replaced the part after the initial guess with
        // one Halley iteration. This reduces accuracy, but saves 2 divisions
        // which helps performance a lot.
        const B: u32 = 709957561;
        fn initial_guess(x: f32) -> f32 {
            let bits = x.to_bits();
            // divide by 3 using multiplication and bitshift
            // this is only correct if bits <= 2^31, which is true for all
            // positive f32 values
            let div = ((bits as u64 * 1431655766) >> 32) as u32;
            f32::from_bits(div + B)
        }
        let t = Vec3A::from_array(x.to_array().map(initial_guess));

        // one halley iteration
        let s = t * t * t;
        t * (s + 2.0 * x) / (2.0 * s + x)
    }
}

#[allow(clippy::excessive_precision)]
fn srgb_to_oklab_impl<O: Operations>(srgb: Vec3A) -> Vec3A {
    let rgb = O::srgb_to_linear(srgb);

    let lms = Vec3A::new(
        rgb.dot(Vec3A::new(0.4122214708, 0.5363325363, 0.0514459929)),
        rgb.dot(Vec3A::new(0.2119034982, 0.6806995451, 0.1073969566)),
        rgb.dot(Vec3A::new(0.0883024619, 0.2817188376, 0.6299787005)),
    );
    let lms = O::cbrt(lms);

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

    // the clamping is necessary for out-of-gamut colors
    O::linear_to_srgb(rgb).clamp(Vec3A::ZERO, Vec3A::ONE)
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
                        (fast_oklab - ref_oklab).abs().max_element() < 0.001,
                        "{color:?} -> fast: {fast_oklab:?} vs ref: {ref_oklab:?}"
                    );

                    let srgb = fast_oklab_to_srgb(fast_oklab);

                    assert!(
                        (color - srgb).abs().max_element() < 0.0025,
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
                    assert!(
                        srgb.max_element() <= 1.0,
                        "{color:?} -> {fast_oklab:?} -> {srgb:?}"
                    );
                    assert!(
                        srgb.min_element() >= 0.0,
                        "{color:?} -> {fast_oklab:?} -> {srgb:?}"
                    );
                }
            }
        }
    }

    pub struct Scalar<O>(O);
    impl<O: Operations> Scalar<O> {
        fn srgb_to_linear(c: f32) -> f32 {
            O::srgb_to_linear(Vec3A::splat(c)).x
        }
        fn linear_to_srgb(c: f32) -> f32 {
            O::linear_to_srgb(Vec3A::splat(c)).x
        }
        fn cbrt(x: f32) -> f32 {
            O::cbrt(Vec3A::splat(x)).x
        }
    }
    type RefScalar = Scalar<Reference>;
    type FastScalar = Scalar<Fast>;

    #[test]
    fn test_linear_srgb() {
        for c in 0..=255 {
            let c = c as f32 / 255.0;
            let l = RefScalar::srgb_to_linear(c);
            let c2 = RefScalar::linear_to_srgb(l);

            assert!((c - c2).abs() < 1e-6, "{c} -> {c2}");
        }

        for c in 0..=255 {
            let c = c as f32 / 255.0;
            let l = FastScalar::srgb_to_linear(c);
            let c2 = FastScalar::linear_to_srgb(l);

            assert!((c - c2).abs() < 2.5e-3, "{c} -> {c2}");
            assert!((0.0..=1.0).contains(&l), "{c} -> {l}");
            assert!((0.0..=1.0).contains(&c2), "{c} -> {l}");
        }

        assert_eq!(RefScalar::srgb_to_linear(0.0), 0.0);
        assert!((RefScalar::srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert_eq!(FastScalar::linear_to_srgb(0.0), 0.0);
        assert!((FastScalar::srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_error_fast_srgb_to_linear() {
        assert_eq!(
            get_error_stats(RefScalar::srgb_to_linear, FastScalar::srgb_to_linear),
            "Error: avg=0.00007546 max=0.00012287 for 0.641"
        );
    }
    #[test]
    fn test_error_fast_linear_to_srgb() {
        assert_eq!(
            get_error_stats(RefScalar::linear_to_srgb, FastScalar::linear_to_srgb),
            "Error: avg=0.00105456 max=0.00236708 for 0.730"
        );
    }
    #[test]
    fn test_error_fast_cbrt() {
        assert_eq!(
            get_error_stats(RefScalar::cbrt, FastScalar::cbrt),
            "Error: avg=0.00000283 max=0.00001299 for 0.250"
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
