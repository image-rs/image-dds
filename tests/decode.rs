use std::path::{Path, PathBuf};

use ddsd::*;
use util::Castable;
use Precision::*;

mod util;

#[test]
fn decode_all_dds_files() {
    fn get_png_path(dds_path: &Path) -> PathBuf {
        util::test_data_dir()
            .join("output")
            .join(dds_path.parent().unwrap().file_name().unwrap())
            .join(dds_path.file_name().unwrap())
            .with_extension("png")
    }
    fn dds_to_png_8bit(
        dds_path: &PathBuf,
        png_path: &PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (image, _) = util::read_dds_png_compatible(dds_path)?;

        // compare to PNG
        util::compare_snapshot_png_u8(png_path, &image)?;

        Ok(())
    }

    let mut failed_count = 0;
    for dds_path in util::example_dds_files() {
        if let Err(e) = dds_to_png_8bit(&dds_path, &get_png_path(&dds_path)) {
            let path = dds_path.strip_prefix(util::test_data_dir()).unwrap();
            eprintln!("Failed to convert {:?}: {}", path, e);
            failed_count += 1;
        }
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
}

#[test]
fn decode_rect() {
    let files = [
        // "normal" format
        "images/uncompressed/DX9 B4G4R4A4_UNORM.dds",
        // This one is optimized for mem-copying
        "images/uncompressed/DX10 R8_UNORM.dds",
    ]
    .map(|x| util::test_data_dir().join(x));

    fn get_png_path(dds_path: &Path) -> PathBuf {
        util::test_data_dir()
            .join("output-rect")
            .join(dds_path.file_name().unwrap())
            .with_extension("png")
    }
    fn dds_to_png_8bit(
        dds_path: &PathBuf,
        png_path: &PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (image, _) = util::read_dds_rect_as_u8(dds_path, Rect::new(47, 2, 63, 35))?;

        // compare to PNG
        util::compare_snapshot_png_u8(png_path, &image)?;

        Ok(())
    }

    let mut failed_count = 0;
    for test_image in files {
        if let Err(e) = dds_to_png_8bit(&test_image, &get_png_path(&test_image)) {
            let path = test_image.strip_prefix(util::test_data_dir()).unwrap();
            eprintln!("Failed to convert {:?}: {}", path, e);
            failed_count += 1;
        }
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
}

#[test]
fn decode_all_color_formats() {
    trait NormMax {
        const NORM_MAX: Self;
    }
    impl NormMax for u8 {
        const NORM_MAX: Self = u8::MAX;
    }
    impl NormMax for u16 {
        const NORM_MAX: Self = u16::MAX;
    }
    impl NormMax for f32 {
        const NORM_MAX: Self = 1.0;
    }

    fn convert_channels<T>(data: &[T], from: Channels, to: Channels) -> Vec<T>
    where
        T: Copy + Default + Castable + NormMax,
    {
        if from == to {
            return data.to_vec();
        }

        fn convert<const N: usize, const M: usize, T>(
            data: &[T],
            f: impl Fn([T; N]) -> [T; M],
        ) -> Vec<T>
        where
            T: Copy + Default + Castable,
        {
            let pixels = data.len() / N;
            let mut result: Vec<T> = vec![Default::default(); pixels * M];

            let data_n: &[[T; N]] = util::cast_slice(data);
            let result_m: &mut [[T; M]] = util::cast_slice_mut(&mut result);

            for (i, o) in data_n.iter().zip(result_m.iter_mut()) {
                *o = f(*i);
            }

            result
        }

        match (from, to) {
            // already handled
            (Channels::Grayscale, Channels::Grayscale)
            | (Channels::Alpha, Channels::Alpha)
            | (Channels::Rgb, Channels::Rgb)
            | (Channels::Rgba, Channels::Rgba) => unreachable!(),

            (Channels::Grayscale, Channels::Alpha) => todo!(),
            (Channels::Grayscale, Channels::Rgb) => convert(data, |[g]| [g, g, g]),
            (Channels::Grayscale, Channels::Rgba) => convert(data, |[g]| [g, g, g, T::NORM_MAX]),
            (Channels::Alpha, Channels::Grayscale) => todo!(),
            (Channels::Alpha, Channels::Rgb) => todo!(),
            (Channels::Alpha, Channels::Rgba) => {
                convert(data, |[a]| [T::default(), T::default(), T::default(), a])
            }
            (Channels::Rgb, Channels::Grayscale) => todo!(),
            (Channels::Rgb, Channels::Alpha) => todo!(),
            (Channels::Rgb, Channels::Rgba) => convert(data, |[r, g, b]| [r, g, b, T::NORM_MAX]),
            (Channels::Rgba, Channels::Grayscale) => todo!(),
            (Channels::Rgba, Channels::Alpha) => convert(data, |[_, _, _, a]| [a]),
            (Channels::Rgba, Channels::Rgb) => convert(data, |[r, g, b, _]| [r, g, b]),
        }
    }
    fn u16_to_u8(data: &[u16]) -> Vec<u8> {
        fn n8(x: u16) -> u8 {
            ((x as u32 * 255 + 32895) >> 16) as u8
        }
        data.iter().copied().map(n8).collect()
    }
    fn f32_to_u8(data: &[f32]) -> Vec<u8> {
        fn n8(x: f32) -> u8 {
            (x * 255.0 + 0.5) as u8
        }
        data.iter().copied().map(n8).collect()
    }

    fn test_color_formats(dds_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let (reference, reader) = util::read_dds::<u8>(dds_path)?;
        let format = reader.format();

        for channels in format.supported_channels() {
            if format.supported_precisions().contains(U8) && channels != reference.channels {
                let image = util::read_dds_with_channels::<u8>(dds_path, channels)?.0;
                let reference = convert_channels(&reference.data, reference.channels, channels);
                assert!(reference == image.data)
            }
            if format.supported_precisions().contains(U16) {
                let image = util::read_dds_with_channels::<u16>(dds_path, channels)?.0;
                let reference = convert_channels(&reference.data, reference.channels, channels);
                assert!(reference == u16_to_u8(&image.data))
            }
            if format.supported_precisions().contains(F32) {
                let image = util::read_dds_with_channels::<f32>(dds_path, channels)?.0;
                let reference = convert_channels(&reference.data, reference.channels, channels);
                assert!(reference == f32_to_u8(&image.data))
            }
        }

        Ok(())
    }

    let mut failed_count = 0;
    for dds_path in util::example_dds_files() {
        if let Err(e) = test_color_formats(&dds_path) {
            let path = dds_path.strip_prefix(util::test_data_dir()).unwrap();
            eprintln!("Failed for {:?}: {}", path, e);
            failed_count += 1;
        }
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
}
