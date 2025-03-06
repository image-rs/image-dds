use std::{
    fs::File,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use ddsd::*;
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
        dds_path: &Path,
        png_path: &Path,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let name = dds_path.file_name().unwrap().to_str().unwrap();
        if name.contains("DX10 A8_UNORM") {
            println!("debugger");
        }
        let (image, _) = util::read_dds_png_compatible(dds_path)?;

        // compare to PNG
        _ = util::compare_snapshot_png_u8(png_path, &image);

        let hex = util::hash_hex(&image.data);
        Ok(hex)
    }

    let mut summaries = util::OutputSummaries::new("_hashes");

    for dds_path in util::example_dds_files() {
        let png_path = get_png_path(&dds_path);

        let file_output = match dds_to_png_8bit(&dds_path, &png_path) {
            Ok(hash) => hash,
            Err(e) => {
                let path = dds_path.strip_prefix(util::test_data_dir()).unwrap();
                eprintln!("Failed to convert {:?}: {}", path, e);
                format!("Error: {}", e)
            }
        };

        summaries.add_output_file(&png_path, &file_output);
    }

    summaries.snapshot_or_fail()
}

#[test]
fn decode_bc6_fuzz_hdr() {
    let bc_fuzz_output_dir = util::test_data_dir().join("output/bc fuzz");

    let get_output_dds_path = |dds_path: &Path| -> PathBuf {
        bc_fuzz_output_dir
            .join(dds_path.file_name().unwrap())
            .with_extension("dds")
    };
    fn test(
        dds_path: &Path,
        output_dds_path: &Path,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let mut file = File::open(dds_path)?;
        let decoder = DdsDecoder::new(&mut file)?;
        let format = decoder.format();
        if !matches!(format, DecodeFormat::BC6H_SF16 | DecodeFormat::BC6H_UF16) {
            return Ok(None);
        }

        let (image, _) = util::read_dds_with_channels(dds_path, Channels::Rgb)?;

        // compare to PNG and ignore any errors
        _ = util::compare_snapshot_dds_f32(output_dds_path, &image);

        let hex = util::hash_hex(util::as_bytes(&image.data));

        Ok(Some(hex))
    }

    let mut summaries = util::OutputSummaries::new("_hashes-bc6-hdr");

    for dds_path in util::example_dds_files_in("bc fuzz") {
        let output_dds_path = get_output_dds_path(&dds_path);
        match test(&dds_path, &output_dds_path) {
            Ok(None) => {} // ignore
            Ok(Some(hex)) => summaries.add_output_file(&output_dds_path, &hex),
            Err(e) => summaries.add_output_file_error(&output_dds_path, &*e),
        }
    }

    summaries.snapshot_or_fail()
}

#[test]
fn decode_rect() {
    let files = [
        // "normal" format
        "images/uncompressed/DX9 B4G4R4A4_UNORM.dds",
        // This one is optimized for mem-copying
        "images/uncompressed/DX10 R8_UNORM.dds",
        // Sub-sampled formats
        "images/sub-sampled/DX9 R8G8_B8G8_UNORM.dds",
        // Block-compressed formats
        "images/bc/DX10 BC7_UNORM.dds",
    ]
    .map(|x| util::test_data_dir().join(x));

    fn stringify_rect(rect: Rect) -> String {
        format!("{},{} {},{}", rect.x, rect.y, rect.width, rect.height)
    }
    fn get_png_path(dds_path: &Path, suffix: &str) -> PathBuf {
        let name = dds_path.file_name().unwrap().to_string_lossy().to_string();
        let mut name = name.replace(".dds", "");
        name += " - ";
        name += suffix;

        util::test_data_dir()
            .join("output-rect")
            .join(name)
            .with_extension("png")
    }
    fn single_rect(dds_path: &Path, rect: Rect) -> Result<(), Box<dyn std::error::Error>> {
        let (image, _) = util::read_dds_rect_as_u8(dds_path, rect)?;

        // compare to PNG
        util::compare_snapshot_png_u8(&get_png_path(dds_path, &stringify_rect(rect)), &image)?;

        Ok(())
    }
    /// This reads the image into a 200x100 RGBA output buffer.
    /// The trick is that it read the image as multiple patches.
    fn patchwork(dds_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let width = 200;
        let height = 100;
        let mut image = util::Image {
            data: vec![0_u8; width * height * 4],
            channels: Channels::Rgba,
            size: Size::new(width as u32, height as u32),
        };

        // read dds
        let mut file = File::open(dds_path)?;
        let decoder = DdsDecoder::new(&mut file)?;
        let size = decoder.header().size();
        let format = decoder.format();
        let target_color = ColorFormat::new(Channels::Rgba, U8);
        if !format.supports(target_color) {
            return Err("Format does not support decoding as RGBA U8".into());
        }

        // read in the whole DDS surface, because we need to read it multiple times
        let surface_byte_len = decoder.layout().texture().unwrap().main().data_len();
        let mut surface = vec![0_u8; surface_byte_len as usize];
        file.read_exact(&mut surface)?;

        // patch it all together
        let break_points = [0.0, 0.2, 0.3333, 0.6, 0.62, 1.0];
        let patches_x = break_points.map(|x| f32::round(x * size.width as f32) as u32);
        let patches_y = break_points.map(|x| f32::round(x * size.height as f32) as u32);
        let skip = (2, 2);
        for (y_index, y_window) in patches_y.windows(2).enumerate() {
            for (x_index, x_window) in patches_x.windows(2).enumerate() {
                if (x_index, y_index) == skip {
                    continue;
                }

                let rect = Rect::new(
                    x_window[0],
                    y_window[0],
                    x_window[1] - x_window[0],
                    y_window[1] - y_window[0],
                );

                let image_x = rect.x as usize + x_index;
                let image_y = rect.y as usize + y_index;

                let stride = image.stride();
                format.decode_rect(
                    &mut Cursor::new(surface.as_slice()),
                    size,
                    rect,
                    target_color,
                    &mut image.data[(image_y * stride + image_x * 4)..],
                    stride,
                )?;
            }
        }

        // compare to PNG
        util::compare_snapshot_png_u8(&get_png_path(dds_path, "patchwork"), &image)?;

        Ok(())
    }

    let mut failed_count = 0;
    for test_image in files {
        let mut test = |rect| {
            if let Err(e) = single_rect(&test_image, rect) {
                let path = test_image.strip_prefix(util::test_data_dir()).unwrap();
                eprintln!("Failed to convert {:?}: {}", path, e);
                failed_count += 1;
            }
        };

        test(Rect::new(47, 2, 63, 35));
        // single pixel to cover certain edge cases
        test(Rect::new(9, 41, 1, 1));
        test(Rect::new(10, 51, 1, 1));

        let mut test_patchwork = || {
            if let Err(e) = patchwork(&test_image) {
                let path = test_image.strip_prefix(util::test_data_dir()).unwrap();
                eprintln!("Failed to convert {:?}: {}", path, e);
                failed_count += 1;
            }
        };

        test_patchwork();
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
}

#[test]
fn decode_all_color_formats() {
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

        let all_channels = [
            Channels::Alpha,
            Channels::Grayscale,
            Channels::Rgb,
            Channels::Rgba,
        ];
        for channels in all_channels
            .iter()
            .copied()
            .filter(|x| format.supports_channels(*x))
        {
            if format.supports_precision(U8) && channels != reference.channels {
                let image = util::read_dds_with_channels::<u8>(dds_path, channels)?.0;
                let reference =
                    util::convert_channels(&reference.data, reference.channels, channels);
                assert!(
                    reference == image.data,
                    "Failed {:?} for {:?}",
                    channels,
                    dds_path
                );
            }
            if format.supports_precision(U16) {
                let image = util::read_dds_with_channels::<u16>(dds_path, channels)?.0;
                let reference =
                    util::convert_channels(&reference.data, reference.channels, channels);
                assert!(
                    reference == u16_to_u8(&image.data),
                    "Failed {:?} for {:?}",
                    channels,
                    dds_path
                )
            }
            if format.supports_precision(F32) {
                let image = util::read_dds_with_channels::<f32>(dds_path, channels)?.0;
                let reference =
                    util::convert_channels(&reference.data, reference.channels, channels);
                assert!(
                    reference == f32_to_u8(&image.data),
                    "Failed {:?} for {:?}",
                    channels,
                    dds_path
                )
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

/// A test for BC6H SF16 blocks which decode to -INF.
///
/// These blocks are pretty rare, so the BC fuzz tests unfortunately don't
/// cover them. Hence this test. The test block were found using brute force.
#[test]
fn neg_infinity_bc6_blocks() {
    let blocks = [0x800000000f_u128, 0xb80000000f_u128, 0xf80000000f_u128];

    for block in blocks {
        // create a little DDS file that only contains this block
        let mut dds_file: Vec<u8> = Vec::new();
        util::write_simple_dds_header(&mut dds_file, Size::new(4, 4), DxgiFormat::BC6H_SF16)
            .unwrap();
        dds_file.extend_from_slice(&block.to_le_bytes());

        // decode it
        let (image, _) = util::decode_dds_with_channels::<f32>(
            &Options::default(),
            dds_file.as_slice(),
            Channels::Rgb,
        )
        .unwrap();

        let has_non_finite = image.data.iter().copied().any(|x| !x.is_finite());
        assert!(has_non_finite, "Block {:#x} did not decode to -INF", block);
    }
}
