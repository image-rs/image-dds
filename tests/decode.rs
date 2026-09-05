use dds::{header::*, *};
use std::{
    fs::File,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use util::{Rect, Snapshot};

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
            // println!("debugger");
        }
        let (image, _) = util::read_dds_png_compatible(dds_path)?;

        // compare to PNG
        _ = util::PngSnapshot.write(png_path, &image)?;

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
                eprintln!("Failed to convert {path:?}: {e}");
                format!("Error: {e}")
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
        let header = Header::read(&mut file, &Default::default())?;
        let format = Format::from_header(&header)?;
        if !matches!(format, Format::BC6H_SF16 | Format::BC6H_UF16) {
            return Ok(None);
        }

        let (image, _) = util::read_dds_with_channels(dds_path, Channels::Rgb)?;

        // compare to PNG and ignore any errors
        _ = util::DdsF32Snapshot.result(output_dds_path, &image);

        let hex = util::hash_hex_f32(&image.data);

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
        "images/uncompressed/DX10 R1_UNORM.dds",
        // Bi-planar formats
        "images/bi-planar/DX10 NV12.dds",
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
        util::PngSnapshot.result(&get_png_path(dds_path, &stringify_rect(rect)), &image)
    }
    /// This reads the image into a 200x100 RGBA output buffer.
    /// The trick is that it read the image as multiple patches.
    fn patchwork(dds_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let width = 200;
        let height = 100;
        let mut image =
            util::Image::new_empty(Channels::Rgba, Size::new(width as u32, height as u32));

        // read dds
        let mut file = File::open(dds_path)?;
        let header = Header::read(&mut file, &Default::default())?;
        let format = Format::from_header(&header)?;
        let surface_size = header.size();

        // read in the whole DDS surface, because we need to read it multiple times
        let layout = DataLayout::from_header(&header)?;
        let surface_byte_len = layout.texture().unwrap().main().data_len();
        let mut surface = vec![0_u8; surface_byte_len as usize];
        file.read_exact(&mut surface)?;

        // patch it all together
        let break_points = [0.0, 0.2, 0.3333, 0.6, 0.62, 1.0];
        let patches_x = break_points.map(|x| f32::round(x * surface_size.width as f32) as u32);
        let patches_y = break_points.map(|x| f32::round(x * surface_size.height as f32) as u32);
        let skip = (2, 2);
        for (y_index, y_window) in patches_y.windows(2).enumerate() {
            for (x_index, x_window) in patches_x.windows(2).enumerate() {
                if (x_index, y_index) == skip {
                    continue;
                }

                let offset = Offset::new(x_window[0], y_window[0]);
                let size = Size::new(x_window[1] - x_window[0], y_window[1] - y_window[0]);

                let image = image.view_mut().cropped(
                    Offset::new(offset.x + x_index as u32, offset.y + y_index as u32),
                    size,
                );

                dds::decode_rect(
                    &mut Cursor::new(surface.as_slice()),
                    image,
                    offset,
                    surface_size,
                    format,
                    &DecodeOptions::default(),
                )?;
            }
        }

        // compare to PNG
        util::PngSnapshot.result(&get_png_path(dds_path, "patchwork"), &image)
    }

    let mut failed_count = 0;
    for test_image in files {
        let mut test = |rect| {
            if let Err(e) = single_rect(&test_image, rect) {
                let path = test_image.strip_prefix(util::test_data_dir()).unwrap();
                eprintln!("Failed to convert {path:?}: {e}");
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
                eprintln!("Failed to convert {path:?}: {e}");
                failed_count += 1;
            }
        };

        test_patchwork();
    }
    if failed_count > 0 {
        panic!("{failed_count} tests failed");
    }
}

/// Checks that decoding an empty rectangle is (1) supported and (2) behaves
/// the same way skipping the surface would.
#[test]
fn decode_empty_rect() {
    let width: u32 = 7;
    let height: u32 = 13;
    let mut dummy_data = vec![0_u8; (width * height) as usize];
    let header = Header::new_image(width, height, Format::R8_UNORM);
    let expected_pos = width as u64 * height as u64;

    // first: skip the surface
    {
        let mut reader = Cursor::new(&mut dummy_data);
        let mut decoder = Decoder::from_header(&mut reader, header.clone()).unwrap();
        decoder.skip_surface().unwrap();
        assert_eq!(reader.position(), expected_pos);
    }

    // second: decode an empty rectangle
    {
        let mut reader = Cursor::new(&mut dummy_data);
        let mut decoder = Decoder::from_header(&mut reader, header.clone()).unwrap();
        let image = ImageViewMut::new(&mut [], Size::new(0, 0), ColorFormat::RGBA_U8).unwrap();
        decoder.read_surface_rect(image, Offset::ZERO).unwrap();
        assert_eq!(reader.position(), expected_pos);
    }
}

/// Checks that all color formats are decoded correctly.
///
/// The idea here is that if you decode as u8, you should get same result as
/// decoding as u16/f32 and then converting to u8.
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
        let (reference, _) =
            util::read_dds_with_settings::<u8>(dds_path, util::ReadSettings::no_cube_map())?;

        let all_channels = [
            Channels::Alpha,
            Channels::Grayscale,
            Channels::Rgb,
            Channels::Rgba,
        ];
        for channels in all_channels.iter().copied() {
            if channels != reference.channels {
                let image = util::read_dds_with_channels::<u8>(dds_path, channels)?.0;
                let reference =
                    util::convert_channels(&reference.data, reference.channels, channels);
                assert!(
                    reference == image.data,
                    "Failed {channels:?} for {dds_path:?}"
                );
            }
            {
                let image = util::read_dds_with_channels::<u16>(dds_path, channels)?.0;
                let reference =
                    util::convert_channels(&reference.data, reference.channels, channels);
                assert!(
                    reference == u16_to_u8(&image.data),
                    "Failed {channels:?} for {dds_path:?}"
                )
            }
            {
                let image = util::read_dds_with_channels::<f32>(dds_path, channels)?.0;
                let reference =
                    util::convert_channels(&reference.data, reference.channels, channels);
                assert!(
                    reference == f32_to_u8(&image.data),
                    "Failed {channels:?} for {dds_path:?}"
                )
            }
        }

        Ok(())
    }

    let mut failed_count = 0;
    for dds_path in util::example_dds_files() {
        if let Err(e) = test_color_formats(&dds_path) {
            let path = dds_path.strip_prefix(util::test_data_dir()).unwrap();
            eprintln!("Failed for {path:?}: {e}");
            failed_count += 1;
        }
    }
    if failed_count > 0 {
        panic!("{failed_count} tests failed");
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
            &ParseOptions::default(),
            std::io::Cursor::new(dds_file.as_slice()),
            Channels::Rgb,
        )
        .unwrap();

        let has_non_finite = image.data.iter().copied().any(|x| !x.is_finite());
        assert!(has_non_finite, "Block {block:#x} did not decode to -INF");
    }
}

#[test]
fn test_unaligned() {
    // dummy image data of the encoded image
    let mut dummy_data = vec![0_u8; 4096];
    util::fill_random(&mut dummy_data);

    // aligned and unaligned buffers
    let mut buffer = vec![0_u32; 4096];
    let (first, second) = buffer.split_at_mut(2048);
    let aligned_buffer = util::as_bytes_mut(first);
    let unaligned_buffer = &mut util::as_bytes_mut(second)[1..];

    let size = Size::new(7, 7);
    let options = DecodeOptions::default();

    for format in [
        Format::R8G8B8A8_UNORM,
        Format::R16G16_UNORM,
        Format::R32_FLOAT,
        Format::AYUV,
        Format::R1_UNORM,
        Format::R8G8_B8G8_UNORM,
        Format::BC1_UNORM,
    ] {
        for &color in util::ALL_COLORS {
            let stride = size.width as usize * color.bytes_per_pixel() as usize;
            let bytes = stride * size.height as usize;

            let aligned = &mut aligned_buffer[..bytes];
            let unaligned = &mut unaligned_buffer[..bytes];

            let aligned_view = ImageViewMut::new(aligned, size, color).unwrap();
            let unaligned_view = ImageViewMut::new(unaligned, size, color).unwrap();

            dds::decode(&mut dummy_data.as_slice(), aligned_view, format, &options).unwrap();
            dds::decode(&mut dummy_data.as_slice(), unaligned_view, format, &options).unwrap();

            assert_eq!(aligned, unaligned, "Failed for {format:?} {color:?}");
        }
    }
}

#[test]
fn test_row_pitch() {
    // dummy image data of the encoded image
    let surface_size = Size::new(7, 13);
    let mut dummy_data = vec![0_u8; surface_size.pixels() as usize * 16];
    util::fill_random_deterministic(&mut dummy_data, None);

    let options = DecodeOptions::default();

    for &format in util::ALL_FORMATS {
        for &color in util::ALL_COLORS {
            let bpp = color.bytes_per_pixel() as usize;

            let mut cont_buffer = vec![0_u8; surface_size.pixels() as usize * bpp];
            let cont_view = ImageViewMut::new(&mut cont_buffer, surface_size, color).unwrap();

            let non_cont_size = Size::new(50, 55);
            let non_cont_offset = Offset::new(8, 31);
            let mut non_cont_buffer = vec![255_u8; non_cont_size.pixels() as usize * bpp];
            let non_cont_view = ImageViewMut::new(&mut non_cont_buffer, non_cont_size, color)
                .unwrap()
                .cropped(non_cont_offset, surface_size);

            assert_eq!(cont_view.size(), non_cont_view.size());

            dds::decode(&mut dummy_data.as_slice(), cont_view, format, &options).unwrap();
            dds::decode(&mut dummy_data.as_slice(), non_cont_view, format, &options).unwrap();

            let mut cont_view = ImageViewMut::new(&mut cont_buffer, surface_size, color).unwrap();
            let mut non_cont_view = ImageViewMut::new(&mut non_cont_buffer, non_cont_size, color)
                .unwrap()
                .cropped(non_cont_offset, surface_size);

            for (r1, r2) in cont_view.rows_mut().zip(non_cont_view.rows_mut()) {
                assert_eq!(r1, r2, "Failed for {format:?} {color:?}");
            }
        }
    }
}

mod errors {
    use super::*;

    fn new_decoder_32x32() -> Decoder<Cursor<Vec<u8>>> {
        let header = Header::new_image(32, 32, Format::R8G8B8A8_UNORM);
        let len = DataLayout::from_header(&header).unwrap().data_len();
        let data = vec![0_u8; len as usize];
        Decoder::from_header(Cursor::new(data), header).unwrap()
    }
    fn new_decoder_16x16x16() -> Decoder<Cursor<Vec<u8>>> {
        let header = Header::new_volume(16, 16, 16, Format::R8G8B8A8_UNORM);
        let len = DataLayout::from_header(&header).unwrap().data_len();
        let data = vec![0_u8; len as usize];
        Decoder::from_header(Cursor::new(data), header).unwrap()
    }

    #[test]
    fn wrong_surface_size() {
        let mut image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(32, 16));
        let mut decoder = new_decoder_32x32();

        let result = decoder.read_surface(image.view_mut());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DecodingError::UnexpectedSurfaceSize));
        assert_eq!(err.to_string(), "Unexpected surface size");
    }

    #[test]
    fn no_more_surfaces() {
        let mut image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(32, 32));
        let mut decoder = new_decoder_32x32();
        decoder.read_surface(image.view_mut()).unwrap(); // read first and only surface
        decoder.rewind_to_previous_surface().unwrap(); // rewind to the first surface
        decoder.skip_surface().unwrap(); // skip the first surface

        let result = decoder.read_surface(image.view_mut());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DecodingError::NoMoreSurfaces));
        assert_eq!(err.to_string(), "No more surfaces to decode");
    }

    #[test]
    fn not_a_cube_map() {
        let mut image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(32 * 4, 32 * 3));
        let mut decoder = new_decoder_32x32();

        let result = decoder.read_cube_map(image.view_mut());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DecodingError::NotACubeMap));
        assert_eq!(err.to_string(), "The DDS file is not a cube map");
    }

    #[test]
    fn cannot_skip_mipmaps_in_volume() {
        let mut image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(16, 16));
        let mut decoder = new_decoder_16x16x16();
        decoder.skip_mipmaps().unwrap(); // does nothing
        decoder.read_surface(image.view_mut()).unwrap(); // read first depth slice

        let result = decoder.skip_mipmaps();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DecodingError::CannotSkipMipmapsInVolume));
        assert_eq!(
            err.to_string(),
            "Cannot skip mipmaps within a volume texture"
        );
    }

    #[test]
    fn memory_limit() {
        fn new_decoder_nv12() -> Decoder<Cursor<Vec<u8>>> {
            let header = Header::new_image(64, 64, Format::NV12);
            let len = DataLayout::from_header(&header).unwrap().data_len();
            let data = vec![0_u8; len as usize];
            Decoder::from_header(Cursor::new(data), header).unwrap()
        }

        let mut image = util::Image::<u8>::new_empty(Channels::Rgb, Size::new(64, 64));
        let mut decoder = new_decoder_nv12();

        // reading without a memory limit works just fine
        decoder.read_surface(image.view_mut()).unwrap();
        decoder.rewind_to_start().unwrap();
        decoder.read_surface(image.view_mut()).unwrap();
        decoder.rewind_to_start().unwrap();

        // memory limit results in an error
        decoder.options.memory_limit = 0;
        let result = decoder.read_surface(image.view_mut());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DecodingError::MemoryLimitExceeded));
        assert_eq!(err.to_string(), "Memory limit exceeded");
    }

    #[test]
    fn rect_out_of_bounds() {
        // decodes a dummy 2x3 image to GRAY U8
        let decode_dummy = |rect: Rect| {
            let mut image = util::Image::<u8>::new_empty(Channels::Grayscale, rect.size());
            let data: &[u8] = &[0, 0, 0, 0, 0, 0];

            dds::decode_rect(
                &mut std::io::Cursor::new(data),
                image.view_mut(),
                rect.offset(),
                Size::new(2, 3),
                Format::R8_UNORM,
                &DecodeOptions::default(),
            )
        };

        let result = decode_dummy(Rect::new(0, 0, 100, 100));
        assert!(matches!(result, Err(DecodingError::RectOutOfBounds)));
        assert_eq!(
            format!("{}", result.unwrap_err()),
            "Rectangle is out of bounds of the image size"
        );

        let result = decode_dummy(Rect::new(2, 2, 1, 1));
        assert!(matches!(result, Err(DecodingError::RectOutOfBounds)));

        // even empty rect can be OoB
        let result = decode_dummy(Rect::new(4, 0, 0, 0));
        assert!(matches!(result, Err(DecodingError::RectOutOfBounds)));
        // edge case: empty rect at the end of the image
        let result = decode_dummy(Rect::new(2, 3, 0, 0));
        assert!(matches!(result, Ok(())));
    }

    #[test]
    fn out_of_memory() {
        // To decode a P016 image, the decoder needs to allocate a buffer for
        // the first plane. Creating an invalid image with width=2^32-1, each
        // line of the first plane will require roughly 8GB of memory. Forcing
        // the decoder to read 4096 lines, will cause it allocate 4096*8GB=32TB
        // of memory, which should be OOM.
        let output_size = Size::new(1, 4096);
        let mut output_buffer = vec![0_u8; output_size.pixels() as usize];
        let surface_size = Size::new(u32::MAX, output_size.height);

        let mut options = DecodeOptions::default();
        options.memory_limit = usize::MAX; // disable memory limit

        let result = dds::decode_rect(
            &mut std::io::empty(),
            ImageViewMut::new(&mut output_buffer, output_size, ColorFormat::GRAYSCALE_U8).unwrap(),
            Offset::new(0, 0),
            surface_size,
            Format::P016,
            &options,
        );

        assert!(matches!(result, Err(DecodingError::MemoryLimitExceeded)));
    }
}
