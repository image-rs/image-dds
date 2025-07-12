use dds::{header::*, *};

use std::{fs::File, io::Seek, num::NonZero, path::PathBuf};

mod util;

fn get_header_byte_len(header: &Header) -> u64 {
    4 + header.byte_len() as u64
}

#[test]
fn parse_data_layout_of_all_dds_files() {
    for dds_path in util::example_dds_files() {
        let mut file = File::open(&dds_path).expect("Failed to open file");
        let file_len = file.metadata().unwrap().len();

        let options = ParseOptions::new_permissive(Some(file_len));
        let decoder_result = Decoder::new_with_options(&mut file, &options);
        let info = match decoder_result {
            Ok(info) => util::DdsInfo::from_decoder(&info),
            Err(e) => panic!("Failed to decode {e}\nFile: {file:?}"),
        };

        let header = info.header;

        // skip cubemaps with array_size == 6 for now
        // https://github.com/RunDevelopment/dds/issues/4
        if let Some(dx10) = header.dx10() {
            if dx10.array_size == 6 {
                continue;
            }
        }

        let data_len = file_len - get_header_byte_len(&header);
        let expected_len = info.layout.data_len();
        assert_eq!(data_len, expected_len, "File: {:?}", &dds_path);
    }
}

#[test]
fn full_layout_snapshot() {
    let mut files: Vec<_> = util::example_dds_files()
        .into_iter()
        .map(|p| {
            let name = p
                .strip_prefix(util::test_data_dir())
                .unwrap()
                .to_str()
                .unwrap()
                .replace("\\", "/")
                .trim_matches('/')
                .to_owned();
            (name, p)
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));

    fn strict_header(dds_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open(dds_path)?;
        let file_len = file.metadata()?.len();

        let mut options = ParseOptions::default();
        options.permissive = false;
        let decoder = Decoder::new_with_options(&mut file, &options)?;

        let data_len = file_len - get_header_byte_len(decoder.header());
        if data_len != decoder.layout().data_len() {
            return Err("Data length mismatch".into());
        }

        Ok(())
    }

    fn validate_region(layout: &DataLayout) {
        let mut offset: u64 = 0;

        match layout {
            DataLayout::Texture(texture) => {
                assert_eq!(texture.data_offset(), 0);

                for surface in texture.iter_mips() {
                    assert_eq!(surface.data_offset(), offset);
                    assert_eq!(
                        surface.data_offset() + surface.data_len(),
                        surface.data_end()
                    );
                    offset += surface.data_len();
                }
            }
            DataLayout::Volume(volume) => {
                assert_eq!(volume.data_offset(), 0);

                for v in volume.iter_mips() {
                    assert_eq!(v.data_offset(), offset);
                    assert_eq!(v.data_offset() + v.data_len(), v.data_end());

                    for slice in v.iter_depth_slices() {
                        assert_eq!(slice.data_offset(), offset);
                        assert_eq!(slice.data_offset() + slice.data_len(), slice.data_end());
                        offset += slice.data_len();
                    }

                    assert_eq!(offset, v.data_end());
                }
            }
            DataLayout::TextureArray(texture_array) => {
                assert_eq!(texture_array.data_offset(), 0);

                for texture in texture_array.iter() {
                    assert_eq!(texture.data_offset(), offset);
                    assert_eq!(
                        texture.data_offset() + texture.data_len(),
                        texture.data_end()
                    );

                    for surface in texture.iter_mips() {
                        assert_eq!(surface.data_offset(), offset);
                        assert_eq!(
                            surface.data_offset() + surface.data_len(),
                            surface.data_end()
                        );
                        offset += surface.data_len();
                    }

                    assert_eq!(offset, texture.data_end());
                }
            }
        }

        assert_eq!(layout.data_offset(), 0);
        assert_eq!(layout.data_len(), layout.data_end());
        assert_eq!(layout.data_len(), offset);
    }

    fn collect_info(dds_path: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
        let mut file = File::open(dds_path)?;
        let file_len = file.metadata()?.len();

        let options = ParseOptions::new_permissive(Some(file_len));
        let decoder = Decoder::new_with_options(&mut file, &options)?;

        let header = decoder.header().clone();
        let format = decoder.format();
        let layout = decoder.layout();
        validate_region(&layout);
        if layout.is_cube_map() {
            assert!(header.is_cube_map());
        }

        let mut output = String::new();

        if let Err(e) = strict_header(dds_path) {
            output.push_str(&format!("Error if strict: {e}\n\n"));
        }

        let data_len = file_len - get_header_byte_len(&header);
        if data_len != layout.data_len() {
            output.push_str(&format!(
                "Data length mismatch: {} != {}\n\n",
                data_len,
                layout.data_len()
            ));
        }

        // RAW HEADER
        file.seek(std::io::SeekFrom::Start(4))?;
        let raw_header = RawHeader::read(&mut file)?;
        util::pretty_print_raw_header(&mut output, &raw_header);
        output.push('\n');

        // HEADER
        util::pretty_print_header(&mut output, &header);

        // FORMAT INFO
        output.push_str("\nPixel Format:\n");
        output.push_str(&format!("    format: {format:?}"));
        if header.is_srgb() {
            output.push_str(" (sRGB)");
        }
        output.push_str(&format!(
            "\n    pixel_info: {:?}\n",
            PixelInfo::from(format)
        ));

        // LAYOUT
        output.push('\n');
        util::pretty_print_data_layout(&mut output, &layout);

        Ok(output)
    }

    // create expected info
    let mut output = String::new();
    for (name, dds_path) in files {
        output.push_str(&name);
        output.push('\n');

        let info = match collect_info(&dds_path) {
            Ok(info) => info,
            Err(e) => format!("Error: {e}"),
        };

        for line in info.lines() {
            if line.is_empty() {
                output.push('\n');
            } else {
                output.push_str(&format!("    {line}\n"));
            }
        }

        output.push('\n');
        output.push('\n');
        output.push('\n');
    }

    util::compare_snapshot_text(&util::test_data_dir().join("layout_snapshot.txt"), &output)
        .unwrap();
}

#[test]
fn iter_and_get_volume() {
    let header_volume = Dx10Header {
        height: 128,
        width: 256,
        depth: Some(4),
        mipmap_count: NonZero::new(5).unwrap(),
        dxgi_format: DxgiFormat::R8G8B8A8_UNORM,
        resource_dimension: ResourceDimension::Texture3D,
        misc_flag: MiscFlags::empty(),
        array_size: 1,
        alpha_mode: AlphaMode::Unknown,
    }
    .into();

    let layout = DataLayout::from_header(&header_volume).unwrap();
    assert!(matches!(layout, DataLayout::Volume(_)));
    assert!(layout.texture().is_none());
    assert!(layout.texture_array().is_none());

    let volume = layout.volume().unwrap();

    let from_iter: Vec<VolumeDescriptor> = volume.iter_mips().collect();
    let from_get: Vec<VolumeDescriptor> = (0..u8::MAX).map_while(|i| volume.get(i)).collect();
    assert_eq!(from_iter, from_get);

    assert_eq!(volume.main(), volume.get(0).unwrap());

    for volume in volume.iter_mips() {
        let from_iter: Vec<SurfaceDescriptor> = volume.iter_depth_slices().collect();
        let from_get: Vec<SurfaceDescriptor> = (0..u32::MAX)
            .map_while(|i| volume.get_depth_slice(i))
            .collect();
        assert_eq!(from_iter, from_get);
    }
}

#[test]
fn iter_and_get_texture_array() {
    let header_texture_array = Dx10Header {
        height: 128,
        width: 256,
        depth: None,
        mipmap_count: NonZero::new(5).unwrap(),
        dxgi_format: DxgiFormat::R8G8B8A8_UNORM,
        resource_dimension: ResourceDimension::Texture2D,
        misc_flag: MiscFlags::empty(),
        array_size: 4,
        alpha_mode: AlphaMode::Unknown,
    }
    .into();

    let layout = DataLayout::from_header(&header_texture_array).unwrap();
    assert!(matches!(layout, DataLayout::TextureArray(_)));
    assert!(layout.texture().is_none());
    assert!(layout.volume().is_none());

    let array = layout.texture_array().unwrap();
    assert!(array.len() == 4);
    assert!(!array.is_empty());

    let from_iter: Vec<Texture> = array.iter().collect();
    let from_get: Vec<Texture> = (0..usize::MAX).map_while(|i| array.get(i)).collect();
    assert_eq!(from_iter, from_get);

    for texture in array.iter() {
        let from_iter: Vec<SurfaceDescriptor> = texture.iter_mips().collect();
        let from_get: Vec<SurfaceDescriptor> = (0..u8::MAX).map_while(|i| texture.get(i)).collect();
        assert_eq!(from_iter, from_get);

        assert_eq!(texture.main(), texture.get(0).unwrap());
    }
}

#[test]
fn empty_array() {
    #![allow(clippy::len_zero)]

    let header_texture_array = Dx10Header {
        height: 128,
        width: 256,
        depth: None,
        mipmap_count: NonZero::new(5).unwrap(),
        dxgi_format: DxgiFormat::R8G8B8A8_UNORM,
        resource_dimension: ResourceDimension::Texture2D,
        misc_flag: MiscFlags::empty(),
        array_size: 0, // empty
        alpha_mode: AlphaMode::Unknown,
    }
    .into();

    let layout = DataLayout::from_header(&header_texture_array).unwrap();
    let array = layout.texture_array().unwrap();

    assert!(array.len() == 0);
    assert!(array.is_empty());
    assert!(array.iter().next().is_none());
    assert!(array.data_len() == 0);
}

#[test]
fn weird_and_invalid_headers() {
    let headers: &[Header] = &[
        // just some simple headers
        //
        Dx9Header::new_image(100, 100, FourCC::DXT1.into()).into(),
        Dx9Header::new_cube_map(100, 100, FourCC::DXT1.into()).into(),
        Dx9Header::new_cube_map(100, 100, FourCC::DXT1.into())
            .with_cube_map_faces(CubeMapFaces::POSITIVE_X | CubeMapFaces::NEGATIVE_Y)
            .into(),
        Dx9Header::new_volume(100, 100, 4, FourCC::DXT1.into()).into(),
        Dx10Header::new_image(100, 100, DxgiFormat::BC1_UNORM).into(),
        Dx10Header::new_cube_map(100, 100, DxgiFormat::BC1_UNORM).into(),
        Dx10Header::new_cube_map(100, 100, DxgiFormat::BC1_UNORM)
            .with_array_size(4)
            .into(),
        Dx10Header::new_volume(100, 100, 4, DxgiFormat::BC1_UNORM).into(),
        Dx10Header::new_image(100, 1, DxgiFormat::R8_UNORM)
            .with_resource_dimension(ResourceDimension::Texture1D)
            .into(),
        //
        // too many mipmaps
        Header::new_image(1, 1, Format::BC1_UNORM).with_mipmap_count(123456),
        //
        // unknown pixel format
        Dx10Header::new_image(100, 100, DxgiFormat::UNKNOWN).into(),
        Dx9Header::new_image(100, 100, FourCC::NONE.into()).into(),
        // despite the invalid pixel format, we can create a proper layout
        Dx9Header::new_image(
            100,
            100,
            MaskPixelFormat {
                flags: PixelFormatFlags::empty(),
                rgb_bit_count: RgbBitCount::Count16,
                r_bit_mask: 0,
                g_bit_mask: 0,
                b_bit_mask: 0,
                a_bit_mask: 0,
            }
            .into(),
        )
        .into(),
        //
        // zero dimension
        Header::new_image(0, 100, Format::BC1_UNORM),
        Header::new_image(100, 0, Format::BC1_UNORM),
        Header::new_volume(100, 100, 0, Format::BC1_UNORM),
        Header::new_cube_map(100, 0, Format::BC1_UNORM),
        // volume without depth
        Header::new_volume(100, 100, 100, Format::BC1_UNORM).with_dimensions(10, 10, None),
        //
        // cube map with huge array size
        Dx10Header::new_cube_map(100, 100, DxgiFormat::BC1_UNORM)
            .with_array_size(u32::MAX)
            .into(),
        //
        // HUGE files
        Header::new_image(u32::MAX, u32::MAX, Format::R8_UNORM),
        Header::new_image(u32::MAX, u32::MAX, Format::R8_UNORM).with_mipmap_count(5),
        Header::new_image(u32::MAX, u32::MAX, Format::R16_UNORM),
        Dx10Header::new_image(u32::MAX, 2, DxgiFormat::R8_UNORM)
            .with_array_size(u32::MAX)
            .into(),
        Header::new_volume(u32::MAX, u32::MAX, 1, Format::R8_UNORM),
        Header::new_volume(u32::MAX, u32::MAX, 2, Format::R8_UNORM),
        Header::new_volume(u32::MAX, u32::MAX, 1, Format::R8_UNORM).with_mipmap_count(5),
        Header::new_volume(u32::MAX, u32::MAX, 1, Format::R16_UNORM),
        //
        // non-2D cube map
        Dx10Header::new_cube_map(100, 100, DxgiFormat::BC1_UNORM)
            .with_resource_dimension(ResourceDimension::Texture1D)
            .into(),
        Dx10Header::new_cube_map(100, 100, DxgiFormat::BC1_UNORM)
            .with_resource_dimension(ResourceDimension::Texture3D)
            .into(),
        Dx9Header::new_volume(100, 100, 1, FourCC::DXT1.into())
            .with_cube_map_faces(CubeMapFaces::ALL)
            .into(),
    ];

    let output = &mut String::new();

    for header in headers {
        util::pretty_print_header(output, header);
        output.push('\n');

        match DataLayout::from_header(header) {
            Ok(layout) => util::pretty_print_data_layout(output, &layout),
            Err(e) => output.push_str(&format!("Error:\n    {e}\n")),
        };

        output.push_str("\n\n\n");
    }

    util::compare_snapshot_text(
        &util::test_data_dir().join("invalid_header_layout.txt"),
        output,
    )
    .unwrap();
}
