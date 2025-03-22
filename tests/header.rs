use std::{collections::HashSet, fs::File};

use dds::{header::*, *};

mod util;

fn get_headers() -> Vec<Header> {
    let mut header_set: HashSet<Header> = util::example_dds_files()
        .into_iter()
        .map(|p| {
            let mut file = File::open(p)?;
            let file_len = file.metadata()?.len();

            let options = ParseOptions::new_permissive(Some(file_len));
            Header::read(&mut file, &options)
        })
        .filter_map(Result::ok)
        .collect();

    add_more_header(|header| {
        header_set.insert(header.clone());
    });

    // sort header
    let mut headers: Vec<Header> = header_set.into_iter().collect();
    headers.sort_by(|a, b| {
        let cmp = a
            .width()
            .cmp(&b.width())
            .then(a.height().cmp(&b.height()))
            .then(a.depth().cmp(&b.depth()))
            .then(a.mipmap_count().get().cmp(&b.mipmap_count().get()));

        match (a, b) {
            (Header::Dx9(a), Header::Dx9(b)) => {
                let cmp = cmp.then(a.caps2.bits().cmp(&b.caps2.bits()));
                match (&a.pixel_format, &b.pixel_format) {
                    (Dx9PixelFormat::FourCC(a), Dx9PixelFormat::FourCC(b)) => {
                        a.0.cmp(&b.0).then(cmp)
                    }
                    (Dx9PixelFormat::Mask(a), Dx9PixelFormat::Mask(b)) => a
                        .flags
                        .bits()
                        .cmp(&b.flags.bits())
                        .then(a.rgb_bit_count.cmp(&b.rgb_bit_count))
                        .then(a.r_bit_mask.cmp(&b.r_bit_mask))
                        .then(a.g_bit_mask.cmp(&b.g_bit_mask))
                        .then(a.b_bit_mask.cmp(&b.b_bit_mask))
                        .then(a.a_bit_mask.cmp(&b.a_bit_mask))
                        .then(cmp),
                    _ => {
                        let a = match &a.pixel_format {
                            Dx9PixelFormat::FourCC(_) => 0,
                            Dx9PixelFormat::Mask(_) => 1,
                        };
                        let b = match &b.pixel_format {
                            Dx9PixelFormat::FourCC(_) => 0,
                            Dx9PixelFormat::Mask(_) => 1,
                        };
                        a.cmp(&b).then(cmp)
                    }
                }
            }
            (Header::Dx10(a), Header::Dx10(b)) => u32::from(a.resource_dimension)
                .cmp(&u32::from(b.resource_dimension))
                .then(a.misc_flag.bits().cmp(&b.misc_flag.bits()))
                .then(u32::from(a.dxgi_format).cmp(&u32::from(b.dxgi_format)))
                .then(a.array_size.cmp(&b.array_size))
                .then(cmp)
                .then(u32::from(a.alpha_mode).cmp(&u32::from(b.alpha_mode))),
            _ => {
                let a = match a {
                    Header::Dx9(_) => 0,
                    Header::Dx10(_) => 1,
                };
                let b = match b {
                    Header::Dx9(_) => 0,
                    Header::Dx10(_) => 1,
                };
                a.cmp(&b).then(cmp)
            }
        }
    });

    headers
}
fn add_more_header(mut add: impl FnMut(&Header)) {
    // create some simple headers
    add(&Header::new_image(100, 100, DxgiFormat::BC1_UNORM_SRGB));
    add(&Header::new_cube_map(100, 100, DxgiFormat::BC1_UNORM_SRGB));
    add(&Header::new_volume(
        100,
        100,
        100,
        DxgiFormat::BC1_UNORM_SRGB,
    ));
    add(&Dx9Header::new_image(100, 100, FourCC::DXT1.into()).into());
    add(&Dx9Header::new_cube_map(100, 100, FourCC::DXT1.into()).into());
    add(&Dx9Header::new_volume(100, 100, 100, FourCC::DXT1.into()).into());

    // DX9 partial cube map
    let partial_cube_map = Dx9Header::new_cube_map(64, 64, FourCC::DXT1.into())
        .with_cube_map_faces(CubeMapFaces::POSITIVE_X);
    assert!(partial_cube_map.is_cube_map());
    assert!(partial_cube_map.cube_map_faces().unwrap().count() == 1);
    add(&partial_cube_map.into());

    let mut dx9_volume_cube = Dx9Header::new_volume(64, 64, 64, FourCC::DXT1.into());
    dx9_volume_cube.caps2 = Caps2::VOLUME | Caps2::CUBE_MAP | Caps2::CUBE_MAP_ALL_FACES;
    assert!(dx9_volume_cube.is_volume());
    assert!(dx9_volume_cube.is_cube_map());
    add(&dx9_volume_cube.into());

    // Fun DX10
    add(&Header::new_image(100, 100, DxgiFormat::UNKNOWN));

    add(&Dx10Header::new_image(100, 100, DxgiFormat::BC1_UNORM)
        .with_array_size(3)
        .into());

    add(&Dx10Header::new_image(100, 100, DxgiFormat::BC1_UNORM)
        .with_alpha_mode(AlphaMode::Premultiplied)
        .into());
    add(&Dx10Header::new_image(100, 100, DxgiFormat::BC1_UNORM)
        .with_alpha_mode(AlphaMode::Opaque)
        .into());
    add(&Dx10Header::new_image(100, 100, DxgiFormat::BC1_UNORM)
        .with_alpha_mode(AlphaMode::Custom)
        .into());

    let dx10_volume_cube = Dx10Header::new_volume(64, 64, 64, DxgiFormat::BC1_UNORM)
        .with_misc_flags(MiscFlags::TEXTURE_CUBE);
    assert!(dx10_volume_cube.is_volume());
    assert!(dx10_volume_cube.is_cube_map());
    add(&dx10_volume_cube.into());
}

#[test]
fn raw_header_snapshot() {
    let headers = get_headers();

    fn collect_info(header: &Header) -> String {
        let mut output = String::new();

        // HEADER
        util::pretty_print_header(&mut output, header);
        output.push('\n');

        // RAW HEADER
        let raw = header.to_raw();
        util::pretty_print_raw_header(&mut output, &raw);

        output
    }

    // create expected info
    let mut output = String::new();
    for header in headers {
        let info = collect_info(&header);
        output.push_str(&info);
        output.push_str("\n\n\n");
    }

    util::compare_snapshot_text(
        &util::test_data_dir().join("raw_header_snapshot.txt"),
        &output,
    )
    .unwrap();
}

#[test]
fn convert_header_snapshot() {
    let headers = get_headers();

    fn collect_info(header: &Header) -> String {
        let mut output = String::new();

        // HEADER
        util::pretty_print_header(&mut output, header);

        // Convert header
        let converted: Option<(Header, Header)> = if header.dx10().is_some() {
            assert_eq!(header.dx10(), header.to_dx10().as_ref());
            header.to_dx9().and_then(|h| {
                let back = h.to_dx10()?;
                Some((h.into(), back.into()))
            })
        } else {
            assert_eq!(header.dx9(), header.to_dx9().as_ref());
            header.to_dx10().and_then(|h| {
                let back = h.to_dx9()?;
                Some((h.into(), back.into()))
            })
        };

        if let Some((converted, converted_back)) = converted {
            output.push_str(if converted.dx10().is_some() {
                "\ninto_dx10 "
            } else {
                "\ninto_dx9 "
            });
            util::pretty_print_header(&mut output, &converted);

            if &converted_back != header {
                output.push_str("\nChanged when converted back ");
                util::pretty_print_header(&mut output, &converted_back);
            }

            let format = Format::from_header(header).ok();
            let converted_format = Format::from_header(&converted).ok();
            let converted_back_format = Format::from_header(&converted_back).ok();

            if (converted_format, converted_back_format) != (format, format) {
                output.push_str("\nChanged format during conversion:\n");
                output.push_str(&format!("    Original:       {:?}\n", format));
                output.push_str(&format!("    Converted:      {:?}\n", converted_format));
                if converted_back_format != format {
                    output.push_str(&format!(
                        "    Converted back: {:?}\n",
                        converted_back_format
                    ));
                }
            }
        } else {
            output.push_str(&format!(
                "\nCan't be converted to {}\n",
                if header.dx10().is_some() {
                    "DX9"
                } else {
                    "DX10"
                }
            ));
        }

        output
    }

    // create expected info
    let mut output = String::new();
    for header in headers {
        let info = collect_info(&header);
        output.push_str(&info);
        output.push_str("\n\n\n");
    }

    util::compare_snapshot_text(
        &util::test_data_dir().join("convert_header_snapshot.txt"),
        &output,
    )
    .unwrap();
}

#[test]
fn raw_header_read_write() {
    // RawHeader is not supposed to so any validation and should any garbage
    // data just fine. In particular, `RawHeader::write` should perfectly
    // reproduce the bytes read by `RawHeader::read`

    {
        // DX9 path
        let garbage = [1234_u32.to_le(); 31];
        let garbage_bytes = util::as_bytes(&garbage);
        let raw = RawHeader::read(&mut &garbage_bytes[..]).unwrap();
        assert_eq!(raw.size, 1234);
        let mut written = Vec::new();
        raw.write(&mut written).unwrap();
        assert_eq!(garbage_bytes, written.as_slice());
    }
    {
        // DX10 path
        let mut garbage = [1234_u32.to_le(); 36];
        garbage[19] = PixelFormatFlags::FOURCC.bits().to_le();
        garbage[20] = u32::from(FourCC::DX10).to_le();
        let garbage_bytes = util::as_bytes(&garbage);
        let raw = RawHeader::read(&mut &garbage_bytes[..]).unwrap();
        assert_eq!(raw.size, 1234);
        assert_eq!(raw.pixel_format.flags, PixelFormatFlags::FOURCC);
        assert_eq!(raw.pixel_format.four_cc, FourCC::DX10);
        assert!(raw.dx10.is_some());
        assert_eq!(raw.dx10.as_ref().unwrap().dxgi_format, 1234);
        let mut written = Vec::new();
        raw.write(&mut written).unwrap();
        assert_eq!(garbage_bytes, written.as_slice());
    }
    {
        // *almost* DX10 path
        // This checks that RawHeader checks for BOTH the flag and the fourCC
        let mut garbage = [1234_u32.to_le(); 31];
        garbage[19] = 0;
        garbage[20] = u32::from(FourCC::DX10).to_le();
        let garbage_bytes = util::as_bytes(&garbage);
        let raw = RawHeader::read(&mut &garbage_bytes[..]).unwrap();
        assert_eq!(raw.size, 1234);
        assert_eq!(raw.pixel_format.flags, PixelFormatFlags::empty());
        assert_eq!(raw.pixel_format.four_cc, FourCC::DX10);
        assert!(raw.dx10.is_none());
        let mut written = Vec::new();
        raw.write(&mut written).unwrap();
        assert_eq!(garbage_bytes, written.as_slice());
    }
}

#[test]
fn header_write_read() {
    for header in get_headers() {
        let mut bytes = Vec::new();
        header.write(&mut bytes).unwrap();

        let mut options = ParseOptions::default();
        let parsed_strict = Header::read(&mut &bytes[..], &options).unwrap();
        assert_eq!(header, parsed_strict);

        options.permissive = true;
        let parsed_permissive = Header::read(&mut &bytes[..], &options).unwrap();
        assert_eq!(header, parsed_permissive);
    }
}

#[test]
fn magic_bytes() {
    let original_header = Header::new_image(123, 345, DxgiFormat::BC1_UNORM);

    let mut bytes = Vec::new();
    original_header.write(&mut bytes).unwrap();

    assert!(Header::read_magic(&mut &bytes[..]).is_ok());
    let mut options = ParseOptions::default();
    let parsed = Header::read(&mut &bytes[..], &options).unwrap();
    assert_eq!(original_header, parsed);
    options.skip_magic_bytes = true;
    let parsed_without_magic = Header::read(&mut &bytes[4..], &options).unwrap();
    assert_eq!(original_header, parsed_without_magic);

    let invalid_magic_bytes = "what am I doing?".as_bytes();
    assert!(matches!(
        Header::read_magic(&mut &invalid_magic_bytes[..]),
        Err(HeaderError::InvalidMagicBytes(_))
    ));
}

/// A collection of weird and invalid DDS header to test header parsing
#[test]
fn weird_and_invalid_headers() {
    fn valid_dx9_fourcc() -> RawHeader {
        Header::from(Dx9Header::new_image(123, 345, FourCC::DXT1.into())).to_raw()
    }
    fn valid_dx9_masked() -> RawHeader {
        Header::from(Dx9Header::new_image(
            123,
            345,
            MaskPixelFormat {
                flags: PixelFormatFlags::RGB,
                rgb_bit_count: RgbBitCount::Count32,
                r_bit_mask: 0xff,
                g_bit_mask: 0x00ff,
                b_bit_mask: 0x0000ff,
                a_bit_mask: 0x000000ff,
            }
            .into(),
        ))
        .to_raw()
    }
    fn valid_dx10() -> RawHeader {
        let header = Dx10Header::new_image(123, 345, DxgiFormat::BC1_UNORM)
            .with_alpha_mode(AlphaMode::Unknown);
        Header::from(header).to_raw()
    }

    fn apply_edit(mut raw: RawHeader, f: fn(&mut RawHeader)) -> RawHeader {
        f(&mut raw);
        raw
    }
    fn apply_edit_dx10(mut raw: RawHeader, f: fn(&mut RawDx10Header)) -> RawHeader {
        f(raw.dx10.as_mut().unwrap());
        raw
    }

    let raw_headers = [
        // valid headers for sanity checking
        valid_dx9_fourcc(),
        valid_dx9_masked(),
        valid_dx10(),
        //
        // invalid header size
        apply_edit(valid_dx9_fourcc(), |raw| raw.size = 0),
        apply_edit(valid_dx9_fourcc(), |raw| raw.size = 123),
        apply_edit(valid_dx9_fourcc(), |raw| {
            // This is an invalid header size, but one we accept because of
            // the game Stalker
            raw.size = 24;
        }),
        //
        // invalid pixel format size
        apply_edit(valid_dx9_fourcc(), |raw| raw.pixel_format.size = 0),
        apply_edit(valid_dx9_fourcc(), |raw| raw.pixel_format.size = 123),
        apply_edit(valid_dx9_fourcc(), |raw| {
            // This is an invalid pixel format size, but one we accept because of
            // the game Flat Out 2
            raw.size = 24;
        }),
        //
        // invalid pixel format flags
        apply_edit(valid_dx9_fourcc(), |raw| {
            raw.pixel_format.flags = PixelFormatFlags::empty()
        }),
        //
        // invalid pixel format rgb bit count
        apply_edit(valid_dx9_masked(), |raw| raw.pixel_format.rgb_bit_count = 7),
        //
        // invalid dxgi_format
        apply_edit_dx10(valid_dx10(), |dx10| dx10.dxgi_format = 0),
        apply_edit_dx10(valid_dx10(), |dx10| dx10.dxgi_format = 1234),
        apply_edit_dx10(valid_dx10(), |dx10| dx10.dxgi_format = u32::MAX),
        //
        // invalid resource_dimension
        apply_edit_dx10(valid_dx10(), |dx10| dx10.resource_dimension = 0),
        apply_edit_dx10(valid_dx10(), |dx10| dx10.resource_dimension = 1),
        apply_edit_dx10(valid_dx10(), |dx10| dx10.resource_dimension = 2),
        apply_edit_dx10(valid_dx10(), |dx10| dx10.resource_dimension = 123),
        //
        // invalid alpha_mode
        apply_edit_dx10(valid_dx10(), |dx10| dx10.misc_flags2 = u32::MAX),
        //
        // invalid array_size
        apply_edit_dx10(valid_dx10(), |dx10| {
            dx10.resource_dimension = 4; // Texture 3D
            dx10.array_size = 0;
        }),
        apply_edit_dx10(valid_dx10(), |dx10| {
            dx10.resource_dimension = 4; // Texture 3D
            dx10.array_size = 123;
        }),
    ];

    let output = &mut String::new();
    for raw in raw_headers.iter() {
        util::pretty_print_raw_header(output, raw);

        let mut options = ParseOptions::default();

        output.push_str("\nStrict parsing ");
        options.permissive = false;
        let strict = Header::from_raw(raw, &options);
        match &strict {
            Ok(strict) => util::pretty_print_header(output, strict),
            Err(err) => output.push_str(&format!("error: {}\n", err)),
        }

        output.push_str("\nPermissive parsing ");
        options.permissive = true;
        let permissive = Header::from_raw(raw, &options);
        if strict.is_ok() && strict.as_ref().ok() == permissive.as_ref().ok() {
            output.push_str("resulted in the same header.\n");
        } else {
            match &permissive {
                Ok(permissive) => util::pretty_print_header(output, permissive),
                Err(err) => output.push_str(&format!("error: {}\n", err)),
            }
        }

        output.push_str("\n\n\n");
    }

    util::compare_snapshot_text(&util::test_data_dir().join("header_parsing.txt"), output).unwrap()
}
