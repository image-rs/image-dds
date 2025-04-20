use dds::{header::*, *};

mod util;

fn short_name(channels: Channels) -> &'static str {
    match channels {
        Channels::Grayscale => "Gray",
        Channels::Alpha => "Alpha",
        Channels::Rgb => "RGB",
        Channels::Rgba => "RGBA",
    }
}

#[test]
fn supported_formats_metadata() {
    let mut table = util::PrettyTable::from_header(&[
        "Format",
        "Color",
        "bpp",
        "Header",
        "FourCC",
        "Encoding",
        "Dithering",
        "Split",
        "Size Mult",
    ]);

    let gaps_at = [
        ("Uncompressed", Format::R8G8B8_UNORM),
        ("Subsampled", Format::R1_UNORM),
        ("Bi-planar", Format::NV12),
        ("Block Compression", Format::BC1_UNORM),
        ("ASTC", Format::ASTC_4X4_UNORM),
        ("Non-standard", Format::BC3_UNORM_RXGB),
    ];

    for format in util::ALL_FORMATS.iter().copied() {
        if let Some((category, _)) = gaps_at.iter().find(|(_, gap)| *gap == format) {
            table.add_empty_row();
            table
                .get_mut(0, table.height() - 1)
                .push_str(&format!("— *{}*", category));
        }

        let bpp = {
            let size = 16 * 9 * 5 * 7;
            let size = Size::new(size, size);
            let bits = PixelInfo::from(format).surface_bytes(size).unwrap() * 8;
            let bpp = bits as f64 / size.pixels() as f64;
            // round to one decimal
            let bpp = (bpp * 10.0).round() / 10.0;
            if bpp == bpp.round() {
                format!("{:.0}", bpp)
            } else {
                format!("{:.1}", bpp)
            }
        };

        let supports_dx10 = Header::new_image(1, 1, format).to_dx10().is_some();
        let supports_dx9 = Header::new_image(1, 1, format).to_dx9().is_some();
        let dx_support = match (supports_dx9, supports_dx10) {
            (true, true) => "☑️",
            (true, false) => "DX9",
            (false, true) => "DX10",
            (false, false) => "❌",
        };

        let four_cc = if let Ok(four_cc) = FourCC::try_from(format) {
            let f = format!("{:?}", four_cc);
            if let Some(short) = f.strip_prefix("FourCC(").and_then(|f| f.strip_suffix(")")) {
                short.to_string()
            } else {
                f
            }
        } else {
            "".to_string()
        };

        let (encoding, dithering, split, size_mul) =
            if let Some(encoding) = format.encoding_support() {
                let mut dithering = String::new();
                if encoding.dithering() != Dithering::None {
                    let possible_dithering = match format.channels() {
                        Channels::Grayscale => Dithering::Color,
                        Channels::Alpha => Dithering::Alpha,
                        Channels::Rgb => Dithering::Color,
                        Channels::Rgba => Dithering::ColorAndAlpha,
                    };
                    if encoding.dithering() == possible_dithering {
                        dithering.push_str("✔️");
                    } else {
                        match encoding.dithering() {
                            Dithering::Color => dithering.push_str("Color only"),
                            Dithering::Alpha => dithering.push_str("Alpha only"),
                            _ => {}
                        }
                    }

                    if encoding.local_dithering() {
                        dithering.push_str(" (local)");
                    }
                }

                let split = if let Some(split_height) = encoding.split_height() {
                    if split_height.get() == 1 {
                        "✔️".to_string()
                    } else {
                        format!("✔️ ({})", split_height)
                    }
                } else {
                    "❌".to_string()
                };

                let size_mul = if let Some(size_multiple) = encoding.size_multiple() {
                    format!("{}x{}", size_multiple.width, size_multiple.height)
                } else {
                    "".to_string()
                };

                ("✔️".to_string(), dithering, split, size_mul)
            } else {
                (
                    "❌".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                )
            };

        table.add_row(&[
            format!("{:?}", format),
            format!(
                "{:5} {:?}",
                short_name(format.channels()),
                format.precision()
            ),
            bpp,
            dx_support.to_string(),
            four_cc,
            encoding,
            dithering,
            split,
            size_mul,
        ]);
    }

    let mut output = String::new();
    let add_lines = |output: &mut String, lines: &[&str]| {
        for line in lines {
            output.push_str(line);
            output.push('\n');
        }
    };
    add_lines(&mut output,&[
        "# Supported Formats",
        "",
        "<!-- This file is generated by `tests/format.rs`. -->",
        "",
        "**Legend:**",
        "- **Format:** The name of the format",
        "- **Color:** The native channels and precision of the image data",
        "- **bpp:** The number of bits per pixel",
        "- **Header:** The header version(s) that supports this format (☑️ if both DX9 and DX10 support it)",
        "- **FourCC:** The FourCC code for this format, if applicable",
        "- **Encoding:** Whether this format supports encoding",
        "- **Dithering:** Whether encoding with dithering is supported",
        "- **Split:** Whether format supports splitting the image into lines for parallel encoding",
        "- **Size Mult:** Only images with dimensions that are multiples of this value can be encoded (if no value is shown, the format supports any size)",
        "",
    ]);
    table.print_markdown(&mut output);

    util::compare_snapshot_text(
        &util::test_data_dir().join("../supported-formats.md"),
        &output,
    )
    .unwrap();
}

#[test]
fn format_conversion() {
    for &format in util::ALL_FORMATS {
        if format == Format::BC3_UNORM_NORMAL {
            continue;
        }

        if let Ok(dxgi) = DxgiFormat::try_from(format) {
            let roundtrip = Format::from_dxgi(dxgi).unwrap();
            assert_eq!(format, roundtrip, "DXGI -> Format -> DXGI: {:?}", format);
        }

        if let Ok(four_cc) = FourCC::try_from(format) {
            let roundtrip = Format::from_four_cc(four_cc).unwrap();
            assert_eq!(
                format, roundtrip,
                "FourCC -> Format -> FourCC: {:?}",
                format
            );
        }
    }
}
