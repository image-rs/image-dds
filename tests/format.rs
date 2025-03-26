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
fn format_metadata() {
    let mut table = util::PrettyTable::from_header(&["Format", "C", "P", "bpp", "Encoding"]);
    table.add_empty_row();
    let mut conv_table = util::PrettyTable::from_header(&["Format", "DXGI", "FourCC", "Masked"]);
    conv_table.add_empty_row();

    let gaps_at = [
        Format::R1_UNORM,
        Format::NV12,
        Format::BC1_UNORM,
        Format::BC3_UNORM_RXGB,
    ];

    for format in util::ALL_FORMATS.iter().copied() {
        if gaps_at.contains(&format) {
            table.add_empty_row();
            conv_table.add_empty_row();
        }

        let encoding = if let Some(encoding) = format.encoding_support() {
            let mut out = "✔️ ".to_string();

            if let Some(block_height) = encoding.split_height() {
                out.push_str(&format!("split={:?} ", block_height));
            }
            if encoding.dithering() != Dithering::None {
                out.push_str(&format!("dithering={:?} ", encoding.dithering()));
                if encoding.local_dithering() {
                    out.push_str("(local) ");
                }
            }
            if encoding.size_multiple() != SizeMultiple::ONE {
                out.push_str(&format!(
                    "size_mul={}x{} ",
                    encoding.size_multiple().width_multiple, encoding.size_multiple().height_multiple
                ));
            }

            out.trim().to_string()
        } else {
            "❌".to_string()
        };

        table.add_row(&[
            format!("{:?}", format),
            short_name(format.channels()).to_string(),
            format!("{:?}", format.precision()),
            format!("{:?}", PixelInfo::from(format).bits_per_pixel()),
            encoding,
        ]);

        let dxgi = if let Ok(dxgi) = DxgiFormat::try_from(format) {
            format!("{:?}", u32::from(dxgi))
        } else {
            "-".to_string()
        };
        let four_cc = if let Ok(four_cc) = FourCC::try_from(format) {
            format!("{:?}", four_cc)
        } else {
            "".to_string()
        };
        let masked = if let Ok(Dx9PixelFormat::Mask(masked)) = Dx9PixelFormat::try_from(format) {
            let flags = format!("{:?}", masked.flags);
            let mut flags = flags.strip_prefix("PixelFormatFlags").unwrap();
            if !flags.contains(' ') {
                flags = flags.strip_prefix("(").unwrap().strip_suffix(")").unwrap();
            }
            if masked.flags == PixelFormatFlags::RGBA {
                flags = "RGBA";
            }
            format!(
                "flags:{:9} rgb_bits:{} r:{:x} g:{:x} b:{:x} a:{:x}",
                flags,
                masked.rgb_bit_count as u32,
                masked.r_bit_mask,
                masked.g_bit_mask,
                masked.b_bit_mask,
                masked.a_bit_mask
            )
        } else {
            "".to_string()
        };

        conv_table.add_row(&[format!("{:?}", format), dxgi, four_cc, masked]);
    }

    let mut output = table.to_string();
    output.push_str("\n\n\n");
    output.push_str(&conv_table.to_string());

    util::compare_snapshot_text(&util::test_data_dir().join("format_metadata.txt"), &output)
        .unwrap();
}

#[test]
fn format_conversion() {
    for &format in util::ALL_FORMATS {
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
