use ddsd::*;

mod util;

const CHANNELS: &[Channels] = &[
    Channels::Grayscale,
    Channels::Alpha,
    Channels::Rgb,
    Channels::Rgba,
];
const PRECISION: &[Precision] = &[Precision::U8, Precision::U16, Precision::F32];

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
    let mut table =
        util::PrettyTable::from_header(&["Format", "C", "P", "Decoding (CxP)", "Encoding"]);
    table.add_empty_row();

    let gaps_at = [Format::R1_UNORM, Format::BC1_UNORM, Format::BC3_UNORM_RXGB];

    for format in util::ALL_FORMATS.iter().copied() {
        if gaps_at.contains(&format) {
            table.add_empty_row();
        }

        let supported_channels: Vec<Channels> = CHANNELS
            .iter()
            .copied()
            .filter(|&ch| format.supports_channels(ch))
            .collect();
        let supported_precision: Vec<Precision> = PRECISION
            .iter()
            .copied()
            .filter(|&prec| format.supports_precision(prec))
            .collect();
        let decoding = if supported_channels == CHANNELS && supported_precision == PRECISION {
            "Supported for all".to_string()
        } else if supported_channels.is_empty() || supported_precision.is_empty() {
            "Not supported".to_string()
        } else {
            let mut support = String::new();
            if supported_channels == CHANNELS {
                support.push_str("all");
            } else {
                let mut first = true;
                for ch in supported_channels {
                    if !first {
                        support.push(',');
                    }
                    first = false;
                    support.push_str(short_name(ch));
                }
            }

            support.push_str(" x ");

            if supported_precision == PRECISION {
                support.push_str("all");
            } else {
                let mut first = true;
                for prec in supported_precision {
                    if !first {
                        support.push(',');
                    }
                    first = false;
                    support.push_str(&format!("{:?}", prec));
                }
            }

            support
        };

        let encoding = if let Some(encoding) = format.encoding() {
            format!("{:?}", encoding)
        } else {
            "Not supported".to_string()
        };

        table.add_row(&[
            format!("{:?}", format),
            short_name(format.channels()).to_string(),
            format!("{:?}", format.precision()),
            decoding,
            encoding,
        ]);
    }

    util::compare_snapshot_text(
        &util::test_data_dir().join("format_metadata.txt"),
        &table.to_string(),
    )
    .unwrap();
}
