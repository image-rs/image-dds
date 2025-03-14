use ddsd::*;

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
    let mut table = util::PrettyTable::from_header(&["Format", "C", "P", "Encoding"]);
    table.add_empty_row();

    let gaps_at = [Format::R1_UNORM, Format::BC1_UNORM, Format::BC3_UNORM_RXGB];

    for format in util::ALL_FORMATS.iter().copied() {
        if gaps_at.contains(&format) {
            table.add_empty_row();
        }

        let encoding = if let Some(encoding) = format.encoding() {
            format!("{:?}", encoding)
        } else {
            "Not supported".to_string()
        };

        table.add_row(&[
            format!("{:?}", format),
            short_name(format.channels()).to_string(),
            format!("{:?}", format.precision()),
            encoding,
        ]);
    }

    util::compare_snapshot_text(
        &util::test_data_dir().join("format_metadata.txt"),
        &table.to_string(),
    )
    .unwrap();
}
