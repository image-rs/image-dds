use std::{collections::HashSet, fs::File};

use ddsd::*;

mod util;

fn get_headers() -> Vec<Header> {
    let header_set: HashSet<Header> = util::example_dds_files()
        .into_iter()
        .map(|p| {
            let mut file = File::open(p)?;
            let file_len = file.metadata()?.len();

            let mut options = ParseOptions::default();
            options.permissive = true;
            options.file_len = Some(file_len);
            Header::read(&mut file, &options)
        })
        .filter_map(Result::ok)
        .collect();

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
            header.clone().into_dx9().and_then(|h| {
                let back = h.to_dx10()?;
                Some((h.into(), back.into()))
            })
        } else {
            header.clone().into_dx10().and_then(|h| {
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
            output.push_str("\nCan't be converted\n");
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
