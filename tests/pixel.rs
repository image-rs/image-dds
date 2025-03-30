use dds::{header::*, *};

use std::fs::File;

mod util;

#[test]
fn from_header() {
    // Verify that the PixelInfo::from_header() gets the same result as PixelInfo::from(SupportedFormat)
    for dds_path in util::example_dds_files() {
        let mut file = File::open(&dds_path).expect("Failed to open file");
        let options = ParseOptions::new_permissive(Some(
            file.metadata().expect("Failed to get metadata").len(),
        ));
        let header = Header::read(&mut file, &options).expect("Failed to read header");
        let format = Format::from_header(&header).expect("Failed to get format");

        let from_format = PixelInfo::from(format);
        let from_header = PixelInfo::from_header(&header).ok();

        assert_eq!(Some(from_format), from_header, "File: {:?}", &dds_path);
    }
}
