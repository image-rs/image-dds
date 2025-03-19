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
        let info = DdsInfo::read_with_options(&mut file, &options).expect("Failed to decode");

        let from_format = PixelInfo::from(info.format());
        let from_header = PixelInfo::from_header(info.header()).ok();

        assert_eq!(Some(from_format), from_header, "File: {:?}", &dds_path);
    }
}
