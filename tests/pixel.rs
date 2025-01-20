use ddsd::*;

use std::fs::File;

mod util;

#[test]
fn from_header() {
    // Verify that the PixelInfo::from_header() gets the same result as PixelInfo::from(SupportedFormat)
    for dds_path in util::example_dds_files() {
        let mut file = File::open(&dds_path).expect("Failed to open file");
        let decoder = DdsDecoder::new(&mut file).expect("Failed to decode");

        let from_format = PixelInfo::from(decoder.format());
        let from_header = PixelInfo::from_header(decoder.header()).ok();

        assert_eq!(Some(from_format), from_header, "File: {:?}", &dds_path);
    }
}
