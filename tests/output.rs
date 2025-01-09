use std::{fs::File, path::PathBuf};

use ddsd::*;

fn is_ci() -> bool {
    std::env::var("CI").is_ok()
}

fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("test-data");
    path
}

fn get_test_images() -> Vec<TestImage> {
    glob::glob(test_data_dir().join("images/**/*.dds").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .map(|x| x.unwrap())
        // ignore files starting with "_"
        .filter(|x| !x.file_name().unwrap().to_str().unwrap().starts_with('_'))
        .map(|x| TestImage { path: x })
        .collect()
}

struct TestImage {
    path: PathBuf,
}

#[test]
fn file_data_layout() {
    for test_image in get_test_images() {
        let mut file = File::open(&test_image.path).expect("Failed to open file");
        let file_len = file.metadata().unwrap().len();

        let decoder_result = DdsDecoder::new(&mut file);
        let decoder = match decoder_result {
            Ok(decoder) => decoder,
            Err(e) => panic!("Failed to decode {}\nFile: {:?}", e, file),
        };

        let header = decoder.header();
        let header_len = 4 + 124 + if header.dxt10.is_some() { 20 } else { 0 };
        let data_len = file_len - header_len;
        let expected_len = decoder.layout().byte_len();
        assert_eq!(data_len, expected_len, "File: {:?}", &test_image.path);
    }
}

#[test]
fn output() {
    for test_image in get_test_images() {
        let mut file = File::open(&test_image.path).expect("Failed to open file");
        let file_len = file.metadata().unwrap().len();

        let decoder_result = DdsDecoder::new(&mut file);
        let decoder = match decoder_result {
            Ok(decoder) => decoder,
            Err(e) => panic!("Failed to decode {}\nFile: {:?}", e, file),
        };

        // just decode the first surface

        let header = decoder.header();
        let header_len = 4 + 124 + if header.dxt10.is_some() { 20 } else { 0 };
        let data_len = file_len - header_len;
        let expected_len = decoder.layout().byte_len();
        assert_eq!(data_len, expected_len, "File: {:?}", &test_image.path);
    }
}
