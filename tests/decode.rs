use std::{
    fs::File,
    path::{Path, PathBuf},
};

use ddsd::*;

mod util;

fn get_test_images() -> Vec<TestImage> {
    glob::glob(
        util::test_data_dir()
            .join("images/**/*.dds")
            .to_str()
            .unwrap(),
    )
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
fn parse_data_layout_of_all_dds_files() {
    for test_image in get_test_images() {
        let debug = test_image
            .path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("restricted");
        if debug {
            println!("Debugging: {:?}", &test_image.path);
        }

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
        let expected_len = decoder.layout().data_len();
        assert_eq!(data_len, expected_len, "File: {:?}", &test_image.path);
    }
}

#[test]
fn decode_all_dds_files() {
    fn get_png_path(dds_path: &Path) -> PathBuf {
        util::test_data_dir()
            .join("output")
            .join(dds_path.parent().unwrap().file_name().unwrap())
            .join(dds_path.file_name().unwrap())
            .with_extension("png")
    }
    fn dds_to_png_8bit(
        dds_path: &PathBuf,
        png_path: &PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (image, _) = util::read_dds_as_u8(dds_path)?;

        // compare to PNG
        util::compare_snapshot_png_u8(png_path, &image)?;

        Ok(())
    }

    let mut failed_count = 0;
    for test_image in get_test_images() {
        if let Err(e) = dds_to_png_8bit(&test_image.path, &get_png_path(&test_image.path)) {
            let path = test_image.path.strip_prefix(util::test_data_dir()).unwrap();
            eprintln!("Failed to convert {:?}: {}", path, e);
            failed_count += 1;
        }
    }
    if failed_count > 0 {
        panic!("{} tests failed", failed_count);
    }
}
