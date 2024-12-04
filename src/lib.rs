mod data;
mod dxgi_format;
mod error;
mod header;
mod util;
mod color;
mod decode;

pub use data::*;
pub use dxgi_format::*;
pub use error::*;
pub use header::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    // #[test]
    // fn it_works() {
    //     let mut file =
    //         File::open(r"C:\Program Files (x86)\Steam\steamapps\common\DARK SOULS III\Game_mod\other\cubegen.dds").expect("Failed to open file");
    //     let options = HeaderReadOptions {
    //         magic: MagicBytes::Check,
    //         ..Default::default()
    //     };
    //     let full_header = FullHeader::read_with_options(&mut file, &options).unwrap();
    //     let data = DataLayout::from_header(&full_header);
    //     let header = full_header.header;
    //     let bar = file;
    //     assert_eq!(header.size, 124);
    // }

    #[test]
    fn lots_of_files() {
        println!("Searching for DDS files...");
        let dds_files = glob::glob(
            r"C:\Users\micha\Git\ddsd\test-data\valid\**\*.dds",
        )
        .expect("Failed to read glob pattern")
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();

        println!("Found {} DDS files", dds_files.len());
        for (i, file) in dds_files.iter().enumerate() {
            if i % 100 == 0 {
                println!("{}", i);
            }

            let mut file = File::open(file).expect("Failed to open file");
            let total = file.metadata().unwrap().len();

            let full_header = FullHeader::read(&mut file).unwrap();
            let header_len = 4
                + 124
                + if full_header.header_dxt10.is_some() {
                    20
                } else {
                    0
                };
            let data_len = total - header_len;
            let data = DataLayout::from_header(&full_header);
            if let Ok(data) = data {
                let expected_len = data.byte_len() as u64;
                if expected_len != data_len {
                    let j = i + 1;
                    let again = DataLayout::from_header(&full_header);
                    assert!(again.is_ok());
                }
                assert_eq!(data_len, expected_len);
            }
        }
    }
}
