#![forbid(unsafe_code)]

mod data;
mod decode;
mod detect;
mod error;
mod format;
mod header;
mod util;

use std::io::{BufRead, Read};

use data::*;
use error::*;
use format::*;
use header::*;

/// Additional options for the DDS decoder specifying how to read and interpret
/// the header.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Options {
    /// Whether magic bytes should be skipped when reading the header.
    ///
    /// DDS files typically start with the magic bytes `"DDS "`. By default, the
    /// decoder will check for these bytes and error if they are not present.
    ///
    /// If this is set to `true`, the decoder assume that the magic bytes are
    /// not present and immediately start reading the header. This can be used
    /// to read DDS files without magic bytes.
    ///
    /// Defaults to `false`.
    pub skip_magic_bytes: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            skip_magic_bytes: false,
        }
    }
}

pub struct Decoder {
    header: Header,
    format: SupportedFormat,
    layout: DataLayout,
}

impl Decoder {
    /// Creates a new decoder by reading the header from the given reader.
    ///
    /// This is equivalent to calling `Decoder::new_with(r, Options::default())`.
    /// See [`Self::new_with`] for more details.
    pub fn new<R: Read>(r: &mut R) -> Result<Self, DecodeError> {
        Self::new_with(r, Options::default())
    }
    /// Creates a new decoder with the given options by reading the header from the given reader.
    ///
    /// If this operations succeeds, the given reader will be positioned at the start of the data
    /// section. All offsets in [`DataLayout`] are relative to this position.
    pub fn new_with<R: Read>(r: &mut R, options: Options) -> Result<Self, DecodeError> {
        // magic bytes
        if !options.skip_magic_bytes {
            Header::read_magic(r)?;
        }

        // header
        let header = Header::read(r)?;

        Self::from_header_with(header, options)
    }

    pub fn from_header(header: Header) -> Result<Self, DecodeError> {
        Self::from_header_with(header, Options::default())
    }
    pub fn from_header_with(header: Header, options: Options) -> Result<Self, DecodeError> {
        // detect format
        let format = SupportedFormat::from_header(&header)?;

        // data layout
        let layout = DataLayout::from_header(&header, format)?;

        Ok(Self {
            header,
            format,
            layout,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn format(&self) -> SupportedFormat {
        self.format
    }
    pub fn layout(&self) -> &DataLayout {
        &self.layout
    }
}

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
        let dds_files = glob::glob(r"C:\Users\micha\Git\ddsd\test-data\valid\**\*.dds")
            .expect("Failed to read glob pattern")
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();

        println!("Found {} DDS files", dds_files.len());
        for (i, file) in dds_files.iter().enumerate() {
            println!("{}", i);

            let mut file = File::open(file).expect("Failed to open file");
            let file_len = file.metadata().unwrap().len();

            let decoder_result = Decoder::new(&mut file);
            if decoder_result.is_err() {
                println!("Failed to decode file: {:?}", file);
            }

            if let Ok(decoder) = decoder_result {
                let header = decoder.header();
                let header_len = 4 + 124 + if header.dxt10.is_some() { 20 } else { 0 };
                let data_len = file_len - header_len;
                let expected_len = decoder.layout().byte_len();
                if expected_len != data_len {
                    // let again = DataLayout::from_header(&header);
                    // assert!(again.is_ok());
                }
                assert_eq!(data_len, expected_len);
            }
        }
    }
}
