use crate::{DxgiFormat, FourCC};

#[derive(Debug)]
#[non_exhaustive]
pub enum DdsDecodeError {
    InvalidHeader(&'static str),
    UnsupportedDxgiFormat(DxgiFormat),
    UnsupportedFourCC(FourCC),
    TooManyMipMaps(u32),
    Io(std::io::Error),
}

impl From<std::io::Error> for DdsDecodeError {
    fn from(error: std::io::Error) -> Self {
        DdsDecodeError::Io(error)
    }
}
