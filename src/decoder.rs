use std::io::Read;

use crate::{DataLayout, DecodeError, Format, Header, ParseOptions};

pub struct DdsDecoder {
    header: Header,
    format: Format,
    layout: DataLayout,
}

impl DdsDecoder {
    /// Creates a new decoder by reading the header from the given reader.
    ///
    /// This is equivalent to calling `Decoder::new_with(r, Options::default())`.
    /// See [`Self::new_with`] for more details.
    pub fn new<R: Read>(r: &mut R) -> Result<Self, DecodeError> {
        Self::new_with_options(r, &ParseOptions::default())
    }
    /// Creates a new decoder with the given options by reading the header from the given reader.
    ///
    /// If this operations succeeds, the given reader will be positioned at the start of the data
    /// section. All offsets in [`DataLayout`] are relative to this position.
    pub fn new_with_options<R: Read>(
        r: &mut R,
        options: &ParseOptions,
    ) -> Result<Self, DecodeError> {
        let header = Header::read(r, options)?;
        Self::from_header(header)
    }

    pub fn from_header(header: Header) -> Result<Self, DecodeError> {
        // detect format
        let format = Format::from_header(&header)?;

        Self::from_header_with_format(header, format)
    }
    pub fn from_header_with_format(header: Header, format: Format) -> Result<Self, DecodeError> {
        // data layout
        let layout = DataLayout::from_header_with(&header, format.into())?;

        Ok(Self {
            header,
            format,
            layout,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn format(&self) -> Format {
        self.format
    }
    pub fn layout(&self) -> &DataLayout {
        &self.layout
    }
}
