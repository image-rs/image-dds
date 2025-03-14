use std::io::Write;

use bitflags::bitflags;

use crate::{cast, ColorFormat, ColorFormatSet, EncodeError, Precision};

use super::{Dithering, EncodeOptions};

pub(crate) struct Args<'a, 'b> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub color: ColorFormat,
    pub writer: &'b mut dyn Write,
    pub options: EncodeOptions,
}
impl<'a, 'b> Args<'a, 'b> {
    fn from(
        data: &'a [u8],
        width: usize,
        color: ColorFormat,
        writer: &'b mut dyn Write,
        options: EncodeOptions,
    ) -> Result<Self, EncodeError> {
        if data.is_empty() {
            return Err(EncodeError::InvalidLines);
        }

        let bytes_per_pixel = color.bytes_per_pixel() as usize;
        debug_assert!(bytes_per_pixel > 0);
        if data.len() % bytes_per_pixel != 0 {
            return Err(EncodeError::InvalidLines);
        }

        let stride = width * bytes_per_pixel;
        if stride == 0 || data.len() % stride != 0 {
            return Err(EncodeError::InvalidLines);
        }

        let height = data.len() / stride;
        if stride * height != data.len() {
            return Err(EncodeError::InvalidLines);
        }

        Ok(Self {
            data,
            width,
            height,
            color,
            writer,
            options,
        })
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Flags: u8 {
        /// Whether all U8 values will be encoded exactly, meaning no loss of
        /// precision.
        ///
        /// SNORM8 is considered exact.
        const EXACT_U8 = 0x1;
        /// Whether all U16 values will be encoded exactly, meaning no loss of
        /// precision.
        ///
        /// SNORM16 is considered exact.
        ///
        /// This flag implies `EXACT_U8`.
        const EXACT_U16 = 0x2 | Self::EXACT_U8.bits();
        /// Whether all F32 values will be encoded exactly, meaning no loss of
        /// precision.
        ///
        /// This flag implies `EXACT_U16` and `EXACT_U8`.
        const EXACT_F32 = 0x4 | Self::EXACT_U16.bits();
        /// Whether color dithering is supported.
        const DITHER_COLOR = 0x8;
        /// Whether alpha dithering is supported.
        const DITHER_ALPHA = 0x16;
        /// Whether both alpha and color dithering is supported.
        const DITHER_ALL = Self::DITHER_COLOR.bits() | Self::DITHER_ALPHA.bits();
    }
}

impl Flags {
    pub const fn exact_for(precision: Precision) -> Self {
        match precision {
            Precision::U8 => Flags::EXACT_U8,
            Precision::U16 => Flags::EXACT_U16,
            Precision::F32 => Flags::EXACT_F32,
        }
    }
    pub const fn get_dithering(self) -> Dithering {
        let color = self.contains(Flags::DITHER_COLOR);
        let alpha = self.contains(Flags::DITHER_ALPHA);
        Dithering::new(color, alpha)
    }
}

pub(crate) struct Encoder {
    pub color_formats: ColorFormatSet,
    pub flags: Flags,
    pub encode: fn(Args) -> Result<(), EncodeError>,
}
impl Encoder {
    pub const fn copy(color: ColorFormat) -> Self {
        Self {
            color_formats: ColorFormatSet::from_single(color),
            flags: Flags::exact_for(color.precision),
            encode: |args| copy_directly(args),
        }
    }

    pub const fn add_flags(mut self, flags: Flags) -> Self {
        self.flags = self.flags.union(flags);
        self
    }

    pub fn encode(&self, args: Args) -> Result<(), EncodeError> {
        assert!(
            self.color_formats.contains(args.color),
            "Picked the wrong encoder"
        );

        (self.encode)(args)
    }
}
pub(crate) struct EncoderSet {
    pub supported_dithering: Dithering,
    pub encoders: &'static [Encoder],
}
impl EncoderSet {
    pub const fn new(encoders: &'static [Encoder]) -> Self {
        assert!(!encoders.is_empty());

        let mut combined_flags = Flags::empty();
        let mut combined_colors = ColorFormatSet::EMPTY;
        let mut i = 0;
        while i < encoders.len() {
            let e = &encoders[i];
            combined_flags = combined_flags.union(e.flags);
            combined_colors = combined_colors.union(e.color_formats);
            i += 1;
        }

        assert!(
            combined_colors.is_all(),
            "All color formats must be supported"
        );

        let supported_dithering = combined_flags.get_dithering();

        Self {
            supported_dithering,
            encoders,
        }
    }

    fn encoders_for_color(&self, color: ColorFormat) -> impl Iterator<Item = &Encoder> {
        self.encoders
            .iter()
            .filter(move |e| e.color_formats.contains(color))
    }
    fn pick_encoder(&self, color: ColorFormat, options: &EncodeOptions) -> &Encoder {
        // Firstly, if we have an encoder that can encode the current precision
        // exactly, use it.
        let precision_flag = Flags::exact_for(color.precision);
        for encoder in self.encoders_for_color(color) {
            if encoder.flags.contains(precision_flag) {
                return encoder;
            }
        }

        // Secondly, search for encoders that perform the requested dithering.
        if options.dithering != Dithering::None {
            for encoder in self.encoders_for_color(color) {
                if encoder.flags.get_dithering().intersect(options.dithering) != Dithering::None {
                    return encoder;
                }
            }
        }

        // Lastly, just pick any encoder that can do the job.
        self.encoders_for_color(color)
            .next()
            .expect("all color formats to be supported")
    }

    pub fn encode(
        &self,
        data: &[u8],
        width: u32,
        color: ColorFormat,
        writer: &mut dyn Write,
        options: &EncodeOptions,
    ) -> Result<(), EncodeError> {
        let encoder = self.pick_encoder(color, options);
        let args = Args::from(data, width as usize, color, writer, options.clone())?;
        encoder.encode(args)
    }
}

fn copy_directly(args: Args) -> Result<(), EncodeError> {
    let Args {
        data,
        color,
        writer,
        ..
    } = args;

    // We can always just write everything directly on LE systems
    // and when the precision is U8
    if cfg!(target_endian = "little") || color.precision == Precision::U8 {
        writer.write_all(data)?;
        return Ok(());
    }

    // We need to convert to LE, so we need to allocate a buffer
    let mut buffer = [0_u8; 4096];
    let chuck_size = buffer.len();

    for chunk in data.chunks(chuck_size) {
        debug_assert!(chunk.len() % color.precision.size() as usize == 0);
        let chunk_buffer = &mut buffer[..chunk.len()];
        chunk_buffer.copy_from_slice(chunk);
        cast::slice_ne_to_le(color.precision, chunk_buffer);
        writer.write_all(chunk_buffer)?;
    }

    Ok(())
}
