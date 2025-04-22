use std::{io::Write, num::NonZeroU8};

use bitflags::bitflags;

use crate::{
    cast, ColorFormat, ColorFormatSet, EncodingError, ImageView, Precision, Progress, Report,
};

use super::{
    Dithering, EncodeOptions, EncodingSupport, PreferredGroupSize, SizeMultiple, SIZE_MUL_2X2,
};

pub(crate) struct Args<'a, 'b, 'c, 'd> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub color: ColorFormat,
    pub writer: &'b mut dyn Write,
    pub progress: Option<&'c mut Progress<'d>>,
    pub options: EncodeOptions,
}
impl<'a, 'b, 'c, 'd> Args<'a, 'b, 'c, 'd> {
    fn from(
        image: ImageView<'a>,
        writer: &'b mut dyn Write,
        progress: Option<&'c mut Progress<'d>>,
        options: EncodeOptions,
    ) -> Result<Self, EncodingError> {
        Ok(Self {
            data: image.data(),
            width: image.width() as usize,
            height: image.height() as usize,
            color: image.color(),
            writer,
            progress,
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
    group_size: PreferredGroupSize,

    pub encode: fn(Args) -> Result<(), EncodingError>,
}
impl Encoder {
    pub const fn new(
        color_formats: ColorFormatSet,
        flags: Flags,
        encode: fn(Args) -> Result<(), EncodingError>,
    ) -> Self {
        Self {
            color_formats,
            flags,
            group_size: PreferredGroupSize::EntireImage,
            encode,
        }
    }
    pub const fn new_universal(encode: fn(Args) -> Result<(), EncodingError>) -> Self {
        Self::new(ColorFormatSet::ALL, Flags::empty(), encode)
    }
    pub const fn copy(color: ColorFormat) -> Self {
        Self {
            color_formats: ColorFormatSet::from_single(color),
            flags: Flags::exact_for(color.precision),
            group_size: PreferredGroupSize::EntireImage,
            encode: |args| copy_directly(args),
        }
    }

    pub const fn add_flags(mut self, flags: Flags) -> Self {
        self.flags = self.flags.union(flags);
        self
    }
    pub const fn with_group_size(mut self, group_size: PreferredGroupSize) -> Self {
        self.group_size = group_size;
        self
    }

    pub fn encode(&self, args: Args) -> Result<(), EncodingError> {
        assert!(
            self.color_formats.contains(args.color),
            "Picked the wrong encoder"
        );

        (self.encode)(args)
    }
}
bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct EncodeFormatFlags: u8 {
        const DITHER_COLOR = 0x1;
        const DITHER_ALPHA = 0x2;
        /// Whether dithering is done within a block, instead of globally.
        const LOCAL_DITHERING = 0x4;
    }
}
pub(crate) struct EncoderSet {
    flags: EncodeFormatFlags,
    split_height: Option<NonZeroU8>,
    size_multiple: Option<SizeMultiple>,
    encoders: &'static [Encoder],
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

        let mut flags = EncodeFormatFlags::empty();
        if supported_dithering.color() {
            flags = flags.union(EncodeFormatFlags::DITHER_COLOR);
        }
        if supported_dithering.alpha() {
            flags = flags.union(EncodeFormatFlags::DITHER_ALPHA);
        }

        Self {
            flags,
            split_height: NonZeroU8::new(1),
            size_multiple: None,
            encoders,
        }
    }
    pub const fn new_bc(encoders: &'static [Encoder]) -> Self {
        let mut set = Self::new(encoders);
        set.flags = set.flags.union(EncodeFormatFlags::LOCAL_DITHERING);
        set.split_height = NonZeroU8::new(4);
        set
    }
    pub const fn new_bi_planar(encoders: &'static [Encoder]) -> Self {
        let mut set = Self::new(encoders);
        set.split_height = None;
        set.size_multiple = Some(SIZE_MUL_2X2);
        set
    }

    pub const fn supported_dithering(&self) -> Dithering {
        Dithering::new(
            self.flags.contains(EncodeFormatFlags::DITHER_COLOR),
            self.flags.contains(EncodeFormatFlags::DITHER_ALPHA),
        )
    }
    pub const fn local_dithering(&self) -> bool {
        self.flags.contains(EncodeFormatFlags::LOCAL_DITHERING)
    }

    pub const fn encoding_support(&self) -> EncodingSupport {
        EncodingSupport {
            dithering: self.supported_dithering(),
            split_height: self.split_height,
            local_dithering: self.local_dithering(),
            size_multiple: self.size_multiple,
            group_size: self.encoders[0].group_size,
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
        writer: &mut dyn Write,
        image: ImageView,
        progress: Option<&mut Progress>,
        options: &EncodeOptions,
    ) -> Result<(), EncodingError> {
        let encoder = self.pick_encoder(image.color(), options);
        let args = Args::from(image, writer, progress, options.clone())?;
        encoder.encode(args)
    }
}

fn copy_directly(args: Args) -> Result<(), EncodingError> {
    let Args {
        data,
        color,
        writer,
        mut progress,
        ..
    } = args;

    progress.report(0.0);

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
