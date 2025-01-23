use crate::{
    cast, convert_channels_untyped, decode::read_write::PixelArgs, Channels, Norm, Precision,
};

use super::read_write::ProcessPixelsFn;

pub(crate) fn adapt(mut adapter: impl Adapter, from: Channels, to: Channels, precision: Precision) {
    if from == to {
        adapter.direct();
    } else {
        match precision {
            Precision::U8 => adapter.map::<u8>(from, to),
            Precision::U16 => adapter.map::<u16>(from, to),
            Precision::F32 => adapter.map::<f32>(from, to),
        }
    }
}

pub(crate) trait Adapter {
    fn direct(&mut self);

    fn map<Precision>(&mut self, from: Channels, to: Channels)
    where
        Precision: cast::Castable + cast::IntoNeBytes + Norm,
        [Precision; 1]: cast::IntoNeBytes,
        [Precision; 3]: cast::IntoNeBytes,
        [Precision; 4]: cast::IntoNeBytes;
}

pub(crate) struct UncompressedAdapter<'a, 'b> {
    pub encoded: &'a [u8],
    pub decoded: &'b mut [u8],
    pub process_fn: ProcessPixelsFn,
}
impl Adapter for UncompressedAdapter<'_, '_> {
    fn direct(&mut self) {
        (self.process_fn)(PixelArgs(self.encoded, self.decoded))
    }

    fn map<Precision>(&mut self, from: Channels, to: Channels)
    where
        Precision: cast::Castable + cast::IntoNeBytes + Norm,
        [Precision; 1]: cast::IntoNeBytes,
        [Precision; 3]: cast::IntoNeBytes,
        [Precision; 4]: cast::IntoNeBytes,
    {
        // bytes per pixel
        let buffer_bpp = from.count() as usize * size_of::<Precision>();
        let decoded_bpp = to.count() as usize * size_of::<Precision>();
        debug_assert!(self.decoded.len() % decoded_bpp == 0);
        let pixels = self.decoded.len() / decoded_bpp;
        debug_assert!(self.encoded.len() % pixels == 0);
        let encoded_bpp = self.encoded.len() / pixels;

        // Allocate a small buffer on the stack and slice it to the correct size
        let mut buffer = [0_u8; 1024];
        let buffer_pixels = buffer.len() / buffer_bpp;
        let buffer = &mut buffer[..(buffer_pixels * buffer_bpp)];

        // convert the pixels in small chunks
        for (encoded, decoded) in self
            .encoded
            .chunks(buffer_pixels * encoded_bpp)
            .zip(self.decoded.chunks_mut(buffer_pixels * decoded_bpp))
        {
            let chunk_pixels = encoded.len() / encoded_bpp;
            debug_assert!(chunk_pixels == decoded.len() / decoded_bpp);

            let chunk_buffer = &mut buffer[..(chunk_pixels * buffer_bpp)];

            // decode pixels into buffer
            (self.process_fn)(PixelArgs(encoded, chunk_buffer));

            // convert pixels
            convert_channels_untyped::<Precision>(from, to, chunk_buffer, decoded);
        }
    }
}
