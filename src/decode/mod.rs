mod bc1;

pub(crate) trait DecodeUncompressed<const CHANNELS: usize> {
    type DecodedPixel;
    type NativeType: Normalized;

    fn decode_pixel_native(&self, input: Self::DecodedPixel) -> [Self::NativeType; CHANNELS];

    fn decode_pixel_u8(&self, input: Self::DecodedPixel) -> [u8; CHANNELS] {
        self.decode_pixel_native(input).map(Normalized::to_u8)
    }
    fn decode_pixel_u16(&self, input: Self::DecodedPixel) -> [u16; CHANNELS] {
        self.decode_pixel_native(input).map(Normalized::to_u16)
    }
    fn decode_pixel_f32(&self, input: Self::DecodedPixel) -> [f32; CHANNELS] {
        self.decode_pixel_native(input).map(Normalized::to_f32)
    }
}

pub(crate) trait Normalized {
    fn to_u8(self) -> u8;
    fn to_u16(self) -> u16;
    fn to_f32(self) -> f32;
}

impl Normalized for u8 {
    fn to_u8(self) -> u8 {
        self
    }

    fn to_u16(self) -> u16 {
        // TODO: This is not correct.
        (self as u16) << 8 | self as u16
    }

    fn to_f32(self) -> f32 {
        let factor = 1.0 / Self::MAX as f32;
        self as f32 * factor
    }
}

impl Normalized for u16 {
    fn to_u8(self) -> u8 {
        // TODO: This is not correct.
        (self >> 8) as u8
    }

    fn to_u16(self) -> u16 {
        self
    }

    fn to_f32(self) -> f32 {
        let factor = 1.0 / Self::MAX as f32;
        self as f32 * factor
    }
}

impl Normalized for f32 {
    fn to_u8(self) -> u8 {
        (self * u8::MAX as f32 + 0.5) as u8
    }

    fn to_u16(self) -> u16 {
        (self * u16::MAX as f32 + 0.5) as u16
    }

    fn to_f32(self) -> f32 {
        self
    }
}
