pub struct Bc1Decoder;

impl Bc1Decoder {
    pub fn decode_block(&self, data: &[u8; 8]) {
        let color0 = u16::from_le_bytes([data[0], data[1]]);
        let color1 = u16::from_le_bytes([data[2], data[3]]);
        let indexes = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    }
}
