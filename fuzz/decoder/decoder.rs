#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Just no panic
    let reader = Cursor::new(data);
    let options = dds::header::ParseOptions::new_permissive(None);
    if let Ok(mut decoder) = dds::Decoder::new_with_options(reader, &options) {
        let color = decoder.native_color();
        let size = decoder.main_size();
        let layout = decoder.layout();
        let is_cube_map = layout.is_cube_map();

        if size.width > 312 || size.height > 312 {
            return;
        }

        let mut buf =
            vec![0u8; size.pixels().saturating_mul(color.bytes_per_pixel() as u64) as usize];
        let image = dds::ImageViewMut::new(&mut buf, size, color).expect("Invalid buffer length");

        if is_cube_map {
            _ = decoder.read_cube_map(image);
        } else {
            _ = decoder.read_surface(image);
        }
    }
});
