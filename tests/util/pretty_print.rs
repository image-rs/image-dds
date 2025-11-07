use dds::{header::*, *};

pub fn pretty_print_header(out: &mut String, header: &Header) {
    out.push_str("Header:\n");
    if let Some(d) = header.depth() {
        out.push_str(&format!(
            "    w/h/d: {:?} x {:?} x {:?}\n",
            header.width(),
            header.height(),
            d
        ));
    } else {
        out.push_str(&format!(
            "    w/h: {:?} x {:?}\n",
            header.width(),
            header.height()
        ));
    }
    out.push_str(&format!("    mipmap_count: {:?}\n", header.mipmap_count()));
    match header {
        Header::Dx9(dx9) => {
            if !dx9.caps2.is_empty() {
                out.push_str(&format!("    caps2: {:?}\n", dx9.caps2));
            }

            match &dx9.pixel_format {
                Dx9PixelFormat::FourCC(four_cc) => {
                    out.push_str(&format!("    format: {four_cc:?}\n"));
                }
                Dx9PixelFormat::Mask(pixel_format) => {
                    out.push_str("    format: masked\n");
                    out.push_str(&format!("        flags: {:?}\n", pixel_format.flags));
                    out.push_str(&format!(
                        "        rgb_bit_count: {:?}\n",
                        pixel_format.rgb_bit_count as u32
                    ));
                    out.push_str(&format!(
                        "        bit_mask: r:0x{:x} g:0x{:x} b:0x{:x} a:0x{:x}\n",
                        pixel_format.r_bit_mask,
                        pixel_format.g_bit_mask,
                        pixel_format.b_bit_mask,
                        pixel_format.a_bit_mask
                    ));
                }
            }
        }
        Header::Dx10(dx10) => {
            out.push_str(&format!("    DX10: {:?}\n", dx10.resource_dimension));
            out.push_str(&format!("        dxgi_format: {:?}\n", dx10.dxgi_format));
            if !dx10.misc_flag.is_empty() {
                out.push_str(&format!("        misc_flag: {:?}\n", dx10.misc_flag));
            }
            if dx10.array_size != 1 {
                out.push_str(&format!("        array_size: {:?}\n", dx10.array_size));
            }
            if dx10.alpha_mode != AlphaMode::Unknown {
                out.push_str(&format!("        alpha_mode: {:?}\n", dx10.alpha_mode));
            }
        }
    };
}

pub fn pretty_print_raw_header(out: &mut String, raw: &RawHeader) {
    out.push_str("Raw Header:\n");

    if raw.size != 124 {
        out.push_str(&format!("    size: {:?}\n", raw.size));
    }
    out.push_str(&format!("    flags: {:?}\n", raw.flags));

    if raw.flags.contains(DdsFlags::DEPTH) {
        out.push_str(&format!(
            "    w/h/d: {:?} x {:?} x {:?}\n",
            raw.width, raw.height, raw.depth
        ));
    } else {
        out.push_str(&format!(
            "    w/h: {:?} x {:?} (x {:?})\n",
            raw.width, raw.height, raw.depth
        ));
    }

    let size = raw.pitch_or_linear_size;
    if raw.flags.contains(DdsFlags::PITCH) && !raw.flags.contains(DdsFlags::LINEAR_SIZE) {
        out.push_str(&format!("    pitch: {size:?}\n"));
    } else if !raw.flags.contains(DdsFlags::PITCH) && raw.flags.contains(DdsFlags::LINEAR_SIZE) {
        out.push_str(&format!("    linear_size: {size:?}\n"));
    } else {
        out.push_str(&format!("    pitch_or_linear_size: {size:?}\n"));
    }

    out.push_str(&format!("    mipmap_count: {:?}", raw.mipmap_count));
    if !raw.flags.contains(DdsFlags::MIPMAP_COUNT) {
        out.push_str("  (not specified)");
    }
    out.push('\n');

    if raw.reserved1.iter().any(|&x| x != 0) {
        out.push_str("    reserved1:\n");
        let zero_prefix = raw.reserved1.iter().take_while(|&&x| x == 0).count();
        if zero_prefix > 0 {
            out.push_str(&format!("        0..={}: 0\n", zero_prefix - 1));
        }
        for i in zero_prefix..raw.reserved1.len() {
            out.push_str(&format!("           {i:>2}: "));

            let n = raw.reserved1[i];
            let bytes = n.to_le_bytes();

            if bytes.iter().all(|x| x.is_ascii_alphanumeric()) {
                for byte in bytes {
                    out.push(byte as char);
                }
                out.push_str(" (ASCII)");
            } else {
                out.push_str(&format!("{n:#010X} {n}"));
            }

            out.push('\n');
        }
    }

    if raw.pixel_format.flags == PixelFormatFlags::FOURCC
        && raw.pixel_format.rgb_bit_count == 0
        && raw.pixel_format.r_bit_mask == 0
        && raw.pixel_format.g_bit_mask == 0
        && raw.pixel_format.b_bit_mask == 0
        && raw.pixel_format.a_bit_mask == 0
    {
        out.push_str(&format!(
            "    pixel_format: {:?}\n",
            raw.pixel_format.four_cc
        ));
    } else {
        out.push_str("    pixel_format:\n");
        out.push_str(&format!("        flags: {:?}\n", raw.pixel_format.flags));
        if raw.pixel_format.four_cc != FourCC::NONE {
            out.push_str(&format!(
                "        four_cc: {:?}\n",
                raw.pixel_format.four_cc
            ));
        }
        out.push_str(&format!(
            "        rgb_bit_count: {:?}\n",
            raw.pixel_format.rgb_bit_count
        ));
        out.push_str(&format!(
            "        bit_mask: r:0x{:x} g:0x{:x} b:0x{:x} a:0x{:x}\n",
            raw.pixel_format.r_bit_mask,
            raw.pixel_format.g_bit_mask,
            raw.pixel_format.b_bit_mask,
            raw.pixel_format.a_bit_mask
        ));
    }

    out.push_str(&format!("    caps: {:?}", raw.caps));
    if !raw.flags.contains(DdsFlags::CAPS) {
        out.push_str("  (not specified)");
    }
    out.push('\n');

    out.push_str(&format!("    caps2: {:?}\n", raw.caps2));
    if raw.caps3 != 0 || raw.caps4 != 0 || raw.reserved2 != 0 {
        out.push_str(&format!("    caps3: {:?}\n", raw.caps3));
        out.push_str(&format!("    caps4: {:?}\n", raw.caps4));
        out.push_str(&format!("    reserved2: {:?}\n", raw.reserved2));
    }

    if let Some(dx10) = &raw.dx10 {
        out.push_str("    DX10:\n");

        out.push_str("        dxgi_format: ");
        if let Ok(dxgi) = DxgiFormat::try_from(dx10.dxgi_format) {
            out.push_str(&format!("{dxgi:?}"));
        } else {
            out.push_str(&format!("{:?}", dx10.dxgi_format));
        }
        out.push('\n');

        out.push_str("        resource_dimension: ");
        if let Ok(dim) = ResourceDimension::try_from(dx10.resource_dimension) {
            out.push_str(&format!("{dim:?}"));
        } else {
            out.push_str(&format!("{:?}", dx10.resource_dimension));
        }
        out.push('\n');

        out.push_str(&format!("        misc_flag: {:?}\n", dx10.misc_flag));
        out.push_str(&format!("        array_size: {:?}\n", dx10.array_size));
        out.push_str(&format!("        misc_flags2: {:?}\n", dx10.misc_flags2));
    }
}

pub fn pretty_print_data_layout(out: &mut String, layout: &DataLayout) {
    out.push_str("Layout: ");
    let pixels = layout.pixel_info();
    match layout {
        DataLayout::Texture(texture) => {
            out.push_str(&format!(
                "Texture ({} bytes @ {:?})\n",
                texture.data_len(),
                pixels
            ));
            for (i, surface) in texture.iter_mips().enumerate() {
                out.push_str(&format!(
                    "    Surface[{i}] {}x{} ({} bytes)\n",
                    surface.width(),
                    surface.height(),
                    surface.data_len()
                ));
            }
        }
        DataLayout::Volume(volume) => {
            out.push_str(&format!(
                "Volume ({} bytes @ {:?})\n",
                volume.data_len(),
                pixels
            ));
            for (i, volume) in volume.iter_mips().enumerate() {
                out.push_str(&format!(
                    "    Volume[{i}] {}x{}x{} ({} bytes)\n",
                    volume.width(),
                    volume.height(),
                    volume.depth(),
                    volume.data_len()
                ));
                for (i, surface) in volume.iter_depth_slices().enumerate() {
                    out.push_str(&format!(
                        "        Surface[{i}] {}x{} ({} bytes)\n",
                        surface.width(),
                        surface.height(),
                        surface.data_len()
                    ));
                }
            }
        }
        DataLayout::TextureArray(texture_array) => {
            out.push_str(&format!(
                "TextureArray len:{} kind:{:?} ({} bytes @ {:?})\n",
                texture_array.len(),
                texture_array.kind(),
                texture_array.data_len(),
                pixels
            ));
            for (i, texture) in texture_array.iter().enumerate() {
                out.push_str(&format!(
                    "    Texture[{i}] ({} bytes)\n",
                    texture.data_len()
                ));
                for (i, surface) in texture.iter_mips().enumerate() {
                    out.push_str(&format!(
                        "        Surface[{i}] {}x{} ({} bytes)\n",
                        surface.width(),
                        surface.height(),
                        surface.data_len()
                    ));
                }
            }
        }
    }
}
