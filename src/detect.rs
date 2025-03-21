//! Internal module for detecting supported formats from DXGI, FourCC, and
//! DDS pixel formats.

use crate::header::{
    AlphaMode, Dx10Header, DxgiFormat, FourCC, MaskPixelFormat, PixelFormatFlags, RgbBitCount,
};
use crate::Format;

pub(crate) const fn special_cases(dx10: &Dx10Header) -> Option<Format> {
    if matches!(dx10.alpha_mode, AlphaMode::Premultiplied) {
        match dx10.dxgi_format {
            DxgiFormat::BC2_UNORM => return Some(Format::BC2_UNORM_PREMULTIPLIED_ALPHA),
            DxgiFormat::BC3_UNORM => return Some(Format::BC3_UNORM_PREMULTIPLIED_ALPHA),
            _ => {}
        }
    }

    None
}

pub(crate) const fn dxgi_format_to_supported(dxgi_format: DxgiFormat) -> Option<Format> {
    match dxgi_format {
        // uncompressed formats
        DxgiFormat::R8G8B8A8_TYPELESS
        | DxgiFormat::R8G8B8A8_UNORM
        | DxgiFormat::R8G8B8A8_UNORM_SRGB => Some(Format::R8G8B8A8_UNORM),
        DxgiFormat::R8G8B8A8_SNORM => Some(Format::R8G8B8A8_SNORM),
        DxgiFormat::B8G8R8A8_TYPELESS
        | DxgiFormat::B8G8R8A8_UNORM
        | DxgiFormat::B8G8R8A8_UNORM_SRGB => Some(Format::B8G8R8A8_UNORM),
        DxgiFormat::B8G8R8X8_TYPELESS
        | DxgiFormat::B8G8R8X8_UNORM
        | DxgiFormat::B8G8R8X8_UNORM_SRGB => Some(Format::B8G8R8X8_UNORM),
        DxgiFormat::B5G6R5_UNORM => Some(Format::B5G6R5_UNORM),
        DxgiFormat::B5G5R5A1_UNORM => Some(Format::B5G5R5A1_UNORM),
        DxgiFormat::B4G4R4A4_UNORM => Some(Format::B4G4R4A4_UNORM),
        DxgiFormat::A4B4G4R4_UNORM => Some(Format::A4B4G4R4_UNORM),
        DxgiFormat::R8_TYPELESS | DxgiFormat::R8_UNORM => Some(Format::R8_UNORM),
        DxgiFormat::R8_SNORM => Some(Format::R8_SNORM),
        DxgiFormat::R8G8_UNORM => Some(Format::R8G8_UNORM),
        DxgiFormat::R8G8_SNORM => Some(Format::R8G8_SNORM),
        DxgiFormat::A8_UNORM => Some(Format::A8_UNORM),
        DxgiFormat::R16_TYPELESS | DxgiFormat::R16_UNORM => Some(Format::R16_UNORM),
        DxgiFormat::R16_SNORM => Some(Format::R16_SNORM),
        DxgiFormat::R16_FLOAT => Some(Format::R16_FLOAT),
        DxgiFormat::R16G16_TYPELESS | DxgiFormat::R16G16_UNORM => Some(Format::R16G16_UNORM),
        DxgiFormat::R16G16_SNORM => Some(Format::R16G16_SNORM),
        DxgiFormat::R16G16_FLOAT => Some(Format::R16G16_FLOAT),
        DxgiFormat::R16G16B16A16_TYPELESS | DxgiFormat::R16G16B16A16_UNORM => {
            Some(Format::R16G16B16A16_UNORM)
        }
        DxgiFormat::R16G16B16A16_SNORM => Some(Format::R16G16B16A16_SNORM),
        DxgiFormat::R16G16B16A16_FLOAT => Some(Format::R16G16B16A16_FLOAT),
        DxgiFormat::R10G10B10A2_TYPELESS | DxgiFormat::R10G10B10A2_UNORM => {
            Some(Format::R10G10B10A2_UNORM)
        }
        DxgiFormat::R11G11B10_FLOAT => Some(Format::R11G11B10_FLOAT),
        DxgiFormat::R9G9B9E5_SHAREDEXP => Some(Format::R9G9B9E5_SHAREDEXP),
        DxgiFormat::R32_TYPELESS | DxgiFormat::R32_FLOAT => Some(Format::R32_FLOAT),
        DxgiFormat::R32G32_TYPELESS | DxgiFormat::R32G32_FLOAT => Some(Format::R32G32_FLOAT),
        DxgiFormat::R32G32B32_TYPELESS | DxgiFormat::R32G32B32_FLOAT => {
            Some(Format::R32G32B32_FLOAT)
        }
        DxgiFormat::R32G32B32A32_TYPELESS | DxgiFormat::R32G32B32A32_FLOAT => {
            Some(Format::R32G32B32A32_FLOAT)
        }
        DxgiFormat::R10G10B10_XR_BIAS_A2_UNORM => Some(Format::R10G10B10_XR_BIAS_A2_UNORM),
        DxgiFormat::AYUV => Some(Format::AYUV),
        DxgiFormat::Y410 => Some(Format::Y410),
        DxgiFormat::Y416 => Some(Format::Y416),

        // sub-sampled formats
        DxgiFormat::R8G8_B8G8_UNORM => Some(Format::R8G8_B8G8_UNORM),
        DxgiFormat::G8R8_G8B8_UNORM => Some(Format::G8R8_G8B8_UNORM),
        DxgiFormat::YUY2 => Some(Format::YUY2),
        DxgiFormat::Y210 => Some(Format::Y210),
        DxgiFormat::Y216 => Some(Format::Y216),
        DxgiFormat::R1_UNORM => Some(Format::R1_UNORM),

        // bi-planar formats
        DxgiFormat::NV12 => Some(Format::NV12),
        DxgiFormat::P010 => Some(Format::P010),
        DxgiFormat::P016 => Some(Format::P016),

        // block compression formats
        DxgiFormat::BC1_TYPELESS | DxgiFormat::BC1_UNORM | DxgiFormat::BC1_UNORM_SRGB => {
            Some(Format::BC1_UNORM)
        }
        DxgiFormat::BC2_TYPELESS | DxgiFormat::BC2_UNORM | DxgiFormat::BC2_UNORM_SRGB => {
            Some(Format::BC2_UNORM)
        }
        DxgiFormat::BC3_TYPELESS | DxgiFormat::BC3_UNORM | DxgiFormat::BC3_UNORM_SRGB => {
            Some(Format::BC3_UNORM)
        }
        DxgiFormat::BC4_TYPELESS | DxgiFormat::BC4_UNORM => Some(Format::BC4_UNORM),
        DxgiFormat::BC4_SNORM => Some(Format::BC4_SNORM),
        DxgiFormat::BC5_TYPELESS | DxgiFormat::BC5_UNORM => Some(Format::BC5_UNORM),
        DxgiFormat::BC5_SNORM => Some(Format::BC5_SNORM),
        DxgiFormat::BC6H_TYPELESS | DxgiFormat::BC6H_UF16 => Some(Format::BC6H_UF16),
        DxgiFormat::BC6H_SF16 => Some(Format::BC6H_SF16),
        DxgiFormat::BC7_TYPELESS | DxgiFormat::BC7_UNORM | DxgiFormat::BC7_UNORM_SRGB => {
            Some(Format::BC7_UNORM)
        }
        _ => None,
    }
}

pub(crate) const fn four_cc_to_dxgi(four_cc: FourCC) -> Option<DxgiFormat> {
    match four_cc {
        FourCC::DXT1 => Some(DxgiFormat::BC1_UNORM),
        FourCC::DXT3 => Some(DxgiFormat::BC2_UNORM),
        FourCC::DXT5 => Some(DxgiFormat::BC3_UNORM),

        FourCC::ATI1 => Some(DxgiFormat::BC4_UNORM),
        FourCC::BC4U => Some(DxgiFormat::BC4_UNORM),
        FourCC::BC4S => Some(DxgiFormat::BC4_SNORM),

        FourCC::ATI2 => Some(DxgiFormat::BC5_UNORM),
        FourCC::BC5U => Some(DxgiFormat::BC5_UNORM),
        FourCC::BC5S => Some(DxgiFormat::BC5_SNORM),

        FourCC::RGBG => Some(DxgiFormat::R8G8_B8G8_UNORM),
        FourCC::GRGB => Some(DxgiFormat::G8R8_G8B8_UNORM),

        FourCC::YUY2 => Some(DxgiFormat::YUY2),

        // Some old encoders use the FOURCC field to store D3DFORMAT constants:
        // https://learn.microsoft.com/en-us/windows/win32/direct3d9/d3dformat
        // See https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide
        // for more details.
        //
        // We can theoretically support most of them. However, testing them
        // is hard because there aren't many programs that produce them
        // (AFAIK).
        FourCC(36) => Some(DxgiFormat::R16G16B16A16_UNORM),
        FourCC(110) => Some(DxgiFormat::R16G16B16A16_SNORM),
        FourCC(111) => Some(DxgiFormat::R16_FLOAT),
        FourCC(112) => Some(DxgiFormat::R16G16_FLOAT),
        FourCC(113) => Some(DxgiFormat::R16G16B16A16_FLOAT),
        FourCC(114) => Some(DxgiFormat::R32_FLOAT),
        FourCC(115) => Some(DxgiFormat::R32G32_FLOAT),
        FourCC(116) => Some(DxgiFormat::R32G32B32A32_FLOAT),

        _ => None,
    }
}
pub(crate) const fn dxgi_to_four_cc(dxgi: DxgiFormat) -> Option<FourCC> {
    match dxgi {
        DxgiFormat::BC1_UNORM => Some(FourCC::DXT1),
        DxgiFormat::BC2_UNORM => Some(FourCC::DXT3),
        DxgiFormat::BC3_UNORM => Some(FourCC::DXT5),
        DxgiFormat::BC4_UNORM => Some(FourCC::BC4U),
        DxgiFormat::BC4_SNORM => Some(FourCC::BC4S),
        DxgiFormat::BC5_UNORM => Some(FourCC::BC5U),
        DxgiFormat::BC5_SNORM => Some(FourCC::BC5S),

        DxgiFormat::R8G8_B8G8_UNORM => Some(FourCC::RGBG),
        DxgiFormat::G8R8_G8B8_UNORM => Some(FourCC::GRGB),

        DxgiFormat::YUY2 => Some(FourCC::YUY2),

        // See `four_cc_to_dxgi`
        DxgiFormat::R16G16B16A16_UNORM => Some(FourCC(36)),
        DxgiFormat::R16G16B16A16_SNORM => Some(FourCC(110)),
        DxgiFormat::R16_FLOAT => Some(FourCC(111)),
        DxgiFormat::R16G16_FLOAT => Some(FourCC(112)),
        DxgiFormat::R16G16B16A16_FLOAT => Some(FourCC(113)),
        DxgiFormat::R32_FLOAT => Some(FourCC(114)),
        DxgiFormat::R32G32_FLOAT => Some(FourCC(115)),
        DxgiFormat::R32G32B32A32_FLOAT => Some(FourCC(116)),

        _ => None,
    }
}

pub(crate) const fn four_cc_to_supported(four_cc: FourCC) -> Option<Format> {
    // quick and easy, convert to DXGI first
    if let Some(dxgi_format) = four_cc_to_dxgi(four_cc) {
        return dxgi_format_to_supported(dxgi_format);
    }

    // now everything that doesn't have a DXGI format equivalent
    match four_cc {
        FourCC::DXT2 => Some(Format::BC2_UNORM_PREMULTIPLIED_ALPHA),
        FourCC::DXT4 => Some(Format::BC3_UNORM_PREMULTIPLIED_ALPHA),

        FourCC::RXGB => Some(Format::BC3_UNORM_RXGB),

        FourCC::UYVY => Some(Format::UYVY),

        _ => None,
    }
}

pub(crate) fn masked_to_supported(pf: &MaskPixelFormat) -> Option<Format> {
    KNOWN_PIXEL_FORMATS.iter().find_map(
        |(p, _, format)| {
            if p.matches(pf) {
                Some(*format)
            } else {
                None
            }
        },
    )
}
pub(crate) fn masked_to_dxgi(pf: &MaskPixelFormat) -> Option<DxgiFormat> {
    KNOWN_PIXEL_FORMATS
        .iter()
        .find_map(|(p, dxgi, _)| if p.matches(pf) { *dxgi } else { None })
}
pub(crate) fn dxgi_to_masked(dxgi_format: DxgiFormat) -> Option<MaskPixelFormat> {
    KNOWN_PIXEL_FORMATS.iter().find_map(|(p, dxgi, _)| {
        if *dxgi == Some(dxgi_format) {
            Some(p.to_masked())
        } else {
            None
        }
    })
}
pub(crate) fn supported_to_masked(format: Format) -> Option<MaskPixelFormat> {
    KNOWN_PIXEL_FORMATS.iter().find_map(|(p, _, f)| {
        if *f == format {
            Some(p.to_masked())
        } else {
            None
        }
    })
}

struct PFPattern {
    flags: PixelFormatFlags,
    rgb_bit_count: RgbBitCount,
    r_bit_mask: u32,
    g_bit_mask: u32,
    b_bit_mask: u32,
    a_bit_mask: u32,
}
impl PFPattern {
    fn matches(&self, pf: &MaskPixelFormat) -> bool {
        pf.flags == self.flags
            && pf.rgb_bit_count == self.rgb_bit_count
            && pf.r_bit_mask == self.r_bit_mask
            && pf.g_bit_mask == self.g_bit_mask
            && pf.b_bit_mask == self.b_bit_mask
            && pf.a_bit_mask == self.a_bit_mask
    }
    const fn with_flags(mut self, flags: PixelFormatFlags) -> Self {
        self.flags = flags;
        self
    }
    fn to_masked(&self) -> MaskPixelFormat {
        MaskPixelFormat {
            flags: self.flags,
            rgb_bit_count: self.rgb_bit_count,
            r_bit_mask: self.r_bit_mask,
            g_bit_mask: self.g_bit_mask,
            b_bit_mask: self.b_bit_mask,
            a_bit_mask: self.a_bit_mask,
        }
    }
}
const KNOWN_PIXEL_FORMATS: &[(PFPattern, Option<DxgiFormat>, Format)] = {
    const fn parse_bit_count(bit_count: u32) -> RgbBitCount {
        match bit_count {
            8 => RgbBitCount::Count8,
            16 => RgbBitCount::Count16,
            24 => RgbBitCount::Count24,
            32 => RgbBitCount::Count32,
            _ => panic!("Invalid bit count"),
        }
    }
    const fn alpha_only(bit_count: u32, a_mask: u32) -> PFPattern {
        PFPattern {
            flags: PixelFormatFlags::ALPHA,
            rgb_bit_count: parse_bit_count(bit_count),
            r_bit_mask: 0,
            g_bit_mask: 0,
            b_bit_mask: 0,
            a_bit_mask: a_mask,
        }
    }
    const fn grayscale(bit_count: u32, r_mask: u32) -> PFPattern {
        PFPattern {
            flags: PixelFormatFlags::LUMINANCE,
            rgb_bit_count: parse_bit_count(bit_count),
            r_bit_mask: r_mask,
            g_bit_mask: 0,
            b_bit_mask: 0,
            a_bit_mask: 0,
        }
    }
    const fn rgb(bit_count: u32, r_mask: u32, g_mask: u32, b_mask: u32) -> PFPattern {
        PFPattern {
            flags: PixelFormatFlags::RGB,
            rgb_bit_count: parse_bit_count(bit_count),
            r_bit_mask: r_mask,
            g_bit_mask: g_mask,
            b_bit_mask: b_mask,
            a_bit_mask: 0,
        }
    }
    const fn rgba(bit_count: u32, r_mask: u32, g_mask: u32, b_mask: u32, a_mask: u32) -> PFPattern {
        PFPattern {
            flags: PixelFormatFlags::RGBA,
            rgb_bit_count: parse_bit_count(bit_count),
            r_bit_mask: r_mask,
            g_bit_mask: g_mask,
            b_bit_mask: b_mask,
            a_bit_mask: a_mask,
        }
    }
    const fn snorm(
        bit_count: u32,
        r_mask: u32,
        g_mask: u32,
        b_mask: u32,
        a_mask: u32,
    ) -> PFPattern {
        PFPattern {
            flags: PixelFormatFlags::BUMP_DUDV,
            rgb_bit_count: parse_bit_count(bit_count),
            r_bit_mask: r_mask,
            g_bit_mask: g_mask,
            b_bit_mask: b_mask,
            a_bit_mask: a_mask,
        }
    }

    let rgb_luminance = PixelFormatFlags::RGB.union(PixelFormatFlags::LUMINANCE);

    use Format::*;

    &[
        // alpha
        (alpha_only(8, 0xFF), Some(DxgiFormat::A8_UNORM), A8_UNORM),
        // grayscale
        (grayscale(8, 0xFF), Some(DxgiFormat::R8_UNORM), R8_UNORM),
        (
            grayscale(8, 0xFF).with_flags(rgb_luminance),
            Some(DxgiFormat::R8_UNORM),
            R8_UNORM,
        ),
        (
            grayscale(16, 0xFFFF),
            Some(DxgiFormat::R16_UNORM),
            R16_UNORM,
        ),
        // rgb
        (
            rgb(16, 0xF800, 0x07E0, 0x001F),
            Some(DxgiFormat::B5G6R5_UNORM),
            B5G6R5_UNORM,
        ),
        (
            rgb(32, 0xFF0000, 0xFF00, 0xFF),
            Some(DxgiFormat::B8G8R8X8_UNORM),
            B8G8R8X8_UNORM,
        ),
        (
            rgb(32, 0xFFFF, 0xFFFF0000, 0),
            Some(DxgiFormat::R16G16_UNORM),
            R16G16_UNORM,
        ),
        (
            rgb(16, 0xFF, 0xFF00, 0),
            Some(DxgiFormat::R8G8_UNORM),
            R8G8_UNORM,
        ),
        (rgb(24, 0xFF0000, 0xFF00, 0xFF), None, B8G8R8_UNORM),
        (rgb(24, 0xFF, 0xFF00, 0xFF0000), None, R8G8B8_UNORM),
        // rgba
        (
            rgba(16, 0xF00, 0xF0, 0xF, 0xF000),
            Some(DxgiFormat::B4G4R4A4_UNORM),
            B4G4R4A4_UNORM,
        ),
        (
            rgba(16, 0x7C00, 0x3E0, 0x1F, 0x8000),
            Some(DxgiFormat::B5G5R5A1_UNORM),
            B5G5R5A1_UNORM,
        ),
        (
            rgba(32, 0xFF0000, 0xFF00, 0xFF, 0xFF000000),
            Some(DxgiFormat::B8G8R8A8_UNORM),
            B8G8R8A8_UNORM,
        ),
        (
            rgba(32, 0xFF, 0xFF00, 0xFF0000, 0xFF000000),
            Some(DxgiFormat::R8G8B8A8_UNORM),
            R8G8B8A8_UNORM,
        ),
        (
            rgba(32, 0x3FF00000, 0xFFC00, 0x3FF, 0xC0000000),
            Some(DxgiFormat::R10G10B10A2_UNORM),
            R10G10B10A2_UNORM,
        ),
        // snorm
        (
            snorm(32, 0xFF, 0xFF00, 0xFF0000, 0xFF000000),
            Some(DxgiFormat::R8G8B8A8_SNORM),
            R8G8B8A8_SNORM,
        ),
        (
            snorm(16, 0xFF, 0xFF00, 0, 0),
            Some(DxgiFormat::R8G8_SNORM),
            R8G8_SNORM,
        ),
        (
            snorm(32, 0xFFFF, 0xFFFF0000, 0, 0),
            Some(DxgiFormat::R16G16_SNORM),
            R16G16_SNORM,
        ),
        // special
        (
            // I have no idea why, but LUMINANCE + ALPHAPIXELS is used for R8G8_UNORM
            PFPattern {
                flags: PixelFormatFlags::LUMINANCE_ALPHA,
                rgb_bit_count: RgbBitCount::Count16,
                r_bit_mask: 0xFF,
                g_bit_mask: 0,
                b_bit_mask: 0,
                a_bit_mask: 0xFF00,
            },
            Some(DxgiFormat::R8G8_UNORM),
            R8G8_UNORM,
        ),
    ]
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxgi_four_cc_round_trip() {
        // DXGI -> Four CC -> DXGI
        for dxgi in DxgiFormat::all() {
            if let Some(four_cc) = dxgi_to_four_cc(dxgi) {
                assert_eq!(Some(dxgi), four_cc_to_dxgi(four_cc));
            }
        }
    }
}
