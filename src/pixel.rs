use crate::{util::div_ceil, DecodeError, DxgiFormat, Header, Size, SupportedFormat};

/// This describes the number of bits per pixel and the layout of pixels within
/// a surface.
///
/// DDS supports a variety of pixel formats, including
///
/// - simple uncompressed (but quantized) formats like `R8G8B8A8_UNORM` and
///   `B5G6R5_UNORM`,
/// - block-compressed formats like `BC1_UNORM` (DXT1),
/// - chroma sub-sampled formats like `NV12` which store the luma and chroma in
///   different planes,
/// - and more.
///
/// The main purpose of this enum is to provide a way to calculate the byte size
/// of a surface to create the [`DataLayout`] of a DDS file.
///
/// ## Unsupported pixel formats
///
/// Note that [`PixelInfo`] can even describe pixel formats `ddsd` does not
/// support. This is by design to allow users to get the data layout for DDS
/// files this library doesn't fully support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelInfo {
    /// Each pixel has a fixed number of bytes, regardless of the dimensions of
    /// the surface.
    Fixed { bytes_per_pixel: u8 },
    /// Pixels are grouped into blocks of constant byte size.
    ///
    /// This is used for block-compressed and channel-packed sub-sampled pixel
    /// formats. E.g. `BC1_UNORM` (`DXT1`) has 8 bytes per 4x4 block, and
    /// `R8G8B8G8_UNORM` has 4 bytes per pair (2x1 block).
    Block {
        bytes_per_block: u8,
        block_size: (u8, u8),
    },
    /// Pixels are stored as sub-sampled YUV (or YCbCr) samples in separate
    /// planes.
    ///
    /// Separate planes means that the Y and U/V components for one pixel are
    /// stored in separate arrays. One example of this is the
    /// [`NV12` format](https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering#nv12).
    /// This an 8-bit YUV 4:2:0 (meaning that U and V are sub-sampled to half
    /// width and height) with the following memory layout:
    ///
    /// ```text
    /// ---> Memory addresses increase (each cell is 8 bit) -->
    /// Y0 | Y1 | Y2 | Y3 | ... | Y(w*h-1) | U0 | V0 | U1 | V1 | ... | U(w/2*h/2-1) | V(w/2*h/2-1)
    /// \---------------------------------/ \----------------------------------------------------/
    ///              Y samples                                 U and V samples
    ///             (w*h bytes)                                 (w*h/2 bytes)
    /// ```
    ///
    /// The Y samples are stored in one plane, while the U and V samples are
    /// channel-packed in a second plane.
    ChromaSubSampled {
        bytes_per_sample: u8,
        sub_sampling: (u8, u8),
    },
}
impl PixelInfo {
    pub fn fixed(bytes_per_pixel: u8) -> Self {
        Self::Fixed { bytes_per_pixel }
    }
    pub fn block(bytes_per_block: u8, block_size: (u8, u8)) -> Self {
        Self::Block {
            bytes_per_block,
            block_size,
        }
    }
    pub fn chroma(bytes_per_sample: u8, sub_sampling: (u8, u8)) -> Self {
        Self::ChromaSubSampled {
            bytes_per_sample,
            sub_sampling,
        }
    }

    pub fn from_header(header: &Header) -> Result<Self, DecodeError> {
        if let Some(dx10) = &header.dxt10 {
            dx10.dxgi_format
                .try_into()
                .map_err(|_| DecodeError::UnsupportedDxgiFormat(dx10.dxgi_format))
        } else if let Some(four_cc) = header.pixel_format.four_cc {
            SupportedFormat::from_four_cc(four_cc)
                .map(Into::into)
                .ok_or(DecodeError::UnsupportedFourCC(four_cc))
        } else {
            let bit_count = header.pixel_format.rgb_bit_count;
            if bit_count > 0 && bit_count <= 32 && bit_count % 8 == 0 {
                Ok(PixelInfo::fixed((bit_count / 8) as u8))
            } else {
                Err(DecodeError::UnsupportedPixelFormat)
            }
        }
    }

    /// Returns the number of bits per pixel.
    ///
    /// If the number of bits per pixel is not an integer, the result is rounded
    /// up to the nearest integer.
    pub fn bits_per_pixel(&self) -> u32 {
        match *self {
            Self::Fixed { bytes_per_pixel } => bytes_per_pixel as u32 * 8,
            Self::Block {
                bytes_per_block,
                block_size,
            } => {
                let bits_per_block = bytes_per_block as u32 * 8;
                let pixels_per_block = block_size.0 as u32 * block_size.1 as u32;
                div_ceil(bits_per_block, pixels_per_block)
            }
            Self::ChromaSubSampled {
                bytes_per_sample,
                sub_sampling,
            } => {
                let bits_per_sample = bytes_per_sample as u32 * 8;
                let sub_sampling = sub_sampling.0 as u32 * sub_sampling.1 as u32;
                bits_per_sample + div_ceil(2 * bits_per_sample, sub_sampling)
            }
        }
    }

    /// Returns the number of bytes a surface with the given dimensions takes
    /// up in the data section of a DDS file.
    ///
    /// If an overflow occurs, `None` is returned. This typically happens when
    /// the surface is unrealistically large, hinting at a modified, corrupted,
    /// or otherwise invalid DDS file.
    pub fn surface_bytes(&self, size: Size) -> Option<u64> {
        match *self {
            Self::Fixed { bytes_per_pixel } => size.pixels().checked_mul(bytes_per_pixel as u64),
            Self::Block {
                bytes_per_block,
                block_size,
            } => {
                let blocks_x = div_ceil(size.width, block_size.0 as u32);
                let blocks_y = div_ceil(size.height, block_size.1 as u32);
                // This cannot overflow, because both factors are u32.
                let blocks = blocks_x as u64 * blocks_y as u64;
                blocks.checked_mul(bytes_per_block as u64)
            }
            Self::ChromaSubSampled {
                bytes_per_sample,
                sub_sampling,
            } => {
                let chroma_x = div_ceil(size.width, sub_sampling.0 as u32);
                let chroma_y = div_ceil(size.height, sub_sampling.1 as u32);
                // This cannot overflow, because both factors are u32.
                let samples_chroma = chroma_x as u64 * chroma_y as u64;
                let samples = size.pixels().checked_add(samples_chroma.checked_mul(2)?)?;
                samples.checked_mul(bytes_per_sample as u64)
            }
        }
    }
}

impl From<SupportedFormat> for PixelInfo {
    fn from(value: SupportedFormat) -> Self {
        use SupportedFormat as F;

        match value {
            // 1 bytes per pixel
            F::R8_UNORM | F::R8_SNORM | F::A8_UNORM => Self::fixed(1),
            // 2 bytes per pixel
            F::B5G6R5_UNORM
            | F::B5G5R5A1_UNORM
            | F::B4G4R4A4_UNORM
            | F::R8G8_UNORM
            | F::R8G8_SNORM
            | F::R16_UNORM
            | F::R16_SNORM
            | F::R16_FLOAT => Self::fixed(2),
            // 3 bytes per pixel
            F::R8G8B8_UNORM | F::B8G8R8_UNORM => Self::fixed(3),
            // 4 bytes per pixel
            F::R8G8B8A8_UNORM
            | F::R8G8B8A8_SNORM
            | F::B8G8R8A8_UNORM
            | F::B8G8R8X8_UNORM
            | F::R16G16_UNORM
            | F::R16G16_SNORM
            | F::R10G10B10A2_UNORM
            | F::R11G11B10_FLOAT
            | F::R9G9B9E5_SHAREDEXP
            | F::R16G16_FLOAT
            | F::R32_FLOAT
            | F::R10G10B10_XR_BIAS_A2_UNORM => Self::fixed(4),
            // 8 bytes per pixel
            F::R16G16B16A16_UNORM
            | F::R16G16B16A16_SNORM
            | F::R16G16B16A16_FLOAT
            | F::R32G32_FLOAT => Self::fixed(8),
            // 12 bytes per pixel
            F::R32G32B32_FLOAT => Self::fixed(12),
            // 16 bytes per pixel
            F::R32G32B32A32_FLOAT => Self::fixed(16),

            // sub-sampled formats
            // 4 bytes per one 2x1 block
            F::R8G8_B8G8_UNORM | F::G8R8_G8B8_UNORM => Self::block(4, (2, 1)),

            // block compression formats
            // 8 bytes per one 4x4 block
            F::BC1_UNORM | F::BC4_UNORM | F::BC4_SNORM => Self::block(8, (4, 4)),
            // 16 bytes per one 4x4 block
            F::BC2_UNORM
            | F::BC3_UNORM
            | F::BC5_UNORM
            | F::BC5_SNORM
            | F::BC6H_SF16
            | F::BC6H_UF16
            | F::BC7_UNORM => Self::block(16, (4, 4)),
        }
    }
}

impl TryFrom<DxgiFormat> for PixelInfo {
    type Error = ();

    fn try_from(value: DxgiFormat) -> Result<Self, Self::Error> {
        use DxgiFormat as F;

        match value {
            // Fixed

            // 1 bytes per pixel
            F::R8_TYPELESS | F::R8_UNORM | F::R8_UINT | F::R8_SNORM | F::R8_SINT | F::A8_UNORM => {
                Ok(Self::fixed(1))
            }
            // 2 bytes per pixel
            F::R8G8_TYPELESS
            | F::R8G8_UNORM
            | F::R8G8_UINT
            | F::R8G8_SNORM
            | F::R8G8_SINT
            | F::R16_TYPELESS
            | F::R16_FLOAT
            | F::D16_UNORM
            | F::R16_UNORM
            | F::R16_UINT
            | F::R16_SNORM
            | F::R16_SINT
            | F::B5G6R5_UNORM
            | F::B5G5R5A1_UNORM
            | F::B4G4R4A4_UNORM => Ok(Self::fixed(2)),
            // 4 bytes per pixel
            F::R10G10B10A2_TYPELESS
            | F::R10G10B10A2_UNORM
            | F::R10G10B10A2_UINT
            | F::R11G11B10_FLOAT
            | F::R8G8B8A8_TYPELESS
            | F::R8G8B8A8_UNORM
            | F::R8G8B8A8_UNORM_SRGB
            | F::R8G8B8A8_UINT
            | F::R8G8B8A8_SNORM
            | F::R8G8B8A8_SINT
            | F::R16G16_TYPELESS
            | F::R16G16_FLOAT
            | F::R16G16_UNORM
            | F::R16G16_UINT
            | F::R16G16_SNORM
            | F::R16G16_SINT
            | F::R32_TYPELESS
            | F::D32_FLOAT
            | F::R32_FLOAT
            | F::R32_UINT
            | F::R32_SINT
            | F::R24G8_TYPELESS
            | F::D24_UNORM_S8_UINT
            | F::R24_UNORM_X8_TYPELESS
            | F::X24_TYPELESS_G8_UINT
            | F::R9G9B9E5_SHAREDEXP
            | F::B8G8R8A8_UNORM
            | F::B8G8R8X8_UNORM
            | F::R10G10B10_XR_BIAS_A2_UNORM
            | F::B8G8R8A8_TYPELESS
            | F::B8G8R8A8_UNORM_SRGB
            | F::B8G8R8X8_TYPELESS
            | F::B8G8R8X8_UNORM_SRGB
            | F::AYUV
            | F::Y410
            | F::V408 => Ok(Self::fixed(4)),
            // 8 bytes per pixel
            F::R16G16B16A16_TYPELESS
            | F::R16G16B16A16_FLOAT
            | F::R16G16B16A16_UNORM
            | F::R16G16B16A16_UINT
            | F::R16G16B16A16_SNORM
            | F::R16G16B16A16_SINT
            | F::R32G32_TYPELESS
            | F::R32G32_FLOAT
            | F::R32G32_UINT
            | F::R32G32_SINT
            | F::R32G8X24_TYPELESS
            | F::D32_FLOAT_S8X24_UINT
            | F::R32_FLOAT_X8X24_TYPELESS
            | F::X32_TYPELESS_G8X24_UINT
            | F::Y416 => Ok(Self::fixed(8)),
            // 12 bytes per pixel
            F::R32G32B32_TYPELESS | F::R32G32B32_FLOAT | F::R32G32B32_UINT | F::R32G32B32_SINT => {
                Ok(Self::fixed(12))
            }
            // 16 bytes per pixel
            F::R32G32B32A32_TYPELESS
            | F::R32G32B32A32_FLOAT
            | F::R32G32B32A32_UINT
            | F::R32G32B32A32_SINT => Ok(Self::fixed(16)),

            // Special:
            // R1_UNORM is technically a fixed bits-per-pixel format, but since
            // it's only 1 bit per pixel, it effectively behaves like a 8x1
            // block format with 1 byte per block.
            // Note: I couldn't find any official documentation on this format,
            // and guessed the memory layout based on vibes.
            // TODO: Verify this.
            F::R1_UNORM => Ok(Self::block(1, (8, 1))),

            // Sub-sampled 2x1
            F::R8G8_B8G8_UNORM | F::G8R8_G8B8_UNORM => Ok(Self::block(4, (2, 1))),
            // YUV 4:2:2 formats
            // https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering#yuy2
            // https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats#y216-and-y210
            F::YUY2 => Ok(Self::block(4, (2, 1))),
            F::Y210 | F::Y216 => Ok(Self::block(8, (2, 1))),

            // Block compression formats
            // 8 bytes per 4x4 block
            F::BC1_TYPELESS
            | F::BC1_UNORM
            | F::BC1_UNORM_SRGB
            | F::BC4_TYPELESS
            | F::BC4_UNORM
            | F::BC4_SNORM => Ok(Self::block(8, (4, 4))),
            // 16 bytes per 4x4 block
            F::BC2_TYPELESS
            | F::BC2_UNORM
            | F::BC2_UNORM_SRGB
            | F::BC3_TYPELESS
            | F::BC3_UNORM
            | F::BC3_UNORM_SRGB
            | F::BC5_TYPELESS
            | F::BC5_UNORM
            | F::BC5_SNORM
            | F::BC6H_TYPELESS
            | F::BC6H_UF16
            | F::BC6H_SF16
            | F::BC7_TYPELESS
            | F::BC7_UNORM
            | F::BC7_UNORM_SRGB => Ok(Self::block(16, (4, 4))),

            // (4:2:0) bytes = w*h + 2 * ceil(w/2)*ceil(h/2)
            F::NV12 | F::OPAQUE_420 => Ok(Self::chroma(1, (2, 2))),
            // (4:2:0) bytes = 2*(w*h + 2 * ceil(w/2)*ceil(h/2))
            F::P010 | F::P016 => Ok(Self::chroma(2, (2, 2))),
            // (4:1:1) bytes = w*h + 2 * ceil(w/4)*h
            F::NV11 => Ok(Self::chroma(1, (4, 1))),
            // (4:2:2) bytes = w*h + 2 * ceil(w/2)*h
            F::P208 => Ok(Self::chroma(1, (2, 1))),

            // Palette formats
            F::AI44 | F::IA44 | F::P8 => Ok(Self::fixed(1)),
            F::A8P8 => Ok(Self::fixed(2)),

            // TODO: I couldn't find anything about the memory layout of this.
            // F::V208 => Err(()),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn bits_per_pixel() {
        // This test verifies that `bits_per_pixel` is implemented correctly,
        // using `surface_bytes` as the reference implementation. This is a
        // reasonable reference, since `surface_bytes` is foundational to this
        // library and tested implicitly in all tests reading DDS files.
        for i in 0..256_u32 {
            if let Some(pixel) = DxgiFormat::try_from(i)
                .ok()
                .and_then(|f| PixelInfo::try_from(f).ok())
            {
                let size = 2 * 3 * 4 * 5 * 6;
                let size = Size::new(size, size);
                let bits_per_pixel_from_size =
                    pixel.surface_bytes(size).unwrap() * 8 / size.pixels();

                assert_eq!(
                    pixel.bits_per_pixel(),
                    bits_per_pixel_from_size as u32,
                    "Failed for {:?}",
                    pixel
                );
            }
        }
    }

    #[test]
    fn from_dxgi() {
        // if it's a valid DXGI_FORMAT, it should be a valid PixelSize
        for i in 0..256_u32 {
            if let Ok(format) = DxgiFormat::try_from(i) {
                if matches!(format, DxgiFormat::UNKNOWN | DxgiFormat::V208) {
                    continue;
                }

                let result = PixelInfo::try_from(format);
                assert!(result.is_ok(), "Failed for {:?}", format);
            }
        }
    }
}
