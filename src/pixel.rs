use crate::header::{Dx9PixelFormat, DxgiFormat, Header};
use crate::{util::div_ceil, Format, FormatError, Size};

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
/// of a surface to create the [`DataLayout`](crate::DataLayout) of a DDS file.
///
/// ## Unsupported pixel formats
///
/// Note that [`PixelInfo`] can describe pixel formats are **not** supported for
/// decoding. This is by design to allow users to get the data layout for DDS
/// files this library doesn't fully support.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelInfo {
    /// Each pixel has a fixed number of bytes, regardless of the dimensions of
    /// the surface.
    ///
    /// This is only the case for uncompressed image formats. E.g.
    /// `R8G8B8A8_UNORM`.
    Fixed { bytes_per_pixel: u8 },
    /// Pixels are grouped into blocks of constant byte size.
    ///
    /// This is used for block-compressed and channel-packed sub-sampled pixel
    /// formats. E.g. `BC1_UNORM` (`DXT1`) has 8 bytes per 4x4 block, and
    /// `R8G8B8G8_UNORM` has 4 bytes per pair (2x1 block).
    Block(BlockPixelInfo),
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
    BiPlanar(BiPlanarPixelInfo),
}
const fn pack_2_u4((a, b): (u8, u8)) -> u8 {
    debug_assert!(a < 16);
    debug_assert!(b < 16);
    a | (b << 4)
}
const fn unpack_2_u4(packed: u8) -> (u8, u8) {
    (packed & 0x0F, packed >> 4)
}
/// See [`PixelInfo::Block`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockPixelInfo {
    bytes_per_block: u8,
    size: u8,
}
impl BlockPixelInfo {
    const fn new(bytes_per_block: u8, block_size: (u8, u8)) -> Self {
        Self {
            bytes_per_block,
            size: pack_2_u4(block_size),
        }
    }

    pub const fn bytes_per_block(&self) -> u8 {
        self.bytes_per_block
    }
    /// Returns the `(width, height)` of a block.
    pub const fn size(&self) -> (u8, u8) {
        unpack_2_u4(self.size)
    }
    /// Returns the number of pixels in a block.
    pub const fn pixels(&self) -> u8 {
        let (x, y) = self.size();
        x * y
    }
}
/// See [`PixelInfo::BiPlanar`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BiPlanarPixelInfo {
    bytes: u8,
    plane2_sub_sampling: u8,
}
impl BiPlanarPixelInfo {
    const fn new(
        plane1_bytes_per_pixel: u8,
        plane2_bytes_per_sample: u8,
        plane2_sub_sampling: (u8, u8),
    ) -> Self {
        Self {
            bytes: pack_2_u4((plane1_bytes_per_pixel, plane2_bytes_per_sample)),
            plane2_sub_sampling: pack_2_u4(plane2_sub_sampling),
        }
    }

    pub const fn plane1_bytes_per_pixel(&self) -> u8 {
        unpack_2_u4(self.bytes).0
    }
    pub const fn plane2_bytes_per_sample(&self) -> u8 {
        unpack_2_u4(self.bytes).1
    }
    pub const fn plane2_sub_sampling(&self) -> (u8, u8) {
        unpack_2_u4(self.plane2_sub_sampling)
    }
}

impl PixelInfo {
    pub const fn fixed(bytes_per_pixel: u8) -> Self {
        Self::Fixed { bytes_per_pixel }
    }
    pub const fn block(bytes_per_block: u8, block_size: (u8, u8)) -> Self {
        Self::Block(BlockPixelInfo::new(bytes_per_block, block_size))
    }
    pub const fn bi_planar(
        bytes_per_pixel: u8,
        bytes_per_sample: u8,
        sub_sampling: (u8, u8),
    ) -> Self {
        Self::BiPlanar(BiPlanarPixelInfo::new(
            bytes_per_pixel,
            bytes_per_sample,
            sub_sampling,
        ))
    }

    pub fn from_header(header: &Header) -> Result<Self, FormatError> {
        match header {
            Header::Dx9(dx9) => match &dx9.pixel_format {
                Dx9PixelFormat::FourCC(four_cc) => Format::from_four_cc(*four_cc)
                    .map(Into::into)
                    .ok_or(FormatError::UnsupportedFourCC(*four_cc)),
                Dx9PixelFormat::Mask(pixel_format) => {
                    Ok(PixelInfo::fixed(pixel_format.rgb_bit_count as u8 / 8))
                }
            },
            Header::Dx10(dx10) => dx10
                .dxgi_format
                .try_into()
                .map_err(|_| FormatError::UnsupportedDxgiFormat(dx10.dxgi_format)),
        }
    }

    /// Returns the number of bits per pixel.
    ///
    /// If the number of bits per pixel is not an integer, the result is rounded
    /// up to the nearest integer.
    pub fn bits_per_pixel(&self) -> u32 {
        match *self {
            Self::Fixed { bytes_per_pixel } => bytes_per_pixel as u32 * 8,
            Self::Block(block) => {
                let bits_per_block = block.bytes_per_block() as u32 * 8;
                div_ceil(bits_per_block, block.pixels() as u32)
            }
            Self::BiPlanar(bi_planar) => {
                let plane1_bits_per_pixel = bi_planar.plane1_bytes_per_pixel() as u32 * 8;
                let plane2_sub_sampling = bi_planar.plane2_sub_sampling();
                let sub_sampling = plane2_sub_sampling.0 as u32 * plane2_sub_sampling.1 as u32;
                plane1_bits_per_pixel
                    + div_ceil(bi_planar.plane2_bytes_per_sample() as u32 * 8, sub_sampling)
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
            Self::Block(block) => {
                let block_size = block.size();
                let blocks_x = div_ceil(size.width, block_size.0 as u32);
                let blocks_y = div_ceil(size.height, block_size.1 as u32);
                // This cannot overflow, because both factors are u32.
                let blocks = blocks_x as u64 * blocks_y as u64;
                blocks.checked_mul(block.bytes_per_block() as u64)
            }
            Self::BiPlanar(bi_planar) => {
                let plane1_bytes = size
                    .pixels()
                    .checked_mul(bi_planar.plane1_bytes_per_pixel() as u64)?;

                let plane2_sub_sampling = bi_planar.plane2_sub_sampling();
                let chroma_x = div_ceil(size.width, plane2_sub_sampling.0 as u32);
                let chroma_y = div_ceil(size.height, plane2_sub_sampling.1 as u32);
                // This cannot overflow, because both factors are u32.
                let samples_chroma = chroma_x as u64 * chroma_y as u64;
                let plane2_bytes =
                    samples_chroma.checked_mul(bi_planar.plane2_bytes_per_sample() as u64)?;

                plane1_bytes.checked_add(plane2_bytes)
            }
        }
    }
}

impl std::fmt::Debug for PixelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed { bytes_per_pixel } => {
                write!(f, "Fixed({bytes_per_pixel} bytes/px)")
            }
            Self::Block(block) => write!(
                f,
                "Block({} bytes/{}x{}px)",
                block.bytes_per_block(),
                block.size().0,
                block.size().1
            ),
            Self::BiPlanar(bi_planar) => write!(
                f,
                "BiPlanar(plane1: {} bytes/px, plane2: {} bytes/{}x{}px)",
                bi_planar.plane1_bytes_per_pixel(),
                bi_planar.plane2_bytes_per_sample(),
                bi_planar.plane2_sub_sampling().0,
                bi_planar.plane2_sub_sampling().1
            ),
        }
    }
}

impl From<Format> for PixelInfo {
    fn from(value: Format) -> Self {
        use Format as F;

        match value {
            // 3 bytes per pixel
            F::R8G8B8_UNORM | F::B8G8R8_UNORM => Self::fixed(3),

            // sub-sampled formats
            // 4 bytes per one 2x1 block
            F::UYVY => Self::block(4, (2, 1)),

            // block compression formats
            // 16 bytes per one 4x4 block
            F::BC2_UNORM_PREMULTIPLIED_ALPHA
            | F::BC3_UNORM_PREMULTIPLIED_ALPHA
            | F::BC3_UNORM_RXGB => Self::block(16, (4, 4)),

            _ => {
                // All other formats should have a DXGI equivalent with known pixel info.
                // PANIC SAFETY: `tests/format.rs` contains a test that
                // exhaustively verifies that all formats have a pixel info.
                // This test would fail if the `unwrap` were to panic.
                let dxgi = DxgiFormat::try_from(value).unwrap();
                PixelInfo::try_from(dxgi).unwrap()
            }
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
            | F::B4G4R4A4_UNORM
            | F::A4B4G4R4_UNORM => Ok(Self::fixed(2)),
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

            // Sub-sampled 2x1
            F::R8G8_B8G8_UNORM | F::G8R8_G8B8_UNORM => Ok(Self::block(4, (2, 1))),
            // YUV 4:2:2 formats
            // https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering#yuy2
            // https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats#y216-and-y210
            F::YUY2 => Ok(Self::block(4, (2, 1))),
            F::Y210 | F::Y216 => Ok(Self::block(8, (2, 1))),
            F::R1_UNORM => Ok(Self::block(1, (8, 1))),

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

            // ASTC formats
            F::ASTC_4X4_TYPELESS | F::ASTC_4X4_UNORM | F::ASTC_4X4_UNORM_SRGB => {
                Ok(Self::block(16, (4, 4)))
            }
            F::ASTC_5X4_TYPELESS | F::ASTC_5X4_UNORM | F::ASTC_5X4_UNORM_SRGB => {
                Ok(Self::block(16, (5, 4)))
            }
            F::ASTC_5X5_TYPELESS | F::ASTC_5X5_UNORM | F::ASTC_5X5_UNORM_SRGB => {
                Ok(Self::block(16, (5, 5)))
            }
            F::ASTC_6X5_TYPELESS | F::ASTC_6X5_UNORM | F::ASTC_6X5_UNORM_SRGB => {
                Ok(Self::block(16, (6, 5)))
            }
            F::ASTC_6X6_TYPELESS | F::ASTC_6X6_UNORM | F::ASTC_6X6_UNORM_SRGB => {
                Ok(Self::block(16, (6, 6)))
            }
            F::ASTC_8X5_TYPELESS | F::ASTC_8X5_UNORM | F::ASTC_8X5_UNORM_SRGB => {
                Ok(Self::block(16, (8, 5)))
            }
            F::ASTC_8X6_TYPELESS | F::ASTC_8X6_UNORM | F::ASTC_8X6_UNORM_SRGB => {
                Ok(Self::block(16, (8, 6)))
            }
            F::ASTC_8X8_TYPELESS | F::ASTC_8X8_UNORM | F::ASTC_8X8_UNORM_SRGB => {
                Ok(Self::block(16, (8, 8)))
            }
            F::ASTC_10X5_TYPELESS | F::ASTC_10X5_UNORM | F::ASTC_10X5_UNORM_SRGB => {
                Ok(Self::block(16, (10, 5)))
            }
            F::ASTC_10X6_TYPELESS | F::ASTC_10X6_UNORM | F::ASTC_10X6_UNORM_SRGB => {
                Ok(Self::block(16, (10, 6)))
            }
            F::ASTC_10X8_TYPELESS | F::ASTC_10X8_UNORM | F::ASTC_10X8_UNORM_SRGB => {
                Ok(Self::block(16, (10, 8)))
            }
            F::ASTC_10X10_TYPELESS | F::ASTC_10X10_UNORM | F::ASTC_10X10_UNORM_SRGB => {
                Ok(Self::block(16, (10, 10)))
            }
            F::ASTC_12X10_TYPELESS | F::ASTC_12X10_UNORM | F::ASTC_12X10_UNORM_SRGB => {
                Ok(Self::block(16, (12, 10)))
            }
            F::ASTC_12X12_TYPELESS | F::ASTC_12X12_UNORM | F::ASTC_12X12_UNORM_SRGB => {
                Ok(Self::block(16, (12, 12)))
            }

            // Bi-planar formats
            // (4:2:0) bytes = w*h + 2 * ceil(w/2)*ceil(h/2)
            F::NV12 | F::OPAQUE_420 => Ok(Self::bi_planar(1, 2, (2, 2))),
            // (4:2:0) bytes = 2*(w*h + 2 * ceil(w/2)*ceil(h/2))
            F::P010 | F::P016 => Ok(Self::bi_planar(2, 4, (2, 2))),
            // (4:1:1) bytes = w*h + 2 * ceil(w/4)*h
            F::NV11 => Ok(Self::bi_planar(1, 2, (4, 1))),
            // (4:2:2) bytes = w*h + 2 * ceil(w/2)*h
            F::P208 => Ok(Self::bi_planar(1, 2, (2, 1))),

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
    use crate::header::*;
    use crate::*;

    #[test]
    fn bits_per_pixel() {
        // This test verifies that `bits_per_pixel` is implemented correctly,
        // using `surface_bytes` as the reference implementation. This is a
        // reasonable reference, since `surface_bytes` is foundational to this
        // library and tested implicitly in all tests reading DDS files.
        for dxgi in DxgiFormat::all() {
            if let Ok(pixel) = PixelInfo::try_from(dxgi) {
                let size = 2 * 3 * 4 * 5 * 6;
                let size = Size::new(size, size);
                let bits_per_pixel_from_size =
                    util::div_ceil(pixel.surface_bytes(size).unwrap() * 8, size.pixels());

                assert_eq!(
                    pixel.bits_per_pixel(),
                    bits_per_pixel_from_size as u32,
                    "Failed for {pixel:?}"
                );
            }
        }
    }

    #[test]
    fn from_dxgi() {
        // if it's a valid DXGI_FORMAT, it should be a valid PixelSize
        for dxgi in DxgiFormat::all() {
            if matches!(dxgi, DxgiFormat::UNKNOWN | DxgiFormat::V208) {
                continue;
            }

            let result = PixelInfo::try_from(dxgi);
            assert!(result.is_ok(), "Failed for {dxgi:?}");
        }
    }

    #[test]
    fn dxgi_supported() {
        // This test verifies that equivalent DxgiFormat and SupportFormat
        // have the same PixelInfo.
        for dxgi in DxgiFormat::all() {
            if let Some(format) = Format::from_dxgi(dxgi) {
                let dxgi_info = PixelInfo::try_from(dxgi).unwrap();
                let format_info = PixelInfo::from(format);

                assert_eq!(dxgi_info, format_info, "Failed for {dxgi:?}");
            }
        }
    }
}
