//! This module is not a test per se, but a script for generating test files.
//!
//! This script is responsible for generating random block-compression images
//! that exhaustively test certain properties.

use std::{fs::File, io::Write};

use ddsd::{header::*, *};

mod util;

fn create_bc_data<const N: usize>(
    mut w: impl Write,
    blocks_x: u32,
    blocks_y: u32,
    format: DxgiFormat,
    mut gen: impl FnMut(u32, u32) -> [u8; N],
) -> Result<(), std::io::Error> {
    let pixel_info = PixelInfo::try_from(format).unwrap();
    if let PixelInfo::Block {
        bytes_per_block,
        block_size,
    } = pixel_info
    {
        assert_eq!(N, bytes_per_block as usize);
        assert_eq!((4, 4), block_size);
    } else {
        panic!("Not a block format");
    }

    // Header
    util::write_simple_dds_header(&mut w, Size::new(blocks_x * 4, blocks_y * 4), format)?;

    // now for the actual data
    for y in 0..blocks_y {
        for x in 0..blocks_x {
            let block = gen(x, y);
            w.write_all(&block)?;
        }
    }

    Ok(())
}

fn random_block<const N: usize>(rng: &mut impl rand::Rng) -> [u8; N] {
    let mut block = [0; N];
    rng.fill_bytes(&mut block);
    block
}

fn set_lsb(block: &mut u128, count: u8, value: u128) {
    *block &= !((1 << count) - 1);
    *block |= value;
}
fn push_bc7_mode(block: &mut u128, mode: u8) {
    *block = (*block << (mode + 1)) | (1 << mode);
}

#[test]
fn bc7_mode_0() {
    // Mode 0 has 4 partition bits that we want to check exhaustively.
    let mut file = File::create("test-data/images/bc fuzz/bc7 mode 0.dds").unwrap();
    let mut rng = util::create_rng();
    create_bc_data(&mut file, 256, 16, DxgiFormat::BC7_UNORM, |_, y| {
        let mut block = u128::from_le_bytes(random_block(&mut rng));

        set_lsb(&mut block, 4, y as u128);
        push_bc7_mode(&mut block, 0);
        block.to_le_bytes()
    })
    .unwrap();
}

#[test]
fn bc7_mode_1_2_3() {
    // Mode 1/2/3 have 6 partition bits and otherwise nothing interesting
    for mode in 1..=3 {
        let mut file =
            File::create(format!("test-data/images/bc fuzz/bc7 mode {}.dds", mode)).unwrap();
        let mut rng = util::create_rng();
        create_bc_data(&mut file, 128, 64, DxgiFormat::BC7_UNORM, |_, y| {
            let mut block = u128::from_le_bytes(random_block(&mut rng));

            set_lsb(&mut block, 6, y as u128);
            push_bc7_mode(&mut block, mode);
            block.to_le_bytes()
        })
        .unwrap();
    }
}

#[test]
fn bc7_mode_4() {
    // Mode 4 has 2 bits rotations and 1 index mode bit
    let mut file = File::create("test-data/images/bc fuzz/bc7 mode 4.dds").unwrap();
    let mut rng = util::create_rng();
    create_bc_data(&mut file, 256, 8, DxgiFormat::BC7_UNORM, |_, y| {
        let mut block = u128::from_le_bytes(random_block(&mut rng));

        set_lsb(&mut block, 3, y as u128);
        push_bc7_mode(&mut block, 4);
        block.to_le_bytes()
    })
    .unwrap();
}

#[test]
fn bc7_mode_5() {
    // Mode 5 has 2 bits rotations
    let mut file = File::create("test-data/images/bc fuzz/bc7 mode 5.dds").unwrap();
    let mut rng = util::create_rng();
    create_bc_data(&mut file, 256, 4, DxgiFormat::BC7_UNORM, |_, y| {
        let mut block = u128::from_le_bytes(random_block(&mut rng));

        set_lsb(&mut block, 2, y as u128);
        push_bc7_mode(&mut block, 5);
        block.to_le_bytes()
    })
    .unwrap();
}

#[test]
fn bc7_mode_6() {
    // Mode 6 has no special bits, so pure random is enough
    let mut file = File::create("test-data/images/bc fuzz/bc7 mode 6.dds").unwrap();
    let mut rng = util::create_rng();
    create_bc_data(&mut file, 64, 64, DxgiFormat::BC7_UNORM, |_, _| {
        let mut block = u128::from_le_bytes(random_block(&mut rng));
        push_bc7_mode(&mut block, 6);
        block.to_le_bytes()
    })
    .unwrap();
}

#[test]
fn bc7_mode_7() {
    // Mode 7 has 6 partition bits that we want to check exhaustively.
    let mut file = File::create("test-data/images/bc fuzz/bc7 mode 7.dds").unwrap();
    let mut rng = util::create_rng();
    create_bc_data(&mut file, 128, 64, DxgiFormat::BC7_UNORM, |_, y| {
        let mut block = u128::from_le_bytes(random_block(&mut rng));

        set_lsb(&mut block, 6, y as u128);
        push_bc7_mode(&mut block, 7);
        block.to_le_bytes()
    })
    .unwrap();
}

#[test]
fn bc6_modes() {
    let pure_random = |file: &mut File, format: DxgiFormat, mode: u8, mode_bits: u8| {
        let mut rng = util::create_rng();
        create_bc_data(file, 64, 32, format, |_, _| {
            let mut block = u128::from_le_bytes(random_block(&mut rng));
            set_lsb(&mut block, mode_bits, mode as u128);
            block.to_le_bytes()
        })
        .unwrap();
    };

    let dir = "test-data/images/bc fuzz";
    for format in [DxgiFormat::BC6H_SF16, DxgiFormat::BC6H_UF16] {
        let name = match format {
            DxgiFormat::BC6H_SF16 => "SF16",
            DxgiFormat::BC6H_UF16 => "UF16",
            _ => unreachable!(),
        };

        let foo = |mode: u8, mode_bits: u8| {
            let mode_name = match mode & 0b11 {
                0b11 => "mode one",
                _ => "mode two",
            };
            let file_path = match mode_bits {
                2 => format!("{dir}/bc6 {name} {mode_name} {mode:0>2b}.dds"),
                5 => format!("{dir}/bc6 {name} {mode_name} {mode:0>5b}.dds"),
                _ => unreachable!(),
            };
            let mut file = File::create(file_path).unwrap();
            pure_random(&mut file, format, mode, mode_bits);
        };

        for mode in [0b00, 0b01] {
            foo(mode, 2);
        }
        for mode in [
            0b00010, 0b00110, 0b01010, 0b01110, 0b10010, 0b10110, 0b11010, 0b11110, 0b00011,
            0b00111, 0b01011, 0b01111,
        ] {
            foo(mode, 5);
        }
    }
}
