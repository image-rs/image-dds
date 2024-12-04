fn u32s_as_bytes(buffer: &mut [u32]) -> &mut [u8] {
    // SAFETY: Slice guarantees the slice::len() * std::mem::size_of::<T>() is <= isize::MAX.
    // SAFETY: Therefore, this multiplication will not overflow.
    let len = buffer.len() * 4;
    let ptr = buffer.as_mut_ptr() as *mut u8;
    unsafe {
        // SAFETY: This is valid because the pointer and length comes from a slice.
        std::slice::from_raw_parts_mut(ptr, len)
    }
}

pub fn read_u32s(reader: &mut impl std::io::Read, buffer: &mut [u32]) -> std::io::Result<()> {
    reader.read_exact(u32s_as_bytes(buffer))?;
    for i in buffer.iter_mut() {
        *i = u32::from_le(*i);
    }
    Ok(())
}
