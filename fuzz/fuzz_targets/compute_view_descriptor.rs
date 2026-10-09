#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::{BufferView, ViewAliasing, ViewDescriptor, ViewLayout};

// Arbitrary descriptor metadata must be rejected or validated without a
// panic or overflowing address arithmetic. This is the launch-binding
// boundary shared by native CPU, Metal, and CUDA adapters.
fuzz_target!(|data: &[u8]| {
    let word = |offset: usize| -> usize {
        data.get(offset..offset + 8)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u64::from_le_bytes)
            .unwrap_or_default() as usize
    };
    let rank = word(0) % 5;
    let shape = (0..rank).map(|axis| word(8 + axis * 8)).collect::<Vec<_>>();
    let strides = (0..rank)
        .map(|axis| word(48 + axis * 8))
        .collect::<Vec<_>>();
    let descriptor = ViewDescriptor {
        rank,
        shape,
        byte_offset: word(88),
        element_strides: strides,
        layout: match data.get(3).copied().unwrap_or_default() % 3 {
            0 => ViewLayout::Strided,
            1 => ViewLayout::RowMajor,
            _ => ViewLayout::ColumnMajor,
        },
        dtype: if data.first().is_some_and(|byte| byte & 1 == 0) {
            "f64"
        } else {
            "f32"
        },
        device: if data.get(1).is_some_and(|byte| byte & 1 == 0) {
            "cpu"
        } else {
            "metal"
        },
        aliasing: match data.get(2).copied().unwrap_or_default() % 3 {
            0 => ViewAliasing::ReadOnly,
            1 => ViewAliasing::Disjoint,
            _ => ViewAliasing::OverlapSafe,
        },
    };
    let _ = descriptor.validate();
    let _ = descriptor.validate_for::<f32>("cpu");
    let _ = descriptor.validate_for_buffer::<f64>(word(96), "cpu");
    if let Ok(count) = descriptor.checked_element_count() {
        let logical_index = if count == 0 { 0 } else { word(136) % count };
        let _ = descriptor.checked_element_index::<f64>(logical_index, "cpu");
        let _ = descriptor.checked_byte_end_exclusive::<f64>("cpu");
    }
    let _ = BufferView::new(word(104), word(112), word(120), word(128));
});
