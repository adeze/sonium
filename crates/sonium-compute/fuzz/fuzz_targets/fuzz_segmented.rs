#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::segmented_sum;

fuzz_target!(|bytes: &[u8]| {
    let value_len = usize::from(bytes.first().copied().unwrap_or(0) % 17);
    let segment_count = usize::from(bytes.get(1).copied().unwrap_or(0) % 9);
    let values: Vec<f64> = (0..value_len)
        .map(|index| f64::from(bytes.get(index + 2).copied().unwrap_or(0)) / 17.0)
        .collect();
    let mut offsets = Vec::with_capacity(segment_count.saturating_add(1));
    for index in 0..segment_count.saturating_add(1) {
        let raw = bytes.get(index + 2 + value_len).copied().unwrap_or(0);
        offsets.push(i32::from(raw) - 32);
    }
    let mut output = vec![0.0; segment_count];
    let _ = segmented_sum(&values, &offsets, &mut output);
});
