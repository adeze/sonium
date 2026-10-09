#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::{Backend, CpuBackend, DeviceBufferLike, ViewAliasing};

fuzz_target!(|bytes: &[u8]| {
    let backend = CpuBackend::new();
    let buffer_len = bytes.get(0).copied().unwrap_or_default() as usize;
    let buffer = backend
        .raw_alloc::<f64>(buffer_len)
        .expect("CPU fuzz allocation must be available");
    let read = |offset: usize| {
        let start = offset.min(bytes.len());
        let end = (start + std::mem::size_of::<usize>()).min(bytes.len());
        let mut value = 0usize;
        for (shift, byte) in bytes[start..end].iter().enumerate() {
            value |= (*byte as usize) << (shift * 8);
        }
        value
    };
    let offset = read(1);
    let len = read(1 + std::mem::size_of::<usize>());
    let stride = read(1 + 2 * std::mem::size_of::<usize>());

    let checked = backend.checked_view_descriptor_with_aliasing(
        &buffer,
        offset,
        len,
        stride,
        ViewAliasing::ReadOnly,
    );
    if let Ok(descriptor) = checked {
        descriptor
            .validate_for_buffer::<f64>(buffer.len(), backend.capabilities().backend_name)
            .expect("a checked descriptor must validate against its source buffer");
    }
});
