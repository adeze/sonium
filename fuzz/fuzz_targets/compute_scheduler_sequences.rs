#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::{DeviceBudget, DeviceScheduler, WorkLease, WorkRequest};

// Bounded admission/completion/cancellation sequences must never panic or
// leave counters beyond their configured quota. Dropping retained leases
// covers the cancellation/refund path used during abrupt caller teardown.
fuzz_target!(|data: &[u8]| {
    let scheduler = DeviceScheduler::new([
        DeviceBudget::new("fuzz-a", 256, 4),
        DeviceBudget::new("fuzz-b", 128, 2),
    ])
    .unwrap();
    let mut leases: Vec<WorkLease> = Vec::new();
    for &opcode in data.iter().take(128) {
        match opcode % 6 {
            0 | 1 => {
                let request =
                    WorkRequest::new(usize::from(opcode) * 7 % 300, usize::from(opcode >> 4) % 6);
                if let Ok(lease) = scheduler.acquire_on(usize::from(opcode >> 7), request) {
                    leases.push(lease);
                }
            }
            2 if !leases.is_empty() => {
                let _ = leases.swap_remove(0).complete();
            }
            3 if !leases.is_empty() => {
                let _ = leases.swap_remove(0).cancel();
            }
            4 => {
                let _ = scheduler.mark_device_lost(usize::from(opcode >> 7));
            }
            _ if !leases.is_empty() => {
                drop(leases.swap_remove(leases.len() - 1));
            }
            _ => {}
        }
        for snapshot in scheduler.snapshots().unwrap() {
            assert!(snapshot.bytes_in_use <= snapshot.max_bytes);
            assert!(snapshot.active <= snapshot.max_active);
        }
    }
});
