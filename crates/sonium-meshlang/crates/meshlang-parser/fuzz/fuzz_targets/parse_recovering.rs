#![no_main]

use libfuzzer_sys::fuzz_target;
use meshlang_parser::{ParseOptions, parse_recovering};

fuzz_target!(|bytes: &[u8]| {
    // The public parser accepts valid UTF-8. Lossy conversion exercises every
    // arbitrary byte sequence while preserving deterministic replacement and
    // never constructing an invalid Rust string.
    let source = String::from_utf8_lossy(bytes);
    let report = parse_recovering(
        &source,
        "fuzz-input.mlg",
        ParseOptions {
            max_diagnostics: 16,
            max_nesting: 64,
            ..ParseOptions::default()
        },
    );
    assert!(report.diagnostics.len() <= 16);
});
