#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_bem::core::io::nc_format::parse_nc_input_string;
use std::path::PathBuf;

// Fuzz the legacy NC.inp boundary without allowing arbitrary input to turn
// into a filesystem operation. The parser must return a typed error or a
// valid config, never panic on malformed sections or numeric tokens.
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = parse_nc_input_string(&text, PathBuf::from("."));
});
