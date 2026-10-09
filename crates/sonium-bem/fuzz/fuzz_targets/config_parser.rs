#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_bem::core::io::native::{ConfigFormat, parse_config};

fuzz_target!(|bytes: &[u8]| {
    let source = String::from_utf8_lossy(bytes);
    let format = if bytes.first().copied().unwrap_or_default() % 2 == 0 {
        ConfigFormat::Json
    } else {
        ConfigFormat::Toml
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = parse_config(&source, format);
    }));
    assert!(result.is_ok(), "BEM config parser panicked on fuzz input");
});
