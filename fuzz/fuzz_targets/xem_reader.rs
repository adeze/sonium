#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::Value;
use sonium_xem::{RoomConfig, io::read_external_room_value_checked};

// Exercise strict typed and external XEM readers with arbitrary UTF-8
// replacements. Both readers must reject malformed or unknown input with a
// typed error and must not panic.
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = RoomConfig::from_json_checked(&text);
    let _ = RoomConfig::from_json_migrated(&text);
    if let Ok(value) = serde_json::from_str::<Value>(&text) {
        let _ = read_external_room_value_checked(value);
    }
});
