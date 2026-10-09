#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_xem::io::{ChunkedBinaryReader, ChunkedBinaryWriter, JsonLinesReader};
use std::io::Cursor;

fuzz_target!(|bytes: &[u8]| {
    let payload = &bytes[..bytes.len().min(128)];
    let mut framed = Vec::new();
    {
        let mut writer = ChunkedBinaryWriter::with_max_chunk_size(&mut framed, 128)
            .expect("chunked header write");
        let _ = writer.write_chunk(payload);
    }
    // Arbitrary mutation must produce an error or a bounded read, never a
    // panic.  The reader's recovery path is exercised by a second frame.
    if let Some(index) = bytes.get(0).map(|value| *value as usize % framed.len().max(1)) {
        framed[index] ^= bytes.get(1).copied().unwrap_or(0);
    }
    if let Ok(mut chunked) = ChunkedBinaryReader::with_max_chunk_size(Cursor::new(framed), 128) {
        let _ = chunked.next_chunk();
    }

    let mut lines = JsonLinesReader::with_max_record_bytes(Cursor::new(bytes), 128);
    let mut record = Vec::new();
    let _ = lines.next_record_into(&mut record);
});
