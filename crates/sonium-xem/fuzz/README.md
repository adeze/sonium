# sonium-xem fuzzing

Install `cargo-fuzz`, then run from this directory:

```bash
cargo fuzz run streaming_readers -- -max_len=512
```

The target mutates chunk checksums/headers and arbitrary JSON-lines input to
exercise bounded recovery and malformed-record handling.
