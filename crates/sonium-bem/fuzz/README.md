# sonium-bem fuzzing

Install `cargo-fuzz`, then run from this directory:

```bash
cargo fuzz run config_parser -- -max_len=4096
```

Arbitrary UTF-8-repaired JSON/TOML inputs must return a typed parse error or a
valid configuration; they must never panic.
