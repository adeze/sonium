# sonium-fft fuzzing

Install `cargo-fuzz`, then run from this directory:

```bash
cargo fuzz run planner_round_trip -- -max_len=512
```

The target covers smooth and Bluestein planner lengths and keeps minimized
round-trip regressions under `corpus/planner_round_trip/`.
