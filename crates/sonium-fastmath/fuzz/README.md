# sonium-fastmath fuzzing

The `sincos_inputs` target exercises finite, subnormal, huge, NaN, and
infinite bit patterns through scalar sin/cos, phasor, derivative, and `f32`
entry points.  Run a bounded smoke campaign with:

```bash
cargo fuzz run sincos_inputs -- -runs=100 -max_len=512 -rss_limit_mb=256
```
