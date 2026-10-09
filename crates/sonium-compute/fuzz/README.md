# sonium-compute fuzz targets

These targets exercise checked CPU backend operations and malformed segmented
reduction metadata. They are intentionally bounded to small buffers so a
fuzzing worker cannot turn an input-controlled shape into an allocation bomb.

```bash
cargo fuzz run fuzz_compute_blas
cargo fuzz run fuzz_segmented
```

The checked APIs must reject malformed shapes with `BackendError` rather than
panic or read outside a buffer.
