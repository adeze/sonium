# XEM JSON Schema artifacts

These files are generated from the runtime validators:

```bash
just xem-schema crates/sonium-xem/schemas
```

The export example and the Rust tests compare the checked-in artifacts with the
canonical schema functions, so a schema change cannot silently omit its
published artifact. The compatibility workflow parses both files with an
independent JSON parser and exercises the room v0 migration plus the v1 room
and multiphysics golden envelopes through the Python and Node readers.
