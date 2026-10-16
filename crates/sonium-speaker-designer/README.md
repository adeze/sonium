# Speaker Designer

Initial native two-way passive crossover workbench using the `adeze/gpui-toolkit` fork. This is an offline design preview, not a receiver controller or acoustically validated design.

## Run

Keep the two repositories beside each other:

```sh
git clone https://github.com/adeze/sonium.git
git clone https://github.com/adeze/gpui-toolkit.git
cd sonium
cargo run -p sonium-speaker-designer --features desktop
```

The desktop feature uses the sibling toolkit's `gpui-ui-kit` and `gpui-miniapp` crates. GPUI's third-party source is shared with that toolkit and resolved in `Cargo.lock`. The desktop app is deliberately outside the default workspace members; numerical tests do not require a GUI.

```sh
cargo test -p sonium-speaker-designer --lib
cargo check -p sonium-speaker-designer --features desktop
cargo clippy -p sonium-speaker-designer --features desktop --all-targets
```

## Initial workflow

Import a response and an impedance file for each of the woofer and tweeter, then edit the six component values. The fixed topology is:

- Woofer: series resistor and inductor, followed by a driver with a parallel capacitor.
- Tweeter: series resistor and capacitor, followed by a driver with a parallel inductor.

Series resistance may be zero (a true wire bypass); inductance and capacitance must be positive. The circuit is solved with Sonium's existing complex modified-nodal-analysis implementation. Electrical loading uses the imported driver impedance, not a nominal eight-ohm substitution.

Response/phase and input-impedance plots share a logarithmic frequency grid over the intersection of all four imported datasets. Driver polarity and delay affect the coherent acoustic sum. The built-in synthetic demo uses explicitly labeled flat responses and resistive loads; it is not measurement evidence.

New project clears the workspace; Undo/Redo retains up to 20 edits, including imports and project changes (Cmd-Z / Cmd-Shift-Z). Open/Save as JSON uses a versioned document containing the imported samples. Saves require a new filename and refuse overwrite; imports and project files are limited to 16 MiB. There is no unsaved-change close prompt yet. The inspector and plot panes scroll independently.

## Import contract

Files contain exactly three whitespace-separated numeric columns, with at least two data rows:

```text
# Response: frequency_Hz magnitude_dB_SPL phase_degrees
100 85.0 -15.0
1000 86.0 -30.0
```

For ZMA files, magnitude is impedance in ohms. Frequencies must be finite, positive and strictly increasing. Impedance magnitudes must be positive. Blank lines, an initial UTF-8 BOM and whole-line comments starting with `#`, `;`, `*` or `//` are accepted. Missing phase, extra columns, nonfinite values, duplicate frequencies and disjoint frequency ranges are rejected. Files are not sorted, phase is not fabricated, and response data is not extrapolated.

The importer currently does not support comma-delimited CSV, arbitrary text headers, magnitude-only responses or automatic minimum-phase reconstruction. Complex Cartesian interpolation on log frequency handles ordinary phase wrap; sparse samples with large physical phase rotation still require adequate original sampling.

## Measurement references and interpretation

Both response files must describe compatible measurement conditions: drive voltage, distance, SPL reference and time origin. The reference voltage/distance fields describe those imported conditions; they do not renormalize existing response data. Predictions multiply each imported complex response by its circuit terminal-to-source voltage ratio and the explicitly selected polarity/delay. The input impedance is derived from the circuit's delivered source current.

Imported data and original source names remain in project state. Editing reference metadata does not establish that the measurement references agree. The app cannot infer phase alignment or acoustic provenance from an FRD/ZMA file. Confirm those independently before interpreting a coherent sum.

The design calculations use `f64`/`Complex64`; graph coordinates may use display precision. Exact cancellation has a finite -300 dB display floor. Plotted phase is wrapped to ±180 degrees. Passive component losses are represented only by the explicit series resistors in this initial topology.

## Scope

This first version has a fixed two-way network, not arbitrary schematic wiring. It does not include enclosure/BEM studies, automatic optimization, component power/thermal qualification, preferred-value rounding, angular directivity, receiver transfer or manufacturing export. No decompiled VituixCAD implementation is included.
