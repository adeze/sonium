# Waveguide optimization examples

Two configurations, two different purposes. Do not mix them up.

## `waveguide_optimization_smoke.json` — smoke test only

Flat, in-phase 1 mm/s piston velocity for both drivers from 200 Hz to 5 kHz.
This is a solver smoke test: it exercises the full waveguide/optimization
pipeline with a trivially predictable drive. It is **not** a physical
optimization case — no real two-way radiates like this — and it must not be
used to draw acoustic conclusions. Retained for CI and regression checks.

## `waveguide_optimization.json` — realistic benchmark

A credibility benchmark for waveguide shape optimization. Both drivers are
*typed* electro-acoustic units (`driver_model`) with lumped
Thiele–Small-consistent curves and a passive crossover network, replacing the
flat in-phase velocity of the smoke test.

Driver chain (per unit, at the 2.83 Vrms ≈ 1 W/8 Ω reference):

```
Z_coil(f) = Re + jω(Le + L_series) − j/(ωC_series)
Z_mech(f) = Rms + jω·Mms + 1/(jω·Cms)        (excludes radiation load)
v(f)      = Bl·V / (Z_coil·Z_mech + Bl²)      [RMS m/s]
Z_el(f)   = Z_coil + Bl² / Z_mech
```

Units:

- **Woofer** (6.5″ class): Re=5.6 Ω, Le=0.35 mH, Bl=6.5 T·m, Mms=14 g,
  Cms=1.1 mm/N (fs≈40 Hz), Qms=4, Sd=133 cm², with an 0.8 mH series
  low-pass inductor (electrical corner ≈ 780 Hz, compounding the
  mass roll-off). Folded into `inductance_curve`.
- **Tweeter** (25 mm dome): Re=4.6 Ω, Le=0.05 mH, Bl=2.3 T·m, Mms=0.4 g,
  Cms=78 µm/N (fs≈900 Hz), Qms=3, Sd=5.3 cm², with a 120 µF series
  high-pass capacitor (`series_capacitance_f`; electrical resonance with Le
  ≈ 2.05 kHz, acoustic high-pass corner near 1.8–2 kHz).

Because the crossover lives in the driver network, the velocity transfer
curves, the electrical impedance curves, and the coupled
radiation-matrix solve (`solve_typed_driver_coupling`) all see the same
complex passbands — including crossover phase. Around the 1–3 kHz acoustic
crossover both drivers radiate with a large, frequency-dependent relative
phase, which is exactly the regime the smoke test cannot represent.

Curve generation: `generate_waveguide_benchmark.py` (same directory) rebuilds
the JSON from these equations, keeping the curves internally consistent.

Mesh-convergence status: the sweep uses the production λ_min/8 constant-element
target at 5 kHz (see review §2.7). A shortlisted mesh-convergence study for
this benchmark (candidate-ordering preservation at λ/4–λ/6) is still pending
and is required before quoting absolute optimization outcomes.
