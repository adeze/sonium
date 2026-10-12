# Original CRASH box reconstruction inputs

`model.json` extracts the original 140-Q4 CRASH box cards. There are 154
shell nodes plus the explicit rigid-body master 155. It includes the original
Johnson–Cook card, thickness/quad flags, initial velocity, wall node set,
rigid-body card and slave set, and the 100 kg point mass at node 14. This
artifact is reconstruction input; a native original-response comparison
remains pending and no numerical solve is claimed here.

The source model is Copyright (C) 2026 SISW Siemens Industry Software Inc.,
licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
Source: OpenCourant `qa-tests/miniqa/CRASH/crashsection/box1_42/data/BOX1BL_0000.rad`.
The starter and engine deck SHA-256 values must match the existing independently
harvested `goldens/miniqa/CRASH/crashsection/box1_42.json` before extraction.

The deck literally declares kg/m/s, with E=205, density=7.8e-6 and box length
50 in those units. The reconstruction uses a fixed physical similarity
transform: length and time multiply by 0.001, stress and density by 1e9.
This gives E=205 GPa, density=7800 kg/m³, length=50 mm and duration=100 us.
Mass, velocity, energy and momentum stay unchanged; forces multiply by 1000.
This is a similarity transform, not a correction of the original unit labels.
Johnson–Cook reference rate multiplies by 1000, preserving the rate ratio.
The independent mass assembly gives shell mass 0.04095000147 kg, total mass
240.04095000147 kg, initial wall-projected kinetic energy 7681.24488004 J,
and initial wall impulse -0.016380000588 Ns. These match the original first
TH sample within its rounding bands. They validate reconstruction and
initial wall projection, rather than an evolving shell response.
Geometry, material and motion constants come only from original cards, not
from the golden response. The original golden checksum is included.

The rigid-wall TH `~1` channel is accumulated normal momentum transfer, not
instantaneous force. Engine `constraints/general/rwall/rgwal0.F` accumulates
`FSAV(1) += FXN` and sets force separately via division by the step interval.
Thus wall impulse is unchanged under the similarity transform; its time
derivative gives the force. Preserve this distinction in the native replay.

Reproduce:

```sh
python3 scripts/extract_crash_box.py STARTER_DECK ENGINE_DECK \
  crates/sonium-qa/goldens/miniqa/CRASH/crashsection/box1_42.json OUTPUT_DIRECTORY
```

The development example `examples/crash_box_probe.rs` now evolves the full
140-element mesh with current Q4 frames, native LAW2 layer history, wall
velocity projection and accumulated impulse, rigid-body translation/rotation,
and the point mass. Its CSV output is a diagnostic and is not registered as
a passing QA comparison. The default uses the production 3D LAW2 return
with plane-stress closure and the production hourglass convenience defaults.

Optional diagnostic switches isolate reference formulation differences:
`CRASH_BOX_PROJECTION=1` uses an experimental in-plane explicit projection
with the unfiltered maximum in-plane strain-rate measure;
`CRASH_BOX_ROTARY=1` adds the reference numerical rotary inertia
`m_node * area / 9`; `CRASH_BOX_HG=1` reduces the convenience damping scale
to the original blank-card Hm/Hf defaults, while `CRASH_BOX_HG=2` replaces
it with an experimental modal viscoelastic control. The latter also includes
transverse-displacement stabilization, which now also has configurable
viscous and stiffness coefficients in the production kernel. These exploratory adapters have not established full shell equivalence
and must not be treated as validated production material or hourglass APIs.

```sh
CRASH_BOX_STEPS=20000 CRASH_BOX_PROJECTION=1 CRASH_BOX_ROTARY=1 \
  CRASH_BOX_HG=2 cargo run -p sonium-qa --features qa-binaries \
  --example crash_box_probe > crash_box_probe.csv
```

Current diagnostic 5000/10000/20000-step comparisons cover the entire
original harvested interval. At 20000 steps, peak-normalized history errors
are about 2.0% for internal energy, 2.2% for wall impulse, and 43% for node
17's y displacement. The last quantity changes by less than 0.01% between
10000 and 20000 steps. Timestep refinement therefore does not resolve the
remaining formulation discrepancy. No original-impact accuracy pass is
claimed. The independent clamped circular plate benchmark is separately
scoped and does not close this requirement.

The production warped-Q4 frame now projects its first edge into the
center tangent plane before normalization. Without that projection the
local basis was nonorthogonal for warped quads. A rotated bilinear-saddle
regression checks the analytic projected area, orthonormal axes and
preservation of force/velocity work.

The original TYPE1 thickness rule samples the two surfaces and intervening
uniform levels. `coqini.F` and `layini.F` select piecewise-linear stress
interpolation, not Gauss integration. For three levels, membrane weights
are `[t/4, t/2, t/4]`; moment weights are `[-t²/12, 0, t²/12]`.
The production `LayerStates::new_piecewise_linear` implements the analytic
hat-function integrals for 2–11 levels, including distinct moment weights.
The elastic regression verifies `N = C t epsilon` and `M = C t³ kappa / 12`
for every count and two thickness scales. `LayerStates::new` retains Gauss
integration as an explicit separate rule.

Further example options are `CRASH_BOX_LAYERS=1` for this surface-sampled
rule, `CRASH_BOX_FRAME=1` for an experimental symmetric convected frame,
and `CRASH_BOX_RATE=1` for a common section energy-equivalent strain rate.
The latter matches the membrane-plus-bending rate definition in `cforc3.F`,
used by all material layers. These options are diagnostic; they have not
closed the original-impact comparison. A run with all three enabled at
5000 steps still has a large node-17 y-displacement error. The comparable
1100-step run changes that endpoint by less than 0.01%, so matching the
reference's roughly 1100 time steps does not explain the discrepancy.

The production transverse hourglass regression compares decay of its
normal checkerboard mode against the independent damped-mode solution and
checks that an affine normal-velocity field with matching rigid rotation
produces no stabilization force. This establishes the added mode control;
it does not establish reference hourglass-formulation equivalence.

An elastic counterfactual isolates the material return from the remaining
shell discrepancy. `CRASH_BOX_ELASTIC=1` sets the native LAW2 yield and cap
to `1e30 Pa` and hardening coefficient to zero, keeping the original elastic
constants and other cards. In a separate reference-deck copy, set LAW2's
`a` and `sigmax` to `1e21` in the original literal units and `b` to zero.
These are the same values under the fixed similarity transform. Keep the
original thickness, node sets, constraints and engine cards.

```sh
CRASH_BOX_ELASTIC=1 CRASH_BOX_END_S=0.00010001 CRASH_BOX_RATE=1 \
  CRASH_BOX_LAYERS=1 CRASH_BOX_FRAME=1 CRASH_BOX_PROJECTION=1 \
  CRASH_BOX_ROTARY=1 CRASH_BOX_HG=2 \
  cargo run -p sonium-qa --features qa-binaries \
  --example crash_box_probe > crash_box_elastic.csv
```

`CRASH_BOX_END_S` overrides the duration in physical seconds; its default
remains the original golden's final stamp. Interpolate the native history
at the reference TH stamps after applying the time/length similarity
transform. The elastic reference's final TH stamp is 99.07125 us, which
differs from the original plastic run's 99.03467 us.

Both elastic histories have zero plastic work. At 5000 steps the native
node-17 y history still has about 75% peak-normalized error; its value at
the reference's final stamp is -0.03424 mm versus -0.14316 mm. Internal
energy and wall-impulse history errors are about 7% and 6%, respectively.
The discrepancy therefore persists without plastic yielding. Correcting
the diagnostic stabilization wave speed to LAW2's `sqrt(E/rho)` has a
negligible effect on that displacement gap. Elastic shell dynamics and
stabilization still require investigation.

A fresh unmodified reference run reproduces all 100 archived time stamps
and all samples of eight compared channels exactly: internal, kinetic,
hourglass and plastic energies, wall impulse, and node-17 x/y/z displacement.
The original reference remains unchanged. Neither this reproduction nor
the elastic counterfactual closes the native original-impact requirement.
