# Propeller attenuation calibration study

Status: literature-informed engineering prior, not a universal performance
claim. Last reviewed 2026-07-15.

## Scope

The toroidal and serration terms in this crate are source-generation modifiers.
They are not a resolved LES/FW-H solution of a toroidal blade or serration
microgeometry. Equation correctness and dimensional consistency determine how
pressures scale; they cannot determine the absolute attenuation. The observable
calibration contract is therefore the predicted band reduction relative to a
specified conventional propeller, not any one local pressure-relief constant.

The crate reference case is a two-blade, five-inch propeller at 18,000 RPM,
3 N nominal thrust, with reductions integrated over 1–5 kHz using the stated
A-weighting and harmonic source model. Changing blade family, thrust, observer,
RPM, inflow turbulence, microphone arc, band, or comparison convention can move
the answer outside the intervals below.

## Comparison conventions

Noise reductions are only meaningful when the comparison is explicit:

- **Fixed RPM:** easy to reproduce but a modified blade may produce less
  thrust. Some apparent acoustic benefit is then unloading rather than a
  quieter way of producing the same useful output.
- **Matched thrust:** the preferred product/design comparison. RPM, pitch, or
  loading is adjusted until thrust agrees; electrical and torque limits should
  also be reported.
- **Observer-specific versus integrated power:** toroidal designs can
  redistribute directivity. A large reduction at one microphone is not
  necessarily the same reduction in sound power.
- **Tonal versus broadband:** this crate represents reductions through a
  harmonic source proxy. Serration literature often reports broadband spectra,
  so direct numerical equality should not be expected without calibration.

## Toroidal-loop prior

Published toroidal-propeller results do not support one geometry-independent
number. Recent numerical and experimental studies show benefits that depend on
loop closure, loading, advance ratio, tip Mach number, and observer angle, with
some conditions showing modest benefit or tonal redistribution.

For the reference observer/band, adopt:

- centre: **5.5 dB**;
- sensitivity interval: **2–8 dB**.

The current research-control reference produces about **5.92 dB**, which is a
defensible provisional centre. It must not be described as an absolute
toroidal-propeller attenuation.

## Serration priors

Serration effectiveness depends on tooth height and wavelength relative to the
hydrodynamic/acoustic length scales, not chord fraction alone. Leading-edge
serrations require turbulent inflow; trailing-edge serrations interact with the
boundary-layer thickness and convected gust wavelength. Benefits can disappear
or become penalties outside the effective band. Recent propeller experiments
also show that thrust loss can materially reduce the fixed-RPM advantage.

Adopt these observable priors:

| Modification and comparison | Centre | Sensitivity interval |
| --- | ---: | ---: |
| Leading-edge serration, effective turbulent-inflow midband | 4 dB | 0–7 dB |
| Trailing-edge serration, fixed RPM | 3.5 dB | 0–6 dB |
| Trailing-edge serration, matched thrust | 1.5 dB | 0–4 dB |
| Toroidal + leading/trailing-edge treatment | 7 dB | 4–10 dB |

The current 18,000 RPM research-control example gives approximately 1.65 dB
for leading plus trailing serrations alone and 8.89 dB for the combined
toroidal/serrated design. The first number is a tonal-harmonic proxy rather than
a turbulent-inflow broadband calibration; the second is plausible but near the
optimistic side of the combined prior.

## Geometry priors and binary ranges

The optimization binary starts at:

- amplitude `h/c = 0.018`;
- wavelength `lambda/c = 0.055`;
- treated span fraction `0.72`.

It permits only:

- `h/c = 0.006–0.035`;
- `lambda/c = 0.035–0.120`;
- treated span `0.40–0.95`.

These are design-search bounds, not a statement that every point is effective.
Future calibration should replace chord-only inputs with nondimensional groups
such as `h/delta`, `lambda/delta`, and `lambda/L_t`, where `delta` is the local
boundary-layer thickness and `L_t` the inflow integral scale.

The optimizer now defaults to no more than **5% thrust loss**. The CLI permits
up to **20%** only for explicit trade studies. A 20% default allows the search
to buy a misleading noise reduction through aerodynamic unloading. The binary
also limits the five-inch reference speed to 3,000–35,000 RPM and reports the
3 N source assumption; feasibility at the extremes requires an aerodynamic or
measured operating map.

The default optimizer also rejects candidates whose surrogate claims more than
the combined prior's 10 dB upper boundary. This is a model-validity guard, not
a claim that a physical design can never exceed 10 dB. Such a candidate must be
validated or used with a recalibrated prior before it is selected as an optimum.

## How to calibrate the model

### Test matrix

Use at least three independently manufactured specimens per geometry. Test the
conventional baseline, toroidal-only, leading-edge-only, trailing-edge-only,
and combined design over:

- at least five RPM or advance-ratio points;
- fixed-RPM and matched-thrust comparisons;
- low-turbulence inflow plus two characterized turbulence grids;
- multiple tooth amplitudes and wavelengths spanning the search range;
- a microphone arc or hemisphere sufficient to separate directivity changes
  from sound-power changes.

Record thrust, torque, electrical power, RPM, density, temperature, humidity,
inflow turbulence spectrum, integral length scale, background spectrum, and
blade geometry. Repeat measurements after remounting to capture installation
variance.

### Inference

1. Calibrate the conventional loading source first; do not use attenuation
   modifiers to hide a wrong baseline.
2. Fit toroidal, leading-edge, and trailing-edge observables separately with a
   hierarchical model across RPM, observer, and specimen.
3. Fit interaction terms only after the individual mechanisms are identified.
   Adding individual dB reductions assumes independent multiplicative powers
   and will generally overpredict a combined design.
4. Reserve entire RPMs, observers, and at least one geometry for validation.
5. Report posterior predictive intervals. Until those exist, use the ranges in
   this document as sensitivity intervals and flag out-of-range predictions.

### Acceptance checks

For a calibrated reference family, require:

- thrust and torque errors reported alongside acoustic errors;
- band-reduction median error below about 2 dB on held-out conditions;
- tonal frequencies correct by construction and tonal amplitudes without a
  systematic harmonic trend;
- no claimed improvement when the confidence/predictive interval overlaps a
  material acoustic penalty;
- matched-thrust results as the primary design conclusion.

Acoustic measurements should use a calibrated engineering or precision method
appropriate to the facility, with background and reflection checks. Propeller
aerodynamic performance should be reduced to stated atmospheric conditions and
include uncertainty in the thrust balance and RPM control.

## Model constants versus calibrated observables

The research-control model currently uses local relief and coherence constants
to produce the reference reductions. Those internal constants are deliberately
not exposed as global dB knobs: applying a fixed 5 dB multiplier at every
frequency would erase band, radial, and coherence behavior. Regression tests
instead require the reference toroidal and combined 1–5 kHz outputs to remain
inside the empirical intervals. When measurements become available, tune the
local constants jointly against spectra and directivity while keeping these
observable tests.

## References

- Toroidal-propeller numerical/experimental study, *Drones* 8(12), 753:
  <https://doi.org/10.3390/drones8120753>
- AIAA 2026 toroidal-propeller experiment:
  <https://doi.org/10.2514/6.2026-0025>
- Multi-propeller serration experiment including aerodynamic trade-offs,
  *Applied Acoustics* (2025):
  <https://doi.org/10.1016/j.apacoust.2025.111045>
- Lee et al., trailing-edge serration noise and aerodynamic performance:
  <https://doi.org/10.1016/j.apacoust.2018.11.020>
- Leading-edge serration response under turbulent inflow:
  <https://doi.org/10.1063/1.4907798>
- Serration wavelength relative to turbulence length scale, *Journal of Fluid
  Mechanics*: <https://doi.org/10.1017/jfm.2017.141>
- Brooks, Pope, and Marcolini, baseline airfoil self-noise framework, NASA
  RP-1218: <https://ntrs.nasa.gov/citations/19890016302>
- ISO 3744 sound-power engineering method:
  <https://www.iso.org/standard/52055.html>
