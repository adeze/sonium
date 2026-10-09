# PC-fan empirical calibration study

Status: engineering prior, not a completed product calibration. Last reviewed
2026-07-15.

## Scope and interpretation

The source and propagation equations in this crate have been checked for units,
reference-distance handling, and pressure-versus-power conversions. That does
not determine the absolute broadband level, installed cavity response, or the
correct operating target. Those quantities need measurements. This document
separates literature-supported scaling laws from provisional empirical
coefficients and defines the priors used by `pcfan-opt`.

The reference article is a conventional 120 mm, eight-blade PC fan at 600 RPM.
`bench` means free-air or a low-reflection flow bench. `chassis` means the same
fan installed behind realistic restrictions in a mid-tower enclosure. Airflow
is a target supplied to the optimizer, not flow prescribed to the CFD solver.

## Airflow target

### Why 55 CFM at 600 RPM is not credible

55 CFM is 0.02596 m3/s. With the model's 54 mm blade-tip and 18 mm hub radii,
the annulus area is about 0.00814 m2, giving a mean axial velocity of 3.19 m/s.
The tip speed at 600 RPM is 3.39 m/s, so the implied flow coefficient
`phi = V_axial / U_tip` is about 0.94. That is far outside a sensible operating
region for a compact axial fan with a hub, frame, shroud clearance, and finite
blade loading. It also treats a typical full-speed catalogue airflow as though
it were available at one third to one fifth of catalogue speed.

Manufacturer free-air ratings give a useful first prior when reduced to 600 RPM
with the fan similarity law `Q proportional to N`. Representative 120 mm data
are:

| Fan | Rated free-air point | Similarity-law value at 600 RPM |
| --- | ---: | ---: |
| ARCTIC P12 PWM PST | 56.3 CFM at 1800 RPM | 18.8 CFM |
| ARCTIC P12 Silent | 24.1 CFM at 1050 RPM | 13.8 CFM |
| ARCTIC P12 Max | 81.0 CFM at 3300 RPM | 14.7 CFM |
| ARCTIC F12 | about 53 CFM at 1350 RPM | about 23.6 CFM |
| be quiet! Silent Wings 4 120 mm PWM high-speed | 76.7 CFM at 2500 RPM | 18.4 CFM |

This is not a substitute for a low-speed fan curve: motor commutation, leakage,
Reynolds number, and the test fixture prevent exact linear scaling. It is,
however, sufficient to reject 55 CFM and establish a defensible initial range.

The adopted 600 RPM priors are:

| Condition | Central target | Sensitivity interval |
| --- | ---: | ---: |
| Free-air bench | 20 CFM | 14–24 CFM |
| Chassis installed | 15 CFM | 10–20 CFM |

For other speeds, the binary scales these values linearly with RPM. Treat this
as a similarity prior, especially outside roughly 600–3300 RPM. An explicit
`--target-airflow-cfm` remains available and emits a warning outside the
RPM-scaled interval.

The installed centre is 25% below free air. A real system operating point must
instead be found by intersecting the measured fan and system-resistance curves.
The 10–20 CFM interval deliberately covers weak and severe restrictions.

## Broadband source coefficients

The Brooks-Pope-Marcolini, Amiet, tip-leakage, rotor-stator-interaction, support,
and motor terms provide useful frequency, velocity, geometry, and turbulence
scaling. Their absolute multipliers are not universal literature constants.
They depend on boundary-layer state, inflow spectrum, surface finish, clearance,
strut placement, motor structure, reference distance, and the measurement
fixture. The current values are therefore prior medians:

| Contribution | Model coefficient | Calibration interval | Uncertainty treatment |
| --- | ---: | ---: | --- |
| Trailing-edge broadband | `4.0e-3` | `1.3e-3`–`1.2e-2` | factor 3 |
| Leading-edge inflow | `8.5e-6` | `8.5e-7`–`8.5e-5` | factor 10 |
| Tip leakage | `1.5e-3` | `3.0e-4`–`7.5e-3` | factor 5 |
| Rotor-stator interaction | `7.35e-5` | `2.45e-5`–`2.21e-4` | factor 3 |
| Support broadband | `6.0e-4` | `1.2e-4`–`3.0e-3` | factor 5 |
| Motor broadband | `2.5e-4` | `5.0e-5`–`1.25e-3` | factor 5 |
| Motor tone | `1.4e-4` | `2.8e-5`–`7.0e-4` | factor 5 |

These intervals are sensitivity bounds, not statistical confidence intervals.
They should be replaced by posterior distributions after calibration. Fit
source families separately where possible; fitting one overall gain cannot
identify which mechanism is wrong and will not extrapolate across RPM,
clearance, or inflow turbulence.

Recommended measurement separation:

1. Clean inflow, several RPMs, several tip clearances: fit trailing-edge and
   tip-leakage terms.
2. Repeat with a characterized turbulence grid: fit the leading-edge term to
   measured turbulence intensity and integral length scale.
3. Change blade/strut count or clocking while holding the rotor fixed: identify
   rotor-stator interaction.
4. Locked-rotor or rotor-removed motor/support measurements: bound fixed
   hardware contributions.
5. Fit in one-third-octave or narrow bands, retaining an absolute microphone
   calibration and background-noise subtraction.

## Chassis cavity calibration

`PcFanEnclosureConfig::modal_damping` is a modal loss factor `eta = 1/Q`, not a
viscous damping ratio. A single value cannot represent panel radiation,
openings, porous absorbers, leakage, and frequency-dependent loss, but it is a
useful first regularization of the rigid-box modal sum.

The adopted prior is `eta = 0.08` (`Q = 12.5`) with a required sensitivity
range of `0.03–0.20` (`Q` about 33–5). The previous 0.05 value was defensible
for a lightly damped cavity but too sharp as an uncalibrated general default.
The binary accepts `--modal-loss-factor` only inside the sensitivity interval.

Calibrate the cavity with the fan source removed or held constant:

1. Measure swept-sine transfer functions from the fan plane to all listener or
   vent locations used in the simulation.
2. Estimate modal centre frequencies and half-power bandwidths, then initialize
   `eta_n = Delta f_n / f_n`.
3. Fit source/receiver coupling and loss independently; do not absorb incorrect
   positions or open-boundary leakage into damping.
4. Validate on held-out source and microphone positions. If residuals remain
   structured, use frequency-dependent/modal damping or measured transfer
   functions rather than forcing one loss factor.

## Calibration and validation design

Use at least three nominally identical samples and randomize test order. Record
RPM, voltage/current, ambient pressure, temperature, humidity, static pressure,
flow rate, microphone geometry, background spectrum, turbulence intensity, and
integral length scale. Suggested operating matrix:

- 400, 600, 900, 1200, and the fan's rated maximum RPM;
- free delivery plus at least four static-pressure points;
- clean inflow and two characterized turbulence grids;
- at least three tip clearances;
- free-field and two representative chassis restriction states.

Fit on a subset of RPM/pressure points and validate on the remainder. Report
band errors, overall A-weighted error, tonal frequency/amplitude error, and
airflow/static-pressure error. An engineering release target is median band
error below 3 dB with no systematic RPM trend; stricter claims require a stated
uncertainty budget.

Air performance should follow ISO 5801 or ANSI/AMCA 210. Acoustic power should
follow an appropriate calibrated sound-power method such as ISO 3744, while IT
equipment installations can also use ECMA-275 conventions. Catalogue airflow
is useful only when its test condition is known.

## References

- Brooks, Pope, and Marcolini, *Airfoil Self-Noise and Prediction*, NASA
  RP-1218: <https://ntrs.nasa.gov/citations/19890016302>
- Amiet, high-frequency gust response / leading-edge noise:
  <https://doi.org/10.1016/S0022-460X(75)80105-2>
- Fukano and Jang, tip-clearance and rotating-blade noise:
  <https://doi.org/10.1016/S0022-460X(03)00815-0>
- Automotive cavity damping study:
  <https://doi.org/10.4271/2020-01-1524>
- ISO 5801 fan performance testing:
  <https://www.iso.org/standard/53963.html>
- ANSI/AMCA 210 laboratory fan aerodynamic performance: AMCA 210-25,
  Air Movement and Control Association official publication page.
- ISO 3744 sound-power engineering method:
  <https://www.iso.org/standard/52055.html>
- ECMA-275 measurement of airborne acoustical noise from information
  technology and telecommunications equipment:
  <https://ecma-international.org/publications-and-standards/standards/ecma-275/>
- ARCTIC P12 product family: <https://www.arctic.de/en/P12-PWM-PST/ACFAN00120A>
- ARCTIC P12 Max: <https://www.arctic.de/en/P12-Max/ACFAN00280A>
- ARCTIC P12 Silent: <https://www.arctic.de/en/P12-Silent/ACFAN00130A>
- be quiet! Silent Wings 4 120 mm PWM high-speed:
  <https://www.bequiet.com/en/casefans/3703>

