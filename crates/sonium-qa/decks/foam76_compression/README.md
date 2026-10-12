# Actual quadratic LAW76 compression replay

This original single-brick SI fixture explicitly selects material 1, LAW76,
with `IFORM=0`, `IQUAD=1`, `ICONV=1`, plastic Poisson ratio 0.5, no rates or
failure. All nodal kinematics are prescribed; this proves material evolution
and does not claim a solved spatial foam field or reproduce the original
LOI76-directory LAW66 shell fixture.

E=350 MPa, elastic Poisson ratio=0, density=433 kg/m³, side=10 mm.
Initial tension/compression/shear strengths are 500/900/420 kPa. All strengths
have multiplier `1+20 ep` through ep=0.5. The engine shear curve abscissa is
`sqrt(3)/2 ep`; its last point is therefore sqrt(3)/4, with ordinate 11 times
initial shear stress. The native quadratic surface has A0=3 tau²,
A1=9 tau²(C-T)/(CT), A2=9(CT-3 tau²)/(CT). These curves are convex without
an engine correction. Paths stay within the isochoric pressure-return domain.

The load reaches −0.04 mm at 1 ms, holds until 1.5 ms,
returns to zero at 2.5 ms, and holds until 3 ms. The completed independent
engine run has zero starter errors/warnings and 120 TH stamps. Full integration
Isolid=18 and DT/BRICK factor 0.05 avoid a reduced-integration hourglass replay.

Native replay integrates logarithmic axial strain from the imposed displacement,
not harvested stresses, plastic strains, or energies. It compares all six
stresses, plastic strain, density, displacement, and midpoint stress work
at every stamp. The pressure-blind counterfactual is separately evolved and
must disagree by at least 1% of the 500 kPa stress scale. Stress error must
be <=0.2%, plastic-strain/work error <=0.5%, density/geometry error <=1e-5.
No output-dependent bounds or input calibration is used.

Reproduce in a scratch directory:

```sh
python3 scripts/generate_foam76_deck.py RUN_DIRECTORY --mode compression
```

Run `starter_linux64_gf -i foam76_cube_0000.rad > starter.log`, then
`engine_linux64_gf -i foam76_cube_0001.rad > engine.log`, then
`th_to_csv_linux64_gf foam76_cubeT01` in that directory. Use `RAD_CFG_PATH`
for the engine hm_cfg_files and `LD_LIBRARY_PATH` for extlib/hm_reader/linux64.
Archive only the completed clean run:

```sh
python3 scripts/generate_foam76_deck.py OUTPUT_DIRECTORY --mode compression \
  --harvest-from RUN_DIRECTORY --engine-exe ENGINE_EXECUTABLE
cargo run -p sonium-qa --features qa-binaries --bin qa-fem-foam-laws
```

The generator reuses the original single-brick kinematic scaffold in
`scripts/generate_foam33_deck.py`. Golden metadata records deck, CSV, engine
log and executable SHA-256, engine commit, material parameters and path.
The binary checks archived artifact hashes, model/path and CSV mapping before replay.
