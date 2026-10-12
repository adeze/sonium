This original SI fixture checks the plain `/MAT/FOAM_PLAS` LAW33 card,
Ka=0 without tabulated/rate multipliers. Its parameters equal the existing
FOAM_V10 material card after unit conversion: density 433 kg/m3, E=350 MPa,
A=-177 kPa, B=677 kPa, C=-3.34, P0=100 kPa, phi=0.1, gamma0=0.

The 10 mm cube uses one fully integrated Isolid=18 brick. The left face
is fixed; the right face has prescribed axial displacement and fixed
transverse displacement. It compresses by 5% over 1 ms, holds for 0.5 ms,
unloads to its original geometry over 1 ms, and holds for 0.5 ms. The
engine brick timestep factor is 0.05. All kinematics are prescribed, so
this fixture is a material-history comparison, not a native spatial solve.
It does not reproduce the original FOAM_V10 multi-element falling-pad case.

The double-precision OpenCourant run completed normally with 5820 cycles
and no starter errors or warnings. `golden.json` records deck, CSV, log,
and engine executable SHA-256 checksums and engine commit
33e685176cccf0c539a3ce07aa2096985a284e2a. `history.csv` retains every TH
stamp. The converter labels brick variables 24..34: activity, six C1
stresses, energy, density, and two unused fields. Moving-node variable 35
is axial displacement. QA validates this header mapping before reading it.

Reproduce from the repository:

```sh
python3 scripts/generate_foam33_deck.py RUN_DIRECTORY
```

In that scratch directory run `starter_linux64_gf -i foam33_cube_0000.rad`
with stdout redirected to `starter.log`, then
`engine_linux64_gf -i foam33_cube_0001.rad > engine.log`, then
`th_to_csv_linux64_gf foam33_cubeT01`. Set `RAD_CFG_PATH` to the engine's
`hm_cfg_files` directory and `LD_LIBRARY_PATH` to `extlib/hm_reader/linux64`
if required. Archive the completed run:

```sh
python3 scripts/generate_foam33_deck.py OUTPUT_DIRECTORY \
  --harvest-from RUN_DIRECTORY --engine-exe ENGINE_EXECUTABLE
cargo run -p sonium-qa --features qa-binaries --bin qa-fem-foam-laws
```

Native `FoamPlas33` follows the plain-card equations verified in engine
`mat033/sigeps33.F`, CASE(1). It takes corotational strain increments and
a geometrical volume ratio supplied independently by the caller. QA drives
it from the imposed displacement curve; no harvested stresses, energy,
or density are fed into the native history. Each TH interval is substepped
and split at load-path corners. Work integrates `sigma V0 dJ`, appropriate
for fixed transverse area. A second native replay disables trapped-air
pressure and must differ detectably from the harvested stress history.

The separate original LOIS integrity case reports zero numerical solves.
LAW33 and LAW51 material histories are separate physics cases; the foam P1
remains pending until the native solved spatial foam cases are verified.
