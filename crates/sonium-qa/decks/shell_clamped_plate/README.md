This original circular plate fixture is written by
`sonium_qa::rad::shell_clamped_plate_case`. It is a clamped elastic disk
(radius 1 m, thickness 0.02 m, E = 200 GPa, nu = 0.3, density 7800 kg/m3)
under 1000 Pa downward pressure. The 8-ring, 32-sector mesh has QEPH
quadrilaterals and center triangles. All six rim DOFs are fixed. Tributary
nodal areas distribute the pressure; their sum is pi m2. The load ramps
linearly to full pressure at 0.8 s and holds through 1 s.

`history.csv` and `engine-output.txt` were harvested from a normally
terminated double-precision OpenCourant engine run, commit
33e685176cccf0c539a3ce07aa2096985a284e2a, on 2026-10-12 (234205 cycles).
`golden.json` records deck, binary, CSV, and log SHA-256 checksums. The
new archive supplements the existing `probes.json`; it does not rewrite
its original reference values.

To reproduce, copy both checked-in `.rad` files into an empty scratch
directory, run `starter_linux64_gf -i shell_clamped_plate_0000.rad`, then
`engine_linux64_gf -i shell_clamped_plate_0001.rad > engine.log`, then
`th_to_csv_linux64_gf shell_clamped_plateT01`. Set `RAD_CFG_PATH` to the
engine installation's `hm_cfg_files` directory and `LD_LIBRARY_PATH` to
its `extlib/hm_reader/linux64` directory if required. From the repository:

```sh
python3 scripts/harvest_shell_plate.py RUN_DIRECTORY ENGINE_EXECUTABLE OUTPUT_DIRECTORY
cargo run -p sonium-qa --features qa-binaries --bin qa-fem-crash-shells
```

The native comparison evolves two independent CST/DKT triangular meshes
(8x32 and 16x64 polar grids) from rest using explicit dynamics and kinetic
damping, with every interior unknown free. It uses the same section,
rim boundary conditions, and tributary pressure prescription. It checks
center displacement and strain energy against the engine's >=0.9 s hold
mean, plus the Kirchhoff circular-plate center displacement, spatial
refinement, physical free-force residual, and strain-energy/work identity.
There is no native dense static solve. Native damping history is not the
engine's physical ramp history, so this gate validates equilibrium only.
The archive's hold variation and kinetic/internal-energy ratio are also
bounded, independently of the native field.

The original CRASH vendor fixtures remain separately reported integrity
checks with zero native solves. This plate gate validates elastic shell
behavior; it does not validate the vendor impact cases' rigid walls,
self-contact, added masses, or rate-dependent plasticity. The audit item
for those CRASH response comparisons remains pending.
