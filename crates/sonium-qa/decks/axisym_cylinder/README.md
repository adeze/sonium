# Axisymmetric cylinder reference

This original fixture uses `/ANALY N2D3D=1` and 64 `/QUAD` elements in the
YZ plane: Y is radius and Z is axial. The cylinder has radii 0.05/0.10 m,
length 0.05 m, E=200 GPa, nu=0.3 and density 7800 kg/m³. Axial translations
are fixed for plane strain; radial translations are free. Bore pressure
ramps smoothly to 1 MPa over 0.002 s and holds until 0.0025 s.

The engine's axisymmetric CLOAD convention applies the radius weight during
force/work integration (`engine/source/loads/general/force.F90`, `n2d=1`
branch). Deck nodal loads therefore integrate pressure over axial length per
unit circumference. Output energies are per radian.

`history.csv` and `engine-output.txt` were harvested from a normally terminated
OpenCourant double-precision run. `golden.json` records deck, engine, log and
history SHA-256 checksums. Its 80 pressure-hold samples start at 0.0021 s.
The QA gate verifies the saved artifacts, compares native static displacement
and strain energy with this hold, and checks native refinement against Lamé
fields. It does not compare the transient ramp.

Regenerate decks with `python3 scripts/generate_axisymmetric_deck.py <scratch>`.
Run the starter and engine there using `-i axisym_cylinder_0000.rad` and
`-i axisym_cylinder_0001.rad`, saving engine stdout to `engine.log`. Convert
`axisym_cylinderT01` with `th_to_csv_linux64_gf`. Then harvest with:

```sh
python3 scripts/generate_axisymmetric_deck.py crates/sonium-qa/decks/axisym_cylinder \
  --harvest-from <scratch> --engine-exe <engine-binary>
```

The harvest command copies the converted CSV and engine log into this
directory. Run `qa-fem-axisym` to verify the native comparison.
