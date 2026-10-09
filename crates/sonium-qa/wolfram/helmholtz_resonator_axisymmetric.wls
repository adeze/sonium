(* Literal reference for Wolfram PDEModels/Acoustics/ModelCollection/
   HelmholtzResonator. The geometry, unit mouth pressure, hard walls, and
   0.82 r_neck correction follow the published page. *)
Needs["NDSolve`FEM`"];

cAir = 343.2;
rhoAir = 1.204;
l = 0.05;
rNeck = 0.01;
rCavity = 0.05;
lTotal = l + 0.82 rNeck;
formula = cAir/(2 Pi) Sqrt[rNeck^2/(lTotal (4/3) rCavity^3)];

neck = Rectangle[{0, -l}, {rNeck, 0}];
cavity = Disk[{0, -(l + rCavity - rNeck/2)}, rCavity, {3 Pi/2, 5 Pi/2}];
bottle = RegionUnion[cavity, neck];
mesh = ToElementMesh[bottle, MaxCellMeasure -> {"Length" -> 0.0015}];
cavityMesh = ToElementMesh[cavity, MaxCellMeasure -> {"Length" -> 0.0015}];

vars = {p[r, z], omega, {r, z}};
pars = <|
  "MassDensity" -> rhoAir,
  "SoundSpeed" -> cAir,
  "RegionSymmetry" -> "Axisymmetric"
|>;
model = AcousticPDEComponent[vars, pars];
mouth = AcousticPressureCondition[
  z == 0,
  vars,
  pars,
  <|"AcousticPressure" -> 1|>
];
solution = ParametricNDSolveValue[
  {model == 0, mouth},
  p,
  Element[{r, z}, mesh],
  {omega}
];
frequencies = Table[formula (0.76 + 0.02 index), {index, 0, 24}];
area = NIntegrate[1, Element[{r, z}, cavityMesh]];
response = Table[
  field = solution[2 Pi frequency];
  {frequency, NIntegrate[Abs[field[r, z]], Element[{r, z}, cavityMesh]]/area},
  {frequency, frequencies}
];
numericalPeak = First[First[MaximalBy[response, Last]]];

Print[ExportString[
  <|
    "schema_version" -> 1,
    "provenance" -> "literal Wolfram Helmholtz Resonator PDEModels geometry",
    "neck_length_m" -> l,
    "neck_radius_m" -> rNeck,
    "cavity_radius_m" -> rCavity,
    "resonance_formula_hz" -> N[formula],
    "numerical_peak_hz" -> N[numericalPeak],
    "published_numerical_peak_hz" -> 180.158,
    "published_formula_error_percent" -> 2.6811
  |>,
  "RawJSON",
  "Compact" -> True
]];
