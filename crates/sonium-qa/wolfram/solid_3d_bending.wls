(* Literal extractor for SolidMechanics-FEM-Stationary-3D-0003. *)

Needs["NDSolve`FEM`"];

vars = {{u[x, y, z], v[x, y, z], w[x, y, z]}, {x, y, z}};
pars = <|
  "YoungModulus" -> 210000,
  "PoissonRatio" -> 0.3,
  "MassDensity" -> 7.85*10^-6
|>;
operator = SolidMechanicsPDEComponent[vars, pars];
fixed = SolidFixedCondition[x == 0, vars, pars];
load = SolidBoundaryLoadValue[
  x == 200, vars, pars, <|"Force" -> {0, 0, -10000}|>
];
mesh = ToElementMesh[
  Cuboid[{0, 0, 0}, {200, 40, 40}],
  "MaxCellMeasure" -> 200,
  "MeshElementType" -> TetrahedronElement
];
displacement = NDSolveValue[
  {operator == load, fixed},
  {u[x, y, z], v[x, y, z], w[x, y, z]},
  Element[{x, y, z}, mesh]
];
probe = Abs[displacement[[3]][200, 20, 0]];
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "SolidMechanics-FEM-Stationary-3D-0003",
    "dimensions_mm" -> {200, 40, 40},
    "young_modulus_n_mm2" -> 210000,
    "poisson_ratio" -> 0.3,
    "total_force_n" -> {0, 0, -10000},
    "probe_mm" -> {200, 20, 0},
    "reference_deflection_mm" -> N[25/42, 16],
    "computed_deflection_mm" -> probe
  |>,
  "RawJSON"
]];
