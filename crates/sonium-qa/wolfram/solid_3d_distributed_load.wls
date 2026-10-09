(* Literal extractor for SolidMechanics-FEM-Stationary-3D-0004. *)

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
  z == 40, vars, pars, <|"Force" -> {0, 0, -50*200}|>
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
maximumDeflection = Max[Abs[displacement[[3]]["ValuesOnGrid"]]];
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "SolidMechanics-FEM-Stationary-3D-0004",
    "dimensions_mm" -> {200, 40, 40},
    "young_modulus_n_mm2" -> 210000,
    "poisson_ratio" -> 0.3,
    "top_force_n" -> {0, 0, -10000},
    "reference_deflection_mm" -> N[25/112, 16],
    "computed_maximum_deflection_mm" -> maximumDeflection
  |>,
  "RawJSON"
]];
