(* Literal extractor for SolidMechanics-FEM-Eigenmode-3D-0001. *)

Needs["NDSolve`FEM`"];

vars = {{u[x, y, z], v[x, y, z], w[x, y, z]}, {x, y, z}};
pars = <|
  "YoungModulus" -> 210*10^9,
  "PoissonRatio" -> 0.28,
  "MassDensity" -> 7800
|>;
operator = SolidMechanicsPDEComponent[vars, pars];
fixed = SolidFixedCondition[x == 0, vars, pars];
mesh = ToElementMesh[
  Cuboid[{0, 0, 0}, {1/2, 0.05, 0.02}],
  "MaxCellMeasure" -> 0.00001,
  "MeshElementType" -> TetrahedronElement
];
{values, modes} = NDEigensystem[
  {operator, fixed},
  {u[x, y, z], v[x, y, z], w[x, y, z]},
  Element[{x, y, z}, mesh],
  1
];
frequency = Sqrt[values[[1]]/pars["MassDensity"]]/(2*Pi);
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "SolidMechanics-FEM-Eigenmode-3D-0001",
    "dimensions_m" -> {1/2, 0.05, 0.02},
    "young_modulus_pa" -> 210*10^9,
    "poisson_ratio" -> 0.28,
    "density_kg_m3" -> 7800,
    "reference_frequency_hz" -> 67.0,
    "computed_frequency_hz" -> frequency
  |>,
  "RawJSON"
]];
