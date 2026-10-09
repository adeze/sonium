(* Literal extractor for SolidMechanics-FEM-Eigenmode-2D-0001. *)

Needs["NDSolve`FEM`"];

vars = {{u[x, y], v[x, y]}, {x, y}};
pars = <|
  "YoungModulus" -> 210*10^9,
  "PoissonRatio" -> 3/10,
  "MassDensity" -> 7850,
  "Thickness" -> 1,
  "SolidMechanicsModelForm" -> "PlaneStress"
|>;
operator = SolidMechanicsPDEComponent[vars, pars];
fixed = SolidFixedCondition[x == 0, vars, pars];
mesh = ToElementMesh[Rectangle[{0, 0}, {10, 0.7}]];
{values, modes} = NDEigensystem[
  {operator, fixed},
  {u[x, y], v[x, y]},
  Element[{x, y}, mesh],
  1
];
frequency = Sqrt[values[[1]]/pars["MassDensity"]]/(2*Pi);
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "SolidMechanics-FEM-Eigenmode-2D-0001",
    "dimensions_m" -> {10, 0.7},
    "young_modulus_pa" -> 210*10^9,
    "poisson_ratio" -> 3/10,
    "density_kg_m3" -> 7850,
    "reference_frequency_hz" -> 5.85289,
    "computed_frequency_hz" -> frequency
  |>,
  "RawJSON"
]];
