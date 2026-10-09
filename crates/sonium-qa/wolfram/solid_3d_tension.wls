(* Literal extractor for SolidMechanics-FEM-Stationary-3D-0001. *)

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
  x == 50, vars, pars, <|"Force" -> {10000, 0, 0}|>
];
mesh = ToElementMesh[
  Cuboid[{0, 0, 0}, {50, 10, 20}],
  "MaxCellMeasure" -> 10,
  "MeshElementType" -> TetrahedronElement
];
displacement = NDSolveValue[
  {operator == load, fixed},
  {u[x, y, z], v[x, y, z], w[x, y, z]},
  Element[{x, y, z}, mesh]
];
strain = SolidMechanicsStrain[vars, pars, displacement];
stress = SolidMechanicsStress[vars, pars, strain];
vonMises = VonMisesStress[vars, pars, stress];
samplePoints = Flatten[
  Table[{xSample, ySample, zSample},
    {xSample, 30, 50, 5}, {ySample, 0, 10, 5}, {zSample, 0, 20, 5}],
  2
];

Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "SolidMechanics-FEM-Stationary-3D-0001",
    "dimensions_mm" -> {50, 10, 20},
    "young_modulus_n_mm2" -> 210000,
    "poisson_ratio" -> 0.3,
    "total_force_n" -> 10000,
    "reference_elongation_mm" -> 0.012,
    "reference_axial_stress_n_mm2" -> 50,
    "maximum_elongation_mm" -> Max[displacement[[1]]["ValuesOnGrid"]],
    "sample_points_mm" -> samplePoints,
    "sigma_xx_n_mm2" -> (stress[[1, 1]] @@ # & /@ samplePoints),
    "von_mises_n_mm2" -> (vonMises @@ # & /@ samplePoints)
  |>,
  "RawJSON"
]];
