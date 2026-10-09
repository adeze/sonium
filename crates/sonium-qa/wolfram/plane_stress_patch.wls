(* Literal extractor for SolidMechanics-FEM-Stationary-2D-PlaneStress-0001. *)

Needs["NDSolve`FEM`"];

vars = {{u[x, y], v[x, y]}, {x, y}};
pars = <|
  "YoungModulus" -> 3*10^6,
  "PoissonRatio" -> 1/5,
  "SolidMechanicsModelForm" -> "PlaneStress",
  "Thickness" -> 4
|>;
operator = SolidMechanicsPDEComponent[vars, pars];
fixed = DirichletCondition[{u[x, y] == 0, v[x, y] == 0}, x == 54];
roller = DirichletCondition[u[x, y] == 0, x == 0];
load = {0, NeumannValue[-50*4, y == 12]};
coordinates = {
  {0, 5}, {0, 12}, {6, 0}, {6, 5},
  {20, 0}, {20, 12}, {54, 0}, {54, 12}
};
indices = {{1, 4, 6, 2}, {3, 5, 6, 4}, {5, 7, 8, 6}};
mesh = ToElementMesh[
  "Coordinates" -> coordinates,
  "MeshElements" -> {QuadElement[indices]}
];
{ufun, vfun} = NDSolveValue[
  {operator == load, roller, fixed},
  {u, v},
  Element[{x, y}, mesh]
];
nodalDisplacements = {ufun[#[[1]], #[[2]]], vfun[#[[1]], #[[2]]]} & /@ coordinates;
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "SolidMechanics-FEM-Stationary-2D-PlaneStress-0001",
    "coordinates" -> coordinates,
    "quad_indices" -> indices,
    "young_modulus" -> 3*10^6,
    "poisson_ratio" -> 1/5,
    "thickness" -> 4,
    "nodal_displacements" -> nodalDisplacements
  |>,
  "RawJSON"
]];
