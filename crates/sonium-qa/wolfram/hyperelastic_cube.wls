(* Literal extractor for the stationary compressible Neo-Hookean cube in
   Wolfram's Hyperelasticity tutorial. Wolfram execution is opt-in. *)

Needs["NDSolve`FEM`"];

cubeMesh = ToElementMesh[
  Cuboid[], MaxCellMeasure -> 0.05,
  "MeshElementType" -> TetrahedronElement
];
vars = {{u[x, y, z], v[x, y, z], w[x, y, z]}, {x, y, z}};
pdeModel[unknowns_, parameters_] := {
  SolidMechanicsPDEComponent[unknowns, parameters] == {0, 0, 0},
  SolidFixedCondition[x == 0, unknowns, parameters],
  SolidDisplacementCondition[x == 1, unknowns, parameters,
    <|"Displacement" -> {p, None, None}|>]
};

c10 = 2881.;
d1 = 10.^-6;
parameters = <|
  "SolidMechanicsMaterialModel" -> "NeoHookean",
  "ShearModulus" -> 2. c10,
  "BulkModulus" -> 2./d1,
  "MassDensity" -> 980.,
  "Compressibility" -> "Compressible"
|>;
extension = 0.004;
(* ParametricNDSolveValue in current command-line Wolfram Engine releases can
   pass an unevaluated displacement parameter into the nonlinear FEM residual,
   producing ParametricNDSolveValue::fempsf/FindRoot::dfmin before Newton even
   starts. This extractor needs one published load only, so substitute that
   value before discretization and solve the resulting numerical problem. *)
solution = NDSolveValue[
  pdeModel[vars, parameters] /. p -> extension,
  {u, v, w},
  {x, y, z} \[Element] cubeMesh,
  AccuracyGoal -> 6, PrecisionGoal -> 6,
  Method -> {"PDEDiscretization" -> {"FiniteElement",
    "PDESolveOptions" -> {
      "FindRootOptions" -> {Method -> "Newton", DampingFactor -> 0.25,
        MaxIterations -> 200}
    }}}
];
samplePoints = {{0.25, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.75, 0.5, 0.5}};
sampleDisplacements = N[
  ({solution[[1]][Sequence @@ #], solution[[2]][Sequence @@ #],
      solution[[3]][Sequence @@ #]} &) /@ samplePoints, 16
];

result = <|
  "schema_version" -> 1,
  "case_id" -> "Hyperelasticity-NeoHookean-Stationary-Cube",
  "dimensions_m" -> {1., 1., 1.},
  "shear_modulus_pa" -> parameters["ShearModulus"],
  "bulk_modulus_pa" -> parameters["BulkModulus"],
  "mass_density_kg_m3" -> parameters["MassDensity"],
  "prescribed_extension_m" -> extension,
  "sample_points_m" -> samplePoints,
  "sample_displacement_m" -> sampleDisplacements
|>;
WriteString[$Output, ExportString[result, "RawJSON", "Compact" -> True], "\n"];
