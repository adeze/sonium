(* Literal linear part of Wolfram's Beam-Spring-Mass example. *)

Needs["NDSolve`FEM`"];
$HistoryLength = 0;

length = 0.5;
height = 0.1;
thickness = height;
domain = Rectangle[{0, 0}, {length, height}];
mesh = ToElementMesh[domain];
parameters = <|
   "ShearModulus" -> 441000.,
   "Thickness" -> thickness,
   "PoissonRatio" -> 0.49|>;
variables = {{u[x, y], v[x, y]}, {x, y}};
operator = SolidMechanicsPDEComponent[variables, parameters];
fixed = SolidFixedCondition[x == 0, variables, parameters];
load = SolidBoundaryLoadValue[
   x == length, variables, parameters, <|"Force" -> {0, -force}|>];
solution = ParametricNDSolveValue[
   {operator == load, fixed}, variables[[1]],
   Element[variables[[-1]], mesh], {force}];

forces = {10., 25., 50., 75., 100.};
displacements = solution /@ forces;
point = {length, height/2};
displacementMagnitudes = Sqrt[
     Total[(displacements /. Thread[{x, y} -> point])^2, {2}]];
data = Transpose[{displacementMagnitudes, forces}];
interpolation = Interpolation[data];
derivative = D[interpolation[position], position];
springConstant = Mean[Head[derivative]["ValuesOnGrid"]];
massDensity = 1100.;
mass = massDensity Area[MeshRegion[mesh]] thickness;

Print@ExportString[
  <|"schemaVersion" -> 1,
    "forcesNewton" -> forces,
    "displacementsMeter" -> displacementMagnitudes,
    "springConstantNewtonPerMeter" -> springConstant,
    "massKilogram" -> mass,
    "publishedSpringConstantNewtonPerMeter" -> 256.322|>,
  "RawJSON"];
