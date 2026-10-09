(* Literal Wolfram Acoustic Wave Diffraction double-slit reference.
   Emits angular extrema on a 0.5 m receiver arc as machine-readable JSON. *)

Needs["NDSolve`FEM`"];
$HistoryLength = 0;

vars = {p[x, y], omega, {x, y}};
cAir = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoAir = QuantityMagnitude[ThermodynamicData["Air", "Density"]];
parameters = <|"SoundSpeed" -> cAir, "MassDensity" -> rhoAir|>;

bottom = 0.;
radius = 0.6;
chamberHeight = 0.25;
slitWidth = 0.03;
slitSeparation = 0.2;
screenThickness = 0.01;
wavelength = 0.1;
frequency = cAir/wavelength;
incidentPressure = 5.;

domain = RegionUnion[
   Disk[{0, chamberHeight}, radius, {0, Pi}],
   Rectangle[{-radius, bottom}, {radius, chamberHeight - screenThickness}],
   Rectangle[{
     -slitSeparation/2 - slitWidth/2, chamberHeight - screenThickness}, {
     -slitSeparation/2 + slitWidth/2, chamberHeight}],
   Rectangle[{
     slitSeparation/2 - slitWidth/2, chamberHeight - screenThickness}, {
     slitSeparation/2 + slitWidth/2, chamberHeight}]
   ];

incoming = AcousticRadiationValue[
   y == bottom, vars, parameters,
   <|"SoundIncidentPressure" -> incidentPressure|>];
outgoing = AcousticAbsorbingValue[
   y >= chamberHeight - screenThickness, vars, parameters];
pde = AcousticPDEComponent[vars, parameters] == incoming + outgoing /. 
   omega -> 2 Pi frequency;

solution = NDSolveValue[
   pde, p, Element[{x, y}, domain],
   Method -> {"PDEDiscretization" -> {"FiniteElement",
       "MeshOptions" -> {"MaxCellMeasure" -> {
           "Length" -> wavelength/12}}}}];

receiverRadius = 0.5;
pressureMagnitude[angle_] := Abs@solution[
   receiverRadius Sin[angle],
   chamberHeight + receiverRadius Cos[angle]];
window = 8 Degree;
sampleCount = 41;
extremum[expected_, maximum_] := Module[{angles, values, ordering},
  angles = Subdivide[expected - window, expected + window, sampleCount - 1];
  values = pressureMagnitude /@ angles;
  ordering = Ordering[values, If[maximum, -1, 1]];
  First@angles[[ordering]]
  ];

analyticMaxima = ArcSin[# wavelength/slitSeparation] & /@ {-1., 0., 1.};
analyticMinima = ArcSin[# wavelength/slitSeparation] & /@ {-1.5, -0.5, 0.5, 1.5};
numericalMaxima = extremum[#, True] & /@ analyticMaxima;
numericalMinima = extremum[#, False] & /@ analyticMinima;

degrees[values_] := N[values 180/Pi];
Print@ExportString[
  <|"schemaVersion" -> 1,
    "analyticMaximaDegrees" -> degrees[analyticMaxima],
    "analyticMinimaDegrees" -> degrees[analyticMinima],
    "numericalMaximaDegrees" -> degrees[numericalMaxima],
    "numericalMinimaDegrees" -> degrees[numericalMinima]|>,
  "RawJSON", "Compact" -> True];
