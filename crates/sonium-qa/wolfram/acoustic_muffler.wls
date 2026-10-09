(* Literal extractor for Wolfram's Acoustic Muffler example.
   The two official STL files are proprietary Wolfram installation assets and
   are deliberately not redistributed by Sonium. *)

Needs["NDSolve`FEM`"];
$HistoryLength = 0;

supportDirectory = FileNameJoin[{
   $InstallationDirectory, "SystemFiles", "Components", "PDEModels",
   "SupportFiles"
   }];
classicalFile = FileNameJoin[{supportDirectory, "ClassicalMuffler.stl"}];
perforatedFile = FileNameJoin[{supportDirectory, "PerforatedMuffler.stl"}];

If[!And @@ (FileExistsQ /@ {classicalFile, perforatedFile}),
 Print@ExportString[
   <|"available" -> False,
     "missingFiles" -> Select[{classicalFile, perforatedFile}, Not@*FileExistsQ]|>,
   "RawJSON"];
 Exit[2]
 ];

vars = {p[x, y, z], omega, {x, y, z}};
cAir = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoAir = QuantityMagnitude[ThermodynamicData["Air", "Density"]];
parameters = <|"SoundSpeed" -> cAir, "MassDensity" -> rhoAir|>;

length = 1.;
inletRadius = 0.05;
secondaryRadius = 0.03;
extensionLength = 0.2;
inlet = -extensionLength;
outlet = length + extensionLength;
pipeArea = Pi inletRadius^2;
incidentPressure = 1.;

radiation = AcousticRadiationValue[
   z <= inlet, vars, parameters,
   <|"SoundIncidentPressure" -> incidentPressure,
     "BoundaryUnitNormal" -> {0, 0, -1}|>
   ];
absorbing = AcousticAbsorbingValue[z >= outlet, vars, parameters];

maximumEdgeLength = (cAir/1000.)/12.;
classicalBoundaryMesh = ToBoundaryMesh@Import[classicalFile];
perforatedBoundaryMesh = ToBoundaryMesh@Import[perforatedFile];
classicalMesh = ToElementMesh[
   classicalBoundaryMesh, MaxCellMeasure -> {"Length" -> maximumEdgeLength}];
perforatedMesh = ToElementMesh[
   perforatedBoundaryMesh, MaxCellMeasure -> {"Length" -> maximumEdgeLength}];

frequencies = Range[20., 1000., 10.];
solve[mesh_, pde_] := Module[{solution},
  solution = ParametricNDSolveValue[
    pde, p, Element[{x, y, z}, mesh], {omega}];
  Table[solution[2 Pi frequency], {frequency, frequencies}]
  ];

airPDE = AcousticPDEComponent[vars, parameters] == radiation + absorbing;
classicalSolutions = solve[classicalMesh, airPDE];
perforatedSolutions = solve[perforatedMesh, airPDE];

flowResistivity = 50000.;
padThickness = 0.02;
k = omega/cAir;
characteristicImpedance = rhoAir cAir;
kComplex = k (1 +
     0.136 (rhoAir omega/(2 Pi flowResistivity))^-0.641 -
     0.322 I (rhoAir omega/(2 Pi flowResistivity))^-0.502);
zComplex = characteristicImpedance (1 +
     0.081 (rhoAir omega/(2 Pi flowResistivity))^-0.699 -
     0.191 I (rhoAir omega/(2 Pi flowResistivity))^-0.556);
cComplex = omega/kComplex;
rhoComplex = kComplex zComplex/omega;
materialRules = {
   c -> If[x^2/(0.25 - padThickness)^2 +
       y^2/(0.1 - padThickness)^2 >= 1, cComplex, cAir],
   rho -> If[x^2/(0.25 - padThickness)^2 +
       y^2/(0.1 - padThickness)^2 >= 1, rhoComplex, rhoAir]
   };
paddedPDE = AcousticPDEComponent[
     vars, <|"SoundSpeed" -> c, "MassDensity" -> rho, materialRules|>
     ] == radiation + absorbing;
paddedSolutions = solve[classicalMesh, paddedPDE];

outletPressure[solutions_] := Table[
   Abs[solutions[[index]][0.15, 0., outlet]],
   {index, Length[frequencies]}];
incidentPower = incidentPressure^2/(2 rhoAir cAir) pipeArea;
transmissionLoss[solutions_] := Module[{outletPower},
  outletPower = outletPressure[solutions]^2/(2 rhoAir cAir) pipeArea;
  10 Log[10, incidentPower/outletPower]
  ];

Print@ExportString[
  <|"available" -> True,
    "frequencyHz" -> frequencies,
    "classicalTransmissionLossDb" -> transmissionLoss[classicalSolutions],
    "perforatedTransmissionLossDb" -> transmissionLoss[perforatedSolutions],
    "paddedTransmissionLossDb" -> transmissionLoss[paddedSolutions],
    "airSoundSpeedMetersPerSecond" -> cAir,
    "airDensityKilogramsPerCubicMeter" -> rhoAir,
    "officialSupportFiles" -> {classicalFile, perforatedFile}|>,
  "RawJSON"];
