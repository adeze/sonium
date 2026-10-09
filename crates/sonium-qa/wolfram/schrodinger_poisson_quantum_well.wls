(* Literal Wolfram model-collection Schrödinger--Poisson quantum well.
   The final line is the machine-readable contract consumed by the optional
   WOLFRAMSCRIPT path in the Rust accuracy test. *)

Needs["NDSolve`FEM`"];

eg[x_] := 1426 + 1247 x;
confinement[z_] := Piecewise[{{0, 200 <= z <= 300}}, 0.66 (eg[0.2] - eg[0])];
domain = Line[{{0}, {500}}];
mesh = ToElementMesh[domain, "MaxCellMeasure" -> 10];
doping[z_] := Piecewise[{{QuantityMagnitude[Quantity[2 10^18, "Centimeters"^-3], "SIBase"], 200 <= z <= 300}}, 0];
sheetDensity = NIntegrate[doping[z] 10^-10, {z, 0, 500}];
electronCharge = -QuantityMagnitude[Quantity[1, "ElementaryCharge"], "SIBase"];
wavefunctionScale = QuantityMagnitude[Quantity[1, 1/Sqrt["Angstroms"]], 1/Sqrt["Meters"]];

chargeDensity[z_, unscaled_] := Module[{scaled = wavefunctionScale unscaled[z]},
  electronCharge (sheetDensity scaled Conjugate[scaled] - doping[z])
];

schVars = {psi[z], {z}};
schPars = <|
  "SchrodingerPotential" -> Quantity[confinement[z], "Millielectronvolts"],
  "ReducedPlanckConstant" -> Quantity[1, "ReducedPlanckConstant"],
  "ScaleUnits" -> {"Meters" -> "Angstroms"},
  "Mass" -> 0.066 Entity["Particle", "Electron"][EntityProperty["Particle", "Mass"]]
|>;
poissonVars = {phi[z], {z}};
poissonPars = <|
  "RelativePermittivity" -> 13.18,
  "VolumeChargeDensity" -> Quantity[chargeDensity[z, wavefunction], "Coulombs"/"Meters"^3],
  "ScaleUnits" -> {"Meters" -> "Angstroms"}
|>;
poissonBc = ElectricPotentialCondition[z == 0 || z == 500, poissonVars, poissonPars,
  <|"ElectricPotential" -> 0|>];
potentialScale = QuantityMagnitude[
  Quantity[1, "Angstroms"^2 "Kilograms"/("Amperes" "Seconds"^3)],
  "Millielectronvolts"/"Coulombs"
];

solveSchrodinger[extra_] := Module[{operator, energy, wavefunction},
  schPars["SchrodingerPotential"] = Quantity[confinement[z], "Millielectronvolts"] + extra;
  operator = SchrodingerPDEComponent[schVars, schPars];
  {{energy}, {wavefunction}} = NDEigensystem[operator, psi, Element[{z}, mesh], 1];
  {energy, wavefunction}
];
solvePoisson[wavefunction_] := Module[{operator},
  poissonPars["VolumeChargeDensity"] = Quantity[chargeDensity[z, wavefunction], "Coulombs"/"Meters"^3];
  operator = ElectrostaticPDEComponent[poissonVars, poissonPars];
  NDSolveValue[{operator == 0, poissonBc}, phi, Element[z, mesh]]
];
toMeV[value_] := QuantityMagnitude[UnitConvert[
  Quantity[value, "Kilograms" "Angstroms"^2/"Seconds"^2], "Millielectronvolts"]];

{energy, wavefunction} = solveSchrodinger[0 Quantity[1, "Millielectronvolts"]];
energies = {toMeV[energy]};
Do[
  potential = solvePoisson[wavefunction];
  {energy, wavefunction} = solveSchrodinger[
    electronCharge potentialScale potential[z] Quantity[1, "Millielectronvolts"]];
  AppendTo[energies, toMeV[energy]],
  {5}
];

payload = <|"schema_version" -> 1, "energies_mev" -> N[energies, 12]|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
