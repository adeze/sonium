(* Wolfram verification case
   HeatTransfer-FEM-Transient-2DAxisym-Single-HeatTransfer-0001.
   This script is optional and is not invoked by the Rust test. *)

Needs["NDSolve`FEM`"];

vars = {temperature[t, r, z], t, {r, z}};
parameters = <|
  "ThermalConductivity" -> 52.,
  "MassDensity" -> 7850.,
  "SpecificHeatCapacity" -> 460.,
  "RegionSymmetry" -> "Axisymmetric"
|>;
operator = HeatTransferPDEComponent[vars, parameters];
temperatureCondition = DirichletCondition[
  temperature[t, r, z] == 1000.,
  r == 0.3 || z == 0. || z == 0.4
];
initialCondition = temperature[0., r, z] == 0.;
domain = Rectangle[{0., 0.}, {0.3, 0.4}];
endTime = 190.;
solution = NDSolveValue[
  {operator == 0., temperatureCondition, initialCondition},
  temperature,
  {t, 0., endTime},
  Element[{r, z}, domain]
];

payload = <|
  "schema_version" -> 1,
  "case_id" ->
    "HeatTransfer-FEM-Transient-2DAxisym-Single-HeatTransfer-0001",
  "end_time_s" -> endTime,
  "probe_rz_m" -> {0.1, 0.3},
  "published_probe_temperature" -> 186.5,
  "computed_probe_temperature" -> N[solution[endTime, 0.1, 0.3], 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
