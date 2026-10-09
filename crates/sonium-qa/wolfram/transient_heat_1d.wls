(* Wolfram verification cases
   HeatTransfer-FEM-Transient-1D-Single-HeatTransfer-0001 and -0002.
   This script is optional and is not invoked by the Rust test. *)

Needs["NDSolve`FEM`"];

vars1 = {temperature1[t, x], t, {x}};
parameters1 = <|
  "ThermalConductivity" -> 35.,
  "MassDensity" -> 7200.,
  "SpecificHeatCapacity" -> 440.5
|>;
operator1 = HeatTransferPDEComponent[vars1, parameters1];
leftCondition1 = HeatTemperatureCondition[
  x == 0., vars1, parameters1,
  <|"SurfaceTemperature" -> 100. Sin[Pi t/40.]|>
];
rightCondition1 = HeatTemperatureCondition[x == 0.1, vars1, parameters1];
initialCondition1 = temperature1[0., x] == 0.;
domain1 = Line[{{0.}, {0.1}}];
endTime1 = 32.;
solution1 = NDSolveValue[
  {operator1 == 0., leftCondition1, rightCondition1, initialCondition1},
  temperature1,
  {t, 0., endTime1},
  Element[{x}, domain1]
];

vars2 = {temperature2[t, x], t, {x}};
parameters2 = <|
  "ThermalConductivity" -> 1.,
  "MassDensity" -> 1.,
  "SpecificHeatCapacity" -> 1.
|>;
operator2 = HeatTransferPDEComponent[vars2, parameters2];
fluxCondition2 = HeatFluxValue[
  x == 0., vars2, parameters2, <|"HeatFlux" -> -1.|>
];
temperatureCondition2 = HeatTemperatureCondition[
  x == 1., vars2, parameters2, <|"SurfaceTemperature" -> 25.|>
];
initialCondition2 = temperature2[0., x] == 25.;
domain2 = Line[{{0.}, {1.}}];
endTime2 = 0.2;
reference2[time_, position_] :=
  (24. + position)
  + 8./Pi^2 Cos[Pi position/2.] Exp[-Pi^2 time/4.]
  + 8./(9. Pi^2) Cos[3. Pi position/2.] Exp[-9. Pi^2 time/4.];
solution2 = NDSolveValue[
  {
    operator2 == fluxCondition2,
    temperatureCondition2,
    initialCondition2
  },
  temperature2,
  {t, 0., endTime2},
  Element[{x}, domain2]
];

payload = <|
  "schema_version" -> 2,
  "cases" -> {
    <|
      "case_id" -> "HeatTransfer-FEM-Transient-1D-Single-HeatTransfer-0001",
      "end_time_s" -> endTime1,
      "probe_x_m" -> 0.02,
      "published_probe_temperature" -> 36.6,
      "computed_probe_temperature" -> N[solution1[endTime1, 0.02], 16]
    |>,
    <|
      "case_id" -> "HeatTransfer-FEM-Transient-1D-Single-HeatTransfer-0002",
      "end_time_s" -> endTime2,
      "sample_x_m" -> {0., 1.},
      "reference_temperature" ->
        (N[reference2[endTime2, #], 16] & /@ {0., 1.}),
      "computed_temperature" ->
        (N[solution2[endTime2, #], 16] & /@ {0., 1.})
    |>
  }
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
