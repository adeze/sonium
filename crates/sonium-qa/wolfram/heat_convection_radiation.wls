(* Literal Wolfram verification case
   FEM-Stationary-2D-Single-HeatTransfer-0002-A from:
   https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransferVerificationTests.html?view=all
*)

Needs["NDSolve`FEM`"];

vars = {T[x, y], {x, y}};
pars = <|"ThermalConductivity" -> 3|>;
op = HeatTransferPDEComponent[vars, pars];
publishedTemperature = {1111.38, 1092.37, 1064.21};

temperatureCondition = HeatTemperatureCondition[
  x == 0 || x == 0.02,
  vars,
  pars,
  <|"SurfaceTemperature" -> 1173|>
];
convection = HeatTransferValue[
  y == 0.01,
  vars,
  pars,
  <|"HeatTransferCoefficient" -> 50, "AmbientTemperature" -> 323|>
];
radiation = HeatRadiationValue[
  y == 0.01,
  vars,
  pars,
  <|"Emissivity" -> 0.7, "AmbientTemperature" -> 323|>
];
domain = Rectangle[{0, 0}, {0.02, 0.01}];
pde = {op == convection + radiation, temperatureCondition};
temperature = NDSolveValue[pde, T, Element[{x, y}, domain]];

samplePoints = {{0.005, 0}, {0.005, 0.005}, {0.01, 0.005}};
sampleTemperature = N[temperature @@@ samplePoints, 16];

payload = <|
  "schema_version" -> 1,
  "width_m" -> 0.02,
  "height_m" -> 0.01,
  "conductivity_w_m_k" -> 3,
  "side_temperature_k" -> 1173,
  "ambient_temperature_k" -> 323,
  "convection_w_m2_k" -> 50,
  "emissivity" -> 0.7,
  "sample_points_m" -> samplePoints,
  "temperature_k" -> sampleTemperature,
  "published_temperature_k" -> publishedTemperature,
  "published_relative_error_norm" ->
    N[Norm[(sampleTemperature - publishedTemperature)/publishedTemperature], 16]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
