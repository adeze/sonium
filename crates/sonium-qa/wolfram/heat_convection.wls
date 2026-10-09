(* Literal Wolfram/NAFEMS verification case
   FEM-Stationary-2D-Single-HeatTransfer-0001-A from:
   https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransferVerificationTests.html?view=all
*)

Needs["NDSolve`FEM`"];

vars = {T[x, y], {x, y}};
pars = <|"ThermalConductivity" -> 52|>;
op = HeatTransferPDEComponent[vars, pars];
publishedTemperature = 18.25;

temperatureCondition = HeatTemperatureCondition[
  y == 0,
  vars,
  pars,
  <|"SurfaceTemperature" -> 100|>
];
convection = HeatTransferValue[
  x == 0.6 || y == 1,
  vars,
  pars,
  <|"HeatTransferCoefficient" -> 750, "AmbientTemperature" -> 0|>
];
domain = Rectangle[{0, 0}, {0.6, 1}];
pde = {op == convection, temperatureCondition};
temperature = NDSolveValue[pde, T, Element[{x, y}, domain]];

samplePoint = {0.6, 0.2};
sampleTemperature = N[temperature @@ samplePoint, 16];

payload = <|
  "schema_version" -> 1,
  "width_m" -> 0.6,
  "height_m" -> 1.0,
  "conductivity_w_m_k" -> 52,
  "fixed_temperature_k" -> 100,
  "ambient_temperature_k" -> 0,
  "convection_w_m2_k" -> 750,
  "sample_point_m" -> samplePoint,
  "temperature_k" -> sampleTemperature,
  "published_temperature_k" -> publishedTemperature,
  "published_relative_error" ->
    N[Abs[(sampleTemperature - publishedTemperature)/publishedTemperature], 16]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
