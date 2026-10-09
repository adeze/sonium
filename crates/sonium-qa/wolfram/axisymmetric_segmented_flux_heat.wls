(* Literal Wolfram/NAFEMS verification case
   HeatTransfer-FEM-Stationary-2DAxisym-Single-HeatTransfer-0002-A from:
   https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransferVerificationTests.html?view=all
*)

Needs["NDSolve`FEM`"];

vars = {T[r, z], {r, z}};
pars = <|"ThermalConductivity" -> 52, "RegionSymmetry" -> "Axisymmetric"|>;
operator = HeatTransferPDEComponent[vars, pars];
height = 0.14;
fluxZMax = 0.1;
fluxZMin = 0.04;
innerRadius = 0.02;
outerRadius = 0.1;
publishedTemperature = 332.97;

flux = HeatFluxValue[
  r == innerRadius && fluxZMin <= z <= fluxZMax,
  vars,
  pars,
  <|"HeatFlux" -> 5 10^5|>
];
temperatureCondition = HeatTemperatureCondition[
  z == 0 || r == outerRadius || z == height,
  vars,
  pars,
  <|"SurfaceTemperature" -> 273.15|>
];
domain = Polygon[{
  {innerRadius, 0.},
  {outerRadius, 0},
  {outerRadius, height},
  {innerRadius, height},
  {innerRadius, fluxZMax},
  {innerRadius, fluxZMin}
}];
temperature = NDSolveValue[
  {operator == flux, temperatureCondition},
  T,
  Element[{r, z}, domain]
];

samplePoint = {0.04, 0.04};
sampleTemperature = N[temperature @@ samplePoint, 16];

payload = <|
  "schema_version" -> 1,
  "inner_radius_m" -> innerRadius,
  "outer_radius_m" -> outerRadius,
  "height_m" -> height,
  "flux_z_interval_m" -> {fluxZMin, fluxZMax},
  "conductivity_w_m_k" -> 52,
  "heat_flux_w_m2" -> 5 10^5,
  "fixed_temperature_k" -> 273.15,
  "sample_point_m" -> samplePoint,
  "temperature_k" -> sampleTemperature,
  "published_temperature_k" -> publishedTemperature,
  "published_relative_error" ->
    N[Abs[(sampleTemperature - publishedTemperature)/publishedTemperature], 16]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
