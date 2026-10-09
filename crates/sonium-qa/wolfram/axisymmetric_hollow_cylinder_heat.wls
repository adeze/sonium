(* Published Wolfram verification case
   HeatTransfer-FEM-Stationary-2DAxisym-Single-HeatTransfer-0001:
   https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransferVerificationTests.html?view=all

   The exact logarithmic temperature field is printed by the official page.
*)
Off[General::munfl];

innerRadius = 1.;
outerRadius = 2.;
height = 0.5;
conductivity = 10.;
innerHeatFlux = 100.;
outerTemperature = 10.;
sampleRadii = {1., 1.25, 1.5, 1.75, 2.};
temperature[r_] := 10. (1. - Log[r/outerRadius]);

payload = <|
  "schema_version" -> 1,
  "case_id" -> "HeatTransfer-FEM-Stationary-2DAxisym-Single-HeatTransfer-0001",
  "radii_m" -> {innerRadius, outerRadius},
  "height_m" -> height,
  "conductivity_w_m_k" -> conductivity,
  "inner_heat_flux_w_m2" -> innerHeatFlux,
  "outer_temperature_k" -> outerTemperature,
  "sample_radii_m" -> sampleRadii,
  "temperature_k" -> (N[temperature[#], 16] & /@ sampleRadii)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
