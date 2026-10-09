(* Sonium reverse example based on the governing formula published in:
   https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/ModelCollection/HelmholtzResonator.html?view=all

   This cylindrical cabinet-plus-port geometry is not the literal Wolfram
   bottle geometry.  It independently evaluates the compact-resonator
   reference used to check Sonium's coupled interior-FEM/exterior-BEM sweep.
*)
c = 343.;
rho = 1.204;
boxRadius = 0.15;
boxDepth = 0.25;
portRadius = 0.03;
portLength = 0.10;
cavityVolume = Pi boxRadius^2 boxDepth;
portArea = Pi portRadius^2;
effectivePortLength = portLength + (0.85 + 0.61) portRadius;
resonanceFrequency =
  c/(2 Pi) Sqrt[portArea/(cavityVolume effectivePortLength)];

payload = <|
  "schema_version" -> 1,
  "provenance" -> "Sonium reverse example using Wolfram Helmholtz formula",
  "sound_speed_m_s" -> c,
  "density_kg_m3" -> rho,
  "box_radius_m" -> boxRadius,
  "box_depth_m" -> boxDepth,
  "port_radius_m" -> portRadius,
  "port_length_m" -> portLength,
  "cavity_volume_m3" -> cavityVolume,
  "port_area_m2" -> portArea,
  "effective_port_length_m" -> effectivePortLength,
  "resonance_frequency_hz" -> resonanceFrequency
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
