(* Gaussian verification of Wolfram's diffusion-only mass-transport equation. *)
diffusivity = 1.; initialSigma = 0.5; finalTime = 0.01;
sampleX = {-1., -0.5, 0., 0.5, 1.};
concentration[x_, t_] := With[
  {variance = initialSigma^2 + 2 diffusivity t},
  Exp[-x^2/(2 variance)]/Sqrt[2 Pi variance]
];
payload = <|
  "schema_version" -> 1, "diffusivity_m2_s" -> diffusivity,
  "initial_sigma_m" -> initialSigma, "final_time_s" -> finalTime,
  "sample_x_m" -> sampleX,
  "concentration" -> (N[concentration[#, finalTime], 16] & /@ sampleX)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
