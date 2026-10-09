(* Sinusoidal-plate case from Wolfram's heat-transfer verification family:
   https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransferVerificationTests.html?view=all
*)
Off[General::munfl];

conductivity = 3.;
samplePoints = {{0.25, 0.25}, {0.50, 0.25}, {0.50, 0.50}, {0.75, 0.50}, {0.75, 0.75}};
temperature[x_, y_] := Sin[Pi x] Sin[Pi y];

(* This exact manufactured field satisfies
   -Div[k Grad[T]] = 2 k Pi^2 Sin[Pi x] Sin[Pi y]
   with T=0 on every side of the unit square. *)
payload = <|
  "schema_version" -> 1,
  "conductivity_w_m_k" -> conductivity,
  "sample_points" -> samplePoints,
  "temperature" -> (N[temperature @@ #, 16] & /@ samplePoints)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
