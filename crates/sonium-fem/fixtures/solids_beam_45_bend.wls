(* Independent Wolfram reference for the 45-degree elastica bend.
   Derives, at 50-digit working precision, the tip position of a cantilever
   bent by an end moment to constant curvature kappa = theta/L with a
   45-degree tip rotation: tipX = sin(theta)/kappa,
   tipY = (1 - cos(theta))/kappa. The Rust side consumes
   solids_beam_45_bend_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_beam_45_bend.wls > fixtures/solids_beam_45_bend_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

len = 10;
theta = Pi/4;
kappa = theta/len;
tipX = Sin[theta]/kappa;
tipY = (1 - Cos[theta])/kappa;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "beam_45_bend",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "length_m" -> 10,
    "tip_rotation_rad" -> "Pi/4"
  |>,
  "values" -> <|
    "tip_x" -> <|
      "values" -> {dec[tipX]},
      "exact" -> "sin(theta)/kappa"
    |>,
    "tip_y" -> <|
      "values" -> {dec[tipY]},
      "exact" -> "(1-cos(theta))/kappa"
    |>
  |>,
  "tolerances" -> <|
    "tip_x_abs" -> 1.0*^-9,
    "tip_x_rel" -> 5.0*^-3,
    "tip_y_abs" -> 1.0*^-9,
    "tip_y_rel" -> 5.0*^-3
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
