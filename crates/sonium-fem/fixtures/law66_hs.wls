(* Independent Wolfram reference for the LAW66 tension blend.
   Derives, at 50-digit working precision, the card-aluminum tension
   weight at five pressures (three interior, two outside the blend
   range) and the blended static yield at ep = 0.01. The card curves
   are symmetric, so the static yield equals the tabulated value at
   every pressure. All inputs are exact rationals.
   The Rust side consumes law66_hs_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law66_hs.wls > fixtures/law66_hs_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

pc = 500*10^6;
pt = 600*10^6;
blend[p_] := Which[p < -pt, 1, p > pc, 0, True, (pc - p)/(pc + pt)];
tabulated = 90*10^6 + ((1/100)/(8/100))*(170*10^6 - 90*10^6);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law66_hs",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "pc_pa" -> 500*10^6,
    "pt_pa" -> 600*10^6,
    "ep" -> 0.01,
    "curve_knots" -> {0.0, 0.08},
    "curve_mpa" -> {90, 170}
  |>,
  "values" -> <|
    "blend_neg" -> <|
      "values" -> {dec[blend[-100*10^6]]},
      "exact" -> "(500+100)/1100 = 6/11"
    |>,
    "blend_zero" -> <|
      "values" -> {dec[blend[0]]},
      "exact" -> "500/1100 = 5/11"
    |>,
    "blend_pos" -> <|
      "values" -> {dec[blend[100*10^6]]},
      "exact" -> "(500-100)/1100 = 4/11"
    |>,
    "blend_out_neg" -> <|
      "values" -> {dec[blend[-700*10^6]]},
      "exact" -> "1 outside the tension limit"
    |>,
    "blend_out_pos" -> <|
      "values" -> {dec[blend[600*10^6]]},
      "exact" -> "0 outside the compression limit"
    |>,
    "static_0_01" -> <|
      "values" -> {dec[tabulated]},
      "exact" -> "90 + (0.01/0.08) 80 MPa = 100 MPa"
    |>
  |>,
  "tolerances" -> <|
    "blend_neg_abs" -> 1.0*^-15,
    "blend_neg_rel" -> 1.0*^-12,
    "blend_zero_abs" -> 1.0*^-15,
    "blend_zero_rel" -> 1.0*^-12,
    "blend_pos_abs" -> 1.0*^-15,
    "blend_pos_rel" -> 1.0*^-12,
    "blend_out_neg_abs" -> 1.0*^-15,
    "blend_out_neg_rel" -> 1.0*^-12,
    "blend_out_pos_abs" -> 1.0*^-15,
    "blend_out_pos_rel" -> 1.0*^-12,
    "static_0_01_abs" -> 1.0*^-6,
    "static_0_01_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
