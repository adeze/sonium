(* Independent Wolfram reference for LAW50 rated yield tables.
   Derives, at 50-digit working precision, piecewise-linear yield values
   of a two-curve table (rates 0 and 10) at interior, edge, and clamped
   points. All inputs are exact rationals; every pin below is rational.
   The Rust side consumes law50_soc_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law50_soc.wls > fixtures/law50_soc_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

lo[x_] := 1 + 2*x;
hi[x_] := 2 + 2*x;
rate[y0_, y1_, r_] := y0 + Clip[r/10, {0, 1}]*(y1 - y0);
yield[x_, r_] := rate[lo[x], hi[x], r];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law50_soc",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "young_all_pa" -> 100,
    "lo_nodes" -> {{0, 1}, {1, 3}},
    "hi_nodes" -> {{0, 2}, {1, 4}},
    "rates" -> {0, 10}
  |>,
  "values" -> <|
    "yield_mid_lo" -> <|
      "values" -> {dec[yield[1/2, 0]]},
      "exact" -> "lo(1/2) = 2"
    |>,
    "yield_mid_hi" -> <|
      "values" -> {dec[yield[1/2, 10]]},
      "exact" -> "hi(1/2) = 3"
    |>,
    "yield_mid_mid" -> <|
      "values" -> {dec[yield[1/2, 5]]},
      "exact" -> "(2 + 3)/2 = 5/2"
    |>,
    "yield_qtr_mid" -> <|
      "values" -> {dec[yield[1/4, 5]]},
      "exact" -> "(3/2 + 5/2)/2 = 2"
    |>,
    "yield_clamp_lo" -> <|
      "values" -> {dec[yield[1/2, -3]]},
      "exact" -> "clamped to lo = 2"
    |>,
    "yield_clamp_hi" -> <|
      "values" -> {dec[yield[1/2, 99]]},
      "exact" -> "clamped to hi = 3"
    |>,
    "yield_x0_r7" -> <|
      "values" -> {dec[yield[0, 7]]},
      "exact" -> "1 + 0.7 = 17/10"
    |>,
    "yield_x1_r3" -> <|
      "values" -> {dec[yield[1, 3]]},
      "exact" -> "3 + 0.3 = 33/10"
    |>
  |>,
  "tolerances" -> <|
    "yield_mid_lo_abs" -> 1.0*^-12,
    "yield_mid_lo_rel" -> 1.0*^-12,
    "yield_mid_hi_abs" -> 1.0*^-12,
    "yield_mid_hi_rel" -> 1.0*^-12,
    "yield_mid_mid_abs" -> 1.0*^-12,
    "yield_mid_mid_rel" -> 1.0*^-12,
    "yield_qtr_mid_abs" -> 1.0*^-12,
    "yield_qtr_mid_rel" -> 1.0*^-12,
    "yield_clamp_lo_abs" -> 1.0*^-12,
    "yield_clamp_lo_rel" -> 1.0*^-12,
    "yield_clamp_hi_abs" -> 1.0*^-12,
    "yield_clamp_hi_rel" -> 1.0*^-12,
    "yield_x0_r7_abs" -> 1.0*^-12,
    "yield_x0_r7_rel" -> 1.0*^-12,
    "yield_x1_r3_abs" -> 1.0*^-12,
    "yield_x1_r3_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
