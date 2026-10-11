(* Independent Wolfram reference for the LAW36 tabulated yield.
   Derives, at 50-digit working precision, the card-aluminum nominal
   yield at a knot, inside a segment, at the last knot, and past the
   grid (end-slope extrapolation), plus the in-segment slope. The last
   two knots share 170 MPa, so the end slope is exactly zero. All
   inputs are exact rationals.
   The Rust side consumes law36_barlat_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law36_barlat.wls > fixtures/law36_barlat_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

x0 = 0; y0 = 90*10^6;
xA = 36/1000; yA = 156*10^6;
xB = 45/1000; yB = 162*10^6;
xMax = 75/1000; yMax = 170*10^6;
endSlope = (170*10^6 - 170*10^6)/((75/1000) - (72/1000));

yield04 = yA + ((4/100) - xA)/(xB - xA)*(yB - yA);
slope04 = (yB - yA)/(xB - xA);
yield10 = yMax + ((1/10) - xMax)*endSlope;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law36_barlat",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "knot_0" -> 0.0,
    "yield_0_pa" -> 90*10^6,
    "seg_lo" -> 0.036,
    "seg_hi" -> 0.045,
    "yield_lo_pa" -> 156*10^6,
    "yield_hi_pa" -> 162*10^6,
    "xmax" -> 0.075,
    "yield_max_pa" -> 170*10^6
  |>,
  "values" -> <|
    "yield_0" -> <|
      "values" -> {dec[y0]},
      "exact" -> "first knot 90 MPa"
    |>,
    "yield_0_04" -> <|
      "values" -> {dec[yield04]},
      "exact" -> "156 + (0.004/0.009) 6 MPa"
    |>,
    "slope_0_04" -> <|
      "values" -> {dec[slope04]},
      "exact" -> "6 MPa / 0.009"
    |>,
    "yield_0_075" -> <|
      "values" -> {dec[yMax]},
      "exact" -> "last knot 170 MPa"
    |>,
    "yield_0_10" -> <|
      "values" -> {dec[yield10]},
      "exact" -> "170 MPa + 0.025 * 0 end slope"
    |>
  |>,
  "tolerances" -> <|
    "yield_0_abs" -> 1.0*^-6,
    "yield_0_rel" -> 1.0*^-12,
    "yield_0_04_abs" -> 1.0*^-6,
    "yield_0_04_rel" -> 1.0*^-12,
    "slope_0_04_abs" -> 1.0*^-6,
    "slope_0_04_rel" -> 1.0*^-12,
    "yield_0_075_abs" -> 1.0*^-6,
    "yield_0_075_rel" -> 1.0*^-12,
    "yield_0_10_abs" -> 1.0*^-6,
    "yield_0_10_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
