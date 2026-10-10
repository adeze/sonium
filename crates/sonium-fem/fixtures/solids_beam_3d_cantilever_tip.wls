(* Independent Wolfram reference for the 3-D cantilever tip.
   Derives, at 50-digit working precision, the tip deflection w = F L^3/(3 E I)
   and tip slope s = F L^2/(2 E I) for a square-section cantilever loaded in
   the x-y plane. All inputs are exact rationals. The Rust side consumes
   solids_beam_3d_cantilever_tip_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_beam_3d_cantilever_tip.wls > fixtures/solids_beam_3d_cantilever_tip_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

load = 1000;
len = 1;
young = 200*10^9;
width = 5/100;
height = 5/100;

inertia = height*width^3/12;
tipDeflection = load*len^3/(3*young*inertia);
tipSlope = load*len^2/(2*young*inertia);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "beam_3d_cantilever_tip",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "load_n" -> 1000,
    "length_m" -> 1,
    "youngs_modulus_pa" -> 200*10^9,
    "width_m" -> 0.05,
    "height_m" -> 0.05
  |>,
  "values" -> <|
    "tip_deflection" -> <|
      "values" -> {dec[tipDeflection]},
      "exact" -> "F*L^3/(3*E*I)"
    |>,
    "tip_slope" -> <|
      "values" -> {dec[tipSlope]},
      "exact" -> "F*L^2/(2*E*I)"
    |>
  |>,
  "tolerances" -> <|
    "tip_deflection_abs" -> 1.0*^-15,
    "tip_deflection_rel" -> 1.0*^-9,
    "tip_slope_abs" -> 1.0*^-15,
    "tip_slope_rel" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
