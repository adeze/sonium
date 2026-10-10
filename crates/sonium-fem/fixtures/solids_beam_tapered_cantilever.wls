(* Independent Wolfram reference for the tapered cantilever.
   Derives, at 50-digit working precision, the tip deflection, tip slope,
   and axial tip displacement of a unit cantilever (E = 200 GPa) with
   constant 0.05 m out-of-plane height and in-plane width tapering
   linearly from 0.05 m at the root to 0.025 m at the tip, under a
   1000 N transverse tip load in the taper plane (plus a
   separate 1000 N axial load for the axial value). Unit-load integrals
   over the exact tapered I(x) and A(x); all inputs are exact rationals.
   The Rust side consumes solids_beam_tapered_cantilever_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_beam_tapered_cantilever.wls > fixtures/solids_beam_tapered_cantilever_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

len = 1;
loadP = 1000;
axialP = 1000;
young = 200*10^9;
heightH = 5/100;
rootB = 5/100;
tipB = 5/200;

b[x_] := rootB + (tipB - rootB)*x/len;
iOf[x_] := heightH*b[x]^3/12;
aOf[x_] := heightH*b[x];

tipDeflection = loadP*Integrate[(len - x)^2/(young*iOf[x]), {x, 0, len}];
tipSlope = loadP*Integrate[(len - x)/(young*iOf[x]), {x, 0, len}];
axialTip = Integrate[axialP/(young*aOf[x]), {x, 0, len}];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "beam_tapered_cantilever",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "length_m" -> 1,
    "tip_load_n" -> 1000,
    "axial_load_n" -> 1000,
    "youngs_modulus_pa" -> 200*10^9,
    "height_m" -> 0.05,
    "root_width_m" -> 0.05,
    "tip_width_m" -> 0.025
  |>,
  "values" -> <|
    "tip_deflection" -> <|
      "values" -> {dec[tipDeflection]},
      "exact" -> "P*Integral[(L-x)^2/(E I(x)), {x, 0, L}]"
    |>,
    "tip_slope" -> <|
      "values" -> {dec[tipSlope]},
      "exact" -> "P*Integral[(L-x)/(E I(x)), {x, 0, L}]"
    |>,
    "axial_tip" -> <|
      "values" -> {dec[axialTip]},
      "exact" -> "Integral[Pax/(E A(x)), {x, 0, L}]"
    |>
  |>,
  "tolerances" -> <|
    "tip_deflection_abs" -> 1.0*^-15,
    "tip_deflection_rel" -> 1.0*^-5,
    "tip_slope_abs" -> 1.0*^-15,
    "tip_slope_rel" -> 1.0*^-5,
    "axial_tip_abs" -> 1.0*^-15,
    "axial_tip_rel" -> 1.0*^-5
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
