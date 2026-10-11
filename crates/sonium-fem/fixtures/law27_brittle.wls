(* Independent Wolfram reference for the LAW27 brittle damage ramp.
   Derives, at 50-digit working precision, the steel-brittle directional
   damage at six principal strains: zero below eps_t, linear ramp capped
   at dmax, dmax at and above eps_m. All inputs are exact rationals.
   The Rust side consumes law27_brittle_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law27_brittle.wls > fixtures/law27_brittle_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

epsT = 8/100;
epsM = 15/100;
dmax = 999/1000;
damage[p_] := Piecewise[{{0, p <= epsT}, {dmax, p >= epsM}}, Min[(p - epsT)/(epsM - epsT), dmax]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law27_brittle",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "eps_t" -> 0.08,
    "eps_m" -> 0.15,
    "dmax" -> 0.999
  |>,
  "values" -> <|
    "damage_0_05" -> <|
      "values" -> {dec[damage[5/100]]},
      "exact" -> "0 below eps_t"
    |>,
    "damage_0_08" -> <|
      "values" -> {dec[damage[8/100]]},
      "exact" -> "0 at eps_t"
    |>,
    "damage_0_115" -> <|
      "values" -> {dec[damage[115/1000]]},
      "exact" -> "(0.115-0.08)/0.07 = 1/2"
    |>,
    "damage_knee" -> <|
      "values" -> {dec[damage[epsT + dmax*(epsM - epsT)]]},
      "exact" -> "ramp reaches dmax at eps_t + dmax (eps_m - eps_t)"
    |>,
    "damage_0_15" -> <|
      "values" -> {dec[damage[15/100]]},
      "exact" -> "dmax at eps_m"
    |>,
    "damage_0_20" -> <|
      "values" -> {dec[damage[20/100]]},
      "exact" -> "dmax above eps_m"
    |>
  |>,
  "tolerances" -> <|
    "damage_0_05_abs" -> 1.0*^-15,
    "damage_0_05_rel" -> 1.0*^-12,
    "damage_0_08_abs" -> 1.0*^-15,
    "damage_0_08_rel" -> 1.0*^-12,
    "damage_0_115_abs" -> 1.0*^-15,
    "damage_0_115_rel" -> 1.0*^-12,
    "damage_knee_abs" -> 1.0*^-15,
    "damage_knee_rel" -> 1.0*^-12,
    "damage_0_15_abs" -> 1.0*^-15,
    "damage_0_15_rel" -> 1.0*^-12,
    "damage_0_20_abs" -> 1.0*^-15,
    "damage_0_20_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
