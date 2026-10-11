(* Independent Wolfram reference for the LAW22 damage factor.
   Derives, at 50-digit working precision, the card-aluminum damage
   factor at five plastic strains. The quasi-static strength is exactly
   100 MPa (perfectly plastic core capped at the same level). Et is the
   total-strain softening tangent (Reference Guide Figure 78), so the
   ep-space damage slope converts it with the undamaged elastic
   follower: |Et| / (S (1 + |Et| / E)) = 175/9, and damage is the exact
   rational 175/9 (ep - 1/10) clamped to [0, 1].
   The Rust side consumes law22_dl_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law22_dl.wls > fixtures/law22_dl_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

strength = 100*10^6;
epsD = 1/10;
et = -2000*10^6;
youngs = 70*10^9;
damage[ep_] := Min[Max[-et*(ep - epsD)/(strength*(1 + (-et)/youngs)), 0], 1];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law22_dl",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "strength_pa" -> 100*10^6,
    "damage_strain" -> 0.1,
    "softening_slope_pa" -> -2000*10^6,
    "youngs_pa" -> 70*10^9
  |>,
  "values" -> <|
    "damage_0_05" -> <|
      "values" -> {dec[damage[5/100]]},
      "exact" -> "clamp(175/9 (0.05-0.1)) = 0"
    |>,
    "damage_0_11" -> <|
      "values" -> {dec[damage[11/100]]},
      "exact" -> "175/9 (0.11-0.1) = 7/36"
    |>,
    "damage_0_12" -> <|
      "values" -> {dec[damage[12/100]]},
      "exact" -> "175/9 (0.12-0.1) = 7/18"
    |>,
    "damage_0_15" -> <|
      "values" -> {dec[damage[15/100]]},
      "exact" -> "175/9 (0.15-0.1) = 35/36"
    |>,
    "damage_0_20" -> <|
      "values" -> {dec[damage[20/100]]},
      "exact" -> "clamp(175/9 (0.20-0.1)) = 1"
    |>
  |>,
  "tolerances" -> <|
    "damage_0_05_abs" -> 1.0*^-15,
    "damage_0_05_rel" -> 1.0*^-12,
    "damage_0_11_abs" -> 1.0*^-15,
    "damage_0_11_rel" -> 1.0*^-12,
    "damage_0_12_abs" -> 1.0*^-15,
    "damage_0_12_rel" -> 1.0*^-12,
    "damage_0_15_abs" -> 1.0*^-15,
    "damage_0_15_rel" -> 1.0*^-12,
    "damage_0_20_abs" -> 1.0*^-15,
    "damage_0_20_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
