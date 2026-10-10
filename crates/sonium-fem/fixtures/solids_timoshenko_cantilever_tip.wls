(* Independent Wolfram reference for the Timoshenko cantilever tip.
   Derives, at 50-digit working precision, the bending part FL^3/(3EI),
   the shear part FL/(kappa*G*A), their sum, and the tip rotation
   FL^2/(2EI). The shear modulus is the exact rational E/(2(1+nu)) with
   nu = 3/10; all inputs are exact. The Rust side consumes
   solids_timoshenko_cantilever_tip_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_timoshenko_cantilever_tip.wls > fixtures/solids_timoshenko_cantilever_tip_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

fLoad = 100;
len = 1;
young = 200*10^9;
secondMoment = 10^-8;
poisson = 3/10;
shear = young/(2*(1 + poisson));
area = 10^-4;
kappa = 5/6;

bending = fLoad*len^3/(3*young*secondMoment);
shearPart = fLoad*len/(kappa*shear*area);
tip = bending + shearPart;
rotation = fLoad*len^2/(2*young*secondMoment);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "timoshenko_cantilever_tip",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "load_n" -> 100,
    "length_m" -> 1,
    "youngs_modulus_pa" -> 200*10^9,
    "second_moment_m4" -> 1.0*^-8,
    "shear_modulus_pa" -> N[shear, 17],
    "area_m2" -> 1.0*^-4,
    "shear_factor" -> 5/6
  |>,
  "values" -> <|
    "tip_deflection" -> <|
      "values" -> {dec[tip]},
      "exact" -> "F*L^3/(3*E*I) + F*L/(kappa*G*A)"
    |>,
    "bending_part" -> <|
      "values" -> {dec[bending]},
      "exact" -> "F*L^3/(3*E*I)"
    |>,
    "shear_part" -> <|
      "values" -> {dec[shearPart]},
      "exact" -> "F*L/(kappa*G*A)"
    |>,
    "tip_rotation" -> <|
      "values" -> {dec[rotation]},
      "exact" -> "F*L^2/(2*E*I)"
    |>
  |>,
  "tolerances" -> <|
    "tip_deflection_abs" -> 1.0*^-15,
    "tip_deflection_rel" -> 1.0*^-12,
    "bending_part_abs" -> 1.0*^-15,
    "bending_part_rel" -> 1.0*^-12,
    "shear_part_abs" -> 1.0*^-17,
    "shear_part_rel" -> 1.0*^-12,
    "tip_rotation_abs" -> 1.0*^-15,
    "tip_rotation_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
