(* Independent Wolfram reference for the Euler-Bernoulli cantilever tip.
   Derives, at 50-digit working precision, the tip deflection FL^3/(3EI),
   the tip slope FL^2/(2EI), and the midspan deflection for a prismatic
   cantilever with a transverse tip load. All inputs are exact rationals;
   N[..., 50] is applied once per exported value. The Rust side consumes
   solids_eb_cantilever_tip_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_eb_cantilever_tip.wls > fixtures/solids_eb_cantilever_tip_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

fLoad = 100;
len = 1;
young = 200*10^9;
secondMoment = 10^-8;

tip = fLoad*len^3/(3*young*secondMoment);
slope = fLoad*len^2/(2*young*secondMoment);
mid = fLoad*(len/2)^2*(3*len - len/2)/(6*young*secondMoment);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "eb_cantilever_tip",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "load_n" -> 100,
    "length_m" -> 1,
    "youngs_modulus_pa" -> 200*10^9,
    "second_moment_m4" -> 1.0*^-8
  |>,
  "values" -> <|
    "tip_deflection" -> <|
      "values" -> {dec[tip]},
      "exact" -> "F*L^3/(3*E*I)"
    |>,
    "tip_slope" -> <|
      "values" -> {dec[slope]},
      "exact" -> "F*L^2/(2*E*I)"
    |>,
    "midspan_deflection" -> <|
      "values" -> {dec[mid]},
      "exact" -> "F*(L/2)^2*(3*L-L/2)/(6*E*I)"
    |>
  |>,
  "tolerances" -> <|
    "tip_deflection_abs" -> 1.0*^-15,
    "tip_deflection_rel" -> 1.0*^-12,
    "tip_slope_abs" -> 1.0*^-15,
    "tip_slope_rel" -> 1.0*^-12,
    "midspan_deflection_abs" -> 1.0*^-15,
    "midspan_deflection_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
