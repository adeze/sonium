(* Independent Wolfram reference for the Ogden rubber QA.
   Derives, at 50-digit working precision, the incompressible uniaxial
   Cauchy stress of the Treloar three-term fit at two stretches. All
   inputs are exact rationals; powers stay exact until N[..., 50].
   The Rust side consumes solids_ogden_uniaxial_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_ogden_uniaxial.wls > fixtures/solids_ogden_uniaxial_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

mu = {618/1000*10^6, 118/100000*10^6, -981/100000*10^6};
alpha = {13/10, 5, -2};
cauchy[l_] := Sum[mu[[i]]*(l^alpha[[i]] - l^(-alpha[[i]]/2)), {i, 3}];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "ogden_uniaxial",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "mu_pa" -> {618000.0, 1180.0, -9810.0},
    "alpha" -> {1.3, 5.0, -2.0},
    "stretch_tension" -> 1.5,
    "stretch_compression" -> 0.8
  |>,
  "values" -> <|
    "cauchy_tension" -> <|
      "values" -> {dec[cauchy[3/2]]},
      "exact" -> "Sum mu_i (l^a_i - l^(-a_i/2)) at l=3/2"
    |>,
    "cauchy_compression" -> <|
      "values" -> {dec[cauchy[4/5]]},
      "exact" -> "Sum mu_i (l^a_i - l^(-a_i/2)) at l=4/5"
    |>
  |>,
  "tolerances" -> <|
    "cauchy_tension_abs" -> 1.0*^-6,
    "cauchy_tension_rel" -> 1.0*^-12,
    "cauchy_compression_abs" -> 1.0*^-6,
    "cauchy_compression_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
