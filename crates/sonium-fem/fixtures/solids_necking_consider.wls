(* Independent Wolfram reference for the necking-bar QA.
   Derives, at 50-digit working precision, the Consider uniform necking
   strains of three Hollomon exponents and the Bridgman correction at
   three neck geometries. The full necking bar has no closed form; these
   values plus the incremental reference and the Radioss deck are the legs.
   Regen: wolframscript -script fixtures/solids_necking_consider.wls > fixtures/solids_necking_consider_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

bridgman[x_] := 1/((1 + 2/x)*Log[1 + x/2]);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "necking_consider",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "hollomon_exponents" -> {0.1, 0.2, 0.3},
    "a_over_r" -> {0.5, 1.0, 2.0}
  |>,
  "values" -> <|
    "uniform_strain" -> <|
      "values" -> {dec[1/10], dec[2/10], dec[3/10]},
      "exact" -> "eps_uniform = n (Considere)"
    |>,
    "bridgman" -> <|
      "values" -> {dec[bridgman[1/2]], dec[bridgman[1]], dec[bridgman[2]]},
      "exact" -> "1/((1+2/x) ln(1+x/2))"
    |>
  |>,
  "tolerances" -> <|
    "uniform_strain_abs" -> 1.0*^-15,
    "uniform_strain_rel" -> 1.0*^-12,
    "bridgman_abs" -> 1.0*^-15,
    "bridgman_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
