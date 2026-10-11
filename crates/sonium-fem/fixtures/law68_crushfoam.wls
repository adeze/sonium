(* Independent Wolfram reference for LAW68 Cosserat crushable foam.
   Derives, at 50-digit working precision, the couple modulus, crush
   strengths, and the couple-stress radial return (elastic advance and
   capped return with plastic flow). All pins below are exact rationals.
   The Rust side consumes law68_crushfoam_golden.json; this script is
   the derivation.
   Regen: wolframscript -script fixtures/law68_crushfoam.wls > fixtures/law68_crushfoam_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

young = 2*10^9; nu = 1/10; alpha = 3/2; y0 = 5*10^6; hard = 50*10^6;
emc = 11*10^8; numc = 1/10;
gc = emc/(2*(1 + numc));
crush[c_] := y0 + hard*Max[c, 0];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law68_crushfoam",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "young_pa" -> 2*10^9,
    "poisson" -> 0.1,
    "alpha" -> 1.5,
    "initial_yield_pa" -> 5*10^6,
    "hardening_pa" -> 50*10^6,
    "couple_young_pa" -> 11*10^8,
    "couple_poisson" -> 0.1
  |>,
  "values" -> <|
    "couple_modulus" -> <|
      "values" -> {dec[gc]},
      "exact" -> "11*10^8/(2*1.1) = 5*10^8"
    |>,
    "crush_virgin" -> <|
      "values" -> {dec[crush[0]]},
      "exact" -> "5*10^6"
    |>,
    "crush_tenth" -> <|
      "values" -> {dec[crush[1/10]]},
      "exact" -> "5*10^6 + 5*10^6 = 10^7"
    |>,
    "couple_elastic" -> <|
      "values" -> {dec[2*gc/10^4]},
      "exact" -> "2 Gc 10^-4 = 10^5"
    |>,
    "couple_capped" -> <|
      "values" -> {dec[crush[0]]},
      "exact" -> "trial 10^8 caps at 5*10^6"
    |>,
    "couple_flow" -> <|
      "values" -> {dec[(2*gc/10 - crush[0])/(2*gc)]},
      "exact" -> "(10^8 - 5*10^6)/10^9 = 19/200"
    |>
  |>,
  "tolerances" -> <|
    "couple_modulus_abs" -> 1.0*^-6,
    "couple_modulus_rel" -> 1.0*^-12,
    "crush_virgin_abs" -> 1.0*^-6,
    "crush_virgin_rel" -> 1.0*^-12,
    "crush_tenth_abs" -> 1.0*^-6,
    "crush_tenth_rel" -> 1.0*^-12,
    "couple_elastic_abs" -> 1.0*^-9,
    "couple_elastic_rel" -> 1.0*^-12,
    "couple_capped_abs" -> 1.0*^-6,
    "couple_capped_rel" -> 1.0*^-12,
    "couple_flow_abs" -> 1.0*^-15,
    "couple_flow_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
