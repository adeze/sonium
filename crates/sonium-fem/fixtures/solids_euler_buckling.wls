(* Independent Wolfram reference for the Euler buckling load.
   Derives, at 50-digit working precision, P_cr = pi^2 E I/(K L)^2 for
   pinned (K=1), clamped (K=1/2), cantilever (K=2), and clamped-pinned
   (K=7/10) end conditions. Pi and K stay exact until N[..., 50]. The
   Rust side consumes solids_euler_buckling_golden.json; this script is
   the derivation.
   Regen: wolframscript -script fixtures/solids_euler_buckling.wls > fixtures/solids_euler_buckling_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

young = 200*10^9;
secondMoment = 10^-8;
len = 1;

pcr[k_] := Pi^2*young*secondMoment/(k*len)^2;
pinned = pcr[1];
clamped = pcr[1/2];
cantilever = pcr[2];
clampedPinned = pcr[7/10];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "euler_buckling",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_modulus_pa" -> 200*10^9,
    "second_moment_m4" -> 1.0*^-8,
    "length_m" -> 1
  |>,
  "values" -> <|
    "pcr_pinned" -> <|
      "values" -> {dec[pinned]},
      "exact" -> "Pi^2 E I/L^2, K=1"
    |>,
    "pcr_clamped" -> <|
      "values" -> {dec[clamped]},
      "exact" -> "Pi^2 E I/(L/2)^2, K=1/2"
    |>,
    "pcr_cantilever" -> <|
      "values" -> {dec[cantilever]},
      "exact" -> "Pi^2 E I/(2L)^2, K=2"
    |>,
    "pcr_clamped_pinned" -> <|
      "values" -> {dec[clampedPinned]},
      "exact" -> "Pi^2 E I/(0.7L)^2, K=7/10"
    |>
  |>,
  "tolerances" -> <|
    "pcr_pinned_abs" -> 1.0*^-9,
    "pcr_pinned_rel" -> 1.0*^-12,
    "pcr_clamped_abs" -> 1.0*^-9,
    "pcr_clamped_rel" -> 1.0*^-12,
    "pcr_cantilever_abs" -> 1.0*^-9,
    "pcr_cantilever_rel" -> 1.0*^-12,
    "pcr_clamped_pinned_abs" -> 1.0*^-9,
    "pcr_clamped_pinned_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
