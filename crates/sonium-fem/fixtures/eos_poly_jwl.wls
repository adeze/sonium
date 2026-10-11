(* Independent Wolfram reference for the polynomial and JWL EOS.
   Derives, at 50-digit working precision, the steel polynomial pressure
   in compression, expansion (quadratic term dropped), and at high
   energy, plus the TNT JWL pressure at two relative volumes and the
   volume branch of the burn fraction. All inputs are exact rationals;
   exps stay exact until N[..., 50].
   The Rust side consumes eos_poly_jwl_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/eos_poly_jwl.wls > fixtures/eos_poly_jwl_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

c0 = 0; c1 = 169/100*10^11; c2 = 31/10*10^11; c3 = 0;
c4 = 183/100; c5 = 183/100;
poly[mu_, e_] := c0 + c1*mu + If[mu < 0, 0, c2*mu^2] + c3*mu^3 + (c4 + c5*mu)*e;

a = 37121/10000*10^11;
b = 323/10000*10^11;
r1 = 415/100; r2 = 95/100; om = 3/10;
rho0 = 1630; detV = 6930; pcj = 21/100*10^11; e0 = 7/100*10^11;
jwl[v_, e_, q_] := a*(1 - om/(r1*v))*Exp[-r1*v] + b*(1 - om/(r2*v))*Exp[-r2*v] + om*(e + q)/v;
bf1[v_] := rho0*detV^2/pcj*(1 - v);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "eos_poly_jwl",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "steel_c1_pa" -> 1.69*^11,
    "steel_c2_pa" -> 3.1*^11,
    "steel_c4" -> 1.83,
    "steel_c5" -> 1.83,
    "tnt_a_pa" -> 3.7121*^11,
    "tnt_b_pa" -> 3.23*^9,
    "tnt_r1" -> 4.15,
    "tnt_r2" -> 0.95,
    "tnt_omega" -> 0.3,
    "tnt_rho0" -> 1630,
    "tnt_d" -> 6930,
    "tnt_pcj_pa" -> 2.1*^10,
    "tnt_e0" -> 7.0*^9
  |>,
  "values" -> <|
    "poly_compression" -> <|
      "values" -> {dec[poly[1/10, 0]]},
      "exact" -> "1.69e11 0.1 + 3.1e11 0.01 = 2.0e10"
    |>,
    "poly_expansion" -> <|
      "values" -> {dec[poly[-5/100, 10^9]]},
      "exact" -> "1.69e11 (-0.05) + (1.83 - 1.83 0.05) 1e9"
    |>,
    "poly_hot" -> <|
      "values" -> {dec[poly[2/10, 5*10^9]]},
      "exact" -> "1.69e11 0.2 + 3.1e11 0.04 + (1.83 + 1.83 0.2) 5e9"
    |>,
    "jwl_unit_volume" -> <|
      "values" -> {dec[jwl[1, e0, 0]]},
      "exact" -> "A (1-w/R1) e^-R1 + B (1-w/R2) e^-R2 + w E0"
    |>,
    "jwl_expanded" -> <|
      "values" -> {dec[jwl[2, e0, 0]]},
      "exact" -> "A (1-w/2R1) e^-2R1 + B (1-w/2R2) e^-2R2 + w E0/2"
    |>,
    "burn_half_volume" -> <|
      "values" -> {dec[bf1[1/2]]},
      "exact" -> "(rho0 D^2/P_CJ) (1 - 1/2)"
    |>
  |>,
  "tolerances" -> <|
    "poly_compression_abs" -> 1.0*^-6,
    "poly_compression_rel" -> 1.0*^-12,
    "poly_expansion_abs" -> 1.0*^-6,
    "poly_expansion_rel" -> 1.0*^-12,
    "poly_hot_abs" -> 1.0*^-6,
    "poly_hot_rel" -> 1.0*^-12,
    "jwl_unit_volume_abs" -> 1.0*^-6,
    "jwl_unit_volume_rel" -> 1.0*^-12,
    "jwl_expanded_abs" -> 1.0*^-6,
    "jwl_expanded_rel" -> 1.0*^-12,
    "burn_half_volume_abs" -> 1.0*^-15,
    "burn_half_volume_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
