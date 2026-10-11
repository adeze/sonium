(* Independent Wolfram reference for the central-difference amplification factor.
   Derives, at 50-digit working precision, the magnitude |A|, the phase arg(A),
   and the period ratio T_num/T = O/arg(A) of the principal root of the undamped
   central-difference characteristic equation u_{n+1} - 2u_n + u_{n-1} = -O^2 u_n
   with the ansatz u_n = A^n, i.e. A = 1 - O^2/2 +/- I O Sqrt[1-O^2/4], at the five
   values O = w dt in {0.1, 1.0, 1.99, 2.0, 2.01}. All inputs are exact rationals;
   N[..., 50] is applied once per exported value. Above the stability limit
   (O = 2.01) both roots are real and negative; the dominant (larger-magnitude)
   root sets |A| > 1 with phase Pi, and the period ratio is the formal quotient
   O/Pi, meaningful only when stable. The Rust side consumes
   solids_cd_amplification_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_cd_amplification.wls > fixtures/solids_cd_amplification_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, prec]]]];

omegas = {1/10, 1, 199/100, 2, 201/100};

root[om_] := If[om <= 2, 1 - om^2/2 + I*om*Sqrt[1 - om^2/4], 1 - om^2/2 - om*Sqrt[om^2/4 - 1]];

absVals = Abs[root[#]] & /@ omegas;
argVals = Arg[root[#]] & /@ omegas;
periodVals = (#/Arg[root[#]]) & /@ omegas;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "cd_amplification",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "omega_dt" -> N[omegas]
  |>,
  "values" -> <|
    "abs_A" -> <|
      "values" -> dec /@ absVals,
      "exact" -> "Abs[1-O^2/2+I*O*Sqrt[1-O^2/4]] (|A|=1 for O<=2; dominant real root above)"
    |>,
    "arg_A" -> <|
      "values" -> dec /@ argVals,
      "exact" -> "Arg[1-O^2/2+I*O*Sqrt[1-O^2/4]] (Pi above the limit)"
    |>,
    "period_ratio" -> <|
      "values" -> dec /@ periodVals,
      "exact" -> "T_num/T = O/Arg[A] (formal quotient above the limit)"
    |>
  |>,
  "tolerances" -> <|
    "abs_A_abs" -> 1.0*^-15,
    "abs_A_rel" -> 1.0*^-12,
    "arg_A_abs" -> 1.0*^-15,
    "arg_A_rel" -> 1.0*^-12,
    "period_ratio_abs" -> 1.0*^-15,
    "period_ratio_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
