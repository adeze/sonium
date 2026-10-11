(* Independent Wolfram reference for the LAW44 Cowper-Symonds flow.
   Derives, at 50-digit working precision, the card-metal analytic flow
   (a + b ep^n) times 1 + (rate/c)^(1/p) at quasi-static, mid, and fast
   rates, plus two rate-factor pins. All inputs are exact rationals;
   powers stay exact until N[..., 50].
   The Rust side consumes law44_cs_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law44_cs.wls > fixtures/law44_cs_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

a = 50*10^6;
b = 100*10^6;
nn = 1/2;
cc = 100;
pp = 5;
smax = 90*10^6;
factor[r_] := If[r <= 0, 1, 1 + (r/cc)^(1/pp)];
flow[ep_, r_] := Min[(a + b*ep^nn)*factor[r], smax*factor[r]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law44_cs",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "a_pa" -> 50*10^6,
    "b_pa" -> 100*10^6,
    "n" -> 0.5,
    "c" -> 100,
    "p" -> 5,
    "max_stress_pa" -> 90*10^6
  |>,
  "values" -> <|
    "flow_quasi" -> <|
      "values" -> {dec[flow[1/100, 0]]},
      "exact" -> "(50 + 100 sqrt(0.01)) MPa = 60 MPa"
    |>,
    "flow_mid" -> <|
      "values" -> {dec[flow[4/100, 100]]},
      "exact" -> "(50 + 100 sqrt(0.04)) 2 MPa = 140 MPa"
    |>,
    "flow_fast" -> <|
      "values" -> {dec[flow[1/100, 1000]]},
      "exact" -> "60 (1 + 10^(1/5)) MPa"
    |>,
    "factor_100" -> <|
      "values" -> {dec[factor[100]]},
      "exact" -> "1 + 1^(1/5) = 2"
    |>,
    "factor_1000" -> <|
      "values" -> {dec[factor[1000]]},
      "exact" -> "1 + 10^(1/5)"
    |>
  |>,
  "tolerances" -> <|
    "flow_quasi_abs" -> 1.0*^-6,
    "flow_quasi_rel" -> 1.0*^-12,
    "flow_mid_abs" -> 1.0*^-6,
    "flow_mid_rel" -> 1.0*^-12,
    "flow_fast_abs" -> 1.0*^-6,
    "flow_fast_rel" -> 1.0*^-12,
    "factor_100_abs" -> 1.0*^-15,
    "factor_100_rel" -> 1.0*^-12,
    "factor_1000_abs" -> 1.0*^-15,
    "factor_1000_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
