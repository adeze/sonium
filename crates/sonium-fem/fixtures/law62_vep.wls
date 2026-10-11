(* Independent Wolfram reference for the LAW62 Ogden stresses.
   Derives, at 50-digit working precision, the card-rubber Ogden
   principal Cauchy stresses at a volume-preserving stretch, plus the
   ground shear modulus and long-term stiffness share. Stretches are
   exact (6/5, Sqrt[5/6], Sqrt[5/6]) with unit Jacobian.
   The Rust side consumes law62_vep_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law62_vep.wls > fixtures/law62_vep_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

mu = {2*10^6, 1*10^6};
al = {2, -2};
nu = 495/1000;
beta = nu/(1 - 2*nu);
lam = {6/5, Sqrt[5/6], Sqrt[5/6]};
jac = lam[[1]]*lam[[2]]*lam[[3]];
ogden[k_] := Sum[2*mu[[i]]/(al[[i]]*jac)*(lam[[k]]^al[[i]] - jac^(-al[[i]]*beta)), {i, 2}];
g0 = mu[[1]] + mu[[2]];
gamInf = 1 - 3/10;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law62_vep",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "mu_pa" -> {2*10^6, 1*10^6},
    "alpha" -> {2, -2},
    "nu" -> 0.495,
    "lambda" -> {1.2, "1/sqrt(1.2)", "1/sqrt(1.2)"},
    "prony_gamma" -> 0.3
  |>,
  "values" -> <|
    "ogden_principal" -> <|
      "values" -> {dec[ogden[1]], dec[ogden[2]], dec[ogden[3]]},
      "exact" -> "sum 2 mu/(alpha J) (lam^alpha - J^(-alpha be))"
    |>,
    "ground_shear" -> <|
      "values" -> {dec[g0]},
      "exact" -> "2 + 1 MPa"
    |>,
    "gamma_infinity" -> <|
      "values" -> {dec[gamInf]},
      "exact" -> "1 - 0.3"
    |>
  |>,
  "tolerances" -> <|
    "ogden_principal_abs" -> 1.0*^-6,
    "ogden_principal_rel" -> 1.0*^-12,
    "ground_shear_abs" -> 1.0*^-6,
    "ground_shear_rel" -> 1.0*^-12,
    "gamma_infinity_abs" -> 1.0*^-15,
    "gamma_infinity_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
