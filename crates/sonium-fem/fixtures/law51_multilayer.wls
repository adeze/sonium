(* Independent Wolfram reference for the LAW51 JWL mixture.
   Derives, at 50-digit working precision, JWL pressure and its
   derivatives at (mu = 3/10, e = 2*10^5), the energy derivative
   (exact rational), mixture pressure at burn 1/4, and programmed
   burn-fraction pins (exact rationals).
   The Rust side consumes law51_multilayer_golden.json; this script is
   the derivation.
   Regen: wolframscript -script fixtures/law51_multilayer.wls > fixtures/law51_multilayer_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

a = 2*10^9; b = 5*10^8; r1 = 9/2; r2 = 6/5; om = 1/4; rho0 = 1500;
jwl[mu_, e_] := With[{eta = 1 + mu},
  a*(1 - om*eta/r1)*Exp[-r1/eta] + b*(1 - om*eta/r2)*Exp[-r2/eta] + om*rho0*eta*e];
djwlDmu[mu_, e_] := With[{eta = 1 + mu},
  a*Exp[-r1/eta]*(r1/eta^2 - om/eta - om/r1)
  + b*Exp[-r2/eta]*(r2/eta^2 - om/eta - om/r2) + om*rho0*e];
djwlDe[mu_] := om*rho0*(1 + mu);

ar = 1*10^9;
jwr[mu_, e_] := With[{eta = 1 + mu},
  ar*(1 - om*eta/r1)*Exp[-r1/eta] + b*(1 - om*eta/r2)*Exp[-r2/eta] + om*rho0*eta*e];

mu = 3/10; ee = 2*10^5; burn = 1/4;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law51_multilayer",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "a_pa" -> 2*10^9,
    "b_pa" -> 5*10^8,
    "r1" -> 4.5,
    "r2" -> 1.2,
    "omega" -> 0.25,
    "rho0" -> 1500,
    "e0" -> 1*10^5,
    "reacted_a_pa" -> 1*10^9,
    "mu" -> 0.3,
    "energy" -> 2*10^5,
    "burn" -> 0.25
  |>,
  "values" -> <|
    "pressure" -> <|
      "values" -> {dec[jwl[mu, ee]]},
      "exact" -> "JWL closed form at (3/10, 2*10^5)"
    |>,
    "dp_dmu" -> <|
      "values" -> {dec[djwlDmu[mu, ee]]},
      "exact" -> "d/dmu of the JWL closed form"
    |>,
    "dp_de" -> <|
      "values" -> {dec[djwlDe[mu]]},
      "exact" -> "omega rho0 eta = 975/2"
    |>,
    "mixture" -> <|
      "values" -> {dec[(1 - burn)*jwl[mu, ee] + burn*jwr[mu, ee]]},
      "exact" -> "(3 Pu + Pr)/4 at burn 1/4"
    |>,
    "burn_ramp" -> <|
      "values" -> {dec[(3 - 2)/2]},
      "exact" -> "(t - tdet)/tburn = 1/2 at t = 3"
    |>,
    "burn_before" -> <|
      "values" -> {dec[0]},
      "exact" -> "clamped 0 before detonation"
    |>,
    "burn_instant_on" -> <|
      "values" -> {dec[1]},
      "exact" -> "instant burn at tdet"
    |>,
    "burn_instant_off" -> <|
      "values" -> {dec[0]},
      "exact" -> "instant burn stays 0 before tdet"
    |>
  |>,
  "tolerances" -> <|
    "pressure_abs" -> 1.0*^-6,
    "pressure_rel" -> 1.0*^-12,
    "dp_dmu_abs" -> 1.0*^-6,
    "dp_dmu_rel" -> 1.0*^-12,
    "dp_de_abs" -> 1.0*^-12,
    "dp_de_rel" -> 1.0*^-12,
    "mixture_abs" -> 1.0*^-6,
    "mixture_rel" -> 1.0*^-12,
    "burn_ramp_abs" -> 1.0*^-15,
    "burn_ramp_rel" -> 1.0*^-15,
    "burn_before_abs" -> 1.0*^-15,
    "burn_before_rel" -> 1.0*^-15,
    "burn_instant_on_abs" -> 1.0*^-15,
    "burn_instant_on_rel" -> 1.0*^-15,
    "burn_instant_off_abs" -> 1.0*^-15,
    "burn_instant_off_rel" -> 1.0*^-15
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
