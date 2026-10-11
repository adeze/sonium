(* Independent Wolfram reference for penalized Hertz normal contact.
   Derives, at 50-digit working precision, the Hertz contact radius,
   approach, and tangent stiffness of a steel sphere on a rigid plane,
   plus the penalty-augmented total approach and series stiffness at
   three penalty values. The penalty compliance F/k_p sits in series
   with the Hertz spring: total approach a^2/R + F/k_p, effective
   stiffness 1/(1/k_H + 1/k_p) with tangent k_H = 2 E* a. All inputs
   are exact rationals; the cube root stays exact until the final
   N[..., 50]. The Rust side consumes
   solids_penalty_hertz_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_penalty_hertz.wls > fixtures/solids_penalty_hertz_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

force = 1000;
radius = 1/100;
young = 200*10^9;
poisson = 3/10;
penalties = {1*10^8, 1*10^10, 1*10^12};

estar = young/(1 - poisson^2);
contactA = (3*force*radius/(4*estar))^(1/3);
approachH = contactA^2/radius;
stiffH = 2*estar*contactA;
approachTotal = Table[approachH + force/kp, {kp, penalties}];
stiffEff = Table[1/(1/stiffH + 1/kp), {kp, penalties}];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "penalty_hertz",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "force_n" -> 1000,
    "radius_m" -> 0.01,
    "youngs_modulus_pa" -> 200*10^9,
    "poisson_ratio" -> 0.3,
    "penalty_stiffness_n_per_m" -> {1.0*^8, 1.0*^10, 1.0*^12}
  |>,
  "values" -> <|
    "effective_modulus" -> <|
      "values" -> {dec[estar]},
      "exact" -> "E/(1-nu^2)"
    |>,
    "contact_radius" -> <|
      "values" -> {dec[contactA]},
      "exact" -> "(3 F R/(4 E*))^(1/3)"
    |>,
    "hertz_approach" -> <|
      "values" -> {dec[approachH]},
      "exact" -> "a^2/R"
    |>,
    "hertz_stiffness" -> <|
      "values" -> {dec[stiffH]},
      "exact" -> "2 E* a"
    |>,
    "approach_total" -> <|
      "values" -> Map[dec, approachTotal],
      "exact" -> "a^2/R + F/k_p per k_p"
    |>,
    "effective_stiffness" -> <|
      "values" -> Map[dec, stiffEff],
      "exact" -> "1/(1/k_H + 1/k_p) per k_p"
    |>
  |>,
  "tolerances" -> <|
    "effective_modulus_abs" -> 1.0*^-6,
    "effective_modulus_rel" -> 1.0*^-12,
    "contact_radius_abs" -> 1.0*^-15,
    "contact_radius_rel" -> 1.0*^-12,
    "hertz_approach_abs" -> 1.0*^-15,
    "hertz_approach_rel" -> 1.0*^-12,
    "hertz_stiffness_abs" -> 1.0*^-6,
    "hertz_stiffness_rel" -> 1.0*^-12,
    "approach_total_abs" -> 1.0*^-15,
    "approach_total_rel" -> 1.0*^-12,
    "effective_stiffness_abs" -> 1.0*^-6,
    "effective_stiffness_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
