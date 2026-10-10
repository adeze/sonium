(* Independent Wolfram reference for the Hertz pressure profile.
   Derives, at 50-digit working precision, the effective modulus, contact
   radius, peak pressure, half-radius pressure, and normal approach of a
   steel sphere on a steel half-space. All inputs are exact rationals; the
   cube root stays exact until the final N[..., 50]. The Rust side consumes
   solids_hertz_pressure_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_hertz_pressure.wls > fixtures/solids_hertz_pressure_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

force = 1000;
radius = 5/100;
young = 200*10^9;
poisson = 3/10;

estar = 1/((1 - poisson^2)/young + (1 - poisson^2)/young);
contactA = (3*force*radius/(4*estar))^(1/3);
p0 = 3*force/(2*Pi*contactA^2);
pHalf = p0*Sqrt[1 - (1/2)^2];
approach = contactA^2/radius;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "hertz_pressure",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "force_n" -> 1000,
    "radius_m" -> 0.05,
    "youngs_modulus_pa" -> 200*10^9,
    "poisson_ratio" -> 0.3
  |>,
  "values" -> <|
    "effective_modulus" -> <|
      "values" -> {dec[estar]},
      "exact" -> "1/((1-nu^2)/E+(1-nu^2)/E)"
    |>,
    "contact_radius" -> <|
      "values" -> {dec[contactA]},
      "exact" -> "(3 F R/(4 E*))^(1/3)"
    |>,
    "peak_pressure" -> <|
      "values" -> {dec[p0]},
      "exact" -> "3 F/(2 pi a^2)"
    |>,
    "pressure_half_radius" -> <|
      "values" -> {dec[pHalf]},
      "exact" -> "p0 sqrt(1-(1/2)^2)"
    |>,
    "approach" -> <|
      "values" -> {dec[approach]},
      "exact" -> "a^2/R"
    |>
  |>,
  "tolerances" -> <|
    "effective_modulus_abs" -> 1.0*^-6,
    "effective_modulus_rel" -> 1.0*^-12,
    "contact_radius_abs" -> 1.0*^-15,
    "contact_radius_rel" -> 1.0*^-12,
    "peak_pressure_abs" -> 1.0*^-6,
    "peak_pressure_rel" -> 1.0*^-12,
    "pressure_half_radius_abs" -> 1.0*^-6,
    "pressure_half_radius_rel" -> 1.0*^-12,
    "approach_abs" -> 1.0*^-15,
    "approach_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
