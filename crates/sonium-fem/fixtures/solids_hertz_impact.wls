(* Independent Wolfram reference for the Hertz impact QA.
   Derives, at 50-digit working precision, the maximum approach, peak
   force, and Rayleigh contact time of an elastic sphere impact from the
   Hertzian energy balance. All inputs are exact rationals; powers stay
   exact until N[..., 50]. The Rust side consumes
   solids_hertz_impact_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_hertz_impact.wls > fixtures/solids_hertz_impact_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

mass = 2;
vel = 3;
radius = 5/100;
estar = 110*10^9;

approach = (15*mass*vel^2/(16*estar*Sqrt[radius]))^(2/5);
peak = 4/3*estar*Sqrt[radius]*approach^(3/2);
time = 294/100*approach/vel;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "hertz_impact",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "mass_kg" -> 2,
    "velocity" -> 3,
    "radius_m" -> 0.05,
    "effective_modulus_pa" -> 110*10^9
  |>,
  "values" -> <|
    "max_approach" -> <|
      "values" -> {dec[approach]},
      "exact" -> "(15 m V^2/(16 E* Sqrt[R]))^(2/5)"
    |>,
    "peak_force" -> <|
      "values" -> {dec[peak]},
      "exact" -> "(4/3) E* Sqrt[R] delta^(3/2)"
    |>,
    "contact_time" -> <|
      "values" -> {dec[time]},
      "exact" -> "2.94 delta/V (Rayleigh)"
    |>
  |>,
  "tolerances" -> <|
    "max_approach_abs" -> 1.0*^-15,
    "max_approach_rel" -> 1.0*^-12,
    "peak_force_abs" -> 1.0*^-6,
    "peak_force_rel" -> 1.0*^-12,
    "contact_time_abs" -> 1.0*^-15,
    "contact_time_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
