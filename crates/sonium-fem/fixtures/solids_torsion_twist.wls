(* Independent Wolfram reference for pure-torsion twist.
   Derives, at 50-digit working precision, the torsional constant J = pi r^4/2,
   the twist phi = T L/(G J), and the peak shear stress tau = T r/J for a
   solid circular shaft. All inputs are exact rationals. The Rust side consumes
   solids_torsion_twist_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_torsion_twist.wls > fixtures/solids_torsion_twist_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

torque = 500;
len = 6/5;
radius = 2/100;
shear = 79*10^9;

polar = Pi*radius^4/2;
twist = torque*len/(shear*polar);
maxShear = torque*radius/polar;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "torsion_twist",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "torque_n_m" -> 500,
    "length_m" -> 1.2,
    "radius_m" -> 0.02,
    "shear_modulus_pa" -> 79*10^9
  |>,
  "values" -> <|
    "torsional_constant" -> <|
      "values" -> {dec[polar]},
      "exact" -> "Pi*r^4/2"
    |>,
    "twist" -> <|
      "values" -> {dec[twist]},
      "exact" -> "T*L/(G*J)"
    |>,
    "max_shear_stress" -> <|
      "values" -> {dec[maxShear]},
      "exact" -> "T*r/J"
    |>
  |>,
  "tolerances" -> <|
    "torsional_constant_abs" -> 1.0*^-20,
    "torsional_constant_rel" -> 1.0*^-12,
    "twist_abs" -> 1.0*^-15,
    "twist_rel" -> 1.0*^-12,
    "max_shear_stress_abs" -> 1.0*^-9,
    "max_shear_stress_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
