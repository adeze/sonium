(* Independent Wolfram reference for the cohesive patch QA.
   Derives, at 50-digit working precision, the epoxy bilinear law onset
   and failure openings, the peak traction, and the dissipated-energy
   fraction at the midpoint opening. All inputs are exact rationals.
   The Rust side consumes solids_cohesive_patch_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_cohesive_patch.wls > fixtures/solids_cohesive_patch_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

stiff = 1*10^12;
strength = 30*10^6;
energy = 500;

d0 = strength/stiff;
df = 2*energy/strength;
dmax = (d0 + df)/2;
branch = (df - dmax)/(df - d0);
dissipated = energy*(dmax - branch*d0)/df;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "cohesive_patch",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "penalty_stiffness" -> 1*10^12,
    "strength_pa" -> 30*10^6,
    "fracture_energy" -> 500
  |>,
  "values" -> <|
    "onset_opening" -> <|
      "values" -> {dec[d0]},
      "exact" -> "sigma_c / K"
    |>,
    "failure_opening" -> <|
      "values" -> {dec[df]},
      "exact" -> "2 G / sigma_c"
    |>,
    "peak_traction" -> <|
      "values" -> {dec[strength]},
      "exact" -> "sigma_c"
    |>,
    "dissipated_at_midpoint" -> <|
      "values" -> {dec[dissipated]},
      "exact" -> "G (dmax - branch d0)/df at dmax=(d0+df)/2"
    |>,
    "fracture_energy_check" -> <|
      "values" -> {dec[energy]},
      "exact" -> "G"
    |>
  |>,
  "tolerances" -> <|
    "onset_opening_abs" -> 1.0*^-18,
    "onset_opening_rel" -> 1.0*^-12,
    "failure_opening_abs" -> 1.0*^-18,
    "failure_opening_rel" -> 1.0*^-12,
    "peak_traction_abs" -> 1.0*^-6,
    "peak_traction_rel" -> 1.0*^-12,
    "dissipated_at_midpoint_abs" -> 1.0*^-9,
    "dissipated_at_midpoint_rel" -> 1.0*^-9,
    "fracture_energy_check_abs" -> 1.0*^-9,
    "fracture_energy_check_rel" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
