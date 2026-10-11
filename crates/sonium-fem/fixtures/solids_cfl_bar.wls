(* Independent Wolfram reference for the 1-D bar CFL limit.
   Derives, at 50-digit working precision, the bar sound speed c = Sqrt[E/rho],
   the critical step dt_crit = L_e/c per element length, and the critical step
   of a stepped mesh via the min rule dt = Min[L_e]/c. Steel bar:
   E = 200 GPa, rho = 7800 kg/m^3; L_e in {0.01, 0.1, 1.0} m; stepped mesh
   {1.0, 0.1, 0.01} m. All inputs are exact rationals; N[..., 50] is applied
   once per exported value. The Rust side consumes solids_cfl_bar_golden.json;
   this script is the derivation.
   Regen: wolframscript -script fixtures/solids_cfl_bar.wls > fixtures/solids_cfl_bar_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, prec]]]];

young = 200*10^9;
rho = 7800;
sound = Sqrt[young/rho];
les = {1/100, 1/10, 1};
dts = #/sound & /@ les;
stepped = {1, 1/10, 1/100};
dtStepped = Min[stepped]/sound;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "cfl_bar",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_modulus_pa" -> young,
    "density_kg_m3" -> rho,
    "lengths_m" -> N[les],
    "stepped_lengths_m" -> N[stepped]
  |>,
  "values" -> <|
    "sound_speed" -> <|
      "values" -> {dec[sound]},
      "exact" -> "Sqrt[E/rho]"
    |>,
    "dt_crit" -> <|
      "values" -> dec /@ dts,
      "exact" -> "L_e/c per length"
    |>,
    "dt_crit_stepped" -> <|
      "values" -> {dec[dtStepped]},
      "exact" -> "Min[L_e]/c (stepped-mesh min rule)"
    |>
  |>,
  "tolerances" -> <|
    "sound_speed_abs" -> 1.0*^-15,
    "sound_speed_rel" -> 1.0*^-12,
    "dt_crit_abs" -> 1.0*^-15,
    "dt_crit_rel" -> 1.0*^-12,
    "dt_crit_stepped_abs" -> 1.0*^-15,
    "dt_crit_stepped_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
