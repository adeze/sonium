(* Independent Wolfram reference for fabric membranes and concrete.
   Derives, at 50-digit working precision, LAW19 tension/compression
   split values, a LAW58 load/unload pair, Mazars damage at a uniaxial
   state, the hydrostatic cap pressure, and the Duvaut-Lions halving
   ratio. Transcendentals print at 50 digits; the rest are rationals.
   The Rust side consumes fabric_concrete_golden.json; this script is
   the derivation.
   Regen: wolframscript -script fixtures/fabric_concrete.wls > fixtures/fabric_concrete_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

e1t = 10*10^6; e1c = 1*10^6; g12 = 2*10^6;
eu = 20*10^6; loadCurve[x_] := 10^6*x/(1/10);

e0 = 1/10^4; at = 4/5; bt = 15000;
mazarsTension[eq_] := 1 - (1 - at)*e0/eq - at*Exp[-bt*(eq - e0)];

young = 30*10^9; nu = 1/5; bulk = young/(3*(1 - 2*nu));

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "fabric_concrete",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "fabric_e1t_pa" -> 10*10^6,
    "fabric_e1c_pa" -> 1*10^6,
    "fabric_g12_pa" -> 2*10^6,
    "fabric_eu_pa" -> 20*10^6,
    "concrete_young_pa" -> 30*10^9,
    "concrete_poisson" -> 0.2,
    "mazars_strain0" -> 1*10^-4,
    "mazars_at" -> 0.8,
    "mazars_bt" -> 15000,
    "concrete_cap0_pa" -> 30*10^6
  |>,
  "values" -> <|
    "fabric_tension" -> <|
      "values" -> {dec[e1t/100]},
      "exact" -> "10^7 * 0.01 = 10^5"
    |>,
    "fabric_compression" -> <|
      "values" -> {dec[-e1c/100]},
      "exact" -> "-10^6 * 0.01 = -10^4"
    |>,
    "fabric_shear" -> <|
      "values" -> {dec[g12/100]},
      "exact" -> "2*10^6 * 0.01 = 2*10^4"
    |>,
    "fabric_loaded" -> <|
      "values" -> {dec[loadCurve[1/20]]},
      "exact" -> "loading curve at 0.05 = 5*10^5"
    |>,
    "fabric_unloaded" -> <|
      "values" -> {dec[loadCurve[1/20] - eu*(1/20 - 1/50)]},
      "exact" -> "5*10^5 - 2*10^7 * 0.03 = -10^5"
    |>,
    "mazars_uniaxial" -> <|
      "values" -> {dec[mazarsTension[2/10^4]]},
      "exact" -> "9/10 - (4/5) e^-1.5 at eq = 2*10^-4"
    |>,
    "mazars_below_threshold" -> <|
      "values" -> {dec[0]},
      "exact" -> "0 below the threshold strain"
    |>,
    "concrete_cap" -> <|
      "values" -> {dec[30*10^6]},
      "exact" -> "trial 5*10^8 caps at 3*10^7"
    |>,
    "concrete_trial_pressure" -> <|
      "values" -> {dec[bulk*3/100]},
      "exact" -> "K * 0.03 = 5*10^8"
    |>,
    "concrete_visco_ratio" -> <|
      "values" -> {dec[1/2]},
      "exact" -> "(inviscid + w old)/(1 + w) halves at w = 1"
    |>
  |>,
  "tolerances" -> <|
    "fabric_tension_abs" -> 1.0*^-9,
    "fabric_tension_rel" -> 1.0*^-12,
    "fabric_compression_abs" -> 1.0*^-12,
    "fabric_compression_rel" -> 1.0*^-12,
    "fabric_shear_abs" -> 1.0*^-9,
    "fabric_shear_rel" -> 1.0*^-12,
    "fabric_loaded_abs" -> 1.0*^-9,
    "fabric_loaded_rel" -> 1.0*^-12,
    "fabric_unloaded_abs" -> 1.0*^-9,
    "fabric_unloaded_rel" -> 1.0*^-12,
    "mazars_uniaxial_abs" -> 1.0*^-12,
    "mazars_uniaxial_rel" -> 1.0*^-12,
    "mazars_below_threshold_abs" -> 1.0*^-15,
    "mazars_below_threshold_rel" -> 1.0*^-15,
    "concrete_cap_abs" -> 1.0*^-6,
    "concrete_cap_rel" -> 1.0*^-9,
    "concrete_trial_pressure_abs" -> 1.0*^-6,
    "concrete_trial_pressure_rel" -> 1.0*^-12,
    "concrete_visco_ratio_abs" -> 1.0*^-15,
    "concrete_visco_ratio_rel" -> 1.0*^-15
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
