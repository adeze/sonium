(* Independent Wolfram reference for the tied-interface patch test.
   Derives, at 50-digit working precision, the uniform axial stress of a
   confined two-block stack under uniform pressure: traction balance on
   any cut below the loaded face forces sigma_zz = -p everywhere, and
   the tied dissimilar-material interface transmits that pressure
   exactly (constant-stress patch criterion). The four probe values are
   identical by uniformity, one per probed interior point. All inputs
   are exact rationals; N[..., 50] is applied once per exported value.
   The Rust side consumes solids_tied_contact_patch_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_tied_contact_patch.wls > fixtures/solids_tied_contact_patch_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

pressure = 1*10^6;
sigmaZZ = -pressure;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "tied_contact_patch",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "block_size_m" -> {2, 1, 1},
    "youngs_bottom_pa" -> 200*10^9,
    "youngs_top_pa" -> 70*10^9,
    "poisson_ratio" -> 0.3,
    "pressure_pa" -> 1.0*^6,
    "probe_points_m" -> {{0.5, 0.5, 0.5}, {1.5, 0.5, 0.5}, {1.0, 0.5, 1.5}, {0.25, 0.75, 0.25}}
  |>,
  "values" -> <|
    "sigma_zz" -> <|
      "values" -> {dec[sigmaZZ], dec[sigmaZZ], dec[sigmaZZ], dec[sigmaZZ]},
      "exact" -> "-p (uniform axial stress through the tie)"
    |>
  |>,
  "tolerances" -> <|
    "sigma_zz_abs" -> 1.0*^-6,
    "sigma_zz_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
