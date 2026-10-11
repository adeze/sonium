(* Independent Wolfram reference for uniform mass-scaling frequency shift.
   Free-free 2-DOF spring chain, m = 1 kg per node, k = 1000 N/m:
   K = k {{1,-1},{-1,1}}, M = m I, eigenfrequencies w1 = 0 (rigid mode),
   w2 = Sqrt[2k/m]. Uniform scaling by s multiplies every mass by s^2,
   shifting each eigenfrequency w -> w/s and the modal critical step
   dt = 2/w_max by s, with per-node added mass (s^2-1) m, for
   s in {1.0, 2.0, 10.0}. Rigid-mode kinetic energy at v = {1,1} m/s
   scales by exactly s^2, i.e. the scaling-law error
   KE_scaled - s^2 KE_unscaled is exactly 0. All inputs are exact
   rationals; N[..., 50] is applied once per exported value. The Rust
   side consumes solids_mass_scaling_shift_golden.json; this script is
   the derivation.
   Regen: wolframscript -script fixtures/solids_mass_scaling_shift.wls > fixtures/solids_mass_scaling_shift_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, prec]]]];

mass = 1;
stiff = 1000;
scales = {1, 2, 10};
vel = {1, 1};
omegaElastic = Sqrt[2*stiff/mass];
omegaScaled = omegaElastic/# & /@ scales;
dtScaled = 2/# & /@ omegaScaled;
added = (#^2 - 1)*mass & /@ scales;
keUnscaled = (mass*vel[[1]]^2 + mass*vel[[2]]^2)/2;
keScaled = (#^2*mass*vel[[1]]^2 + #^2*mass*vel[[2]]^2)/2 & /@ scales;
keError = Table[keScaled[[i]] - scales[[i]]^2*keUnscaled, {i, Length[scales]}];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "mass_scaling_shift",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "mass_kg" -> mass,
    "stiffness_n_m" -> stiff,
    "scales" -> N[scales],
    "rigid_velocity_m_s" -> N[vel]
  |>,
  "values" -> <|
    "omega_rigid" -> <|
      "values" -> {dec[0]},
      "exact" -> "0 (rigid mode unaffected by uniform scaling)"
    |>,
    "omega_elastic_unscaled" -> <|
      "values" -> {dec[omegaElastic]},
      "exact" -> "Sqrt[2*k/m]"
    |>,
    "omega_elastic_scaled" -> <|
      "values" -> dec /@ omegaScaled,
      "exact" -> "Sqrt[2*k/m]/s per scale"
    |>,
    "dt_crit_scaled" -> <|
      "values" -> dec /@ dtScaled,
      "exact" -> "2*s/Sqrt[2*k/m] per scale"
    |>,
    "added_mass_per_node" -> <|
      "values" -> dec /@ added,
      "exact" -> "(s^2-1)*m per scale"
    |>,
    "ke_rigid_unscaled" -> <|
      "values" -> {dec[keUnscaled]},
      "exact" -> "m*(v1^2+v2^2)/2"
    |>,
    "ke_rigid_scaled" -> <|
      "values" -> dec /@ keScaled,
      "exact" -> "s^2*m*(v1^2+v2^2)/2 per scale"
    |>,
    "ke_scaling_error" -> <|
      "values" -> dec /@ keError,
      "exact" -> "KE_scaled - s^2*KE_unscaled = 0 per scale"
    |>
  |>,
  "tolerances" -> <|
    "omega_rigid_abs" -> 1.0*^-15,
    "omega_rigid_rel" -> 1.0*^-12,
    "omega_elastic_unscaled_abs" -> 1.0*^-15,
    "omega_elastic_unscaled_rel" -> 1.0*^-12,
    "omega_elastic_scaled_abs" -> 1.0*^-15,
    "omega_elastic_scaled_rel" -> 1.0*^-12,
    "dt_crit_scaled_abs" -> 1.0*^-15,
    "dt_crit_scaled_rel" -> 1.0*^-12,
    "added_mass_per_node_abs" -> 1.0*^-15,
    "added_mass_per_node_rel" -> 1.0*^-12,
    "ke_rigid_unscaled_abs" -> 1.0*^-15,
    "ke_rigid_unscaled_rel" -> 1.0*^-12,
    "ke_rigid_scaled_abs" -> 1.0*^-15,
    "ke_rigid_scaled_rel" -> 1.0*^-12,
    "ke_scaling_error_abs" -> 1.0*^-15,
    "ke_scaling_error_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
