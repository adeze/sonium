(* Independent Wolfram reference for the 1-D spring and rivet.
   Derives, at 50-digit working precision, the linear/nonlinear spring
   forces, diagonal-axis projection, rupture energy, critical step,
   and the rivet interaction utilizations. All inputs are exact
   rationals; roots stay exact until N[..., 50].
   The Rust side consumes solids_spring_rivet_1d_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_spring_rivet_1d.wls > fixtures/solids_spring_rivet_1d_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Linear spring k = 2000, c = 5 at d = 1/20, v = 2:
   F = k d + c v. *)
linForce = 2000*(1/20) + 5*2;

(* Nonlinear spring k = 1000 plus curve (0,0)-(1/10,50) at d = 1/20:
   F = k d + 25; tangent k + 500. Clamped end at d = 1:
   F = 1000 + 50. *)
nlForce = 1000*(1/20) + 25;
nlTangent = 1000 + 500;
clampForce = 1000*1 + 50;

(* Diagonal spring (0,0,0)-(3,4,0), rest 4, k = 1000:
   length 5, elongation 1, axis (3/5,4/5,0), F = 1000. *)
axisForce = {600, 800, 0};

(* Trapezoidal energy of the 0 -> 1/10 ramp to 100 N. *)
energy = (0 + 100)/2*(1/10);

(* Critical step for m1 = m2 = 2, k = 1000:
   dt = 2/Sqrt[k (1/m1 + 1/m2)]. *)
dtCrit = 2/Sqrt[1000*(1/2 + 1/2)];

(* Rivet penalty 1e6/1e3, capacities 5000/2000/50.
   Shear 1000 N: U = (1000/2000)^2 = 1/4.
   Plus moment 30 N m: U = 1/4 + (30/50)^2 = 61/100.
   Axial 6000 N: U = (6000/5000)^2 = 36/25. *)
rivetShear = (1000/2000)^2;
rivetCombined = (1000/2000)^2 + (30/50)^2;
rivetAxial = (6000/5000)^2;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "spring_rivet_1d",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "linear" -> "k 2000, c 5, d 0.05, v 2",
    "nonlinear" -> "k 1000, curve (0,0)-(0.1,50), d 0.05",
    "diagonal" -> "(0,0,0)-(3,4,0), rest 4, k 1000",
    "ramp" -> "0 -> 0.1 m to 100 N",
    "step" -> "m1 = m2 = 2, k 1000",
    "rivet" -> "penalty 1e6/1e3, caps 5000/2000/50"
  |>,
  "values" -> <|
    "linear_force" -> <|
      "values" -> {dec[linForce]},
      "exact" -> "k d + c v"
    |>,
    "nonlinear_force" -> <|
      "values" -> {dec[nlForce]},
      "exact" -> "k d + curve(d)"
    |>,
    "nonlinear_tangent" -> <|
      "values" -> {dec[nlTangent]},
      "exact" -> "k + curve slope"
    |>,
    "clamped_force" -> <|
      "values" -> {dec[clampForce]},
      "exact" -> "k d + endpoint ordinate"
    |>,
    "axis_force" -> <|
      "values" -> dec /@ axisForce,
      "exact" -> "F axis on node 2"
    |>,
    "ramp_energy" -> <|
      "values" -> {dec[energy]},
      "exact" -> "(F0 + F1)/2 (d1 - d0)"
    |>,
    "critical_step" -> <|
      "values" -> {dec[dtCrit]},
      "exact" -> "2/Sqrt[k (1/m1 + 1/m2)]"
    |>,
    "rivet_shear" -> <|
      "values" -> {dec[rivetShear]},
      "exact" -> "(Fs/Fsmax)^2"
    |>,
    "rivet_combined" -> <|
      "values" -> {dec[rivetCombined]},
      "exact" -> "(Fs/Fsmax)^2 + (M/Mmax)^2"
    |>,
    "rivet_axial" -> <|
      "values" -> {dec[rivetAxial]},
      "exact" -> "(Fa/Famax)^2"
    |>
  |>,
  "tolerances" -> <|
    "linear_force_abs" -> 1.0*^-9,
    "linear_force_rel" -> 1.0*^-12,
    "nonlinear_force_abs" -> 1.0*^-9,
    "nonlinear_force_rel" -> 1.0*^-12,
    "nonlinear_tangent_abs" -> 1.0*^-9,
    "nonlinear_tangent_rel" -> 1.0*^-12,
    "clamped_force_abs" -> 1.0*^-9,
    "clamped_force_rel" -> 1.0*^-12,
    "axis_force_abs" -> 1.0*^-9,
    "axis_force_rel" -> 1.0*^-12,
    "ramp_energy_abs" -> 1.0*^-12,
    "ramp_energy_rel" -> 1.0*^-12,
    "critical_step_abs" -> 1.0*^-15,
    "critical_step_rel" -> 1.0*^-12,
    "rivet_shear_abs" -> 1.0*^-15,
    "rivet_shear_rel" -> 1.0*^-12,
    "rivet_combined_abs" -> 1.0*^-15,
    "rivet_combined_rel" -> 1.0*^-12,
    "rivet_axial_abs" -> 1.0*^-15,
    "rivet_axial_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
