(* Independent Wolfram reference for RBODY rigid kinematics QA.
   Derives, at 50-digit working precision, the pair distances of a 4-node
   tetra under x = R(theta(t)) x0 + t(t) with theta(t) = 0.3 t about z,
   the rotation determinants, the angular velocity recovered from the
   skew part of Rdot . Transpose[R], and the slave positions.
   All inputs are exact rationals; Sqrt, Sin, and Cos stay exact until
   N[..., 50]. The Rust side consumes
   solids_rbody_kinematics_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_rbody_kinematics.wls > fixtures/solids_rbody_kinematics_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

theta[t_] := (3/10) t;
rot[t_] := {{Cos[theta[t]], -Sin[theta[t]], 0}, {Sin[theta[t]], Cos[theta[t]], 0}, {0, 0, 1}};
trans[t_] := {t/10, t/5, (3/10) t};
nodes0 = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
pos[t_] := Table[rot[t] . n + trans[t], {n, nodes0}];
pairs = {{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
dist[t_, {i_, j_}] := Norm[pos[t][[i]] - pos[t][[j]]];

(* Angular velocity from the skew part of Rdot . R^T. *)
rdot[t_] := D[rot[tt], tt] /. tt -> t;
skew[t_] := rdot[t] . Transpose[rot[t]];
omega[t_] := {(skew[t][[3, 2]] - skew[t][[2, 3]])/2, (skew[t][[1, 3]] - skew[t][[3, 1]])/2, (skew[t][[2, 1]] - skew[t][[1, 2]])/2};

t0 = 0; t05 = 1/2; t1 = 1;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "rbody_kinematics",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "nodes" -> {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    "rotation_axis" -> {0, 0, 1},
    "theta_rate_rad_s" -> 0.3,
    "translation_rate_m_s" -> {0.1, 0.2, 0.3},
    "times_s" -> {0, 0.5, 1.0}
  |>,
  "values" -> <|
    "pair_distances_t0" -> <|
      "values" -> Table[dec[dist[t0, p]], {p, pairs}],
      "exact" -> "Norm[R(0).(xi-xj)] over 6 tetra pairs"
    |>,
    "pair_distances_t05" -> <|
      "values" -> Table[dec[dist[t05, p]], {p, pairs}],
      "exact" -> "Norm[R(0.15).(xi-xj)] over 6 tetra pairs"
    |>,
    "pair_distances_t1" -> <|
      "values" -> Table[dec[dist[t1, p]], {p, pairs}],
      "exact" -> "Norm[R(0.3).(xi-xj)] over 6 tetra pairs"
    |>,
    "det_rotation_t0" -> <|
      "values" -> {dec[Det[rot[t0]]]},
      "exact" -> "Det[R(0)] = 1"
    |>,
    "det_rotation_t05" -> <|
      "values" -> {dec[Det[rot[t05]]]},
      "exact" -> "Det[R(0.15)] = 1"
    |>,
    "det_rotation_t1" -> <|
      "values" -> {dec[Det[rot[t1]]]},
      "exact" -> "Det[R(0.3)] = 1"
    |>,
    "omega_t0" -> <|
      "values" -> Table[dec[omega[t0][[k]]], {k, 3}],
      "exact" -> "axial(Rdot.R^T) at t = 0"
    |>,
    "omega_t05" -> <|
      "values" -> Table[dec[omega[t05][[k]]], {k, 3}],
      "exact" -> "axial(Rdot.R^T) at t = 0.5"
    |>,
    "omega_t1" -> <|
      "values" -> Table[dec[omega[t1][[k]]], {k, 3}],
      "exact" -> "axial(Rdot.R^T) at t = 1"
    |>,
    "positions_t05" -> <|
      "values" -> Flatten[Table[dec[c], {n, pos[t05]}, {c, n}]],
      "exact" -> "R(0.15).x0 + t(0.5), node-major"
    |>,
    "positions_t1" -> <|
      "values" -> Flatten[Table[dec[c], {n, pos[t1]}, {c, n}]],
      "exact" -> "R(0.3).x0 + t(1), node-major"
    |>
  |>,
  "tolerances" -> <|
    "pair_distances_t0_abs" -> 1.0*^-12,
    "pair_distances_t0_rel" -> 1.0*^-12,
    "pair_distances_t05_abs" -> 1.0*^-12,
    "pair_distances_t05_rel" -> 1.0*^-12,
    "pair_distances_t1_abs" -> 1.0*^-12,
    "pair_distances_t1_rel" -> 1.0*^-12,
    "det_rotation_t0_abs" -> 1.0*^-14,
    "det_rotation_t0_rel" -> 1.0*^-14,
    "det_rotation_t05_abs" -> 1.0*^-14,
    "det_rotation_t05_rel" -> 1.0*^-14,
    "det_rotation_t1_abs" -> 1.0*^-14,
    "det_rotation_t1_rel" -> 1.0*^-14,
    "omega_t0_abs" -> 1.0*^-12,
    "omega_t0_rel" -> 1.0*^-12,
    "omega_t05_abs" -> 1.0*^-12,
    "omega_t05_rel" -> 1.0*^-12,
    "omega_t1_abs" -> 1.0*^-12,
    "omega_t1_rel" -> 1.0*^-12,
    "positions_t05_abs" -> 1.0*^-12,
    "positions_t05_rel" -> 1.0*^-12,
    "positions_t1_abs" -> 1.0*^-12,
    "positions_t1_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
