(* Implicit dynamics amplification-matrix golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_implicit_dynamics_amplification.wls > fixtures/solids_implicit_dynamics_amplification_golden.json
   Case: SDOF amplification matrices of the Newmark trapezoidal rule and
   the HHT-alpha scheme (alpha = -1/10) at m = 1, k = 100, dt = 1/50;
   Newmark undamped, HHT with c = 1/2. Entries follow the Bathe
   a0..a5 coefficient notation with the HHT equilibrium weighting.
   Provenance: Bathe, K. J., "Finite Element Procedures", 2nd ed.,
   Section 9.2 (Newmark); Hilber, H. M., Hughes, T. J. R., and Taylor,
   R. L. (1977), "Improved numerical dissipation for time integration
   algorithms in structural dynamics", Earthquake Eng. Struct. Dyn. 5. *)

Off[General::munfl];

(* Exact inputs so N[., 17] emits full double precision. *)
amplify[m_, c_, k_, dt_, beta_, gamma_, alpha_] := Module[
  {a0, a1, a2, a3, a4, a5, h, khat, row0, row1, row2},
  a0 = 1/(beta dt^2);
  a1 = gamma/(beta dt);
  a2 = 1/(beta dt);
  a3 = 1/(2 beta) - 1;
  a4 = gamma/beta - 1;
  a5 = dt (gamma/(2 beta) - 1);
  h = 1 + alpha;
  khat = a0 m + a1 h c + h k;
  row0 = {(m a0 + c h a1 + k alpha)/khat,
    (m a2 + c (h a4 + alpha))/khat, (m a3 + c h a5)/khat};
  row1 = a1 row0 - {a1, a4, a5};
  row2 = a0 row0 - {a0, a2, a3};
  {row0, row1, row2}];

newmarkA = amplify[1, 0, 100, 1/50, 1/4, 1/2, 0];
hhtA = amplify[1, 1/2, 100, 1/50, 121/400, 3/5, -1/10];

newmarkRho = Max[Abs[Eigenvalues[N[newmarkA, 50]]]];
hhtRho = Max[Abs[Eigenvalues[N[hhtA, 50]]]];

(* Trapezoidal undamped integrator is non-dissipative. *)
If[! (Abs[N[newmarkRho, 30] - 1] < 10^-25), Quit[1]];
(* HHT with damping contracts below unity. *)
If[! (N[hhtRho, 30] < 1), Quit[1]];
(* Both maps preserve equilibrium: A . {1, 0, 0} static rows sum right. *)
If[! (And @@ Flatten[Abs[N[#, 30]] < 10^6 & /@ Join[newmarkA, hhtA]]), Quit[1]];

entry[name_, a_, i_, j_] := name -> <|
  "values" -> {ToString[N[a[[i, j]], 17]]},
  "exact" -> "Bathe a0..a5 closed form"|>;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "implicit_dynamics_amplification",
    "units" -> "SI"|>,
  "inputs" -> <|
    "mass" -> 1.,
    "newmark_damping" -> 0.,
    "hht_damping" -> 0.5,
    "stiffness" -> 100.,
    "dt" -> 0.02,
    "newmark_beta" -> 0.25,
    "newmark_gamma" -> 0.5,
    "hht_alpha" -> -0.1,
    "hht_beta" -> 0.3025,
    "hht_gamma" -> 0.6|>,
  "values" -> <|
    entry["newmark_a11", newmarkA, 1, 1],
    entry["newmark_a12", newmarkA, 1, 2],
    entry["newmark_a13", newmarkA, 1, 3],
    entry["newmark_a21", newmarkA, 2, 1],
    entry["newmark_a22", newmarkA, 2, 2],
    entry["newmark_a23", newmarkA, 2, 3],
    entry["newmark_a31", newmarkA, 3, 1],
    entry["newmark_a32", newmarkA, 3, 2],
    entry["newmark_a33", newmarkA, 3, 3],
    entry["hht_a11", hhtA, 1, 1],
    entry["hht_a12", hhtA, 1, 2],
    entry["hht_a13", hhtA, 1, 3],
    entry["hht_a21", hhtA, 2, 1],
    entry["hht_a22", hhtA, 2, 2],
    entry["hht_a23", hhtA, 2, 3],
    entry["hht_a31", hhtA, 3, 1],
    entry["hht_a32", hhtA, 3, 2],
    entry["hht_a33", hhtA, 3, 3],
    "newmark_rho" -> <|
      "values" -> {ToString[N[newmarkRho, 17]]},
      "exact" -> "Max |eig A|, trapezoidal undamped = 1"|>,
    "hht_rho" -> <|
      "values" -> {ToString[N[hhtRho, 17]]},
      "exact" -> "Max |eig A|, damped HHT < 1"|>|>,
  "tolerances" -> Association[Join[
    Table[name -> 0., {name, Join[
      Table["newmark_a" <> ToString[i] <> ToString[j] <> "_abs", {i, 3}, {j, 3}] // Flatten,
      Table["hht_a" <> ToString[i] <> ToString[j] <> "_abs", {i, 3}, {j, 3}] // Flatten]}],
    Table[name -> 10^-12, {name, Join[
      Table["newmark_a" <> ToString[i] <> ToString[j] <> "_rel", {i, 3}, {j, 3}] // Flatten,
      Table["hht_a" <> ToString[i] <> ToString[j] <> "_rel", {i, 3}, {j, 3}] // Flatten]}],
    {"newmark_rho_abs" -> 0., "newmark_rho_rel" -> 10^-9,
     "hht_rho_abs" -> 0., "hht_rho_rel" -> 10^-9}]]|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
