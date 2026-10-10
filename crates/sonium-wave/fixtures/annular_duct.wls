(* Independent Wolfram reference for hard-wall annular-duct radial roots.
   Solves J_m'(s x) Y_m'(x) - J_m'(x) Y_m'(s x) = 0 by bracketing sign
   changes on a fine grid and refining with FindRoot at 50-digit
   working precision. The Rust side consumes annular_duct_golden.json
   through annular_duct_radial_root. Rank convention (matching Rust):
   m = 0 counts nonzero roots from radial_mode 1; m >= 1 from
   radial_mode 0. Hub ratio s = 0.4.
   Run: wolframscript -script annular_duct.wls > annular_duct_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86-64. *)

sigma = SetPrecision[2/5, 60];
(* NOTE: keep the discriminant on ONE line: in .wls a newline after a
   complete expression silently splits the definition (see history). *)
disc[m_, x_] := (D[BesselJ[m, t], t] /. t -> sigma x) (D[BesselY[m, t], t] /. t -> x) - (D[BesselJ[m, t], t] /. t -> x) (D[BesselY[m, t], t] /. t -> sigma x);

(* Bracket the n-th positive root by scanning for sign changes. *)
findRoot[m_, rank_] := Module[{grid, brackets, x},
  grid = Range[SetPrecision[1/100, 60], 40, SetPrecision[1/200, 60]];
  brackets = Select[Partition[grid, 2, 1],
    disc[m, #[[1]]] disc[m, #[[2]]] < 0 &];
  x = (brackets[[rank, 1]] + brackets[[rank, 2]])/2;
  FindRoot[disc[m, xx] == 0, {xx, x},
    WorkingPrecision -> 55, AccuracyGoal -> 45, PrecisionGoal -> 45]
];

roots = <|
  "m0_rank1" -> (xx /. findRoot[0, 1]),
  "m0_rank2" -> (xx /. findRoot[0, 2]),
  "m1_rank0" -> (xx /. findRoot[1, 1]),
  "m1_rank1" -> (xx /. findRoot[1, 2]),
  "m3_rank0" -> (xx /. findRoot[3, 1])
|>;

prec = 50;
golden = <|
  "schema_version" -> 1,
  "case_id" -> "wave.annular-duct.v1",
  "convention" -> "Hard-wall annulus sigma = 0.4; chi = kappa a roots of J_m'(s x)Y_m'(x) - J_m'(x)Y_m'(s x) = 0",
  "generator" -> "crates/sonium-wave/fixtures/annular_duct.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> 55,
  "hub_tip_ratio" -> N[sigma, prec],
  "roots" -> <|
    "m0_rank1" -> N[roots["m0_rank1"], prec],
    "m0_rank2" -> N[roots["m0_rank2"], prec],
    "m1_rank0" -> N[roots["m1_rank0"], prec],
    "m1_rank1" -> N[roots["m1_rank1"], prec],
    "m3_rank0" -> N[roots["m3_rank0"], prec]
  |>,
  "tolerances" -> <|
    "root_abs" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
