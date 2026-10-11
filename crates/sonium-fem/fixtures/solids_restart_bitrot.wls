(* Independent Wolfram reference for the restart bitrot guard.
   Pins, at 50-digit working precision, the reference sums and sums of
   squares of a fixed 4-node ExplicitState. All state entries are exact
   dyadic rationals (exactly representable in binary64), so the Rust
   replay test parses the inputs bit-exactly, round-trips the state
   through write_restart/read_restart, and recomputes these sums.
   Any serializer drift shows up as a golden mismatch; any flipped
   bit shows up as a PartialEq failure in the replay test.
   Regen: wolframscript -script fixtures/solids_restart_bitrot.wls > fixtures/solids_restart_bitrot_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Fixed 4-node state; every entry is an exact dyadic rational. *)
positions = {1/2, -1/4, 7/2, -5/2, 7/8, -17/2, 11/16, -13/4, 31/2, -17/8, 19/2, -41/2};
velocities = {-1/8, 5/4, -13/2, 9/2, -11/16, 27/2, -15/8, 17/4, -37/2, 21/16, -23/2, 51/2};
masses = {1/2, 3/4, 11/2, 7/8};
forces = {3/2, -5/4, 15/2, -9/8, 11/2, -25/2, 15/16, -17/4, 39/2, -21/8, 47/2, -25/2};
simTime = 3/8;
simStep = 41;
simDt = 1/1024;

sumPos = Total[positions];
sumVel = Total[velocities];
sumMas = Total[masses];
sumFrc = Total[forces];
sqPos = Total[positions^2];
sqVel = Total[velocities^2];
sqMas = Total[masses^2];
sqFrc = Total[forces^2];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "restart_bitrot",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "node_count" -> 4,
    "restart_schema" -> 1,
    "positions" -> N[positions],
    "velocities" -> N[velocities],
    "masses" -> N[masses],
    "forces" -> N[forces],
    "time_s" -> N[simTime],
    "step" -> simStep,
    "dt_s" -> N[simDt]
  |>,
  "values" -> <|
    "sum_positions" -> <|
      "values" -> {dec[sumPos]},
      "exact" -> "Total[positions]"
    |>,
    "sum_velocities" -> <|
      "values" -> {dec[sumVel]},
      "exact" -> "Total[velocities]"
    |>,
    "sum_masses" -> <|
      "values" -> {dec[sumMas]},
      "exact" -> "Total[masses]"
    |>,
    "sum_forces" -> <|
      "values" -> {dec[sumFrc]},
      "exact" -> "Total[forces]"
    |>,
    "sumsq_positions" -> <|
      "values" -> {dec[sqPos]},
      "exact" -> "Total[positions^2]"
    |>,
    "sumsq_velocities" -> <|
      "values" -> {dec[sqVel]},
      "exact" -> "Total[velocities^2]"
    |>,
    "sumsq_masses" -> <|
      "values" -> {dec[sqMas]},
      "exact" -> "Total[masses^2]"
    |>,
    "sumsq_forces" -> <|
      "values" -> {dec[sqFrc]},
      "exact" -> "Total[forces^2]"
    |>
  |>,
  "tolerances" -> <|
    "sum_positions_abs" -> 1.0*^-12,
    "sum_positions_rel" -> 1.0*^-12,
    "sum_velocities_abs" -> 1.0*^-12,
    "sum_velocities_rel" -> 1.0*^-12,
    "sum_masses_abs" -> 1.0*^-12,
    "sum_masses_rel" -> 1.0*^-12,
    "sum_forces_abs" -> 1.0*^-12,
    "sum_forces_rel" -> 1.0*^-12,
    "sumsq_positions_abs" -> 1.0*^-12,
    "sumsq_positions_rel" -> 1.0*^-12,
    "sumsq_velocities_abs" -> 1.0*^-12,
    "sumsq_velocities_rel" -> 1.0*^-12,
    "sumsq_masses_abs" -> 1.0*^-12,
    "sumsq_masses_rel" -> 1.0*^-12,
    "sumsq_forces_abs" -> 1.0*^-12,
    "sumsq_forces_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
