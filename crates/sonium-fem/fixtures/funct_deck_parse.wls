(* Independent Wolfram reference for /FUNCT + /TABLE deck parsing.
   Derives, at 50-digit working precision, function evaluations on
   parsed curves, a /TABLE/0 bilinear pin, and /TABLE/1 union-grid
   resampling pins. The deck text below is documentation; the Rust
   side replays the same text through the production parser and
   checks these values. All pins below are exact rationals.
   The Rust side consumes funct_deck_parse_golden.json; this script
   is the derivation.
   Regen: wolframscript -script fixtures/funct_deck_parse.wls > fixtures/funct_deck_parse_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(*
/FUNCT/10
curve 1
0.0 1.0
1.0 2.0
/FUNCT/20
curve 2
0.0 1.0
1.0 4.0
/TABLE/1/3
rate table
2
10 1.0e-4
20 1.0
/TABLE/0/9
data table
2 2 2 0 0
0.0 1.0
0.0 10.0
1.0 2.0 3.0 4.0
/END
*)

f10[x_] := 1 + x;
f20[x_] := 1 + 3*x;
yfrac = (1/2 - 1/10^4)/(1 - 1/10^4);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "funct_deck_parse",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "funct_ids" -> {10, 20},
    "table1_id" -> 3,
    "table0_id" -> 9
  |>,
  "values" -> <|
    "funct10_mid" -> <|
      "values" -> {dec[f10[1/2]]},
      "exact" -> "1 + 1/2 = 3/2"
    |>,
    "funct20_mid" -> <|
      "values" -> {dec[f20[1/2]]},
      "exact" -> "1 + 3/2 = 5/2"
    |>,
    "table1_row0" -> <|
      "values" -> {dec[f10[1]]},
      "exact" -> "on-grid row reproduces funct 10: 2"
    |>,
    "table1_row1" -> <|
      "values" -> {dec[f20[1]]},
      "exact" -> "on-grid row reproduces funct 20: 4"
    |>,
    "table1_blend" -> <|
      "values" -> {dec[f10[1/2] + yfrac*(f20[1/2] - f10[1/2])]},
      "exact" -> "3/2 + (4999/9999)*1 = 39995/19998"
    |>,
    "table0_center" -> <|
      "values" -> {dec[(1 + 2 + 3 + 4)/4]},
      "exact" -> "bilinear center = 5/2"
    |>,
    "table0_corner" -> <|
      "values" -> {dec[4]},
      "exact" -> "eval(1, 10) = 4"
    |>
  |>,
  "tolerances" -> <|
    "funct10_mid_abs" -> 1.0*^-12,
    "funct10_mid_rel" -> 1.0*^-12,
    "funct20_mid_abs" -> 1.0*^-12,
    "funct20_mid_rel" -> 1.0*^-12,
    "table1_row0_abs" -> 1.0*^-12,
    "table1_row0_rel" -> 1.0*^-12,
    "table1_row1_abs" -> 1.0*^-12,
    "table1_row1_rel" -> 1.0*^-12,
    "table1_blend_abs" -> 1.0*^-12,
    "table1_blend_rel" -> 1.0*^-12,
    "table0_center_abs" -> 1.0*^-12,
    "table0_center_rel" -> 1.0*^-12,
    "table0_corner_abs" -> 1.0*^-12,
    "table0_corner_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
