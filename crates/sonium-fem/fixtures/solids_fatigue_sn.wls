(* Fatigue S-N golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_fatigue_sn.wls > fixtures/solids_fatigue_sn_golden.json
   Case: Basquin lives, Goodman correction, two-block Miner damage, an
   independent ASTM E1049 four-point rainflow count on [0,1,-1,1,-1,0],
   and a spotweld force-life damage sum.
   Provenance: Basquin, O. H. (1910), "The exponential law of endurance
   tests", Proc. ASTM 10; Miner, M. A. (1945), "Cumulative damage in
   fatigue", J. Appl. Mech. 12; ASTM E1049-85 rainflow counting. *)

Off[General::munfl];

(* Basquin N = (Sa/Sf)^(1/b), Goodman Sa/(1-Sm/Sut). *)
basquin[sa_, sf_, b_] := (sa/sf)^(1/b);
goodman[sa_, sm_, sut_] := sa/(1 - sm/sut);

sfSN = 1000;
bSN = -1/10;
n500 = basquin[500, sfSN, bSN];
n300 = basquin[300, sfSN, bSN];
nGood = basquin[goodman[300, 300, 1200], sfSN, bSN];
minerTwo = 100/basquin[300, sfSN, bSN] + 1000/basquin[150, sfSN, bSN];

If[! (n500 == 1024), Quit[1]];
If[! (nGood == (2/5)^-10), Quit[1]];

(* Independent four-point rainflow (ASTM E1049 Fig. 5 rule). *)
reversals[s_] := Module[{pts, out = {}, dir = 0, slope},
  pts = First /@ Split[s];
  If[Length[pts] < 2, Return[pts]];
  AppendTo[out, pts[[1]]];
  Do[
    slope = Sign[pts[[i + 1]] - pts[[i]]];
    If[slope != 0 && dir != 0 && slope != dir, AppendTo[out, pts[[i]]]];
    If[slope != 0, dir = slope],
    {i, Length[pts] - 1}];
  AppendTo[out, Last[pts]];
  out];
rainflow[s_] := Module[{rev, stack = {}, cyc = {}},
  rev = reversals[s];
  Do[
    AppendTo[stack, p];
    While[Length[stack] >= 4,
      {a, b, c, d} = Take[stack, -4];
      {x, y, z} = {Abs[a - b], Abs[b - c], Abs[c - d]};
      If[y <= x && y <= z,
        AppendTo[cyc, {y, (b + c)/2, 1}];
        stack = Drop[stack, {-3, -2}],
        Break[]]],
    {p, rev}];
  Do[AppendTo[cyc, {Abs[stack[[i]] - stack[[i + 1]]],
    (stack[[i]] + stack[[i + 1]])/2, 1/2}], {i, Length[stack] - 1}];
  cyc];

rf = rainflow[{0, 1, -1, 1, -1, 0}];
full = Select[rf, #[[3]] == 1 &];
half = Select[rf, #[[3]] == 1/2 &];
If[! (Length[full] == 1 && full[[1, 1]] == 2 && full[[1, 2]] == 0), Quit[1]];
If[! (Sort[half[[All, 1]]] == {1, 1, 2}), Quit[1]];

(* Spotweld force-life: F0 = 1000, b = -1/5, 0/500 reversals. *)
fnLife[fa_, f0_, b_] := (fa/f0)^(1/b);
spotHist = Table[If[EvenQ[i], 0, 500], {i, 0, 9}];
spotCyc = rainflow[spotHist];
spotDamage = Total[#3/fnLife[#1/2, 1000, -1/5] & @@@ spotCyc];
spotTotal = Total[spotCyc[[All, 3]]];

If[! (fnLife[250, 1000, -1/5] == 1024), Quit[1]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "fatigue_sn",
    "units" -> "SI"|>,
  "inputs" -> <|
    "basquin_sf" -> 1000.,
    "basquin_b" -> -0.1,
    "goodman_mean" -> 300.,
    "goodman_uts" -> 1200.,
    "fn_f0" -> 1000.,
    "fn_b" -> -0.2|>,
  "values" -> <|
    "basquin_n500" -> <|
      "values" -> {ToString[N[n500, 17]]},
      "exact" -> "(1/2)^-10 = 1024"|>,
    "basquin_n300" -> <|
      "values" -> {ToString[N[n300, 17]]},
      "exact" -> "(3/10)^-10"|>,
    "goodman_n" -> <|
      "values" -> {ToString[N[nGood, 17]]},
      "exact" -> "(2/5)^-10 after Goodman"|>,
    "miner_two_block" -> <|
      "values" -> {ToString[N[minerTwo, 17]]},
      "exact" -> "100/N(300)+1000/N(150)"|>,
    "rainflow_full_range" -> <|
      "values" -> {ToString[N[full[[1, 1]], 17]]},
      "exact" -> "ASTM four-point full cycle"|>,
    "rainflow_full_mean" -> <|
      "values" -> {ToString[N[full[[1, 2]], 17]]},
      "exact" -> "ASTM four-point full cycle"|>,
    "rainflow_full_count" -> <|
      "values" -> {ToString[N[Total[full[[All, 3]]], 17]]},
      "exact" -> "one full cycle"|>,
    "rainflow_half_total" -> <|
      "values" -> {ToString[N[Total[half[[All, 3]]], 17]]},
      "exact" -> "three residue halves"|>,
    "spotweld_life_250" -> <|
      "values" -> {ToString[N[fnLife[250, 1000, -1/5], 17]]},
      "exact" -> "(1/4)^-5 = 1024"|>,
    "spotweld_damage" -> <|
      "values" -> {ToString[N[spotDamage, 17]]},
      "exact" -> "Miner over rainflow of 0/500 history"|>,
    "spotweld_cycle_total" -> <|
      "values" -> {ToString[N[spotTotal, 17]]},
      "exact" -> "counted cycle weights"|>|>,
  "tolerances" -> <|
    "basquin_n500_abs" -> 0.,
    "basquin_n500_rel" -> 10^-12,
    "basquin_n300_abs" -> 0.,
    "basquin_n300_rel" -> 10^-12,
    "goodman_n_abs" -> 0.,
    "goodman_n_rel" -> 10^-12,
    "miner_two_block_abs" -> 0.,
    "miner_two_block_rel" -> 10^-12,
    "rainflow_full_range_abs" -> 0.,
    "rainflow_full_range_rel" -> 10^-12,
    "rainflow_full_mean_abs" -> 10^-15,
    "rainflow_full_mean_rel" -> 0.,
    "rainflow_full_count_abs" -> 0.,
    "rainflow_full_count_rel" -> 10^-12,
    "rainflow_half_total_abs" -> 0.,
    "rainflow_half_total_rel" -> 10^-12,
    "spotweld_life_250_abs" -> 0.,
    "spotweld_life_250_rel" -> 10^-12,
    "spotweld_damage_abs" -> 0.,
    "spotweld_damage_rel" -> 10^-12,
    "spotweld_cycle_total_abs" -> 0.,
    "spotweld_cycle_total_rel" -> 10^-12|>|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
