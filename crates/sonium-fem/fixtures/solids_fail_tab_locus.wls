(* Independent Wolfram reference for the tabulated failure locus.
   Derives, at 50-digit working precision, triaxiality interpolation,
   rate blending, above-table extrapolation, (959) damage, and the
   rupture verdict for a small exact-rational FAIL_TAB grid. All inputs
   are exact rationals; the Rust side consumes
   solids_fail_tab_locus_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_fail_tab_locus.wls > fixtures/solids_fail_tab_locus_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Grid: triaxiality nodes with base ordinates; slow row = base/2. *)
triax = {-1, -1/3, 0, 1/3, 2/3};
base = {3, 5/6, 3/10, 9/25, 1/20};
slow = base/2;
fast = base;
rSlow = 1/100000;
rFast = 1/2;

(* Piecewise-linear triaxiality interpolation on the fast row. *)
triaxInterp[y_, t_] := Module[{i},
  i = LengthWhile[triax, # < t &];
  If[i == 0, y[[1]],
   If[i >= Length[triax], y[[-1]],
    y[[i]] + (t - triax[[i]])/(triax[[i + 1]] - triax[[i]])*(y[[i + 1]] - y[[i]])]]];

efNode = triaxInterp[fast, 1/3];
efMidTriax = triaxInterp[fast, 1/6];

(* Linear rate blend between the rows at r = 1/4. *)
fracBlend = (1/4 - rSlow)/(rFast - rSlow);
efBlend = triaxInterp[slow, 1/3] + fracBlend*(triaxInterp[fast, 1/3] - triaxInterp[slow, 1/3]);

(* Linear extrapolation above the last row at r = 1. *)
slope = (triaxInterp[fast, 1/3] - triaxInterp[slow, 1/3])/(rFast - rSlow);
efExtrap = triaxInterp[fast, 1/3] + slope*(1 - rFast);

(* Equation (959): dD = (dep/ef) n Dp^(1-1/n), Dp = 2, n = 2. *)
dep = 1/100;
efD = 9/25;
dD959 = (dep/efD)*2*2^(1 - 1/2);

(* Rupture verdict: damage 1.15 against Dcrit 1.1. *)
ruptureFlag = If[115/100 > 11/10, 1, 0];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "fail_tab_locus",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "triax_nodes" -> {-1.0, -0.3333333333333333, 0.0, 0.3333333333333333, 0.6666666666666666},
    "base_ordinates" -> {3.0, 0.8333333333333334, 0.3, 0.36, 0.05},
    "slow_rate" -> 0.00001,
    "fast_rate" -> 0.5,
    "slow_scale" -> 0.5,
    "dp" -> 2.0,
    "exponent" -> 2.0,
    "dcrit" -> 1.1
  |>,
  "values" -> <|
    "ef_node" -> <|
      "values" -> {dec[efNode]},
      "exact" -> "fast(1/3) = 9/25"
    |>,
    "ef_mid_triax" -> <|
      "values" -> {dec[efMidTriax]},
      "exact" -> "fast(1/6) = (0.30+0.36)/2"
    |>,
    "ef_blend_rate" -> <|
      "values" -> {dec[efBlend]},
      "exact" -> "slow(1/3)+((1/4-rSlow)/(rFast-rSlow))(fast(1/3)-slow(1/3))"
    |>,
    "ef_extrapolate" -> <|
      "values" -> {dec[efExtrap]},
      "exact" -> "fast(1/3)+((fast(1/3)-slow(1/3))/(rFast-rSlow))(1-rFast)"
    |>,
    "damage_959" -> <|
      "values" -> {dec[dD959]},
      "exact" -> "(dep/ef) n Dp^(1-1/n), dep=1/100, ef=9/25, Dp=2, n=2"
    |>,
    "rupture_flag" -> <|
      "values" -> {dec[ruptureFlag]},
      "exact" -> "1.15 > 1.1"
    |>
  |>,
  "tolerances" -> <|
    "ef_node_abs" -> 1.0*^-15,
    "ef_node_rel" -> 1.0*^-12,
    "ef_mid_triax_abs" -> 1.0*^-15,
    "ef_mid_triax_rel" -> 1.0*^-12,
    "ef_blend_rate_abs" -> 1.0*^-15,
    "ef_blend_rate_rel" -> 1.0*^-12,
    "ef_extrapolate_abs" -> 1.0*^-15,
    "ef_extrapolate_rel" -> 1.0*^-12,
    "damage_959_abs" -> 1.0*^-15,
    "damage_959_rel" -> 1.0*^-12,
    "rupture_flag_abs" -> 1.0*^-15,
    "rupture_flag_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
