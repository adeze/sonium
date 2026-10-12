(* Independent Wolfram reference for LAW50 honeycomb strain paths.
   Integrates, at 100-digit working precision, the pseudo-plastic
   orthotropic honeycomb equations (diagonal elasticity plus an
   independent per-direction cap) along four strain paths: axis-1
   compression, axis-2 compression, axis-3 compression (each a
   100-step engineering-strain ramp to 30% crush with the lateral
   directions held fixed, which keeps their stress exactly zero
   under the decoupled elasticity), and a symmetric axis-1 cycle
   0 -> +emax -> -emax -> 0. Each direction carries two rated
   yield curves (strain rates 0 and 10 /s) evaluated by clamped
   piecewise-linear lookup with linear rate interpolation; every
   step runs at rate 5 /s exactly (dt = de/5), pinning the
   interpolation midpoint. The abscissa uses the LAW50 default
   negated-component-strain basis (Iflag = -1), so compression
   walks up the tabulated curves while tension clamps at the
   first knot. The plastic increment per step is the capped
   strain remainder, accumulated in the equivalent norm. All
   inputs are exact rationals converted once with N[..., 100];
   the loop then runs at 100-digit working precision with no
   machine-precision literals, and N[..., 50] is applied once
   per exported value. Stations record the yielding flag and
   accumulated epbar at fifths of each path. The Rust side
   consumes solids_law50_paths_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/solids_law50_paths.wls > fixtures/solids_law50_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 100;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact inputs: moduli 100 Pa per direction; per-direction rated
   yield curves (rates 0 and 10 /s) with directional crush
   emphasis; emax = 0.3 crush; 100 steps per leg. *)
modX = {100, 100, 100, 100, 100, 100};
loX = {0, 1}; hiX = {0, 1};
loY = {{1, 3}, {3/2, 4}, {2, 5}, {1, 1}, {1, 1}, {1, 1}};
hiY = {{2, 4}, {5/2, 5}, {3, 6}, {2, 2}, {2, 2}, {2, 2}};
rateLoX = 0; rateHiX = 10;
emaxX = 3/10;
nLegX = 100;

modE = N[modX, prec];
xlo = N[loX, prec]; xhi = N[hiX, prec];
ylo = N[loY, prec]; yhi = N[hiY, prec];
rLo = N[rateLoX, prec]; rHi = N[rateHiX, prec];
emax = N[emaxX, prec];
nLeg = nLegX;
de = emax/nLeg;
dt = de/5;

(* Clamped piecewise-linear lookup on a two-knot grid. *)
functEval[x_, xs_, ys_] := If[x <= xs[[1]], ys[[1]],
  If[x >= xs[[2]], ys[[2]],
    ys[[1]] + (x - xs[[1]])/(xs[[2]] - xs[[1]])*(ys[[2]] - ys[[1]])]];

(* Rated yield of one direction at abscissa and rate. *)
ratedYield[dir_, x_, r_] := Module[{flo, fhi, frac},
  flo = functEval[x, xlo, ylo[[dir]]];
  fhi = functEval[x, xhi, yhi[[dir]]];
  If[r <= rLo, flo,
    If[r >= rHi, fhi,
      frac = (r - rLo)/(rHi - rLo);
      flo + frac*(fhi - flo)]]
];

(* Von Mises norm of a full C1 stress (closure only). *)
vonMisesFull[sig_] := Module[{p, s0, s1, s2},
  p = (sig[[1]] + sig[[2]] + sig[[3]])/3;
  s0 = sig[[1]] - p; s1 = sig[[2]] - p; s2 = sig[[3]] - p;
  Sqrt[(3/2)*(s0^2 + s1^2 + s2^2) + 3*(sig[[4]]^2 + sig[[5]]^2 + sig[[6]]^2)]
];

(* Drive a fixed-increment path; stations list 1-based step indices.
   Rates are fresh raw component magnitudes |d|/dt per step (the
   cutoff is infinite, so no filtering applies). *)
runFixed[steps_, stations_] := Module[{sig, str, ep, i, d, r, t, x,
  cap, trial, capped, plas, dep, yld, yl, epst},
  sig = {0, 0, 0, 0, 0, 0}; str = {0, 0, 0, 0, 0, 0}; ep = 0;
  yl = {}; epst = {};
  Do[d = steps[[i]];
    r = Abs[d]/dt;
    t = str + d;
    x = -t;
    cap = Table[Abs[ratedYield[dir, x[[dir]], r[[dir]]]], {dir, 1, 6}];
    trial = Table[sig[[dir]] + modE[[dir]]*d[[dir]], {dir, 1, 6}];
    capped = Table[If[Abs[trial[[dir]]] <= cap[[dir]], trial[[dir]],
      Sign[trial[[dir]]]*cap[[dir]]], {dir, 1, 6}];
    yld = Or @@ Table[Abs[trial[[dir]]] > cap[[dir]], {dir, 1, 6}];
    plas = Table[d[[dir]] - (capped[[dir]] - sig[[dir]])/modE[[dir]], {dir, 1, 6}];
    dep = Sqrt[plas.plas];
    sig = capped;
    ep = ep + dep;
    str = str + d;
    If[MemberQ[stations, i], AppendTo[yl, yld]; AppendTo[epst, ep]],
    {i, 1, Length[steps]}];
  {sig, ep, yl, epst}
];

st100 = {20, 40, 60, 80, 100};
stCyc = {80, 160, 240, 320, 400};

ax1 = runFixed[Table[{-de, 0, 0, 0, 0, 0}, {nLeg}], st100];
ax2 = runFixed[Table[{0, -de, 0, 0, 0, 0}, {nLeg}], st100];
ax3 = runFixed[Table[{0, 0, -de, 0, 0, 0}, {nLeg}], st100];
cycDes = Join[Table[de, {nLeg}], Table[-de, {2*nLeg}], Table[de, {nLeg}]];
cyc = runFixed[Table[{cycDes[[i]], 0, 0, 0, 0, 0}, {i, 1, Length[cycDes]}], stCyc];

(* Loop-closure distance: von Mises of the cycle end stress. *)
closure = vonMisesFull[cyc[[1]]];

(* Precision tripwire on every exported state value. *)
chkList[ax1[[1]], "ax1_sigma"];
chk[ax1[[2]], "ax1_epbar"];
chkList[ax1[[4]], "ax1_epbar_st"];
chkList[ax2[[1]], "ax2_sigma"];
chk[ax2[[2]], "ax2_epbar"];
chkList[ax2[[4]], "ax2_epbar_st"];
chkList[ax3[[1]], "ax3_sigma"];
chk[ax3[[2]], "ax3_epbar"];
chkList[ax3[[4]], "ax3_epbar_st"];
chkList[cyc[[1]], "cycle_sigma"];
chk[cyc[[2]], "cycle_epbar"];
chkList[cyc[[4]], "cycle_epbar_st"];
chk[closure, "cycle_closure"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law50_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "moduli_pa" -> N[modX],
    "yield_lo" -> N[loY],
    "yield_hi" -> N[hiY],
    "rate_lo" -> N[rateLoX],
    "rate_hi" -> N[rateHiX],
    "step_rate" -> N[de/dt],
    "emax" -> N[emaxX],
    "steps_per_leg" -> nLegX,
    "stations_100" -> st100,
    "stations_cycle" -> stCyc,
    "dt_s" -> N[dt]
  |>,
  "values" -> <|
    "ax1_sigma" -> <|
      "values" -> dec /@ ax1[[1]],
      "exact" -> "axis-1 crush to 0.3, rated caps at 5 /s"
    |>,
    "ax1_epbar" -> <|
      "values" -> {dec[ax1[[2]]]},
      "exact" -> "Sum capped-remainder norms over 100 crush steps"
    |>,
    "ax1_yielding" -> <|
      "values" -> flag /@ ax1[[3]],
      "exact" -> "any direction capped at steps {20,40,60,80,100}"
    |>,
    "ax1_epbar_st" -> <|
      "values" -> dec /@ ax1[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "ax2_sigma" -> <|
      "values" -> dec /@ ax2[[1]],
      "exact" -> "axis-2 crush to 0.3, rated caps at 5 /s"
    |>,
    "ax2_epbar" -> <|
      "values" -> {dec[ax2[[2]]]},
      "exact" -> "Sum capped-remainder norms over 100 crush steps"
    |>,
    "ax2_yielding" -> <|
      "values" -> flag /@ ax2[[3]],
      "exact" -> "any direction capped at steps {20,40,60,80,100}"
    |>,
    "ax2_epbar_st" -> <|
      "values" -> dec /@ ax2[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "ax3_sigma" -> <|
      "values" -> dec /@ ax3[[1]],
      "exact" -> "axis-3 crush to 0.3, rated caps at 5 /s"
    |>,
    "ax3_epbar" -> <|
      "values" -> {dec[ax3[[2]]]},
      "exact" -> "Sum capped-remainder norms over 100 crush steps"
    |>,
    "ax3_yielding" -> <|
      "values" -> flag /@ ax3[[3]],
      "exact" -> "any direction capped at steps {20,40,60,80,100}"
    |>,
    "ax3_epbar_st" -> <|
      "values" -> dec /@ ax3[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "cycle_sigma" -> <|
      "values" -> dec /@ cyc[[1]],
      "exact" -> "axis-1 0->+emax->-emax->0 crush cycle"
    |>,
    "cycle_epbar" -> <|
      "values" -> {dec[cyc[[2]]]},
      "exact" -> "Sum capped-remainder norms over 400 cycle steps"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[3]],
      "exact" -> "any direction capped at steps {80,160,240,320,400}"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[4]],
      "exact" -> "accumulated epbar at steps {80,160,240,320,400}"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "von Mises of the cycle end stress"
    |>
  |>,
  "tolerances" -> <|
    "ax1_sigma_abs" -> 1.0*^-9,
    "ax1_sigma_rel" -> 1.0*^-12,
    "ax1_epbar_abs" -> 1.0*^-12,
    "ax1_epbar_rel" -> 1.0*^-12,
    "ax1_yielding_abs" -> 0.0,
    "ax1_yielding_rel" -> 0.0,
    "ax1_epbar_st_abs" -> 1.0*^-12,
    "ax1_epbar_st_rel" -> 1.0*^-12,
    "ax2_sigma_abs" -> 1.0*^-9,
    "ax2_sigma_rel" -> 1.0*^-12,
    "ax2_epbar_abs" -> 1.0*^-12,
    "ax2_epbar_rel" -> 1.0*^-12,
    "ax2_yielding_abs" -> 0.0,
    "ax2_yielding_rel" -> 0.0,
    "ax2_epbar_st_abs" -> 1.0*^-12,
    "ax2_epbar_st_rel" -> 1.0*^-12,
    "ax3_sigma_abs" -> 1.0*^-9,
    "ax3_sigma_rel" -> 1.0*^-12,
    "ax3_epbar_abs" -> 1.0*^-12,
    "ax3_epbar_rel" -> 1.0*^-12,
    "ax3_yielding_abs" -> 0.0,
    "ax3_yielding_rel" -> 0.0,
    "ax3_epbar_st_abs" -> 1.0*^-12,
    "ax3_epbar_st_rel" -> 1.0*^-12,
    "cycle_sigma_abs" -> 1.0*^-9,
    "cycle_sigma_rel" -> 1.0*^-12,
    "cycle_epbar_abs" -> 1.0*^-12,
    "cycle_epbar_rel" -> 1.0*^-12,
    "cycle_yielding_abs" -> 0.0,
    "cycle_yielding_rel" -> 0.0,
    "cycle_epbar_st_abs" -> 1.0*^-12,
    "cycle_epbar_st_rel" -> 1.0*^-12,
    "cycle_closure_abs" -> 1.0*^-9,
    "cycle_closure_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
