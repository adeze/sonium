(* Independent Wolfram reference for LAW44 Cowper-Symonds strain paths.
   Integrates, at 400-digit working precision, the von Mises
   radial-return equations with combined iso-kinematic hardening
   (Chard = 1, fully kinematic) and the Cowper-Symonds rate factor
   along six strain paths: uniaxial extension with the lateral strain
   solved per step for zero lateral stress (bisection, 50 passes),
   pure shear (gamma_xy ramp), volumetric (eps I ramp), a symmetric
   axial cycle 0 -> +emax -> -emax -> 0 with the same lateral solve,
   and two rate legs (solved uniaxial at nominal 1e-3 and 1e2 1/s).
   The state is carried in deviatoric/hydrostatic split form
   (textbook formulation) plus the Prager back-stress; the plastic
   multiplier on each yielding step solves the consistency equation
   qXi - (3G + Ck(ep+dep)) dep - sig0 == 0 with the end-point
   hardening slope Ck, via 170-pass bisection (independent of the
   80-pass bisection the Rust return-mapping uses). The back-stress
   commits with the exact rated surface difference
   (sig(epNew)-sig(epOld)) * ratio along the returned-stress
   direction, matching the production commit; the return's internal
   slope-based back-stress is discarded, as in production. Flow
   stress is (a + b ep^n)(1 + (rate/c)^(1/p)) capped at the rated
   maximum; quasi-static legs run at rate 0 (factor exactly 1) and
   rate legs use the tensor-Frobenius total rate |d|/dt per step.
   Rupture and tensile limits stay open (verified via a Gershgorin
   bound on the accumulated total strain). All inputs are exact
   rationals converted once with N[..., 400]; the loop then runs at
   400-digit working precision with no machine-precision literals
   (covers the ~0.7-digit-per-step significance-tracking decay of
   the kinematic-hardening feedback over 400 cycle steps),
   and N[..., 50] is applied once per exported value. Stations
   record the yielding flag and accumulated epbar at fifths of each
   path. The Rust side consumes solids_law44_paths_golden.json;
   this script is the derivation.
   Regen: wolframscript -script fixtures/solids_law44_paths.wls > fixtures/solids_law44_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 400;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact card inputs: /MAT/LAW44 card example, E = 20.5 GPa,
   nu = 0.3, a = 50 MPa, b = 100 MPa, n = 0.5, Chard = 1,
   SIGMA_max0 = 90 MPa, c = 100 /s, p = 5, ICC scaled, VP total. *)
eX = 205/10*10^9;
nuX = 3/10;
aX = 50*10^6;
bX = 100*10^6;
nX = 1/2;
chardX = 1;
smaxX = 90*10^6;
cX = 100;
pX = 5;
t1X = 10^20;
emaxX = 2/100;
nLegX = 100;
latX = 50;
rateSlowX = 10^-3;
rateFastX = 10^2;

E0 = N[eX, prec];
nu0 = N[nuX, prec];
A0 = N[aX, prec];
B0 = N[bX, prec];
nH = N[nX, prec];
chard = N[chardX, prec];
smax0 = N[smaxX, prec];
cCS = N[cX, prec];
pCS = N[pX, prec];
G0 = E0/(2*(1 + nu0));
K0 = E0/(3*(1 - 2*nu0));
G2 = 2*G0;
G3 = 3*G0;
emax = N[emaxX, prec];
nLeg = nLegX;
latIt = latX;

(* Cowper-Symonds factor 1 + (rate/c)^(1/p), quiet at rate <= 0. *)
rateFactor[r_] := If[r <= 0, 1, 1 + (r/cCS)^(1/pCS)];

(* Rated cap under ICC = scaled. *)
ratedCap[r_] := smax0*rateFactor[r];

(* Capped analytic flow stress (Eq. 238 branch, no yield function). *)
flowStress[ep_, r_] :=
  Min[(A0 + B0*Max[ep, 0]^nH)*rateFactor[r], ratedCap[r]];

(* Hardening slope with the documented 1e-12 ep floor, zero past cap. *)
hardSlope[ep_, r_] := Module[{raw, cap},
  raw = flowStress[ep, r];
  cap = ratedCap[r];
  If[raw >= cap, 0, B0*nH*Max[ep, N[10^-12, prec]]^(nH - 1)*rateFactor[r]]
];

(* Von Mises norm from a deviatoric C1 tensor (exact 3, 1/2). *)
vonMisesDev[dev_] := Sqrt[3*((dev[[1]]^2 + dev[[2]]^2 + dev[[3]]^2)/2
  + dev[[4]]^2 + dev[[5]]^2 + dev[[6]]^2)];

(* Full C1 stress from split state (export and probes only). *)
sigOf[dev_, p_] := {dev[[1]] + p, dev[[2]] + p, dev[[3]] + p,
  dev[[4]], dev[[5]], dev[[6]]};

(* Von Mises norm of a full C1 stress. *)
vonMisesFull[sig_] := Module[{p},
  p = (sig[[1]] + sig[[2]] + sig[[3]])/3;
  vonMisesDev[{sig[[1]] - p, sig[[2]] - p, sig[[3]] - p,
    sig[[4]], sig[[5]], sig[[6]]}]
];

(* Tensor-Frobenius total rate of a C1 increment (VP = total). *)
totalRate[d_, dt_] := Sqrt[d[[1]]^2 + d[[2]]^2 + d[[3]]^2
  + (d[[4]]^2 + d[[5]]^2 + d[[6]]^2)/2]/dt;

(* 170-pass bisection solve of
   qXi - (3G + Ck(ep+dep)) dep - sig0 == 0 in dep, with the
   production bracket and doubling. The residual assert
   independently validates the root. *)
solveDep[qXi_, ep_, rate_, sig0_] := Module[{hi, a, b, c, fc, dep, f, k},
  hi = (qXi - sig0)/G3;
  For[k = 1, k <= 60, k++,
    If[qXi - (G3 + Max[hardSlope[ep + hi, rate], 0])*hi - sig0 <= 0, Break[]];
    hi = 2*hi;
  ];
  a = 0; b = hi;
  Do[
    c = (a + b)/2;
    fc = qXi - (G3 + Max[hardSlope[ep + c, rate], 0])*c - sig0;
    If[fc > 0, a = c, b = c],
    {170}];
  dep = (a + b)/2;
  f = qXi - (G3 + Max[hardSlope[ep + dep, rate], 0])*dep - sig0;
  If[Abs[f] > 10^-50*qXi, Print["SOLVE FAILURE"]; Abort[]];
  dep
];

(* One combined-hardening trial step in split form:
   {devNew, pNew, dep, yielding}. Back-stress enters through the
   relative trial stress; the recoil runs along it. *)
j2step[dev_, p_, back_, ep_, d_, rate_] := Module[{vol, pT, devT, xi, qXi, sig0, dep},
  vol = d[[1]] + d[[2]] + d[[3]];
  pT = p + K0*vol;
  devT = {dev[[1]] + G2*(d[[1]] - vol/3),
    dev[[2]] + G2*(d[[2]] - vol/3),
    dev[[3]] + G2*(d[[3]] - vol/3),
    dev[[4]] + G2*(1/2)*d[[4]],
    dev[[5]] + G2*(1/2)*d[[5]],
    dev[[6]] + G2*(1/2)*d[[6]]};
  xi = devT - back;
  qXi = vonMisesDev[xi];
  sig0 = flowStress[0, rate];
  If[qXi <= sig0,
    {devT, pT, 0, False},
    dep = solveDep[qXi, ep, rate, sig0];
    {devT - (G3*dep/qXi)*xi, pT, dep, True}
  ]
];

(* Production commit of the back-stress: exact rated surface
   difference along the returned-stress direction. *)
commitBack[devN_, pN_, back_, epOld_, dep_] := Module[{sig, q, p, devS,
  quasiNew, quasiOld, ratio, capEst, step, dir},
  sig = sigOf[devN, pN];
  q = vonMisesFull[sig];
  p = (sig[[1]] + sig[[2]] + sig[[3]])/3;
  devS = {sig[[1]] - p, sig[[2]] - p, sig[[3]] - p,
    sig[[4]], sig[[5]], sig[[6]]};
  quasiNew = flowStress[epOld + dep, 0];
  quasiOld = flowStress[Max[epOld, 0], 0];
  ratio = Min[Max[q/quasiNew, 10^-3], 10^3];
  capEst = smax0*ratio;
  step = If[quasiNew <= 0,
    Max[hardSlope[epOld + dep, 0]*dep, 0],
    Max[Min[quasiNew*ratio, capEst] - Min[Max[quasiOld, 0]*ratio, capEst], 0]];
  If[step > 0 && q > 10^-12, back + step*devS/q, back]
];

(* Gershgorin bound on the largest principal total strain
   (tensor shear is half the engineering shear). *)
gershBound[t_] := Max[
  Abs[t[[1]]] + Abs[t[[4]]]/2 + Abs[t[[6]]]/2,
  Abs[t[[2]]] + Abs[t[[4]]]/2 + Abs[t[[5]]]/2,
  Abs[t[[3]]] + Abs[t[[6]]]/2 + Abs[t[[5]]]/2];

(* Lateral trial stress of one axial step (bisection probe). *)
stepYY[dev_, p_, back_, ep_, de_, dl_, rate_] := Module[{u},
  u = j2step[dev, p, back, ep, {de, dl, dl, 0, 0, 0}, rate];
  u[[1, 2]] + u[[2]]
];

(* Lateral strain for zero lateral stress (bisection, latIt passes).
   The rate-leg probe rebuilds the total rate per trial dl. *)
solveDL[dev_, p_, back_, ep_, de_, dtOpt_] := Module[{lo, hi, mid, blo, bhi, w, k, pr},
  pr[dl_] := Module[{r},
    r = If[dtOpt === 0, 0, totalRate[{de, dl, dl, 0, 0, 0}, dtOpt]];
    stepYY[dev, p, back, ep, de, dl, r]];
  If[de >= 0, lo = -de; hi = 0, lo = 0; hi = -de];
  blo = pr[lo];
  bhi = pr[hi];
  For[k = 1, k <= 20, k++,
    If[blo < 0 && bhi > 0, Break[]];
    w = Abs[hi - lo]; lo = lo - w; hi = hi + w;
    blo = pr[lo];
    bhi = pr[hi];
  ];
  If[! (blo < 0 && bhi > 0), Print["BRACKET FAILURE"]; Abort[]];
  Do[mid = (lo + hi)/2;
    If[pr[mid] > 0, hi = mid, lo = mid],
    {latIt}];
  (lo + hi)/2
];

(* Drive a fixed-increment path; stations list 1-based step indices. *)
runFixed[steps_, stations_, dtOpt_] := Module[{dev, p, back, ep, tot, i, d, r, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; back = {0, 0, 0, 0, 0, 0};
  ep = 0; tot = {0, 0, 0, 0, 0, 0};
  yl = {}; epst = {};
  Do[d = steps[[i]];
    r = If[dtOpt === 0, 0, totalRate[d, dtOpt]];
    u = j2step[dev, p, back, ep, d, r];
    dev = u[[1]]; p = u[[2]];
    back = If[u[[4]], commitBack[dev, p, back, ep, u[[3]]], back];
    ep = ep + u[[3]];
    tot = tot + d;
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[steps]}];
  {sigOf[dev, p], ep, yl, epst, back, tot}
];

(* Drive an axial path with the lateral solve. *)
runAxial[des_, stations_, dtOpt_] := Module[{dev, p, back, ep, tot, i, de, dl, d, r, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; back = {0, 0, 0, 0, 0, 0};
  ep = 0; tot = {0, 0, 0, 0, 0, 0};
  yl = {}; epst = {};
  Do[de = des[[i]];
    dl = solveDL[dev, p, back, ep, de, dtOpt];
    d = {de, dl, dl, 0, 0, 0};
    r = If[dtOpt === 0, 0, totalRate[d, dtOpt]];
    u = j2step[dev, p, back, ep, d, r];
    dev = u[[1]]; p = u[[2]];
    back = If[u[[4]], commitBack[dev, p, back, ep, u[[3]]], back];
    ep = ep + u[[3]];
    tot = tot + d;
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[des]}];
  {sigOf[dev, p], ep, yl, epst, back, tot}
];

de = emax/nLeg;
dg = emax/nLeg;
dv = emax/nLeg;
st100 = {20, 40, 60, 80, 100};
stCyc = {80, 160, 240, 320, 400};
dtSlow = de/N[rateSlowX, prec];
dtFast = de/N[rateFastX, prec];

uni = runAxial[Table[de, {nLeg}], st100, 0];
shear = runFixed[Table[{0, 0, 0, dg, 0, 0}, {nLeg}], st100, 0];
vol = runFixed[Table[{dv, dv, dv, 0, 0, 0}, {nLeg}], st100, 0];
cycDes = Join[Table[de, {nLeg}], Table[-de, {2*nLeg}], Table[de, {nLeg}]];
cyc = runAxial[cycDes, stCyc, 0];
slow = runAxial[Table[de, {nLeg}], st100, dtSlow];
fast = runAxial[Table[de, {nLeg}], st100, dtFast];

(* Tensile-scale guard: totals stay 19 orders below the limit. *)
Do[If[gershBound[leg[[6]]] > 1, Print["TENSILE SCALE TRIPPED"]; Abort[]],
  {leg, {uni, shear, vol, cyc, slow, fast}}];

(* Consistency residual of the relative stress at the cycle end. *)
cycSig = cyc[[1]]; cycBack = cyc[[5]]; cycEp = cyc[[2]];
closure = vonMisesFull[cycSig - cycBack] - flowStress[0, 0];

(* Precision tripwire on every exported state value. *)
chkList[uni[[1]], "uni_sigma"];
chk[uni[[2]], "uni_epbar"];
chkList[uni[[4]], "uni_epbar_st"];
chkList[uni[[5]], "uni_back"];
chkList[shear[[1]], "shear_sigma"];
chk[shear[[2]], "shear_epbar"];
chkList[shear[[4]], "shear_epbar_st"];
chkList[vol[[1]], "vol_sigma"];
chk[vol[[2]], "vol_epbar"];
chkList[vol[[4]], "vol_epbar_st"];
chkList[cyc[[1]], "cycle_sigma"];
chk[cyc[[2]], "cycle_epbar"];
chkList[cyc[[4]], "cycle_epbar_st"];
chkList[slow[[1]], "slow_sigma"];
chk[slow[[2]], "slow_epbar"];
chkList[slow[[4]], "slow_epbar_st"];
chkList[fast[[1]], "fast_sigma"];
chk[fast[[2]], "fast_epbar"];
chkList[fast[[4]], "fast_epbar_st"];
chk[closure, "cycle_closure"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law44_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_pa" -> N[eX],
    "poisson" -> N[nuX],
    "a_pa" -> N[aX],
    "b_pa" -> N[bX],
    "n" -> N[nX],
    "chard" -> N[chardX],
    "max_stress_pa" -> N[smaxX],
    "c_per_s" -> N[cX],
    "p" -> N[pX],
    "emax" -> N[emaxX],
    "steps_per_leg" -> nLegX,
    "lateral_iterations" -> latX,
    "stations_100" -> st100,
    "stations_cycle" -> stCyc,
    "rate_slow" -> N[rateSlowX],
    "rate_fast" -> N[rateFastX],
    "dt_slow_s" -> N[dtSlow],
    "dt_fast_s" -> N[dtFast]
  |>,
  "values" -> <|
    "uni_sigma" -> <|
      "values" -> dec /@ uni[[1]],
      "exact" -> "CS radial return, Chard = 1, lateral solved per step"
    |>,
    "uni_epbar" -> <|
      "values" -> {dec[uni[[2]]]},
      "exact" -> "Sum dep over 100 solved uniaxial steps"
    |>,
    "uni_yielding" -> <|
      "values" -> flag /@ uni[[3]],
      "exact" -> "qXi > sig0 at steps {20,40,60,80,100}"
    |>,
    "uni_epbar_st" -> <|
      "values" -> dec /@ uni[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "uni_back" -> <|
      "values" -> dec /@ uni[[5]],
      "exact" -> "Prager back-stress, exact surface-difference commits"
    |>,
    "shear_sigma" -> <|
      "values" -> dec /@ shear[[1]],
      "exact" -> "CS radial return under gamma_xy ramp"
    |>,
    "shear_epbar" -> <|
      "values" -> {dec[shear[[2]]]},
      "exact" -> "Sum dep over 100 shear steps"
    |>,
    "shear_yielding" -> <|
      "values" -> flag /@ shear[[3]],
      "exact" -> "qXi > sig0 at steps {20,40,60,80,100}"
    |>,
    "shear_epbar_st" -> <|
      "values" -> dec /@ shear[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "vol_sigma" -> <|
      "values" -> dec /@ vol[[1]],
      "exact" -> "hydrostatic elastic, p = K 3 emax"
    |>,
    "vol_epbar" -> <|
      "values" -> {dec[vol[[2]]]},
      "exact" -> "zero, J2 never yields hydrostatically"
    |>,
    "vol_yielding" -> <|
      "values" -> flag /@ vol[[3]],
      "exact" -> "all False on the volumetric path"
    |>,
    "vol_epbar_st" -> <|
      "values" -> dec /@ vol[[4]],
      "exact" -> "all zero on the volumetric path"
    |>,
    "cycle_sigma" -> <|
      "values" -> dec /@ cyc[[1]],
      "exact" -> "CS radial return over 0->+emax->-emax->0, solved lateral"
    |>,
    "cycle_epbar" -> <|
      "values" -> {dec[cyc[[2]]]},
      "exact" -> "Sum dep over 400 cycle steps"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[3]],
      "exact" -> "qXi > sig0 at steps {80,160,240,320,400}"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[4]],
      "exact" -> "accumulated epbar at steps {80,160,240,320,400}"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "q(sig - back) - sig0_quasi at the cycle end"
    |>,
    "slow_sigma" -> <|
      "values" -> dec /@ slow[[1]],
      "exact" -> "uniaxial at 1e-3 1/s, CS factor ~1.1"
    |>,
    "slow_epbar" -> <|
      "values" -> {dec[slow[[2]]]},
      "exact" -> "Sum dep over the slow rate leg"
    |>,
    "slow_yielding" -> <|
      "values" -> flag /@ slow[[3]],
      "exact" -> "qXi > sig0 at steps {20,40,60,80,100}"
    |>,
    "slow_epbar_st" -> <|
      "values" -> dec /@ slow[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "fast_sigma" -> <|
      "values" -> dec /@ fast[[1]],
      "exact" -> "uniaxial at 1e2 1/s, CS factor ~2.04"
    |>,
    "fast_epbar" -> <|
      "values" -> {dec[fast[[2]]]},
      "exact" -> "Sum dep over the fast rate leg"
    |>,
    "fast_yielding" -> <|
      "values" -> flag /@ fast[[3]],
      "exact" -> "qXi > sig0 at steps {20,40,60,80,100}"
    |>,
    "fast_epbar_st" -> <|
      "values" -> dec /@ fast[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>
  |>,
  "tolerances" -> <|
    "uni_sigma_abs" -> 1.0*^-6,
    "uni_sigma_rel" -> 1.0*^-12,
    "uni_epbar_abs" -> 1.0*^-12,
    "uni_epbar_rel" -> 1.0*^-12,
    "uni_yielding_abs" -> 0.0,
    "uni_yielding_rel" -> 0.0,
    "uni_epbar_st_abs" -> 1.0*^-12,
    "uni_epbar_st_rel" -> 1.0*^-12,
    "uni_back_abs" -> 1.0*^-6,
    "uni_back_rel" -> 1.0*^-12,
    "shear_sigma_abs" -> 1.0*^-6,
    "shear_sigma_rel" -> 1.0*^-12,
    "shear_epbar_abs" -> 1.0*^-12,
    "shear_epbar_rel" -> 1.0*^-12,
    "shear_yielding_abs" -> 0.0,
    "shear_yielding_rel" -> 0.0,
    "shear_epbar_st_abs" -> 1.0*^-12,
    "shear_epbar_st_rel" -> 1.0*^-12,
    "vol_sigma_abs" -> 1.0*^-6,
    "vol_sigma_rel" -> 1.0*^-12,
    "vol_epbar_abs" -> 0.0,
    "vol_epbar_rel" -> 0.0,
    "vol_yielding_abs" -> 0.0,
    "vol_yielding_rel" -> 0.0,
    "vol_epbar_st_abs" -> 0.0,
    "vol_epbar_st_rel" -> 0.0,
    "cycle_sigma_abs" -> 1.0*^-6,
    "cycle_sigma_rel" -> 1.0*^-12,
    "cycle_epbar_abs" -> 1.0*^-12,
    "cycle_epbar_rel" -> 1.0*^-12,
    "cycle_yielding_abs" -> 0.0,
    "cycle_yielding_rel" -> 0.0,
    "cycle_epbar_st_abs" -> 1.0*^-12,
    "cycle_epbar_st_rel" -> 1.0*^-12,
    "cycle_closure_abs" -> 1.0*^-6,
    "cycle_closure_rel" -> 1.0*^-12,
    "slow_sigma_abs" -> 1.0*^-6,
    "slow_sigma_rel" -> 1.0*^-12,
    "slow_epbar_abs" -> 1.0*^-12,
    "slow_epbar_rel" -> 1.0*^-12,
    "slow_yielding_abs" -> 0.0,
    "slow_yielding_rel" -> 0.0,
    "slow_epbar_st_abs" -> 1.0*^-12,
    "slow_epbar_st_rel" -> 1.0*^-12,
    "fast_sigma_abs" -> 1.0*^-6,
    "fast_sigma_rel" -> 1.0*^-12,
    "fast_epbar_abs" -> 1.0*^-12,
    "fast_epbar_rel" -> 1.0*^-12,
    "fast_yielding_abs" -> 0.0,
    "fast_yielding_rel" -> 0.0,
    "fast_epbar_st_abs" -> 1.0*^-12,
    "fast_epbar_st_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
