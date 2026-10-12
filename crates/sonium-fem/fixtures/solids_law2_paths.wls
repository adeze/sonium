(* Independent Wolfram reference for LAW2 Johnson-Cook strain paths.
   Integrates, at 300-digit working precision, the von Mises radial-return
   equations with 4340-steel Johnson-Cook hardening along six strain
   paths: uniaxial extension with the lateral strain solved per step for
   zero lateral stress (bisection, 50 passes), pure shear (gamma_xy
   ramp), volumetric (eps I ramp), a symmetric axial cycle
   0 -> +emax -> -emax -> 0 with the same lateral solve, and two rate
   legs (solved uniaxial at nominal 1e-3 and 1e3 1/s). The state is
   carried in deviatoric/hydrostatic split form (textbook formulation);
   the plastic multiplier on each yielding step is solved with the
   Illinois (modified regula falsi) method on the consistency equation
   qTrial - 3G dep - sig(ep + dep) == 0, independently of the bisection
   the Rust return-mapping uses. Flow stress is
   (A + B ep^n)(1 + C ln(rate/ref))(1 - T*^m) with the quiet
   below-reference rate branch and adiabatic heating T += q dep/Cv from
   the committed plastic work; rate legs estimate the rate from the
   step equivalent increment over dt. All inputs are exact rationals
   converted once with N[..., 300]; the loop then runs at 300-digit
   working precision with no machine-precision literals, and N[..., 50]
   is applied once per exported value. Stations record the yielding
   flag and accumulated epbar at fifths of each path. The Rust side
   consumes solids_law2_paths_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/solids_law2_paths.wls > fixtures/solids_law2_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 300;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact card inputs: 4340 steel, E = 200 GPa, nu = 0.3, JC
   A = 792 MPa, B = 510 MPa, n = 0.26, C = 0.014, m = 1.03,
   Tmelt = 1793 K, Tref = 298 K, Cv = 7830*475 J/(m^3 K),
   caps and rupture open at 1e30, isotropic hardening. *)
eX = 200*10^9;
nuX = 3/10;
aX = 792*10^6;
bX = 510*10^6;
nX = 26/100;
cX = 14/1000;
mX = 103/100;
tmX = 1793;
trX = 298;
cvX = 7830*475;
capX = 10^30;
rupX = 10^30;
refX = 1;
emaxX = 2/100;
nLegX = 100;
latX = 50;
rateSlowX = 10^-3;
rateFastX = 10^3;

E0 = N[eX, prec];
nu0 = N[nuX, prec];
A0 = N[aX, prec];
B0 = N[bX, prec];
nH = N[nX, prec];
cR = N[cX, prec];
mT = N[mX, prec];
tmelt = N[tmX, prec];
tref = N[trX, prec];
cv = N[cvX, prec];
cap0 = N[capX, prec];
rup0 = N[rupX, prec];
refR = N[refX, prec];
G0 = E0/(2*(1 + nu0));
K0 = E0/(3*(1 - 2*nu0));
G2 = 2*G0;
G3 = 3*G0;
emax = N[emaxX, prec];
nLeg = nLegX;
latIt = latX;

(* Johnson-Cook rate factor with the quiet quasi-static branch. *)
rateFactor[r_] := If[r <= refR, 1, Max[1 + cR*Log[r/refR], 10^-3]];

(* Thermal factor 1 - T*^m with clamped homologous temperature. *)
thermalFactor[temp_] := Module[{ts},
  ts = Min[Max[(temp - tref)/(tmelt - tref), 0], 1];
  1 - ts^mT
];

(* Unrated hardening curve. *)
flowRaw[epe_] := A0 + B0*epe^nH;

(* Rated, capped flow stress at plastic strain, rate, temperature. *)
flowStress[epe_, rate_, temp_] :=
  Min[flowRaw[epe]*rateFactor[rate]*thermalFactor[temp], cap0*rateFactor[rate]];

(* Pure bisection (170 passes) solve of
   qTrial - 3G dep - flow(ep + dep) == 0 directly in dep. The bracket
   [0, over/3G] always straddles the root for a hardening law.
   Bisection is used instead of Newton because Newton iterates the
   residual, whose -3G dep term feeds dep's own tracked error back
   into every step (measured x30 error growth per committed step);
   bisection only reads signs, so the midpoint chain carries no
   self-amplification and the tracked error stays bounded like the
   closed-form laws. 170 passes resolve the bracket (hi <= 2e-4) to
   width ~1e-55 with sign tests decidable by 200+ orders of margin.
   The residual assert (1e-50 of qTrial) independently validates the
   root and catches any harmful sign-test misfire. *)
solveDep[qT_, ep_, rate_, temp_] := Module[{rf, tf, cap, sy0, over, hi, a, b, c, fc, dep, f},
  rf = rateFactor[rate];
  tf = thermalFactor[temp];
  cap = cap0*rf;
  sy0 = Min[flowRaw[ep]*rf*tf, cap];
  over = qT - sy0;
  hi = over/G3;
  a = 0; b = hi;
  Do[
    c = (a + b)/2;
    fc = qT - G3*c - Min[flowRaw[ep + c]*rf*tf, cap];
    If[fc > 0, a = c, b = c],
    {170}];
  dep = (a + b)/2;
  f = qT - G3*dep - Min[flowRaw[ep + dep]*rf*tf, cap];
  If[Abs[f] > 10^-50*qT, Print["SOLVE FAILURE"]; Abort[]];
  dep
];

(* Von Mises norm from a deviatoric C1 tensor (exact 3, 1/2). *)
vonMisesDev[dev_] := Sqrt[3*((dev[[1]]^2 + dev[[2]]^2 + dev[[3]]^2)/2
  + dev[[4]]^2 + dev[[5]]^2 + dev[[6]]^2)];

(* Equivalent strain increment of a C1 increment (rate estimate). *)
equivIncr[d_] := Module[{m, d0, d1, d2, sh},
  m = (d[[1]] + d[[2]] + d[[3]])/3;
  d0 = d[[1]] - m; d1 = d[[2]] - m; d2 = d[[3]] - m;
  sh = (d[[4]]^2 + d[[5]]^2 + d[[6]]^2)/2;
  Sqrt[(2/3)*(d0^2 + d1^2 + d2^2 + sh)]
];

(* One J2 trial step in split form: {devNew, pNew, dep, yielding}.
   dtOpt = 0 selects the quasi-static rate; else rate = equiv/dt. *)
j2step[dev_, p_, ep_, temp_, d_, dtOpt_] := Module[{vol, pT, devT, qT, rate, sy, dep, ratio},
  vol = d[[1]] + d[[2]] + d[[3]];
  pT = p + K0*vol;
  devT = {dev[[1]] + G2*(d[[1]] - vol/3),
    dev[[2]] + G2*(d[[2]] - vol/3),
    dev[[3]] + G2*(d[[3]] - vol/3),
    dev[[4]] + G2*(1/2)*d[[4]],
    dev[[5]] + G2*(1/2)*d[[5]],
    dev[[6]] + G2*(1/2)*d[[6]]};
  qT = vonMisesDev[devT];
  rate = If[dtOpt === 0, refR, equivIncr[d]/dtOpt];
  sy = flowStress[ep, rate, temp];
  If[qT <= sy,
    {devT, pT, 0, False},
    dep = solveDep[qT, ep, rate, temp];
    If[ep + dep >= rup0, Print["RUPTURE"]; Abort[]];
    ratio = 1 - G3*dep/qT;
    {ratio*devT, pT, dep, True}
  ]
];

(* Full C1 stress from split state (export and probes only). *)
sigOf[dev_, p_] := {dev[[1]] + p, dev[[2]] + p, dev[[3]] + p,
  dev[[4]], dev[[5]], dev[[6]]};

(* Lateral trial stress of one axial step (bisection probe). *)
stepYY[dev_, p_, ep_, temp_, de_, dl_, dtOpt_] := Module[{u},
  u = j2step[dev, p, ep, temp, {de, dl, dl, 0, 0, 0}, dtOpt];
  u[[1, 2]] + u[[2]]
];

(* Lateral strain for zero lateral stress (bisection, latIt passes). *)
solveDL[dev_, p_, ep_, temp_, de_, dtOpt_] := Module[{lo, hi, mid, blo, bhi, w, k},
  If[de >= 0, lo = -de; hi = 0, lo = 0; hi = -de];
  blo = stepYY[dev, p, ep, temp, de, lo, dtOpt];
  bhi = stepYY[dev, p, ep, temp, de, hi, dtOpt];
  For[k = 1, k <= 20, k++,
    If[blo < 0 && bhi > 0, Break[]];
    w = Abs[hi - lo]; lo = lo - w; hi = hi + w;
    blo = stepYY[dev, p, ep, temp, de, lo, dtOpt];
    bhi = stepYY[dev, p, ep, temp, de, hi, dtOpt];
  ];
  If[! (blo < 0 && bhi > 0), Print["BRACKET FAILURE"]; Abort[]];
  Do[mid = (lo + hi)/2;
    If[stepYY[dev, p, ep, temp, de, mid, dtOpt] > 0, hi = mid, lo = mid],
    {latIt}];
  (lo + hi)/2
];

(* Drive a fixed-increment path; stations list 1-based step indices. *)
runFixed[steps_, stations_, dtOpt_] := Module[{dev, p, ep, temp, i, d, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0; temp = tref;
  yl = {}; epst = {};
  Do[d = steps[[i]];
    u = j2step[dev, p, ep, temp, d, dtOpt];
    dev = u[[1]]; p = u[[2]];
    temp = temp + vonMisesDev[u[[1]]]*u[[3]]/cv;
    ep = ep + u[[3]];
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[steps]}];
  {sigOf[dev, p], ep, yl, epst, temp}
];

(* Drive an axial path with the lateral solve. *)
runAxial[des_, stations_, dtOpt_] := Module[{dev, p, ep, temp, i, de, dl, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0; temp = tref;
  yl = {}; epst = {};
  Do[de = des[[i]];
    dl = solveDL[dev, p, ep, temp, de, dtOpt];
    u = j2step[dev, p, ep, temp, {de, dl, dl, 0, 0, 0}, dtOpt];
    dev = u[[1]]; p = u[[2]];
    temp = temp + vonMisesDev[u[[1]]]*u[[3]]/cv;
    ep = ep + u[[3]];
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[des]}];
  {sigOf[dev, p], ep, yl, epst, temp}
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

cycSig = cyc[[1]]; cycEp = cyc[[2]]; cycTemp = cyc[[5]];
pCyc = (cycSig[[1]] + cycSig[[2]] + cycSig[[3]])/3;
devCyc = {cycSig[[1]] - pCyc, cycSig[[2]] - pCyc, cycSig[[3]] - pCyc,
  cycSig[[4]], cycSig[[5]], cycSig[[6]]};
closure = vonMisesDev[devCyc] - flowStress[cycEp, refR, cycTemp];

(* Precision tripwire: every exported state value must keep 55 digits
   of absolute accuracy (closure excluded: it carries the thermal lag
   and legitimately cancels partway). *)
chkList[uni[[1]], "uni_sigma"];
chk[uni[[2]], "uni_epbar"];
chkList[uni[[4]], "uni_epbar_st"];
chk[uni[[5]], "uni_temp"];
chkList[shear[[1]], "shear_sigma"];
chk[shear[[2]], "shear_epbar"];
chkList[shear[[4]], "shear_epbar_st"];
chk[shear[[5]], "shear_temp"];
chkList[vol[[1]], "vol_sigma"];
chk[vol[[2]], "vol_epbar"];
chkList[vol[[4]], "vol_epbar_st"];
chk[vol[[5]], "vol_temp"];
chkList[cyc[[1]], "cycle_sigma"];
chk[cyc[[2]], "cycle_epbar"];
chkList[cyc[[4]], "cycle_epbar_st"];
chk[cyc[[5]], "cycle_temp"];
chkList[slow[[1]], "slow_sigma"];
chk[slow[[2]], "slow_epbar"];
chkList[slow[[4]], "slow_epbar_st"];
chk[slow[[5]], "slow_temp"];
chkList[fast[[1]], "fast_sigma"];
chk[fast[[2]], "fast_epbar"];
chkList[fast[[4]], "fast_epbar_st"];
chk[fast[[5]], "fast_temp"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law2_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_pa" -> N[eX],
    "poisson" -> N[nuX],
    "a_pa" -> N[aX],
    "b_pa" -> N[bX],
    "n" -> N[nX],
    "c" -> N[cX],
    "m" -> N[mX],
    "t_melt_k" -> N[tmX],
    "t_ref_k" -> N[trX],
    "heat_capacity" -> N[cvX],
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
      "exact" -> "JC radial return, bisection dep, lateral solved per step"
    |>,
    "uni_epbar" -> <|
      "values" -> {dec[uni[[2]]]},
      "exact" -> "Sum dep over 100 solved uniaxial steps"
    |>,
    "uni_yielding" -> <|
      "values" -> flag /@ uni[[3]],
      "exact" -> "qTrial > sig(ep) at steps {20,40,60,80,100}"
    |>,
    "uni_epbar_st" -> <|
      "values" -> dec /@ uni[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "uni_temp" -> <|
      "values" -> {dec[uni[[5]]]},
      "exact" -> "Tref + Sum q dep/Cv over the uniaxial path"
    |>,
    "shear_sigma" -> <|
      "values" -> dec /@ shear[[1]],
      "exact" -> "JC radial return under gamma_xy ramp"
    |>,
    "shear_epbar" -> <|
      "values" -> {dec[shear[[2]]]},
      "exact" -> "Sum dep over 100 shear steps"
    |>,
    "shear_yielding" -> <|
      "values" -> flag /@ shear[[3]],
      "exact" -> "qTrial > sig(ep) at steps {20,40,60,80,100}"
    |>,
    "shear_epbar_st" -> <|
      "values" -> dec /@ shear[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "shear_temp" -> <|
      "values" -> {dec[shear[[5]]]},
      "exact" -> "Tref + Sum q dep/Cv over the shear path"
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
    "vol_temp" -> <|
      "values" -> {dec[vol[[5]]]},
      "exact" -> "Tref, no plastic work on the volumetric path"
    |>,
    "cycle_sigma" -> <|
      "values" -> dec /@ cyc[[1]],
      "exact" -> "JC radial return over 0->+emax->-emax->0, solved lateral"
    |>,
    "cycle_epbar" -> <|
      "values" -> {dec[cyc[[2]]]},
      "exact" -> "Sum dep over 400 cycle steps"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[3]],
      "exact" -> "qTrial > sig(ep) at steps {80,160,240,320,400}"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[4]],
      "exact" -> "accumulated epbar at steps {80,160,240,320,400}"
    |>,
    "cycle_temp" -> <|
      "values" -> {dec[cyc[[5]]]},
      "exact" -> "Tref + Sum q dep/Cv over the cycle"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "q - sig(ep, T) at the cycle end incl. one-step thermal lag"
    |>,
    "slow_sigma" -> <|
      "values" -> dec /@ slow[[1]],
      "exact" -> "uniaxial at 1e-3 1/s, quiet branch == quasi-static"
    |>,
    "slow_epbar" -> <|
      "values" -> {dec[slow[[2]]]},
      "exact" -> "Sum dep over the slow rate leg"
    |>,
    "slow_yielding" -> <|
      "values" -> flag /@ slow[[3]],
      "exact" -> "qTrial > sig(ep) at steps {20,40,60,80,100}"
    |>,
    "slow_epbar_st" -> <|
      "values" -> dec /@ slow[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "slow_temp" -> <|
      "values" -> {dec[slow[[5]]]},
      "exact" -> "Tref + Sum q dep/Cv over the slow rate leg"
    |>,
    "fast_sigma" -> <|
      "values" -> dec /@ fast[[1]],
      "exact" -> "uniaxial at 1e3 1/s with the log rate factor"
    |>,
    "fast_epbar" -> <|
      "values" -> {dec[fast[[2]]]},
      "exact" -> "Sum dep over the fast rate leg"
    |>,
    "fast_yielding" -> <|
      "values" -> flag /@ fast[[3]],
      "exact" -> "qTrial > sig(ep) at steps {20,40,60,80,100}"
    |>,
    "fast_epbar_st" -> <|
      "values" -> dec /@ fast[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "fast_temp" -> <|
      "values" -> {dec[fast[[5]]]},
      "exact" -> "Tref + Sum q dep/Cv over the fast rate leg"
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
    "uni_temp_abs" -> 1.0*^-9,
    "uni_temp_rel" -> 1.0*^-12,
    "shear_sigma_abs" -> 1.0*^-6,
    "shear_sigma_rel" -> 1.0*^-12,
    "shear_epbar_abs" -> 1.0*^-12,
    "shear_epbar_rel" -> 1.0*^-12,
    "shear_yielding_abs" -> 0.0,
    "shear_yielding_rel" -> 0.0,
    "shear_epbar_st_abs" -> 1.0*^-12,
    "shear_epbar_st_rel" -> 1.0*^-12,
    "shear_temp_abs" -> 1.0*^-9,
    "shear_temp_rel" -> 1.0*^-12,
    "vol_sigma_abs" -> 1.0*^-6,
    "vol_sigma_rel" -> 1.0*^-12,
    "vol_epbar_abs" -> 0.0,
    "vol_epbar_rel" -> 0.0,
    "vol_yielding_abs" -> 0.0,
    "vol_yielding_rel" -> 0.0,
    "vol_epbar_st_abs" -> 0.0,
    "vol_epbar_st_rel" -> 0.0,
    "vol_temp_abs" -> 0.0,
    "vol_temp_rel" -> 0.0,
    "cycle_sigma_abs" -> 1.0*^-6,
    "cycle_sigma_rel" -> 1.0*^-12,
    "cycle_epbar_abs" -> 1.0*^-12,
    "cycle_epbar_rel" -> 1.0*^-12,
    "cycle_yielding_abs" -> 0.0,
    "cycle_yielding_rel" -> 0.0,
    "cycle_epbar_st_abs" -> 1.0*^-12,
    "cycle_epbar_st_rel" -> 1.0*^-12,
    "cycle_temp_abs" -> 1.0*^-9,
    "cycle_temp_rel" -> 1.0*^-12,
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
    "slow_temp_abs" -> 1.0*^-9,
    "slow_temp_rel" -> 1.0*^-12,
    "fast_sigma_abs" -> 1.0*^-6,
    "fast_sigma_rel" -> 1.0*^-12,
    "fast_epbar_abs" -> 1.0*^-12,
    "fast_epbar_rel" -> 1.0*^-12,
    "fast_yielding_abs" -> 0.0,
    "fast_yielding_rel" -> 0.0,
    "fast_epbar_st_abs" -> 1.0*^-12,
    "fast_epbar_st_rel" -> 1.0*^-12,
    "fast_temp_abs" -> 1.0*^-9,
    "fast_temp_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
