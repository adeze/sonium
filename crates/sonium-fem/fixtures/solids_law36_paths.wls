(* Independent Wolfram reference for LAW36 tabulated-metal strain paths.
   Integrates, at 50-digit working precision, the von Mises radial-return
   equations with the card-aluminum piecewise-linear hardening curve along
   four strain paths: uniaxial extension with the lateral strain solved per
   step for zero lateral stress (bisection, 50 passes), pure shear
   (gamma_xy ramp), volumetric (eps I ramp), and a symmetric axial cycle
   0 -> +emax -> -emax -> 0 with the same lateral solve. The state is
   carried in deviatoric/hydrostatic split form (textbook formulation);
   the plastic multiplier on each yielding step is solved in closed form
   per curve segment (dep = (qTrial - a - b ep)/(3G + b), walking forward
   while ep + dep crosses a knot, end-slope extrapolation past the last
   knot), which is algebraically identical to the consistency equation
   the Rust return-mapping bisects. All inputs are exact rationals
   converted once with N[..., 200]; the loop then runs at 200-digit
   working precision with no machine-precision literals (covers the
   significance-tracking decay of 400 implicit steps), and N[..., 50]
   is applied once per exported value. Stations record the
   yielding flag and accumulated epbar at fifths of each path. The
   failure station rebuilds total strain (elastic plus accumulated
   plastic tensor) and applies the production principal-strain
   convention. The Rust side consumes
   solids_law36_paths_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_law36_paths.wls > fixtures/solids_law36_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 200;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact card inputs: E = 60.4 GPa, nu = 0.33, single tabulated curve
   90..170 MPa over 7.5% plastic strain, all limits open. *)
eX = 604/10*10^9;
nuX = 33/100;
knotX = {0, 1/4000, 1/1000, 9/1000, 17/1000, 21/1000, 36/1000, 45/1000,
  55/1000, 72/1000, 75/1000};
knotY = {90, 100, 104, 121, 136, 143, 156, 162, 165, 170, 170}*10^6;
rupX = 10^20; t1X = 10^20; t2X = 2*10^20; delX = 3*10^20;
emaxX = 5/100;
nX = 100;
latX = 50;

E0 = N[eX, prec];
nu0 = N[nuX, prec];
G0 = E0/(2*(1 + nu0));
K0 = E0/(3*(1 - 2*nu0));
G2 = 2*G0;
G3 = 3*G0;
kX = N[knotX, prec];
kY = N[knotY, prec];
nk = Length[kX];
segSlope = Table[(kY[[i + 1]] - kY[[i]])/(kX[[i + 1]] - kX[[i]]), {i, 1, nk - 1}];
emax = N[emaxX, prec];
nLeg = nX;
latIt = latX;

(* Tabulated yield with end-slope extrapolation past the last knot. *)
yieldAt[ep_] := Module[{i = 1},
  If[ep <= kX[[1]], Return[kY[[1]]]];
  If[ep >= kX[[nk]],
    Return[kY[[nk]] + (ep - kX[[nk]])*segSlope[[nk - 1]]]];
  While[ep > kX[[i + 1]], i++];
  kY[[i]] + (ep - kX[[i]])*segSlope[[i]]
];

(* Von Mises norm from a deviatoric C1 tensor (exact 1/2, 3). *)
vonMisesDev[dev_] := Sqrt[3*((dev[[1]]^2 + dev[[2]]^2 + dev[[3]]^2)/2
  + dev[[4]]^2 + dev[[5]]^2 + dev[[6]]^2)];

(* Closed-form plastic multiplier: per-segment analytic root of
   qTrial - 3G dep - yield(ep + dep) == 0, walking forward across knots. *)
solveDep[qT_, ep_] := Module[{j = 1, a, b, dep},
  While[j < nk - 1 && ep > kX[[j + 1]], j++];
  While[True,
    If[j >= nk,
      b = segSlope[[nk - 1]];
      a = kY[[nk]] - b*kX[[nk]];
      dep = (qT - a - b*ep)/(G3 + b);
      Return[Max[dep, 0]],
      b = segSlope[[j]];
      a = kY[[j]] - b*kX[[j]];
      dep = (qT - a - b*ep)/(G3 + b);
      If[ep + dep <= kX[[j + 1]], Return[Max[dep, 0]]];
      j++
    ]
  ]
];

(* One J2 trial step in split form: {devNew, pNew, dep, yielding}. *)
j2step[dev_, p_, ep_, d_] := Module[{vol, pT, devT, qT, sy, dep, ratio},
  vol = d[[1]] + d[[2]] + d[[3]];
  pT = p + K0*vol;
  devT = {dev[[1]] + G2*(d[[1]] - vol/3),
    dev[[2]] + G2*(d[[2]] - vol/3),
    dev[[3]] + G2*(d[[3]] - vol/3),
    dev[[4]] + G2*(1/2)*d[[4]],
    dev[[5]] + G2*(1/2)*d[[5]],
    dev[[6]] + G2*(1/2)*d[[6]]};
  qT = vonMisesDev[devT];
  sy = yieldAt[ep];
  If[qT <= sy,
    {devT, pT, 0, False},
    dep = solveDep[qT, ep];
    ratio = 1 - G3*dep/qT;
    {ratio*devT, pT, dep, True}
  ]
];

(* Full C1 stress from split state (export and probes only). *)
sigOf[dev_, p_] := {dev[[1]] + p, dev[[2]] + p, dev[[3]] + p,
  dev[[4]], dev[[5]], dev[[6]]};

(* Lateral trial stress of one axial step (bisection probe). *)
stepYY[dev_, p_, ep_, de_, dl_] := Module[{u},
  u = j2step[dev, p, ep, {de, dl, dl, 0, 0, 0}];
  u[[1, 2]] + u[[2]]
];

(* Lateral strain for zero lateral stress (bisection, latIt passes). *)
solveDL[dev_, p_, ep_, de_] := Module[{lo, hi, mid, blo, bhi, w, k},
  If[de >= 0, lo = -de; hi = 0, lo = 0; hi = -de];
  blo = stepYY[dev, p, ep, de, lo];
  bhi = stepYY[dev, p, ep, de, hi];
  For[k = 1, k <= 20, k++,
    If[blo < 0 && bhi > 0, Break[]];
    w = Abs[hi - lo]; lo = lo - w; hi = hi + w;
    blo = stepYY[dev, p, ep, de, lo];
    bhi = stepYY[dev, p, ep, de, hi];
  ];
  If[! (blo < 0 && bhi > 0), Print["BRACKET FAILURE"]; Abort[]];
  Do[mid = (lo + hi)/2;
    If[stepYY[dev, p, ep, de, mid] > 0, hi = mid, lo = mid],
    {latIt}];
  (lo + hi)/2
];

(* Associative flow direction ((3/2) dev/q) from split state. *)
flowDir[dev_, p_] := Module[{q},
  q = vonMisesDev[dev];
  If[q <= 10^-12, {0, 0, 0, 0, 0, 0}, (3/2)*dev/q]
];

(* Drive a fixed-increment path; stations list 1-based step indices. *)
runFixed[steps_, stations_] := Module[{dev, p, ep, i, d, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0;
  yl = {}; epst = {};
  Do[d = steps[[i]];
    u = j2step[dev, p, ep, d];
    dev = u[[1]]; p = u[[2]]; ep = ep + u[[3]];
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[steps]}];
  {sigOf[dev, p], ep, yl, epst}
];

(* Drive an axial path with the lateral solve; also tracks the plastic
   tensor for the end-of-path failure-station rebuild. *)
runAxial[des_, stations_] := Module[{dev, p, ep, plas, i, de, dl, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0;
  plas = {0, 0, 0, 0, 0, 0};
  yl = {}; epst = {};
  Do[de = des[[i]];
    dl = solveDL[dev, p, ep, de];
    u = j2step[dev, p, ep, {de, dl, dl, 0, 0, 0}];
    dev = u[[1]]; p = u[[2]]; ep = ep + u[[3]];
    plas = plas + u[[3]]*flowDir[u[[1]], u[[2]]];
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[des]}];
  {sigOf[dev, p], ep, yl, epst, plas, dev, p}
];

de = emax/nLeg;
dg = emax/nLeg;
dv = emax/nLeg;
st100 = {20, 40, 60, 80, 100};
stCyc = {80, 160, 240, 320, 400};

uni = runAxial[Table[de, {nLeg}], st100];
shear = runFixed[Table[{0, 0, 0, dg, 0, 0}, {nLeg}], st100];
vol = runFixed[Table[{dv, dv, dv, 0, 0, 0}, {nLeg}], st100];
cycDes = Join[Table[de, {nLeg}], Table[-de, {2*nLeg}], Table[de, {nLeg}]];
cyc = runAxial[cycDes, stCyc];

(* Failure station: total-strain rebuild at the cycle end, with the
   production principal-strain convention (shear slots halved). *)
cycSig = cyc[[1]]; cycEp = cyc[[2]]; cycPlas = cyc[[5]];
cycDev = cyc[[6]]; cycP = cyc[[7]];
elasCyc = {cycDev[[1]]/G2 + cycP/(3*K0),
  cycDev[[2]]/G2 + cycP/(3*K0),
  cycDev[[3]]/G2 + cycP/(3*K0),
  cycDev[[4]]/G2, cycDev[[5]]/G2, cycDev[[6]]/G2};
totCyc = elasCyc + cycPlas;
tenCyc = {{totCyc[[1]], totCyc[[4]]/2, totCyc[[6]]/2},
  {totCyc[[4]]/2, totCyc[[2]], totCyc[[5]]/2},
  {totCyc[[6]]/2, totCyc[[5]]/2, totCyc[[3]]}};
firstCyc = Max[Eigenvalues[tenCyc]];
ruptured = cycEp >= N[rupX, prec];
deleted = firstCyc >= N[delX, prec];
tscale = Which[firstCyc <= N[t1X, prec], 1,
  firstCyc >= N[t2X, prec], 0,
  True, (N[t2X, prec] - firstCyc)/(N[t2X, prec] - N[t1X, prec])];
closure = vonMisesDev[cycDev] - yieldAt[cycEp];

(* Precision tripwire: every exported state value must keep 55 digits
   of absolute accuracy (closure excluded: it legitimately cancels). *)
chkList[uni[[1]], "uni_sigma"];
chk[uni[[2]], "uni_epbar"];
chkList[uni[[4]], "uni_epbar_st"];
chkList[shear[[1]], "shear_sigma"];
chk[shear[[2]], "shear_epbar"];
chkList[shear[[4]], "shear_epbar_st"];
chkList[vol[[1]], "vol_sigma"];
chk[vol[[2]], "vol_epbar"];
chkList[vol[[4]], "vol_epbar_st"];
chkList[cyc[[1]], "cycle_sigma"];
chk[cyc[[2]], "cycle_epbar"];
chkList[cyc[[4]], "cycle_epbar_st"];
chk[tscale, "failure_tscale"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law36_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_pa" -> N[eX],
    "poisson" -> N[nuX],
    "emax" -> N[emaxX],
    "steps_per_leg" -> nX,
    "lateral_iterations" -> latX,
    "stations_100" -> st100,
    "stations_cycle" -> stCyc,
    "curve_x" -> N[knotX],
    "curve_y_mpa" -> N[knotY/10^6]
  |>,
  "values" -> <|
    "uni_sigma" -> <|
      "values" -> dec /@ uni[[1]],
      "exact" -> "radial return, tabulated yield, lateral solved per step"
    |>,
    "uni_epbar" -> <|
      "values" -> {dec[uni[[2]]]},
      "exact" -> "Sum dep over 100 solved uniaxial steps"
    |>,
    "uni_yielding" -> <|
      "values" -> flag /@ uni[[3]],
      "exact" -> "qTrial > yield(ep) at steps {20,40,60,80,100}"
    |>,
    "uni_epbar_st" -> <|
      "values" -> dec /@ uni[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "shear_sigma" -> <|
      "values" -> dec /@ shear[[1]],
      "exact" -> "radial return under gamma_xy ramp"
    |>,
    "shear_epbar" -> <|
      "values" -> {dec[shear[[2]]]},
      "exact" -> "Sum dep over 100 shear steps"
    |>,
    "shear_yielding" -> <|
      "values" -> flag /@ shear[[3]],
      "exact" -> "qTrial > yield(ep) at steps {20,40,60,80,100}"
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
      "exact" -> "radial return over 0->+emax->-emax->0, solved lateral"
    |>,
    "cycle_epbar" -> <|
      "values" -> {dec[cyc[[2]]]},
      "exact" -> "Sum dep over 400 cycle steps"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[3]],
      "exact" -> "qTrial > yield(ep) at steps {80,160,240,320,400}"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[4]],
      "exact" -> "accumulated epbar at steps {80,160,240,320,400}"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "q(sigma) - yield(ep) at the cycle end, on-surface ~ 0"
    |>,
    "failure_ruptured" -> <|
      "values" -> {flag[ruptured]},
      "exact" -> "ep >= 1e20 at the cycle end"
    |>,
    "failure_deleted" -> <|
      "values" -> {flag[deleted]},
      "exact" -> "max principal total strain >= 3e20 at the cycle end"
    |>,
    "failure_tscale" -> <|
      "values" -> {dec[tscale]},
      "exact" -> "Zhao tensile scale at the cycle end, 1 below t1"
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
    "cycle_closure_rel" -> 1.0*^-9,
    "failure_ruptured_abs" -> 0.0,
    "failure_ruptured_rel" -> 0.0,
    "failure_deleted_abs" -> 0.0,
    "failure_deleted_rel" -> 0.0,
    "failure_tscale_abs" -> 1.0*^-15,
    "failure_tscale_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
