(* Independent Wolfram reference for LAW62 hyper-viscoelastic
   strain paths. Integrates, at 100-digit working precision, the
   compressible-Ogden plus Simo-finite-viscoelasticity equations
   along four strain paths: uniaxial extension with the lateral
   strain solved per step for zero lateral stress (bisection, 50
   passes), pure shear (gamma_xy ramp), small volumetric
   (0.01/axis, the near-incompressible bulk response), and a
   symmetric axial cycle 0 -> +emax -> -emax -> 0 with the same
   lateral solve. Every step runs at dt = 1 ms through the exact
   Prony update (decay plus gain on the instantaneous-Ogden
   deviatoric increment). Principal stretches come from an exact
   diagonal shortcut on axis-aligned paths and from Eigensystem
   on the shear path (an independent eigensolver; the
   rotate-diagonal-rotate-back reconstruction is basis invariant).
   The law carries no plasticity or damage: yielding stays False
   and epbar stays zero, and the viscous branch overstress is
   pinned as the internal variable alongside stress. All inputs
   are exact rationals converted once with N[..., 100]; the loop
   then runs at 100-digit working precision with no
   machine-precision literals, and N[..., 50] is applied once
   per exported value. Stations record the yielding flag and
   accumulated epbar at fifths of each path. The Rust side
   consumes solids_law62_paths_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/solids_law62_paths.wls > fixtures/solids_law62_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 100;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact card inputs: nu = 0.495, Ogden pairs (2 MPa, 2) and
   (1 MPa, -2), one Prony branch (30%, 10 ms), open viscosity. *)
nuX = 495/1000;
ogMuX = {2*10^6, 1*10^6};
ogAlX = {2, -2};
prGamX = {3/10};
prTauX = {1/100};
muMaxX = 10^30;
dtX = 10^-3;
emaxX = 2/10;
evmaxX = 1/100;
nLegX = 100;
latX = 50;

nu0 = N[nuX, prec];
ogMu = N[ogMuX, prec];
ogAl = N[ogAlX, prec];
prGam = N[prGamX, prec];
prTau = N[prTauX, prec];
muMax = N[muMaxX, prec];
dt = N[dtX, prec];
emax = N[emaxX, prec];
evmax = N[evmaxX, prec];
nLeg = nLegX;
latIt = latX;
npair = Length[ogMu];
nbr = Length[prGam];

betaOg = nu0/(1 - 2*nu0);
g0 = Total[ogMu];
gammaInf = 1 - Total[prGam];
moduli = Table[Min[prGam[[i]]*g0, muMax/prTau[[i]]], {i, 1, nbr}];
share = moduli/g0;
decay = Table[Exp[-dt/prTau[[i]]], {i, 1, nbr}];
gain = Table[If[dt/prTau[[i]] > 10^-12,
  (1 - Exp[-dt/prTau[[i]]])/(dt/prTau[[i]]), 1], {i, 1, nbr}];

(* Exact Ogden principal Cauchy stresses at principal stretches. *)
ogdenPrincipal[lam_] := Module[{jac, out, a, al, mu, jt, pre},
  jac = lam[[1]]*lam[[2]]*lam[[3]];
  out = {0, 0, 0};
  Do[mu = ogMu[[k]]; al = ogAl[[k]];
    jt = jac^(-al*betaOg);
    pre = 2*mu/(al*jac);
    Do[out[[a]] = out[[a]] + pre*(lam[[a]]^al - jt), {a, 1, 3}],
    {k, 1, npair}];
  out
];

(* C1 tensor from a full 3x3 matrix (symmetric part). *)
c1FromFull[m_] := {m[[1, 1]], m[[2, 2]], m[[3, 3]],
  (m[[1, 2]] + m[[2, 1]])/2, (m[[2, 3]] + m[[3, 2]])/2,
  (m[[1, 3]] + m[[3, 1]])/2};

(* Full 3x3 matrix from a C1 tensor (tensor shear slots). *)
fullFromC1[t_] := {{t[[1]], t[[4]], t[[6]]},
  {t[[4]], t[[2]], t[[5]]}, {t[[6]], t[[5]], t[[3]]}};

(* Full 3x3 matrix from a C1 Hencky tensor (halved shears). *)
henckyFull[h_] := {{h[[1]], h[[4]]/2, h[[6]]/2},
  {h[[4]]/2, h[[2]], h[[5]]/2}, {h[[6]]/2, h[[5]]/2, h[[3]]}};

(* Principal values plus rotation columns of a Hencky state:
   exact diagonal shortcut when shears vanish, Eigensystem
   otherwise (independent eigensolver, invariant rebuild). *)
principalSystem[h_] := If[h[[4]] == 0 && h[[5]] == 0 && h[[6]] == 0,
  {{h[[1]], h[[2]], h[[3]]},
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
  Module[{sys},
    sys = Eigensystem[N[henckyFull[h], prec]];
    {sys[[1]], Transpose[sys[[2]]]}]
];

(* Instantaneous Ogden deviatoric stress in C1. *)
ogdenDev[h_] := Module[{vals, rot, lam, og, pOg, sOg},
  {vals, rot} = principalSystem[h];
  lam = Exp[vals];
  og = ogdenPrincipal[lam];
  pOg = (og[[1]] + og[[2]] + og[[3]])/3;
  sOg = og - {pOg, pOg, pOg};
  c1FromFull[rot.DiagonalMatrix[sOg].Transpose[rot]]
];

(* One viscoelastic trial: {stress, henckyNew, branchesNew, s0Trial}.
   s0Old is the cached instantaneous deviatoric of the committed
   state (identical input, identical value). *)
trialStep[h_, branches_, s0old_, d_] := Module[{hT, vals, rot, lam,
  og, pOg, sOg, s0T, bNew, full, i, sFull},
  hT = h + d;
  {vals, rot} = principalSystem[hT];
  lam = Exp[vals];
  og = ogdenPrincipal[lam];
  pOg = (og[[1]] + og[[2]] + og[[3]])/3;
  sOg = og - {pOg, pOg, pOg};
  s0T = c1FromFull[rot.DiagonalMatrix[sOg].Transpose[rot]];
  bNew = Table[decay[[i]]*branches[[i]]
    + share[[i]]*gain[[i]]*(s0T - s0old), {i, 1, nbr}];
  full = DiagonalMatrix[gammaInf*sOg + {pOg, pOg, pOg}];
  Do[full = full + Transpose[rot].fullFromC1[bNew[[i]]].rot,
    {i, 1, nbr}];
  sFull = rot.full.Transpose[rot];
  {c1FromFull[sFull], hT, bNew, s0T}
];

(* Von Mises norm of a full C1 stress. *)
vonMisesFull[sig_] := Module[{p, s0, s1, s2},
  p = (sig[[1]] + sig[[2]] + sig[[3]])/3;
  s0 = sig[[1]] - p; s1 = sig[[2]] - p; s2 = sig[[3]] - p;
  Sqrt[(3/2)*(s0^2 + s1^2 + s2^2) + 3*(sig[[4]]^2 + sig[[5]]^2 + sig[[6]]^2)]
];

(* Lateral trial stress of one axial step (bisection probe). *)
stepYY[h_, branches_, s0old_, de_, dl_] := Module[{u},
  u = trialStep[h, branches, s0old, {de, dl, dl, 0, 0, 0}];
  u[[1, 2]]
];

(* Lateral strain for zero lateral stress (bisection, latIt passes). *)
solveDL[h_, branches_, s0old_, de_] := Module[{lo, hi, mid, blo, bhi, w, k},
  If[de >= 0, lo = -de; hi = 0, lo = 0; hi = -de];
  blo = stepYY[h, branches, s0old, de, lo];
  bhi = stepYY[h, branches, s0old, de, hi];
  For[k = 1, k <= 20, k++,
    If[blo < 0 && bhi > 0, Break[]];
    w = Abs[hi - lo]; lo = lo - w; hi = hi + w;
    blo = stepYY[h, branches, s0old, de, lo];
    bhi = stepYY[h, branches, s0old, de, hi];
  ];
  If[! (blo < 0 && bhi > 0), Print["BRACKET FAILURE"]; Abort[]];
  Do[mid = (lo + hi)/2;
    If[stepYY[h, branches, s0old, de, mid] > 0, hi = mid, lo = mid],
    {latIt}];
  (lo + hi)/2
];

(* Drive a fixed-increment path; stations list 1-based step indices. *)
runFixed[steps_, stations_] := Module[{h, branches, s0, i, d, u, yl, epst},
  h = {0, 0, 0, 0, 0, 0};
  branches = Table[{0, 0, 0, 0, 0, 0}, {nbr}];
  s0 = ogdenDev[h];
  yl = {}; epst = {};
  Do[d = steps[[i]];
    u = trialStep[h, branches, s0, d];
    h = u[[2]]; branches = u[[3]]; s0 = u[[4]];
    If[MemberQ[stations, i], AppendTo[yl, False]; AppendTo[epst, 0]],
    {i, 1, Length[steps]}];
  {trialStressOnly[h, branches], h, branches, yl, epst}
];

(* Stress rebuild at the committed end state (export only). *)
trialStressOnly[h_, branches_] := Module[{vals, rot, lam, og, pOg, sOg, full},
  {vals, rot} = principalSystem[h];
  lam = Exp[vals];
  og = ogdenPrincipal[lam];
  pOg = (og[[1]] + og[[2]] + og[[3]])/3;
  sOg = og - {pOg, pOg, pOg};
  full = DiagonalMatrix[gammaInf*sOg + {pOg, pOg, pOg}];
  Do[full = full + Transpose[rot].fullFromC1[branches[[i]]].rot,
    {i, 1, nbr}];
  c1FromFull[rot.full.Transpose[rot]]
];

(* Drive an axial path with the lateral solve, tracking the first
   branch norm at stations for the internal-variable pin. *)
runAxial[des_, stations_] := Module[{h, branches, s0, i, de, dl, u,
  yl, epst, bst},
  h = {0, 0, 0, 0, 0, 0};
  branches = Table[{0, 0, 0, 0, 0, 0}, {nbr}];
  s0 = ogdenDev[h];
  yl = {}; epst = {}; bst = {};
  Do[de = des[[i]];
    dl = solveDL[h, branches, s0, de];
    u = trialStep[h, branches, s0, {de, dl, dl, 0, 0, 0}];
    h = u[[2]]; branches = u[[3]]; s0 = u[[4]];
    If[MemberQ[stations, i], AppendTo[yl, False]; AppendTo[epst, 0];
      AppendTo[bst, vonMisesFull[branches[[1]]]]],
    {i, 1, Length[des]}];
  {trialStressOnly[h, branches], h, branches, yl, epst, bst}
];

de = emax/nLeg;
dg = emax/nLeg;
dv = evmax/nLeg;
st100 = {20, 40, 60, 80, 100};
stCyc = {80, 160, 240, 320, 400};

uni = runAxial[Table[de, {nLeg}], st100];
shear = runFixed[Table[{0, 0, 0, dg, 0, 0}, {nLeg}], st100];
vol = runFixed[Table[{dv, dv, dv, 0, 0, 0}, {nLeg}], st100];
cycDes = Join[Table[de, {nLeg}], Table[-de, {2*nLeg}], Table[de, {nLeg}]];
cyc = runAxial[cycDes, stCyc];

(* Residual stress after the strain loop closes (branch memory). *)
closure = vonMisesFull[cyc[[1]]];

(* Precision tripwire on every exported state value. *)
chkList[uni[[1]], "uni_sigma"];
chkList[uni[[2]], "uni_hencky"];
chkList[uni[[3, 1]], "uni_branch"];
chkList[uni[[6]], "uni_branch_st"];
chkList[shear[[1]], "shear_sigma"];
chkList[vol[[1]], "vol_sigma"];
chkList[cyc[[1]], "cycle_sigma"];
chk[closure, "cycle_closure"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law62_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "poisson" -> N[nuX],
    "ogden_mu_pa" -> N[ogMuX],
    "ogden_alpha" -> N[ogAlX],
    "prony_gamma" -> N[prGamX],
    "prony_tau_s" -> N[prTauX],
    "dt_s" -> N[dtX],
    "emax" -> N[emaxX],
    "evmax" -> N[evmaxX],
    "steps_per_leg" -> nLegX,
    "lateral_iterations" -> latX,
    "stations_100" -> st100,
    "stations_cycle" -> stCyc
  |>,
  "values" -> <|
    "uni_sigma" -> <|
      "values" -> dec /@ uni[[1]],
      "exact" -> "Ogden-Prony uniaxial, lateral solved per step"
    |>,
    "uni_yielding" -> <|
      "values" -> flag /@ uni[[4]],
      "exact" -> "all False, hyper-viscoelasticity never yields"
    |>,
    "uni_epbar_st" -> <|
      "values" -> dec /@ uni[[5]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "uni_hencky" -> <|
      "values" -> dec /@ uni[[2]],
      "exact" -> "committed Hencky strain, solved lateral included"
    |>,
    "uni_branch" -> <|
      "values" -> dec /@ uni[[3, 1]],
      "exact" -> "first Prony overstress at the uniaxial end"
    |>,
    "uni_branch_st" -> <|
      "values" -> dec /@ uni[[6]],
      "exact" -> "first-branch von Mises at steps {20,40,60,80,100}"
    |>,
    "shear_sigma" -> <|
      "values" -> dec /@ shear[[1]],
      "exact" -> "Ogden-Prony under gamma_xy ramp"
    |>,
    "shear_yielding" -> <|
      "values" -> flag /@ shear[[4]],
      "exact" -> "all False, hyper-viscoelasticity never yields"
    |>,
    "shear_epbar_st" -> <|
      "values" -> dec /@ shear[[5]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "vol_sigma" -> <|
      "values" -> dec /@ vol[[1]],
      "exact" -> "Ogden bulk response at 0.01 Hencky per axis"
    |>,
    "vol_yielding" -> <|
      "values" -> flag /@ vol[[4]],
      "exact" -> "all False, hyper-viscoelasticity never yields"
    |>,
    "vol_epbar_st" -> <|
      "values" -> dec /@ vol[[5]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "cycle_sigma" -> <|
      "values" -> dec /@ cyc[[1]],
      "exact" -> "Ogden-Prony over 0->+emax->-emax->0, solved lateral"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[4]],
      "exact" -> "all False, hyper-viscoelasticity never yields"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[5]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "von Mises of the cycle end stress"
    |>
  |>,
  "tolerances" -> <|
    "uni_sigma_abs" -> 1.0*^-5,
    "uni_sigma_rel" -> 1.0*^-12,
    "uni_yielding_abs" -> 0.0,
    "uni_yielding_rel" -> 0.0,
    "uni_epbar_st_abs" -> 0.0,
    "uni_epbar_st_rel" -> 0.0,
    "uni_hencky_abs" -> 1.0*^-12,
    "uni_hencky_rel" -> 1.0*^-12,
    "uni_branch_abs" -> 1.0*^-6,
    "uni_branch_rel" -> 1.0*^-9,
    "uni_branch_st_abs" -> 1.0*^-6,
    "uni_branch_st_rel" -> 1.0*^-9,
    "shear_sigma_abs" -> 1.0*^-5,
    "shear_sigma_rel" -> 1.0*^-12,
    "shear_yielding_abs" -> 0.0,
    "shear_yielding_rel" -> 0.0,
    "shear_epbar_st_abs" -> 0.0,
    "shear_epbar_st_rel" -> 0.0,
    "vol_sigma_abs" -> 1.0*^-4,
    "vol_sigma_rel" -> 1.0*^-12,
    "vol_yielding_abs" -> 0.0,
    "vol_yielding_rel" -> 0.0,
    "vol_epbar_st_abs" -> 0.0,
    "vol_epbar_st_rel" -> 0.0,
    "cycle_sigma_abs" -> 1.0*^-4,
    "cycle_sigma_rel" -> 1.0*^-12,
    "cycle_yielding_abs" -> 0.0,
    "cycle_yielding_rel" -> 0.0,
    "cycle_epbar_st_abs" -> 0.0,
    "cycle_epbar_st_rel" -> 0.0,
    "cycle_closure_abs" -> 1.0*^-4,
    "cycle_closure_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
