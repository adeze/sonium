(* Independent Wolfram reference for the LAW27 brittle-damage strain paths.
   Integrates, at 400-digit working precision, the von Mises radial-return
   core (bisection dep: the n = 1/2 radical form would double ep's
   tracked error each step through its direct -ep term) with the LAW27
   total-strain brittle damage, zero-strength-flow rebooking, and the
   eps_f deletion latch over five strain paths: solved-lateral uniaxial
   tension to 0.10 (damage, intact), fixed shear to 0.43 (non-coaxial
   damage at rebuilt principals +-0.11), volumetric compression to
   -0.10 (elastic, tension-only damage stays shut), a solved-lateral
   +-0.10 cycle (damage follows the totals, ends on the surface with
   no thermal lag), and solved-lateral tension to 0.18 (deletion past
   0.16 zeroes the stress; constrained tension self-limits at
   marginal damage through pressure scaling and never deletes).
   The volumetric leg runs in
   compression because tensile volumetric strain has a triply degenerate
   principal basis, whose intact direction is eigensolver-dependent.
   All inputs are exact rationals; export is N[..., 50] via CForm.
   The Rust side replays law27_paths_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/solids_law27_paths.wls > fixtures/solids_law27_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 400;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact card inputs: brittle steel, E = 200 GPa, nu = 0.3, JC core
   A = 250 MPa, B = 500 MPa, n = 1/2, c = 0 (rate-free), m = 0 and
   Cv = 0 (no thermal part), brittle dirs eps_t = 0.08, eps_m = 0.15,
   dmax = 0.999, eps_f = 0.16, core caps open at 1e30. *)
eX = 200*10^9;
nuX = 3/10;
aX = 250*10^6;
bX = 500*10^6;
nX = 1/2;
trX = 298;
epsTX = 8/100;
epsMX = 15/100;
dmaxX = 999/1000;
epsFX = 16/100;
emaxX = 10/100;
emaxShearX = 43/100;
emaxVolX = -10/100;
emaxTenX = 18/100;
nLegX = 100;
latX = 50;

E0 = N[eX, prec];
nu0 = N[nuX, prec];
A0 = N[aX, prec];
B0 = N[bX, prec];
tref = N[trX, prec];
epsT = N[epsTX, prec];
epsM = N[epsMX, prec];
dmax = N[dmaxX, prec];
epsF = N[epsFX, prec];
G0 = E0/(2*(1 + nu0));
K0 = E0/(3*(1 - 2*nu0));
G2 = 2*G0;
G3 = 3*G0;
emax = N[emaxX, prec];
emaxShear = N[emaxShearX, prec];
emaxVol = N[emaxVolX, prec];
emaxTen = N[emaxTenX, prec];
nLeg = nLegX;
latIt = latX;

(* JC core flow stress (rate factor and thermal factor are exactly 1:
   the card sets c = 0, m = 0), capped at 1e30 for fidelity. *)
flow[epe_] := Min[A0 + B0*Sqrt[epe], 10^30];

(* Pure bisection (170 passes) solve of
   qTrial - 3G dep - flow(ep + dep) == 0 directly in dep. The bracket
   [0, over/3G] always straddles the root for a hardening law.
   Bisection is used instead of the n = 1/2 radical form because that
   form's direct -ep term doubles ep's tracked error every step
   (measured x2/step, dragging the stress along at x3G); bisection
   reads ep only slope-weighted inside flow, so the tracked error
   stays bounded like the other closed-form laws. 170 passes resolve
   the bracket (hi <= 3e-3) to width ~1e-54 with sign tests decidable
   by 200+ orders of margin. The residual assert (1e-50 of qTrial)
   independently validates the root. *)
solveDep[qT_, ep_] := Module[{sy0, over, hi, a, b, c, fc, dep, f},
  sy0 = flow[ep];
  over = qT - sy0;
  hi = over/G3;
  a = 0; b = hi;
  Do[
    c = (a + b)/2;
    fc = qT - G3*c - flow[ep + c];
    If[fc > 0, a = c, b = c],
    {170}];
  dep = (a + b)/2;
  f = qT - G3*dep - flow[ep + dep];
  If[Abs[f] > 10^-50*qT, Print["SOLVE FAILURE"]; Abort[]];
  dep
];

(* Von Mises norm from a deviatoric C1 tensor (exact 3, 1/2). *)
vonMisesDev[dev_] := Sqrt[3*((dev[[1]]^2 + dev[[2]]^2 + dev[[3]]^2)/2
  + dev[[4]]^2 + dev[[5]]^2 + dev[[6]]^2)];

(* Von Mises norm from a full C1 stress. *)
vonMisesFull[s6_] := Module[{p, dev},
  p = (s6[[1]] + s6[[2]] + s6[[3]])/3;
  dev = {s6[[1]] - p, s6[[2]] - p, s6[[3]] - p, s6[[4]], s6[[5]], s6[[6]]};
  vonMisesDev[dev]
];

(* Equivalent of a C1 strain increment (engineering shear slots). *)
equivIncr[d_] := Module[{m, d0, d1, d2, sh},
  m = (d[[1]] + d[[2]] + d[[3]])/3;
  d0 = d[[1]] - m; d1 = d[[2]] - m; d2 = d[[3]] - m;
  sh = (d[[4]]^2 + d[[5]]^2 + d[[6]]^2)/2;
  Sqrt[(2/3)*(d0^2 + d1^2 + d2^2 + sh)]
];

(* Equivalent of a deviatoric C1 tensor with tensor shear slots. *)
equivDeviatoric[t_] := Sqrt[(2/3)*(t[[1]]^2 + t[[2]]^2 + t[[3]]^2
  + 2*(t[[4]]^2 + t[[5]]^2 + t[[6]]^2))];

(* Elastic strain of a C1 stress with bulk and shear moduli. *)
elasticStrain[s6_] := Module[{p},
  p = (s6[[1]] + s6[[2]] + s6[[3]])/3;
  {(s6[[1]] - p)/G2 + p/(3*K0),
   (s6[[2]] - p)/G2 + p/(3*K0),
   (s6[[3]] - p)/G2 + p/(3*K0),
   s6[[4]]/G2, s6[[5]]/G2, s6[[6]]/G2}
];

(* Full 3x3 matrix from a C1 strain tensor (engineering shear halved). *)
fullFromC1Strain[t_] := {{t[[1]], t[[4]]/2, t[[6]]/2},
  {t[[4]]/2, t[[2]], t[[5]]/2}, {t[[6]]/2, t[[5]]/2, t[[3]]}};

(* Full 3x3 matrix from a C1 stress tensor (shear already tensor). *)
fullFromC1Stress[t_] := {{t[[1]], t[[4]], t[[6]]},
  {t[[4]], t[[2]], t[[5]]}, {t[[6]], t[[5]], t[[3]]}};

(* C1 tensor from a 3x3 matrix (off-diagonal pairs averaged). *)
c1FromFull[m_] := {m[[1, 1]], m[[2, 2]], m[[3, 3]],
  (m[[1, 2]] + m[[2, 1]])/2, (m[[2, 3]] + m[[3, 2]])/2,
  (m[[1, 3]] + m[[3, 1]])/2};

(* Full C1 stress from split state (export and probes only). *)
sigOf[dev_, p_] := {dev[[1]] + p, dev[[2]] + p, dev[[3]] + p,
  dev[[4]], dev[[5]], dev[[6]]};

(* Principal system of a C1 strain tensor: values largest first with
   the eigenvector columns in the same order. Eigenvector signs are
   arbitrary; the brittle scaling below is sign-invariant. *)
principalSystem[t6_] := Module[{vals, vecs, ord},
  {vals, vecs} = Eigensystem[fullFromC1Strain[t6]];
  ord = Ordering[vals, All, Greater];
  {vals[[ord]], Transpose[vecs[[ord]]]}
];

(* Brittle damage at a principal strain (tension only). *)
damageAt[p_] := If[p <= epsT, 0,
  If[p >= epsM, dmax, Min[(p - epsT)/(epsM - epsT), dmax]]];

(* Directional brittle scaling of a trial stress in the
   strain-principal basis (third direction intact). *)
applyBrittle[s6_, d1_, d2_, q_] := Module[{keep, full, hat, back},
  If[d1 <= 0 && d2 <= 0, Return[s6]];
  keep = {1 - d1, 1 - d2, 1};
  full = fullFromC1Stress[s6];
  hat = Table[Sqrt[keep[[i]]*keep[[j]]]*
    (Transpose[q].full.q)[[i, j]], {i, 3}, {j, 3}];
  back = q.hat.Transpose[q];
  c1FromFull[back]
];

(* Associative flow direction of a C1 stress with the 1e-12 production
   floor: {direction, hasDirection}, zero vector when tiny. *)
flowDir[s6_] := Module[{p, dev, q},
  p = (s6[[1]] + s6[[2]] + s6[[3]])/3;
  dev = {s6[[1]] - p, s6[[2]] - p, s6[[3]] - p,
    s6[[4]], s6[[5]], s6[[6]]};
  q = vonMisesDev[dev];
  If[q <= 10^-12, {{0, 0, 0, 0, 0, 0}, False}, {(3/2)*dev/q, True}]
];

(* Trial total strain: elastic part plus plastic accumulation plus the
   step flow along the trial direction (no direction: plastic alone). *)
trialTotal[plastic_, s6_, dep_, dir_] :=
  elasticStrain[s6] + plastic + dep*dir;

(* One LAW27 trial step in split form:
   {devNew, pNew, depBooked, yielding, {d1, d2}, rupturedNow}.
   Runs the JC core return, rebuilds trial totals, rebooks the full
   equivalent increment once brittle damage opens, latches deletion
   at eps_f (stress zeroed, dep kept), else applies the brittle
   scaling. Undamaged steps return the split core state directly
   (no reassembly round-trip). The core rupture assert documents
   that 1e30 never binds. *)
updateInner[dev_, p_, ep_, plastic_, dmg_, ruptured_, d6_] := Module[
  {vol, pT, devT, qT, sy, depCore, dev1, yld, trial6, fd, dir,
   total, ps, vals, qmat, d1, d2, dep, total2, ps2, s6, pNew, devNew},
  If[ruptured, Return[{{0, 0, 0, 0, 0, 0}, 0, 0, True, dmg, True}]];
  If[ep >= 10^30, Print["CORE RUPTURE"]; Abort[]];
  vol = d6[[1]] + d6[[2]] + d6[[3]];
  pT = p + K0*vol;
  devT = {dev[[1]] + G2*(d6[[1]] - vol/3),
    dev[[2]] + G2*(d6[[2]] - vol/3),
    dev[[3]] + G2*(d6[[3]] - vol/3),
    dev[[4]] + G2*(1/2)*d6[[4]],
    dev[[5]] + G2*(1/2)*d6[[5]],
    dev[[6]] + G2*(1/2)*d6[[6]]};
  qT = vonMisesDev[devT];
  sy = flow[ep];
  If[qT <= sy,
    depCore = 0; dev1 = devT; yld = False,
    depCore = solveDep[qT, ep];
    dev1 = (1 - G3*depCore/qT)*devT; yld = True];
  trial6 = sigOf[dev1, pT];
  fd = flowDir[trial6]; dir = fd[[1]];
  total = trialTotal[plastic, trial6, depCore, dir];
  ps = principalSystem[total]; vals = ps[[1]]; qmat = ps[[2]];
  d1 = damageAt[vals[[1]]]; d2 = damageAt[vals[[2]]];
  dep = depCore;
  If[d1 > 0 || d2 > 0,
    dep = equivIncr[d6]; yld = True;
    total2 = trialTotal[plastic, trial6, dep, dir];
    ps2 = principalSystem[total2]; vals = ps2[[1]]; qmat = ps2[[2]];
    d1 = damageAt[vals[[1]]]; d2 = damageAt[vals[[2]]];
  ];
  If[vals[[1]] >= epsF || vals[[2]] >= epsF,
    Return[{{0, 0, 0, 0, 0, 0}, 0, dep, True, {d1, d2}, True}]];
  If[d1 <= 0 && d2 <= 0, Return[{dev1, pT, dep, yld, {d1, d2}, False}]];
  s6 = applyBrittle[trial6, d1, d2, qmat];
  pNew = (s6[[1]] + s6[[2]] + s6[[3]])/3;
  devNew = {s6[[1]] - pNew, s6[[2]] - pNew, s6[[3]] - pNew,
    s6[[4]], s6[[5]], s6[[6]]};
  {devNew, pNew, dep, yld, {d1, d2}, False}
];

(* Commit a trial step: advance stress, ep, and the plastic tensor
   (accumulated-flow fallback at zero stress, virgin freeze-out at
   eq <= 1e-30), then refresh damage and the rupture latch from
   committed totals. State: {dev, p, ep, plastic, {d1, d2}, ruptured}. *)
commitStep[dev_, p_, ep_, plastic_, dmg_, ruptured_, u_] := Module[
  {devN, pN, epN, s6, fd, dir, ok, eq, plN, total, ps, vals, rupN},
  devN = u[[1]]; pN = u[[2]]; epN = ep + u[[3]];
  s6 = sigOf[devN, pN];
  fd = flowDir[s6]; dir = fd[[1]]; ok = fd[[2]];
  plN = plastic;
  If[ok, plN = plastic + u[[3]]*dir,
    eq = equivDeviatoric[plastic];
    If[eq > 10^-30, plN = plastic + u[[3]]*plastic/eq]];
  total = elasticStrain[s6] + plN;
  ps = principalSystem[total]; vals = ps[[1]];
  rupN = ruptured || vals[[1]] >= epsF || vals[[2]] >= epsF;
  {devN, pN, epN, plN, {damageAt[vals[[1]]], damageAt[vals[[2]]]}, rupN}
];

(* Lateral trial stress of one axial step (bisection probe). *)
stepYY[dev_, p_, ep_, plastic_, dmg_, ruptured_, de_, dl_] := Module[{u},
  u = updateInner[dev, p, ep, plastic, dmg, ruptured, {de, dl, dl, 0, 0, 0}];
  u[[1, 2]] + u[[2]]
];

(* Lateral strain for zero lateral stress (bisection, latIt passes).
   Post-deletion (and across the rupture swath, where no strict
   bracket exists) the lateral stress is identically zero, so the
   lateral strain is unconstrained: return 0 by deterministic
   convention. Exact-zero probes are exactly the ruptured ones on
   both sides, so the fallback triggers identically. *)
solveDL[dev_, p_, ep_, plastic_, dmg_, ruptured_, de_] := Module[
  {lo, hi, mid, blo, bhi, w, k},
  If[ruptured, Return[0]];
  If[de >= 0, lo = -de; hi = 0, lo = 0; hi = -de];
  blo = stepYY[dev, p, ep, plastic, dmg, ruptured, de, lo];
  bhi = stepYY[dev, p, ep, plastic, dmg, ruptured, de, hi];
  For[k = 1, k <= 20, k++,
    If[blo < 0 && bhi > 0, Break[]];
    w = Abs[hi - lo]; lo = lo - w; hi = hi + w;
    blo = stepYY[dev, p, ep, plastic, dmg, ruptured, de, lo];
    bhi = stepYY[dev, p, ep, plastic, dmg, ruptured, de, hi];
  ];
  If[! (blo < 0 && bhi > 0),
    If[blo == 0 || bhi == 0, Return[0]];
    Print["BRACKET FAILURE"]; Abort[]];
  Do[mid = (lo + hi)/2;
    If[stepYY[dev, p, ep, plastic, dmg, ruptured, de, mid] > 0,
      hi = mid, lo = mid],
    {latIt}];
  (lo + hi)/2
];

(* Drive a fixed-increment path; stations list 1-based step indices.
   Returns {sigma, ep, yielding, epStations, temp, damage, dmgStations,
   rupturedFlags}; temperature stays Tref (Cv = 0: no heating). *)
runFixed[steps_, stations_] := Module[
  {dev, p, ep, pl, dmg, rup, i, d, u, st, yl, epst, dgst, rp},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0;
  pl = {0, 0, 0, 0, 0, 0}; dmg = {0, 0}; rup = False;
  yl = {}; epst = {}; dgst = {}; rp = {};
  Do[d = steps[[i]];
    u = updateInner[dev, p, ep, pl, dmg, rup, d];
    st = commitStep[dev, p, ep, pl, dmg, rup, u];
    dev = st[[1]]; p = st[[2]]; ep = st[[3]];
    pl = st[[4]]; dmg = st[[5]]; rup = st[[6]];
    If[MemberQ[stations, i],
      AppendTo[yl, u[[4]]]; AppendTo[epst, ep];
      AppendTo[dgst, dmg[[1]]]; AppendTo[rp, rup]],
    {i, 1, Length[steps]}];
  {sigOf[dev, p], ep, yl, epst, tref, dmg, dgst, rp}
];

(* Drive an axial path with the lateral solve. *)
runAxial[des_, stations_] := Module[
  {dev, p, ep, pl, dmg, rup, i, de, dl, u, st, yl, epst, dgst, rp},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0;
  pl = {0, 0, 0, 0, 0, 0}; dmg = {0, 0}; rup = False;
  yl = {}; epst = {}; dgst = {}; rp = {};
  Do[de = des[[i]];
    dl = solveDL[dev, p, ep, pl, dmg, rup, de];
    u = updateInner[dev, p, ep, pl, dmg, rup, {de, dl, dl, 0, 0, 0}];
    st = commitStep[dev, p, ep, pl, dmg, rup, u];
    dev = st[[1]]; p = st[[2]]; ep = st[[3]];
    pl = st[[4]]; dmg = st[[5]]; rup = st[[6]];
    If[MemberQ[stations, i],
      AppendTo[yl, u[[4]]]; AppendTo[epst, ep];
      AppendTo[dgst, dmg[[1]]]; AppendTo[rp, rup]],
    {i, 1, Length[des]}];
  {sigOf[dev, p], ep, yl, epst, tref, dmg, dgst, rp}
];

de = emax/nLeg;
dg = emaxShear/nLeg;
dv = emaxVol/nLeg;
dten = emaxTen/nLeg;
st100 = {20, 40, 60, 80, 100};
stCyc = {80, 160, 240, 320, 400};

uni = runAxial[Table[de, {nLeg}], st100];
shear = runFixed[Table[{0, 0, 0, dg, 0, 0}, {nLeg}], st100];
vol = runFixed[Table[{dv, dv, dv, 0, 0, 0}, {nLeg}], st100];
cycDes = Join[Table[de, {nLeg}], Table[-de, {2*nLeg}], Table[de, {nLeg}]];
cyc = runAxial[cycDes, stCyc];
tension = runAxial[Table[dten, {nLeg}], st100];

(* Cycle closure: q minus the unrated core flow at the cycle end.
   The final reloading leg yields in tension, so the end state sits
   on the grown surface (~0); no thermal part exists (m = 0), so no
   thermal lag offsets it. *)
closure = vonMisesFull[cyc[[1]]] - flow[cyc[[2]]];

(* Precision tripwire: every exported state value must keep 55 digits
   of absolute accuracy. *)
chkList[uni[[1]], "uni_sigma"];
chk[uni[[2]], "uni_epbar"];
chkList[uni[[4]], "uni_epbar_st"];
chk[uni[[5]], "uni_temp"];
chkList[uni[[6]], "uni_damage"];
chkList[uni[[7]], "uni_dmg_st"];
chkList[shear[[1]], "shear_sigma"];
chk[shear[[2]], "shear_epbar"];
chkList[shear[[4]], "shear_epbar_st"];
chk[shear[[5]], "shear_temp"];
chkList[shear[[6]], "shear_damage"];
chkList[shear[[7]], "shear_dmg_st"];
chkList[vol[[1]], "vol_sigma"];
chk[vol[[2]], "vol_epbar"];
chkList[vol[[4]], "vol_epbar_st"];
chk[vol[[5]], "vol_temp"];
chkList[vol[[6]], "vol_damage"];
chkList[vol[[7]], "vol_dmg_st"];
chkList[cyc[[1]], "cycle_sigma"];
chk[cyc[[2]], "cycle_epbar"];
chkList[cyc[[4]], "cycle_epbar_st"];
chk[cyc[[5]], "cycle_temp"];
chkList[cyc[[6]], "cycle_damage"];
chkList[cyc[[7]], "cycle_dmg_st"];
chk[closure, "cycle_closure"];
chkList[tension[[1]], "tension_sigma"];
chk[tension[[2]], "tension_epbar"];
chkList[tension[[4]], "tension_epbar_st"];
chk[tension[[5]], "tension_temp"];
chkList[tension[[6]], "tension_damage"];
chkList[tension[[7]], "tension_dmg_st"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law27_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_pa" -> N[eX],
    "poisson" -> N[nuX],
    "a_pa" -> N[aX],
    "b_pa" -> N[bX],
    "n" -> N[nX],
    "t_ref_k" -> N[trX],
    "eps_t" -> N[epsTX],
    "eps_m" -> N[epsMX],
    "dmax" -> N[dmaxX],
    "eps_f" -> N[epsFX],
    "emax" -> N[emaxX],
    "emax_shear" -> N[emaxShearX],
    "emax_vol" -> N[emaxVolX],
    "emax_tension" -> N[emaxTenX],
    "steps_per_leg" -> nLegX,
    "lateral_iterations" -> latX,
    "stations_100" -> st100,
    "stations_cycle" -> stCyc
  |>,
  "values" -> <|
    "uni_sigma" -> <|
      "values" -> dec /@ uni[[1]],
      "exact" -> "JC core return with brittle scaling, lateral solved per step"
    |>,
    "uni_epbar" -> <|
      "values" -> {dec[uni[[2]]]},
      "exact" -> "Sum booked dep over 100 solved uniaxial steps"
    |>,
    "uni_yielding" -> <|
      "values" -> flag /@ uni[[3]],
      "exact" -> "core yield or damage flow at steps {20,40,60,80,100}"
    |>,
    "uni_epbar_st" -> <|
      "values" -> dec /@ uni[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "uni_temp" -> <|
      "values" -> {dec[uni[[5]]]},
      "exact" -> "Tref, Cv = 0 disables heating"
    |>,
    "uni_damage" -> <|
      "values" -> dec /@ uni[[6]],
      "exact" -> "[d1, d2] at total axial 0.10: ramp value, zero lateral"
    |>,
    "uni_dmg_st" -> <|
      "values" -> dec /@ uni[[7]],
      "exact" -> "d1 at steps {20,40,60,80,100}, onset at step 80"
    |>,
    "shear_sigma" -> <|
      "values" -> dec /@ shear[[1]],
      "exact" -> "JC core return with non-coaxial brittle scaling in shear"
    |>,
    "shear_epbar" -> <|
      "values" -> {dec[shear[[2]]]},
      "exact" -> "Sum booked dep over 100 shear steps"
    |>,
    "shear_yielding" -> <|
      "values" -> flag /@ shear[[3]],
      "exact" -> "core yield or damage flow at steps {20,40,60,80,100}"
    |>,
    "shear_epbar_st" -> <|
      "values" -> dec /@ shear[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "shear_temp" -> <|
      "values" -> {dec[shear[[5]]]},
      "exact" -> "Tref, Cv = 0 disables heating"
    |>,
    "shear_damage" -> <|
      "values" -> dec /@ shear[[6]],
      "exact" -> "[d1, d2] at rebuilt principals +-0.11: ramp, zero middle"
    |>,
    "shear_dmg_st" -> <|
      "values" -> dec /@ shear[[7]],
      "exact" -> "d1 at steps {20,40,60,80,100}, onset at step 80"
    |>,
    "vol_sigma" -> <|
      "values" -> dec /@ vol[[1]],
      "exact" -> "hydrostatic elastic, p = K 3 emaxVol"
    |>,
    "vol_epbar" -> <|
      "values" -> {dec[vol[[2]]]},
      "exact" -> "zero, J2 never yields hydrostatically"
    |>,
    "vol_yielding" -> <|
      "values" -> flag /@ vol[[3]],
      "exact" -> "all False on the compressive volumetric path"
    |>,
    "vol_epbar_st" -> <|
      "values" -> dec /@ vol[[4]],
      "exact" -> "all zero on the compressive volumetric path"
    |>,
    "vol_temp" -> <|
      "values" -> {dec[vol[[5]]]},
      "exact" -> "Tref, no plastic work on the volumetric path"
    |>,
    "vol_damage" -> <|
      "values" -> dec /@ vol[[6]],
      "exact" -> "[0, 0], damage is tension-only"
    |>,
    "vol_dmg_st" -> <|
      "values" -> dec /@ vol[[7]],
      "exact" -> "all zero on the compressive volumetric path"
    |>,
    "cycle_sigma" -> <|
      "values" -> dec /@ cyc[[1]],
      "exact" -> "brittle core return over 0->+emax->-emax->0, solved lateral"
    |>,
    "cycle_epbar" -> <|
      "values" -> {dec[cyc[[2]]]},
      "exact" -> "Sum booked dep over 400 cycle steps"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[3]],
      "exact" -> "core yield or damage flow at steps {80,160,240,320,400}"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[4]],
      "exact" -> "accumulated epbar at steps {80,160,240,320,400}"
    |>,
    "cycle_temp" -> <|
      "values" -> {dec[cyc[[5]]]},
      "exact" -> "Tref, Cv = 0 disables heating"
    |>,
    "cycle_damage" -> <|
      "values" -> dec /@ cyc[[6]],
      "exact" -> "[d1, d2] at the cycle end, damage follows the totals"
    |>,
    "cycle_dmg_st" -> <|
      "values" -> dec /@ cyc[[7]],
      "exact" -> "d1 at steps {80,160,240,320,400}"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "q - sig(ep) at the cycle end, ~0 on the surface, no lag"
    |>,
    "tension_sigma" -> <|
      "values" -> dec /@ tension[[1]],
      "exact" -> "zeroed past deletion at eps_f = 0.16"
    |>,
    "tension_epbar" -> <|
      "values" -> {dec[tension[[2]]]},
      "exact" -> "Sum booked dep over the constrained tension leg"
    |>,
    "tension_yielding" -> <|
      "values" -> flag /@ tension[[3]],
      "exact" -> "core yield, damage flow, then deletion at steps 20..100"
    |>,
    "tension_epbar_st" -> <|
      "values" -> dec /@ tension[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "tension_temp" -> <|
      "values" -> {dec[tension[[5]]]},
      "exact" -> "Tref, Cv = 0 disables heating"
    |>,
    "tension_damage" -> <|
      "values" -> dec /@ tension[[6]],
      "exact" -> "[dmax, 0] past eps_m in the axial direction"
    |>,
    "tension_dmg_st" -> <|
      "values" -> dec /@ tension[[7]],
      "exact" -> "d1 at steps {20,40,60,80,100}, saturating past eps_m"
    |>,
    "tension_ruptured" -> <|
      "values" -> flag /@ tension[[8]],
      "exact" -> "deletion latch at steps {20,40,60,80,100}"
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
    "uni_temp_abs" -> 0.0,
    "uni_temp_rel" -> 0.0,
    "uni_damage_abs" -> 1.0*^-12,
    "uni_damage_rel" -> 1.0*^-12,
    "uni_dmg_st_abs" -> 1.0*^-12,
    "uni_dmg_st_rel" -> 1.0*^-12,
    "shear_sigma_abs" -> 1.0*^-6,
    "shear_sigma_rel" -> 1.0*^-12,
    "shear_epbar_abs" -> 1.0*^-12,
    "shear_epbar_rel" -> 1.0*^-12,
    "shear_yielding_abs" -> 0.0,
    "shear_yielding_rel" -> 0.0,
    "shear_epbar_st_abs" -> 1.0*^-12,
    "shear_epbar_st_rel" -> 1.0*^-12,
    "shear_temp_abs" -> 0.0,
    "shear_temp_rel" -> 0.0,
    "shear_damage_abs" -> 1.0*^-12,
    "shear_damage_rel" -> 1.0*^-12,
    "shear_dmg_st_abs" -> 1.0*^-12,
    "shear_dmg_st_rel" -> 1.0*^-12,
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
    "vol_damage_abs" -> 0.0,
    "vol_damage_rel" -> 0.0,
    "vol_dmg_st_abs" -> 0.0,
    "vol_dmg_st_rel" -> 0.0,
    "cycle_sigma_abs" -> 1.0*^-6,
    "cycle_sigma_rel" -> 1.0*^-12,
    "cycle_epbar_abs" -> 1.0*^-12,
    "cycle_epbar_rel" -> 1.0*^-12,
    "cycle_yielding_abs" -> 0.0,
    "cycle_yielding_rel" -> 0.0,
    "cycle_epbar_st_abs" -> 1.0*^-12,
    "cycle_epbar_st_rel" -> 1.0*^-12,
    "cycle_temp_abs" -> 0.0,
    "cycle_temp_rel" -> 0.0,
    "cycle_damage_abs" -> 1.0*^-12,
    "cycle_damage_rel" -> 1.0*^-12,
    "cycle_dmg_st_abs" -> 1.0*^-12,
    "cycle_dmg_st_rel" -> 1.0*^-12,
    "cycle_closure_abs" -> 1.0*^-6,
    "cycle_closure_rel" -> 1.0*^-12,
    "tension_sigma_abs" -> 1.0*^-6,
    "tension_sigma_rel" -> 1.0*^-12,
    "tension_epbar_abs" -> 1.0*^-12,
    "tension_epbar_rel" -> 1.0*^-12,
    "tension_yielding_abs" -> 0.0,
    "tension_yielding_rel" -> 0.0,
    "tension_epbar_st_abs" -> 1.0*^-12,
    "tension_epbar_st_rel" -> 1.0*^-12,
    "tension_temp_abs" -> 0.0,
    "tension_temp_rel" -> 0.0,
    "tension_damage_abs" -> 1.0*^-12,
    "tension_damage_rel" -> 1.0*^-12,
    "tension_dmg_st_abs" -> 1.0*^-12,
    "tension_dmg_st_rel" -> 1.0*^-12,
    "tension_ruptured_abs" -> 0.0,
    "tension_ruptured_rel" -> 0.0
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
