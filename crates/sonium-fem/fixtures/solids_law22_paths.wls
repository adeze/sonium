(* Independent Wolfram reference for LAW22 ductile-damage strain paths.
   Integrates, at 200-digit working precision, the von Mises radial-return
   equations with the card-aluminum perfectly-plastic core (100 MPa) and
   explicit ductile damage along six strain paths: uniaxial extension
   with the lateral strain solved per step for zero lateral stress
   (bisection, 50 passes), pure shear (gamma_xy ramp), volumetric
   (eps I ramp), a symmetric axial cycle 0 -> +emax -> -emax -> 0 with
   the same lateral solve, hydrostatic compression, and constrained
   uniaxial extension. The state is carried in deviatoric/hydrostatic
   split form (textbook formulation); the plastic multiplier is solved
   in closed form (dep = (qTrial - (1-d) S)/(3G_d), exact for the
   linear residual), independently of the bisection the Rust
   return-mapping uses. Damage degrades the moduli
   (E_d = (1-d) E, nu_d = d/2 + (1-d) nu, bulk preserved) and the
   surface ((1-d) S), and integrates explicitly from the end-of-step
   plastic strain; past saturation (d = 1) the point carries pressure
   on the undamaged bulk modulus with frozen plastic strain. All inputs
   are exact rationals converted once with N[..., 200]; the loop then
   runs at 200-digit working precision with no machine-precision
   literals, and N[..., 50] is applied once per exported value.
   Stations record the yielding flag and accumulated epbar at fifths of
   each path. The Rust side consumes solids_law22_paths_golden.json;
   this script is the derivation.
   Regen: wolframscript -script fixtures/solids_law22_paths.wls > fixtures/solids_law22_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 200;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact card inputs: aluminum, E = 70 GPa, nu = 0.3, perfectly plastic
   A = 100 MPa (B = 0), cap 100 MPa, rupture at ep = 0.2, damage from
   ep = 0.1 with softening slope Et = -2000 MPa, no rate effects. *)
eX = 70*10^9;
nuX = 3/10;
aX = 100*10^6;
capX = 100*10^6;
rupX = 2/10;
dmgX = 1/10;
etX = -2000*10^6;
emaxX = 12/100;
nLegX = 100;
latX = 50;

E0 = N[eX, prec];
nu0 = N[nuX, prec];
S0 = N[aX, prec];
cap0 = N[capX, prec];
rup0 = N[rupX, prec];
dmgEp = N[dmgX, prec];
Et = N[etX, prec];
Kbulk = E0/(3*(1 - 2*nu0));
emax = N[emaxX, prec];
nLeg = nLegX;
latIt = latX;

(* Damaged moduli (bulk exactly preserved) and strength. *)
damE[d_] := (1 - d)*E0;
damNu[d_] := (1/2)*d + (1 - d)*nu0;
damG[d_] := damE[d]/(2*(1 + damNu[d]));
damK[d_] := damE[d]/(3*(1 - 2*damNu[d]));
damS[d_] := (1 - d)*S0;

(* Softening slope in plastic-strain space (negative). *)
etEp = Et/(1 + (-Et/E0));

(* Trial damage after a plastic increment (explicit Euler on d).
   The perfectly-plastic core has zero hardening slope, so the rate is
   -Et_ep/S against the undamaged base strength. *)
trialDamage[epNew_, dep_, dOld_] := Module[{rateD},
  If[epNew <= dmgEp || dep <= 0 || Et >= 0, Return[dOld]];
  rateD = -etEp/S0;
  Min[Max[dOld + rateD*dep, 0], 1]
];

(* Von Mises norm from a deviatoric C1 tensor (exact 3, 1/2). *)
vonMisesDev[dev_] := Sqrt[3*((dev[[1]]^2 + dev[[2]]^2 + dev[[3]]^2)/2
  + dev[[4]]^2 + dev[[5]]^2 + dev[[6]]^2)];

(* One LAW22 trial step in split form: {devNew, pNew, dep, yielding}.
   Saturated points (d = 1) ride pressure only with frozen ep. *)
j2step[dev_, p_, ep_, dmg_, d_] := Module[{vol, pT, devT, qT, sy, dep, ratio, g2, g3},
  If[dmg >= 1 - 10^-12,
    vol = d[[1]] + d[[2]] + d[[3]];
    pT = p + Kbulk*vol;
    Return[{{0, 0, 0, 0, 0, 0}, pT, 0, True}]];
  g2 = 2*damG[dmg];
  g3 = 3*damG[dmg];
  vol = d[[1]] + d[[2]] + d[[3]];
  pT = p + damK[dmg]*vol;
  devT = {dev[[1]] + g2*(d[[1]] - vol/3),
    dev[[2]] + g2*(d[[2]] - vol/3),
    dev[[3]] + g2*(d[[3]] - vol/3),
    dev[[4]] + g2*(1/2)*d[[4]],
    dev[[5]] + g2*(1/2)*d[[5]],
    dev[[6]] + g2*(1/2)*d[[6]]};
  qT = vonMisesDev[devT];
  sy = damS[dmg];
  If[qT <= sy,
    {devT, pT, 0, False},
    dep = (qT - sy)/g3;
    If[ep + dep >= rup0, Print["RUPTURE"]; Abort[]];
    ratio = 1 - g3*dep/qT;
    {ratio*devT, pT, dep, True}
  ]
];

(* Full C1 stress from split state (export and probes only). *)
sigOf[dev_, p_] := {dev[[1]] + p, dev[[2]] + p, dev[[3]] + p,
  dev[[4]], dev[[5]], dev[[6]]};

(* Lateral trial stress of one axial step (bisection probe). *)
stepYY[dev_, p_, ep_, dmg_, de_, dl_] := Module[{u},
  u = j2step[dev, p, ep, dmg, {de, dl, dl, 0, 0, 0}];
  u[[1, 2]] + u[[2]]
];

(* Lateral strain for zero lateral stress (bisection, latIt passes). *)
solveDL[dev_, p_, ep_, dmg_, de_] := Module[{lo, hi, mid, blo, bhi, w, k},
  If[de >= 0, lo = -de; hi = 0, lo = 0; hi = -de];
  blo = stepYY[dev, p, ep, dmg, de, lo];
  bhi = stepYY[dev, p, ep, dmg, de, hi];
  For[k = 1, k <= 20, k++,
    If[blo < 0 && bhi > 0, Break[]];
    w = Abs[hi - lo]; lo = lo - w; hi = hi + w;
    blo = stepYY[dev, p, ep, dmg, de, lo];
    bhi = stepYY[dev, p, ep, dmg, de, hi];
  ];
  If[! (blo < 0 && bhi > 0), Print["BRACKET FAILURE"]; Abort[]];
  Do[mid = (lo + hi)/2;
    If[stepYY[dev, p, ep, dmg, de, mid] > 0, hi = mid, lo = mid],
    {latIt}];
  (lo + hi)/2
];

(* Drive a fixed-increment path; stations list 1-based step indices. *)
runFixed[steps_, stations_] := Module[{dev, p, ep, dmg, i, d, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0; dmg = 0;
  yl = {}; epst = {};
  Do[d = steps[[i]];
    u = j2step[dev, p, ep, dmg, d];
    dev = u[[1]]; p = u[[2]]; ep = ep + u[[3]];
    dmg = trialDamage[ep, u[[3]], dmg];
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[steps]}];
  {sigOf[dev, p], ep, yl, epst, dmg}
];

(* Drive an axial path with the lateral solve. *)
runAxial[des_, stations_] := Module[{dev, p, ep, dmg, i, de, dl, u, yl, epst},
  dev = {0, 0, 0, 0, 0, 0}; p = 0; ep = 0; dmg = 0;
  yl = {}; epst = {};
  Do[de = des[[i]];
    dl = solveDL[dev, p, ep, dmg, de];
    u = j2step[dev, p, ep, dmg, {de, dl, dl, 0, 0, 0}];
    dev = u[[1]]; p = u[[2]]; ep = ep + u[[3]];
    dmg = trialDamage[ep, u[[3]], dmg];
    If[MemberQ[stations, i], AppendTo[yl, u[[4]]]; AppendTo[epst, ep]],
    {i, 1, Length[des]}];
  {sigOf[dev, p], ep, yl, epst, dmg}
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
hydro = runFixed[Table[{-dv, -dv, -dv, 0, 0, 0}, {nLeg}], st100];
constr = runFixed[Table[{de, 0, 0, 0, 0, 0}, {nLeg}], st100];

cycSig = cyc[[1]]; cycEp = cyc[[2]]; cycDmg = cyc[[5]];
pCyc = (cycSig[[1]] + cycSig[[2]] + cycSig[[3]])/3;
devCyc = {cycSig[[1]] - pCyc, cycSig[[2]] - pCyc, cycSig[[3]] - pCyc,
  cycSig[[4]], cycSig[[5]], cycSig[[6]]};
closure = vonMisesDev[devCyc] - (1 - cycDmg)*S0;

(* Precision tripwire: every exported state value must keep 55 digits
   of absolute accuracy (closure excluded: the saturated cycle ends at
   zero strength and legitimately cancels). *)
chkList[uni[[1]], "uni_sigma"];
chk[uni[[2]], "uni_epbar"];
chkList[uni[[4]], "uni_epbar_st"];
chk[uni[[5]], "uni_damage"];
chkList[shear[[1]], "shear_sigma"];
chk[shear[[2]], "shear_epbar"];
chkList[shear[[4]], "shear_epbar_st"];
chk[shear[[5]], "shear_damage"];
chkList[vol[[1]], "vol_sigma"];
chk[vol[[2]], "vol_epbar"];
chkList[vol[[4]], "vol_epbar_st"];
chk[vol[[5]], "vol_damage"];
chkList[cyc[[1]], "cycle_sigma"];
chk[cyc[[2]], "cycle_epbar"];
chkList[cyc[[4]], "cycle_epbar_st"];
chk[cyc[[5]], "cycle_damage"];
chkList[hydro[[1]], "hydro_sigma"];
chk[hydro[[2]], "hydro_epbar"];
chkList[hydro[[4]], "hydro_epbar_st"];
chk[hydro[[5]], "hydro_damage"];
chkList[constr[[1]], "constr_sigma"];
chk[constr[[2]], "constr_epbar"];
chkList[constr[[4]], "constr_epbar_st"];
chk[constr[[5]], "constr_damage"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law22_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_pa" -> N[eX],
    "poisson" -> N[nuX],
    "strength_pa" -> N[aX],
    "rupture_strain" -> N[rupX],
    "damage_strain" -> N[dmgX],
    "softening_slope_pa" -> N[etX],
    "emax" -> N[emaxX],
    "steps_per_leg" -> nLegX,
    "lateral_iterations" -> latX,
    "stations_100" -> st100,
    "stations_cycle" -> stCyc
  |>,
  "values" -> <|
    "uni_sigma" -> <|
      "values" -> dec /@ uni[[1]],
      "exact" -> "damaged radial return, lateral solved per step"
    |>,
    "uni_epbar" -> <|
      "values" -> {dec[uni[[2]]]},
      "exact" -> "Sum dep over 100 solved uniaxial steps"
    |>,
    "uni_yielding" -> <|
      "values" -> flag /@ uni[[3]],
      "exact" -> "qTrial > (1-d) S at steps {20,40,60,80,100}"
    |>,
    "uni_epbar_st" -> <|
      "values" -> dec /@ uni[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "uni_damage" -> <|
      "values" -> {dec[uni[[5]]]},
      "exact" -> "explicit damage at the uniaxial end, partial"
    |>,
    "shear_sigma" -> <|
      "values" -> dec /@ shear[[1]],
      "exact" -> "radial return under gamma_xy ramp, below damage onset"
    |>,
    "shear_epbar" -> <|
      "values" -> {dec[shear[[2]]]},
      "exact" -> "Sum dep over 100 shear steps"
    |>,
    "shear_yielding" -> <|
      "values" -> flag /@ shear[[3]],
      "exact" -> "qTrial > S at steps {20,40,60,80,100}"
    |>,
    "shear_epbar_st" -> <|
      "values" -> dec /@ shear[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "shear_damage" -> <|
      "values" -> {dec[shear[[5]]]},
      "exact" -> "zero, shear stays below the damage strain"
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
    "vol_damage" -> <|
      "values" -> {dec[vol[[5]]]},
      "exact" -> "zero, no plastic flow on the volumetric path"
    |>,
    "cycle_sigma" -> <|
      "values" -> dec /@ cyc[[1]],
      "exact" -> "damaged return over 0->+emax->-emax->0, solved lateral"
    |>,
    "cycle_epbar" -> <|
      "values" -> {dec[cyc[[2]]]},
      "exact" -> "Sum dep over 400 cycle steps, frozen at saturation"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[3]],
      "exact" -> "yielding at steps {80,160,240,320,400}"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[4]],
      "exact" -> "accumulated epbar at steps {80,160,240,320,400}"
    |>,
    "cycle_damage" -> <|
      "values" -> {dec[cyc[[5]]]},
      "exact" -> "1, the cycle saturates damage"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "q - (1-d) S at the cycle end, zero at saturation"
    |>,
    "hydro_sigma" -> <|
      "values" -> dec /@ hydro[[1]],
      "exact" -> "hydrostatic elastic compression, p = -K 3 emax"
    |>,
    "hydro_epbar" -> <|
      "values" -> {dec[hydro[[2]]]},
      "exact" -> "zero, J2 never yields hydrostatically"
    |>,
    "hydro_yielding" -> <|
      "values" -> flag /@ hydro[[3]],
      "exact" -> "all False on the compression leg"
    |>,
    "hydro_epbar_st" -> <|
      "values" -> dec /@ hydro[[4]],
      "exact" -> "all zero on the compression leg"
    |>,
    "hydro_damage" -> <|
      "values" -> {dec[hydro[[5]]]},
      "exact" -> "zero, no plastic flow on the compression leg"
    |>,
    "constr_sigma" -> <|
      "values" -> dec /@ constr[[1]],
      "exact" -> "radial return under laterally constrained extension"
    |>,
    "constr_epbar" -> <|
      "values" -> {dec[constr[[2]]]},
      "exact" -> "Sum dep over 100 constrained steps"
    |>,
    "constr_yielding" -> <|
      "values" -> flag /@ constr[[3]],
      "exact" -> "qTrial > S at steps {20,40,60,80,100}"
    |>,
    "constr_epbar_st" -> <|
      "values" -> dec /@ constr[[4]],
      "exact" -> "accumulated epbar at steps {20,40,60,80,100}"
    |>,
    "constr_damage" -> <|
      "values" -> {dec[constr[[5]]]},
      "exact" -> "zero, constrained path stays below damage onset"
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
    "uni_damage_abs" -> 1.0*^-12,
    "uni_damage_rel" -> 1.0*^-12,
    "shear_sigma_abs" -> 1.0*^-6,
    "shear_sigma_rel" -> 1.0*^-12,
    "shear_epbar_abs" -> 1.0*^-12,
    "shear_epbar_rel" -> 1.0*^-12,
    "shear_yielding_abs" -> 0.0,
    "shear_yielding_rel" -> 0.0,
    "shear_epbar_st_abs" -> 1.0*^-12,
    "shear_epbar_st_rel" -> 1.0*^-12,
    "shear_damage_abs" -> 0.0,
    "shear_damage_rel" -> 0.0,
    "vol_sigma_abs" -> 1.0*^-6,
    "vol_sigma_rel" -> 1.0*^-12,
    "vol_epbar_abs" -> 0.0,
    "vol_epbar_rel" -> 0.0,
    "vol_yielding_abs" -> 0.0,
    "vol_yielding_rel" -> 0.0,
    "vol_epbar_st_abs" -> 0.0,
    "vol_epbar_st_rel" -> 0.0,
    "vol_damage_abs" -> 0.0,
    "vol_damage_rel" -> 0.0,
    "cycle_sigma_abs" -> 1.0*^-6,
    "cycle_sigma_rel" -> 1.0*^-12,
    "cycle_epbar_abs" -> 1.0*^-12,
    "cycle_epbar_rel" -> 1.0*^-12,
    "cycle_yielding_abs" -> 0.0,
    "cycle_yielding_rel" -> 0.0,
    "cycle_epbar_st_abs" -> 1.0*^-12,
    "cycle_epbar_st_rel" -> 1.0*^-12,
    "cycle_damage_abs" -> 0.0,
    "cycle_damage_rel" -> 0.0,
    "cycle_closure_abs" -> 1.0*^-9,
    "cycle_closure_rel" -> 1.0*^-9,
    "hydro_sigma_abs" -> 1.0*^-6,
    "hydro_sigma_rel" -> 1.0*^-12,
    "hydro_epbar_abs" -> 0.0,
    "hydro_epbar_rel" -> 0.0,
    "hydro_yielding_abs" -> 0.0,
    "hydro_yielding_rel" -> 0.0,
    "hydro_epbar_st_abs" -> 0.0,
    "hydro_epbar_st_rel" -> 0.0,
    "hydro_damage_abs" -> 0.0,
    "hydro_damage_rel" -> 0.0,
    "constr_sigma_abs" -> 1.0*^-6,
    "constr_sigma_rel" -> 1.0*^-12,
    "constr_epbar_abs" -> 1.0*^-12,
    "constr_epbar_rel" -> 1.0*^-12,
    "constr_yielding_abs" -> 0.0,
    "constr_yielding_rel" -> 0.0,
    "constr_epbar_st_abs" -> 1.0*^-12,
    "constr_epbar_st_rel" -> 1.0*^-12,
    "constr_damage_abs" -> 0.0,
    "constr_damage_rel" -> 0.0
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
