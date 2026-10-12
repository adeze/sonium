(* Independent Wolfram reference for LAW51 multimat strain paths.
   Integrates, at 100-digit working precision, the two-material
   programmed-burn JWL mixture equations along four strain paths:
   uniaxial-strain tension (mu -> -0.02), pure shear (mu frozen,
   pinning the exactly-zero deviatoric response while burn mixes
   the pressure at constant volume), volumetric compression
   (mu -> +0.3), and a symmetric volumetric cycle
   0 -> +0.3 -> -0.3 -> 0. Each step evaluates the mixture
   pressure P = (1-F) Pu + F Pr at the trial volumetric strain
   with start-of-step energies and burn fraction, returns
   hydrostatic stress -(P - Pext) I, then advances the energies
   adiabatically (de = P dmu/(rho0 eta^2) per material) and ramps
   burn linearly over 200 steps. Detonation starts at t = 0, so
   100-step legs end half-burned and the 400-step cycle ends
   fully burned. All inputs are exact rationals converted once
   with N[..., 100]; the loop then runs at 100-digit working
   precision with no machine-precision literals, and N[..., 50]
   is applied once per exported value. The law is rate
   independent (no plasticity, no damage): yielding stays False
   and epbar stays zero on every station. The Rust side consumes
   solids_law51_paths_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_law51_paths.wls > fixtures/solids_law51_paths_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 100;
outprec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, outprec]]]];
flag[b_] := If[b, "1", "0"];
chk[x_, nm_] := If[Accuracy[x] < 55, Print["PRECISION COLLAPSE in ", nm]; Abort[]];
chkList[l_, nm_] := Do[chk[v, nm], {v, l}];

(* Exact inputs: unreacted JWL a = 2 GPa, reacted a = 1 GPa,
   b = 0.5 GPa, r1 = 4.5, r2 = 1.2, omega = 0.25, rho0 = 1500,
   e0 = 1e5 J/kg; no ambient pressure; burn over 200 steps. *)
aUX = 2*10^9; aRX = 1*10^9;
bX = 5*10^8;
r1X = 45/10; r2X = 12/10;
omX = 1/4;
rhoX = 1500;
e0X = 10^5;
pextX = 0;
tdetX = 0;
tburnStepsX = 200;
dtX = 10^-6;
deX = 2*10^-4;
dgX = 2*10^-3;
dvX = 10^-3;
nLegX = 100;

aU = N[aUX, prec]; aR = N[aRX, prec];
bJ = N[bX, prec];
r1 = N[r1X, prec]; r2 = N[r2X, prec];
om = N[omX, prec];
rho0 = N[rhoX, prec];
e0 = N[e0X, prec];
pext = N[pextX, prec];
tdet = N[tdetX, prec];
dt = N[dtX, prec];
tburn = tburnStepsX*dt;
de = N[deX, prec];
dg = N[dgX, prec];
dv = N[dvX, prec];
nLeg = nLegX;
etaFloor = N[10^-12, prec];

(* Compression ratio with the production floor. *)
etaOf[mu_] := Max[1 + mu, etaFloor];

(* JWL pressure at volumetric strain and specific energy. *)
jwlP[a_, mu_, e_] := Module[{eta},
  eta = etaOf[mu];
  a*(1 - om*eta/r1)*Exp[-r1/eta]
    + bJ*(1 - om*eta/r2)*Exp[-r2/eta]
    + om*rho0*eta*e
];

(* Programmed burn fraction at time. *)
burnAt[t_] := Min[Max[(t - tdet)/tburn, 0], 1];

(* Mixture pressure at trial mu with start-of-step energies/burn. *)
mixP[mu_, eu_, er_, burn_] :=
  (1 - burn)*jwlP[aU, mu, eu] + burn*jwlP[aR, mu, er];

(* Drive a fixed-increment path; stations list 1-based step indices. *)
runFixed[steps_, stations_] := Module[{mu, eu, er, burn, time, press,
  i, d, dmu, muT, etaT, pu, pr, yl, epst, net},
  mu = 0; eu = e0; er = e0; burn = 0; time = 0; press = 0;
  yl = {}; epst = {};
  Do[d = steps[[i]];
    dmu = -(d[[1]] + d[[2]] + d[[3]]);
    muT = mu + dmu;
    press = mixP[muT, eu, er, burn];
    etaT = etaOf[muT];
    pu = jwlP[aU, muT, eu];
    pr = jwlP[aR, muT, er];
    eu = eu + pu/(rho0*etaT^2)*dmu;
    er = er + pr/(rho0*etaT^2)*dmu;
    mu = muT;
    time = time + dt;
    burn = burnAt[time];
    If[MemberQ[stations, i], AppendTo[yl, False]; AppendTo[epst, 0]],
    {i, 1, Length[steps]}];
  net = press - pext;
  {{-net, -net, -net, 0, 0, 0}, press, mu, burn, eu, er, yl, epst}
];

st100 = {20, 40, 60, 80, 100};
stCyc = {80, 160, 240, 320, 400};

uni = runFixed[Table[{de, 0, 0, 0, 0, 0}, {nLeg}], st100];
shear = runFixed[Table[{0, 0, 0, dg, 0, 0}, {nLeg}], st100];
vol = runFixed[Table[{-dv, -dv, -dv, 0, 0, 0}, {nLeg}], st100];
cycDvs = Join[Table[-dv, {nLeg}], Table[dv, {2*nLeg}], Table[-dv, {nLeg}]];
cyc = runFixed[Table[{cycDvs[[i]], cycDvs[[i]], cycDvs[[i]], 0, 0, 0},
  {i, 1, Length[cycDvs]}], stCyc];

(* Adiabatic hysteresis: end pressure minus the fully-burned
   virgin pressure (burn, not hysteresis, moves the baseline). *)
prVirgin = jwlP[aR, 0, e0];
closure = cyc[[2]] - prVirgin;

(* Precision tripwire on every exported state value. *)
chkList[uni[[1]], "uni_sigma"];
chk[uni[[2]], "uni_pressure"];
chk[uni[[3]], "uni_mu"];
chk[uni[[4]], "uni_burn"];
chk[uni[[5]], "uni_eu"];
chk[uni[[6]], "uni_er"];
chkList[shear[[1]], "shear_sigma"];
chk[shear[[2]], "shear_pressure"];
chk[shear[[3]], "shear_mu"];
chk[shear[[4]], "shear_burn"];
chk[shear[[5]], "shear_eu"];
chk[shear[[6]], "shear_er"];
chkList[vol[[1]], "vol_sigma"];
chk[vol[[2]], "vol_pressure"];
chk[vol[[3]], "vol_mu"];
chk[vol[[4]], "vol_burn"];
chk[vol[[5]], "vol_eu"];
chk[vol[[6]], "vol_er"];
chkList[cyc[[1]], "cycle_sigma"];
chk[cyc[[2]], "cycle_pressure"];
chk[cyc[[3]], "cycle_mu"];
chk[cyc[[4]], "cycle_burn"];
chk[cyc[[5]], "cycle_eu"];
chk[cyc[[6]], "cycle_er"];
chk[closure, "cycle_closure"];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law51_paths",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "a_unreacted_pa" -> N[aUX],
    "a_reacted_pa" -> N[aRX],
    "b_pa" -> N[bX],
    "r1" -> N[r1X],
    "r2" -> N[r2X],
    "omega" -> N[omX],
    "rho0" -> N[rhoX],
    "e0" -> N[e0X],
    "p_ext_pa" -> N[pextX],
    "burn_steps" -> tburnStepsX,
    "dt_s" -> N[dtX],
    "de_tension" -> N[deX],
    "dg_shear" -> N[dgX],
    "dv_vol" -> N[dvX],
    "steps_per_leg" -> nLegX,
    "stations_100" -> st100,
    "stations_cycle" -> stCyc
  |>,
  "values" -> <|
    "uni_sigma" -> <|
      "values" -> dec /@ uni[[1]],
      "exact" -> "hydrostatic -(P-Pext) I at mu = -0.02, half burn"
    |>,
    "uni_pressure" -> <|
      "values" -> {dec[uni[[2]]]},
      "exact" -> "mixture pressure at the tension end"
    |>,
    "uni_mu" -> <|
      "values" -> {dec[uni[[3]]]},
      "exact" -> "volumetric strain -0.02 at the tension end"
    |>,
    "uni_burn" -> <|
      "values" -> {dec[uni[[4]]]},
      "exact" -> "burn fraction 1/2 at the tension end"
    |>,
    "uni_eu" -> <|
      "values" -> {dec[uni[[5]]]},
      "exact" -> "unreacted adiabatic energy at the tension end"
    |>,
    "uni_er" -> <|
      "values" -> {dec[uni[[6]]]},
      "exact" -> "reacted adiabatic energy at the tension end"
    |>,
    "uni_yielding" -> <|
      "values" -> flag /@ uni[[7]],
      "exact" -> "all False, the mixture never yields"
    |>,
    "uni_epbar_st" -> <|
      "values" -> dec /@ uni[[8]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "shear_sigma" -> <|
      "values" -> dec /@ shear[[1]],
      "exact" -> "hydrostatic only, shear slots exactly zero"
    |>,
    "shear_pressure" -> <|
      "values" -> {dec[shear[[2]]]},
      "exact" -> "burn-mixed pressure at mu = 0"
    |>,
    "shear_mu" -> <|
      "values" -> {dec[shear[[3]]]},
      "exact" -> "zero, shear carries no volume"
    |>,
    "shear_burn" -> <|
      "values" -> {dec[shear[[4]]]},
      "exact" -> "burn fraction 1/2 at the shear end"
    |>,
    "shear_eu" -> <|
      "values" -> {dec[shear[[5]]]},
      "exact" -> "e0, no volumetric work under shear"
    |>,
    "shear_er" -> <|
      "values" -> {dec[shear[[6]]]},
      "exact" -> "e0, no volumetric work under shear"
    |>,
    "shear_yielding" -> <|
      "values" -> flag /@ shear[[7]],
      "exact" -> "all False, the mixture never yields"
    |>,
    "shear_epbar_st" -> <|
      "values" -> dec /@ shear[[8]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "vol_sigma" -> <|
      "values" -> dec /@ vol[[1]],
      "exact" -> "hydrostatic -(P-Pext) I at mu = +0.3, half burn"
    |>,
    "vol_pressure" -> <|
      "values" -> {dec[vol[[2]]]},
      "exact" -> "mixture pressure at the crush end"
    |>,
    "vol_mu" -> <|
      "values" -> {dec[vol[[3]]]},
      "exact" -> "volumetric strain +0.3 at the crush end"
    |>,
    "vol_burn" -> <|
      "values" -> {dec[vol[[4]]]},
      "exact" -> "burn fraction 1/2 at the crush end"
    |>,
    "vol_eu" -> <|
      "values" -> {dec[vol[[5]]]},
      "exact" -> "unreacted adiabatic energy at the crush end"
    |>,
    "vol_er" -> <|
      "values" -> {dec[vol[[6]]]},
      "exact" -> "reacted adiabatic energy at the crush end"
    |>,
    "vol_yielding" -> <|
      "values" -> flag /@ vol[[7]],
      "exact" -> "all False, the mixture never yields"
    |>,
    "vol_epbar_st" -> <|
      "values" -> dec /@ vol[[8]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "cycle_sigma" -> <|
      "values" -> dec /@ cyc[[1]],
      "exact" -> "hydrostatic at mu = 0, full burn, cycled energies"
    |>,
    "cycle_pressure" -> <|
      "values" -> {dec[cyc[[2]]]},
      "exact" -> "mixture pressure at the cycle end"
    |>,
    "cycle_mu" -> <|
      "values" -> {dec[cyc[[3]]]},
      "exact" -> "zero, the volume cycle closes"
    |>,
    "cycle_burn" -> <|
      "values" -> {dec[cyc[[4]]]},
      "exact" -> "burn fraction 1 at the cycle end"
    |>,
    "cycle_eu" -> <|
      "values" -> {dec[cyc[[5]]]},
      "exact" -> "unreacted adiabatic energy at the cycle end"
    |>,
    "cycle_er" -> <|
      "values" -> {dec[cyc[[6]]]},
      "exact" -> "reacted adiabatic energy at the cycle end"
    |>,
    "cycle_yielding" -> <|
      "values" -> flag /@ cyc[[7]],
      "exact" -> "all False, the mixture never yields"
    |>,
    "cycle_epbar_st" -> <|
      "values" -> dec /@ cyc[[8]],
      "exact" -> "all zero, no plastic flow"
    |>,
    "cycle_closure" -> <|
      "values" -> {dec[closure]},
      "exact" -> "P_end - Pr(0, e0), adiabatic hysteresis"
    |>
  |>,
  "tolerances" -> <|
    "uni_sigma_abs" -> 1.0*^-6,
    "uni_sigma_rel" -> 1.0*^-12,
    "uni_pressure_abs" -> 1.0*^-6,
    "uni_pressure_rel" -> 1.0*^-12,
    "uni_mu_abs" -> 1.0*^-12,
    "uni_mu_rel" -> 1.0*^-12,
    "uni_burn_abs" -> 1.0*^-12,
    "uni_burn_rel" -> 1.0*^-12,
    "uni_eu_abs" -> 1.0*^-6,
    "uni_eu_rel" -> 1.0*^-9,
    "uni_er_abs" -> 1.0*^-6,
    "uni_er_rel" -> 1.0*^-9,
    "uni_yielding_abs" -> 0.0,
    "uni_yielding_rel" -> 0.0,
    "uni_epbar_st_abs" -> 0.0,
    "uni_epbar_st_rel" -> 0.0,
    "shear_sigma_abs" -> 1.0*^-6,
    "shear_sigma_rel" -> 1.0*^-12,
    "shear_pressure_abs" -> 1.0*^-6,
    "shear_pressure_rel" -> 1.0*^-12,
    "shear_mu_abs" -> 0.0,
    "shear_mu_rel" -> 0.0,
    "shear_burn_abs" -> 1.0*^-12,
    "shear_burn_rel" -> 1.0*^-12,
    "shear_eu_abs" -> 1.0*^-9,
    "shear_eu_rel" -> 1.0*^-12,
    "shear_er_abs" -> 1.0*^-9,
    "shear_er_rel" -> 1.0*^-12,
    "shear_yielding_abs" -> 0.0,
    "shear_yielding_rel" -> 0.0,
    "shear_epbar_st_abs" -> 0.0,
    "shear_epbar_st_rel" -> 0.0,
    "vol_sigma_abs" -> 1.0*^-6,
    "vol_sigma_rel" -> 1.0*^-12,
    "vol_pressure_abs" -> 1.0*^-6,
    "vol_pressure_rel" -> 1.0*^-12,
    "vol_mu_abs" -> 1.0*^-12,
    "vol_mu_rel" -> 1.0*^-12,
    "vol_burn_abs" -> 1.0*^-12,
    "vol_burn_rel" -> 1.0*^-12,
    "vol_eu_abs" -> 1.0*^-6,
    "vol_eu_rel" -> 1.0*^-9,
    "vol_er_abs" -> 1.0*^-6,
    "vol_er_rel" -> 1.0*^-9,
    "vol_yielding_abs" -> 0.0,
    "vol_yielding_rel" -> 0.0,
    "vol_epbar_st_abs" -> 0.0,
    "vol_epbar_st_rel" -> 0.0,
    "cycle_sigma_abs" -> 1.0*^-6,
    "cycle_sigma_rel" -> 1.0*^-12,
    "cycle_pressure_abs" -> 1.0*^-6,
    "cycle_pressure_rel" -> 1.0*^-12,
    "cycle_mu_abs" -> 1.0*^-12,
    "cycle_mu_rel" -> 1.0*^-12,
    "cycle_burn_abs" -> 0.0,
    "cycle_burn_rel" -> 0.0,
    "cycle_eu_abs" -> 1.0*^-6,
    "cycle_eu_rel" -> 1.0*^-9,
    "cycle_er_abs" -> 1.0*^-6,
    "cycle_er_rel" -> 1.0*^-9,
    "cycle_yielding_abs" -> 0.0,
    "cycle_yielding_rel" -> 0.0,
    "cycle_epbar_st_abs" -> 0.0,
    "cycle_epbar_st_rel" -> 0.0,
    "cycle_closure_abs" -> 1.0*^-6,
    "cycle_closure_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
