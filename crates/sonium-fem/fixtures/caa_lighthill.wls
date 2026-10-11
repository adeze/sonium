(* Independent Wolfram reference for the sonium-fem compact Lighthill kernel.
   Derives, at 50-digit working precision, exact (radiating + reactive) and
   far-field-only pressures of an axial quadrupole at kr = 1, 10, 100, the
   transverse far-field null (pure reactive remainder), the quadrupole
   directional factor at five polar angles plus its sphere normalization,
   the 1/r spreading ratio, and a reciprocity delta. The Rust side consumes
   caa_lighthill_golden.json through lighthill_pressure,
   lighthill_far_field_pressure, directivity_factor, and
   lighthill_reciprocity_gap.
   Run: wolframscript -script caa_lighthill.wls > caa_lighthill_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86-64. *)

om = 1000; c = 340;
k = om/c;
tAxx = 2;
tMat = {{tAxx, 0, 0}, {0, 0, 0}, {0, 0, 0}};

green[R_] := Exp[-I k R]/(4 Pi R);
acoef[R_] := green[R] (-k^2 + 3/R^2 + I 3 k/R);
bcoef[R_] := green[R] (-1/R^2 - I k/R);
contract[n_] := n . tMat . n;
exactP[n_, R_] := acoef[R] contract[n] + bcoef[R] Tr[tMat];
farP[n_, R_] := -k^2 green[R] contract[n];

(* --- kr sweep on the quadrupole axis: 1/r^3 -> 1/r transition. --- *)
krCases = Table[
  Module[{R = kr/k, n = {1, 0, 0}, ex, fa},
    ex = exactP[n, R]; fa = farP[n, R];
    <|"kr" -> kr, "observer" -> R n,
      "exact" -> ex, "far" -> fa, "reactive" -> ex - fa|>
  ],
  {kr, {1, 10, 100}}
];

(* --- Transverse observer: far field nulls, exact is pure reactive. --- *)
rT = 100/k; nT = {0, 1, 0};
exT = exactP[nT, rT]; faT = farP[nT, rT];

(* --- Directivity D(th) = Txx cos^2 th and its sphere normalization. --- *)
angles = {0, Pi/6, Pi/4, Pi/3, Pi/2};
factors = Table[
  With[{n = {Cos[th], Sin[th], 0}}, contract[n]],
  {th, angles}
];
(* (4 pi)^-1 Integrate[|D|^2, sphere] = |Txx|^2/5 for Txx-only. *)
normAnalytic = Abs[tAxx]^2/5;

(* --- Spreading ratio of the radiating term (exactly 1/2). --- *)
rNear = 100/k; rFar = 200/k;
spreadRatio = Abs[farP[{1, 0, 0}, rFar]]/Abs[farP[{1, 0, 0}, rNear]];

(* --- Reciprocity delta (exactly zero). --- *)
pA = {3, 1/2, -1/5}; pB = {-1, 2, 7/10};
recDelta = Module[{dAB, dBA},
  dAB = pB - pA; dBA = pA - pB;
  exactP[dAB/Norm[dAB], Norm[dAB]] - exactP[dBA/Norm[dBA], Norm[dBA]]
];

prec = 50;
cPair[z_] := {N[Re[z], prec], N[Im[z], prec]};
rList[v_] := N[v, prec];
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.caa-lighthill.v1",
  "convention" -> "SI units; exp(i wt) harmonic; exp(-i k r) propagation; compact axial Lighthill quadrupole",
  "generator" -> "crates/sonium-fem/fixtures/caa_lighthill.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "angular_frequency" -> N[om, prec],
  "speed_of_sound" -> N[c, prec],
  "stress_moment" -> Map[cPair, N[tMat, prec], {2}],
  "kr_cases" -> Table[
    <|"kr" -> N[case["kr"], prec],
      "observer" -> rList[case["observer"]],
      "exact" -> cPair[case["exact"]],
      "far" -> cPair[case["far"]],
      "reactive" -> cPair[case["reactive"]]|>,
    {case, krCases}
  ],
  "transverse" -> <|
    "observer" -> rList[rT nT],
    "exact" -> cPair[exT],
    "far" -> cPair[faT]
  |>,
  "directivity" -> <|
    "angles_rad" -> rList[angles],
    "factors" -> cPair /@ factors,
    "normalization" -> N[normAnalytic, prec]
  |>,
  "spreading" -> <|
    "distance_near" -> N[rNear, prec],
    "distance_far" -> N[rFar, prec],
    "ratio" -> N[spreadRatio, prec]
  |>,
  "reciprocity" -> <|
    "point_a" -> rList[pA],
    "point_b" -> rList[pB],
    "delta" -> cPair[recDelta]
  |>,
  "tolerances" -> <|
    "pressure_abs" -> 1.0*^-9,
    "factor_abs" -> 1.0*^-12,
    "ratio_abs" -> 1.0*^-15,
    "norm_rel" -> 1.0*^-5
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
