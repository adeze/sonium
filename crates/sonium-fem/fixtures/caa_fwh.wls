(* Independent Wolfram reference for the sonium-fem planar FW-H kernel.
   Derives, at 50-digit working precision, single-segment thickness /
   loading / total pressures with the aero-convention Hankel kernel
   G = -i/4 H0^(2)(kr) for a loading-only, a thickness-only, and a
   combined oblique fixture; a closed four-segment square contour fed
   with line-source data; and the Hankel spreading ratio at kr = 100.
   The Rust side consumes caa_fwh_golden.json through
   stationary_fwh_pressure_2d.
   Run: wolframscript -script caa_fwh.wls > caa_fwh_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86-64. *)

om = 1000; rho = 12/10; c = 340;
k = om/c;

(* Aero convention: conjugate of the exp(-iwt) wave-crate kernel. *)
green[R_] := -I/4 HankelH2[0, k R];
dgreen[proj_, R_] := -(I k/4) HankelH2[1, k R] proj;
thick[vn_, R_, L_] := I om rho vn green[R] L;
load[p_, proj_, R_, L_] := p dgreen[proj, R] L;

segEval[pos_, nrm_, L_, p_, vn_, obs_] := Module[{R, proj, th, lo},
  R = Norm[obs - pos];
  proj = ((obs - pos)/R) . nrm;
  th = thick[vn, R, L]; lo = load[p, proj, R, L];
  <|"observer" -> obs, "pressure" -> p, "normal_velocity" -> vn,
    "thickness" -> th, "loading" -> lo, "total" -> th + lo|>
];

(* --- Single-segment fixtures (cf. fwh_monopole_patch). --- *)
pos = {1, 0}; nrm = {1, 0}; area = 2;
caseA = segEval[pos, nrm, area, 3/2 - I/2, 0, {4, 0}];
caseB = segEval[pos, nrm, area, 0, 1/100 + 2 I/100, {4, 0}];
caseC = segEval[pos, nrm, area, 1 + I, 1/50 - I/100, {4, 1}];

(* --- Closed square contour with line-source data. --- *)
lineP[R_] := green[R];
lineVn[R_] := ((I k/4) HankelH2[1, k R])/(-I om rho);
squareSegs = {
  <|"position" -> {1, 0}, "outward_normal" -> {1, 0}, "length" -> 2|>,
  <|"position" -> {-1, 0}, "outward_normal" -> {-1, 0}, "length" -> 2|>,
  <|"position" -> {0, 1}, "outward_normal" -> {0, 1}, "length" -> 2|>,
  <|"position" -> {0, -1}, "outward_normal" -> {0, -1}, "length" -> 2|>
};
obsD = {3, 1/2};
caseDsegs = Table[
  With[{R = Norm[seg["position"]]},
    segEval[seg["position"], seg["outward_normal"], seg["length"],
      lineP[R], lineVn[R], obsD]],
  {seg, squareSegs}
];
caseDtot = Total[#["total"] & /@ caseDsegs];
caseDthi = Total[#["thickness"] & /@ caseDsegs];
caseDloa = Total[#["loading"] & /@ caseDsegs];

(* --- Hankel spreading ratio at kr = 100. --- *)
rNear = 100/k; rFar = 200/k;
spreadRatio = Abs[green[rFar]]/Abs[green[rNear]];

prec = 50;
cPair[z_] := {N[Re[z], prec], N[Im[z], prec]};
rList[v_] := N[v, prec];
caseJson[case_] := <|
  "observer" -> rList[case["observer"]],
  "pressure" -> cPair[case["pressure"]],
  "normal_velocity" -> cPair[case["normal_velocity"]],
  "thickness" -> cPair[case["thickness"]],
  "loading" -> cPair[case["loading"]],
  "total" -> cPair[case["total"]]
|>;
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.caa-fwh-2d.v1",
  "convention" -> "SI units; exp(i wt) harmonic; planar FW-H contour; Hankel kernel G = -i/4 H0^(2)(kr)",
  "generator" -> "crates/sonium-fem/fixtures/caa_fwh.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "angular_frequency" -> N[om, prec],
  "density" -> N[rho, prec],
  "speed_of_sound" -> N[c, prec],
  "segment" -> <|
    "position" -> rList[pos],
    "outward_normal" -> rList[nrm],
    "length" -> N[area, prec]
  |>,
  "case_a" -> caseJson[caseA],
  "case_b" -> caseJson[caseB],
  "case_c" -> caseJson[caseC],
  "case_d" -> <|
    "observer" -> rList[obsD],
    "segments" -> Table[
      <|"position" -> rList[squareSegs[[i]]["position"]],
        "outward_normal" -> rList[squareSegs[[i]]["outward_normal"]],
        "length" -> N[squareSegs[[i]]["length"], prec],
        "pressure" -> cPair[caseDsegs[[i]]["pressure"]],
        "normal_velocity" -> cPair[caseDsegs[[i]]["normal_velocity"]],
        "thickness" -> cPair[caseDsegs[[i]]["thickness"]],
        "loading" -> cPair[caseDsegs[[i]]["loading"]],
        "total" -> cPair[caseDsegs[[i]]["total"]]|>,
      {i, 4}
    ],
    "thickness" -> cPair[caseDthi],
    "loading" -> cPair[caseDloa],
    "total" -> cPair[caseDtot]
  |>,
  "spreading" -> <|
    "distance_near" -> N[rNear, prec],
    "distance_far" -> N[rFar, prec],
    "ratio" -> N[spreadRatio, prec]
  |>,
  "tolerances" -> <|
    "component_abs" -> 1.0*^-9,
    "ratio_abs" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
