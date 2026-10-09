(* Independent Wolfram reference for the sonium-fem stationary FW-H kernel.
   Derives, at 50-digit working precision, single-patch thickness / loading /
   total pressures for three fixtures: (A) loading-only on-axis, (B)
   thickness-only on-axis, (C) combined oblique observer. The Rust side
   consumes fwh_monopole_patch_golden.json through stationary_fwh_pressure.
   Run: wolframscript -script fwh_monopole_patch.wls (bounded: < 60 s, < 8 MB out).
   Engine: Wolfram Language 15.0.0 for Linux x86-64 (verified 2026-09-27). *)

om = 1000; rho = 12/10; c = 340;
k = om/c;
pos = {1, 0, 0}; nrm = {1, 0, 0}; area = 2;

green[R_] := Exp[-I k R]/(4 Pi R);
dgreen[proj_, R_] := green[R] (I k + 1/R) proj;
thick[vn_, R_] := I om rho vn green[R] area;
load[p_, proj_, R_] := p dgreen[proj, R] area;

(* --- Case A: loading only, on-axis observer. --- *)
obsA = {4, 0, 0}; RA = Norm[obsA - pos]; projA = ((obsA - pos)/RA) . nrm;
pA = 3/2 - I/2; vnA = 0;
thickA = thick[vnA, RA]; loadA = load[pA, projA, RA]; totalA = thickA + loadA;

(* --- Case B: thickness only, on-axis observer. --- *)
pB = 0; vnB = 1/100 + 2 I/100;
thickB = thick[vnB, RA]; loadB = load[pB, projA, RA]; totalB = thickB + loadB;

(* --- Case C: combined, oblique observer. --- *)
obsC = {4, 1, 0}; RC = Norm[obsC - pos]; projC = ((obsC - pos)/RC) . nrm;
pC = 1 + I; vnC = 1/50 - I/100;
thickC = thick[vnC, RC]; loadC = load[pC, projC, RC]; totalC = thickC + loadC;

prec = 50;
cPair[z_] := {N[Re[z], prec], N[Im[z], prec]};
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.fwh-single-patch.v1",
  "convention" -> "SI units; exp(i wt) harmonic; exp(-i k r) propagation; stationary permeable FW-H",
  "generator" -> "crates/sonium-fem/fixtures/fwh_monopole_patch.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "angular_frequency" -> om,
  "density" -> N[rho, prec],
  "speed_of_sound" -> c,
  "patch" -> <|
    "position" -> pos,
    "outward_normal" -> nrm,
    "area" -> area
  |>,
  "case_a" -> <|
    "observer" -> obsA,
    "pressure" -> cPair[pA],
    "normal_velocity" -> cPair[vnA],
    "thickness" -> cPair[thickA],
    "loading" -> cPair[loadA],
    "total" -> cPair[totalA]
  |>,
  "case_b" -> <|
    "observer" -> obsA,
    "pressure" -> cPair[pB],
    "normal_velocity" -> cPair[vnB],
    "thickness" -> cPair[thickB],
    "loading" -> cPair[loadB],
    "total" -> cPair[totalB]
  |>,
  "case_c" -> <|
    "observer" -> obsC,
    "pressure" -> cPair[pC],
    "normal_velocity" -> cPair[vnC],
    "thickness" -> cPair[thickC],
    "loading" -> cPair[loadC],
    "total" -> cPair[totalC]
  |>,
  "tolerances" -> <|
    "component_abs" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
