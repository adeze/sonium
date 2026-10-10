(* Scordelis-Lo roof golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_shell_scordelis_lo.wls > fixtures/solids_shell_scordelis_lo_golden.json
   Case: cylindrical roof R = 25, L = 50, t = 0.25, E = 4.32e8, nu = 0,
   uniform normal pressure q = 90, half-angle 40 deg, diaphragm ends.
   Target: vertical deflection at the free-edge midpoint -0.3024.
   Provenance: Scordelis, A. C. and Lo, K. S. (1964), "Computer analysis of
   cylindrical shells", J. Am. Concrete Inst. 61; target as tabulated in
   Belytschko, Liu, and Moran, "Nonlinear Finite Elements for Continua and
   Structures", 2nd ed. The benchmark has no closed form; this script
   records the published inputs/target, checks dimensionless consistency,
   and emits the golden JSON. *)

Off[General::munfl];

radiusM = 25.;
lengthM = 50.;
thickM = 0.25;
ymodPa = 4.32*10^8;
nu = 0.;
loadPa = 90.;
targetW = -0.3024;

thin = thickM/radiusM;
memStiff = ymodPa thickM;
loadScale = loadPa radiusM^2/memStiff;

If[! (thin < 0.05), Quit[1]];
If[! (targetW < 0), Quit[1]];
If[! (Abs[targetW]/loadScale > 0.01 && Abs[targetW]/loadScale < 10000), Quit[1]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "shell_scordelis_lo",
    "units" -> "SI"|>,
  "inputs" -> <|
    "radius_m" -> radiusM,
    "length_m" -> lengthM,
    "thickness_m" -> thickM,
    "youngs_modulus_pa" -> ymodPa,
    "poisson_ratio" -> nu,
    "pressure_load_pa" -> loadPa,
    "half_angle_deg" -> 40.|>,
  "values" -> <|
    "free_edge_deflection" -> <|
      "values" -> {ToString[N[targetW, 16]]},
      "exact" -> "Scordelis-Lo (1964) tabulated free-edge midpoint deflection; no closed form"|>|>,
  "tolerances" -> <|
    "free_edge_deflection_abs" -> 0.,
    "free_edge_deflection_rel" -> 0.03|>|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
