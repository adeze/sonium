(* Pinched hemisphere golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_shell_pinched_hemisphere.wls > fixtures/solids_shell_pinched_hemisphere_golden.json
   Case: hemisphere R = 10, t = 0.04, 18-degree polar hole, E = 6.825e7,
   nu = 0.3, two inward + two outward diametral point loads P = 2.
   Target: radial deflection under the load 0.0940.
   Provenance: MacNeal, R. H. and Harder, R. L. (1985), "A proposed standard
   set of problems to test finite element accuracy", FEAD 1; loads and target
   as tabulated in Belytschko, Liu, and Moran, "Nonlinear Finite Elements for
   Continua and Structures", 2nd ed. The benchmark has no closed form; this
   script records the published inputs/target, checks dimensionless
   consistency, and emits the golden JSON. *)

Off[General::munfl];

radiusM = 10.;
thickM = 0.04;
holeDeg = 18.;
ymodPa = 6.825*10^7;
nu = 0.3;
ploadN = 2.;
targetU = 0.0940;

thin = thickM/radiusM;
bendStiff = ymodPa thickM^3/(12 (1 - nu^2));
flexScale = ploadN radiusM^2/(bendStiff/radiusM);

If[! (thin < 0.01), Quit[1]];
If[! (targetU > 0), Quit[1]];
If[! (targetU/flexScale > 0.001 && targetU/flexScale < 1000), Quit[1]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "shell_pinched_hemisphere",
    "units" -> "SI"|>,
  "inputs" -> <|
    "radius_m" -> radiusM,
    "thickness_m" -> thickM,
    "hole_deg" -> holeDeg,
    "youngs_modulus_pa" -> ymodPa,
    "poisson_ratio" -> nu,
    "pinch_load_n" -> ploadN|>,
  "values" -> <|
    "radial_inward" -> <|
      "values" -> {ToString[N[targetU, 16]]},
      "exact" -> "MacNeal-Harder (1985) tabulated pinch deflection; no closed form"|>,
    "radial_outward" -> <|
      "values" -> {ToString[N[targetU, 16]]},
      "exact" -> "MacNeal-Harder (1985) tabulated pinch deflection; no closed form"|>|>,
  "tolerances" -> <|
    "radial_inward_abs" -> 0.,
    "radial_inward_rel" -> 0.07,
    "radial_outward_abs" -> 0.,
    "radial_outward_rel" -> 0.07|>|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
