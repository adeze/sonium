(* Clamped circular plate golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_shell_clamped_plate.wls > fixtures/solids_shell_clamped_plate_golden.json
   Case: clamped disk a = 1, t = 0.02, E = 200e9, nu = 0.3,
   uniform pressure q = 1000 (Kirchhoff theory).
   Targets: center deflection w0 = -q a^4/(64 D), flexural rigidity D,
   hoop moment M_t(r) = q/16 [(1+nu) a^2 - (1+3nu) r^2] at the C3 probe
   radius r = 0.1875, and its M1 = M_t/t^2 normalization.
   Provenance: Timoshenko, S. and Woinowsky-Krieger, S., "Theory of Plates
   and Shells", 2nd ed., Article 24 (axisymmetrical bending). *)

Off[General::munfl];

(* Exact inputs so N[., 17] emits full double precision. *)
radiusM = 1;
thickM = 2/100;
ymodPa = 200*10^9;
nu = 3/10;
loadPa = 1000;
probeR = 3/16;

rigidity = ymodPa thickM^3/(12 (1 - nu^2));
centerW = -loadPa radiusM^4/(64 rigidity);
hoopM = loadPa/16 ((1 + nu) radiusM^2 - (1 + 3 nu) probeR^2);
probeM1 = hoopM/thickM^2;

If[! (rigidity > 0), Quit[1]];
If[! (centerW < 0), Quit[1]];
If[! (hoopM > 0 && probeM1 > 0), Quit[1]];
(* Center moment limit: M_t(0) = q a^2 (1+nu)/16. *)
If[! (Abs[loadPa/16 (1 + nu) radiusM^2 - 81.25] < 10^-9), Quit[1]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "shell_clamped_plate",
    "units" -> "SI"|>,
  "inputs" -> <|
    "radius_m" -> radiusM,
    "thickness_m" -> thickM,
    "youngs_modulus_pa" -> ymodPa,
    "poisson_ratio" -> nu,
    "pressure_load_pa" -> loadPa,
    "probe_radius_m" -> probeR|>,
  "values" -> <|
    "center_deflection" -> <|
      "values" -> {ToString[N[centerW, 17]]},
      "exact" -> "-q a^4/(64 D), Timoshenko plates Art. 24"|>,
    "flexural_rigidity" -> <|
      "values" -> {ToString[N[rigidity, 17]]},
      "exact" -> "E t^3/(12 (1-nu^2))"|>,
    "hoop_moment" -> <|
      "values" -> {ToString[N[hoopM, 17]]},
      "exact" -> "q/16 [(1+nu) a^2 - (1+3nu) r^2] at r = 0.1875"|>,
    "probe_m1" -> <|
      "values" -> {ToString[N[probeM1, 17]]},
      "exact" -> "M_t/t^2 shell resultant normalization"|>|>,
  "tolerances" -> <|
    "center_deflection_abs" -> 0.,
    "center_deflection_rel" -> 0.05,
    "flexural_rigidity_abs" -> 0.,
    "flexural_rigidity_rel" -> 10^-12,
    "hoop_moment_abs" -> 0.,
    "hoop_moment_rel" -> 10^-12,
    "probe_m1_abs" -> 0.,
    "probe_m1_rel" -> 10^-12|>|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
