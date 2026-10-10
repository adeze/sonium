(* Twisted beam golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_shell_twisted_beam.wls > fixtures/solids_shell_twisted_beam_golden.json
   Case: cantilever L = 12, width 1.1, t = 0.32, 90-degree twist about the
   beam axis, E = 29e6, nu = 0.22, unit tip loads in the tip width
   (in-plane) and tip thickness (out-of-plane) directions.
   Targets: in-plane tip deflection 0.005424, out-of-plane 0.001754.
   Provenance: MacNeal, R. H. and Harder, R. L. (1985), "A proposed standard
   set of problems to test finite element accuracy", FEAD 1. The twisted
   targets have no closed form; this script records the published
   inputs/targets, cross-checks the straight-beam (zero twist) closed forms
   that bound them, and emits the golden JSON. *)

Off[General::munfl];

lengthM = 12.;
widthM = 1.1;
thickM = 0.32;
ymodPa = 29.*10^6;
nu = 0.22;
ploadN = 1.;
targetIn = 0.005424;
targetOut = 0.001754;

iWeak = thickM widthM^3/12;
iStrong = widthM thickM^3/12;
straightIn = ploadN lengthM^3/(3 ymodPa iWeak);
straightOut = ploadN lengthM^3/(3 ymodPa iStrong);

If[! (targetIn > 0 && targetOut > 0), Quit[1]];
If[! (targetIn > straightIn), Quit[1]];
If[! (targetOut < straightOut), Quit[1]];
If[! (targetIn > targetOut), Quit[1]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "shell_twisted_beam",
    "units" -> "SI"|>,
  "inputs" -> <|
    "length_m" -> lengthM,
    "width_m" -> widthM,
    "thickness_m" -> thickM,
    "youngs_modulus_pa" -> ymodPa,
    "poisson_ratio" -> nu,
    "tip_load_n" -> ploadN,
    "twist_deg" -> 90.|>,
  "values" -> <|
    "tip_in_plane" -> <|
      "values" -> {ToString[N[targetIn, 16]]},
      "exact" -> "MacNeal-Harder (1985) tabulated twisted-beam deflection; no closed form"|>,
    "tip_out_of_plane" -> <|
      "values" -> {ToString[N[targetOut, 16]]},
      "exact" -> "MacNeal-Harder (1985) tabulated twisted-beam deflection; no closed form"|>|>,
  "tolerances" -> <|
    "tip_in_plane_abs" -> 0.,
    "tip_in_plane_rel" -> 0.05,
    "tip_out_of_plane_abs" -> 0.,
    "tip_out_of_plane_rel" -> 0.05|>|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
