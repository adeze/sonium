(* Modal NVH golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_nvh_modal.wls > fixtures/solids_nvh_modal_golden.json
   Case: SDOF random-vibration closed forms plus single-mode receptance
   spots. One-sided PSDs over Hz: white force variance S0/(8 z w^3),
   Miles RMS sqrt(Pi S0 fn/(4 z)), receptance H = 1/(w^2-w^2+2izww).
   Provenance: Crandall, S. H. and Mark, W. D., "Random Vibration in
   Mechanical Systems" (white-noise integrals); Miles, J. W. (1954),
   "On structural fatigue under random loading", J. Aero. Sci. 21. *)

Off[General::munfl];

(* Exact inputs so N[., 17] emits full double precision. *)
s0force = 2;
zetaF = 1/50;
wnF = 25;
whiteVar = s0force/(8 zetaF wnF^3);
whiteRms = Sqrt[whiteVar];

s0base = 1;
zetaM = 1/20;
fnM = 10;
milesVar = Pi s0base fnM/(4 zetaM);
milesRms = Sqrt[milesVar];

(* Single-mode receptance, wn = 10, z = 1/50, unit modal pair. *)
wnR = 10;
zetaR = 1/50;
h[w_] := 1/(wnR^2 - w^2 + 2 I zetaR wnR w);
h0 = h[0];
hRes = h[wnR];
h2x = h[2 wnR];

If[! (whiteVar == 1/1250), Quit[1]];
If[! (Re[h0] == 1/100 && Im[h0] == 0), Quit[1]];
If[! (Re[hRes] == 0 && Im[hRes] == -1/4), Quit[1]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "nvh_modal",
    "units" -> "SI"|>,
  "inputs" -> <|
    "white_s0" -> 2.,
    "white_zeta" -> 0.02,
    "white_omega_n" -> 25.,
    "miles_s0" -> 1.,
    "miles_zeta" -> 0.05,
    "miles_fn_hz" -> 10.,
    "receptance_omega_n" -> 10.,
    "receptance_zeta" -> 0.02|>,
  "values" -> <|
    "white_variance" -> <|
      "values" -> {ToString[N[whiteVar, 17]]},
      "exact" -> "S0/(8 z w^3), Crandall & Mark"|>,
    "white_rms" -> <|
      "values" -> {ToString[N[whiteRms, 17]]},
      "exact" -> "sqrt of white_variance"|>,
    "miles_variance" -> <|
      "values" -> {ToString[N[milesVar, 17]]},
      "exact" -> "Pi S0 fn/(4 z), Miles 1954"|>,
    "miles_rms" -> <|
      "values" -> {ToString[N[milesRms, 17]]},
      "exact" -> "sqrt of miles_variance"|>,
    "receptance_dc_re" -> <|
      "values" -> {ToString[N[Re[h0], 17]]},
      "exact" -> "1/w^2 at w = 0"|>,
    "receptance_dc_im" -> <|
      "values" -> {ToString[N[Im[h0], 17]]},
      "exact" -> "0 at w = 0"|>,
    "receptance_res_re" -> <|
      "values" -> {ToString[N[Re[hRes], 17]]},
      "exact" -> "0 at resonance"|>,
    "receptance_res_im" -> <|
      "values" -> {ToString[N[Im[hRes], 17]]},
      "exact" -> "-1/(2 z w^2) at resonance"|>,
    "receptance_2x_re" -> <|
      "values" -> {ToString[N[Re[h2x], 17]]},
      "exact" -> "Re H at w = 2w"|>,
    "receptance_2x_im" -> <|
      "values" -> {ToString[N[Im[h2x], 17]]},
      "exact" -> "Im H at w = 2w"|>|>,
  "tolerances" -> <|
    "white_variance_abs" -> 0.,
    "white_variance_rel" -> 10^-12,
    "white_rms_abs" -> 0.,
    "white_rms_rel" -> 10^-12,
    "miles_variance_abs" -> 0.,
    "miles_variance_rel" -> 10^-12,
    "miles_rms_abs" -> 0.,
    "miles_rms_rel" -> 10^-12,
    "receptance_dc_re_abs" -> 0.,
    "receptance_dc_re_rel" -> 10^-12,
    "receptance_dc_im_abs" -> 10^-15,
    "receptance_dc_im_rel" -> 0.,
    "receptance_res_re_abs" -> 10^-15,
    "receptance_res_re_rel" -> 0.,
    "receptance_res_im_abs" -> 0.,
    "receptance_res_im_rel" -> 10^-12,
    "receptance_2x_re_abs" -> 0.,
    "receptance_2x_re_rel" -> 10^-12,
    "receptance_2x_im_abs" -> 0.,
    "receptance_2x_im_rel" -> 10^-12|>|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
