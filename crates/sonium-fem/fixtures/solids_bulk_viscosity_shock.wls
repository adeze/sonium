(* Independent Wolfram reference for bulk-viscosity shock identities.
   Derives, at 50-digit working precision, the piston-shock Rankine-Hugoniot
   jump [p] = rho0*Us*Up with linear Hugoniot Us = c + s*Up for steel, exact
   viscosity samples q = rho*l*(c0*l*d^2 - c1*c*d) in compression (q = 0 in
   expansion), and the steady-shock width law L = Pi*l*Sqrt[c0/s] obtained by
   integrating the quadratic-viscosity profile du/dx = Sqrt[s*u*(Up-u)]/(l*Sqrt[c0])
   between the unshocked (u = 0) and shocked (u = Up) states. The viscous end
   states vanish (q = 0 where d = 0) for any c0/c1, so the jump is preserved
   independent of the viscosity coefficients; the Rust side checks
   self-consistency q(-zc) = gap through the production viscosity kernel,
   where zc = Up*Sqrt[s/c0]/(2*l) is the center strain rate and
   gap = rho0*s*Up^2/4 the Rayleigh-line/Hugoniot gap. The Rust side consumes
   solids_bulk_viscosity_shock_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_bulk_viscosity_shock.wls > fixtures/solids_bulk_viscosity_shock_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Steel piston shock (exact inputs). *)
rho0 = 7800;
c = 4500;
s = 3/2;
up = 100;
len = 1/100;
us = c + s*up;
pJump = rho0*us*up;
gap = rho0*s*up^2/4;

(* Viscosity samples with (c0, c1) = (11/10, 1/20). *)
c0 = 11/10;
c1 = 1/20;
q[d_] := rho0*len*(c0*len*d^2 - c1*c*d);
rates = {-50000, -1000, -100, 0, 500};
qsamples = Table[If[d < 0, q[d], 0], {d, rates}];

(* Width law and center strain rates over the c0 set. *)
c0set = {1/2, 11/10, 2};
widths = Table[Pi*len*Sqrt[c0v/s], {c0v, c0set}];
zcenter = Table[up*Sqrt[s/c0v]/(2*len), {c0v, c0set}];

(* Shock-end viscous pressures vanish for any coefficients. *)
endC0 = {11/10, 2};
endC1 = {1/20, 1/2};
endQ = Table[0, {Length[endC0]}];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "bulk_viscosity_shock",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "density" -> rho0,
    "sound_speed" -> c,
    "hugoniot_slope" -> 1.5,
    "piston_velocity" -> up,
    "element_length" -> 0.01,
    "c0_quadratic" -> 1.1,
    "c1_linear" -> 0.05,
    "sample_rates" -> rates,
    "c0_set" -> {0.5, 1.1, 2.0},
    "end_c0" -> {1.1, 2.0},
    "end_c1" -> {0.05, 0.5}
  |>,
  "values" -> <|
    "shock_speed" -> <|
      "values" -> {dec[us]},
      "exact" -> "Us = c + s*Up"
    |>,
    "pressure_jump" -> <|
      "values" -> {dec[pJump]},
      "exact" -> "[p] = rho0*Us*Up"
    |>,
    "q_samples" -> <|
      "values" -> dec /@ qsamples,
      "exact" -> "q = rho*l*(c0*l*d^2 - c1*c*d) for d < 0, else 0"
    |>,
    "center_strain_rates" -> <|
      "values" -> dec /@ zcenter,
      "exact" -> "zc = Up*Sqrt[s/c0]/(2*l)"
    |>,
    "center_q" -> <|
      "values" -> {dec[gap]},
      "exact" -> "q(-zc; c0, c1 = 0) = rho0*s*Up^2/4 for every c0"
    |>,
    "shock_widths" -> <|
      "values" -> dec /@ widths,
      "exact" -> "L = Pi*l*Sqrt[c0/s]"
    |>,
    "shock_end_q" -> <|
      "values" -> dec /@ endQ,
      "exact" -> "q = 0 at both shock ends for any c0/c1"
    |>
  |>,
  "tolerances" -> <|
    "shock_speed_abs" -> 1.0*^-9,
    "shock_speed_rel" -> 1.0*^-12,
    "pressure_jump_abs" -> 1.0*^-3,
    "pressure_jump_rel" -> 1.0*^-12,
    "q_samples_abs" -> 1.0*^-6,
    "q_samples_rel" -> 1.0*^-12,
    "center_strain_rates_abs" -> 1.0*^-9,
    "center_strain_rates_rel" -> 1.0*^-12,
    "center_q_abs" -> 1.0*^-6,
    "center_q_rel" -> 1.0*^-12,
    "shock_widths_abs" -> 1.0*^-15,
    "shock_widths_rel" -> 1.0*^-12,
    "shock_end_q_abs" -> 0,
    "shock_end_q_rel" -> 0
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
