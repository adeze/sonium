(* Independent Wolfram reference for the sonium-fem 1D elastic bar impact.
   Closed-form d'Alembert solution at 50-digit working precision for the
   bar2_v41 copper-bar analog (rho = 8930 kg/m^3, E = 117 GPa from
   /MAT/LAW3 RHO_0 8.93 / E 0.117 in cm-g-us deck units, L = 0.0324 m
   from the node z extent 0..3.24, v = 227 m/s from /INIVEL Vz -0.0227
   and the deck readme "ALE 2D IMPACT 227M/S"): wall face fixed at x = 0,
   far face free, contact stress rho c v, separation at 2L/c with rigid
   rebound. Sample points sit strictly inside constant regions (no
   characteristic crossings) plus two post-separation free-flight points.
   The Rust side consumes ale_bar2_golden.json through BarImpact.
   Run: wolframscript -script ale_bar2.wls > ale_bar2_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86-64. *)

prec = 50;
rho = 8930;
young = 117*^9;
len = 324/10000;
speed = 227;
c = Sqrt[young/rho];
contact = rho c speed;
separation = 2 len/c;

(* Odd about the wall, even about the free face: 4L periodic. *)
ext[xi_] := Module[{red = Mod[xi + 2 len, 4 len] - 2 len},
  If[red < 0, speed, -speed]];
vel[x_, t_] := If[c t >= 2 len, speed, (ext[x - c t] + ext[x + c t])/2];
stress[x_, t_] := If[c t >= 2 len, 0, rho c (ext[x + c t] - ext[x - c t])/2];

xs = {0, len/5, len/2, 4 len/5, len};
ts = separation*{1/8, 3/8, 5/8, 7/8};
samples = Flatten[Table[
  <|"x" -> x, "t" -> t, "velocity" -> vel[x, t], "stress" -> stress[x, t]|>,
  {x, xs}, {t, ts}], 1];
free = {
  <|"x" -> len/3, "t" -> 5 separation/2, "velocity" -> vel[len/3, 5 separation/2],
    "stress" -> stress[len/3, 5 separation/2]|>,
  <|"x" -> 0, "t" -> 5 separation/2, "velocity" -> vel[0, 5 separation/2],
    "stress" -> stress[0, 5 separation/2]|>
};

(* Energy cross-check: KE + strain at t = L/(2c) equals launch KE. *)
ke0 = rho len speed^2/2;
nq = 2000; wq = len/nq;
tq = len/(2 c);
energy = Sum[
  (rho vel[(i + 1/2) wq, tq]^2/2 + stress[(i + 1/2) wq, tq]^2/(2 young)) wq,
  {i, 0, nq - 1}];
energyDrift = Abs[energy - ke0]/ke0;

nSample[s_] := <|"x" -> N[s["x"], prec], "t" -> N[s["t"], prec],
  "velocity" -> N[s["velocity"], prec], "stress" -> N[s["stress"], prec]|>;
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.ale-bar2.v1",
  "convention" -> "1D elastic normal impact; wall at x=0, free face at x=L; compression negative",
  "generator" -> "crates/sonium-fem/fixtures/ale_bar2.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "density" -> N[rho, prec],
  "young" -> N[young, prec],
  "length" -> N[len, prec],
  "speed" -> N[speed, prec],
  "wave_speed" -> N[c, prec],
  "contact_stress" -> N[contact, prec],
  "separation_time" -> N[separation, prec],
  "samples" -> nSample /@ samples,
  "free_flight" -> nSample /@ free,
  "energy_drift" -> N[energyDrift, prec],
  "tolerances" -> <|
    "derived_rel" -> 1.0*^-12,
    "velocity_abs" -> 1.0*^-6,
    "stress_rel" -> 1.0*^-9,
    "energy_rel" -> 5.0*^-3
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
