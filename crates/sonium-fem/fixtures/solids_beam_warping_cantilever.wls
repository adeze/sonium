(* Independent Wolfram reference for the warping-restrained cantilever.
   Solves the Vlasov torsion ODE G J th' - E Iw th''' = T at 50-digit
   working precision for a 2 m I-beam cantilever (b = 0.1, h = 0.2,
   t = 0.01 m, E = 200 GPa, nu = 0.3) with a 1000 N m tip torque, root
   twist and warping fixed, tip bimoment free. Section constants J and Iw
   replicate the sonium Section3D.i_beam thin-wall formulas exactly. The
   Rust side consumes solids_beam_warping_cantilever_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_beam_warping_cantilever.wls > fixtures/solids_beam_warping_cantilever_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

len = 2;
torqueT = 1000;
young = 200*10^9;
poisson = 3/10;
shearMod = young/(2*(1 + poisson));
flangeB = 1/10;
webH = 2/10;
thick = 1/100;

iyWeak = (2*thick*flangeB^3 + webH*thick^3)/12;
flangeSpacing = webH + thick;
warpingI = iyWeak*flangeSpacing^2/4;
torsionJ = (2*flangeB*thick^3 + webH*thick^3)/3;
gj = shearMod*torsionJ;
eg = young*warpingI;
mu = Sqrt[gj/eg];

sol = th[x] /. DSolve[{gj*th'[x] - eg*th'''[x] == torqueT,
  th[0] == 0, th'[0] == 0, th''[len] == 0}, th[x], x][[1]];
tipTwist = sol /. x -> len;
midTwist = sol /. x -> len/2;
rootBimoment = -eg*(D[sol, {x, 2}] /. x -> 0);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "beam_warping_cantilever",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "length_m" -> 2,
    "tip_torque_n_m" -> 1000,
    "youngs_modulus_pa" -> 200*10^9,
    "poisson_ratio" -> 0.3,
    "flange_width_m" -> 0.1,
    "web_height_m" -> 0.2,
    "thickness_m" -> 0.01,
    "torsion_constant_m4" -> dec[torsionJ],
    "warping_constant_m6" -> dec[warpingI],
    "mu_per_m" -> dec[mu]
  |>,
  "values" -> <|
    "tip_twist" -> <|
      "values" -> {dec[tipTwist]},
      "exact" -> "DSolve Vlasov ODE: (T/GJ)(L - Tanh(mu L)/mu)"
    |>,
    "midspan_twist" -> <|
      "values" -> {dec[midTwist]},
      "exact" -> "DSolve Vlasov ODE evaluated at L/2"
    |>,
    "root_bimoment" -> <|
      "values" -> {dec[rootBimoment]},
      "exact" -> "B = -E Iw th''(0) = -(T/mu) Tanh(mu L)"
    |>
  |>,
  "tolerances" -> <|
    "tip_twist_abs" -> 1.0*^-15,
    "tip_twist_rel" -> 1.0*^-6,
    "midspan_twist_abs" -> 1.0*^-15,
    "midspan_twist_rel" -> 1.0*^-6,
    "root_bimoment_abs" -> 1.0*^-9,
    "root_bimoment_rel" -> 1.0*^-2
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
