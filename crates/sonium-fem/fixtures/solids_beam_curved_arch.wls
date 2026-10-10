(* Independent Wolfram reference for the pinned circular arch.
   Derives, at 50-digit working precision, the horizontal thrust and crown
   deflection of a pinned semicircular arch (radius 1 m, square 0.05 m
   section, E = 200 GPa) under a 1000 N crown point load via Castigliano
   compatibility, plus the pure-membrane radial contraction under 1000 Pa
   uniform external pressure (ring theory; replayed on a quarter ring with
   symmetry BCs, since pinned ends forbid the membrane state). Both
   Euler-Bernoulli (bending + axial) and
   Timoshenko (adds shear) variants are derived; the thrust redundancy is
   solved from the zero-spread compatibility inside this script. The Rust
   side consumes solids_beam_curved_arch_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/solids_beam_curved_arch.wls > fixtures/solids_beam_curved_arch_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

radius = 1;
loadP = 1000;
pressureQ = 1000;
young = 200*10^9;
poisson = 3/10;
width = 5/100;
height = 5/100;
shearKappa = 5/6;

area = width*height;
inertia = height*width^3/12;
shearMod = young/(2*(1 + poisson));

(* Resultants on the right half, theta in [0, Pi/2] from the right pin.
   pLoad stays symbolic so crown deflection follows by differentiation;
   h is the redundant inward thrust. *)
m[th_, h_] := (pLoad/2)*radius*(1 - Cos[th]) - h*radius*Sin[th];
n[th_, h_] := -(h*Sin[th] + (pLoad/2)*Cos[th]);
v[th_, h_] := -h*Cos[th] + (pLoad/2)*Sin[th];

(* Zero relative horizontal spread at the pins (dU/dh = 0). *)
compatEB = Integrate[m[th, h]*(-radius*Sin[th])/(young*inertia)
  + n[th, h]*(-Sin[th])/(young*area), {th, 0, Pi/2}];
thrustEB = h /. Solve[compatEB == 0, h][[1]];
compatTim = compatEB + Integrate[v[th, h]*(-Cos[th])
  /(shearMod*area*shearKappa), {th, 0, Pi/2}];
thrustTim = h /. Solve[compatTim == 0, h][[1]];

dhEB = D[thrustEB, pLoad];
dhTim = D[thrustTim, pLoad];
dMdp[th_, h_, dh_] := (radius/2)*(1 - Cos[th]) - dh*radius*Sin[th];
dNdp[th_, h_, dh_] := -(dh*Sin[th] + Cos[th]/2);
dVdp[th_, h_, dh_] := -dh*Cos[th] + Sin[th]/2;

(* Crown deflection by Castigliano, both halves, bending + axial. *)
crownEB = 2*radius*Integrate[
  (m[th, h]*dMdp[th, h, dhEB]/(young*inertia)
    + n[th, h]*dNdp[th, h, dhEB]/(young*area)) /. h -> thrustEB,
  {th, 0, Pi/2}];
crownTim = 2*radius*Integrate[
  (m[th, h]*dMdp[th, h, dhTim]/(young*inertia)
    + n[th, h]*dNdp[th, h, dhTim]/(young*area)
    + v[th, h]*dVdp[th, h, dhTim]/(shearMod*area*shearKappa)) /. h -> thrustTim,
  {th, 0, Pi/2}];

thrustEBval = thrustEB /. pLoad -> loadP;
thrustTimval = thrustTim /. pLoad -> loadP;
crownEBval = crownEB /. pLoad -> loadP;
crownTimval = crownTim /. pLoad -> loadP;

(* Uniform external pressure: pure membrane state N = -q R. *)
membraneRadial = -pressureQ*radius^2/(young*area);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "beam_curved_arch",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "radius_m" -> 1,
    "crown_load_n" -> 1000,
    "pressure_pa" -> 1000,
    "youngs_modulus_pa" -> 200*10^9,
    "poisson_ratio" -> 0.3,
    "width_m" -> 0.05,
    "height_m" -> 0.05,
    "shear_kappa" -> 5/6
  |>,
  "values" -> <|
    "thrust" -> <|
      "values" -> {dec[thrustEBval]},
      "exact" -> "Castigliano zero-spread: (P/Pi)(R^2-r^2)/(R^2+r^2), r^2=I/A"
    |>,
    "thrust_timoshenko" -> <|
      "values" -> {dec[thrustTimval]},
      "exact" -> "Castigliano zero-spread with shear flexibility"
    |>,
    "crown_deflection" -> <|
      "values" -> {dec[crownEBval]},
      "exact" -> "dU/dP with bending + axial energy over both halves"
    |>,
    "crown_deflection_timoshenko" -> <|
      "values" -> {dec[crownTimval]},
      "exact" -> "dU/dP with bending + axial + shear energy over both halves"
    |>,
    "membrane_radial" -> <|
      "values" -> {dec[membraneRadial]},
      "exact" -> "w = -q R^2/(E A), pure membrane N = -q R"
    |>
  |>,
  "tolerances" -> <|
    "thrust_abs" -> 1.0*^-9,
    "thrust_rel" -> 1.0*^-9,
    "thrust_timoshenko_abs" -> 1.0*^-9,
    "thrust_timoshenko_rel" -> 1.0*^-9,
    "crown_deflection_abs" -> 1.0*^-15,
    "crown_deflection_rel" -> 1.0*^-9,
    "crown_deflection_timoshenko_abs" -> 1.0*^-15,
    "crown_deflection_timoshenko_rel" -> 1.0*^-9,
    "membrane_radial_abs" -> 1.0*^-15,
    "membrane_radial_rel" -> 1.0*^-3
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
