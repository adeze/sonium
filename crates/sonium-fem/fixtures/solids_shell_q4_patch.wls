(* Independent Wolfram reference for the Q4 crash-shell patch.
   Derives, at 50-digit working precision, the unit-square center
   membrane operator row, hourglass vector, area, lumped nodal mass,
   critical step, plane-stress elastic response at exx = 1/1000, the
   membrane resultant, and the first 5-point Gauss-Legendre thickness
   node. All inputs are exact rationals; roots stay exact until N[..., 50].
   The Rust side consumes solids_shell_q4_patch_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_shell_q4_patch.wls > fixtures/solids_shell_q4_patch_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Unit square [0,1]^2, nodes (0,0),(1,0),(1,1),(0,1).
   Center Jacobian J = 1/2 I, so dN/dx = {sx/2}, dN/dy = {sy/2}
   with signs {(-1,-1),(1,-1),(1,1),(-1,1)}. *)
dnDx = {-1/2, 1/2, 1/2, -1/2};
dnDy = {-1/2, -1/2, 1/2, 1/2};
memRow0 = Flatten[Transpose[{dnDx, {0, 0, 0, 0}}]];
memRow1 = Flatten[Transpose[{{0, 0, 0, 0}, dnDy}]];

(* Hourglass pattern orthogonalized: h - (h.x) bx - (h.y) by.
   On the square (h.x) = (h.y) = 0, so gamma = h exactly. *)
h = {1, -1, 1, -1};
xs = {0, 1, 1, 0};
ys = {0, 0, 1, 1};
gamma = h - (h.xs)*dnDx - (h.ys)*dnDy;

area = 1;
rho = 7800;
thick = 1/500;
nodalMass = rho*area*thick/4;
charLen = 1;
sound = 4500;
dtCrit = charLen/sound;

(* Plane-stress uniaxial membrane strain exx = 1/1000. *)
e = 200*10^9;
nu = 3/10;
exx = 1/1000;
sxx = e*exx/(1 - nu^2);
syy = nu*e*exx/(1 - nu^2);
dezz = -nu/(1 - nu)*exx;
nxx = thick*sxx;

(* First 5-point Gauss-Legendre node on [-t/2, t/2]. Numerize the
   exact Solve roots before sorting (Sort orders exact radicals
   canonically, not numerically). *)
gl5 = Sort[N[x /. Solve[LegendreP[5, x] == 0, x], 50]];
z1 = gl5[[1]]*thick/2;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "shell_q4_patch",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "square" -> "unit [0,1]^2",
    "thickness_m" -> 0.002,
    "density" -> 7800,
    "sound_speed" -> 4500,
    "youngs" -> 200.0*^9,
    "poisson" -> 0.3,
    "exx" -> 0.001
  |>,
  "values" -> <|
    "membrane_b_row0" -> <|
      "values" -> dec /@ memRow0,
      "exact" -> "dN/dx row over [u1,v1..u4,v4], exact halves"
    |>,
    "membrane_b_row1" -> <|
      "values" -> dec /@ memRow1,
      "exact" -> "dN/dy row over [u1,v1..u4,v4], exact halves"
    |>,
    "gamma" -> <|
      "values" -> dec /@ gamma,
      "exact" -> "h - (h.x)bx - (h.y)by = {1,-1,1,-1}"
    |>,
    "area" -> <|
      "values" -> {dec[area]},
      "exact" -> "1"
    |>,
    "nodal_mass" -> <|
      "values" -> {dec[nodalMass]},
      "exact" -> "rho A t / 4"
    |>,
    "critical_step" -> <|
      "values" -> {dec[dtCrit]},
      "exact" -> "L / c = 1/4500"
    |>,
    "plane_sxx" -> <|
      "values" -> {dec[sxx]},
      "exact" -> "E exx / (1 - nu^2)"
    |>,
    "plane_syy" -> <|
      "values" -> {dec[syy]},
      "exact" -> "nu E exx / (1 - nu^2)"
    |>,
    "plane_dezz" -> <|
      "values" -> {dec[dezz]},
      "exact" -> "-nu exx / (1 - nu)"
    |>,
    "resultant_nxx" -> <|
      "values" -> {dec[nxx]},
      "exact" -> "t sxx"
    |>,
    "thickness_z1" -> <|
      "values" -> {dec[z1]},
      "exact" -> "first LegendreP[5] root times t/2"
    |>
  |>,
  "tolerances" -> <|
    "membrane_b_row0_abs" -> 1.0*^-15,
    "membrane_b_row0_rel" -> 1.0*^-12,
    "membrane_b_row1_abs" -> 1.0*^-15,
    "membrane_b_row1_rel" -> 1.0*^-12,
    "gamma_abs" -> 1.0*^-15,
    "gamma_rel" -> 1.0*^-12,
    "area_abs" -> 1.0*^-15,
    "area_rel" -> 1.0*^-12,
    "nodal_mass_abs" -> 1.0*^-12,
    "nodal_mass_rel" -> 1.0*^-12,
    "critical_step_abs" -> 1.0*^-18,
    "critical_step_rel" -> 1.0*^-12,
    "plane_sxx_abs" -> 1.0*^-3,
    "plane_sxx_rel" -> 1.0*^-9,
    "plane_syy_abs" -> 1.0*^-3,
    "plane_syy_rel" -> 1.0*^-9,
    "plane_dezz_abs" -> 1.0*^-15,
    "plane_dezz_rel" -> 1.0*^-9,
    "resultant_nxx_abs" -> 1.0*^-6,
    "resultant_nxx_rel" -> 1.0*^-9,
    "thickness_z1_abs" -> 1.0*^-15,
    "thickness_z1_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
