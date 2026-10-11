(* Independent Wolfram reference for the axisymmetric QUAD ring patch.
   Derives, at 50-digit working precision, the unit-square patch
   (r in [1,2], z in [0,1]) centroid operator rows, hourglass vector,
   area, centroid radius, ring volume, lumped nodal mass, critical
   step, the uniaxial-strain elastic response at ezz = 1/1000, the
   radial hoop-resultant identity, the uniform-shear response, and the
   (zero) shock viscosity of the expansion step. All inputs are exact
   rationals; roots and Pi stay exact until N[..., 50].
   The Rust side consumes axisym_ring_patch_golden.json; this script
   is the derivation.
   Regen: wolframscript -script fixtures/axisym_ring_patch.wls > fixtures/axisym_ring_patch_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Patch nodes (r,z): (1,0),(2,0),(2,1),(1,1). Center Jacobian
   J = 1/2 I, so dN/dr = {sx/2}, dN/dz = {sz/2} with node signs
   {(-1,-1),(1,-1),(1,1),(-1,1)}. *)
dnDr = {-1/2, 1/2, 1/2, -1/2};
dnDz = {-1/2, -1/2, 1/2, 1/2};

(* Hourglass pattern orthogonalized: h - (h.r) br - (h.z) bz.
   On the square (h.r) = (h.z) = 0, so gamma = h exactly. *)
h = {1, -1, 1, -1};
rs = {1, 2, 2, 1};
zs = {0, 0, 1, 1};
gamma = h - (h.rs)*dnDr - (h.zs)*dnDz;

area = 1;
rbar = 3/2;
volume = 2*Pi*rbar*area;
rho = 7800;
nodalMass = rho*volume/4;
charLen = 1/Sqrt[2];
sound = 4500;
dtCrit = charLen/sound;

(* Uniaxial-strain elastic step: ezz = 1/1000, err = eth = grz = 0. *)
e = 200*10^9;
nu = 3/10;
lame = e*nu/((1 + nu)*(1 - 2*nu));
mu = e/(2*(1 + nu));
ezz = 1/1000;
szz = (lame + 2*mu)*ezz;
srr = lame*ezz;
sth = lame*ezz;
(* Radial nodal forces sum to the hoop resultant sth*W/rbar. *)
radialSum = sth*volume/rbar;
(* Uniform engineering shear grz = 1/1000 gives srz = mu*grz. *)
grz = 1/1000;
srz = mu*grz;
(* The elastic driver expands (volumetric rate +1/s): no viscosity. *)
bulkQ = 0;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "axisym_ring_patch",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "square" -> "r in [1,2], z in [0,1]",
    "density" -> 7800,
    "sound_speed" -> 4500,
    "youngs" -> 200.0*^9,
    "poisson" -> 0.3,
    "ezz" -> 0.001,
    "grz" -> 0.001
  |>,
  "values" -> <|
    "b_r_row" -> <|
      "values" -> dec /@ dnDr,
      "exact" -> "dN/dr row, exact halves"
    |>,
    "b_z_row" -> <|
      "values" -> dec /@ dnDz,
      "exact" -> "dN/dz row, exact halves"
    |>,
    "gamma" -> <|
      "values" -> dec /@ gamma,
      "exact" -> "h - (h.r)br - (h.z)bz = {1,-1,1,-1}"
    |>,
    "area" -> <|
      "values" -> {dec[area]},
      "exact" -> "1"
    |>,
    "radius_bar" -> <|
      "values" -> {dec[rbar]},
      "exact" -> "3/2"
    |>,
    "volume" -> <|
      "values" -> {dec[volume]},
      "exact" -> "2 Pi rbar A = 3 Pi"
    |>,
    "nodal_mass" -> <|
      "values" -> {dec[nodalMass]},
      "exact" -> "rho V / 4 = 5850 Pi"
    |>,
    "char_length" -> <|
      "values" -> {dec[charLen]},
      "exact" -> "A / dmax = 1/Sqrt[2]"
    |>,
    "critical_step" -> <|
      "values" -> {dec[dtCrit]},
      "exact" -> "L / c = 1/(4500 Sqrt[2])"
    |>,
    "elastic_szz" -> <|
      "values" -> {dec[szz]},
      "exact" -> "(lambda + 2 mu) ezz"
    |>,
    "elastic_srr" -> <|
      "values" -> {dec[srr]},
      "exact" -> "lambda ezz"
    |>,
    "elastic_sth" -> <|
      "values" -> {dec[sth]},
      "exact" -> "lambda ezz"
    |>,
    "radial_sum" -> <|
      "values" -> {dec[radialSum]},
      "exact" -> "sth W / rbar = 2 Pi sth"
    |>,
    "shear_srz" -> <|
      "values" -> {dec[srz]},
      "exact" -> "mu grz"
    |>,
    "bulk_q" -> <|
      "values" -> {dec[bulkQ]},
      "exact" -> "0 in expansion"
    |>
  |>,
  "tolerances" -> <|
    "b_r_row_abs" -> 1.0*^-15,
    "b_r_row_rel" -> 1.0*^-12,
    "b_z_row_abs" -> 1.0*^-15,
    "b_z_row_rel" -> 1.0*^-12,
    "gamma_abs" -> 1.0*^-15,
    "gamma_rel" -> 1.0*^-12,
    "area_abs" -> 1.0*^-15,
    "area_rel" -> 1.0*^-12,
    "radius_bar_abs" -> 1.0*^-15,
    "radius_bar_rel" -> 1.0*^-12,
    "volume_abs" -> 1.0*^-12,
    "volume_rel" -> 1.0*^-12,
    "nodal_mass_abs" -> 1.0*^-9,
    "nodal_mass_rel" -> 1.0*^-12,
    "char_length_abs" -> 1.0*^-15,
    "char_length_rel" -> 1.0*^-12,
    "critical_step_abs" -> 1.0*^-18,
    "critical_step_rel" -> 1.0*^-12,
    "elastic_szz_abs" -> 1.0*^-3,
    "elastic_szz_rel" -> 1.0*^-9,
    "elastic_srr_abs" -> 1.0*^-3,
    "elastic_srr_rel" -> 1.0*^-9,
    "elastic_sth_abs" -> 1.0*^-3,
    "elastic_sth_rel" -> 1.0*^-9,
    "radial_sum_abs" -> 1.0*^-6,
    "radial_sum_rel" -> 1.0*^-9,
    "shear_srz_abs" -> 1.0*^-3,
    "shear_srz_rel" -> 1.0*^-9,
    "bulk_q_abs" -> 1.0*^-15,
    "bulk_q_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
