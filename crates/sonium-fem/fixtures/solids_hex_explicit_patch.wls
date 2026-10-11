(* Independent Wolfram reference for the H8 explicit patch.
   Derives, at 50-digit working precision, the unit-cube volume,
   centroid gradients, hourglass base mode, characteristic length,
   lumped nodal mass, critical step, viscous hourglass coefficient,
   and the uniaxial-strain elastic stress. All inputs are exact
   rationals; roots stay exact until N[..., 50].
   The Rust side consumes solids_hex_explicit_patch_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_hex_explicit_patch.wls > fixtures/solids_hex_explicit_patch_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Unit cube [0,1]^3 in xi-fastest HEX_NODES order.
   Center Jacobian J = 1/2 I, det = 1/8, volume = 8 det = 1.
   Centroid gradients dN/dx = s/4 for node signs s. *)
signs = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
         {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}};
grad0 = signs[[1]]/4;
volume = 1;

(* Flanagan-Belytschko base mode 0: xi*eta over the nodes. *)
hgMode0 = {1, -1, 1, -1, 1, -1, 1, -1};

(* Largest face area 1, so L = V/A = 1. *)
charLen = 1;
rho = 7800;
sound = 4500;
nodalMass = rho*volume/8;
dtCrit = charLen/sound;

(* Viscous hourglass coefficient hg_scale rho c L^2, scale 0.05. *)
hgCoeff = 5/100*rho*sound*charLen^2;

(* Uniaxial-strain elastic stress at exx = 1/1000. *)
e = 200*10^9;
nu = 3/10;
g = e/(2*(1 + nu));
k = e/(3*(1 - 2*nu));
sxx = (k + 4*g/3)/1000;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "hex_explicit_patch",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "cube" -> "unit [0,1]^3",
    "density" -> 7800,
    "sound_speed" -> 4500,
    "hg_scale" -> 0.05,
    "youngs" -> 200.0*^9,
    "poisson" -> 0.3,
    "exx" -> 0.001
  |>,
  "values" -> <|
    "volume" -> <|
      "values" -> {dec[volume]},
      "exact" -> "8 det J(0) = 1"
    |>,
    "grad_node0" -> <|
      "values" -> dec /@ grad0,
      "exact" -> "J^-T dN/dxi at center = {-1/4,-1/4,-1/4}"
    |>,
    "hg_mode0" -> <|
      "values" -> dec /@ hgMode0,
      "exact" -> "xi*eta base mode"
    |>,
    "char_length" -> <|
      "values" -> {dec[charLen]},
      "exact" -> "V / Amax = 1"
    |>,
    "nodal_mass" -> <|
      "values" -> {dec[nodalMass]},
      "exact" -> "rho V / 8"
    |>,
    "critical_step" -> <|
      "values" -> {dec[dtCrit]},
      "exact" -> "L / c = 1/4500"
    |>,
    "hg_coeff" -> <|
      "values" -> {dec[hgCoeff]},
      "exact" -> "hg_scale rho c L^2"
    |>,
    "uniaxial_sxx" -> <|
      "values" -> {dec[sxx]},
      "exact" -> "(K + 4G/3) exx"
    |>
  |>,
  "tolerances" -> <|
    "volume_abs" -> 1.0*^-15,
    "volume_rel" -> 1.0*^-12,
    "grad_node0_abs" -> 1.0*^-15,
    "grad_node0_rel" -> 1.0*^-12,
    "hg_mode0_abs" -> 1.0*^-15,
    "hg_mode0_rel" -> 1.0*^-12,
    "char_length_abs" -> 1.0*^-15,
    "char_length_rel" -> 1.0*^-12,
    "nodal_mass_abs" -> 1.0*^-9,
    "nodal_mass_rel" -> 1.0*^-12,
    "critical_step_abs" -> 1.0*^-18,
    "critical_step_rel" -> 1.0*^-12,
    "hg_coeff_abs" -> 1.0*^-6,
    "hg_coeff_rel" -> 1.0*^-12,
    "uniaxial_sxx_abs" -> 1.0*^-3,
    "uniaxial_sxx_rel" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
