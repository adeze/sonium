(* Independent Wolfram reference for the tetra/wedge patch.
   Derives, at 50-digit working precision, the reference-tetrahedron
   volume, B-matrix entries, characteristic length, lumped mass, and
   uniaxial-strain stress, plus the unit-wedge volume, length, and
   quadrature weight sum. All inputs are exact rationals; roots stay
   exact until N[..., 50].
   The Rust side consumes solids_tetra_penta_patch_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_tetra_penta_patch.wls > fixtures/solids_tetra_penta_patch_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Reference tet (0,0,0),(1,0,0),(0,1,0),(0,0,1): J = I, V = 1/6.
   Gradients: N0 = (-1,-1,-1), N1 = (1,0,0), N2 = (0,1,0), N3 = (0,0,1). *)
tetVol = 1/6;
tetBxx = {-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
tetBxy = {-1, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0};

(* Faces: three right triangles 1/2 plus the hypotenuse face Sqrt[3]/2. *)
tetAmax = Sqrt[3]/2;
tetLen = tetVol/tetAmax;
rho = 7800;
tetMass = rho*tetVol/4;

(* Uniaxial-strain elastic stress at exx = 1/1000. *)
e = 200*10^9;
nu = 3/10;
g = e/(2*(1 + nu));
k = e/(3*(1 - 2*nu));
sxx = (k + 4*g/3)/1000;

(* Unit wedge: bottom (0,0,0),(1,0,0),(0,1,0), top z = 1.
   Triangle area 1/2 times height 1. Faces: 2 x 1/2 triangles,
   quads 1, 1, Sqrt[2]. *)
wedgeVol = 1/2;
wedgeAmax = Sqrt[2];
wedgeLen = wedgeVol/wedgeAmax;
wedgeMass = rho*wedgeVol/6;
quadSum = 1;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "tetra_penta_patch",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "tet" -> "reference (0,0,0),(1,0,0),(0,1,0),(0,0,1)",
    "wedge" -> "unit bottom triangle, height 1",
    "density" -> 7800,
    "youngs" -> 200.0*^9,
    "poisson" -> 0.3,
    "exx" -> 0.001
  |>,
  "values" -> <|
    "tet_volume" -> <|
      "values" -> {dec[tetVol]},
      "exact" -> "det J / 6 = 1/6"
    |>,
    "tet_b_row_xx" -> <|
      "values" -> dec /@ tetBxx,
      "exact" -> "exx row over [ux1,uy1,uz1..]"
    |>,
    "tet_b_row_xy" -> <|
      "values" -> dec /@ tetBxy,
      "exact" -> "gxy row [dNy,dNx,0] per node"
    |>,
    "tet_char_length" -> <|
      "values" -> {dec[tetLen]},
      "exact" -> "V / Amax = 1/(3 Sqrt[3])"
    |>,
    "tet_nodal_mass" -> <|
      "values" -> {dec[tetMass]},
      "exact" -> "rho V / 4"
    |>,
    "uniaxial_sxx" -> <|
      "values" -> {dec[sxx]},
      "exact" -> "(K + 4G/3) exx"
    |>,
    "wedge_volume" -> <|
      "values" -> {dec[wedgeVol]},
      "exact" -> "triangle area 1/2 times height"
    |>,
    "wedge_char_length" -> <|
      "values" -> {dec[wedgeLen]},
      "exact" -> "V / Amax = 1/(2 Sqrt[2])"
    |>,
    "wedge_nodal_mass" -> <|
      "values" -> {dec[wedgeMass]},
      "exact" -> "rho V / 6"
    |>,
    "wedge_quad_sum" -> <|
      "values" -> {dec[quadSum]},
      "exact" -> "reference wedge volume 1"
    |>
  |>,
  "tolerances" -> <|
    "tet_volume_abs" -> 1.0*^-15,
    "tet_volume_rel" -> 1.0*^-12,
    "tet_b_row_xx_abs" -> 1.0*^-15,
    "tet_b_row_xx_rel" -> 1.0*^-12,
    "tet_b_row_xy_abs" -> 1.0*^-15,
    "tet_b_row_xy_rel" -> 1.0*^-12,
    "tet_char_length_abs" -> 1.0*^-15,
    "tet_char_length_rel" -> 1.0*^-12,
    "tet_nodal_mass_abs" -> 1.0*^-9,
    "tet_nodal_mass_rel" -> 1.0*^-12,
    "uniaxial_sxx_abs" -> 1.0*^-3,
    "uniaxial_sxx_rel" -> 1.0*^-9,
    "wedge_volume_abs" -> 1.0*^-14,
    "wedge_volume_rel" -> 1.0*^-12,
    "wedge_char_length_abs" -> 1.0*^-15,
    "wedge_char_length_rel" -> 1.0*^-12,
    "wedge_nodal_mass_abs" -> 1.0*^-9,
    "wedge_nodal_mass_rel" -> 1.0*^-12,
    "wedge_quad_sum_abs" -> 1.0*^-15,
    "wedge_quad_sum_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
