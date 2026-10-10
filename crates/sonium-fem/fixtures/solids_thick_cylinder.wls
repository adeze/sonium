(* Independent Wolfram reference for thick-cylinder stresses.
   Derives, at 50-digit working precision, the Lame radial and hoop stresses
   at the bore, midwall, and outer radius under 10 MPa internal pressure, the
   closed-end axial stress, and the bore radial displacement in both
   plane-strain and plane-stress form. All inputs are exact rationals. The
   Rust side consumes solids_thick_cylinder_golden.json; this script is
   the derivation.
   Regen: wolframscript -script fixtures/solids_thick_cylinder.wls > fixtures/solids_thick_cylinder_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

aInner = 1/10;
bOuter = 2/10;
pInner = 10^7;
young = 200*10^9;
poisson = 3/10;

lame[r_] := Module[{s = pInner*aInner^2/(bOuter^2 - aInner^2)},
  {s*(1 - bOuter^2/r^2), s*(1 + bOuter^2/r^2)}];
{srA, stA} = lame[aInner];
{srM, stM} = lame[(aInner + bOuter)/2];
{srB, stB} = lame[bOuter];
axial = pInner*aInner^2/(bOuter^2 - aInner^2);

(* Plane strain: u = (1+nu)/E * ((1-2 nu) A r - B/r). *)
bigA = pInner*aInner^2/(bOuter^2 - aInner^2);
bigB = pInner*aInner^2*bOuter^2/(bOuter^2 - aInner^2);
dispStrain = (1 + poisson)/young*((1 - 2*poisson)*bigA*aInner - bigB/aInner);
(* Plane stress: u = r (sigma_theta - nu sigma_r)/E. *)
dispStress = aInner*(stA - poisson*srA)/young;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "thick_cylinder",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "inner_radius_m" -> 0.1,
    "outer_radius_m" -> 0.2,
    "inner_pressure_pa" -> 10^7,
    "youngs_modulus_pa" -> 200*10^9,
    "poisson_ratio" -> 0.3
  |>,
  "values" -> <|
    "sigma_r_bore" -> <|
      "values" -> {dec[srA]},
      "exact" -> "Lame: p a^2/(b^2-a^2) (1-b^2/a^2)"
    |>,
    "sigma_theta_bore" -> <|
      "values" -> {dec[stA]},
      "exact" -> "Lame: p a^2/(b^2-a^2) (1+b^2/a^2)"
    |>,
    "sigma_r_mid" -> <|
      "values" -> {dec[srM]},
      "exact" -> "Lame at r=(a+b)/2"
    |>,
    "sigma_theta_mid" -> <|
      "values" -> {dec[stM]},
      "exact" -> "Lame at r=(a+b)/2"
    |>,
    "sigma_r_outer" -> <|
      "values" -> {dec[srB]},
      "exact" -> "Lame at r=b (traction-free)"
    |>,
    "sigma_theta_outer" -> <|
      "values" -> {dec[stB]},
      "exact" -> "Lame at r=b"
    |>,
    "axial_stress" -> <|
      "values" -> {dec[axial]},
      "exact" -> "p a^2/(b^2-a^2), closed ends"
    |>,
    "displacement_bore_plane_strain" -> <|
      "values" -> {dec[dispStrain]},
      "exact" -> "(1+nu)/E ((1-2 nu) A a-B/a)"
    |>,
    "displacement_bore_plane_stress" -> <|
      "values" -> {dec[dispStress]},
      "exact" -> "a (sigma_t-nu sigma_r)/E"
    |>
  |>,
  "tolerances" -> <|
    "sigma_r_bore_abs" -> 1.0*^-9,
    "sigma_r_bore_rel" -> 1.0*^-12,
    "sigma_theta_bore_abs" -> 1.0*^-9,
    "sigma_theta_bore_rel" -> 1.0*^-12,
    "sigma_r_mid_abs" -> 1.0*^-9,
    "sigma_r_mid_rel" -> 1.0*^-12,
    "sigma_theta_mid_abs" -> 1.0*^-9,
    "sigma_theta_mid_rel" -> 1.0*^-12,
    "sigma_r_outer_abs" -> 1.0*^-9,
    "sigma_r_outer_rel" -> 1.0*^-12,
    "sigma_theta_outer_abs" -> 1.0*^-9,
    "sigma_theta_outer_rel" -> 1.0*^-12,
    "axial_stress_abs" -> 1.0*^-9,
    "axial_stress_rel" -> 1.0*^-12,
    "displacement_bore_plane_strain_abs" -> 1.0*^-15,
    "displacement_bore_plane_strain_rel" -> 1.0*^-12,
    "displacement_bore_plane_stress_abs" -> 1.0*^-15,
    "displacement_bore_plane_stress_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
