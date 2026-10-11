(* Independent Wolfram reference for spring property mapping.
   Derives, at 50-digit working precision, the TYPE8/TYPE25/TYPE32
   linear-coefficient maps onto beam sections: effective modulus,
   area, inertias, torsion constant, and shear factors from exact
   rational card inputs. Roots stay exact until N[..., 50].
   The Rust side consumes solids_beam_prop_section_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_beam_prop_section.wls > fixtures/solids_beam_prop_section_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* TYPE8 sample card: Kx = 3e6, Ky = Kz = 1e6, Krx = 2e4,
   Kry = Krz = 5e4, mass 78/10, L = 2, nu = 3/10, rho = 7800.
   Area A = mass/(rho L); E = Kx L / A; G = E/(2(1+nu));
   J = Krx L / G; Iy = Kry L / E; Iz = Krz L / E;
   kappa = Ky L / (G A). *)
kx = 3*10^6;
ky = 10^6;
krx = 2*10^4;
kry = 5*10^4;
mass = 78/10;
len = 2;
nu = 3/10;
rho = 7800;
area = mass/(rho*len);
ea = kx*len;
eEff = ea/area;
g = eEff/(2*(1 + nu));
tors = krx*len/g;
iy = kry*len/eEff;
kappa = ky*len/(g*area);

(* TYPE25 sample card: tension 2e6, shear 8e5, torsion 1e4,
   bend 3e4, mass 39/10, L = 1. Isotropic radial modes. *)
tKx = 2*10^6;
tKs = 8*10^5;
tKrx = 10^4;
tKb = 3*10^4;
tMass = 39/10;
tLen = 1;
tArea = tMass/(rho*tLen);
tEa = tKx*tLen;
tE = tEa/tArea;
tG = tE/(2*(1 + nu));
tJ = tKrx*tLen/tG;
tI = tKb*tLen/tE;

(* TYPE32 sample card: Stif0 = 1e5, L = 1/4, mass 5/100.
   EA = Stif0 L; A = mass/(rho L); E = EA/A. *)
pStif = 10^5;
pLen = 1/4;
pMass = 5/100;
pEa = pStif*pLen;
pArea = pMass/(rho*pLen);
pE = pEa/pArea;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "beam_prop_section",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "type8" -> "Kx 3e6, Ky=Kz 1e6, Krx 2e4, Kry=Krz 5e4, mass 7.8, L 2",
    "type25" -> "tension 2e6, shear 8e5, torsion 1e4, bend 3e4, mass 3.9, L 1",
    "type32" -> "Stif0 1e5, L 0.25, mass 0.05",
    "nu" -> 0.3,
    "rho" -> 7800
  |>,
  "values" -> <|
    "gene_area" -> <|
      "values" -> {dec[area]},
      "exact" -> "mass/(rho L)"
    |>,
    "gene_modulus" -> <|
      "values" -> {dec[eEff]},
      "exact" -> "Kx L / A"
    |>,
    "gene_shear_modulus" -> <|
      "values" -> {dec[g]},
      "exact" -> "E/(2(1+nu))"
    |>,
    "gene_torsion" -> <|
      "values" -> {dec[tors]},
      "exact" -> "Krx L / G"
    |>,
    "gene_iy" -> <|
      "values" -> {dec[iy]},
      "exact" -> "Kry L / E"
    |>,
    "gene_kappa" -> <|
      "values" -> {dec[kappa]},
      "exact" -> "Ky L / (G A)"
    |>,
    "axi_area" -> <|
      "values" -> {dec[tArea]},
      "exact" -> "mass/(rho L)"
    |>,
    "axi_torsion" -> <|
      "values" -> {dec[tJ]},
      "exact" -> "Ktor L / G"
    |>,
    "axi_inertia" -> <|
      "values" -> {dec[tI]},
      "exact" -> "Kb L / E, both axes"
    |>,
    "pre_ea" -> <|
      "values" -> {dec[pEa]},
      "exact" -> "Stif0 L"
    |>,
    "pre_area" -> <|
      "values" -> {dec[pArea]},
      "exact" -> "mass/(rho L)"
    |>,
    "pre_modulus" -> <|
      "values" -> {dec[pE]},
      "exact" -> "EA / A"
    |>
  |>,
  "tolerances" -> <|
    "gene_area_abs" -> 1.0*^-18,
    "gene_area_rel" -> 1.0*^-12,
    "gene_modulus_abs" -> 1.0*^-3,
    "gene_modulus_rel" -> 1.0*^-12,
    "gene_shear_modulus_abs" -> 1.0*^-3,
    "gene_shear_modulus_rel" -> 1.0*^-12,
    "gene_torsion_abs" -> 1.0*^-18,
    "gene_torsion_rel" -> 1.0*^-12,
    "gene_iy_abs" -> 1.0*^-18,
    "gene_iy_rel" -> 1.0*^-12,
    "gene_kappa_abs" -> 1.0*^-15,
    "gene_kappa_rel" -> 1.0*^-12,
    "axi_area_abs" -> 1.0*^-18,
    "axi_area_rel" -> 1.0*^-12,
    "axi_torsion_abs" -> 1.0*^-18,
    "axi_torsion_rel" -> 1.0*^-12,
    "axi_inertia_abs" -> 1.0*^-18,
    "axi_inertia_rel" -> 1.0*^-12,
    "pre_ea_abs" -> 1.0*^-9,
    "pre_ea_rel" -> 1.0*^-12,
    "pre_area_abs" -> 1.0*^-18,
    "pre_area_rel" -> 1.0*^-12,
    "pre_modulus_abs" -> 1.0*^-6,
    "pre_modulus_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
