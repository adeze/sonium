(* Independent Wolfram reference for the Bao-Xue-Wierzbicki locus.
   Derives, at 50-digit working precision, the branch values, shell Lode
   coordinates, and fracture strains of (974)/(1067)-(1069) at the miniqa
   input_51 deck parameters. All inputs are exact rationals; exps and
   powers stay exact until N[..., 50]. The Rust side consumes
   solids_wierzbicki_locus_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_wierzbicki_locus.wls > fixtures/solids_wierzbicki_locus_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

c1 = 87/100;
c2 = 177/100;
c3 = 21/100;
c4 = 56/100;
mm = 2;
nn = 135/1000;

emax[eta_] := c1*Exp[-c2*eta];
emin[eta_] := c3*Exp[-c4*eta];
xiShell[eta_] := -(27/2)*eta*(eta^2 - 1/3);
efLocus[eta_, xi_] :=
  (emax[eta]^nn - (emax[eta]^nn - emin[eta]^nn)*(1 - Abs[xi]^mm)^(1/mm))^(1/nn);

etaUni = 1/3;
xiUni = xiShell[etaUni];
efUni = efLocus[etaUni, xiUni];
efShear = efLocus[0, 0];

etaMid = 1/2;
xiMid = xiShell[etaMid];
efMid = efLocus[etaMid, xiMid];

xiBiax = xiShell[2/3];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "wierzbicki_locus",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "c1" -> 0.87,
    "c2" -> 1.77,
    "c3" -> 0.21,
    "c4" -> 0.56,
    "m" -> 2.0,
    "n" -> 0.135
  |>,
  "values" -> <|
    "emax_uniaxial" -> <|
      "values" -> {dec[emax[etaUni]]},
      "exact" -> "C1 exp(-C2/3)"
    |>,
    "emin_uniaxial" -> <|
      "values" -> {dec[emin[etaUni]]},
      "exact" -> "C3 exp(-C4/3)"
    |>,
    "xi_uniaxial" -> <|
      "values" -> {dec[xiUni]},
      "exact" -> "-(27/2)(1/3)(1/9-1/3) = 1"
    |>,
    "ef_uniaxial" -> <|
      "values" -> {dec[efUni]},
      "exact" -> "locus(1/3, 1) = emax"
    |>,
    "ef_shear" -> <|
      "values" -> {dec[efShear]},
      "exact" -> "locus(0, 0) = emin = C3"
    |>,
    "xi_mid" -> <|
      "values" -> {dec[xiMid]},
      "exact" -> "-(27/2)(1/2)(1/4-1/3) = 9/16"
    |>,
    "ef_mid" -> <|
      "values" -> {dec[efMid]},
      "exact" -> "locus(1/2, 9/16) per (974)"
    |>,
    "xi_biaxial" -> <|
      "values" -> {dec[xiBiax]},
      "exact" -> "-(27/2)(2/3)(4/9-1/3) = -1"
    |>
  |>,
  "tolerances" -> <|
    "emax_uniaxial_abs" -> 1.0*^-15,
    "emax_uniaxial_rel" -> 1.0*^-12,
    "emin_uniaxial_abs" -> 1.0*^-15,
    "emin_uniaxial_rel" -> 1.0*^-12,
    "xi_uniaxial_abs" -> 1.0*^-15,
    "xi_uniaxial_rel" -> 1.0*^-12,
    "ef_uniaxial_abs" -> 1.0*^-15,
    "ef_uniaxial_rel" -> 1.0*^-12,
    "ef_shear_abs" -> 1.0*^-15,
    "ef_shear_rel" -> 1.0*^-12,
    "xi_mid_abs" -> 1.0*^-15,
    "xi_mid_rel" -> 1.0*^-12,
    "ef_mid_abs" -> 1.0*^-15,
    "ef_mid_rel" -> 1.0*^-12,
    "xi_biaxial_abs" -> 1.0*^-15,
    "xi_biaxial_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
