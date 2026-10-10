(* Composites failure-criterion golden reference.
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)
   Regen: wolframscript -script fixtures/solids_composites_failure.wls > fixtures/solids_composites_failure_golden.json
   Case: Hashin (1980) and Puck inter-fiber-fracture exposures for eight
   plane-stress states in ply material coordinates, carbon/epoxy allowables.
   The Puck action-plane search here scans 0.05 degrees (reference
   resolution); the production scan uses 0.5 degrees and must match the
   exposure within 1e-3 relative and the angle within 0.5 degrees absolute.
   Provenance: Hashin, Z. (1980) J. Appl. Mech. 47; Puck, A. and
   Schuermann, H. (1998) Comp. Sci. Tech. 58 and (2002) 62, via
   Deuschle, H. M. (2010) PhD thesis, Universitaet Stuttgart, Sec. 4.2. *)

Off[General::munfl];

xt = 1.95*10^9; xc = 1.48*10^9;
yt = 4.8*10^7; yc = 2.0*10^8;
s12 = 7.9*10^7; s23 = 7.9*10^7;
ptPar = 0.35; pcPar = 0.30; ptPerp = 0.30; pcPerp = 0.25;

(* Hashin 1980 exposures (sqrt of criterion values). *)
hashin[s1_, s2_, t12_] := Module[{sh, ft, fc, mt, mc},
  sh = (t12/s12)^2;
  ft = If[s1 >= 0, Sqrt[Max[0, (s1/xt)^2 + sh]], 0];
  fc = If[s1 < 0, Max[0, -s1/xc], 0];
  mt = If[s2 >= 0, Sqrt[Max[0, (s2/yt)^2 + sh]], 0];
  mc = If[s2 < 0,
    Sqrt[Max[0, ((yc/(2*s23))^2 - 1)*(s2/yc) + (s2/(2*s23))^2 + sh]], 0];
  {ft, fc, mt, mc}];

(* Puck IFF exposure at one fracture angle (radians). *)
rT = yt; rPP = yc/(2*(1 + pcPerp)); rPL = s12;
puckIFF[s2_, t12_, th_] := Module[{sn, tnt, tn1, sh2, mag2, ratio, lin, quad, c2},
  sn = s2*Cos[th]^2;
  tnt = -s2*Sin[th]*Cos[th];
  tn1 = t12*Cos[th];
  sh2 = (tnt/rPP)^2 + (tn1/rPL)^2;
  mag2 = tnt^2 + tn1^2;
  If[sn >= 0,
    ratio = If[mag2 > 0,
      (ptPerp/rPP)*tnt^2/mag2 + (ptPar/rPL)*tn1^2/mag2, ptPerp/rPP];
    lin = 2*ratio*sn;
    c2 = 1 - 2*ratio*rT;
    quad = sh2 + c2*(sn/rT)^2;
    (lin + Sqrt[Max[0, lin^2 + 4*quad]])/2,
    ratio = If[mag2 > 0,
      (pcPerp/rPP)*tnt^2/mag2 + (pcPar/rPL)*tn1^2/mag2, pcPerp/rPP];
    lin = 2*ratio*sn;
    (lin + Sqrt[Max[0, lin^2 + 4*sh2]])/2]];

(* Reference search: 0.05-degree scan over [0, 90]. *)
puckScan[s2_, t12_] := Module[{best = 0., ang = 0., th, e},
  Do[th = d*Pi/180.;
    e = puckIFF[s2, t12, th];
    If[e > best, best = e; ang = d],
    {d, 0., 90., 0.05}];
  {best, ang}];

puck[s1_, s2_, t12_] := Module[{ff, iff},
  ff = If[s1 >= 0, Max[0, s1/xt], Max[0, -s1/xc]];
  iff = puckScan[s2, t12];
  {ff, iff[[1]], iff[[2]]}];

(* Sanity cross-checks before emitting. *)
If[Abs[puckIFF[yt, 0, 0] - 1] > 1.*10^-9, Quit[1]];
If[Abs[puckIFF[0, s12, 0] - 1] > 1.*10^-9, Quit[1]];
comp = puckScan[-yc, 0];
If[Abs[comp[[1]] - 1] > 0.02, Quit[1]];
If[Abs[comp[[2]] - 53.5] > 3., Quit[1]];
If[Abs[hashin[xt, 0, 0][[1]] - 1] > 1.*10^-12, Quit[1]];
If[Abs[hashin[0, -yc, 0][[4]] - 1] > 1.*10^-9, Quit[1]];

states = {
  {"fiber_tension", 1.0*10^9, 0., 0.},
  {"fiber_compression", -1.0*10^9, 0., 0.},
  {"matrix_tension", 0., 3.0*10^7, 0.},
  {"matrix_compression", 0., -1.2*10^8, 0.},
  {"pure_shear", 0., 0., 5.0*10^7},
  {"biaxial_tension_shear", 8.0*10^8, 2.0*10^7, 3.0*10^7},
  {"transverse_compression_shear", 0., -1.0*10^8, 4.0*10^7},
  {"fiber_compression_shear", -8.0*10^8, -5.0*10^7, 3.0*10^7}};

vals = Association[]; tols = Association[];
Do[
  {name, a, b, c} = st;
  h = hashin[a, b, c];
  p = puck[a, b, c];
  vals[name <> "_hashin"] = <|
    "values" -> (ToString[N[#, 16], CForm] & /@ h),
    "exact" -> "Hashin 1980 exposures (fiber-t, fiber-c, matrix-t, matrix-c)"|>;
  tols[name <> "_hashin_abs"] = 0.;
  tols[name <> "_hashin_rel"] = 1.*10^-9;
  vals[name <> "_puck_fiber"] = <|
    "values" -> {ToString[N[p[[1]], 16], CForm]},
    "exact" -> "Puck fiber exposure (max fiber-direction stress)"|>;
  tols[name <> "_puck_fiber_abs"] = 0.;
  tols[name <> "_puck_fiber_rel"] = 1.*10^-9;
  vals[name <> "_puck_iff"] = <|
    "values" -> {ToString[N[p[[2]], 16], CForm]},
    "exact" -> "Puck IFF exposure (0.05-degree action-plane reference scan)"|>;
  tols[name <> "_puck_iff_abs"] = 0.;
  tols[name <> "_puck_iff_rel"] = 1.*10^-3;
  vals[name <> "_puck_angle"] = <|
    "values" -> {ToString[N[p[[3]], 16], CForm]},
    "exact" -> "Puck fracture angle in degrees (reference scan)"|>;
  tols[name <> "_puck_angle_abs"] = 0.5;
  tols[name <> "_puck_angle_rel"] = 0.,
  {st, states}];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026)",
    "protocol" -> "solids-golden-v1",
    "case" -> "composites_failure",
    "units" -> "SI"|>,
  "inputs" -> <|
    "xt_pa" -> xt, "xc_pa" -> xc,
    "yt_pa" -> yt, "yc_pa" -> yc,
    "s12_pa" -> s12, "s23_pa" -> s23,
    "puck_p_t_par" -> ptPar, "puck_p_c_par" -> pcPar,
    "puck_p_t_perp" -> ptPerp, "puck_p_c_perp" -> pcPerp|>,
  "values" -> vals,
  "tolerances" -> tols|>;
WriteString[$Output, ExportString[golden, "RawJSON", "Compact" -> True], "\n"];
