(* Independent Wolfram reference for the sonium-coupling aperture seam.
   Derives, at 50-digit working precision, the active-power audit and the
   measure-weighted adjoint identity for the two-triangle aperture used by
   the seam-contract test. The Rust side consumes coupling_seam_golden.json;
   this script is the independent derivation, not a reprint of the Rust code.
   Run: wolframscript -script coupling_seam.wls (bounded: < 60 s, < 8 MB out).
   Engine: Wolfram Language 15.0.0 for Linux x86-64 (verified 2026-09-26). *)

areas = {1/2, 1/4};
(* Source measures from lumping a third of each incident face measure. *)
sourceMeasures = {1/6, 1/4, 1/4, 1/12};
weights = {{1/3, 1/3, 1/3, 0}, {0, 1/3, 1/3, 1/3}};

pNodal = {2 + I, 1 - 2 I};
pExt = {1 + 0 I, 1 + 0 I};
vAp = {1 + 0 I, 1 + 0 I};

powerSide[press_, vel_] :=
  1/2 Total[Re[press Conjugate[vel]] areas];
pInt = powerSide[pNodal, vAp];
pExtW = powerSide[pExt, vAp];

(* Adjoint check vectors (same as the Rust seam test). *)
xVec = {1 + I/2, -2 + I, 1/4 - 3 I/2, 3};
yVec = {1/2 + I/4, -1 + 2 I};
txVec = weights . xVec;
ahyVec = Table[
  Sum[yVec[[t]] weights[[t, s]] areas[[t]]/sourceMeasures[[s]], {t, 2}],
  {s, 4}];
lhs = Conjugate[txVec] . (yVec areas);
rhs = Conjugate[xVec] . (ahyVec sourceMeasures);

prec = 50;
cPair[z_] := {N[Re[z], prec], N[Im[z], prec]};
cPairList[zs_] := cPair /@ zs;
golden = <|
  "schema_version" -> 1,
  "case_id" -> "coupling.seam-two-triangle.v1",
  "convention" -> "exp(+ikr) phasor, outward normal velocity, SI units",
  "generator" -> "crates/sonium-coupling/fixtures/coupling_seam.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "faces" -> {{0, 1, 2}, {1, 3, 2}},
  "areas_m2" -> N[areas, prec],
  "source_measures_m2" -> N[sourceMeasures, prec],
  "interior_pressure_per_triangle_pa" -> cPairList[pNodal],
  "exterior_pressure_per_triangle_pa" -> cPairList[pExt],
  "velocity_per_triangle_m_s" -> cPairList[vAp],
  "interior_active_power_w" -> N[pInt, prec],
  "exterior_active_power_w" -> N[pExtW, prec],
  "adjoint_lhs" -> cPair[lhs],
  "adjoint_rhs" -> cPair[rhs],
  "adjoint_mismatch" -> N[Abs[lhs - rhs], prec],
  "tolerances" -> <|
    "power_abs_w" -> 1.0*^-12,
    "adjoint_abs" -> 1.0*^-12
  |>
|>;

json = ExportString[golden, "JSON"];
WriteString[$Output, json];
