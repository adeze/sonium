(* Independent Wolfram reference for the sonium-fem transfer seam.
   Derives, at 50-digit working precision: (1) the lumped-integral
   conservation check for a P1 conservative transfer of a linear field on the
   unit simplex tet (exact interpolation, zero correction); (2) the
   inverse-distance panel weights for the two-panel seam fixture; (3) both
   sides of the measure-weighted forward/adjoint dot-product identity for the
   tri-3 transfer used by the seam tests. The Rust side consumes
   transfer_seam_golden.json; this script is the independent derivation.
   Run: wolframscript -script transfer_seam.wls (bounded: < 60 s, < 8 MB out).
   Engine: Wolfram Language 15.0.0 for Linux x86-64 (verified 2026-09-26). *)

(* --- (1) Unit simplex tet, linear field f = x + 2 y + 3 z. --- *)
tetNodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
tetVolume = Abs[Det[Differences[tetNodes]]]/6;
fieldValues = #[[1]] + 2 #[[2]] + 3 #[[3]] & /@ tetNodes;
lumpedIntegral = tetVolume/4 Total[fieldValues];

(* --- (2) Two-panel inverse-distance weights. --- *)
srcCenters = {{0, 0, 0}, {0, 1, 0}};
tgtCenter = {1/1000, 0, 0};
dist2 = Total[(# - tgtCenter)^2] & /@ srcCenters;
floor = 1/10^10;
wts = 1/(Max[Sqrt[#], floor]^2) & /@ dist2;
idwValue = (wts . {3, 3})/Total[wts];

(* --- (3) Tri-3 transfer adjoint identity. --- *)
areas = {1/2, 1/4};
srcMeas = {1/6, 1/4, 1/4, 1/12};
weights = {{1/3, 1/3, 1/3, 0}, {0, 1/3, 1/3, 1/3}};
xVec = {1 + I/2, -2 + I, 1/4 - 3 I/2, 3};
yVec = {1/2 + I/4, -1 + 2 I};
tx = weights . xVec;
ahy = Table[
  Sum[yVec[[t]] weights[[t, s]] areas[[t]]/srcMeas[[s]], {t, 2}],
  {s, 4}];
lhs = Conjugate[tx] . (yVec areas);
rhs = Conjugate[xVec] . (ahy srcMeas);

prec = 50;
cPair[z_] := {N[Re[z], prec], N[Im[z], prec]};
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.transfer-seam-two-cell.v1",
  "convention" -> "SI units; P1 exactness; outward/inverse-distance contracts per module docs",
  "generator" -> "crates/sonium-fem/fixtures/transfer_seam.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "tet_volume" -> N[tetVolume, prec],
  "tet_lumped_integral" -> N[lumpedIntegral, prec],
  "tet_correction" -> 0,
  "panel_idw_weights" -> N[wts, prec],
  "panel_idw_value" -> N[idwValue, prec],
  "transfer_adjoint_lhs" -> cPair[lhs],
  "transfer_adjoint_rhs" -> cPair[rhs],
  "transfer_adjoint_mismatch" -> N[Abs[lhs - rhs], prec],
  "tolerances" -> <|
    "conservation_abs" -> 1.0*^-12,
    "idw_abs" -> 1.0*^-9,
    "adjoint_abs" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
