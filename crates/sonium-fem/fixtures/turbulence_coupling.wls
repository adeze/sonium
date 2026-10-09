(* Independent Wolfram reference for the turbulence coupling seam.
   Derives, at 50-digit working precision: (1) P1 shape gradients on the unit
   simplex tet (closed form); (2) the [flow,k,omega] monolithic block
   composition for 1x1 blocks; (3) the mean eddy-viscosity reduction for a
   two-element fixture. The Rust side consumes turbulence_coupling_golden.json
   through the production assembly path; this script is the independent
   derivation. Run: wolframscript -script turbulence_coupling.wls
   (bounded: < 60 s, < 8 MB out).
   Engine: Wolfram Language 15.0.0 for Linux x86-64 (verified 2026-09-26). *)

(* --- (1) Unit simplex tet P1 shape gradients. --- *)
nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
jacobian = Transpose[{nodes[[2]] - nodes[[1]], nodes[[3]] - nodes[[1]],
   nodes[[4]] - nodes[[1]]}];
invJT = Inverse[Transpose[jacobian]];
refGrads = {{-1, -1, -1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
physGrads = (invJT . #) & /@ refGrads;

(* --- (2) 1x1 block composition: flow 2, scalar 3, zero coupling. --- *)
flowBlock = {{2}};
scalarBlock = {{3}};
monoBlock = ArrayFlatten[{{flowBlock, {{0}}, {{0}}}, {{{0}}, scalarBlock, {{0}}}, {{{0}}, {{0}}, scalarBlock}}];

(* --- (3) Mean eddy viscosity over two elements. --- *)
elementViscosities = {1/2, 1/4};
meanViscosity = Mean[elementViscosities];

prec = 50;
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.turbulence-coupling-two-cell.v1",
  "convention" -> "SI units; P1 exactness; [flow,k,omega] block ordering",
  "generator" -> "crates/sonium-fem/fixtures/turbulence_coupling.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "unit_tet_shape_gradients" -> N[physGrads, prec],
  "monolithic_block" -> N[monoBlock, prec],
  "mean_eddy_viscosity" -> N[meanViscosity, prec],
  "tolerances" -> <|
    "gradient_abs" -> 1.0*^-12,
    "block_abs" -> 1.0*^-12,
    "viscosity_abs" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
