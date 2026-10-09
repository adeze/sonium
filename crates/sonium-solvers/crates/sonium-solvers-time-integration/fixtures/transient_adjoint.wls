(* Independent Wolfram reference for the transient discrete adjoint seam.
   Derives, at 50-digit working precision, the closed-form backward sweep for
   the two-step fixture used by the frozen pcfan regression test: diagonal
   state Jacobians, one BDF history block, and the design-gradient
   accumulation. The Rust side consumes transient_adjoint_golden.json; this
   script is the independent derivation (direct inverses, no GMRES).
   Run: wolframscript -script transient_adjoint.wls (bounded: < 60 s, < 8 MB).
   Engine: Wolfram Language 15.0.0 for Linux x86-64 (verified 2026-09-26). *)

a0 = {2, 3};
a1 = {2, 3};
hDiag = {1/2, 1};
phi0 = {1, 2};
phi1 = {3, 4};
b0 = {1, 2};
b1 = {2, 1};
e0 = 1/10;
e1 = 1/5;

lambda1 = phi1/a1;
rhs0 = phi0 - hDiag*lambda1;
lambda0 = rhs0/a0;
gradient = (e0 + e1) - (b0 . lambda0) - (b1 . lambda1);

prec = 50;
golden = <|
  "schema_version" -> 1,
  "case_id" -> "solvers.transient-adjoint-two-step.v1",
  "convention" -> "R_n = 0 residuals; A_n^T λ_n = φ_{x_n} - Σ_{m>n} H_{m,n}^T λ_m; g = Σ(φ_d - R_d^T λ)",
  "generator" -> "crates/sonium-solvers/crates/sonium-solvers-time-integration/fixtures/transient_adjoint.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "lambda_1" -> N[lambda1, prec],
  "lambda_0" -> N[lambda0, prec],
  "design_gradient" -> {N[gradient, prec]},
  "tolerances" -> <|
    "adjoint_abs" -> 1.0*^-10,
    "gradient_abs" -> 1.0*^-10
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
