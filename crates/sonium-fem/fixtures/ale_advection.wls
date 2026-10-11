(* Independent Wolfram reference for the sonium-fem ALE 1D advection kernel.
   Exact translation of a two-harmonic periodic profile at 50-digit working
   precision: u(x,t) = u0(Mod(x - v t, L)) sampled at cell centers for an
   1600-cell grid (L = 2, v = 1) at half and full periods. The Rust side
   consumes ale_advection_golden.json through Advection1d::run/max_error.
   Tolerance note: the production minmod+Heun scheme clips smooth extrema
   (locally first order); measured max error at this resolution is ~1.7e-3,
   and error_abs = 1.0e-2 below carries a 5x margin over that measurement.
   Run: wolframscript -script ale_advection.wls > ale_advection_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86-64. *)

prec = 50;
cells = 1600;
len = 2;
vel = 1;
cfl = 1/2;
times = {1, 2};
u0[x_] := Sin[Pi x] + 1/4 Sin[2 Pi x + 1];
exact[x_, t_] := u0[Mod[x - vel t, len]];
centers = Table[(i + 1/2) len/cells, {i, 0, cells - 1}];

golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.ale-advection.v1",
  "convention" -> "periodic linear advection; exact solution is translation of u0",
  "generator" -> "crates/sonium-fem/fixtures/ale_advection.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> prec,
  "cells" -> cells,
  "length" -> N[len, prec],
  "velocity" -> N[vel, prec],
  "cfl" -> N[cfl, prec],
  "times" -> N[times, prec],
  "centers" -> N[centers, prec],
  "exact_half" -> N[exact[#, times[[1]]] & /@ centers, prec],
  "exact_full" -> N[exact[#, times[[2]]] & /@ centers, prec],
  "tolerances" -> <|"error_abs" -> 1.0*^-2|>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
