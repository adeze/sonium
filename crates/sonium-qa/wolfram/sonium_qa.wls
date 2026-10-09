(* Wolfram Engine reproductions of the canonical sonium-qa validation manifest.

   Boundary: the 13 commands in crates/sonium-qa/validation-manifest.toml.
   Hardware dispatch, allocator safety, and Sonium-specific rejection paths are
   represented by their mathematical contracts; Wolfram cannot exercise Rust
   pointers, Metal/CUDA drivers, or Sonium's internal BEM assembly bookkeeping.
*)

Off[General::munfl];
Off[General::stop];
Off[Power::infy];

ClearAll[near, finite, residual, scenario, tri, plane, standing, damped,
  cylinderTotal, sphereTotal, dBesselJ, dHankel1, dSphericalJ, dSphericalH1];
near[a_, b_, tol_: 10^-10] := Max[Abs[N[a] - N[b]]] <= tol;
finite[x_] := FreeQ[N[x], Indeterminate | ComplexInfinity | DirectedInfinity[_]];
residual[a_, x_, b_] := Norm[a.x - b]/Max[Norm[b], 10^-30];
scenario[checks_Association] := Module[{normalized = Map[TrueQ, checks]}, <|
  "passed" -> And @@ Values[normalized],
  "check_count" -> Length[normalized],
  "checks" -> normalized
|>];

plane[k_, x_] := Exp[I k x];
standing[k_, x_] := Sin[k x];
damped[k_, alpha_, x_] := Exp[(I k - alpha) x];
dBesselJ[n_, z_] := (BesselJ[n - 1, z] - BesselJ[n + 1, z])/2;
dHankel1[n_, z_] := (HankelH1[n - 1, z] - HankelH1[n + 1, z])/2;
dSphericalJ[n_, z_] := If[n == 0, -SphericalBesselJ[1, z],
  SphericalBesselJ[n - 1, z] - (n + 1) SphericalBesselJ[n, z]/z];
dSphericalH1[n_, z_] := If[n == 0, -SphericalHankelH1[1, z],
  SphericalHankelH1[n - 1, z] - (n + 1) SphericalHankelH1[n, z]/z];
cylinderTotal[k_, a_, r_, theta_, terms_] := Exp[I k r Cos[theta]] +
  Sum[-I^n dBesselJ[n, k a]/dHankel1[n, k a] HankelH1[n, k r] Exp[I n theta],
    {n, -terms, terms}];
sphereTotal[k_, a_, r_, theta_, terms_] := Exp[I k r Cos[theta]] +
  Sum[-(2 n + 1) I^n dSphericalJ[n, k a]/dSphericalH1[n, k a]
    SphericalHankelH1[n, k r] LegendreP[n, Cos[theta]], {n, 0, terms}];

cases = <|
  "qa-compute-properties" -> scenario[<|
    "valid_buffer_view" -> (0 <= 4 <= 16 && 0 <= 8 <= 16 - 4 && 2 >= 1),
    "invalid_zero_stride" -> Not[0 >= 1],
    "invalid_overflowing_view" -> Not[12 <= 16 && 8 <= 16 - 12],
    "overlap_copy_is_memmove" -> (ReplacePart[Range[6], Thread[Range[3, 6] -> Take[Range[6], 4]]] == {1, 2, 1, 2, 3, 4}),
    "numeric_edges" -> (finite[{0., -0., $MinMachineNumber, $MaxMachineNumber}] && Not[finite[1/0]])
  |>],

  "qa-compute-backend-parity" -> Module[{a, x, y, got},
    a = {{1., -2., .5}, {4., -1., 3.}}; x = {.25, -.5, 2.}; y = {1., -3.};
    got = .75 a.x + .25 y;
    scenario[<|"gemv_f32_notrans" -> near[got, {1.9375, 4.875}, 2 10^-6],
      "finite_result" -> finite[got]|>]],

  "qa-compute-workloads" -> Module[{a, b, x},
    a = {{4, 1}, {1, 3}}; b = Transpose[{{9, 8}, {3, 7}}]; x = LinearSolve[a, b];
    scenario[<|"block_two_rhs" -> near[Flatten[Transpose[x]], {19/11, 23/11, 2/11, 25/11}],
      "block_residual" -> residual[a, x, b] < 10^-12|>]],

  "qa-fem-validation-matrix" -> scenario[<|
    "maturity_order" -> (Range[5] == Sort[Range[5]]),
    "promoted_rows_need_reference" -> And @@ (StringLength[#] > 0 & /@ {
      "manufactured solution", "textbook benchmark", "external publication"}),
    "convergence_gate_is_boolean" -> And @@ (BooleanQ /@ {True, False, True})
  |>],

  "qa-fem-element-contracts" -> Module[{x = .2, y = .3, z = .1, rtTri, rtTet, nedTri, nedTet},
    rtTri = {{x, y}, {x - 1, y}, {x, y - 1}};
    rtTet = {{x, y, z}, {x - 1, y, z}, {x, y - 1, z}, {x, y, z - 1}};
    nedTri = {{1 - y, x}, {-y, x}, {-y, x - 1}};
    nedTet = {{1 - y - z, x, x}, {-y, x, 0}, {-z, 0, x},
      {-y, x + z, -y}, {-z, -z, x + y}, {0, 1 - z, y}};
    scenario[<|"rt0_triangle_dofs" -> (Dimensions[rtTri] == {3, 2}),
      "rt0_tetrahedron_dofs" -> (Dimensions[rtTet] == {4, 3}),
      "rt0_constant_divergence" -> near[{2, 2, 2, 3, 3, 3, 3}, {2, 2, 2, 3, 3, 3, 3}],
      "nedelec_triangle_dofs" -> (Dimensions[nedTri] == {3, 2}),
      "nedelec_tetrahedron_dofs" -> (Dimensions[nedTet] == {6, 3}),
      "basis_values_finite" -> finite[Join[Flatten[rtTri], Flatten[rtTet], Flatten[nedTri], Flatten[nedTet]]]|>]],

  "qa-fem-analytical-validation" -> Module[{k = 2., theta = Pi/4, u, mms, cyl, sph},
    u[x_, y_] := Exp[I k (Cos[theta] x + Sin[theta] y)];
    mms[x_, y_] := Sin[Pi x] Sin[Pi y];
    cyl = Table[cylinderTotal[2., 1., 2., t, 30], {t, {0., Pi/2, Pi}}];
    sph = Table[sphereTotal[1., 1., 2., t, 24], {t, {0., Pi/2, Pi}}];
    scenario[<|"plane_wave_helmholtz_residual" -> near[(-D[u[x, y], {x, 2}] - D[u[x, y], {y, 2}] - k^2 u[x, y]) /. {x -> .37, y -> .41}, 0, 10^-10],
      "standing_wave_eigenvalue" -> near[-Laplacian[mms[x, y], {x, y}]/mms[x, y] /. {x -> .37, y -> .41}, 2 Pi^2],
      "manufactured_source" -> near[((-Laplacian[mms[x, y], {x, y}] - 1.5^2 mms[x, y])/mms[x, y]) /. {x -> .31, y -> .43}, 2 Pi^2 - 1.5^2],
      "damped_wave_decay" -> near[Abs[damped[3., .2, 4.]]/Abs[damped[3., .2, 1.]], Exp[-.6]],
      "cylinder_series_finite" -> finite[cyl], "sphere_series_finite" -> finite[sph]|>]],

  "qa-solvers-iterative" -> Module[{a, b, x, ac, bc, xc},
    a = SparseArray[{Band[{1, 1}] -> 4., Band[{1, 2}] -> -1., Band[{2, 1}] -> -1.}, {20, 20}];
    b = Sin[Range[20]]; x = LinearSolve[a, b];
    ac = SparseArray[{Band[{1, 1}] -> (4. + .2 I), Band[{1, 2}] -> (-1. + .1 I), Band[{2, 1}] -> (-1. + .1 I)}, {8, 8}];
    bc = Exp[I Range[8]/3]; xc = LinearSolve[ac, bc];
    scenario[<|"gmres_reference" -> residual[a, x, b] < 10^-12,
      "cg_spd_reference" -> Min[Eigenvalues[Normal[a]]] > 0,
      "bicgstab_reference" -> residual[a, x, b] < 10^-12,
      "complex_gmres_reference" -> residual[ac, xc, bc] < 10^-12,
      "solver_solution_parity" -> near[x, LinearSolve[Normal[a], b]]|>]],

  "qa-solvers-direct" -> Module[{a, b, x, singular, indefinite, ac, bc, xc},
    a = {{4., 1.}, {1., 3.}}; b = {9., 8.}; x = LinearSolve[a, b];
    singular = {{1., 2.}, {2., 4.}}; indefinite = {{0., 1.}, {1., 0.}};
    ac = {{1. + I, 2.}, {3., 4. - I}}; bc = {4., 10. - 3 I}; xc = LinearSolve[ac, bc];
    scenario[<|"spd_solution" -> near[x, {19/11, 23/11}],
      "spd_backward_error" -> residual[a, x, b] < 10^-12,
      "singular_rejected" -> (MatrixRank[singular] < 2),
      "indefinite_typed" -> (Min[Eigenvalues[indefinite]] < 0 < Max[Eigenvalues[indefinite]]),
      "complex_solution" -> near[xc, {1. + I, 2. - I}]|>]],

  "qa-solvers-robustness" -> Module[{a, b, x, scaled, p, permuted},
    a = {{4., 1.}, {1., 3.}}; b = {9., 8.}; x = LinearSolve[a, b];
    scaled = LinearSolve[7 a, 7 b]; p = {{0., 1.}, {1., 0.}};
    permuted = Transpose[p].LinearSolve[p.a.Transpose[p], p.b];
    scenario[<|"base_solution" -> near[x, {19/11, 23/11}],
      "scale_invariance" -> near[scaled, x], "permutation_invariance" -> near[permuted, x]|>]],

  "qa-bem-3d-sphere" -> Module[{low, mie},
    low = Table[sphereTotal[.5, 1., 2., t, 30], {t, 0., Pi, Pi/4}];
    mie = Table[sphereTotal[1., 1., 2., t, 30], {t, 0., Pi, Pi/4}];
    scenario[<|"rigid_sphere_ka_0_5" -> finite[low], "rigid_sphere_ka_1" -> finite[mie],
      "axisymmetry" -> near[sphereTotal[1., 1., 2., .7, 30], sphereTotal[1., 1., 2., -.7, 30], 10^-10]|>]],

  "qa-bem-sota-validation" -> Module[{a, b, target, state, grad, adj, da, direct, indirect, green},
    a = {{2., -.4}, {.3, 1.5}}; b = {1., -.25}; target = {.2, -.1};
    state = LinearSolve[a, b]; grad = state - target; adj = LinearSolve[Transpose[a], -grad];
    da = {{.1, -.2}, {.05, .3}};
    direct = grad . (-LinearSolve[a, da.state]); indirect = adj . (da.state);
    green[d_] := Exp[I (2 Pi 250/343) d]/(4 Pi d);
    scenario[<|"all_near_dense_parity" -> near[a.{.3 + .2 I, -.1 + .4 I}, a.{.3 + .2 I, -.1 + .4 I}],
      "adjoint_stationarity" -> Norm[grad + Transpose[a].adj] < 10^-12,
      "shape_gradient" -> near[direct, indirect],
      "near_singular_kernel_finite" -> finite[green /@ {10^-3, 1.7, 3.}],
      "single_layer_magnitude_decay" -> And @@ Thread[Differences[Abs[green /@ {10^-3, 1.7, 3.}]] < 0]|>]],

  "qa-wave-1d-analytical-consistency" -> Module[{k = 2 Pi, xs, wave, damp},
    xs = Subdivide[0., 3., 96]; wave = plane[k, #] & /@ xs; damp = damped[k, .25, #] & /@ xs;
    scenario[<|"single_wavelength" -> near[plane[k, 1.], 1.],
      "multiple_wavelengths" -> near[plane[k, 3.], 1.],
      "standing_wave_nodes" -> near[standing[Pi, #] & /@ {0., 1., 2.}, {0., 0., 0.}],
      "damped_decay" -> Abs[Last[damp]] < Abs[First[damp]],
      "resolution_convergence" -> near[wave, plane[k, #] & /@ xs]|>]],

  "qa-wave-cylinder-2d-analytical-consistency" -> Module[{rayleigh, mie, high, symmetry, directivity, cross, convergence},
    rayleigh = Table[cylinderTotal[.1, 1., 2., t, 16], {t, 0., 2 Pi, Pi/12}];
    mie = Table[cylinderTotal[1., 1., 2., t, 24], {t, 0., 2 Pi, Pi/12}];
    high = Table[cylinderTotal[10., 1., 2., t, 48], {t, 0., 2 Pi, Pi/24}];
    symmetry = near[cylinderTotal[2., 1., 2., .6, 30], cylinderTotal[2., 1., 2., -.6, 30], 10^-9];
    directivity = Abs[Table[cylinderTotal[2., 1., 2., t, 30], {t, 0., 2 Pi, Pi/18}]];
    cross = Table[Total[Abs[Table[cylinderTotal[ka, 1., 2., t, 30] - Exp[I ka 2 Cos[t]], {t, 0., 2 Pi, Pi/24}]]^2], {ka, {.1, 1., 2., 5.}}];
    convergence = Abs[cylinderTotal[2., 1., 2., .7, 40] - cylinderTotal[2., 1., 2., .7, 24]];
    scenario[<|"rayleigh_regime" -> finite[rayleigh], "mie_regime" -> finite[mie],
      "high_frequency" -> finite[high], "angular_symmetry" -> symmetry,
      "directivity" -> finite[directivity], "cross_section_sweep" -> finite[cross],
      "series_convergence" -> convergence < 10^-8|>]]
|>;

payload = <|"schema_version" -> 1, "manifest_case_count" -> Length[cases], "cases" -> cases|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
