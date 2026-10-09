(* Wolfram Rosenbrock golden for sonium-solvers-optimization.
 *
 * Emits exact f/gradient/Hessian data for f(x,y) = (1-x)^2 + 100*(y-x^2)^2
 * as one compact RawJSON document on stdout; redirect into
 *   crates/sonium-solvers-optimization/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-solvers-optimization/wolfram/rosenbrock_golden.wls \
 *     > crates/sonium-solvers-optimization/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-solvers-optimization-20260921: exact
 * objective gradients and Hessians at fixed points. Optimizer trajectories
 * are implementation-specific and asserted only by convergence to the
 * analytic minimizer (1,1).
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];
vecNum[v_] := num /@ v;
matNum[m_] := Table[num[m[[j, k]]], {j, Length[m]}, {k, Length[m[[1]]]}];

f[{x_, y_}] := (1 - x)^2 + 100*(y - x^2)^2;
grad[{x_, y_}] := D[f[{xx, yy}], {{xx, yy}}] /. {xx -> x, yy -> y};
hess[{x_, y_}] := D[f[{xx, yy}], {{xx, yy}, 2}] /. {xx -> x, yy -> y};

points = {{0, 0}, {1, 1}, {-1, 2}, {1/2, -1}, {-6/5, 1}};

payload = <|
  "schema" -> "sonium-solvers-optimization.wolfram-golden.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " Rosenbrock golden"],
  "function" -> "f(x,y) = (1-x)^2 + 100*(y-x^2)^2",
  "minimizer" -> vecNum[{1, 1}],
  "points" -> Table[<|
    "point" -> vecNum[N[p, prec + 10]],
    "value" -> num[f[p]],
    "gradient" -> vecNum[grad[p]],
    "hessian" -> matNum[hess[p]]
  |>, {p, points}]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
