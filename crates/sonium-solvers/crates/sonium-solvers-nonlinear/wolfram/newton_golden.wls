(* Wolfram Newton-system golden for sonium-solvers-nonlinear.
 *
 * Emits exact solutions, residuals and Jacobians for F(x,y) = (x^2+y^2-2,
 * x-y) as one compact RawJSON document on stdout; redirect into
 *   crates/sonium-solvers-nonlinear/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-solvers-nonlinear/wolfram/newton_golden.wls \
 *     > crates/sonium-solvers-nonlinear/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-solvers-nonlinear-20260921: Newton
 * iterates are implementation-specific and NOT goldenized; solutions,
 * residuals and Jacobians are.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];
vecNum[v_] := num /@ v;
matNum[m_] := Table[num[m[[j, k]]], {j, Length[m]}, {k, Length[m[[1]]]}];

f[{x_, y_}] := {x^2 + y^2 - 2, x - y};
jac[{x_, y_}] := {{2*x, 2*y}, {1, -1}};

solutions = {{1, 1}, {-1, -1}};
probes = {{2, 1/2}, {0, 1}, {-2, 1}};

payload = <|
  "schema" -> "sonium-solvers-nonlinear.wolfram-golden.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " Newton-system golden"],
  "system" -> "F(x,y) = (x^2+y^2-2, x-y)",
  "solutions" -> vecNum /@ N[solutions, prec + 10],
  "probes" -> Table[<|
    "point" -> vecNum[N[p, prec + 10]],
    "residual" -> vecNum[f[p]],
    "jacobian" -> matNum[jac[p]]
  |>, {p, probes}]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
