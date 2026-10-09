(* Wolfram iterative-solver golden for sonium-solvers-iterative.
 *
 * Emits two small exact systems with known spectra and solutions as one
 * compact RawJSON document on stdout; redirect into
 *   crates/sonium-solvers/crates/sonium-solvers-iterative/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-solvers/crates/sonium-solvers-iterative/wolfram/iterative_golden.wls \
 *     > crates/sonium-solvers/crates/sonium-solvers-iterative/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-solvers-iterative-20260921: small
 * nonnormal/indefinite/complex systems with exact solutions and known
 * spectra, so the replay test can check true solution error together with
 * the reported residual and backward error, plus an adjoint identity on the
 * nonsymmetric system. System A is the companion matrix of
 * (x-1)(x-2)(x-3): nonnormal with exact integer spectrum {1,2,3}.
 * System B is Hermitian indefinite, the MINRES model problem, with
 * eigenvalues (-1-Sqrt[17])/2 and (-1+Sqrt[17])/2.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];
vecNum[v_] := num /@ v;
matNum[m_] := Table[num[m[[j, k]]], {j, Length[m]}, {k, Length[m[[1]]]}];
cnum[z_] := <|"re" -> num[Re[z]], "im" -> num[Im[z]]|>;
cvecNum[v_] := cnum /@ v;
cmatNum[m_] := Table[cnum[m[[j, k]]], {j, Length[m]}, {k, Length[m[[1]]]}];

(* System A: nonnormal companion, spectrum {1, 2, 3}. *)
a = {{0, 1, 0}, {0, 0, 1}, {6, -11, 6}};
x1 = {1, 1, 1};
x2 = {1, -1, 2};
y = {1, 2, -1};

(* System B: Hermitian indefinite 2x2. *)
hb = {{1, 1 + I}, {1 - I, -2}};
hxx = {1 + I, 2 - I};

payload = <|
  "meta" -> <|
    "engine" -> StringJoin["Wolfram Engine ", $Version],
    "protocol" -> "audit-oracle-hermetic-v1"
  |>,
  "nonsym_companion_3x3" -> <|
    "a" -> matNum[a],
    "eigenvalues_exact" -> ToString /@ Eigenvalues[a],
    "characteristic_exact" -> ToString[CForm[CharacteristicPolynomial[a, t]]],
    "x1" -> vecNum[x1],
    "b1" -> vecNum[a . x1],
    "x2" -> vecNum[x2],
    "b2" -> vecNum[a . x2],
    "adjoint_probe_y" -> vecNum[y],
    "transpose_y" -> vecNum[Transpose[a] . y]
  |>,
  "hermitian_indefinite_2x2" -> <|
    "a" -> cmatNum[hb],
    "eigenvalues_exact" -> Map[ToString[CForm[#]] &, Eigenvalues[hb]],
    "eigenvalues_num" -> cvecNum[N[Eigenvalues[hb], prec]],
    "x" -> cvecNum[hxx],
    "b" -> cvecNum[hb . hxx]
  |>
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
