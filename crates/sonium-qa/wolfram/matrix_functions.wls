(* Wolfram matrix-function golden for the solver family.
 *
 * Emits matrix-exponential, nonsymmetric-spectrum, wide-SVD and adjoint
 * references as one compact RawJSON document on stdout; redirect into
 *   crates/sonium-qa/wolfram/goldens/matrix-functions-v1.json
 * Run from the repository root:
 *   wolframscript -file crates/sonium-qa/wolfram/matrix_functions.wls \
 *     > crates/sonium-qa/wolfram/goldens/matrix-functions-v1.json
 *
 * Companion to exact_linear_algebra.wls for the dense matrix-function APIs
 * (sonium-solvers-direct expm / wide SVD, sonium-solvers-eigen
 * nonsymmetric_eigenvalues, sonium-solvers-core DenseMatrix::adjoint):
 * exact integer/Gaussian-integer inputs evaluated at 50 digits, consumed
 * hermetically by each crate's wolfram_golden replay test. Integer results
 * stay exact; transcendentals and spectra are 50-digit decimals (plain
 * f64-parseable CForm, no backticks). Eigenvalues sort by {Re, Im} so the
 * replay can match them deterministically after its own sort.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];
cnum[z_] := <|"re" -> num[Re[z]], "im" -> num[Im[z]]|>;
vecNum[v_] := num /@ v;
cvecNum[v_] := cnum /@ v;
matNum[m_] := Table[num[m[[j, k]]], {j, Length[m]}, {k, Length[m[[1]]]}];
cmatNum[m_] := Table[cnum[m[[j, k]]], {j, Length[m]}, {k, Length[m[[1]]]}];
sortedSpectrum[m_] := SortBy[N[Eigenvalues[N[m, prec + 10]], prec], {Re[#], Im[#]} &];

(* 3x3 nonsymmetric exponential case. *)
expmA = {{1, 2, 0}, {-1, 0, 1}, {0, 1, 2}};

(* 2x2 skew generator: exp([[0,-1],[1,0]]) is the rotation by 1 radian. *)
skewA = {{0, -1}, {1, 0}};

(* 4x4 nonsymmetric with mixed real/complex spectrum. *)
eigA = {{2, 1, 0, 0}, {-1, 2, 1, 0}, {0, -1, 1, 2}, {1, 0, -1, 3}};

(* Companion of (x-1)(x^2+1) = x^3-x^2+x-1: exact spectrum {1, I, -I}. *)
compA = {{0, 0, 1}, {1, 0, -1}, {0, 1, 1}};

(* 2x4 full-row-rank wide SVD case. *)
svdW = {{1, 2, 3, 4}, {5, 6, 7, 8}};

(* 2x3 Gaussian-integer adjoint case. *)
adjA = {{1 + I, 2, -I}, {3, -2*I, 1 - I}};

payload = <|
  "schema" -> "sonium-qa.matrix-functions.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " matrix-function golden"],
  "conventions" -> <|
    "matrix_exp" -> "MatrixExp[A]",
    "spectrum_sort" -> "ascending Re, then ascending Im",
    "singular_values" -> "descending SingularValueList",
    "adjoint" -> "Conjugate[A]^T"
  |>,
  "expm_nonsym_3x3" -> <|
    "a" -> matNum[expmA],
    "exp" -> matNum[N[MatrixExp[N[expmA, prec + 10]], prec]]
  |>,
  "expm_skew_2x2" -> <|
    "a" -> matNum[skewA],
    "exp" -> matNum[N[MatrixExp[N[skewA, prec + 10]], prec]]
  |>,
  "nonsym_eig_4x4" -> <|
    "a" -> matNum[eigA],
    "eigenvalues" -> cvecNum[sortedSpectrum[eigA]]
  |>,
  "companion_eig_3x3" -> <|
    "a" -> matNum[compA],
    "eigenvalues" -> cvecNum[sortedSpectrum[compA]]
  |>,
  "svd_wide_2x4" -> <|
    "a" -> matNum[svdW],
    "singular_values" -> vecNum[Sort[N[SingularValueList[N[svdW, prec + 10], Tolerance -> 0], prec], Greater]]
  |>,
  "adjoint_2x3" -> <|
    "a" -> cmatNum[adjA],
    "adjoint" -> cmatNum[Conjugate[Transpose[adjA]]]
  |>
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
