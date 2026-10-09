(* Wolfram exact linear-algebra golden for the solver family.
 *
 * Emits exact small matrices, vectors, products, decompositions and spectra
 * as one compact RawJSON document on stdout; redirect into
 *   crates/sonium-qa/wolfram/goldens/exact-linear-algebra-v1.json
 * Run from the repository root:
 *   wolframscript -file crates/sonium-qa/wolfram/exact_linear_algebra.wls \
 *     > crates/sonium-qa/wolfram/goldens/exact-linear-algebra-v1.json
 *
 * Shared independent oracle per the solver audit family (core, sparse,
 * compute, iterative, direct, eigen, facade): exact integer/rational inputs
 * evaluated at 50 digits, consumed hermetically by each crate's
 * wolfram_golden replay test. Integer results stay exact; rationals and
 * spectra are 50-digit decimals (plain f64-parseable CForm, no backticks).
 * The strided-GEMV, exact/cancellation-reduction, and rectangular-CSR
 * cases extend the compute audit envelope (awkward strides,
 * cancellation-sensitive reductions, rectangular SpMV/SpMM,
 * Hermitian/transpose actions on one independent reference).
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

(* 3x3 nonsymmetric with exact integer solution data. *)
aNon = {{2, 1, 0}, {-1, 3, 1}, {0, -2, 4}};
xNon = {1, -1, 2};
yNon = {1, 0, -1};

(* 4x3 rectangular. *)
aRect = {{1, 2, -1}, {0, 1, 3}, {-2, 0, 1}, {1, -1, 2}};
xRect = {1, 2, -1};
yRect = {1, 0, -1, 2};

(* 3x3 Hermitian with Gaussian-integer entries. *)
aHer = {{2, 1 + I, 0}, {1 - I, 3, -2*I}, {0, 2*I, 4}};
uHer = {1 + I, 2, -1 + 2*I};
vHer = {3 - I, -2 + I, 1};

(* Hilbert 4x4: exact rationals, integer inverse, SPD inertia. *)
hilb = Table[1/(i + j - 1), {i, 4}, {j, 4}];
hilbInv = Inverse[hilb];

(* Symmetric tridiagonal with closed-form spectrum 2-Sqrt[2], 2, 2+Sqrt[2]. *)
tri = {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

(* Ill-scaled diagonal. *)
scl = DiagonalMatrix[{10^-6, 1, 10^3, 10^6}];

(* Symmetric indefinite pivot with spectrum +-1. *)
indef = {{0, 1}, {1, 0}};

(* 3x2 full-column-rank SVD case. *)
svdA = {{1, 2}, {3, 4}, {5, 6}};

(* 3x4 strided GEMV case: exact integers; the compute replay pads rows
 * to lda = 6, exercising non-unit row stride on the same math. *)
aStride = {{1, -2, 3, 0}, {4, 5, -6, 7}, {-8, 9, 10, -11}};
xStride = {2, -1, 3, 1};
yStride = {1, -2, 3};

(* 2x3 complex strided GEMV case (Gaussian integers); the replay pads
 * to lda = 5 and checks NoTrans, Transpose and Hermitian actions. *)
aStrideC = {{1 + I, 2, -I}, {3, -2*I, 1 - I}};
xStrideC = {1, 2 + I, -1};
yStrideC = {2 - I, 1};

(* Exact small reductions (all results are exact in binary f64). *)
redDotU = {1, 2, 3, 4};
redDotV = {5, -6, 7, 8};
redNrmW = {3, 4};
redDotP = {1 + I, 2 - 3*I};
redDotQ = {4, 1 + I};

(* Cancellation-sensitive reduction: exact value 1, while naive f64
 * accumulation yields 0. The replay asserts the documented
 * n*eps*||x|.|y|| error bound, not exactness. *)
cancelX = {10^16, 1, -10^16};
cancelY = {1, 1, 1};

(* 2x3 CSR SpMV/SpMM case with exact integer arithmetic. *)
csrA = {{2, 0, -1}, {0, 3, 4}};
csrX = {5, -6, 7};
csrXt = {1, -1};
csrXRhs = {{5, -6}, {7, 8}, {-1, 2}};

conjDot[u_, v_] := Conjugate[u] . v;
plainDot[u_, v_] := u . v;

payload = <|
  "schema" -> "sonium-qa.exact-linear-algebra.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " exact linear-algebra golden"],
  "conventions" -> <|
    "dot_conj" -> "Sum Conjugate[u_i] v_i",
    "dot_plain" -> "Sum u_i v_i (no conjugation)",
    "norm2" -> "Sqrt[Sum Abs[u_i]^2]",
    "transpose_action" -> "A^T x without conjugation",
    "hermitian_action" -> "Conjugate[A]^T x"
  |>,
  "nonsym_3x3" -> <|
    "a" -> matNum[aNon],
    "x_exact" -> vecNum[xNon],
    "b" -> vecNum[aNon . xNon],
    "atx" -> vecNum[Transpose[aNon] . yNon],
    "y" -> vecNum[yNon]
  |>,
  "rect_4x3" -> <|
    "a" -> matNum[aRect],
    "x_exact" -> vecNum[xRect],
    "b" -> vecNum[aRect . xRect],
    "atx" -> vecNum[Transpose[aRect] . yRect],
    "y" -> vecNum[yRect]
  |>,
  "hermitian_3x3" -> <|
    "a" -> cmatNum[aHer],
    "u" -> cvecNum[uHer],
    "v" -> cvecNum[vHer],
    "dot_conj" -> cnum[conjDot[uHer, vHer]],
    "dot_plain" -> cnum[plainDot[uHer, vHer]],
    "norm2_u" -> num[Sqrt[conjDot[uHer, uHer]]],
    "hv" -> cvecNum[aHer . vHer],
    "hermitian_v" -> cvecNum[Conjugate[Transpose[aHer]] . vHer],
    "transpose_v" -> cvecNum[Transpose[aHer] . vHer]
  |>,
  "hilbert_4x4" -> <|
    "a" -> matNum[hilb],
    "x_exact" -> vecNum[{1, 1, 1, 1}],
    "b" -> vecNum[hilb . {1, 1, 1, 1}],
    "inverse" -> matNum[hilbInv],
    "eigenvalues" -> vecNum[Sort[N[Eigenvalues[N[hilb, prec + 10]], prec]]],
    "inertia" -> {4, 0, 0}
  |>,
  "tridiag_3x3" -> <|
    "a" -> matNum[tri],
    "eigenvalues" -> vecNum[Sort[N[Eigenvalues[N[tri, prec + 10]], prec]]],
    "inertia" -> {3, 0, 0}
  |>,
  "illscaled_diag_4x4" -> <|
    "diag" -> vecNum[Diagonal[scl]],
    "x_exact" -> vecNum[{1, 1, 1, 1}],
    "b" -> vecNum[scl . {1, 1, 1, 1}]
  |>,
  "sym_indef_2x2" -> <|
    "a" -> matNum[indef],
    "eigenvalues" -> vecNum[Sort[N[Eigenvalues[N[indef, prec + 10]], prec]]],
    "inertia" -> {1, 1, 0}
  |>,
  "svd_3x2" -> <|
    "a" -> matNum[svdA],
    "singular_values" -> vecNum[Sort[N[SingularValueList[N[svdA, prec + 10], Tolerance -> 0], prec], Greater]]
  |>,
  "strided_gemv_3x4" -> <|
    "a" -> matNum[aStride],
    "x" -> vecNum[xStride],
    "b" -> vecNum[aStride . xStride],
    "y" -> vecNum[yStride],
    "atx" -> vecNum[Transpose[aStride] . yStride],
    "lda" -> 6
  |>,
  "strided_gemv_c64_2x3" -> <|
    "a" -> cmatNum[aStrideC],
    "x" -> cvecNum[xStrideC],
    "b" -> cvecNum[aStrideC . xStrideC],
    "y" -> cvecNum[yStrideC],
    "hermitian_y" -> cvecNum[Conjugate[Transpose[aStrideC]] . yStrideC],
    "transpose_y" -> cvecNum[Transpose[aStrideC] . yStrideC],
    "lda" -> 5
  |>,
  "reduction_exact" -> <|
    "dot_u" -> vecNum[redDotU],
    "dot_v" -> vecNum[redDotV],
    "dot" -> num[redDotU . redDotV],
    "norm_w" -> vecNum[redNrmW],
    "norm2" -> num[Sqrt[redNrmW . redNrmW]],
    "cdot_p" -> cvecNum[redDotP],
    "cdot_q" -> cvecNum[redDotQ],
    "dot_conj" -> cnum[conjDot[redDotP, redDotQ]],
    "dot_plain" -> cnum[plainDot[redDotP, redDotQ]]
  |>,
  "reduction_cancellation" -> <|
    "x" -> vecNum[cancelX],
    "y" -> vecNum[cancelY],
    "exact" -> num[cancelX . cancelY],
    "bound" -> num[Length[cancelX]*$MachineEpsilon*Total[Abs[cancelX]*Abs[cancelY]]]
  |>,
  "csr_spmv_2x3" -> <|
    "row_ptr" -> {0, 2, 4},
    "col_idx" -> {0, 2, 1, 2},
    "values" -> vecNum[{2, -1, 3, 4}],
    "x" -> vecNum[csrX],
    "y" -> vecNum[csrA . csrX],
    "xt" -> vecNum[csrXt],
    "at_xt" -> vecNum[Transpose[csrA] . csrXt],
    "x_rhs" -> matNum[csrXRhs],
    "y_rhs" -> matNum[csrA . csrXRhs]
  |>
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
