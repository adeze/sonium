(* Wolfram block-preconditioner golden for sonium-solvers-preconditioners.
 *
 * Emits the exact inverse action of a 2-block nonsymmetric matrix as one
 * compact RawJSON document on stdout; redirect into
 *   crates/sonium-solvers-preconditioners/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-solvers-preconditioners/wolfram/block_golden.wls \
 *     > crates/sonium-solvers-preconditioners/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-solvers-preconditioners-20260921:
 * small-block actions, transposed variants and the Schur complement for
 * cross-checking apply/apply_transpose/apply_hermitian. The 2x2 blocks are
 * dense, so ILUT with zero dropping factorizes them exactly and the replay
 * test pins the full block-Jacobi action, not just its sparsity pattern.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];
vecNum[v_] := num /@ v;
matNum[m_] := Table[num[m[[j, k]]], {j, Length[m]}, {k, Length[m[[1]]]}];

(* Two dense nonsymmetric 2x2 blocks, no coupling: block Jacobi is exact. *)
a = {{2, -1, 0, 0}, {-3, 2, 0, 0}, {0, 0, 3, 1}, {0, 0, -2, 3}};
b = {1, 2, 3, 4};
ainv = Inverse[a];

(* Schur complement of the leading 2x2 block: S = A22 - A21 A11^-1 A12. *)
a11 = a[[1 ;; 2, 1 ;; 2]];
a12 = a[[1 ;; 2, 3 ;; 4]];
a21 = a[[3 ;; 4, 1 ;; 2]];
a22 = a[[3 ;; 4, 3 ;; 4]];
schur = a22 - a21 . Inverse[a11] . a12;

payload = <|
  "schema" -> "sonium-solvers-preconditioners.wolfram-golden.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " block-preconditioner golden"],
  "a" -> matNum[a],
  "b" -> vecNum[b],
  "inverse_b" -> vecNum[ainv . b],
  "inverse_tb" -> vecNum[Transpose[ainv] . b],
  "schur_leading_2x2" -> matNum[schur]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
