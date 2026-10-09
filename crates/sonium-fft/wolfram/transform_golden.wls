(* Wolfram exact-transform golden for sonium-fft.
 *
 * Emits exact small DFT/DCT-II/DST-II matrices, transform outputs,
 * a linear convolution and rectangular-window STFT frame spectra as one
 * compact RawJSON document on stdout; redirect into
 *   crates/sonium-fft/fixtures/wolfram-golden-v1.json
 * Run from the repository root:
 *   wolframscript -file crates/sonium-fft/wolfram/transform_golden.wls \
 *     > crates/sonium-fft/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-fft-20260921: exact matrices evaluated
 * at 50 digits with Sonium's unnormalized signs, so a forward/inverse round
 * trip cannot hide paired sign/scale bugs.
 *
 * Conventions (verified against sonium-fft sources):
 * - forward DFT M[j,k] = Exp[-2 Pi I j k / N], unnormalized
 *   (= Sqrt[N] * Conjugate[FourierMatrix[N]], stated explicitly instead of
 *   relying on FourierParameters presets);
 * - inverse DFT = Conjugate[M]/N;
 * - DCT-II out[k] = Sum x[n] Cos[Pi (n+1/2) k / N] (dct.rs, unnormalized);
 * - DST-II out[k] = Sum x[n] Sin[Pi (n+1/2) (k+1) / N] (dct.rs, unnormalized).
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
(* CForm decimals without backtick precision marks; plain f64-parseable. *)
num[value_] := ToString[CForm[N[value, prec]]];
cnum[z_] := <|"re" -> num[Re[z]], "im" -> num[Im[z]]|>;

(* Unnormalized forward DFT matrix from exact roots of unity, carried at
   60 digits so downstream evaluation never hits exact-zero significance
   tracking (N::meprec); outputs are rounded to 50 digits by num[]. *)
dftMatrix[n_] := N[Table[
  Sqrt[n]*Conjugate[FourierMatrix[n][[j + 1, k + 1]]],
  {j, 0, n - 1}, {k, 0, n - 1}
], prec + 10];
dftApply[mat_, vec_] := mat . vec;

dftCase[n_, vec_] := Module[{mat = dftMatrix[n], v = N[vec, prec + 10], fwd, inv},
  fwd = dftApply[mat, v];
  inv = (1/n)*Conjugate[mat] . fwd;
  <|
    "n" -> n,
    "matrix_re" -> Table[num[Re[mat[[j, k]]]], {j, n}, {k, n}],
    "matrix_im" -> Table[num[Im[mat[[j, k]]]], {j, n}, {k, n}],
    "input" -> cnum /@ vec,
    "forward" -> cnum /@ fwd,
    "inverse_of_forward" -> cnum /@ inv
  |>
];

dctMatrix[n_] := N[Table[
  Cos[Pi*(m + 1/2)*k/n], {k, 0, n - 1}, {m, 0, n - 1}
], prec + 10];
dstMatrix[n_] := N[Table[
  Sin[Pi*(m + 1/2)*(k + 1)/n], {k, 0, n - 1}, {m, 0, n - 1}
], prec + 10];

realCase[name_, mat_, vec_] := Module[{out = mat . N[vec, prec + 10]},
  <|
    "n" -> Length[vec],
    "matrix" -> Table[num[mat[[j, k]]], {j, Length[vec]}, {k, Length[vec]}],
    "input" -> num /@ vec,
    "output" -> num /@ out
  |>
];

(* Linear (full) convolution, zero padding. *)
convLhs = {1, 2, 3, 4};
convRhs = {5, -1, 2};
convOut = ListConvolve[convLhs, convRhs, {1, -1}, 0];

(* Rectangular-window STFT: 16-sample ramp, size 8, hop 4. *)
stftSignal = Range[0, 15];
stftFrames = Table[Take[stftSignal, {s + 1, s + 8}], {s, 0, 8, 4}];
stftSpectra = dftApply[dftMatrix[8], N[#, prec + 10] & /@ #] & /@ stftFrames;

payload = <|
  "schema" -> "sonium-fft.wolfram-golden.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " exact-transform golden"],
  "conventions" -> <|
    "dft_forward" -> "M[j,k] = Exp[-2 Pi I j k / N], unnormalized",
    "dft_inverse" -> "Conjugate[M]/N",
    "dct2" -> "out[k] = Sum x[n] Cos[Pi (n+1/2) k / N]",
    "dst2" -> "out[k] = Sum x[n] Sin[Pi (n+1/2) (k+1) / N]"
  |>,
  "dft" -> {
    <|"name" -> "dft-5", "case" -> dftCase[5, {3, -1, 4, 1, -5}]|>,
    <|"name" -> "dft-7-complex", "case" -> dftCase[7, {1 + I, 2 - I, -3, 4 + 2*I, -1 - I, 2, -4 + 3*I}]|>,
    <|"name" -> "dft-8", "case" -> dftCase[8, {1, -2, 3, -4, 5, -6, 7, -8}]|>
  },
  "real" -> {
    <|"name" -> "dct2-8", "case" -> realCase["dct2-8", dctMatrix[8], {1, -1, 2, -2, 3, -3, 4, -4}]|>,
    <|"name" -> "dst2-8", "case" -> realCase["dst2-8", dstMatrix[8], {1, -1, 2, -2, 3, -3, 4, -4}]|>
  },
  "convolution" -> <|
    "lhs" -> num /@ convLhs,
    "rhs" -> num /@ convRhs,
    "output" -> num /@ convOut
  |>,
  "stft_rectangular_8_4" -> <|
    "signal" -> num /@ stftSignal,
    "fft_size" -> 8,
    "hop_size" -> 4,
    "frames" -> (cnum /@ # & /@ stftSpectra)
  |>
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
