(* Wolfram correctly-rounded sin/cos golden for sonium-fastmath.
 *
 * Emits IEEE-754-exact input decimals with correctly rounded sine/cosine
 * values as one compact RawJSON document on stdout; redirect into
 *   crates/sonium-fastmath/fixtures/wolfram-golden-v1.json
 * Run from the repository root:
 *   wolframscript -file crates/sonium-fastmath/wolfram/sincos_golden.wls \
 *     > crates/sonium-fastmath/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-fastmath-20260921: exact-input
 * high-precision Sin/Cos references, especially k*Pi/2 neighbors (adjacent
 * binary64 values computed by exact-rational spacing, not by calling a
 * library routine) and huge exponents. The checked-in MPFR corpus remains
 * the second independent implementation.
 *
 * Values are 30-significant-digit decimals: enough to round-trip any
 * binary64 input or correctly rounded output exactly. Sine/cosine are the
 * 100-digit evaluations rounded once to binary64.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
outPrec = 30;
workPrec = 100;

(* Plain f64-parseable decimals, no backtick precision marks. *)
num[value_] := ToString[CForm[N[value, outPrec]]];
(* Correctly rounded binary64 of an exact/high-precision value. *)
rounded[value_] := N[value, MachinePrecision];

entry[name_, exactInput_] := Module[{s, c},
  s = rounded[N[Sin[exactInput], workPrec]];
  c = rounded[N[Cos[exactInput], workPrec]];
  <|"name" -> name, "x" -> num[N[exactInput, outPrec]],
    "sin" -> num[s], "cos" -> num[c]|>
];

(* Adjacent binary64 values around the nearest double to target, via exact
   rational spacing 2^(e-52); all targets below are normal-range. *)
neighbors[name_, target_] := Module[{d, exact, sp},
  d = N[target, MachinePrecision];
  exact = SetPrecision[d, Infinity];
  (* Binary exponent e with 1 <= m < 2 in d = m*2^e; adjacent normal
     doubles are spaced 2^(e-52) apart. *)
  sp = 2^(Exponent[d, 2] - 52);
  {entry[name <> "_down", exact - sp], entry[name <> "_up", exact + sp]}
];

cases = Join[
  {entry["pos_zero", 0], entry["neg_zero", 0],
   entry["subnormal_min", 2^-1074], entry["tiny_1e_300", 1*^-300],
   entry["half", 1/2], entry["one", 1]},
  Flatten[neighbors["kpi2_" <> ToString[#], #*Pi/2] & /@ Range[1, 8]],
  Flatten[neighbors["kpi_" <> ToString[#], #*Pi] & /@ Range[1, 2]],
  {entry["pi", Pi], entry["two_pi", 2*Pi], entry["ten", 10],
   entry["mega", 10^6], entry["peta", 10^15],
   (* Non-representable magnitudes must use the exact value of the nearest
      binary64 (SetPrecision[..., Infinity]), never the exact integer:
      at 1e100 scale the integer and its nearest double have unrelated
      sines. glibc cross-check confirmed the double-exact convention. *)
   entry["huge_1e100", SetPrecision[10.^100, Infinity]],
   entry["huge_1e300", SetPrecision[10.^300, Infinity]],
   entry["dbl_max", 2^1024 - 2^971]}
];

(* IEEE sin(-0) is -0, but Sin[0] evaluates to positive zero: restore the
   signed zero the neg_zero input must produce. *)
cases = cases /. c_Association /; c["name"] == "neg_zero" :>
  Join[c, <|"x" -> "-0.", "sin" -> "-0."|>];

payload = <|
  "schema" -> "sonium-fastmath.wolfram-golden.v1",
  "precision_digits" -> workPrec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " correctly-rounded sin/cos golden"],
  "conventions" -> <|
    "input" -> "30-digit decimals round-trip to the exact tested bit pattern",
    "output" -> "100-digit evaluation rounded once to binary64",
    "neighbors" -> "adjacent binary64 values via exact-rational 2^(e-52) spacing"
  |>,
  "cases" -> cases
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
