(* Wolfram special-function golden for sonium-wave.
 *
 * Emits arbitrary-precision spherical Bessel/Hankel and Legendre reference
 * values as one compact RawJSON document on stdout; redirect into
 *   crates/sonium-wave/fixtures/special-golden-wolfram-v1.json
 * Run from the repository root:
 *   wolframscript -file crates/sonium-wave/wolfram/special_golden.wls \
 *     > crates/sonium-wave/fixtures/special-golden-wolfram-v1.json
 *
 * Independent oracle per audit-sonium-wave-20260921: certifies recurrence
 * direction, Wronskians, derivatives and low/high argument limits without
 * generating expected values with Sonium (or with the Python decimal
 * generator behind special-golden-v1.json, which remains the second
 * independent implementation).
 *
 * Conventions: j_n / y_n are the real positive-argument spherical functions
 * and derivatives are d/dx. SphericalHankelH1[n, x] = j_n + i y_n, which is
 * exactly how sonium stores harmonic_factor = 1.0 (re = j, im = y).
 * LegendreP[n, m, x] carries the Condon-Shortley (-1)^m phase.
 *)
Off[General::munfl];

(* Exact inputs (rational multiples of Pi) need headroom beyond the target
   precision during internal evaluation; otherwise N::meprec messages would
   pollute the stdout JSON. *)
$MaxExtraPrecision = 500;

prec = 50;
(* CForm renders full arbitrary-precision decimals without the backtick
   precision marks that InputForm emits, so the JSON stays plain numbers
   as strings that Rust f64 parsing accepts. *)
num[value_] := ToString[CForm[N[value, prec]]];

sphericalXs = {1/10, Pi/4, Pi/2, Pi, 2*Pi, 50};
sphericalNames = {"tenth", "pi_over_4", "pi_over_2", "pi", "two_pi", "fifty"};
sphericalOrders = Range[0, 13];

sphJ[n_, x_] := SphericalBesselJ[n, x];
sphY[n_, x_] := SphericalBesselY[n, x];
sphJd[n_, x_] := D[SphericalBesselJ[n, \[FormalX]], \[FormalX]] /. \[FormalX] -> x;
sphYd[n_, x_] := D[SphericalBesselY[n, \[FormalX]], \[FormalX]] /. \[FormalX] -> x;
sphH1Re[n_, x_] := Re[SphericalHankelH1[n, x]];
sphH1Im[n_, x_] := Im[SphericalHankelH1[n, x]];

sphericalCase[name_, x_] := Module[{xc = N[x, prec + 10]},
  <|
    "name" -> name,
    "x" -> num[x],
    "j" -> Table[num[sphJ[n, xc]], {n, sphericalOrders}],
    "y" -> Table[num[sphY[n, xc]], {n, sphericalOrders}],
    "j_derivative" -> Table[num[sphJd[n, xc]], {n, sphericalOrders}],
    "y_derivative" -> Table[num[sphYd[n, xc]], {n, sphericalOrders}],
    "h1_re" -> Table[num[sphH1Re[n, xc]], {n, sphericalOrders}],
    "h1_im" -> Table[num[sphH1Im[n, xc]], {n, sphericalOrders}]
  |>
];

legendreXs = {37/100, -81/100};
legendreNames = {"x_0.37", "x_minus_0.81"};
legendreDegrees = Range[0, 7];
associatedM = 2;
associatedDegrees = Range[associatedM, 7];

legendreCase[name_, x_] := Module[{xc = N[x, prec + 10]},
  <|
    "name" -> name,
    "x" -> num[x],
    "p" -> Table[num[LegendreP[n, xc]], {n, legendreDegrees}],
    "derivative" -> Table[
      num[D[LegendreP[n, \[FormalX]], \[FormalX]] /. \[FormalX] -> xc],
      {n, legendreDegrees}
    ],
    "associated_m" -> associatedM,
    "associated" -> Table[num[LegendreP[n, associatedM, xc]], {n, associatedDegrees}]
  |>
];

payload = <|
  "schema" -> "sonium-wave.special-golden-wolfram.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " arbitrary-precision special functions"],
  "conventions" -> <|
    "spherical" -> "j_n and y_n use the real positive-argument convention; derivatives are d/dx",
    "hankel" -> "h1 = SphericalHankelH1 = j_n + i y_n; matches sonium harmonic_factor = 1.0 packing",
    "legendre" -> "P_n and P_n^m use the Condon-Shortley (-1)^m phase"
  |>,
  "spherical" -> MapThread[sphericalCase, {sphericalNames, sphericalXs}],
  "legendre" -> MapThread[legendreCase, {legendreNames, legendreXs}]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
