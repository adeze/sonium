(* Wolfram oscillator golden for sonium-solvers-time-integration.
 *
 * Emits high-precision cosine samples for undamped (w = 1) and stiff
 * (w = 50) free oscillators as one compact RawJSON document on stdout;
 * redirect into
 *   crates/sonium-solvers-time-integration/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-solvers-time-integration/wolfram/oscillator_golden.wls \
 *     > crates/sonium-solvers-time-integration/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-solvers-time-integration-20260921:
 * high-precision undamped/stiff oscillator references (DSolve-grade exact
 * trajectories sampled at 50 digits). Integrator trajectories are
 * second-order approximations; the replay test asserts the documented
 * error scaling, not exact equality.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];

oscillator[w_, t_] := <|"t" -> num[t], "u" -> num[Cos[w*t]], "v" -> num[-w*Sin[w*t]]|>;

payload = <|
  "schema" -> "sonium-solvers-time-integration.wolfram-golden.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " oscillator golden"],
  "system" -> "u'' + w^2 u = 0, u(0) = 1, v(0) = 0",
  "undamped_w1" -> Table[oscillator[1, t], {t, {1/2, 1, 2, 5, 10}}],
  "stiff_w50" -> Table[oscillator[50, t], {t, {1/10, 1/2, 1, 2}}]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
