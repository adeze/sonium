(* Wolfram crossover golden for sonium-sim-speaker.
 *
 * Emits exact first-order crossover transfers and their frequency
 * derivatives as one compact RawJSON document on stdout; redirect into
 *   crates/sonium-sim-speaker/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-sim-speaker/wolfram/crossover_golden.wls \
 *     > crates/sonium-sim-speaker/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-sim-speaker-20260921 (analytic
 * crossover derivatives; absolute error floors near nulls). The branch
 * model is H(f) = gain * shape(f/fc) * Exp[-I*2*Pi*f*delay] with
 * shapeLP(r) = 1/(1+I*r) and shapeHP(r) = I*r/(1+I*r). Derivatives come
 * from symbolic D[], not from finite differences of a reference
 * implementation, so the replay test pins both the transfer and the
 * sensitivity the product path differentiates.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];
cnum[z_] := <|"re" -> num[Re[z]], "im" -> num[Im[z]]|>;

shapeLP[r_] := 1/(1 + I*r);
shapeHP[r_] := I*r/(1 + I*r);
branchH[f_, fc_, gain_, delay_, shape_] :=
  gain * shape[f/fc] * Exp[-I*2*Pi*f*delay];

cases = {
  <|"kind" -> "LowPass", "fc" -> 1000., "gain" -> 1., "delay" -> 0.|>,
  <|"kind" -> "LowPass", "fc" -> 1000., "gain" -> 0.5, "delay" -> 25.*^-6|>,
  <|"kind" -> "HighPass", "fc" -> 500., "gain" -> 2., "delay" -> 0.|>
};
freqs = {5., 100., 500., 1000., 2000., 8000.};

entry[kind_, fc_, gain_, delay_, f_] := Module[{shape, h, dh},
  shape = If[kind === "LowPass", shapeLP, shapeHP];
  h = branchH[f, fc, gain, delay, shape];
  dh = D[branchH[ff, fc, gain, delay, shape], ff] /. ff -> f;
  <|
    "kind" -> kind, "fc" -> num[fc], "gain" -> num[gain],
    "delay_s" -> num[delay], "f_hz" -> num[f],
    "h" -> cnum[h], "dhdf" -> cnum[dh],
    "exact" -> ToString[CForm[shape[f/fc]]]
  |>];

payload = <|
  "meta" -> <|
    "engine" -> StringJoin["Wolfram Engine ", $Version],
    "protocol" -> "audit-oracle-hermetic-v1"
  |>,
  "cases" -> Flatten[Table[
    entry[c["kind"], c["fc"], c["gain"], c["delay"], f],
    {c, cases}, {f, freqs}]]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
