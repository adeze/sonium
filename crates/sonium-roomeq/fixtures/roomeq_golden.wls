(* Wolfram goldens for sonium-roomeq CEA-2034 metrics + preference score.
   Run: wolframscript -file crates/sonium-roomeq/fixtures/roomeq_golden.wls
   Output: crates/sonium-roomeq/fixtures/wolfram-golden-v1.json
   Mirrors the Rust definitions in cea2034.rs with independent Wolfram
   formulations (direct log-domain energy sums, Correlation, TakeWhile):
   any transcription or convention slip (peak/RMS, weight constants,
   Olive coefficients, band-table reference) shows up as a replay mismatch. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-roomeq", "fixtures", "wolfram-golden-v1.json"}];

(* Plain decimals: strip the MEPREC `prec. precision mark. *)
dec[x_] := First[StringSplit[ToString[N[x, 17], InputForm], "`"]];
decList[l_] := Map[dec, l];

scalar[name_, val_, exact_] := name -> <|
  "value" -> dec[val], "exact" -> exact|>;
vector[name_, vals_, exact_] := name -> <|
  "values" -> decList[vals], "exact" -> exact|>;

(* ---------------- score-case curves (exact one-decimal SPL) ---------------- *)
freq = {50., 100., 200., 300., 1000., 5000., 11000., 15000.};
on   = {78.5, 79.0, 80.5, 81.0, 82.5, 83.0, 82.0, 81.5};
lw   = {79.0, 79.5, 80.0, 81.5, 82.0, 82.5, 82.0, 81.0};
sp   = {70.0, 71.0, 73.0, 76.0, 78.0, 79.0, 78.5, 78.0};
pir  = {77.0, 78.0, 79.5, 80.5, 81.5, 82.0, 81.5, 81.0};
intervals = {{1, 4}, {5, 8}}; (* 1-based halves of the 8-point grid *)

mad[v_] := Mean[Abs[v - Mean[v]]];
nbdOn = Mean[mad /@ {on[[1 ;; 4]], on[[5 ;; 8]]}];
nbdPir = Mean[mad /@ {pir[[1 ;; 4]], pir[[5 ;; 8]]}];

(* SM window: f > 100 Hz, f < 16000 Hz -> indices 3..8 *)
smX = Log10[freq[[3 ;; 8]]];
smY = pir[[3 ;; 8]];
smPir = Correlation[smX, smY]^2;

(* LFX: lw window f in [300, 10000] -> indices 4..6; ref = mean - 6 dB *)
lwRef = Mean[lw[[4 ;; 6]]] - 6.0;
lfxIdx = Select[Range[8], freq[[#]] <= 300.0 && sp[[#]] <= lwRef &];
(* contiguous run from index 1 -> answer is log10 of the next frequency *)
runEnd = LengthWhile[Range[Length[lfxIdx]], lfxIdx[[#]] == # &];
lfxVal = Log10[freq[[lfxIdx[[runEnd]] + 1]]];

pref = 12.69 - 2.49*nbdOn - 2.99*nbdPir - 4.31*lfxVal + 2.32*smPir;

(* ---------------- spinorama energy-aggregation case ---------------- *)
rows = {
  {80., 81., 82.}, {78., 79., 80.}, {82., 82., 82.}, {81., 80., 79.},
  {79., 79., 79.}, {83., 84., 85.}, {77., 78., 79.}, {75., 76., 77.}};
groups = {{1, 2}, {3}, {4}, {5}, {6}, {7}, {8}, {8}};
weights = {1., 1., 1., 1., 1., 1., 1., 2.};

em[s_, w_] := 10*Log10[Total[w*10^(s/10)]/Total[w]];
colMean[col_, idx_] := em[rows[[idx, col]], ConstantArray[1., Length[idx]]];
g = Table[colMean[c, groups[[k]]], {k, 7}, {c, 3}];
lwRow = Table[em[g[[3 ;; 7, c]], ConstantArray[1., 5]], {c, 3}];
spRow = Table[em[rows[[groups[[8]], c]],
  weights[[groups[[8]]]]], {c, 3}];
pirRow = Table[em[{g[[1, c]], lwRow[[c]], spRow[[c]]},
  {0.12, 0.44, 0.44}], {c, 3}];
spin = Join[g[[1 ;; 1]], {lwRow}, g[[3 ;; 7]], {spRow}, {pirRow}];

(* ---------------- octave band table (N = 3) ---------------- *)
count = 3;
p = 2^(1/count); pBand = 2^(1/(2*count));
oIter = Quotient[count*10 + 1, 2];
downs = Table[1290.0/p^i, {i, oIter, 1, -1}];
ups = TakeWhile[Table[1290.0*p^i, {i, oIter}], #/pBand <= 20000.0 &];
centers = Join[downs, {1290.0}, ups];
bands = Map[{#/pBand, #, #*pBand} &, centers];

(* intervals of the 6-point probe grid against the N=3 table *)
probe = {50., 150., 1000., 5000., 11000., 15000.};
bandMin = Max[100.0, First[probe]];
posGE[v_] := Module[{p = FirstPosition[probe, _?(# >= v &), {7}, 1]},
  p[[1]] - 1];
posGT[v_] := Module[{p = FirstPosition[probe, _?(# > v &), {7}, 1]},
  p[[1]] - 1];
kept = Select[bands, bandMin <= #[[2]] <= 12000.0 &];
pairs = Select[{posGE[#[[1]]], posGT[#[[3]]]} & /@ kept,
  #[[1]] < #[[2]] &];

di = lw - sp;

(* ---------------- +12 dB common-offset invariance ---------------- *)
shift = 12.0;
on12 = on + shift; lw12 = lw + shift; sp12 = sp + shift; pir12 = pir + shift;
nbdOn12 = Mean[mad /@ {on12[[1 ;; 4]], on12[[5 ;; 8]]}];
nbdPir12 = Mean[mad /@ {pir12[[1 ;; 4]], pir12[[5 ;; 8]]}];
smPir12 = Correlation[Log10[freq[[3 ;; 8]]], pir12[[3 ;; 8]]]^2;
lwRef12 = Mean[lw12[[4 ;; 6]]] - 6.0;
lfxIdx12 = Select[Range[8], freq[[#]] <= 300.0 && sp12[[#]] <= lwRef12 &];
runEnd12 = LengthWhile[Range[Length[lfxIdx12]], lfxIdx12[[#]] == # &];
lfx12 = Log10[freq[[lfxIdx12[[runEnd12]] + 1]]];
pref12 = 12.69 - 2.49*nbdOn12 - 2.99*nbdPir12 - 4.31*lfx12 + 2.32*smPir12;

(* ---------------- energy-mean gradient weights ---------------- *)
(* d(em)/d(s_i) = 10^(s_i/10)/Sum[10^(s/10)]: the chain-rule weights for
   the two-row group-0 column-0 aggregation. *)
demW = 10^(rows[[{1, 2}, 1]]/10);
demW = demW/Total[demW];

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "values" -> Association[{
    scalar["nbd_on", nbdOn, "Mean of interval MADs of `on`"],
    scalar["nbd_pir", nbdPir, "Mean of interval MADs of `pir`"],
    scalar["sm_pir", smPir, "Correlation[Log10[f], pir]^2 over f in (100, 16000)"],
    scalar["lfx", lfxVal, "Log10[1000] via contiguous low-frequency run"],
    scalar["pref_score", pref, "12.69 - 2.49*nbdOn - 2.99*nbdPir - 4.31*lfx + 2.32*sm"],
    vector["di_curve", di, "lw - sp elementwise"],
    vector["spin_lw", spin[[2]], "energy mean of ER rows 3..7"],
    vector["spin_sp", spin[[8]], "weighted energy mean, weight 2 on row 8"],
    vector["spin_pir", spin[[9]], "0.12/0.44/0.44 energy mean of on/LW/SP"],
    scalar["spin_g0_c0", spin[[1, 1]],
      "two-row log-domain energy mean of {80, 78}"],
    scalar["nbd_on_12", nbdOn12, "+12 dB offset NBD (translation invariant)"],
    scalar["nbd_pir_12", nbdPir12, "+12 dB offset NBD (translation invariant)"],
    scalar["sm_pir_12", smPir12, "+12 dB offset SM (translation invariant)"],
    scalar["lfx_12", lfx12, "+12 dB offset LFX (frequency, invariant)"],
    scalar["pref_12", pref12, "+12 dB common-offset preference score"],
    vector["dem_g0_c0", demW,
      "d(group-0 col-0 energy mean)/d(row SPL), sums to 1"],
    vector["octave3_centers", centers, "1290 Hz reference geometric table"],
    vector["octave3_intervals", Flatten[pairs],
      "flattened (imin, imax) pairs on the probe grid"]}]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];