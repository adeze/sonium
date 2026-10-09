(* Wolfram goldens for sonium-sim-room SPL conventions.
   Run: wolframscript -file crates/sonium-sim-room/fixtures/spl_golden.wls
   Output: crates/sonium-sim-room/fixtures/wolfram-golden-v1.json
   Pins the peak-vs-RMS convention (frequency-domain solvers carry peak
   amplitudes; SPL references RMS, hence the 1/Sqrt[2] factor) and the
   20 uPa reference against an independent implementation. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-sim-room", "fixtures", "wolfram-golden-v1.json"}];

dec[x_] := First[StringSplit[ToString[N[x, 17], InputForm], "`"]];

(* |0.03 + 0.04 I| = 0.05 Pa peak *)
mag = Abs[0.03 + 0.04*I];
rms = mag/Sqrt[2];
spl = 20*Log10[rms/20.*^-6];

(* round trip: back to peak amplitude at phase 0.7 rad *)
peak = Sqrt[2]*20.*^-6*10^(spl/20);

(* rigid 5x4x3 shoebox cavity: Neumann spectrum + zero-mode pressure *)
cavDims = {5., 4., 3.}; cS = 343.0;
cavFreq[{n_, m_, p_}] := (cS/2)*
  Sqrt[(n/cavDims[[1]])^2 + (m/cavDims[[2]])^2 + (p/cavDims[[3]])^2];
cavModes = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {2, 3, 1}};
zeroP = -cS^2/((Times @@ cavDims)*(2*Pi*55.0)^2);

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "values" -> Association[{
    "peak_spl_db" -> <|
      "value" -> dec[spl],
      "exact" -> "20*Log10[(0.05/Sqrt[2])/20*10^-6]"|>,
    "roundtrip_peak_pa" -> <|
      "value" -> dec[peak],
      "exact" -> "Sqrt[2]*20*10^-6*10^(spl/20) = 0.05"|>,
    "cav_f_100" -> <|"value" -> dec[cavFreq[{1, 0, 0}]],
      "exact" -> "c/2 Sqrt[(1/5)^2], rigid 5x4x3"|>,
    "cav_f_010" -> <|"value" -> dec[cavFreq[{0, 1, 0}]],
      "exact" -> "c/2 Sqrt[(1/4)^2], rigid 5x4x3"|>,
    "cav_f_001" -> <|"value" -> dec[cavFreq[{0, 0, 1}]],
      "exact" -> "c/2 Sqrt[(1/3)^2], rigid 5x4x3"|>,
    "cav_f_111" -> <|"value" -> dec[cavFreq[{1, 1, 1}]],
      "exact" -> "c/2 Sqrt[(1/5)^2 + (1/4)^2 + (1/3)^2]"|>,
    "cav_f_231" -> <|"value" -> dec[cavFreq[{2, 3, 1}]],
      "exact" -> "c/2 Sqrt[(2/5)^2 + (3/4)^2 + (1/3)^2]"|>,
    "cav_zero_re" -> <|"value" -> dec[zeroP],
      "exact" -> "-c^2/(V (2 pi 55)^2), V = 60 m^3"|>,
    "cav_zero_im" -> <|"value" -> dec[0.],
      "exact" -> "zero mode is purely real"|>}]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];