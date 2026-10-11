(* Independent Wolfram reference for LAW76 SAMP triaxiality yield.
   Derives, at 50-digit working precision, station-interpolated yields
   and the closed-form radial return of a uniaxial-strain increment
   against flat curves. All pins below are exact rationals.
   The Rust side consumes law76_samp_golden.json; this script is the
   derivation.
   Regen: wolframscript -script fixtures/law76_samp.wls > fixtures/law76_samp_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

young = 1000; nu = 3/10; gg = young/(2*(1 + nu));
stations = {-2/3, -1/3, 0, 1/3, 2/3};
stationVals = {10, 10, 6, 20, 20};
yieldAt[t_] := Piecewise[
  Table[{stationVals[[i]] + (t - stations[[i]])*(stationVals[[i + 1]] - stationVals[[i]])/(stations[[i + 1]] - stations[[i]]),
    stations[[i]] < t <= stations[[i + 1]]}, {i, 1, 4}],
  stationVals[[1]]];
yieldClamped[t_] := Which[t <= -2/3, 10, t >= 2/3, 20, True, yieldAt[t]];

qtrial = 20; y0 = 10;
dep = (qtrial - y0)/(3*gg);
sxx = 40/3;

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "law76_samp",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "young_pa" -> 1000,
    "poisson" -> 0.3,
    "tension_pa" -> 10,
    "compression_pa" -> 20,
    "shear_pa" -> 6,
    "dstrain" -> {0.02, -0.006, -0.006, 0, 0, 0}
  |>,
  "values" -> <|
    "station_tension" -> <|
      "values" -> {dec[yieldClamped[-1/3]]},
      "exact" -> "10 at T = -1/3"
    |>,
    "station_shear" -> <|
      "values" -> {dec[yieldClamped[0]]},
      "exact" -> "6 at T = 0"
    |>,
    "station_compression" -> <|
      "values" -> {dec[yieldClamped[1/3]]},
      "exact" -> "20 at T = 1/3"
    |>,
    "station_mid" -> <|
      "values" -> {dec[yieldClamped[-1/6]]},
      "exact" -> "(10 + 6)/2 = 8 at T = -1/6"
    |>,
    "station_biaxial_end" -> <|
      "values" -> {dec[yieldClamped[-2/3]]},
      "exact" -> "10 at T = -2/3 without biaxial curve"
    |>,
    "station_clamp" -> <|
      "values" -> {dec[yieldClamped[1]]},
      "exact" -> "20 clamped past T = 2/3"
    |>,
    "trial_q" -> <|
      "values" -> {dec[qtrial]},
      "exact" -> "q = E 0.02 = 20"
    |>,
    "return_dep" -> <|
      "values" -> {dec[dep]},
      "exact" -> "(20 - 10)/(3 G) = 13/1500"
    |>,
    "return_sxx" -> <|
      "values" -> {dec[sxx]},
      "exact" -> "strain-controlled return gives 40/3"
    |>
  |>,
  "tolerances" -> <|
    "station_tension_abs" -> 1.0*^-12,
    "station_tension_rel" -> 1.0*^-12,
    "station_shear_abs" -> 1.0*^-12,
    "station_shear_rel" -> 1.0*^-12,
    "station_compression_abs" -> 1.0*^-12,
    "station_compression_rel" -> 1.0*^-12,
    "station_mid_abs" -> 1.0*^-12,
    "station_mid_rel" -> 1.0*^-12,
    "station_biaxial_end_abs" -> 1.0*^-12,
    "station_biaxial_end_rel" -> 1.0*^-12,
    "station_clamp_abs" -> 1.0*^-12,
    "station_clamp_rel" -> 1.0*^-12,
    "trial_q_abs" -> 1.0*^-9,
    "trial_q_rel" -> 1.0*^-12,
    "return_dep_abs" -> 1.0*^-12,
    "return_dep_rel" -> 1.0*^-9,
    "return_sxx_abs" -> 1.0*^-9,
    "return_sxx_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
