(* Wolfram goldens for meshlang-ast SI unit conversions.
   Run: wolframscript -file crates/sonium-meshlang/crates/meshlang-ast/fixtures/units_golden.wls
   Output: crates/sonium-meshlang/crates/meshlang-ast/fixtures/wolfram-golden-v1.json
   Independent SI factors via Quantity/UnitConvert: every to_si_factor in
   the parser table is pinned, including the inch (exact 0.0254), the
   degree (Pi/180), and two compound expressions built with checked
   multiply/divide. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-meshlang", "crates", "meshlang-ast",
    "fixtures", "wolfram-golden-v1.json"}];

(* Plain parseable decimals: CForm never emits MEPREC marks, so exact
   rationals (1/10^6 -> 1.e-6) and transcendentals alike survive. The
   backtick-stripping convention used elsewhere silently drops exponents on
   exact-rational inputs and must not be copied for Quantity-derived
   goldens. *)
dec[x_] := ToString[CForm[N[x, 17]]];
siFactor[q_] := QuantityMagnitude[UnitConvert[q, "SIBase"]];

symbols = {"Millimeters", "Centimeters", "Kilometers", "Inches",
   "Micrometers", "Milliseconds", "Microseconds", "Grams", "Degrees",
   "Kilohertz", "Megahertz", "Kilopascals", "Megapascals", "Percent"};
names = {"mm", "cm", "km", "in", "um", "ms", "us", "g", "deg",
   "kHz", "MHz", "kPa", "MPa", "pct"};
entries = Table[
  names[[i]] -> <|"value" -> dec[siFactor[Quantity[1, symbols[[i]]]]],
    "exact" -> "UnitConvert[1 " <> symbols[[i]] <> ", SIBase]"|>,
  {i, Length[symbols]}];

(* compound expressions: km/ms -> m/s factor, MPa/mm -> Pa/m factor *)
compoundKmMs = QuantityMagnitude[UnitConvert[
    Quantity[1, "Kilometers"]/Quantity[1, "Milliseconds"], "Meters"/"Seconds"]];
compoundMPaMm = QuantityMagnitude[UnitConvert[
    Quantity[1, "Megapascals"]/Quantity[1, "Millimeters"], "Pascals"/"Meters"]];

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "values" -> Association[Join[entries, {
    "km_per_ms" -> <|"value" -> dec[compoundKmMs],
      "exact" -> "UnitConvert[km/ms, m/s]"|>,
    "MPa_per_mm" -> <|"value" -> dec[compoundMPaMm],
      "exact" -> "UnitConvert[MPa/mm, Pa/m]"|>}]]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];