(* Independent Wolfram reference for erosion energy accounting.
   Derives, at 50-digit working precision, the active/eroded ledger split
   for a scripted three-element history plus shell thickness-ratio policy
   verdicts. All inputs are exact rationals. The Rust side consumes
   solids_erosion_energy_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_erosion_energy.wls > fixtures/solids_erosion_energy_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Element volumes (m^3), plastic densities, internal densities (J/m^3). *)
vol = {2/10^6, 1/10^6, 5/10^7};
plasD = {15*10^7, 3*10^8, 2*10^8};
intD = {16*10^7, 32*10^7, 21*10^7};
eroded = {False, True, False};

plas = vol*plasD;
int = vol*intD;
activePlas = Total[Pick[plas, eroded, False]];
erodedPlas = Total[Pick[plas, eroded, True]];
activeInt = Total[Pick[int, eroded, False]];
erodedInt = Total[Pick[int, eroded, True]];

(* Shell thickness-ratio verdicts: five 0.2 layers, ratio 0.4. *)
thick = {1/5, 1/5, 1/5, 1/5, 1/5};
runFrom[failed_, order_] := Module[{sum = 0},
  Do[If[! failed[[i]], Break[]]; sum += thick[[i]], {i, order}];
  sum/Total[thick]];
tripped[failed_] :=
  If[runFrom[failed, Range[5]] > 2/5 || runFrom[failed, Reverse[Range[5]]] > 2/5, 1, 0];
verdictTwo = tripped[{True, True, False, False, False}];
verdictThree = tripped[{True, True, True, False, False}];
verdictSplit = tripped[{True, True, False, True, True}];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "erosion_energy",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "volumes_m3" -> {0.000002, 0.000001, 5.0*^-7},
    "plastic_densities" -> {150000000.0, 300000000.0, 200000000.0},
    "internal_densities" -> {160000000.0, 320000000.0, 210000000.0},
    "eroded" -> {False, True, False},
    "layer_thickness" -> {0.2, 0.2, 0.2, 0.2, 0.2},
    "thick_fail" -> 0.4
  |>,
  "values" -> <|
    "active_plastic" -> <|
      "values" -> {dec[activePlas]},
      "exact" -> "2e-6*1.5e8 + 5e-7*2e8"
    |>,
    "eroded_plastic" -> <|
      "values" -> {dec[erodedPlas]},
      "exact" -> "1e-6*3e8"
    |>,
    "active_internal" -> <|
      "values" -> {dec[activeInt]},
      "exact" -> "2e-6*1.6e8 + 5e-7*2.1e8"
    |>,
    "eroded_internal" -> <|
      "values" -> {dec[erodedInt]},
      "exact" -> "1e-6*3.2e8"
    |>,
    "verdict_two_layers" -> <|
      "values" -> {dec[verdictTwo]},
      "exact" -> "0.4 > 0.4 is false"
    |>,
    "verdict_three_layers" -> <|
      "values" -> {dec[verdictThree]},
      "exact" -> "0.6 > 0.4"
    |>,
    "verdict_split_layers" -> <|
      "values" -> {dec[verdictSplit]},
      "exact" -> "run breaks at the sound layer"
    |>
  |>,
  "tolerances" -> <|
    "active_plastic_abs" -> 1.0*^-9,
    "active_plastic_rel" -> 1.0*^-12,
    "eroded_plastic_abs" -> 1.0*^-9,
    "eroded_plastic_rel" -> 1.0*^-12,
    "active_internal_abs" -> 1.0*^-9,
    "active_internal_rel" -> 1.0*^-12,
    "eroded_internal_abs" -> 1.0*^-9,
    "eroded_internal_rel" -> 1.0*^-12,
    "verdict_two_layers_abs" -> 1.0*^-15,
    "verdict_two_layers_rel" -> 1.0*^-12,
    "verdict_three_layers_abs" -> 1.0*^-15,
    "verdict_three_layers_rel" -> 1.0*^-12,
    "verdict_split_layers_abs" -> 1.0*^-15,
    "verdict_split_layers_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
