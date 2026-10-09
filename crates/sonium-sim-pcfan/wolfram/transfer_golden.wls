(* Wolfram transfer-conservation golden for sonium-sim-pcfan.
 *
 * Emits exact reference-tetrahedron volumes as one compact RawJSON document
 * on stdout; redirect into
 *   crates/sonium-sim-pcfan/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-sim-pcfan/wolfram/transfer_golden.wls \
 *     > crates/sonium-sim-pcfan/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-sim-pcfan-20260921 (transfer-adjoint
 * conservation on small production meshes). The replay transfers a constant
 * state between two nested reference tetrahedra through the production
 * `transfer_pc_fan_cfd_state` path. The reference tet is
 *   (0,0,0), (1,0,0), (0,1,0), (0,0,1) with volume 1/6, and the nested tet
 *   scales it by 1/2 about the origin with volume 1/48. A constant state
 *   transferred between equal-measure meshes is exact with a vanishing
 *   conservative correction; the nested unequal-measure transfer must fail
 *   closed with the two Wolfram volumes as mismatch measures.
 * Wolfram independently derives both volumes from the coordinates;
 * `RegionMeasure[Tetrahedron[...]]` never touches the Rust mesh code.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];

source = Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}];
target = Tetrahedron[{{0, 0, 0}, {1/2, 0, 0}, {0, 1/2, 0}, {0, 0, 1/2}}];

payload = <|
  "meta" -> <|
    "engine" -> StringJoin["Wolfram Engine ", $Version],
    "protocol" -> "audit-oracle-hermetic-v1"
  |>,
  "source_volume" -> <|
    "value" -> num[RegionMeasure[source]],
    "exact" -> "1/6"
  |>,
  "target_volume" -> <|
    "value" -> num[RegionMeasure[target]],
    "exact" -> "1/48"
  |>
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
