(* Literal extractor for Wolfram's furnished-room eigenfrequency example.
   The geometry remains in Wolfram's proprietary PDEModels support assets and
   is deliberately not copied into the Sonium repository. *)

Needs["NDSolve`FEM`"];

supportDirectory = Environment["WOLFRAM_PDEMODELS_SUPPORT_DIR"];
If[!StringQ[supportDirectory] || StringLength[supportDirectory] == 0,
  supportDirectory = FileNameJoin[{
     $InstallationDirectory, "SystemFiles", "Components", "PDEModels",
     "SupportFiles"
  }]
];

geometryPath = FileNameJoin[{supportDirectory, "SmoothWallModel.stl"}];
If[!FileExistsQ[geometryPath],
  Print["SmoothWallModel.stl not found under " <> supportDirectory];
  Exit[2]
];

cAir = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoAir = QuantityMagnitude[ThermodynamicData["Air", "Density"]];
vars = {p[x, y, z], {x, y, z}};
pars = <|"SoundSpeed" -> cAir, "MassDensity" -> rhoAir|>;
bmesh = ToBoundaryMesh[Import[geometryPath]];
h = (cAir/200)/12;
mesh = ToElementMesh[bmesh, MaxCellMeasure -> {"Length" -> h}, "MeshOrder" -> 1];
{vals, funs} = NDEigensystem[
  AcousticPDEComponent[vars, pars], p, Element[{x, y, z}, mesh], 5
];
frequencies = N[Round[
  cAir/(2 Pi) Sqrt[rhoAir vals], 0.001
]];

Print[ExportString[
  <|
    "schema_version" -> 1,
    "source" -> "Wolfram RoomEigenfrequencies furnished room",
    "geometry_asset" -> "SmoothWallModel.stl",
    "sound_speed_m_s" -> N[cAir],
    "mass_density_kg_m3" -> N[rhoAir],
    "maximum_frequency_hz" -> 200,
    "target_edge_length_m" -> N[h],
    "eigenvalues" -> N[vals],
    "frequencies_hz" -> frequencies,
    "published_frequencies_hz" -> {0., 23.152, 33.982, 40.876, 44.8},
    "empty_room_frequencies_hz" -> {0., 22.8896, 34.3344, 41.2648, 45.7792}
  |>,
  "RawJSON"
]];
