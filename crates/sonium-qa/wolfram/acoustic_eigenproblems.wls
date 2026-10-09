(* Published Wolfram PDEModels examples:
   https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/AcousticsFrequencyDomain.html?view=all
   https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/ModelCollection/RoomEigenfrequencies.html?view=all

   The tube spectra are evaluated with NDEigenvalues.  The room values use the
   exact rectangular-room expression published by Wolfram so this script stays
   suitable for a fast validation test instead of launching a large 3D solve.
*)
Off[General::munfl];

c = 343.;
left = 0.;
right = 5.;
length = right - left;

closedEigenvalues = Sort[N[NDEigenvalues[
  -Laplacian[p[x], {x}],
  p[x],
  {x, left, right},
  6,
  Method -> {"SpatialDiscretization" -> {"FiniteElement", {
    "MeshOptions" -> {"MaxCellMeasure" -> 0.01}
  }}}
], 16]];

openEigenvalues = Sort[N[NDEigenvalues[
  {-Laplacian[p[x], {x}], DirichletCondition[p[x] == 0, True]},
  p[x],
  {x, left, right},
  5,
  Method -> {"SpatialDiscretization" -> {"FiniteElement", {
    "MeshOptions" -> {"MaxCellMeasure" -> 0.01}
  }}}
], 16]];

toFrequency[lambda_] := c Sqrt[Max[0., lambda]]/(2 Pi);
closedFrequencies = toFrequency /@ closedEigenvalues;
openFrequencies = toFrequency /@ openEigenvalues;
closedExact = Table[n c/(2 length), {n, 0, 5}];
openExact = Table[n c/(2 length), {n, 1, 5}];

roomDimensions = {7.5, 5., 3.};
roomModeIndices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 0, 0}};
roomExact[index_] := c/2 Sqrt[Total[(index/roomDimensions)^2]];
roomFrequencies = roomExact /@ roomModeIndices;

drivenLength = 2.;
drivenK = 1.;
drivenSampleX = {0., 0.5, 1., 1.5, 2.};
drivenPressure[x_, reflection_] :=
  (Exp[-I drivenK x] + reflection Exp[I drivenK (x - 2 drivenLength)])/
  (1 + reflection Exp[-2 I drivenK drivenLength]);
complexPair[value_] := {Re[N[value, 16]], Im[N[value, 16]]};

payload = <|
  "schema_version" -> 1,
  "sound_speed_m_s" -> c,
  "tube_closed_frequencies_hz" -> closedFrequencies,
  "tube_open_frequencies_hz" -> openFrequencies,
  "tube_closed_exact_hz" -> closedExact,
  "tube_open_exact_hz" -> openExact,
  "room_mode_indices" -> roomModeIndices,
  "room_exact_hz" -> roomFrequencies,
  "driven_tube_sample_x_m" -> drivenSampleX,
  "driven_tube_hard_pressure" ->
    (complexPair /@ (drivenPressure[#, 1.] & /@ drivenSampleX)),
  "driven_tube_soft_pressure" ->
    (complexPair /@ (drivenPressure[#, -1.] & /@ drivenSampleX)),
  "driven_tube_impedance_pressure" ->
    (complexPair /@ (drivenPressure[#, 1./3.] & /@ drivenSampleX))
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
