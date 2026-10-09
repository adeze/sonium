(* Derived driven-room reference using the exact dimensions from Wolfram's
   Room Eigenfrequencies example. The published page is an eigensystem, not a
   BEM model; this modal Green-function sum is explicitly an adaptation. *)

dimensions = {7.5, 5., 3.};
soundSpeed = 343.2;
frequency = 80.;
k = 2. Pi frequency/soundSpeed;
source = {1.1, 1.3, 1.2};
receiver = {5.8, 3.7, 1.6};
maxOrder = 30;

epsilon[n_] := If[n == 0, 1., 2.];
modeWaveNumberSquared[nx_, ny_, nz_] :=
  (nx Pi/dimensions[[1]])^2 +
  (ny Pi/dimensions[[2]])^2 +
  (nz Pi/dimensions[[3]])^2;
modeProduct[nx_, ny_, nz_, point_] :=
  Cos[nx Pi point[[1]]/dimensions[[1]]] *
  Cos[ny Pi point[[2]]/dimensions[[2]]] *
  Cos[nz Pi point[[3]]/dimensions[[3]]];

modalPressure[order_] := Sum[
  epsilon[nx] epsilon[ny] epsilon[nz]/Times @@ dimensions *
  modeProduct[nx, ny, nz, source] *
  modeProduct[nx, ny, nz, receiver] /
  (modeWaveNumberSquared[nx, ny, nz] - k^2),
  {nx, 0, order}, {ny, 0, order}, {nz, 0, order}
];
complexPair[value_] := {Re[N[value, 16]], Im[N[value, 16]]};

payload = <|
  "schema_version" -> 1,
  "provenance" -> "derived modal Green function using Wolfram Room Eigenfrequencies dimensions; not a literal BEM example",
  "dimensions_m" -> dimensions,
  "sound_speed_m_s" -> soundSpeed,
  "frequency_hz" -> frequency,
  "source_m" -> source,
  "receiver_m" -> receiver,
  "max_mode_order" -> maxOrder,
  "receiver_pressure" -> complexPair[modalPressure[maxOrder]],
  "receiver_pressure_order_20" -> complexPair[modalPressure[20]]
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
