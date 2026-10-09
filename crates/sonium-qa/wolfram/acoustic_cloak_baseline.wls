(* Independent rigid-cylinder series for the uncloaked Acoustic Cloak case. *)

cWater = QuantityMagnitude[ThermodynamicData["Water", "SoundSpeed"]];
frequency = 800.;
radius = 1.;
k = 2 Pi frequency/cWater;
termCount = 30;

jDerivative[n_, argument_] := If[
   n == 0,
   -BesselJ[1, argument],
   BesselJ[n - 1, argument] - n/argument BesselJ[n, argument]
   ];
hDerivative[n_, argument_] := If[
   n == 0,
   -HankelH1[1, argument],
   HankelH1[n - 1, argument] - n/argument HankelH1[n, argument]
   ];
coefficient[n_] :=
  -I^n jDerivative[n, k radius]/hDerivative[n, k radius];
totalPressure[r_, theta_] :=
  Exp[I k r Cos[theta]] + Sum[
    If[n == 0, 1, 2] coefficient[n] HankelH1[n, k r] Cos[n theta],
    {n, 0, termCount - 1}];

rings = {1., 1.025, 1.05, 1.1, 1.2};
angles = Subdivide[0., Pi, 360];
maximum = Max@Flatten@Table[
    Abs[totalPressure[r, theta]], {r, rings}, {theta, angles}];

Print@ExportString[
  <|"schemaVersion" -> 1,
    "waterSoundSpeedMetersPerSecond" -> cWater,
    "waveNumberPerMeter" -> k,
    "seriesMaximumPascal" -> maximum,
    "publishedFemMaximumPascal" -> 1.89447|>,
  "RawJSON"];
