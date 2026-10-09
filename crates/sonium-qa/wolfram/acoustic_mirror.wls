(* Literal Wolfram Acoustic Mirror amplification reference. *)

Needs["NDSolve`FEM`"];
$HistoryLength = 0;

vars = {p[t, x, y], t, {x, y}};
cAir = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoAir = QuantityMagnitude[ThermodynamicData["Air", "Density"]];

focalLength = 1.;
thickness = 0.2;
mirrorWidth = 5.;
separation = 10.;
leftMirror = ImplicitRegion[
   -thickness <= x - y^2/(4 focalLength) <= 0 &&
    Abs[y] <= mirrorWidth/2, {x, y}];
rightMirror = ImplicitRegion[
   0 <= (x - separation) + y^2/(4 focalLength) <= thickness &&
    Abs[y] <= mirrorWidth/2, {x, y}];
outerRegion = Rectangle[
   {-0.1 separation, -0.75 mirrorWidth},
   {1.1 separation, 0.75 mirrorWidth}];
domain = RegionDifference[outerRegion, RegionUnion[leftMirror, rightMirror]];

pulseHalfWidth = 0.004;
input[t_] := Evaluate[
   1000 WaveletPsi[
     DGaussianWavelet[1],
     (Pi Sqrt[2])/pulseHalfWidth (t - pulseHalfWidth)]];
regularizedDelta[gamma_, position_List, source_List] := Piecewise[{
    {Times @@ Thread[
        1/(4 gamma) (1 + Cos[Pi/(2 gamma) (position - source)])],
     And @@ Thread[RealAbs[position - source] <= 2 gamma]},
    {0, True}
    }];

maximumFrequency = 600.;
meshSize = cAir/(3 maximumFrequency);
parameters = <|
   "SoundSpeed" -> cAir,
   "Density" -> rhoAir,
   "MonopoleSource" ->
    regularizedDelta[meshSize, {x, y}, {focalLength, 0}] input[t]
   |>;
mesh = ToElementMesh[domain, MaxCellMeasure -> {"Length" -> meshSize}];
silentInitialConditions = {
   p[0, x, y] == 0,
   Derivative[1, 0, 0][p][0, x, y] == 0
   };
absorbing = AcousticAbsorbingValue[
   Abs[y] > mirrorWidth/2 || x < -thickness ||
    x > separation + thickness,
   vars, parameters];
pde = {
   AcousticPDEComponent[vars, parameters] == absorbing,
   silentInitialConditions
   };
endTime = 0.06;
solution = NDSolveValue[
   pde, p, {t, 0, endTime}, Element[{x, y}, mesh]];

freeTravelTime = (separation - 2 focalLength)/cAir;
refocusTime = (separation + 2 focalLength)/cAir;
freeField[t_] := rhoAir input[t - freeTravelTime]/(
   2 Pi Sqrt[separation - 2 focalLength]);
sampleTimes = Range[0., 2 pulseHalfWidth, pulseHalfWidth 10^-3];
mirrorPeak = Max[
   solution[# + refocusTime, separation - focalLength, 0] & /@ sampleTimes];
freePeak = Max[freeField[# + freeTravelTime] & /@ sampleTimes];

Print@ExportString[
  <|"schemaVersion" -> 1,
    "mirrorPeakPascal" -> mirrorPeak,
    "freePeakPascal" -> freePeak,
    "amplificationRatio" -> mirrorPeak/freePeak,
    "publishedAmplificationRatio" -> 2.96217|>,
  "RawJSON"];
