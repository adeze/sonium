(* Literal reference for Wolfram PDEModels/Acoustics/ModelCollection/
   MotorNoiseAnalysis, with and without the optional sound barrier. *)
Needs["NDSolve`FEM`"];

cAir = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoAir = QuantityMagnitude[ThermodynamicData["Air", "Density"]];
rMotor = 1/10;
rDomain = 3;
d = 2 10^-3;
rpm = {15000, 17500, 20000};
omegaMax = 2 Pi Max[rpm]/60;
lambdaMin = cAir/(Max[rpm]/60);
domain = Annulus[{0, 0}, {rMotor, rDomain}];
mesh = ToElementMesh[domain, MaxCellMeasure -> {"Length" -> lambdaMin/12}];

vars = {p[x, y], omega, {x, y}};
pars = <|"SoundSpeed" -> cAir, "MassDensity" -> rhoAir|>;
phi[x_, y_] := ArcTan[x, y];
velocity = 2 I omega d Exp[2 I phi[x, y]];
casing = AcousticNormalVelocityValue[
  x^2 + y^2 == rMotor^2,
  vars,
  pars,
  <|"AcousticSoundParticleVelocity" -> velocity|>
];
outer = AcousticAbsorbingValue[
  x^2 + y^2 == rDomain^2,
  vars,
  pars,
  <|"AcousticSourceDistance" -> 1/(2 rDomain)|>
];
pde = {AcousticPDEComponent[vars, pars] == casing + outer};
pfun = ParametricNDSolveValue[pde, p, Element[{x, y}, mesh], {omega}];
solutions = pfun /@ (2 Pi rpm/60);
pRef = 20 10^-6;
spl = 20 Log10[Abs[#[2, 0]]/pRef] & /@ solutions;

barrierDomain = RegionDifference[
  domain,
  Annulus[{0, 0}, {1, 6/5}, {Pi/4, 3 Pi/4}]
];
barrierMesh = ToElementMesh[
  barrierDomain,
  MaxCellMeasure -> {"Length" -> lambdaMin/12}
];
barrierPfun = ParametricNDSolveValue[
  pde, p, Element[{x, y}, barrierMesh], {omega}
];
barrierSolutions = barrierPfun /@ (2 Pi rpm/60);
barrierSplAtUpper2m =
  20 Log10[Abs[#[0, 2]]/pRef] & /@ barrierSolutions;

Print[ExportString[
  <|
    "schema_version" -> 1,
    "provenance" -> "literal Wolfram Motor Noise Analysis with and without hard barrier",
    "rpm" -> N[rpm],
    "spl_at_2m_db" -> N[spl],
    "barrier_spl_at_upper_2m_db" -> N[barrierSplAtUpper2m],
    "published_spl_at_2m_db" -> {122.286, 126.941, 131.007}
  |>,
  "RawJSON",
  "Compact" -> True
]];
