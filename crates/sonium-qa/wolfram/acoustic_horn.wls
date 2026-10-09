(* Literal numerical extractor for Wolfram's Acoustic Horn model collection
   example. It prints the 96-point straight and curved reflection spectra. *)

Needs["NDSolve`FEM`"];

cAir = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoAir = QuantityMagnitude[ThermodynamicData["Air", "Density"]];
vars = {p[x, y], omega, {x, y}};
pars = <|"SoundSpeed" -> cAir, "MassDensity" -> rhoAir|>;
model = AcousticPDEComponent[vars, pars];

t = a/2;
theta = ArcTan[(b - a)/l];
rules = {r -> 1, a -> 1/15, b -> 3/10, l -> 1/2, d -> 1/2};
rec1 = Rectangle[{-r, a}, {-r + d, a + t}];
rec2 = Polygon @@ (RotationTransform[theta, {-r + d, a}][#] & /@
     (List @@ RegionBoundary[
       Rectangle[{-r + d, a}, {-r + d + l/Cos[theta], a + t}]
     ]));
straightRegion = RegionDifference[
    Disk[{0, 0}, r, {0, Pi}], RegionUnion[rec1, rec2]
  ] /. rules;

curvatureRadius = 2;
{x0, y0} = {x, y} /. Last[NSolve[{
    x^2 + (y - b)^2 == curvatureRadius^2,
    (x + r)^2 + (y - a)^2 == curvatureRadius^2
  }, {x, y}]];
phi = VectorAngle[{-r - x0, a - y0}, {-x0, b - y0}];
gamma = 3/2 Pi - VectorAngle[{0, -1}, {-r - x0, a - y0}];
curve = Annulus[
   {x0, y0}, {curvatureRadius - t, curvatureRadius},
   {gamma, gamma + phi}
];
curvedRegion = RegionDifference[
    Disk[{0, 0}, r, {0, Pi}], curve
  ] /. rules;

h = (cAir/1000)/12;
straightMesh = ToElementMesh[straightRegion, MaxCellMeasure -> {"Length" -> h}];
curvedMesh = ToElementMesh[curvedRegion, MaxCellMeasure -> {"Length" -> h}];
inlet = Circle[{0, 0}, r, {Pi - ArcSin[a/r], Pi}] /. rules;
outlet = Circle[{0, 0}, r, {0, Pi - ArcSin[(a + t)/r]}] /. rules;
inletCondition = AcousticRadiationValue[
  Element[{x, y}, inlet], vars, pars, <|"SoundIncidentPressure" -> 1|>
];
outletCondition = AcousticAbsorbingValue[
   Element[{x, y}, outlet], vars, pars,
   <|"AcousticSourceDistance" -> 1/(2 r)|>
 ] /. rules;
pde = model == inletCondition + outletCondition;

frequencyOverride = Environment["WOLFRAM_HORN_FREQUENCIES"];
frequencies = If[
  StringQ[frequencyOverride] && StringLength[frequencyOverride] > 0,
  N[ToExpression["{" <> frequencyOverride <> "}"]],
  N[Range[50, 1000, 10]]
];
straightSolution = ParametricNDSolveValue[
  pde, p, Element[{x, y}, straightMesh], {omega}
];
curvedSolution = ParametricNDSolveValue[
  pde, p, Element[{x, y}, curvedMesh], {omega}
];
straightFields = straightSolution[2 Pi #] & /@ frequencies;
curvedFields = curvedSolution[2 Pi #] & /@ frequencies;
inletLength = N[ArcLength[inlet]];
reflection[fields_] := Abs[
  NIntegrate[#[x, y], Element[{x, y}, inlet]]/inletLength - 1
] & /@ fields;

Print[ExportString[
  <|
    "schema_version" -> 1,
    "sound_speed_m_s" -> N[cAir],
    "mass_density_kg_m3" -> N[rhoAir],
    "target_edge_length_m" -> N[h],
    "frequencies_hz" -> frequencies,
    "straight_reflection" -> reflection[straightFields],
    "curved_reflection" -> reflection[curvedFields]
  |>,
  "RawJSON", "Compact" -> True
]];
