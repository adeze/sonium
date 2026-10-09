(* Literal deterministic extractor for Wolfram's
   "Multilayer Sphere Heat Transfer" model:
   https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/ModelCollection/MultilayerSphereHeatTransfer.html?view=all

   This script preserves the published three concentric hollow layers,
   material coefficients, angular heat flux, and time interval. It emits
   fixed radial samples instead of relying on the page's plots. Sonium does
   not yet claim a matching Rust regression because the literal case needs a
   conforming marked 3D shell mesh and spatial boundary-flux adapter. *)

Needs["NDSolve`FEM`"];

r0 = 1.;
r1 = 2.;
r2 = 4.;
r3 = 6.;

shape = OpenCascadeShape[Ball[{0, 0, 0}, #]] & /@ {r0, r1, r2, r3};
faces = OpenCascadeShapeFaces /@ shape;
obj = OpenCascadeShapeUnion[Flatten[faces]];
bmesh = OpenCascadeShapeSurfaceMeshToBoundaryMesh[obj];

regionMarkers = <|"layer1" -> 1, "layer2" -> 2, "layer3" -> 3|>;
mesh = ToElementMesh[
   bmesh,
   "RegionHoles" -> {{0, 0, 0}},
   "RegionMarker" -> {
     {{(r0 + r1)/2, 0, 0}, regionMarkers["layer1"]},
     {{(r1 + r2)/2, 0, 0}, regionMarkers["layer2"]},
     {{(r2 + r3)/2, 0, 0}, regionMarkers["layer3"]}
     },
   MaxCellMeasure -> 0.06
   ];

kappa = Which[
   ElementMarker == regionMarkers["layer1"], 1.,
   ElementMarker == regionMarkers["layer2"], 2.,
   ElementMarker == regionMarkers["layer3"], 4.
   ];
rho = 1.;
cp = Which[
   ElementMarker == regionMarkers["layer1"], 1.,
   ElementMarker == regionMarkers["layer2"], 0.5,
   ElementMarker == regionMarkers["layer3"], 4./9.
   ];

vars = {u[t, x, y, z], t, {x, y, z}};
pars = <|
   "MassDensity" -> rho,
   "SpecificHeatCapacity" -> cp,
   "ThermalConductivity" -> kappa
   |>;
pars["BoundaryCondition1"] = <|"SurfaceTemperature" -> 0.|>;

transformRules = Thread[
   {r, theta, phi} ->
    CoordinateTransformData["Cartesian" -> "Spherical", "Mapping", {x, y, z}]
   ];
heatFlux = theta^2 (Pi - theta)^2 phi^2 (Pi - phi)^2;
pars["BoundaryCondition2"] = <|"HeatFlux" -> heatFlux|>;
innerCondition = HeatTemperatureCondition[
     r == r0, vars, pars, "BoundaryCondition1"] /. transformRules;
outerFlux = HeatFluxValue[
     r == r3 && 0 <= theta <= Pi && 0 <= phi <= Pi,
     vars,
     pars,
     "BoundaryCondition2"
     ] /. transformRules;

solution = NDSolveValue[
   {
    HeatTransferPDEComponent[vars, pars] == outerFlux,
    innerCondition,
    u[0, x, y, z] == 0.
    },
   u,
   {t, 0., 50.},
   Element[{x, y, z}, mesh]
   ];

sampleTimes = {5., 10., 20., 30., 45., 50.};
sampleRadii = {1.5, 3., 5.};
directions = <|
   "positiveX" -> {1., 0., 0.},
   "positiveY" -> {0., 1., 0.},
   "negativeY" -> {0., -1., 0.}
   |>;

samples = AssociationMap[
   Function[direction,
    Table[
     <|
      "time" -> time,
      "radius" -> radius,
      "temperature" ->
       N[solution[time, Sequence @@ (radius directions[direction])], 16]
      |>,
     {time, sampleTimes}, {radius, sampleRadii}
     ]
    ],
   Keys[directions]
   ];

(* The linear PDE and concentric materials decouple by spherical-harmonic
   degree. These angular averages expose the exact degree-zero mode used by
   Sonium's radial-average regression without claiming the non-axisymmetric
   field. *)
sphericalAverageFlux = (12. - Pi^2) Pi^4/30.;
sphericalAverageSamples = Table[
  N[1/(4 Pi) NIntegrate[
    solution[
      time,
      radius Sin[theta] Cos[phi],
      radius Sin[theta] Sin[phi],
      radius Cos[theta]
    ] Sin[theta],
    {theta, 0., Pi}, {phi, -Pi, Pi},
    Method -> "LocalAdaptive"
  ], 16],
  {time, sampleTimes}, {radius, sampleRadii}
];

payload = <|
   "source" -> "Wolfram Multilayer Sphere Heat Transfer",
   "radii" -> {r0, r1, r2, r3},
   "conductivities" -> {1., 2., 4.},
   "densities" -> {1., 1., 1.},
   "specificHeatCapacities" -> {1., 0.5, 4./9.},
   "timeInterval" -> {0., 50.},
   "sampleTimes" -> sampleTimes,
   "sampleRadii" -> sampleRadii,
   "sphericalAverageFlux" -> N[sphericalAverageFlux, 16],
   "sphericalAverageSamples" -> sphericalAverageSamples,
   "samples" -> samples
   |>;

WriteString[$Output, ExportString[payload, "RawJSON"]];
