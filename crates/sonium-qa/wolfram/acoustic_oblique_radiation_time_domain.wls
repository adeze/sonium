(* Exact extractor for Radiated Sound Waves with Oblique Incidence. *)
Needs["NDSolve`FEM`"];

cair = 343.2;
rhoair = 1.2041;
omega = 5025.;
tend = 0.002;
theta = Pi/6;
domain = Polygon[{{0, 0}, {1/5, 0},
  {1/5 + Sin[theta] - Sin[theta]^2/5,
   Cos[theta] - (1/5 Cos[theta]) Sin[theta]},
  {Sin[theta], Cos[theta]}}];
mesh = ToElementMesh[domain];

vars2D = {p[t, x, y], t, {x, y}};
pars = <|"SoundSpeed" -> cair, "MassDensity" -> rhoair|>;
k = omega/cair;
kx = k Sin[theta];
ky = k Cos[theta];
incident = Cos[-(kx x + ky y) + omega t];
initialConditions = {
  p[0, x, y] == incident,
  Derivative[1, 0, 0][p][0, x, y] == D[incident, t]
} /. t -> 0;
incoming = AcousticRadiationValue[
  ElementMarker == 1,
  vars2D,
  pars,
  <|"BoundaryUnitNormal" -> {0, 1},
    "SoundDirectionVector" -> {kx, ky},
    "SoundIncidentPressure" -> incident|>
];
outgoing = AcousticAbsorbingValue[ElementMarker == 3, vars2D, pars];
pde = {
  AcousticPDEComponent[vars2D, pars] == incoming + outgoing,
  initialConditions
};
solution = NDSolveValue[pde, p, {t, 0, tend}, Element[{x, y}, mesh]];

samplePoints = mesh["Coordinates"];
Export["acoustic_oblique_radiation_time_domain.json", <|
  "source" -> "Wolfram AcousticsTimeDomain oblique radiation example",
  "angle_rad" -> theta,
  "angular_frequency_rad_s" -> omega,
  "end_time_s" -> tend,
  "points_m" -> samplePoints,
  "pressure_pa" -> (solution[tend, #[[1]], #[[2]]] & /@ samplePoints)
|>, "RawJSON"];
