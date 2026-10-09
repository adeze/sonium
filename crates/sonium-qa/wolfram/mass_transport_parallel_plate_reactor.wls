(* Literal steady parallel-plate reactor from Wolfram's Mass Transport tutorial. *)
Needs["NDSolve`FEM`"];

Ω = Rectangle[{0, 0}, {0.2, 0.5}];
vars = {c[x, y], {x, y}};
pars = <||>;
pars["DiffusionCoefficient"] = 10^-4;
vx = 0;
vavg = 5*10^-3;
pars["MassConvectionVelocity"] = {vx, 2*vavg*(1 - ((x - 0.1)/0.1)^2)};
Γin = MassConcentrationCondition[
  y == 0,
  vars,
  pars,
  <|"MassConcentration" -> 1000|>
];
Γactive = MassFluxValue[
  x == 0.2,
  vars,
  pars,
  <|"MassFlux" -> -0.5, "BoundaryUnitNormal" -> {1, 0}|>
];
Γleft = MassImpermeableBoundaryValue[
  x == 0,
  vars,
  pars,
  <|"BoundaryUnitNormal" -> {-1, 0}|>
];
cfun = NDSolveValue[
  {
    MassTransportPDEComponent[vars, pars] == Γactive + Γleft,
    Γin
  },
  c,
  Element[{x, y}, Ω]
];
samplePoints = {{0., 0.25}, {0.1, 0.25}, {0.2, 0.25}, {0.1, 0.5}, {0.1, 0.}};

result = <|
  "schema_version" -> 1,
  "diffusivity_m2_s" -> 10^-4,
  "average_velocity_m_s" -> 5*10^-3,
  "inlet_concentration" -> 1000.,
  "active_wall_flux" -> -0.5,
  "sample_points" -> samplePoints,
  "samples" -> (cfun[#[[1]], #[[2]]] & /@ samplePoints),
  "area_integral" -> NIntegrate[cfun[x, y], {x, 0, 0.2}, {y, 0, 0.5}]
|>;
Print[ExportString[result, "RawJSON", "Compact" -> True]];
