(* Literal Catalytic Converter from Wolfram's PDEModels collection. *)
Needs["NDSolve`FEM`"];

vars = {c[t, x, y], t, {x, y}};
pars = <|"ModelForm" -> "Conservative"|>;
pars["DiffusionCoefficient"] = 1.6*10^-5;
W = 0.2;
L = 0.5;
Ω = Rectangle[{0, 0}, {W, L}];
vavg = 5*10^-3;
pars["MassConvectionVelocity"] = {0, 2*vavg*(1 - ((x - W/2)/(W/2))^2)};
c0 = 1500;
ics = {c[0, x, y] == c0};
Γin = MassConcentrationCondition[
  y == 0,
  vars,
  pars,
  <|"MassConcentration" -> c0|>
];
Γout = MassOutflowValue[y == L, vars, pars];
k = 10^-6;
S0 = 1000;
Γactive = MassFluxValue[
  x == W,
  vars,
  pars,
  <|"MassFlux" -> -k*c[t, x, y]*(S0 - (c0 - c[t, x, y]))|>
];
tend = 150;
pde = {
  MassTransportPDEComponent[vars, pars] == Γactive + Γout,
  Γin,
  ics
};
cfun = NDSolveValue[pde, c, {t, 0, tend}, Element[{x, y}, Ω]];
samplePoints = {{0.1, 0.25}, {0.2, 0.25}, {0.1, 0.5}, {0.1, 0.}};
outlet = NIntegrate[cfun[tend, x, L], {x, 0, W}]/W;

result = <|
  "schema_version" -> 1,
  "final_time_s" -> tend,
  "sample_points" -> samplePoints,
  "samples" -> (cfun[tend, #[[1]], #[[2]]] & /@ samplePoints),
  "outlet_concentration" -> outlet,
  "reduction" -> c0 - outlet
|>;
Print[ExportString[result, "RawJSON", "Compact" -> True]];
