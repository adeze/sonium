(* Hermetic degree-zero reduction of Wolfram's multilayer-sphere heat model.
   Angular averaging decouples exactly because the coefficients are concentric;
   multiplying by r^2 gives the conservative one-dimensional weak form. *)
Needs["NDSolve`FEM`"];

r0 = 1.; r1 = 2.; r2 = 4.; r3 = 6.;
k[r_] := Piecewise[{{1., r <= r1}, {2., r <= r2}}, 4.];
cp[r_] := Piecewise[{{1., r <= r1}, {0.5, r <= r2}}, 4./9.];
sphericalAverageFlux = (12. - Pi^2) Pi^4/30.;

vars = {u[t, r], t, {r}};
pars = <|
  "MassDensity" -> r^2 cp[r],
  "SpecificHeatCapacity" -> 1.,
  "ThermalConductivity" -> r^2 k[r]
|>;
equation = HeatTransferPDEComponent[vars, pars] ==
  NeumannValue[r3^2 sphericalAverageFlux, r == r3];
solution = NDSolveValue[
  {equation, DirichletCondition[u[t, r] == 0., r == r0], u[0., r] == 0.},
  u, {t, 0., 50.}, Element[{r}, Line[{{r0}, {r3}}]]
];

sampleTimes = {5., 10., 20., 30., 45., 50.};
sampleRadii = {1.5, 3., 5.};
payload = <|
  "provenance" -> "Exact spherical-harmonic degree-zero reduction of Wolfram Multilayer Sphere Heat Transfer",
  "sphericalAverageFlux" -> N[sphericalAverageFlux, 16],
  "sphericalAverageSamples" -> Table[N[solution[time, radius], 16], {time, sampleTimes}, {radius, sampleRadii}]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
