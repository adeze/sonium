(* Exact extractor for the periodic-tube example in AcousticsTimeDomain. *)
Needs["NDSolve`FEM`"];

left = 0.;
right = 0.5;
length = right - left;
tend = 0.002;
cair = 343.2;
rhoair = 1.2041;
domain = Line[{{left}, {right}}];

vars = {p[t, x], t, {x}};
pars = <|"SoundSpeed" -> cair, "MassDensity" -> rhoair|>;
map = Function[x, x + length];
periodic = PeriodicBoundaryCondition[p[t, x], x == left, map];
abcRight = AcousticAbsorbingValue[x == right, vars, pars];
p0[x_] := D[0.125 Erf[(x - 0.25)/0.075], x];
initialConditions = {
  p[0, x] == p0[x],
  Derivative[1, 0][p][0, x] == -cair D[p0[x], x]
};
pde = {
  AcousticPDEComponent[vars, pars] == abcRight,
  periodic,
  initialConditions
};

solution = NDSolveValue[
  pde,
  p,
  {t, 0, tend},
  Element[x, domain],
  Method -> {"PDEDiscretization" -> {"FiniteElement",
    "MeshOptions" -> {"MaxCellMeasure" -> 0.0025}}}
];

sampleX = N[Subdivide[left, right, 200]];
Export["acoustic_periodic_time_domain.json", <|
  "source" -> "Wolfram AcousticsTimeDomain periodic boundary example",
  "sound_speed_m_s" -> cair,
  "length_m" -> length,
  "end_time_s" -> tend,
  "x_m" -> sampleX,
  "pressure_pa" -> (solution[tend, #] & /@ sampleX)
|>, "RawJSON"];
