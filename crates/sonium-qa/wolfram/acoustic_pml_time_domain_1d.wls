(* Standalone extractor for Wolfram's 1D time-domain acoustic PML example. *)

left = 0.;
right = 0.5;
pmlWidth = 0.25;
outerRight = right + pmlWidth;
cair = 343.2;
rhoair = 1.2041;
omega = 5025.;
impedance = rhoair cair;
sigmaMax = 100.;
tend = 0.0025;

sigmaX[x_] := If[x >= right, (x - right)/pmlWidth sigmaMax, 0.];
equations = {
  D[p[t, x], t] + cair sigmaX[x] p[t, x] +
    rhoair cair^2 D[v[t, x], x] == 0,
  D[v[t, x], t] + cair sigmaX[x] v[t, x] +
    1/rhoair D[p[t, x], x] == 0
};
initialConditions = {p[0, x] == 0, v[0, x] == 0};
boundaryConditions = {
  p[t, left] == Sin[omega t],
  v[t, left] == Sin[omega t]/impedance
};

solution = NDSolveValue[
  Join[equations, initialConditions, boundaryConditions],
  {p, v},
  {t, 0, tend},
  {x, left, outerRight},
  Method -> {"PDEDiscretization" -> {"FiniteElement",
    "MeshOptions" -> {"MaxCellMeasure" -> 0.0025}}}
];

sampleX = N[Subdivide[left, outerRight, 300]];
Export["acoustic_pml_time_domain_1d.json", <|
  "source" -> "Wolfram AcousticsTimeDomain 1D PML example",
  "physical_domain_m" -> {left, right},
  "pml_width_m" -> pmlWidth,
  "sigma_max" -> sigmaMax,
  "angular_frequency_rad_s" -> omega,
  "end_time_s" -> tend,
  "x_m" -> sampleX,
  "pressure_pa" -> (solution[[1]][tend, #] & /@ sampleX),
  "particle_velocity_m_s" -> (solution[[2]][tend, #] & /@ sampleX)
|>, "RawJSON"];
