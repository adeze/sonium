(* Standalone extractor for Wolfram's 2D time-domain acoustic PML example. *)

left = 0.;
right = 0.5;
bottom = 0.;
top = 0.5;
pmlWidth = 0.25;
cair = 343.2;
rhoair = 1.2041;
sigmaMax = 20.;
beta = 0.05;
tend = 0.002;
domain = Rectangle[
  {left - pmlWidth, bottom - pmlWidth},
  {right + pmlWidth, top + pmlWidth}
];

sigmaX[x_] := Which[
  x >= right, (x - right)/pmlWidth sigmaMax,
  x <= left, (left - x)/pmlWidth sigmaMax,
  True, 0.
];
sigmaY[y_] := Which[
  y >= top, (y - top)/pmlWidth sigmaMax,
  y <= bottom, (bottom - y)/pmlWidth sigmaMax,
  True, 0.
];

equations = {
  D[p[t, x, y], t] + cair (sigmaX[x] + sigmaY[y]) p[t, x, y] +
    rhoair cair^2 (D[u[t, x, y], x] + D[v[t, x, y], y]) +
    rhoair cair^2 sigmaX[x] D[q[t, x, y], y] +
    rhoair cair^2 sigmaY[y] D[r[t, x, y], x] == 0,
  D[u[t, x, y], t] + cair sigmaX[x] u[t, x, y] +
    1/rhoair D[p[t, x, y], x] == 0,
  D[v[t, x, y], t] + cair sigmaY[y] v[t, x, y] +
    1/rhoair D[p[t, x, y], y] == 0,
  D[q[t, x, y], t] - cair v[t, x, y] == 0,
  D[r[t, x, y], t] - cair u[t, x, y] == 0
};
p0[x_, y_] := Exp[-(x - 0.25)^2/beta^2] Exp[-(y - 0.25)^2/beta^2];
initialConditions = {
  p[0, x, y] == p0[x, y],
  u[0, x, y] == 0,
  v[0, x, y] == 0,
  q[0, x, y] == 0,
  r[0, x, y] == 0
};

solution = NDSolveValue[
  Join[equations, initialConditions],
  {p, u, v, q, r},
  {t, 0, tend},
  Element[{x, y}, domain],
  Method -> {"PDEDiscretization" -> {"FiniteElement",
    "MeshOptions" -> {"MaxCellMeasure" -> 0.0002}}}
];

samplePoints = Flatten[
  Table[{xx, yy},
    {xx, N[Subdivide[left - pmlWidth, right + pmlWidth, 40]]},
    {yy, N[Subdivide[bottom - pmlWidth, top + pmlWidth, 40]]}],
  1
];
Export["acoustic_pml_time_domain_2d.json", <|
  "source" -> "Wolfram AcousticsTimeDomain 2D PML example",
  "physical_domain_m" -> {{left, bottom}, {right, top}},
  "pml_width_m" -> pmlWidth,
  "sigma_max" -> sigmaMax,
  "gaussian_width_m" -> beta,
  "end_time_s" -> tend,
  "points_m" -> samplePoints,
  "pressure_pa" -> (solution[[1]][tend, #[[1]], #[[2]]] & /@ samplePoints)
|>, "RawJSON"];
