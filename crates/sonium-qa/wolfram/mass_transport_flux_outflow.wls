(* Literal transient conservative mass-flux/outflow example from Wolfram's Mass Transport tutorial. *)
Needs["NDSolve`FEM`"];

left = 0;
right = 1;
tend = 100;
Ω = Line[{{left}, {right}}];
ClearAll[q];
q[t_] := If[t <= 50, 10, 0];
vars = {c[t, x], t, {x}};
pars = <|
  "DiffusionCoefficient" -> 0.01,
  "MassConvectionVelocity" -> {0.02},
  "ModelForm" -> "Conservative"
|>;
Γflux = MassFluxValue[x == left, vars, pars, <|"MassFlux" -> q[t]|>];
Γoutflow = MassOutflowValue[
  x == right,
  vars,
  pars,
  <|"BoundaryUnitNormal" -> {1}|>
];
pde = {
  MassTransportPDEComponent[vars, pars] == Γflux + Γoutflow,
  c[0, x] == 0
};
cfun = NDSolveValue[pde, c, {t, 0, tend}, Element[x, Ω]];
sampleX = {0., 0.25, 0.5, 0.75, 1.};
sampleTime = {25., 50., 75., 100.};

result = <|
  "schema_version" -> 1,
  "diffusivity_m2_s" -> 0.01,
  "velocity_m_s" -> 0.02,
  "flux" -> 10.,
  "flux_end_time_s" -> 50.,
  "final_time_s" -> 100.,
  "sample_x_m" -> sampleX,
  "sample_time_s" -> sampleTime,
  "final_profile" -> (cfun[tend, #] & /@ sampleX),
  "outlet_history" -> (cfun[#, right] & /@ sampleTime)
|>;
Print[ExportString[result, "RawJSON", "Compact" -> True]];
