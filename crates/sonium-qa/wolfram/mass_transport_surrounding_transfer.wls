(* Literal CO2 surrounding-transfer example from Wolfram's Mass Transport tutorial. *)
Needs["NDSolve`FEM`"];

left = 0;
right = 1;
Ω = Line[{{left}, {right}}];
vars = {c[t, x], t, {x}};
pars = <|"DiffusionCoefficient" -> 1.64*10^-5|>;
ics = {c[0, x] == 1000*Exp[-(x - 0.5)^2/0.2^2]};
ΓsurFlux = MassTransferValue[
  x == left || x == right,
  vars,
  pars,
  <|"MassTransferCoefficient" -> 8.2*10^-3, "AmbientConcentration" -> 0|>
];
pde = {MassTransportPDEComponent[vars, pars] == ΓsurFlux, ics};
cfun = NDSolveValue[pde, c, {t, 0, 10080}, Element[x, Ω]];
sampleX = {0., 0.25, 0.5, 0.75, 1.};

result = <|
  "schema_version" -> 1,
  "diffusivity_m2_s" -> 1.64*10^-5,
  "transfer_coefficient_m_s" -> 0.0082,
  "ambient_concentration" -> 0.,
  "final_time_s" -> 10080.,
  "sample_x_m" -> sampleX,
  "initial_integral" -> NIntegrate[cfun[0, x], {x, left, right}],
  "final_integral" -> NIntegrate[cfun[10080, x], {x, left, right}],
  "final_profile" -> (cfun[10080, #] & /@ sampleX)
|>;
Print[ExportString[result, "RawJSON", "Compact" -> True]];
