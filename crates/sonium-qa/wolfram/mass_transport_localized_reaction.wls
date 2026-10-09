(* Literal localized-reaction example from Wolfram's Mass Transport tutorial. *)
Needs["NDSolve`FEM`"];

Ω = Rectangle[];
Ωreaction = Rectangle[{4/10, 45/100}, {6/10, 55/100}];
vars = {c[t, x, y], t, {x, y}};
pars = <|"DiffusionCoefficient" -> 0.001|>;
pars["MassReactionRate"] = If[RegionMember[Ωreaction][{x, y}], 1, 0];
pde = {
  MassTransportPDEComponent[vars, pars] == 0,
  c[0, x, y] == 1000
};
cfun = NDSolveValue[pde, c, {t, 0, 100}, Element[{x, y}, Ω]];
samplePoints = {{0., 0.}, {0.2, 0.5}, {0.4, 0.5}, {0.5, 0.5}, {0.8, 0.5}};

result = <|
  "schema_version" -> 1,
  "diffusivity_m2_s" -> 0.001,
  "reaction_rate_s_inv" -> 1.,
  "final_time_s" -> 100.,
  "sample_points" -> samplePoints,
  "final_integral" -> NIntegrate[cfun[100, x, y], {x, 0, 1}, {y, 0, 1}],
  "samples" -> (cfun[100, #[[1]], #[[2]]] & /@ samplePoints)
|>;
Print[ExportString[result, "RawJSON", "Compact" -> True]];
