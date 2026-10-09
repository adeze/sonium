(* Literal extractor for the 1D ABC Gaussian-pulse example. *)
Needs["NDSolve`FEM`"];
cair = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoair = QuantityMagnitude[ThermodynamicData["Air", "Density"]];
left = 0;
right = 1/2;
tend = 0.002;
region = Line[{{left}, {right}}];
beta = 0.1;
p0[x_] := 2*Exp[-x^2/beta^2];
initial = {p[0, x] == p0[x], Derivative[1, 0][p][0, x] == 0};
vars = {p[t, x], t, {x}};
parameters = <|"SoundSpeed" -> cair, "MassDensity" -> rhoair|>;
abc = AcousticAbsorbingValue[x == right, vars, parameters];
equation = {AcousticPDEComponent[vars, parameters] == abc, initial};
solution = NDSolveValue[
  equation, p, {t, 0, tend}, Element[x, region],
  Method -> {"PDEDiscretization" -> {"MethodOfLines",
    "SpatialDiscretization" -> {"FiniteElement",
      "MeshOptions" -> {"MaxCellMeasure" -> {"Length" -> beta/12}}
    }
  }}
];
analytical[t_, x_] :=
  Exp[-(cair*t - x)^2/beta^2] + Exp[-(cair*t + x)^2/beta^2];
samplePoints = N[Range[0, 1/2, 1/200], 16];
errors = Abs[(solution[tend, #] - analytical[tend, #]) & /@ samplePoints];
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "AcousticsTimeDomain-NormalIncidenceABC",
    "sound_speed_m_s" -> cair,
    "density_kg_m3" -> rhoair,
    "length_m" -> right,
    "end_time_s" -> tend,
    "pulse_width_m" -> beta,
    "sample_points_m" -> samplePoints,
    "wolfram_pressure_pa" -> (solution[tend, #] & /@ samplePoints),
    "analytical_pressure_pa" -> (analytical[tend, #] & /@ samplePoints),
    "maximum_absolute_error_pa" -> Max[errors]
  |>,
  "RawJSON"
]];
