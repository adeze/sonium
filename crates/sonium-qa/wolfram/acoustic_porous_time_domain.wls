(* Literal extractor for the porous-absorber example in AcousticsTimeDomain. *)
Needs["NDSolve`FEM`"];
cair = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed"]];
rhoair = QuantityMagnitude[ThermodynamicData["Air", "Density"]];
left = 0;
right = 1/2;
officialEndTime = 0.002;
extendedEndTime = 0.008;
frequency = 1000;
omega = 2*Pi*frequency;
porosity = 0.8;
flowResistivity = 5000;
bulkModulus = rhoair*cair^2*
  (1 + 0.0978*(rhoair*frequency/flowResistivity)^-0.7)^-2;
vars = {p[t, x], t, {x}};
lossyModel = 1/bulkModulus*D[p[t, x], {t, 2}] +
  porosity*flowResistivity/(rhoair*bulkModulus)*D[p[t, x], t] +
  AcousticPDEComponent[{p[t, x], {x}}, <|"MassDensity" -> rhoair|>];
lossyAbc = NeumannValue[
  -(cair/bulkModulus)*Derivative[1, 0][p][t, x] -
  cair*porosity*flowResistivity/(rhoair*bulkModulus)*p[t, x],
  x == right
];
source = DirichletCondition[p[t, x] == Sin[omega*t], x == left];
initial = {p[0, x] == 0, Derivative[1, 0][p][0, x] == 0};
region = Line[{{left}, {right}}];
solution = NDSolveValue[
  {lossyModel == lossyAbc, initial, source},
  p, {t, 0, extendedEndTime}, Element[x, region],
  Method -> {"PDEDiscretization" -> {"MethodOfLines",
    "SpatialDiscretization" -> {"FiniteElement",
      "MeshOptions" -> {"MaxCellMeasure" -> {"Length" -> cair/(12*frequency)}}
    }
  }}
];
k = Sqrt[rhoair*omega^2/bulkModulus -
  I*omega*porosity*flowResistivity/bulkModulus];
amplitude[x_] := Exp[Im[k]*x];
probeAmplitude[x_] := 1/2*(
  NMaxValue[{solution[t, x], 0.007 <= t <= 0.008}, t] -
  NMinValue[{solution[t, x], 0.007 <= t <= 0.008}, t]);
samplePoints = N[Range[0, 1/2, 1/100], 16];
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "AcousticsTimeDomain-PorousAbsorber",
    "frequency_hz" -> frequency,
    "porosity" -> porosity,
    "flow_resistivity_pa_s_m2" -> flowResistivity,
    "bulk_modulus_pa" -> bulkModulus,
    "complex_wavenumber_rad_m" -> {Re[k], Im[k]},
    "official_end_time_s" -> officialEndTime,
    "wolfram_official_field_pa" ->
      (solution[officialEndTime, #] & /@ samplePoints),
    "probe_x_m" -> {0.25, 0.45},
    "wolfram_probe_amplitude_pa" -> {probeAmplitude[0.25], probeAmplitude[0.45]},
    "complex_wavenumber_amplitude_pa" -> {amplitude[0.25], amplitude[0.45]}
  |>,
  "RawJSON"
]];
