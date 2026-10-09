(* Literal extractor for the two-medium example in AcousticsTimeDomain. *)
vars = {p[t, x], t, {x}};
tend = 0.002;
beta = 0.05;
p0[x_] := 2*Exp[-x^2/beta^2];
ics = {p[0, x] == p0[x], Derivative[1, 0][p][0, x] == 0};
pressure = "Pressure" -> Quantity[101325, "Pascals"];
pt1 = {pressure, "Temperature" -> Quantity[250, "Kelvins"]};
pt2 = {pressure, "Temperature" -> Quantity[600, "Kelvins"]};
c1 = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed", pt1]];
rho1 = QuantityMagnitude[ThermodynamicData["Air", "Density", pt1]];
c2 = QuantityMagnitude[ThermodynamicData["Air", "SoundSpeed", pt2]];
rho2 = QuantityMagnitude[ThermodynamicData["Air", "Density", pt2]];
parameters = <|
  "SoundSpeed" -> If[x <= 1/2, c1, c2],
  "MassDensity" -> If[x <= 1/2, rho1, rho2]
|>;
region = Line[{{0}, {1.5}}];
equation = AcousticPDEComponent[vars, parameters] == 0;
solution = NDSolveValue[
  {equation, ics}, p, {t, 0, tend}, Element[x, region],
  Method -> {"PDEDiscretization" -> {"MethodOfLines",
    "SpatialDiscretization" -> {"FiniteElement",
      "MeshOptions" -> {"MaxCellMeasure" -> {"Length" -> beta/12}}
    }
  }}
];
z1 = c1*rho1;
z2 = c2*rho2;
reflection = (z2 - z1)/(z2 + z1);
transmission = 2*z2/(z2 + z1);
Print[ExportString[
  <|
    "schema_version" -> 1,
    "case_id" -> "AcousticsTimeDomain-TwoMedia-GaussianPulse",
    "end_time_s" -> tend,
    "interface_m" -> 0.5,
    "pulse_width_m" -> beta,
    "cold" -> <|"temperature_k" -> 250, "sound_speed_m_s" -> c1,
      "density_kg_m3" -> rho1, "impedance_pa_s_m" -> z1|>,
    "hot" -> <|"temperature_k" -> 600, "sound_speed_m_s" -> c2,
      "density_kg_m3" -> rho2, "impedance_pa_s_m" -> z2|>,
    "reflection_amplitude" -> reflection,
    "transmission_amplitude" -> transmission,
    "wolfram_reflected_minimum_pa" ->
      FindMinValue[solution[tend, x], {x, 0.2, 0.5}],
    "wolfram_transmitted_maximum_pa" ->
      FindMaxValue[solution[tend, x], {x, 0.5, 0.85}]
  |>,
  "RawJSON"
]];
