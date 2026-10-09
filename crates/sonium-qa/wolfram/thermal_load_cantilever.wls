(* Literal deterministic extractor for Wolfram's "Thermal Load" model:
   https://reference.wolfram.com/language/PDEModels/tutorial/Multiphysics/ModelCollection/ThermalLoad.html?view=all

   This preserves the published one-way transient coupling: solve the steel
   beam temperature first, then use that spatial time-dependent field as the
   plane-stress thermal strain temperature. Sonium does not yet claim a Rust
   accuracy test because its public thermoelastic load accepts only a uniform
   scalar temperature and has no matching transient vector problem builder. *)

Needs["NDSolve`FEM`"];

density = 7500.;
specificHeat = 470.;
conductivity = 44.;
youngModulus = 210.*10^9;
thermalExpansion = 12.*10^-6;
poissonRatio = 0.3;
length = 5.;
width = 0.5;
heatFlux = 10000.;
endTime = 10800.;
coldTemperature = 0.;

domain = Rectangle[{0., 0.}, {length, width}];
mesh = ToElementMesh[domain, MaxCellMeasure -> 5.*10^-3];

thermalVariables = {temperature[t, x, y], t, {x, y}};
thermalParameters = <|
   "MassDensity" -> density,
   "SpecificHeatCapacity" -> specificHeat,
   "ThermalConductivity" -> conductivity IdentityMatrix[2]
   |>;
thermalEquation = HeatTransferPDEComponent[
    thermalVariables, thermalParameters];
temperatureCondition = HeatTemperatureCondition[
   x == 0 || y == 0,
   thermalVariables,
   <||>,
   <|"SurfaceTemperature" -> coldTemperature|>
   ];
fluxCondition = HeatFluxValue[
   y == width,
   thermalVariables,
   <||>,
   <|"HeatFlux" -> heatFlux|>
   ];
temperatureFunction = NDSolveValue[
   {
    thermalEquation == fluxCondition,
    temperatureCondition,
    temperature[0, x, y] == coldTemperature
    },
   temperature,
   {t, 0., endTime},
   Element[{x, y}, mesh]
   ];

structuralVariables = {
   {u[t, x, y], v[t, x, y]},
   t,
   {x, y}
   };
structuralParameters = <|
   "SolidMechanicsModelForm" -> "PlaneStress",
   "MassDensity" -> density,
   "YoungModulus" -> youngModulus,
   "PoissonRatio" -> poissonRatio,
   "Thickness" -> 1.,
   "ThermalExpansion" -> thermalExpansion,
   "ThermalStrainTemperature" -> temperatureFunction[t, x, y]
   |>;
structuralEquation = SolidMechanicsPDEComponent[
   structuralVariables, structuralParameters];
fixedCondition = SolidFixedCondition[
   x == 0, structuralVariables, <||>];

{uFunction, vFunction} = NDSolveValue[
   {
    structuralEquation == {0., 0.},
    fixedCondition,
    u[0, x, y] == 0.,
    v[0, x, y] == 0.,
    Derivative[1, 0, 0][u][0, x, y] == 0.,
    Derivative[1, 0, 0][v][0, x, y] == 0.
    },
   {u, v},
   {t, 0., endTime},
   Element[{x, y}, mesh]
   ];

sampleTimes = {0., 1800., 3600., 7200., 10800.};
samplePoints = {
   {length, 0.},
   {length, width/2.},
   {length, width}
   };
samples = Flatten[
   Table[
    <|
     "time_s" -> time,
     "point_m" -> point,
     "temperature_c" ->
      N[temperatureFunction[time, Sequence @@ point], 16],
     "displacement_m" -> {
       N[uFunction[time, Sequence @@ point], 16],
       N[vFunction[time, Sequence @@ point], 16]
       }
     |>,
    {time, sampleTimes}, {point, samplePoints}
    ],
   1
   ];

payload = <|
   "schema_version" -> 1,
   "case_id" -> "Wolfram-ThermalLoad",
   "dimensions_m" -> {length, width, 1.},
   "density_kg_m3" -> density,
   "specific_heat_j_kg_k" -> specificHeat,
   "conductivity_w_m_k" -> conductivity,
   "young_modulus_pa" -> youngModulus,
   "poisson_ratio" -> poissonRatio,
   "thermal_expansion_1_k" -> thermalExpansion,
   "top_heat_flux_w_m2" -> heatFlux,
   "end_time_s" -> endTime,
   "max_abs_u_m" -> N[Max[Abs[uFunction["ValuesOnGrid"]]], 16],
   "max_abs_v_m" -> N[Max[Abs[vFunction["ValuesOnGrid"]]], 16],
   "published_max_abs_u_m" -> 0.0049009,
   "published_max_abs_v_m" -> 0.0243852,
   "samples" -> samples
   |>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True]];
