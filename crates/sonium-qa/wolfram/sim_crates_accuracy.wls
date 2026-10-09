(* Independent Wolfram Engine accuracy references for every sonium-sim-* crate.

   The Rust integration test writes only physical inputs and computed outputs.
   This file recomputes the references from equations, never from Sonium
   implementation helpers, and emits a compact machine-readable error report.

   Regimes:
   - room: direct free-field Green function (absorbing room, image path only),
   - PC fan / propeller: compact rotating-loading source kinematics,
   - speaker modeling: infinite-baffle circular piston in the far field,
   - structural mechanics: linear Euler-Bernoulli cantilever,
   - speaker spline: natural cubic interpolation used by waveguide profiles.
*)

Off[General::munfl];
ClearAll[magnitude, accuracyCheck, scenario, toComplex, naturalSplineData,
  naturalSplineValue, naturalSplineDerivative, pistonDirectivity,
  freeFieldGreen, firstOrderImagePositions, pressureLevelDb, aWeighting,
  mechanicalPistonImpedance, traceProgress];

magnitude[x_] := If[ListQ[x], Norm[Flatten[N[x]]], Abs[N[x]]];
accuracyCheck[actual_, reference_, absoluteTolerance_, relativeTolerance_] :=
  Module[{absoluteError, scale, tolerance},
    absoluteError = magnitude[N[actual] - N[reference]];
    scale = Max[magnitude[reference], absoluteTolerance];
    tolerance = absoluteTolerance + relativeTolerance scale;
    <|
      "passed" -> TrueQ[absoluteError <= tolerance],
      "absolute_error" -> N[absoluteError],
      "relative_error" -> N[absoluteError/scale],
      "tolerance" -> N[tolerance]
    |>
  ];
scenario[checks_Association] := <|
  "passed" -> And @@ Lookup[Values[checks], "passed"],
  "check_count" -> Length[checks],
  "checks" -> checks
|>;
toComplex[pair_] := pair[[1]] + I pair[[2]];
traceProgress[name_] := If[Environment["SONIUM_SIM_ACCURACY_TRACE"] === "1",
  WriteString[$Output, "completed " <> name <> " report\n"]
];

naturalSplineData[points_] := Module[{xs, ys, n, h, matrix, rhs},
  xs = points[[All, 1]]; ys = points[[All, 2]]; n = Length[points];
  h = Differences[xs]; matrix = ConstantArray[0., {n, n}];
  rhs = ConstantArray[0., n];
  matrix[[1, 1]] = 1.; matrix[[n, n]] = 1.;
  Do[
    matrix[[i, i - 1]] = h[[i - 1]];
    matrix[[i, i]] = 2. (h[[i - 1]] + h[[i]]);
    matrix[[i, i + 1]] = h[[i]];
    rhs[[i]] = 6. ((ys[[i + 1]] - ys[[i]])/h[[i]] -
      (ys[[i]] - ys[[i - 1]])/h[[i - 1]]),
    {i, 2, n - 1}
  ];
  <|"x" -> xs, "y" -> ys, "m" -> LinearSolve[matrix, rhs]|>
];
naturalSplineValue[data_, query_] := Module[{xs, ys, ms, i, h, a, b},
  xs = data["x"]; ys = data["y"]; ms = data["m"];
  i = Which[query <= First[xs], 1, query >= Last[xs], Length[xs] - 1,
    True, First@FirstPosition[Partition[xs, 2, 1], {lo_, hi_} /; lo <= query <= hi]];
  h = xs[[i + 1]] - xs[[i]]; a = (xs[[i + 1]] - query)/h;
  b = (query - xs[[i]])/h;
  a ys[[i]] + b ys[[i + 1]] + (a^3 - a) h^2 ms[[i]]/6. +
    (b^3 - b) h^2 ms[[i + 1]]/6.
];
naturalSplineDerivative[data_, query_] := Module[{xs, ys, ms, i, h, a, b},
  xs = data["x"]; ys = data["y"]; ms = data["m"];
  i = Which[query <= First[xs], 1, query >= Last[xs], Length[xs] - 1,
    True, First@FirstPosition[Partition[xs, 2, 1], {lo_, hi_} /; lo <= query <= hi]];
  h = xs[[i + 1]] - xs[[i]]; a = (xs[[i + 1]] - query)/h;
  b = (query - xs[[i]])/h;
  (ys[[i + 1]] - ys[[i]])/h - (3. a^2 - 1.) h ms[[i]]/6. +
    (3. b^2 - 1.) h ms[[i + 1]]/6.
];
pistonDirectivity[ka_, angleDegrees_] := Module[{argument},
  argument = ka Sin[angleDegrees Degree];
  If[Abs[argument] < 10^-12, 1., 2. BesselJ[1, argument]/argument]
];
freeFieldGreen[source_, listener_, frequency_, soundSpeed_] := Module[{distance},
  distance = Norm[listener - source];
  Exp[I 2. Pi frequency distance/soundSpeed]/(4. Pi distance)
];
firstOrderImagePositions[source_, dimensions_] := Join[
  {source},
  Table[ReplacePart[source, axis -> -source[[axis]]], {axis, 3}],
  Table[
    ReplacePart[source, axis -> 2. dimensions[[axis]] - source[[axis]]],
    {axis, 3}
  ]
];
pressureLevelDb[pressure_] := 20. Log[10., Max[Abs[pressure], 10^-30]/(20. 10^-6)];
aWeighting[frequency_] := Module[{f, f2, f4, numerator, denominator, raw, raw1k},
  f = Max[frequency, 1.]; f2 = f^2; f4 = f^4;
  numerator = 12194.^2 f4;
  denominator = (f2 + 20.6^2) Sqrt[(f2 + 107.7^2) (f2 + 737.9^2)] *
    (f2 + 12194.^2);
  raw = numerator/denominator;
  raw1k = 12194.^2 1000.^4/((1000.^2 + 20.6^2)
    Sqrt[(1000.^2 + 107.7^2) (1000.^2 + 737.9^2)]
    (1000.^2 + 12194.^2));
  raw/raw1k
];
mechanicalPistonImpedance[radius_, frequency_, density_, soundSpeed_] :=
  Module[{ka, area, resistance, reactance},
    ka = 2. Pi frequency radius/soundSpeed; area = Pi radius^2;
    resistance = 1. - BesselJ[1, 2. ka]/ka;
    reactance = StruveH[1, 2. ka]/ka;
    density soundSpeed area (resistance + I reactance)
  ];

inputPath = Environment["SONIUM_SIM_ACCURACY_INPUT"];
If[!StringQ[inputPath] || StringLength[inputPath] == 0 || !FileExistsQ[inputPath],
  WriteString[$Output, "SONIUM_SIM_ACCURACY_INPUT must name the Rust JSON fixture\n"];
  Exit[2]
];
input = Import[inputPath, "RawJSON"];
If[Lookup[input, "schema_version", 0] =!= 1, Exit[3]];
cases = input["cases"];

roomReport = Module[{c, actualFree, actualFirstOrder, freeReference,
    firstOrderReference, images, distances, spreading, phaseRatios,
    phaseReference, frequencies},
  c = cases["sonium-sim-room"];
  frequencies = c["frequencies_hz"];
  actualFree = Map[toComplex, c["actual_free_field_pressures"], {2}];
  actualFirstOrder = Map[toComplex, c["actual_first_order_pressures"], {2}];
  freeReference = Table[
    freeFieldGreen[c["source_m"], listener, frequency, c["sound_speed_m_s"]],
    {listener, c["listeners_m"]}, {frequency, frequencies}
  ];
  images = firstOrderImagePositions[c["source_m"], c["room_dimensions_m"]];
  firstOrderReference = Table[
    Total[freeFieldGreen[#, listener, frequency, c["sound_speed_m_s"]] & /@ images],
    {listener, c["listeners_m"]}, {frequency, frequencies}
  ];
  distances = Norm[# - c["source_m"]] & /@ c["listeners_m"];
  spreading = distances Abs[actualFree[[All, 1]]];
  phaseRatios = Table[
    actualFree[[i, j + 1]]/actualFree[[i, j]],
    {i, Length[distances]}, {j, Length[frequencies] - 1}
  ];
  phaseReference = Table[
    Exp[I 2. Pi (frequencies[[j + 1]] - frequencies[[j]]) distances[[i]]/
      c["sound_speed_m_s"]],
    {i, Length[distances]}, {j, Length[frequencies] - 1}
  ];
  scenario[<|
    "direct_green_function_grid" -> accuracyCheck[
      actualFree, freeReference, 10^-13, 3. 10^-12],
    "first_order_image_sum" -> accuracyCheck[
      actualFirstOrder, firstOrderReference, 10^-13, 5. 10^-12],
    "inverse_distance_spreading" -> accuracyCheck[
      spreading, ConstantArray[1./(4. Pi), Length[distances]], 10^-13, 3. 10^-12],
    "frequency_phase_increment" -> accuracyCheck[
      phaseRatios, phaseReference, 10^-12, 5. 10^-12]
  |>]
];
traceProgress["room"];

pcFanReport = Module[{c, diskArea, velocity, bpf, mach, cfmPerM3s,
    indices, frequencies, pressures, harmonicSpl, overallSpl,
    weightedOverallSpl, swlOffset, harmonicSwl},
  c = cases["sonium-sim-pcfan"]; cfmPerM3s = 2118.8800032893;
  diskArea = Pi (c["tip_radius_m"]^2 - c["hub_radius_m"]^2);
  velocity = c["airflow_cfm"]/cfmPerM3s/diskArea;
  bpf = c["blade_count"] Abs[c["rpm"]]/60.;
  mach = Abs[velocity]/Max[c["sound_speed_m_s"], 1.];
  indices = Range[Length[c["actual_harmonic_indices"]]];
  frequencies = indices bpf;
  pressures = c["actual_harmonic_pressures_pa"];
  harmonicSpl = pressureLevelDb /@ pressures;
  overallSpl = pressureLevelDb[Sqrt[Total[pressures^2]]];
  weightedOverallSpl = pressureLevelDb[
    Sqrt[Total[(pressures (aWeighting /@ frequencies))^2]]];
  swlOffset = 10. Log[10., 4. Pi c["reference_distance_m"]^2 (20. 10^-6)^2/
    (c["air_density_kg_m3"] c["sound_speed_m_s"] 10^-12)];
  harmonicSwl = harmonicSpl + swlOffset;
  scenario[<|
    "blade_passing_frequency" -> accuracyCheck[
      c["actual_blade_passing_frequency_hz"], bpf, 10^-12, 10^-12],
    "airflow_to_axial_velocity" -> accuracyCheck[
      c["actual_axial_velocity_m_s"], velocity, 10^-12, 10^-12],
    "axial_mach_number" -> accuracyCheck[
      c["actual_mach_number"], mach, 10^-13, 10^-12],
    "harmonic_indices" -> accuracyCheck[
      c["actual_harmonic_indices"], indices, 0., 0.],
    "harmonic_frequency_lattice" -> accuracyCheck[
      c["actual_harmonic_frequencies_hz"], frequencies, 10^-11, 10^-12],
    "harmonic_pressure_to_spl" -> accuracyCheck[
      c["actual_harmonic_spl_db"], harmonicSpl, 10^-11, 10^-12],
    "harmonic_spherical_power" -> accuracyCheck[
      c["actual_harmonic_swl_db"], harmonicSwl, 10^-10, 10^-12],
    "energetic_overall_spl" -> accuracyCheck[
      c["actual_overall_spl_db"], overallSpl, 10^-11, 10^-12],
    "iec_a_weighted_overall_spl" -> accuracyCheck[
      c["actual_a_weighted_overall_spl_db"], weightedOverallSpl, 10^-10, 10^-12],
    "spherical_overall_sound_power" -> accuracyCheck[
      c["actual_overall_swl_db"], overallSpl + swlOffset, 10^-10, 10^-12]
  |>]
];
traceProgress["PC fan"];

propellerReport = Module[{c, blades, omega, tipMach, loadingForce, base,
    count, indices, frequencies, pressures, weightedPower, band},
  c = cases["sonium-sim-propellers"]; blades = c["blade_count"];
  omega = 2. Pi c["rpm"]/60.;
  tipMach = Abs[omega] c["tip_radius_m"]/c["sound_speed_m_s"];
  loadingForce = Sqrt[c["thrust_n"]^2 +
    (c["torque_nm"]/c["tip_radius_m"])^2];
  base = loadingForce/(4. Pi c["sound_speed_m_s"]);
  count = Length[c["actual_frequencies_hz"]];
  indices = Range[count];
  frequencies = Table[n blades Abs[c["rpm"]]/60., {n, count}];
  pressures = Table[
    base (n blades) Abs[omega] Abs[BesselJ[n blades, n blades tipMach]],
    {n, count}
  ];
  band = c["a_weighting_band_hz"];
  (* The Rust source spectrum reports positive peak-pressure magnitudes, so
     the physical time-mean-square band pressure is p_peak^2/2. *)
  weightedPower = Total[MapThread[
    If[band[[1]] <= #1 <= band[[2]], (#2 aWeighting[#1])^2/2., 0.] &,
    {frequencies, pressures}
  ]];
  scenario[<|
    "harmonic_indices" -> accuracyCheck[
      c["actual_harmonic_indices"], indices, 0., 0.],
    "harmonic_frequencies" -> accuracyCheck[
      c["actual_frequencies_hz"], frequencies, 10^-11, 10^-12],
    "gutin_loading_pressures" -> accuracyCheck[
      c["actual_base_pressures_pa"], pressures, 10^-14, 2. 10^-10],
    "conventional_pressure_handoff" -> accuracyCheck[
      c["actual_pressures_pa"], pressures, 10^-14, 2. 10^-10],
    "smooth_blade_unity_reduction" -> accuracyCheck[
      c["actual_amplitude_reductions"], ConstantArray[1., count], 10^-14, 10^-13],
    "linear_loading_scaling" -> accuracyCheck[
      c["actual_scaled_base_pressures_pa"], c["load_scale"] pressures,
      10^-14, 2. 10^-10],
    "a_weighted_band_power" -> accuracyCheck[
      c["actual_a_weighted_pressure_power"], weightedPower, 10^-20, 2. 10^-12]
  |>]
];
traceProgress["propeller"];

speakerReport = Module[{c, spline, values, derivatives, slope, intercept,
    affineValues, affineDerivatives},
  c = cases["sonium-sim-speaker"];
  spline = naturalSplineData[c["control_points"]];
  values = naturalSplineValue[spline, #] & /@ c["queries"];
  derivatives = naturalSplineDerivative[spline, #] & /@ c["queries"];
  slope = (c["affine_control_points"][[2, 2]] -
    c["affine_control_points"][[1, 2]])/
    (c["affine_control_points"][[2, 1]] - c["affine_control_points"][[1, 1]]);
  intercept = c["affine_control_points"][[1, 2]] -
    slope c["affine_control_points"][[1, 1]];
  affineValues = intercept + slope c["affine_queries"];
  affineDerivatives = ConstantArray[slope, Length[c["affine_queries"]]];
  scenario[<|
    "natural_cubic_values" -> accuracyCheck[
      c["actual_values"], values, 10^-13, 2. 10^-12],
    "natural_cubic_derivatives" -> accuracyCheck[
      c["actual_derivatives"], derivatives, 10^-13, 2. 10^-12],
    "control_point_interpolation" -> accuracyCheck[
      c["actual_values"][[{2, 4, 6, 7}]], c["control_points"][[All, 2]],
      10^-14, 10^-13],
    "affine_reproduction_values" -> accuracyCheck[
      c["actual_affine_values"], affineValues, 10^-13, 2. 10^-12],
    "affine_reproduction_derivatives" -> accuracyCheck[
      c["actual_affine_derivatives"], affineDerivatives, 10^-13, 2. 10^-12]
  |>]
];
traceProgress["speaker spline"];

speakerModelingReport = Module[{c, ka, directivity, k, area, velocity,
    farDirectivity, pressures, impedances, impedanceScale, lowKa,
    normalizedLowImpedance, spreadingRatios, spreadingReference},
  c = cases["sonium-sim-speaker-modeling"];
  ka = 2. Pi c["frequency_hz"] c["radius_m"]/c["sound_speed_m_s"];
  directivity = pistonDirectivity[ka, #] & /@ c["angles_deg"];
  k = 2. Pi c["frequency_hz"]/c["sound_speed_m_s"];
  area = Pi c["radius_m"]^2; velocity = toComplex[c["velocity_rms"]];
  farDirectivity = pistonDirectivity[ka, c["far_field_angle_deg"]];
  pressures = (I c["density_kg_m3"] c["sound_speed_m_s"] k area velocity
    farDirectivity Exp[-I k #]/(2. Pi #)) & /@ c["distances_m"];
  impedances = mechanicalPistonImpedance[c["radius_m"], #,
    c["density_kg_m3"], c["sound_speed_m_s"]] & /@
    c["impedance_frequencies_hz"];
  impedanceScale = c["density_kg_m3"] c["sound_speed_m_s"] area;
  lowKa = 2. Pi First[c["impedance_frequencies_hz"]] c["radius_m"]/
    c["sound_speed_m_s"];
  normalizedLowImpedance = toComplex[First[c["actual_mechanical_radiation_impedances"]]]/
    impedanceScale;
  spreadingRatios = Table[
    toComplex[c["actual_far_field_pressures"][[j + 1]]]/
      toComplex[c["actual_far_field_pressures"][[j]]],
    {j, Length[c["distances_m"]] - 1}
  ];
  spreadingReference = Table[
    c["distances_m"][[j]]/c["distances_m"][[j + 1]]
      Exp[-I k (c["distances_m"][[j + 1]] - c["distances_m"][[j]])],
    {j, Length[c["distances_m"]] - 1}
  ];
  scenario[<|
    "baffled_piston_directivity" -> accuracyCheck[
      c["actual_directivity"], directivity, 10^-13, 5. 10^-12],
    "baffled_piston_far_field" -> accuracyCheck[
      toComplex /@ c["actual_far_field_pressures"], pressures, 10^-13, 5. 10^-12],
    "far_field_spreading_and_phase" -> accuracyCheck[
      spreadingRatios, spreadingReference, 10^-12, 8. 10^-12],
    "full_mechanical_radiation_impedance" -> accuracyCheck[
      toComplex /@ c["actual_mechanical_radiation_impedances"], impedances,
      10^-11, 2. 10^-8],
    "low_ka_resistance_asymptote" -> accuracyCheck[
      Re[normalizedLowImpedance]/(lowKa^2/2.), 1., 10^-7, 0.],
    "low_ka_reactance_asymptote" -> accuracyCheck[
      Im[normalizedLowImpedance]/(8. lowKa/(3. Pi)), 1., 10^-7, 0.]
  |>]
];
traceProgress["speaker modeling"];

structuralReport = Module[{c, secondMoment, reference, xs, nodalReference,
    reactionForce, reactionMoment, strainEnergy},
  c = cases["sonium-sim-structural-mechanics"];
  secondMoment = c["width_m"] c["height_m"]^3/12.;
  reference = c["tip_force_n"] c["length_m"]^3/
    (3. c["youngs_modulus_pa"] secondMoment);
  xs = Range[0, c["elements"]] c["length_m"]/c["elements"];
  nodalReference = ({0.,
      c["tip_force_n"] #^2 (3. c["length_m"] - #)/
        (6. c["youngs_modulus_pa"] secondMoment),
      c["tip_force_n"] # (2. c["length_m"] - #)/
        (2. c["youngs_modulus_pa"] secondMoment)} &) /@ xs;
  reactionForce = {0., -c["tip_force_n"]};
  reactionMoment = -c["tip_force_n"] c["length_m"];
  strainEnergy = c["tip_force_n"]^2 c["length_m"]^3/
    (6. c["youngs_modulus_pa"] secondMoment);
  scenario[<|
    "euler_bernoulli_tip_deflection" -> accuracyCheck[
      c["actual_tip_deflection_m"], reference, 10^-13, 10^-8],
    "linear_solver_residual" -> accuracyCheck[
      c["actual_solver_residual"], 0., 10^-9, 0.],
    "global_force_balance" -> accuracyCheck[
      c["actual_force_balance_residual_n"], 0., 10^-8, 0.],
    "free_dof_equilibrium" -> accuracyCheck[
      c["actual_free_equilibrium_residual_n"], 0., 10^-8, 0.],
    "discrete_energy_balance" -> accuracyCheck[
      c["actual_energy_balance_relative_error"], 0., 10^-10, 0.],
    "nodal_euler_bernoulli_field" -> accuracyCheck[
      c["actual_nodal_displacement"], nodalReference, 10^-11, 5. 10^-8],
    "applied_load_vector" -> accuracyCheck[
      c["actual_applied_force_n"], {0., c["tip_force_n"]}, 10^-12, 10^-13],
    "clamp_reaction_force" -> accuracyCheck[
      c["actual_reaction_force_n"], reactionForce, 10^-8, 10^-11],
    "clamp_reaction_moment" -> accuracyCheck[
      c["actual_reaction_moment_nm"], reactionMoment, 10^-8, 10^-11],
    "strain_energy_clapeyron" -> accuracyCheck[
      c["actual_strain_energy_j"], strainEnergy, 10^-11, 5. 10^-8],
    "maximum_displacement_at_tip" -> accuracyCheck[
      c["actual_max_displacement_m"], Abs[reference], 10^-11, 5. 10^-8]
  |>]
];
traceProgress["structural mechanics"];

reports = <|
  "sonium-sim-pcfan" -> pcFanReport,
  "sonium-sim-propellers" -> propellerReport,
  "sonium-sim-room" -> roomReport,
  "sonium-sim-speaker" -> speakerReport,
  "sonium-sim-speaker-modeling" -> speakerModelingReport,
  "sonium-sim-structural-mechanics" -> structuralReport
|>;
payload = <|
  "schema_version" -> 1,
  "passed" -> And @@ Lookup[Values[reports], "passed"],
  "crates" -> reports
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
