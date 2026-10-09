(* Stationary circular-pipe member of Wolfram's fluid verification family:
   https://reference.wolfram.com/language/PDEModels/tutorial/FluidDynamics/FluidDynamicsVerificationTests.html?view=all
*)
Off[General::munfl];

pipeRadius = 0.05;
dynamicViscosity = 0.001;
pressureGradient = -2.;
sampleRadii = {0., 0.01, 0.02, 0.03, 0.04, 0.05};

velocity = DSolveValue[
  {
    dynamicViscosity/r D[r D[u[r], r], r] == pressureGradient,
    u[pipeRadius] == 0,
    u'[0] == 0
  },
  u,
  r
];

payload = <|
  "schema_version" -> 1,
  "pipe_radius_m" -> pipeRadius,
  "dynamic_viscosity_pa_s" -> dynamicViscosity,
  "pressure_gradient_pa_m" -> pressureGradient,
  "sample_radii_m" -> sampleRadii,
  "axial_velocity_m_s" -> (N[velocity[#], 16] & /@ sampleRadii)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
