(* Independent Wolfram references for canonical 3D acoustic BEM checks.
   The rigid sphere is the spherical analogue of the bare sound-hard obstacle
   in Acoustic Cloak. The pulsating sphere follows the monopole derivation in
   AcousticsFrequencyDomain. *)

Off[General::munfl];
c = 343.;
rho = 1.21;
radius = 1.;

k = 1.;
ka = k radius;
receiverRadius = 2.;
receiverAngles = N[{0, Pi/4, Pi/2, 3 Pi/4, Pi}];
termCount = 30;
jDerivative[n_] :=
  Derivative[1][SphericalBesselJ[n, #] &][ka];
hDerivative[n_] :=
  Derivative[1][SphericalHankelH1[n, #] &][ka];
rigidCoefficient[n_] := jDerivative[n]/hDerivative[n];
rigidPressure[theta_] := Sum[
  (2 n + 1) I^n (
    SphericalBesselJ[n, k receiverRadius]
    - rigidCoefficient[n] SphericalHankelH1[n, k receiverRadius]
  ) LegendreP[n, Cos[theta]],
  {n, 0, termCount - 1}
];
rigidBackscatterAmplitude = I/k Sum[
  (2 n + 1) rigidCoefficient[n] (-1)^n,
  {n, 0, termCount - 1}
];
rigidBackscatterRcs = 4 Pi Abs[rigidBackscatterAmplitude]^2;

(* ka = Pi is the first interior Dirichlet eigenfrequency of the sphere and
   therefore an irregular frequency for the conventional exterior equation. *)
irregularKa = N[Pi, 16];
irregularK = irregularKa/radius;
irregularJDerivative[n_] :=
  Derivative[1][SphericalBesselJ[n, #] &][irregularKa];
irregularHDerivative[n_] :=
  Derivative[1][SphericalHankelH1[n, #] &][irregularKa];
irregularRigidCoefficient[n_] :=
  irregularJDerivative[n]/irregularHDerivative[n];
irregularRigidPressure[theta_] := Sum[
  (2 n + 1) I^n (
    SphericalBesselJ[n, irregularK receiverRadius]
    - irregularRigidCoefficient[n]
      SphericalHankelH1[n, irregularK receiverRadius]
  ) LegendreP[n, Cos[theta]],
  {n, 0, termCount - 1}
];

pulsatingFrequency = 100.;
pulsatingK = 2. Pi pulsatingFrequency/c;
velocity = 1.;
pulsatingHDerivative =
  Derivative[1][SphericalHankelH1[0, #] &][pulsatingK radius];
pulsatingAmplitude = I rho c velocity/pulsatingHDerivative;
pulsatingRadii = {2., 4., 8.};
pulsatingPressure[r_] :=
  pulsatingAmplitude SphericalHankelH1[0, pulsatingK r];
pulsatingKa = pulsatingK radius;
radiationResistance =
  rho c pulsatingKa^2/(1. + pulsatingKa^2);
pulsatingRadiatedPower =
  1./2. 4. Pi radius^2 radiationResistance Abs[velocity]^2;

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "rigid_ka" -> ka,
  "rigid_receiver_radius_m" -> receiverRadius,
  "rigid_receiver_angles_rad" -> receiverAngles,
  "rigid_total_pressure" -> (complexPair[rigidPressure[#]] & /@ receiverAngles),
  "rigid_backscatter_rcs_m2" -> N[rigidBackscatterRcs, 16],
  "rigid_irregular_ka" -> N[irregularKa, 16],
  "rigid_irregular_total_pressure" ->
    (complexPair[irregularRigidPressure[#]] & /@ receiverAngles),
  "pulsating_frequency_hz" -> pulsatingFrequency,
  "pulsating_receiver_radii_m" -> pulsatingRadii,
  "pulsating_pressure_pa" -> (complexPair[pulsatingPressure[#]] & /@ pulsatingRadii),
  "pulsating_radiated_power_w" -> N[pulsatingRadiatedPower, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
