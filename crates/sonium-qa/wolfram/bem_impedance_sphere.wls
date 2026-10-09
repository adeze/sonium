(* Independent Wolfram series for a derived 3D impedance-sphere BEM check.
   Wolfram's PDEModels catalog has no literal 3D BEM sphere example; this
   uses the same plane-wave acoustics convention as the rigid-sphere adapter. *)

radius = 1.;
k = 1.;
ka = k radius;
normalizedAdmittance = 1.;
receiverRadius = 2.;
receiverAngles = {0., Pi/4., Pi/2., 3. Pi/4., Pi};
termCount = 30;

jDerivative[n_] := Derivative[1][SphericalBesselJ[n, #] &][ka];
hDerivative[n_] := Derivative[1][SphericalHankelH1[n, #] &][ka];
(* The physical condition is ∂r p - i k β p = 0. These derivatives are
   with respect to kr, so the common factor k is cancelled in the coefficient. *)
coefficient[n_] :=
  (jDerivative[n] - I normalizedAdmittance SphericalBesselJ[n, ka])/
  (hDerivative[n] - I normalizedAdmittance SphericalHankelH1[n, ka]);
pressure[theta_] := Sum[
  (2 n + 1) I^n (
    SphericalBesselJ[n, k receiverRadius] -
    coefficient[n] SphericalHankelH1[n, k receiverRadius]
  ) LegendreP[n, Cos[theta]],
  {n, 0, termCount - 1}
];
softCoefficient[n_] :=
  SphericalBesselJ[n, ka]/SphericalHankelH1[n, ka];
softPressure[theta_] := Sum[
  (2 n + 1) I^n (
    SphericalBesselJ[n, k receiverRadius] -
    softCoefficient[n] SphericalHankelH1[n, k receiverRadius]
  ) LegendreP[n, Cos[theta]],
  {n, 0, termCount - 1}
];
complexPair[value_] := {Re[N[value, 16]], Im[N[value, 16]]};

payload = <|
  "schema_version" -> 1,
  "provenance" -> "derived Wolfram impedance-sphere partial-wave series; not a literal PDEModels geometry",
  "radius_m" -> radius,
  "ka" -> ka,
  "normalized_admittance" -> complexPair[normalizedAdmittance],
  "receiver_radius_m" -> receiverRadius,
  "receiver_angles_rad" -> N[receiverAngles, 16],
  "total_pressure" -> (complexPair /@ (pressure /@ receiverAngles)),
  "soft_total_pressure" -> (complexPair /@ (softPressure /@ receiverAngles))
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
