(* Published Wolfram verification case
   SolidMechanics-FEM-Stationary-2D-PlaneStress-0002:
   https://reference.wolfram.com/language/PDEModels/tutorial/StructuralMechanics/SolidMechanicsVerificationTests.html?view=all

   The official acceptance quantity is the Kirsch sigma_xx value at the crown
   of a quarter-hole. This script evaluates the same analytical stress
   expression printed by the Wolfram example.
*)
Off[General::munfl];

width = 1.;
height = 1.;
thickness = 0.1;
radius = 0.1;
youngModulus = 200. * 10^9;
poissonRatio = 0.3;
traction = 1000.;
theta = Pi/2;
sigmaX[r_] :=
  traction/2 (1 + radius^2/r^2) -
    traction/2 (1 + 3 radius^4/r^4) Cos[2 theta];

payload = <|
  "schema_version" -> 1,
  "case_id" -> "SolidMechanics-FEM-Stationary-2D-PlaneStress-0002",
  "dimensions_m" -> {width, height, thickness},
  "hole_radius_m" -> radius,
  "young_modulus_pa" -> youngModulus,
  "poisson_ratio" -> poissonRatio,
  "traction_pa" -> traction,
  "analytical_crown_stress_pa" -> N[sigmaX[radius], 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
