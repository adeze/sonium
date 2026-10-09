(* Exact uniaxial material-point reduction of Wolfram's linear-hardening
   Plasticity example. The material, dimensions, maximum displacement, and
   20 displacement multipliers are copied from the published page. *)
youngModulus = 70. 10^9;
yieldStress = 250. 10^7;
hardeningModulus = 150. 10^8;
length = 0.04;
maximumDisplacement = 3.9 10^-3;
multipliers = N[Table[1 - Abs[k - 1], {k, 0.1, 2, 1/10}], 16];
totalStrain = N[multipliers maximumDisplacement/length, 16];

stress = 0.;
accumulatedPlasticStrain = 0.;
previousStrain = 0.;
stressHistory = {};
plasticStrainHistory = {};
Do[
  strainIncrement = strain - previousStrain;
  trialStress = stress + youngModulus strainIncrement;
  currentYield = yieldStress + hardeningModulus accumulatedPlasticStrain;
  excess = Abs[trialStress] - currentYield;
  If[excess <= 0.,
    stress = trialStress,
    plasticIncrement = excess/(youngModulus + hardeningModulus);
    stress = trialStress - youngModulus plasticIncrement Sign[trialStress];
    accumulatedPlasticStrain += plasticIncrement
  ];
  AppendTo[stressHistory, stress];
  AppendTo[plasticStrainHistory, accumulatedPlasticStrain];
  previousStrain = strain,
  {strain, totalStrain}
];

payload = <|
  "schema_version" -> 1,
  "multipliers" -> multipliers,
  "total_strain" -> totalStrain,
  "stress_pa" -> N[stressHistory, 16],
  "accumulated_plastic_strain" -> N[plasticStrainHistory, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
