(* Derived speaker/BEM adaptation, not a literal Wolfram PDEModels geometry.
   Wolfram's Motor Noise example prescribes a harmonic surface velocity on a
   radiating solid.  This redistributable reference specializes that boundary
   pattern to the classical infinite-baffle circular piston. *)

rho = 1.21;
c = 343.;
radius = 0.065;
velocityPeak = 0.01;
ka = 2.;
anglesDeg = {0., 30., 60.};
axialRadii = {2., 4.};

directivity[angleDeg_] := Module[{argument = ka Sin[angleDeg Degree]},
  If[Abs[argument] < 10^-12, 1., 2 BesselJ[1, argument]/argument]
];
pressurePeak[distance_] := rho c ka radius velocityPeak/(2 distance);

payload = <|
  "schema_version" -> 1,
  "provenance" -> "derived adaptation of Wolfram Motor Noise prescribed-velocity boundary; not literal PDEModels geometry",
  "ka" -> ka,
  "angles_deg" -> anglesDeg,
  "normalized_directivity" -> (N[directivity[#], 16] & /@ anglesDeg),
  "axial_radii_m" -> axialRadii,
  "axial_pressure_peak_pa" -> (N[pressurePeak[#], 16] & /@ axialRadii)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
