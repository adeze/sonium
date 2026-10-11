(* sonium-qa fem case fem.airbag-isentrope.v1.
   Isentropic vent-orifice relations of the uniform-pressure airbag
   (Theory Manual equations 1115-1121): critical pressure ratio,
   exit velocity with shocked branch, exit density, mass and energy
   rates, isentropic temperature ratios, and mixture gamma. Two gases
   (textbook gamma = 1.4, deck gamma = 1.667) over subsonic and choked
   pressure ratios. Catches exponent, branch-selection, and
   gamma-averaging slips on the Rust side. *)

Off[General::munfl];

pCritRatio[g_] := (2/(g + 1))^(g/(g - 1));
exitVelocity[g_, p_, down_, rho_] := Module[{vent, ratio},
  vent = Max[down, p pCritRatio[g]];
  ratio = vent/p;
  Sqrt[2 g/(g - 1) p/rho (1 - ratio^((g - 1)/g))]
];
exitDensity[g_, p_, down_, rho_] := Module[{vent},
  vent = Max[down, p pCritRatio[g]];
  rho (vent/p)^(1/g)
];

gDeck = 1667/1000;
cpDeck = 520275/1000;
pChamber = 2/10^4;
rhoChamber = 16/10^10;
holeArea = 840;
inflowH = 30761/100 cpDeck;
ratios = {99/100, 9/10, 7/10, 3/5, 2/5, 1/10};

vents = Table[Module[{down, u, rhoE, mdot},
   down = pChamber r;
   u = exitVelocity[gDeck, pChamber, down, rhoChamber];
   rhoE = exitDensity[gDeck, pChamber, down, rhoChamber];
   mdot = rhoE holeArea u;
   <|
    "ratio" -> N[r, 50],
    "velocity" -> N[u, 50],
    "exit_density" -> N[rhoE, 50],
    "mass_rate" -> N[mdot, 50],
    "energy_rate" -> N[mdot inflowH, 50],
    "temp_ratio" -> N[(Max[down, pChamber pCritRatio[gDeck]]/pChamber)^((gDeck - 1)/gDeck), 50]
   |>
  ], {r, ratios}];

rRef = (cpDeck/gDeck) (gDeck - 1);
mixGamma = ((3/10) cpDeck + (7/10) cpDeck)/((3/10) (cpDeck - rRef) + (7/10) (cpDeck - rRef));

gAir = 14/10;
payload = <|
  "schema_version" -> 1,
  "case" -> "fem.airbag-isentrope.v1",
  "gamma_deck" -> N[gDeck, 50],
  "cp_deck" -> N[cpDeck, 50],
  "gas_constant" -> N[rRef, 50],
  "pressure" -> N[pChamber, 50],
  "density" -> N[rhoChamber, 50],
  "area" -> holeArea,
  "enthalpy" -> N[inflowH, 50],
  "critical_ratio_deck" -> N[pCritRatio[gDeck], 50],
  "critical_ratio_air" -> N[pCritRatio[gAir], 50],
  "air_velocity" -> N[exitVelocity[gAir, 2/10^4, 1/10^4, 12/10^10], 50],
  "mixture_gamma" -> N[mixGamma, 50],
  "vents" -> vents
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
