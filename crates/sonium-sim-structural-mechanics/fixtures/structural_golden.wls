(* Wolfram goldens for sonium-sim-structural-mechanics analytic references.
   Run: wolframscript -file crates/sonium-sim-structural-mechanics/fixtures/structural_golden.wls
   Output: crates/sonium-sim-structural-mechanics/fixtures/wolfram-golden-v1.json
   Euler column load P = Pi^2 E I/(K L)^2 for the introductory pinned steel
   column, and the Navier+Mindlin center deflection of the introductory
   simply-supported square plate under sinusoidal load. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-sim-structural-mechanics", "fixtures",
    "wolfram-golden-v1.json"}];

dec[x_] := StringReplace[
  First[StringSplit[ToString[N[x, 17], InputForm], "`"]], "*^" -> "e"];

(* introductory buckling column: L = 1 m, 0.02 x 0.02 section,
   E = 200 GPa, pinned-pinned K = 1 *)
colI = 0.02*0.02^3/12;
eulerLoad = Pi^2*200.*^9*colI/(1.0*1.0)^2;

(* introductory Mindlin plate: 1 x 1 x 0.1 m, E = 1 MPa, nu = 0.3,
   q0 = 1 MPa sinusoidal, shear correction 5/6 *)
plateA = 1.0; plateB = 1.0; plateT = 0.1;
plateE = 1.*^6; plateNu = 0.3; plateQ = 1.*^6; kappa = 5./6.;
plateLambda = (Pi/plateA)^2 + (Pi/plateB)^2;
plateD = plateE*plateT^3/(12*(1 - plateNu^2));
plateG = plateE/(2*(1 + plateNu));
plateS = kappa*plateG*plateT;
plateW = plateQ*(1/(plateD*plateLambda^2) + 1/(plateS*plateLambda));

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "values" -> Association[{
    "euler_pinned_load_n" -> <|"value" -> dec[eulerLoad],
      "exact" -> "Pi^2*200*10^9*(0.02*0.02^3/12)/1^2"|>,
    "plate_center_deflection_m" -> <|"value" -> dec[plateW],
      "exact" -> "q0*(1/(D*lambda^2) + 1/(S*lambda)), Navier+Mindlin"|>}]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];