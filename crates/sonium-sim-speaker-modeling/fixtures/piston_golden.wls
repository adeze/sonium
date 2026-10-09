(* Wolfram goldens for the baffled-piston radiation impedance
   (sonium-sim-speaker-modeling).
   Run: wolframscript -file crates/sonium-sim-speaker-modeling/fixtures/piston_golden.wls
   Output: crates/sonium-sim-speaker-modeling/fixtures/wolfram-golden-v1.json
   R(ka) = 1 - J1(2 ka)/ka and X(ka) = H1(2 ka)/ka use Wolfram's independent
   BesselJ/StruveH against the crate's numerical Struve quadrature (plus the
   small-ka Taylor branch). Frequencies are carried as fixture inputs so the
   replay forms ka = 2 Pi f a/c from identical decimals. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-sim-speaker-modeling", "fixtures",
    "wolfram-golden-v1.json"}];

(* Plain decimals: strip the MEPREC mark and rewrite WL `*^` exponents
   as `e` so downstream harnesses parse with a plain decimal reader. *)
dec[x_] := StringReplace[
  First[StringSplit[ToString[N[x, 17], InputForm], "`"]], "*^" -> "e"];

cAir = 343.0; rhoAir = 1.204; radius = 0.1;
r[ka_] := 1 - BesselJ[1, 2*ka]/ka;
x[ka_] := StruveH[1, 2*ka]/ka;
freqFor[ka_] := ka*cAir/(2*Pi*radius);

kas = {1.*^-4, 0.5, 1.0, 2.0, 5.0};
entries = Flatten[Table[
    {"f_ka_" <> tag -> <|"value" -> dec[freqFor[ka]], "exact" -> "input frequency"|>,
     "R_ka_" <> tag -> <|"value" -> dec[r[ka]],
       "exact" -> "1 - BesselJ[1, 2 ka]/ka"|>,
     "X_ka_" <> tag -> <|"value" -> dec[x[ka]],
       "exact" -> "StruveH[1, 2 ka]/ka"|>},
    {ka, kas}, {tag, {ToString[ka]}}]];

(* full mechanical impedance at ka = 1: Z = rho c pi a^2 (R + I X) *)
zScale = rhoAir*cAir*Pi*radius^2;
zFull = zScale*(r[1.0] + I*x[1.0]);

(* ---- directivity + far field at the ka = 1 frequency ----
   ka is re-formed from the emitted frequency decimal exactly as the
   replay does (ka = 2 Pi f a/c), so no hidden exact-1.0 assumption. *)
f1 = N[freqFor[1.0], 17];
ka1 = 2*Pi*f1*radius/cAir;
dirFor[thDeg_] := If[thDeg == 0.,
  1., 2*BesselJ[1, ka1*Sin[thDeg*Degree]]/(ka1*Sin[thDeg*Degree])];
dirAngles = {0., 30., 60., 90.};
k1 = 2*Pi*f1/cAir;
area = Pi*radius^2;
vRms = 0.03 - 0.01*I;
rDist = 2.3;
thF = 37*Degree;
dF = 2*BesselJ[1, ka1*Sin[thF]]/(ka1*Sin[thF]);
pF = I*rhoAir*cAir*k1*area*vRms*dF*Exp[-I*k1*rDist]/(2*Pi*rDist);

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "values" -> Association[Join[entries, {
    "Z_full_re" -> <|"value" -> dec[Re[zFull]],
      "exact" -> "rho c pi a^2 R(1), a = 0.1 m, f = c/(2 pi a)"|>,
    "Z_full_im" -> <|"value" -> dec[Im[zFull]],
      "exact" -> "rho c pi a^2 X(1), a = 0.1 m, f = c/(2 pi a)"|>,
    "dir_0" -> <|"value" -> dec[dirFor[0.]],
      "exact" -> "on-axis limit 1"|>,
    "dir_30" -> <|"value" -> dec[dirFor[30.]],
      "exact" -> "2 J1(ka sin30)/(ka sin30)"|>,
    "dir_60" -> <|"value" -> dec[dirFor[60.]],
      "exact" -> "2 J1(ka sin60)/(ka sin60)"|>,
    "dir_90" -> <|"value" -> dec[dirFor[90.]],
      "exact" -> "2 J1(ka)/(ka)"|>,
    "ff_re" -> <|"value" -> dec[Re[pF]],
      "exact" -> "Re[i rho c k A v D e^(-i k r)/(2 pi r)], v = 0.03-0.01i, r = 2.3 m, th = 37 deg"|>,
    "ff_im" -> <|"value" -> dec[Im[pF]],
      "exact" -> "Im[i rho c k A v D e^(-i k r)/(2 pi r)], v = 0.03-0.01i, r = 2.3 m, th = 37 deg"|>}]]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];