(* Wolfram goldens for sonium-wasm analytic room acoustics.
   Run: wolframscript -file crates/sonium-wasm/fixtures/room_modes_golden.wls
   Output: crates/sonium-wasm/fixtures/wolfram-golden-v1.json
   Three independent formulations of the rigid rectangular-room acoustics:
   eigenfrequencies f = c/2 Sqrt[(n/Lx)^2 + (m/Ly)^2 + (p/Lz)^2] with the
   axial/tangential/oblique classification, the driven damped cosine modal
   sum (rigid (0,0,0) mode kept, 4x band filter), and first-order image
   sources with wall-coefficient products. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-wasm", "fixtures", "wolfram-golden-v1.json"}];

dec[x_] := StringReplace[
  First[StringSplit[ToString[N[x, 17], InputForm], "`"]], "*^" -> "e"];
decList[l_] := Map[dec, l];

(* ---------------- eigenmode table ---------------- *)
lx = 4.0; ly = 3.0; lz = 2.5; cAir = 343.0; maxOrd = 2; maxF = 300.0;
modeF[n_, m_, p_] := (cAir/2)*Sqrt[(n/lx)^2 + (m/ly)^2 + (p/lz)^2];
rawModes = Flatten[Table[{n, m, p, modeF[n, m, p]},
  {n, 0, maxOrd}, {m, 0, maxOrd}, {p, 0, maxOrd}], 2];
keptModes = Select[rawModes, #[[4]] <= maxF && #[[1 ;; 3]] != {0, 0, 0} &];
sortedModes = SortBy[keptModes, #[[4]] &];
modeType[{n_, m_, p_}] := Module[{z = Count[{n, m, p}, 0]},
  Which[z == 2, "axial", z == 1, "tangential", True, "oblique"]];
modeEntries = Map[
  <|"n" -> ToString[#[[1]]], "m" -> ToString[#[[2]]],
    "p" -> ToString[#[[3]]], "f" -> dec[#[[4]]],
    "type" -> modeType[#[[1 ;; 3]]]|> &, sortedModes];

(* ---------------- driven modal sum ---------------- *)
wx = 5.0; wy = 4.0; wz = 3.0;
src = {1.0, 2.0, 1.0}; lis = {4.0, 1.0, 2.0};
freq = 100.0; zeta = 0.05;
vol = wx*wy*wz; om = 2*Pi*freq;
pref = cAir^2/vol;
modeTerm[n_, m_, p_] := Module[{wn, srcM, lisM, eps, den},
  wn = cAir*Pi*Sqrt[(n/wx)^2 + (m/wy)^2 + (p/wz)^2];
  If[{n, m, p} != {0, 0, 0} &&
    (wn/(2*Pi) > 4*freq || wn/(2*Pi) < freq/4), Return[0]];
  srcM = Cos[n*Pi*src[[1]]/wx]*Cos[m*Pi*src[[2]]/wy]*Cos[p*Pi*src[[3]]/wz];
  lisM = Cos[n*Pi*lis[[1]]/wx]*Cos[m*Pi*lis[[2]]/wy]*Cos[p*Pi*lis[[3]]/wz];
  eps = (If[# == 0, 1., 2.] & /@ {n, m, p});
  den = (wn^2 - om^2) - I*(2*zeta*wn*om);
  (Times @@ eps)*srcM*lisM/den];
modalP = pref*Sum[modeTerm[n, m, p], {n, 0, 2}, {m, 0, 2}, {p, 0, 2}];

(* ---------------- first-order image sources ---------------- *)
srcI = {1.0, 1.0, 1.0}; dims = {5.0, 4.0, 3.0};
refl = {0.5, 0.6, 0.7, 0.8, 0.9, 0.4};
oneAxis[s_, len_, lo_, up_] :=
  {{s, 1, 0}, {-s, lo, 1}, {2*len - s, up, 1}};
ax = oneAxis[srcI[[1]], dims[[1]], refl[[1]], refl[[2]]];
ay = oneAxis[srcI[[2]], dims[[2]], refl[[3]], refl[[4]]];
az = oneAxis[srcI[[3]], dims[[3]], refl[[5]], refl[[6]]];
combos = Select[Flatten[Table[
    {ix[[1]], iy[[1]], iz[[1]], ix[[2]]*iy[[2]]*iz[[2]],
     ix[[3]] + iy[[3]] + iz[[3]]},
    {ix, ax}, {iy, ay}, {iz, az}], 2], #[[5]] <= 1 &];
sortedCombos = SortBy[combos, {#[[5]] &, #[[1]] &, #[[2]] &, #[[3]] &}];
imageEntries = Map[
  <|"x" -> dec[#[[1]]], "y" -> dec[#[[2]]], "z" -> dec[#[[3]]],
    "coeff" -> dec[#[[4]]], "order" -> ToString[#[[5]]]|> &,
  sortedCombos];

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "modes" -> modeEntries,
  "values" -> Association[{
    "modal_re" -> <|"value" -> dec[Re[modalP]],
      "exact" -> "driven damped cosine modal sum, room 5x4x3, f = 100 Hz"|>,
    "modal_im" -> <|"value" -> dec[Im[modalP]],
      "exact" -> "driven damped cosine modal sum, room 5x4x3, f = 100 Hz"|>}],
  "images" -> imageEntries|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];