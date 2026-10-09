(* Wolfram goldens for the compact FW-H single-element kernel
   (sonium-sim-propellers).
   Run: wolframscript -file crates/sonium-sim-propellers/fixtures/fwh_golden.wls
   Output: crates/sonium-sim-propellers/fixtures/wolfram-golden-v1.json
   One surface element at (3, 4, 0) seen from the origin (r = 5 m exactly):
   monopole = I w rho A vn Exp[-I k r]/(4 Pi r) and
   dipole = I w A cos-th p Exp[-I k r]/(4 Pi c r) with cos-th = -3/5.
   The replay pins the retarded phase sign, the 4 Pi normalization, the
   normal projection, and the peak-phasor multiply-by-I convention. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-sim-propellers", "fixtures",
    "wolfram-golden-v1.json"}];

dec[x_] := StringReplace[
  First[StringSplit[ToString[N[x, 17], InputForm], "`"]], "*^" -> "e"];

freq = 100.0; rho = 1.2; cAir = 343.0;
obs = {0., 0., 0.}; centroid = {3., 4., 0.};
path = obs - centroid;
r = Norm[path];
normal = {1., 0., 0.};
cosTh = normal . (path/r);
area = 0.5;
p = 2.0 + 0.0*I;
vn = 0.01 + 0.02*I;
om = 2*Pi*freq; k = om/cAir;
phase = Exp[-I*k*r];
monoFactor = om*rho*area/(4*Pi*r);
diFactor = om*area*cosTh/(4*Pi*cAir*r);
mono = I*vn*monoFactor*phase;
di = I*p*diFactor*phase;

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "values" -> Association[{
    "distance_m" -> <|"value" -> dec[r], "exact" -> "5"|>,
    "cos_theta" -> <|"value" -> dec[cosTh], "exact" -> "-3/5"|>,
    "mono_re" -> <|"value" -> dec[Re[mono]],
      "exact" -> "Re[I w rho A vn e^(-I k r)/(4 Pi r)]"|>,
    "mono_im" -> <|"value" -> dec[Im[mono]],
      "exact" -> "Im[I w rho A vn e^(-I k r)/(4 Pi r)]"|>,
    "di_re" -> <|"value" -> dec[Re[di]],
      "exact" -> "Re[I w A cos-th p e^(-I k r)/(4 Pi c r)]"|>,
    "di_im" -> <|"value" -> dec[Im[di]],
      "exact" -> "Im[I w A cos-th p e^(-I k r)/(4 Pi c r)]"|>}]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];