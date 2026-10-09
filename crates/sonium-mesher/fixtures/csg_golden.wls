(* Wolfram golden values for sonium-mesher + meshlang CSG volumes.
   Run: wolframscript -file crates/sonium-mesher/fixtures/csg_golden.wls
   Output: crates/sonium-mesher/fixtures/wolfram-golden-v1.json
   Exact polyhedral volumes plus the sphere volume N[4 Pi/3, 17]. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-mesher", "fixtures", "wolfram-golden-v1.json"}];

(* Plain decimals: strip the MEPREC `prec. precision mark so downstream
   harnesses can parse values with a plain decimal reader. *)
dec[x_] := First[StringSplit[ToString[N[x, 17], InputForm], "`"]];

exact[name_, val_] := name -> <|
  "value" -> dec[val],
  "exact" -> ToString[val, InputForm]|>;

(* Partial-overlap difference: Box(2,2,2) minus a unit box translated by
   (3/2, 0, 0); overlap 1/2 x 1 x 1. The translated placement is
   load-bearing: an identity translate would nest fully and give 7. *)
shiftedNotchVol = 2^3 - (1/2)*1*1;

(* Polynomial moments of the [0,2]x[0,3] rectangle: exact area and
   first/second moments for mesh-integral replay. *)
rect = {0, 2, 0, 3};
rectArea = (rect[[2]] - rect[[1]])*(rect[[4]] - rect[[3]]);
rectMx = ((rect[[2]]^2 - rect[[1]]^2)/2)*(rect[[4]] - rect[[3]]);
rectMy = (rect[[2]] - rect[[1]])*((rect[[4]]^2 - rect[[3]]^2)/2);
rectMxx = ((rect[[2]]^3 - rect[[1]]^3)/3)*(rect[[4]] - rect[[3]]);
rectMyy = (rect[[2]] - rect[[1]])*((rect[[4]]^3 - rect[[3]]^3)/3);

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "rect" -> <|
    "area" -> dec[rectArea], "mx" -> dec[rectMx], "my" -> dec[rectMy],
    "mxx" -> dec[rectMxx], "myy" -> dec[rectMyy]|>,
  "values" -> Association[{
    exact["box_2x3x5_volume", 2*3*5],
    exact["notched_cube_volume", 1^3 - (1/2)^3],
    exact["shifted_notch_volume", shiftedNotchVol],
    exact["unit_sphere_volume", (4/3)*Pi],
    exact["hull_unit_cube", 1],
    exact["hull_octahedron", 4/3],
    exact["hull_unit_simplex", 1/6]}]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];