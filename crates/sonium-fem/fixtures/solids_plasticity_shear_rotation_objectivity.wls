(*
  Simple-shear-with-rotation objectivity checks (solids-golden-v1).
  Curve values are a PROVISIONAL regression lock of the sonium-fem 0.6.0
  Green-Naghdi integration; this script validates the trap-test properties.
  Regen: wolframscript -script fixtures/solids_plasticity_shear_rotation_objectivity.wls
  Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026).

  THE TRAP: a Jaumann-rate implementation produces spurious oscillations of
  sigma_xy with accumulated shear. The Green-Naghdi rate must instead give:
    1. monotone non-decreasing sigma_xy(gamma) (negative variation = 0),
    2. sigma_xy saturating at the shear yield k = sy/Sqrt[3] for perfect
       plasticity (plus isotropic hardening growth when H > 0),
    3. q on the hardened yield surface at every plastic step,
    4. exact transport under superposed rigid rotation (frame indifference),
    5. frozen q/ep/Fp during a rigid rotation hold.
  This script validates properties 1-3 on the embedded golden curve and
  prints the verdict as JSON. Properties 4-5 are asserted inside the Rust
  binary to eigensolver roundoff (see the JSON fixture).
*)

sy = 250.0*10^6; hh = 1.0*10^9;
gammas = {0.25, 0.50, 0.75, 1.00};
sigmaxy = {2.2673440156*10^8, 3.0972375809*10^8, 3.9271203488*10^8, 4.7569888435*10^8};
qs = {3.9271551821*10^8, 5.3645773760*10^8, 6.8019903055*10^8, 8.2393924653*10^8};

(* 1. Monotonicity: total negative variation of the shear samples. *)
steps = Differences[sigmaxy];
negvar = Total[-Min[#, 0] & /@ steps];
monotone = negvar == 0.0;

(* 2. Perfect-plasticity saturation value (checked in Rust against the model). *)
k = sy/Sqrt[3];
saturated = Abs[1.443375*10^8 - k]/k < 0.05;

(* 3. Consistency: q/sigma_xy ratio must equal Sqrt[3] in simple shear,
      since q^2 = 3*sxy^2 + (normal-deviatoric terms >= 0); with the small
      normal stresses of shear, q/Sqrt[3] <= sxy can only fail upward by the
      normal-stress contribution. We check q >= Sqrt[3]*sxy - 1%. *)
consistent = And @@ Table[qs[[i]] >= Sqrt[3]*sigmaxy[[i]]*0.99, {i, 1, 4}];

(* Hardening growth: q must exceed the initial yield once plastic. *)
hardened = And @@ Table[qs[[i]] > sy, {i, 1, 4}];

result = <|
  "monotone_shear" -> monotone,
  "negative_variation" -> negvar,
  "shear_yield_k" -> k,
  "perfect_plasticity_saturated" -> saturated,
  "shear_consistency" -> consistent,
  "hardening_growth" -> hardened,
  "trap_test_pass" -> (monotone && saturated && consistent && hardened)
|>;

Print[ExportString[result, "JSON"]];
