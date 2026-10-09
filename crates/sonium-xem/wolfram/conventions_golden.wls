(* Wolfram convention golden for sonium-xem.
 *
 * Emits SPL, angular-frequency and signed-wavenumber references as one
 * compact RawJSON document on stdout; redirect into
 *   crates/sonium-xem/fixtures/wolfram-golden-v1.json
 * (create the fixtures directory first). Run from the repository root:
 *   wolframscript -file crates/sonium-xem/wolfram/conventions_golden.wls \
 *     > crates/sonium-xem/fixtures/wolfram-golden-v1.json
 *
 * Independent oracle per audit-sonium-xem-20260921: unit/phasor convention
 * fixtures with independent signs. SPL uses peak amplitudes referenced to
 * RMS (divide by Sqrt[2]); wavenumbers carry the convention spatial sign.
 *)
Off[General::munfl];

$MaxExtraPrecision = 500;
prec = 50;
num[value_] := ToString[CForm[N[value, prec]]];

pRef = 20*^-6;

(* peak magnitude -> SPL re RMS convention *)
spl[peak_] := 20*Log10[(peak/Sqrt[2])/pRef];

payload = <|
  "schema" -> "sonium-xem.wolfram-golden.v1",
  "precision_digits" -> prec,
  "generator" -> StringJoin["Wolfram Engine ", $Version, " convention golden"],
  "conventions" -> <|
    "spl" -> "peak amplitude, RMS-referenced: 20 log10((|P|/Sqrt[2])/20uPa",
    "wavenumber" -> "k = spatial_sign * 2 Pi f / c; spatial_sign = +1 for ExpMinusIwt, -1 for ExpPlusIwt"
  |>,
  "spl" -> {
    <|"peak" -> num[Sqrt[2]], "spl_db" -> num[spl[Sqrt[2]]]|>,
    <|"peak" -> num[2*Sqrt[2]], "spl_db" -> num[spl[2*Sqrt[2]]]|>,
    <|"peak" -> num[1/2], "spl_db" -> num[spl[1/2]]|>
  },
  "angular" -> {
    <|"freq_hz" -> num[1000], "omega" -> num[2*Pi*1000]|>,
    <|"freq_hz" -> num[440], "omega" -> num[2*Pi*440]|>
  },
  "wavenumber" -> {
    <|"freq_hz" -> num[1000], "c" -> num[343],
      "k_minus" -> num[2*Pi*1000/343], "k_plus" -> num[-2*Pi*1000/343]|>,
    <|"freq_hz" -> num[440], "c" -> num[343],
      "k_minus" -> num[2*Pi*440/343], "k_plus" -> num[-2*Pi*440/343]|>,
    <|"freq_hz" -> num[1/10], "c" -> num[1500],
      "k_minus" -> num[2*Pi*(1/10)/1500], "k_plus" -> num[-2*Pi*(1/10)/1500]|>
  }
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
