# OpenCourant golden schema (`opencourant-golden-v1`)

One JSON per miniqa deck at `goldens/miniqa/<case>.json`, where `<case>`
is the miniqa-relative case dir (e.g. `LOIS/LOI2/BESH/input_44.json`).
The shape extends the `decks/*/probes.json` convention (`case`, engine
stamp, expected values + tolerances) with full time series, deck hashes,
and run provenance. Only harvested numeric facts enter the repo: no
vendor decks, `T01` binaries, `.abf` blobs, or sources.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema` | string | Always `opencourant-golden-v1`; loaders reject anything else. |
| `case` | string | miniqa-relative case dir (`LOIS/LOI2/BESH/input_44`). |
| `meta` | object | Provenance (below). |
| `time` | number[] | Shared time grid in deck time units, strictly the `th_to_csv` time column. |
| `channels` | object | Channel key → `{values, final, min, max, abs_tol, rel_tol}`. |
| `cycle` | object | Last explicit-cycle scalars from the engine `.out` (`CYCLE`, `TIME`, `TSTEP`, `ELTID`, `ERROR`, `IENERGY`, `KENERGYT`, `KENERGYR`, `EXTWORK`, `MASERR`, `TOTALMASS`, `ADDEDMASS`), keyed by the vendor names. |

Every `channels[k].values` has exactly `time.length` samples.

## `meta`

| Key | Type | Meaning |
|---|---|---|
| `engine` | string | Human stamp: `<product> <arch> CommitID <short8>`, e.g. `OpenRadioss linux64_gf CommitID 33e68517`. Product and commit parse from the starter `.out` banner. |
| `engine_commit` | string | Full CommitID hex from the `.out` banner. |
| `starter_exe`, `engine_exe` | string | Executable file names that ran. |
| `deck_sha256.starter` | string | sha256 hex of the `_0000.rad` starter deck. |
| `deck_sha256.engine` | string/null | sha256 hex of the concatenated `_000N.rad` engine decks in run order. |
| `starter_deck` | string | Starter deck file name inside the miniqa `data/` dir. |
| `engine_decks` | string[] | Engine deck file names in run order. |
| `starter_options` | string | Exact starter argv after the exe (`-i <deck> -np <np> [...]`). |
| `engine_options` | string | Exact engine argv per run, joined with ` ; `. |
| `mpi`, `np`, `threads`, `precision` | string/int/int/string | `smp`, starter `-np`, `OMP_NUM_THREADS`, `dp`. |
| `harvest_date` | string | RFC 3339 UTC harvest timestamp. |
| `deck_units` | [string × 3]/null | Unit system from `/BEGIN` (mass, length, time), e.g. `["Mg","mm","s"]`; null when unparseable. |
| `rootname` | string | Deck rootname (`BESHV44`). |
| `normal_termination` | bool | `NORMAL TERMINATION` present in the engine `.out` (harvest requires true). |
| `th_tool` | string | `th_to_csv` binary file name used for conversion. |
| `exe_sha256` | object | sha256 hex of the starter, engine, and `th_to_csv` binaries. |
| `decimation` | object/null | `{stride, kept, of}` when `--max-samples` decimated the grid; null for full series. |
| `local_patches` | string/absent | Local vendor-tree deviations the deck carries. Absent = ran as shipped. Set explicitly via `--local-patches`; each note points at `miniqa/PATCHES.md`, which records the exact diffs. Regen via `qa-oc-replay` inherits the previous golden's note. |
| `th_note` | string/absent | Why this golden carries no TH grid. Absent = full TH golden. Present = cycle-only golden (converter could not produce a CSV); `time` is `[]`, `channels` is `{}`, and `cycle` still holds the last-cycle scalars. Old readers ignore it; loaders reject an empty grid without it. |

## Channel keys

Keys are whitespace-collapsed `th_to_csv` titles (every whitespace run
→ one space, quotes stripped, trimmed) with the trailing `var NN`
suffix removed: `th_to_csv` numbers TH-group variables file-wide and
the numbering shifts between engine versions, so the suffix carries no
identity and must not enter keys. Pairing is by title, never by column
position. Titles that repeat within a harvest (TH groups with several
variables, duplicated part groups) gain a `~k` ordinal in 1-based file
order (`th_nodes 4 ~1`, `th_nodes 4 ~2`, …); unique titles keep their
bare name. A title whose arity changes between harvests renames, which
is a deliberate fail-closed regen event. No `VAR/id`-style renaming is
applied: the physical meaning of a `~k` ordinal lives in deck TH-card
order (a deck-reader concern, Phase 4). Columns titled `(null)`
(unresolvable TH requests, always empty) are dropped from both header
and rows before any other check; short rows are only tolerated over
dropped `(null)` columns.

## Cycle-only goldens

When `th_to_csv` crashes or yields no usable CSV for a deck whose engine
run terminated normally (observed on degenerate T01 files), the harvest
still writes a golden with `time: []`, `channels: {}`, the `cycle`
scalars, and `meta.th_note` explaining why. Replay of such a golden
checks load + deck hashes and reports the cycle-scalar count; `--series`
replay is rejected (no grid to interpolate onto).

## Decimation

`qa-oc-harvest --max-samples N` keeps about N stamps: a uniform stride
over the harvest grid plus every channel's argmin/argmax plus the
endpoints, so no channel's extrema are lost (`decimation: null` means
full series). Kept count is at most `N + 2 * channels + 1`.

## Tolerances

Each channel stores `abs_tol` and `rel_tol`. Harvest sets `rel_tol` to
the vendor `Diff_tolerance` (`6.6e-4`) and `abs_tol` to
`1e-12 * max(1, peak)`. Replay interpolates each provided channel
linearly onto the golden stamps (the provided grid must cover them and
be monotone) and passes a sample when
`|actual - expected| <= abs + rel * |expected|`, with the vendor
near-zero skip (`Zero_tolerance 5e-14`): samples where both sides sit
below it, or differ by less than twice it, are noise. Cycle scalars
compare at the default relative tolerance. Extra provided channels are
ignored; missing channels, uncovered stamps, and non-finite samples
fail.

## Regen

```
REGEN_OC_GOLDENS=1 cargo run --release -p sonium-qa --features qa-binaries --bin qa-oc-harvest -- --deck <case> --max-samples 2000
```

needs `--miniqa`/`OPENCOURANT_MINIQA` and `--exe`/`RADISS_EXE`/
`OPENCOURANT_EXE`. Every stale/missing-golden error names this command
with the failing case filled in. `qa-oc-replay --deck <case>` replays
offline (self-test by default, `--series <csv>` for provided series).

Run scratch resolves as `--scratch-root`, then `SONIUM_QA_SCRATCH`,
then the platform temp dir (which honors `TMPDIR` on Unix); per-deck
scratch is removed on success. This checkout runs scratch under
`<checkout>/tmp` (see root `.gitignore`), never `/tmp`.

## Size policy

Canonical corpus policy (uniform, no per-deck config): every deck is
harvested with `--max-samples 2000`. Decks under the budget keep full
series with no decimation record; the 6 biggest decks (2.4k–12.5k
samples) keep stride ≤ 7 plus per-channel extrema and endpoints.
Rationale (Phase 0 gate measurement): full series total 61.8MB for 78
goldens; `th_to_csv` already emits 7-significant-digit values so
precision rounding saves nothing; uniform `--max-samples 2000` lands
at ~41MB. Verified guarantees (gate fidelity proof): kept stamps are
exact engine values (a strict subgrid subset), endpoints and
per-channel global min/max are preserved exactly, and consecutive
kept stamps are at most one nominal stride apart (≤ 7 on the 6
affected decks; the other 72 keep full series). Replay queries kept
stamps only, so no interpolation error enters replay from decimation;
the accepted cost is coarser temporal checkpoint density on 6 decks.
Dropped stamps do NOT interpolate within replay tolerance in general
(high-frequency content vs the tight 6.6e-4 band), which is why the
guarantee is subset-exactness, not interp-accuracy. Corpus guard:
50MB checked in; the miniqa set is closed (81 decks), so no growth is
expected. The harvester default stays 0 (full series); the policy
lives in `regen_command` and this doc.
