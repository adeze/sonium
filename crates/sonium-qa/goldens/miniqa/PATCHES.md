# Local vendor-tree patches (miniqa goldens)

Three miniqa decks cannot run as shipped: their `qadiags.inc` include file
consists of 20 bare `Copyright>` license-header lines with no `#` comment
marker (the fork's header script stripped them), and each deck includes that
file while the starter is inside an open fixed-format data block
(`-- BLOCK: /SHELL/3` in LOI66). The starter dies with ~95 `ERROR ID
100101` input-format errors and `ERROR TERMINATION` (exit 2). Vendor CI
fails identically on these decks; this is a vendor-HEAD defect, not a
harvester defect.

With owner authorization (option b, 2026-10-10), the 3 local files were
patched by prefixing `#` to every bare `Copyright>` line, turning the
include into pure comments:

- `LOIS/LOI66/test_shell/data/qadiags.inc`
- `LOIS/LOI76/shell/test1/compression/data/qadiags.inc`
- `OPTIONS/desacti/test2/version_5/data/qadiags.inc`

Before md5 (all three, identical): `c3e6214f57ec7fd8ca33fb99a1f8a465`
After md5 (all three, identical): `7a481b641d9a5e57f89fc887b95d4fe5`
Originals backed up under `<checkout>/tmp/vendor_inc_backup/`.
Diff shape (all 20 lines, same in each file):

```diff
1,20c1,20
< Copyright>        OpenRadioss
< Copyright>        Copyright (C) 2026 Siemens
< Copyright>
< ...
---
> # Copyright>        OpenRadioss
> # Copyright>        Copyright (C) 2026 Siemens
> # Copyright>
> # ...
```

Physics neutrality is proven, not assumed: all three patched decks run to
`NORMAL TERMINATION` and their fresh last-cycle lines (cycle, time-step,
energies, totals) are digit-identical to the shipped reference `.out`
files. This is expected — the include carries no cards in any deck (all
`qadiags.inc` copies tree-wide are the identical header-only 20 lines;
the other 78 decks merely include it at keyword level, where the
starter tolerates the garbage with warnings).

Goldens harvested from patched decks carry
`meta.local_patches = "qadiags.inc: prefixed '#' to 20 bare Copyright>
header lines (vendor HEAD starter-fatal; see PATCHES.md)"`, set via
`qa-oc-harvest --local-patches`. The `.rad` decks themselves are
unmodified, so deck-hash verification is unaffected; reverting the
`.inc` fix fails closed (starter exit 2) on the next regen.
