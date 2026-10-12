#!/usr/bin/env python3
"""Publish qa-oc-bench bundles into perf/oc-explicit.md deck tables.

Usage: scripts/perf_oc_publish.py [bundles_dir] [markdown]

Reads sonium.oc-bench/v1 bundles, one row per cell, and rewrites the
per-deck results blocks of the markdown between
<!-- PERF-OC-DECK-BEGIN <deck> --> / <!-- PERF-OC-DECK-END -->
markers. Decks without bundles are left untouched; decks without a
marker block are appended at the end of the Results section (before
"## Reading"). Failing rows (correctness_passed=false) are never
published, per matrix policy.

The block format round-trips the hand-written Phase 6 tables except
row order, which is normalized to ascending ns/dof/cycle.
"""

import glob
import json
import os
import re
import sys

SCHEMA = "sonium.oc-bench/v1"
BEGIN = "<!-- PERF-OC-DECK-BEGIN {deck} -->"
END = "<!-- PERF-OC-DECK-END -->"
READING_HEADING = "## Reading"


def load_cells(bundles_dir):
    decks = {}
    skipped = 0
    for path in sorted(glob.glob(os.path.join(bundles_dir, "*.json"))):
        with open(path) as handle:
            try:
                bundle = json.load(handle)
            except json.JSONDecodeError as error:
                print(f"warn: {path} not JSON ({error}); skipped", file=sys.stderr)
                continue
        if bundle.get("schema") != SCHEMA:
            print(f"warn: {path} schema {bundle.get('schema')}; skipped", file=sys.stderr)
            continue
        protocol = bundle.get("protocol", {})
        for row in bundle.get("rows", []):
            if not row.get("correctness_passed", False):
                skipped += 1
                continue
            decks.setdefault(row["deck"], []).append((row, protocol))
    return decks, skipped


def deck_block(deck, cells):
    dofs = {row["dof"] for row, _ in cells}
    if len(dofs) != 1:
        print(f"warn: {deck} has mixed dof {sorted(dofs)}; using first", file=sys.stderr)
    dof = sorted(dofs)[0]
    cycles = {}
    for row, _ in cells:
        cycles.setdefault(row["side"], set()).add(row["cycles"])
    for side, values in cycles.items():
        if len(values) != 1:
            print(
                f"warn: {deck}/{side} has mixed cycles {sorted(values)}; using first",
                file=sys.stderr,
            )
    engine_cycles = sorted(cycles.get("engine", {0}))[0]
    sonium_cycles = sorted(cycles.get("sonium", {0}))[0]
    repeats = {protocol.get("repeats") for _, protocol in cells}
    header = f"### {deck} (dof {dof}; engine {engine_cycles} cycles, sonium {sonium_cycles} cycles"
    if repeats == {1}:
        header += "; 1 repeat"
    elif len(repeats) != 1:
        print(f"warn: {deck} has mixed protocol repeats {sorted(repeats)}", file=sys.stderr)
        header += "; mixed protocol"
    header += ")"
    ordered = sorted(cells, key=lambda cell: cell[0]["ns_per_dof_cycle"])
    best = min(row["ns_per_dof_cycle"] for row, _ in cells)
    lines = [
        BEGIN.format(deck=deck),
        header,
        "",
        "| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |",
        "| ---- | ------ | ------: | -----: | -----------: | -------: |",
    ]
    for row, _ in ordered:
        ns = row["ns_per_dof_cycle"]
        ns_text = f"{ns:.0f}"
        if ns == best:
            ns_text = f"**{ns_text}**"
        rss_mb = round(row["peak_rss_bytes"] / 1e6)
        lines.append(
            f"| {row['side']} | {row['flavor']} | {row['threads']} | "
            f"{row['wall_s']:.3g} | {ns_text} | {rss_mb} MB |"
        )
    lines.append(END)
    return "\n".join(lines) + "\n"


def main():
    bundles_dir = sys.argv[1] if len(sys.argv) > 1 else "perf/oc"
    markdown = sys.argv[2] if len(sys.argv) > 2 else "perf/oc-explicit.md"
    decks, skipped = load_cells(bundles_dir)
    if skipped:
        print(f"note: {skipped} failing rows unpublished by policy", file=sys.stderr)
    with open(markdown) as handle:
        text = handle.read()
    for deck in sorted(decks):
        block = deck_block(deck, decks[deck])
        pattern = re.compile(
            re.escape(BEGIN.format(deck=deck)) + r".*?" + re.escape(END) + r"\n?",
            re.DOTALL,
        )
        if pattern.search(text):
            text = pattern.sub(block, text)
        else:
            anchor = text.find(READING_HEADING)
            if anchor < 0:
                text = text.rstrip("\n") + "\n\n" + block
            else:
                text = text[:anchor].rstrip("\n") + "\n\n" + block + "\n" + text[anchor:]
        print(f"published {deck}: {len(decks[deck])} cells")
    with open(markdown, "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
