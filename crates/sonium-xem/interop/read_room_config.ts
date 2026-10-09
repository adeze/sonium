#!/usr/bin/env node
/** Independent Node/TypeScript-compatible XEM v1 reader.
 *
 * The file uses only the Node standard library so the compatibility gate does
 * not depend on a generated Rust or third-party schema validator.
 */

import { readFileSync } from "node:fs";

const path = process.argv[2];
if (!path) throw new Error("usage: read_room_config.ts <room-config.json>");
const document = JSON.parse(readFileSync(path, "utf8"));
if (!document || typeof document !== "object" || Array.isArray(document)) {
  throw new Error("XEM room document must be an object");
}

const known = new Set([
  "schema_version", "room", "sources", "listening_positions", "frequencies",
  "boundaries", "solver", "visualization", "metadata", "conventions",
]);
for (const key of Object.keys(document)) {
  if (!known.has(key)) throw new Error(`unknown XEM root field: ${key}`);
}
if (document.schema_version !== 1) throw new Error("compatibility reader supports schema_version 1 only");
const room = document.room;
if (!room || typeof room !== "object" || !["rectangular", "lshaped"].includes(String(room.type))) {
  throw new Error("room type is missing or unsupported");
}
if (!Array.isArray(document.sources) || document.sources.length === 0) throw new Error("at least one source is required");
if (!Array.isArray(document.listening_positions) || document.listening_positions.length === 0) throw new Error("at least one listening position is required");
const frequencies = document.frequencies;
if (!frequencies || typeof frequencies !== "object") throw new Error("frequency configuration is missing");
const min = frequencies.min_freq;
const max = frequencies.max_freq;
const points = frequencies.num_points;
if (typeof min !== "number" || !Number.isFinite(min) || typeof max !== "number" || !Number.isFinite(max)) {
  throw new Error("frequency endpoints must be finite numbers");
}
if (min <= 0 || max <= 0 || min > max) throw new Error("frequency endpoints must be positive and ordered");
if (!Number.isInteger(points) || points < 1 || (points > 1 && min === max)) throw new Error("frequency grid is degenerate");
console.log(JSON.stringify({ schema_version: document.schema_version, sources: document.sources.length, positions: document.listening_positions.length }));
