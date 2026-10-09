#!/usr/bin/env node
/** Dependency-free Node/TypeScript-compatible XEM multiphysics v1 reader. */

import { readFileSync } from "node:fs";

const path = process.argv[2];
if (!path) throw new Error("usage: read_multiphysics_problem.ts <problem.json>");
const document = JSON.parse(readFileSync(path, "utf8"));
if (!document || typeof document !== "object" || Array.isArray(document)) {
  throw new Error("XEM multiphysics document must be an object");
}

const known = new Set([
  "schema_version", "id", "conventions", "domains", "regions", "fields",
  "boundary_conditions", "interface_conditions", "material_laws", "couplings",
  "design_variables", "objectives", "constraints", "provenance",
]);
for (const key of Object.keys(document)) {
  if (!known.has(key)) throw new Error(`unknown XEM multiphysics root field: ${key}`);
}
if (document.schema_version !== 1) throw new Error("compatibility reader supports schema_version 1 only");
if (typeof document.id !== "string" || document.id.trim() === "") throw new Error("multiphysics problem id is required");
for (const key of known) {
  if (key === "schema_version" || key === "id" || key === "conventions" || key === "provenance") continue;
  if (!Array.isArray(document[key])) throw new Error(`${key} must be an array`);
}
if (!document.conventions || typeof document.conventions !== "object" || Array.isArray(document.conventions)) {
  throw new Error("conventions must be an object");
}
if (!document.provenance || typeof document.provenance !== "object" || Array.isArray(document.provenance)) {
  throw new Error("provenance must be an object");
}
console.log(JSON.stringify({
  schema_version: document.schema_version,
  id: document.id,
  domains: document.domains.length,
  fields: document.fields.length,
}));
