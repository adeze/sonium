/**
 * Browser-facing adapter for the generated sonium-roomsim wasm-bindgen module.
 *
 * The adapter deliberately owns no filesystem, solver, or Gmsh capability. It
 * validates the Rust capability contract once and keeps the JSON wire format
 * intact so browser clients cannot accidentally consume a different API
 * version after a package upgrade.
 */

export const EXPECTED_API_NAME = "sonium-meshlang-frontend-v2";
export const EXPECTED_API_SCHEMA = 2;
export const EXPECTED_OPERATIONS = Object.freeze([
  "meshlangFrontendContract",
  "parseMeshLang",
  "validateMeshLangGrammar",
  "previewCanonicalIr",
  "meshProjection",
  "previewCanonicalIrWithRegistry",
  "previewCanonicalIrWithSignedRegistry",
  "previewCanonicalIrWithTrustedRegistry",
  "previewCanonicalIrWithTrustedPackage",
  "MeshLangEditor",
]);

export const PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION = 1;
export const MAX_PROJECTION_MESH_ARTIFACT_BYTES = 64 * 1024 * 1024;
export const MAX_TRUSTED_PACKAGE_BYTES = 8 * 1024 * 1024;
const MAX_PROJECTION_MESH_CELLS = 262_144;
const MAX_PROJECTION_MESH_CELLS_PER_AXIS = 64;

function validateTrustedPackageUrl(url) {
  if (typeof url !== "string" || url.length === 0 || /\s/.test(url)) {
    throw new TypeError("trusted package URL must be a non-empty URL without whitespace");
  }
  let parsed;
  try {
    parsed = new URL(url);
  } catch (error) {
    throw new TypeError(`trusted package URL is invalid: ${error.message}`);
  }
  if (
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.search !== "" ||
    parsed.hash !== ""
  ) {
    throw new TypeError(
      "trusted package URL must be an absolute HTTP(S) URL without credentials, query, or fragment",
    );
  }
  return parsed.href;
}

/** Fetch a bounded, JSON package envelope for trusted Rust-side verification. */
export async function fetchTrustedPackageJson(url, fetchImpl = globalThis.fetch) {
  const endpoint = validateTrustedPackageUrl(url);
  if (typeof fetchImpl !== "function") {
    throw new TypeError("trusted package fetcher must be a function");
  }
  let response;
  try {
    response = await fetchImpl(endpoint, {
      method: "GET",
      headers: { accept: "application/json" },
      redirect: "error",
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`trusted package fetch failed: ${detail}`, { cause: error });
  }
  if (response === null || typeof response !== "object") {
    throw new TypeError("trusted package fetcher returned an invalid response");
  }
  if (response.ok !== true) {
    const status = Number.isInteger(response.status) ? ` (HTTP ${response.status})` : "";
    throw new Error(`trusted package fetch returned an unsuccessful response${status}`);
  }
  const declaredLength = response.headers?.get?.("content-length");
  if (declaredLength !== undefined && declaredLength !== null) {
    const length = Number(declaredLength);
    if (Number.isFinite(length) && length > MAX_TRUSTED_PACKAGE_BYTES) {
      throw new RangeError(
        `trusted package response exceeds ${MAX_TRUSTED_PACKAGE_BYTES}-byte limit`,
      );
    }
  }
  if (typeof response.text !== "function") {
    throw new TypeError("trusted package response must expose text()");
  }
  let body;
  try {
    body = await response.text();
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`trusted package response read failed: ${detail}`, { cause: error });
  }
  if (typeof body !== "string") {
    throw new TypeError("trusted package response text() must return a string");
  }
  if (new TextEncoder().encode(body).byteLength > MAX_TRUSTED_PACKAGE_BYTES) {
    throw new RangeError(
      `trusted package response exceeds ${MAX_TRUSTED_PACKAGE_BYTES}-byte limit`,
    );
  }
  let envelope;
  try {
    envelope = JSON.parse(body);
  } catch (error) {
    throw new TypeError(`trusted package response is invalid JSON: ${error.message}`);
  }
  if (envelope === null || typeof envelope !== "object" || Array.isArray(envelope)) {
    throw new TypeError("trusted package response must be a JSON object envelope");
  }
  return body;
}

function artifactError(message) {
  throw new TypeError(`invalid ProjectionMeshArtifact: ${message}`);
}

function requireFiniteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    artifactError(`${label} must be finite`);
  }
  return value;
}

function requireExactKeys(value, expected, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    artifactError(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const keys = [...expected].sort();
  if (actual.length !== keys.length || actual.some((key, index) => key !== keys[index])) {
    artifactError(`${label} has unexpected fields`);
  }
}

/** Validate and decode a Rust `ProjectionMeshArtifact` JSON payload. */
export function parseProjectionMeshArtifact(json) {
  if (typeof json !== "string") {
    throw new TypeError("ProjectionMeshArtifact JSON must be a string");
  }
  const bytes = new TextEncoder().encode(json).byteLength;
  if (bytes > MAX_PROJECTION_MESH_ARTIFACT_BYTES) {
    throw new RangeError(
      `ProjectionMeshArtifact exceeds ${MAX_PROJECTION_MESH_ARTIFACT_BYTES}-byte limit`,
    );
  }
  let artifact;
  try {
    artifact = JSON.parse(json);
  } catch (error) {
    throw new TypeError(`ProjectionMeshArtifact JSON is invalid: ${error.message}`);
  }
  return validateProjectionMeshArtifact(artifact);
}

/** Validate an already-decoded `ProjectionMeshArtifact` object. */
export function validateProjectionMeshArtifact(artifact) {
  requireExactKeys(artifact, ["schema_version", "mesh"], "artifact");
  if (
    !Number.isInteger(artifact.schema_version) ||
    artifact.schema_version !== PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION
  ) {
    artifactError(
      `schema version ${artifact.schema_version}; expected ${PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION}`,
    );
  }
  const mesh = artifact.mesh;
  requireExactKeys(mesh, ["vertices", "triangles", "bounds", "cells"], "mesh");
  if (
    !Array.isArray(mesh.bounds) ||
    mesh.bounds.length !== 2 ||
    mesh.bounds.some((bound) => !Array.isArray(bound) || bound.length !== 3)
  ) {
    artifactError("bounds must be two 3-vectors");
  }
  for (let axis = 0; axis < 3; axis += 1) {
    const lower = requireFiniteNumber(mesh.bounds[0][axis], `bounds[0][${axis}]`);
    const upper = requireFiniteNumber(mesh.bounds[1][axis], `bounds[1][${axis}]`);
    if (lower >= upper) artifactError(`bounds axis ${axis} must have positive extent`);
  }
  if (
    !Array.isArray(mesh.cells) ||
    mesh.cells.length !== 3 ||
    mesh.cells.some(
      (count) =>
        !Number.isInteger(count) ||
        count <= 0 ||
        count > MAX_PROJECTION_MESH_CELLS_PER_AXIS,
    )
  ) {
    artifactError("cells must contain three bounded positive integers");
  }
  if (mesh.cells[0] * mesh.cells[1] * mesh.cells[2] > MAX_PROJECTION_MESH_CELLS) {
    artifactError("grid exceeds cell limit");
  }
  if (!Array.isArray(mesh.vertices)) artifactError("vertices must be an array");
  mesh.vertices.forEach((vertex, index) => {
    if (!Array.isArray(vertex) || vertex.length !== 3) {
      artifactError(`vertex ${index} must be a 3-vector`);
    }
    vertex.forEach((value, axis) => requireFiniteNumber(value, `vertex ${index}[${axis}]`));
  });
  if (!Array.isArray(mesh.triangles)) artifactError("triangles must be an array");
  mesh.triangles.forEach((triangle, index) => {
    if (
      !Array.isArray(triangle) ||
      triangle.length !== 3 ||
      triangle.some(
        (vertex) =>
          !Number.isInteger(vertex) || vertex < 0 || vertex >= mesh.vertices.length,
      )
    ) {
      artifactError(`triangle ${index} references an invalid vertex`);
    }
    if (new Set(triangle).size !== 3) artifactError(`triangle ${index} is degenerate`);
  });
  return artifact;
}

function parseJsonEnvelope(json, label) {
  if (typeof json !== "string") {
    throw new TypeError(`${label} must return a JSON string`);
  }
  let value;
  try {
    value = JSON.parse(json);
  } catch (error) {
    throw new TypeError(`${label} returned invalid JSON: ${error.message}`);
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must return a JSON object envelope`);
  }
  return value;
}

function requireFunction(exports, name) {
  if (typeof exports[name] !== "function") {
    throw new TypeError(`MeshLang WASM export '${name}' is missing`);
  }
  return exports[name].bind(exports);
}

function validateContract(contract) {
  if (contract.api_name !== EXPECTED_API_NAME) {
    throw new Error(
      `unsupported MeshLang frontend API '${contract.api_name ?? ""}'`,
    );
  }
  if (contract.api_schema !== EXPECTED_API_SCHEMA) {
    throw new Error(
      `unsupported MeshLang frontend schema ${contract.api_schema}; expected ${EXPECTED_API_SCHEMA}`,
    );
  }
  if (contract.filesystem_free !== true || contract.production_meshing !== false) {
    throw new Error("MeshLang frontend contract is not filesystem-free");
  }
  if (
    !Array.isArray(contract.operations) ||
    JSON.stringify(contract.operations) !== JSON.stringify(EXPECTED_OPERATIONS)
  ) {
    throw new Error("MeshLang frontend operation contract drifted");
  }
  for (const field of [
    "parse_schema",
    "grammar_contract_schema",
    "canonical_ir_preview_schema",
    "registry_preview_schema",
  ]) {
    if (!Number.isInteger(contract[field]) || contract[field] < 1) {
      throw new Error(`MeshLang frontend schema field '${field}' is invalid`);
    }
  }
  return contract;
}

/**
 * Adapt an initialized wasm-bindgen module to a stable browser API.
 * @param {Record<string, Function>} wasm generated sonium_roomsim exports
 */
export function createMeshLangFrontend(wasm) {
  if (wasm === null || typeof wasm !== "object" || Array.isArray(wasm)) {
    throw new TypeError("a generated MeshLang WASM module is required");
  }

  const contractCall = requireFunction(wasm, "meshlangFrontendContract");
  const parseCall = requireFunction(wasm, "parseMeshLang");
  const grammarCall = requireFunction(wasm, "validateMeshLangGrammar");
  const previewCall = requireFunction(wasm, "previewCanonicalIr");
  const meshProjectionCall = requireFunction(wasm, "meshProjection");
  const registryCall = requireFunction(wasm, "previewCanonicalIrWithRegistry");
  const signedRegistryCall = requireFunction(
    wasm,
    "previewCanonicalIrWithSignedRegistry",
  );
  const trustedRegistryCall = requireFunction(
    wasm,
    "previewCanonicalIrWithTrustedRegistry",
  );
  const trustedPackageCall = requireFunction(
    wasm,
    "previewCanonicalIrWithTrustedPackage",
  );

  const contract = validateContract(parseJsonEnvelope(contractCall(), "meshlangFrontendContract"));

  const parse = (source, filename) =>
    parseJsonEnvelope(parseCall(source, filename), "parseMeshLang");
  const grammar = () =>
    parseJsonEnvelope(grammarCall(), "validateMeshLangGrammar");
  const preview = (source, filename) =>
    parseJsonEnvelope(previewCall(source, filename), "previewCanonicalIr");
  const meshProjection = (source, filename, projectionName, bounds, cells, tolerance) => {
    const request = JSON.stringify({ bounds, cells, tolerance });
    return parseProjectionMeshArtifact(
      meshProjectionCall(source, filename, projectionName, request),
    );
  };
  const previewWithRegistry = (source, filename, registryJson) =>
    parseJsonEnvelope(
      registryCall(source, filename, registryJson),
      "previewCanonicalIrWithRegistry",
    );
  const previewWithSignedRegistry = (source, filename, registryJson) =>
    parseJsonEnvelope(
      signedRegistryCall(source, filename, registryJson),
      "previewCanonicalIrWithSignedRegistry",
    );
  const previewWithTrustedRegistry = (
    source,
    filename,
    registryJson,
    trustPolicyJson,
  ) =>
    parseJsonEnvelope(
      trustedRegistryCall(source, filename, registryJson, trustPolicyJson),
    "previewCanonicalIrWithTrustedRegistry",
  );
  const previewWithTrustedPackage = (
    source,
    filename,
    packageJson,
    trustPolicyJson,
  ) =>
    parseJsonEnvelope(
      trustedPackageCall(source, filename, packageJson, trustPolicyJson),
      "previewCanonicalIrWithTrustedPackage",
    );
  const previewWithTrustedPackageFromFetch = async (
    source,
    filename,
    packageUrl,
    trustPolicyJson,
    fetchImpl,
  ) =>
    previewWithTrustedPackage(
      source,
      filename,
      await fetchTrustedPackageJson(packageUrl, fetchImpl),
      trustPolicyJson,
    );

  const editor = () => {
    if (typeof wasm.MeshLangEditor !== "function") {
      throw new TypeError("MeshLangEditor export is missing");
    }
    const instance = new wasm.MeshLangEditor();
    return {
      parse(source, filename) {
        return parseJsonEnvelope(instance.parse(source, filename), "MeshLangEditor.parse");
      },
      parseSubtree(source, filename, start, end) {
        return parseJsonEnvelope(
          instance.parseSubtree(source, filename, start, end),
          "MeshLangEditor.parseSubtree",
        );
      },
      invalidate(filename) {
        return instance.invalidate(filename);
      },
      clear() {
        instance.clear();
      },
      stats() {
        return parseJsonEnvelope(instance.stats(), "MeshLangEditor.stats");
      },
    };
  };

  return Object.freeze({
    contract,
    parse,
    grammar,
    preview,
    meshProjection,
    previewWithRegistry,
    previewWithSignedRegistry,
  previewWithTrustedRegistry,
  previewWithTrustedPackage,
  previewWithTrustedPackageFromFetch,
  editor,
  });
}

/**
 * Load the wasm-bindgen package emitted by build-wasm.sh.
 *
 * The optional loader is intentionally host-owned: browser applications can
 * provide their bundler/import-map loader, while tests and workers can inject
 * a module without touching the filesystem. The default remains the published
 * package-relative ESM artifact.
 *
 * @param {() => Promise<Record<string, Function>> | Record<string, Function>} [loadModule]
 */
export async function loadGeneratedMeshLangFrontend(
  loadModule = () => import("./pkg/sonium_roomsim.js"),
) {
  if (typeof loadModule !== "function") {
    throw new TypeError("the MeshLang WASM module loader must be a function");
  }
  let wasm;
  try {
    wasm = await loadModule();
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`failed to load the MeshLang WASM module: ${detail}`, {
      cause: error,
    });
  }
  // wasm-bindgen's web target exposes an async default initializer. Calling it
  // here makes the package entry point usable directly from browser ESM while
  // remaining compatible with test doubles or wasm-pack variants without one.
 if (wasm === null || typeof wasm !== "object" || Array.isArray(wasm)) {
  throw new TypeError(
   "the MeshLang WASM module loader must return an object namespace",
  );
 }
 if (typeof wasm.default === "function") {
  let initialized;
  try {
   initialized = await wasm.default();
  } catch (error) {
   const detail = error instanceof Error ? error.message : String(error);
   throw new Error(`failed to initialize the MeshLang WASM module: ${detail}`, {
    cause: error,
   });
  }
    // wasm-bindgen variants either mutate the namespace in place or return
    // the finalized exports. Preserve both loader contracts so browser
    // workers and bundlers can wrap the generated initializer safely.
    if (initialized !== undefined && initialized !== null) {
  if (typeof initialized !== "object" || Array.isArray(initialized)) {
        throw new TypeError("MeshLang WASM initializer returned a non-object namespace");
      }
      wasm = { ...wasm, ...initialized };
    }
  }
  return createMeshLangFrontend(wasm);
}
