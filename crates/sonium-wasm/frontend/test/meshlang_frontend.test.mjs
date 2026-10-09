import assert from "node:assert/strict";
import test from "node:test";
import {
  EXPECTED_OPERATIONS,
  MAX_TRUSTED_PACKAGE_BYTES,
  createMeshLangFrontend,
  fetchTrustedPackageJson,
  loadGeneratedMeshLangFrontend,
  parseProjectionMeshArtifact,
  validateProjectionMeshArtifact,
} from "../index.mjs";

const source = "geometry model { let body = Box(1.0, 1.0, 1.0) }";

function fakeWasm() {
  const contract = {
    api_schema: 2,
    api_name: "sonium-meshlang-frontend-v2",
    filesystem_free: true,
    production_meshing: false,
    operations: [...EXPECTED_OPERATIONS],
    parse_schema: 1,
    grammar_contract_schema: 1,
    canonical_ir_preview_schema: 1,
    registry_preview_schema: 1,
  };
  let entries = new Set();
  let hits = 0;
  let misses = 0;
  const envelope = (kind, extra = {}) =>
    JSON.stringify({ kind, parse_schema: 1, report: {}, ...extra });
  return {
    meshlangFrontendContract: () => JSON.stringify(contract),
    parseMeshLang: (text, filename) => {
      entries.add(filename);
      misses += 1;
      return envelope("parse", { text });
    },
    validateMeshLangGrammar: () => envelope("grammar"),
      previewCanonicalIr: () => envelope("preview"),
      meshProjection: () =>
        JSON.stringify({
          schema_version: 1,
          mesh: {
            vertices: [],
            triangles: [],
            bounds: [[0, 0, 0], [1, 1, 1]],
            cells: [4, 4, 4],
          },
        }),
    previewCanonicalIrWithRegistry: () => envelope("registry"),
      previewCanonicalIrWithSignedRegistry: () => envelope("signed"),
      previewCanonicalIrWithTrustedRegistry: () => envelope("trusted"),
      previewCanonicalIrWithTrustedPackage: () => envelope("trusted-package"),
    MeshLangEditor: class {
      parse(_text, filename) {
        if (entries.has(filename)) hits += 1;
        else {
          entries.add(filename);
          misses += 1;
        }
        return envelope("editor");
      }
      parseSubtree(_text, _filename, start, end) {
        return envelope("subtree", { start, end });
      }
      invalidate(filename) {
        return entries.delete(filename);
      }
      clear() {
        entries.clear();
      }
      stats() {
        return JSON.stringify({ hits, misses, entries: entries.size, subtree_entries: 0 });
      }
    },
  };
}

test("validates the Rust contract and preserves JSON operation envelopes", () => {
  const frontend = createMeshLangFrontend(fakeWasm());
  assert.equal(frontend.contract.api_name, "sonium-meshlang-frontend-v2");
  assert.equal(frontend.parse(source, "main.mlg").kind, "parse");
  assert.equal(frontend.grammar().kind, "grammar");
  assert.equal(frontend.preview(source, "main.mlg").kind, "preview");
  const artifact = frontend.meshProjection(
    source,
    "main.mlg",
    "model.surface",
    [[0, 0, 0], [1, 1, 1]],
    [4, 4, 4],
    0.01,
  );
  assert.equal(artifact.schema_version, 1);
  assert.deepEqual(artifact.mesh.cells, [4, 4, 4]);
  assert.equal(frontend.previewWithRegistry(source, "main.mlg", "{}").kind, "registry");
  assert.equal(
    frontend.previewWithSignedRegistry(source, "main.mlg", "{}").kind,
    "signed",
  );
    assert.equal(
      frontend.previewWithTrustedRegistry(source, "main.mlg", "{}", "{}").kind,
      "trusted",
    );
    assert.equal(
      frontend.previewWithTrustedPackage(source, "main.mlg", "{}", "{}").kind,
      "trusted-package",
    );
});

test("editor adapter exposes cache invalidation and telemetry", () => {
  const editor = createMeshLangFrontend(fakeWasm()).editor();
  editor.parse(source, "main.mlg");
  editor.parse(source, "main.mlg");
  assert.deepEqual(editor.stats(), { hits: 1, misses: 1, entries: 1, subtree_entries: 0 });
  assert.equal(editor.invalidate("main.mlg"), true);
  assert.deepEqual(editor.stats(), { hits: 1, misses: 1, entries: 0, subtree_entries: 0 });
  editor.clear();
  assert.deepEqual(editor.stats(), { hits: 1, misses: 1, entries: 0, subtree_entries: 0 });
});

test("editor adapter exposes bounded subtree parsing", () => {
  const editor = createMeshLangFrontend(fakeWasm()).editor();
  const result = editor.parseSubtree(source, "main.mlg", 3, source.length);
  assert.equal(result.kind, "subtree");
  assert.equal(result.start, 3);
  assert.equal(result.end, source.length);
});

test("rejects a downgraded or capability-expanding module", () => {
  const wasm = fakeWasm();
  wasm.meshlangFrontendContract = () =>
    JSON.stringify({
    api_schema: 2,
    api_name: "sonium-meshlang-frontend-v2",
      filesystem_free: false,
      production_meshing: true,
      operations: [...EXPECTED_OPERATIONS],
    });
  assert.throws(() => createMeshLangFrontend(wasm), /filesystem-free/);
});

test("rejects operation-order drift and malformed schema versions", () => {
  const reordered = fakeWasm();
  reordered.meshlangFrontendContract = () =>
    JSON.stringify({
    api_schema: 2,
    api_name: "sonium-meshlang-frontend-v2",
      filesystem_free: true,
      production_meshing: false,
      operations: [...EXPECTED_OPERATIONS].reverse(),
      parse_schema: 1,
      grammar_contract_schema: 1,
      canonical_ir_preview_schema: 1,
      registry_preview_schema: 1,
    });
  assert.throws(() => createMeshLangFrontend(reordered), /operation contract drifted/);

  const malformed = fakeWasm();
  malformed.meshlangFrontendContract = () =>
    JSON.stringify({
    api_schema: 2,
    api_name: "sonium-meshlang-frontend-v2",
      filesystem_free: true,
      production_meshing: false,
      operations: [...EXPECTED_OPERATIONS],
      parse_schema: 0,
      grammar_contract_schema: 1,
      canonical_ir_preview_schema: 1,
      registry_preview_schema: 1,
    });
  assert.throws(() => createMeshLangFrontend(malformed), /schema field 'parse_schema'/);
});

test("rejects non-object JSON envelopes from wasm exports", () => {
  const nullEnvelope = fakeWasm();
  nullEnvelope.parseMeshLang = () => "null";
  const nullFrontend = createMeshLangFrontend(nullEnvelope);
  assert.throws(
    () => nullFrontend.parse(source, "malformed-null.mlg"),
    /parseMeshLang must return a JSON object envelope/,
  );

  const arrayEnvelope = fakeWasm();
  arrayEnvelope.validateMeshLangGrammar = () => "[]";
  const arrayFrontend = createMeshLangFrontend(arrayEnvelope);
  assert.throws(
    () => arrayFrontend.grammar(),
    /validateMeshLangGrammar must return a JSON object envelope/,
  );
});

test("initializes an injected generated module before validating its contract", async () => {
  const wasm = fakeWasm();
  let initialized = false;
  wasm.default = async () => {
    initialized = true;
  };
  const frontend = await loadGeneratedMeshLangFrontend(async () => wasm);
  assert.equal(initialized, true);
  assert.equal(frontend.contract.api_name, "sonium-meshlang-frontend-v2");
});

test("accepts initializer-returned wasm export namespaces", async () => {
  const exports = fakeWasm();
  const module = { default: async () => exports };
  const frontend = await loadGeneratedMeshLangFrontend(async () => module);
  assert.equal(frontend.contract.api_name, "sonium-meshlang-frontend-v2");
  assert.equal(frontend.parse(source, "returned-exports.mlg").kind, "parse");
});

test("rejects non-object wasm initializer results", async () => {
  const module = { default: async () => "invalid exports" };
  await assert.rejects(
    () => loadGeneratedMeshLangFrontend(async () => module),
    /initializer returned a non-object namespace/,
  );
});

test("reports host module-loader failures without touching the filesystem", async () => {
  await assert.rejects(
    () => loadGeneratedMeshLangFrontend(async () => {
      throw new Error("fixture unavailable");
    }),
    /failed to load the MeshLang WASM module: fixture unavailable/,
  );
  await assert.rejects(
    () => loadGeneratedMeshLangFrontend(null),
    /module loader must be a function/,
  );
});

test("rejects invalid module namespaces before probing exports", async () => {
  await assert.rejects(
    () => loadGeneratedMeshLangFrontend(async () => null),
    /module loader must return an object namespace/,
  );
  await assert.rejects(
    () => loadGeneratedMeshLangFrontend(async () => []),
    /module loader must return an object namespace/,
  );
  await assert.rejects(
    () => loadGeneratedMeshLangFrontend(async () => "invalid namespace"),
    /module loader must return an object namespace/,
  );
  assert.throws(
    () => createMeshLangFrontend([]),
    /generated MeshLang WASM module is required/,
  );
});

test("reports generated-module initializer failures with a typed boundary", async () => {
  await assert.rejects(
    () =>
      loadGeneratedMeshLangFrontend(async () => ({
        default: async () => {
          throw new Error("wasm init failed");
        },
      })),
    /failed to initialize the MeshLang WASM module: wasm init failed/,
  );
  await assert.rejects(
    () => loadGeneratedMeshLangFrontend(async () => ({ default: async () => [] })),
  /initializer returned a non-object namespace/,
  );
});

test("fetches trusted package JSON through an injected bounded transport", async () => {
  let request;
  const frontend = createMeshLangFrontend(fakeWasm());
  const result = await frontend.previewWithTrustedPackageFromFetch(
    source,
    "transport.mlg",
    "https://registry.example/packages/demo/1.0.json",
    "{\"issuers\":[]}",
    async (url, init) => {
      request = { url, init };
      return {
        ok: true,
        status: 200,
        headers: { get: () => "42" },
        text: async () => JSON.stringify({ schema_version: 1, index: {} }),
      };
    },
  );
  assert.equal(result.kind, "trusted-package");
  assert.equal(request.url, "https://registry.example/packages/demo/1.0.json");
  assert.deepEqual(request.init, {
    method: "GET",
    headers: { accept: "application/json" },
    redirect: "error",
  });
});

test("fails closed for trusted package transport URL, status, size, and JSON errors", async () => {
  const fetcher = async () => ({
    ok: true,
    status: 200,
    text: async () => JSON.stringify({ schema_version: 1 }),
  });
  await assert.rejects(
    () => fetchTrustedPackageJson("file:///tmp/package.json", fetcher),
    /absolute HTTP\(S\) URL/,
  );
  await assert.rejects(
    () => fetchTrustedPackageJson("https://registry.example/package?token=secret", fetcher),
    /without credentials, query, or fragment/,
  );
  await assert.rejects(
    () =>
      fetchTrustedPackageJson("https://registry.example/package.json", async () => ({
        ok: false,
        status: 503,
        text: async () => "{}",
      })),
    /unsuccessful response.*503/,
  );
  await assert.rejects(
    () =>
      fetchTrustedPackageJson("https://registry.example/package.json", async () => ({
        ok: true,
        headers: { get: () => String(MAX_TRUSTED_PACKAGE_BYTES + 1) },
        text: async () => "{}",
      })),
    /exceeds.*byte limit/,
  );
  await assert.rejects(
    () =>
      fetchTrustedPackageJson("https://registry.example/package.json", async () => ({
        ok: true,
        text: async () => "not-json",
      })),
    /invalid JSON/,
  );
});

const projectionArtifact = {
  schema_version: 1,
  mesh: {
    vertices: [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ],
    triangles: [[0, 1, 2]],
    bounds: [
      [0, 0, 0],
    [1, 1, 1],
    ],
    cells: [2, 2, 1],
  },
};

test("validates and decodes the Rust projection mesh artifact contract", () => {
  const json = JSON.stringify(projectionArtifact);
  assert.deepEqual(parseProjectionMeshArtifact(json), projectionArtifact);
  assert.strictEqual(validateProjectionMeshArtifact(projectionArtifact), projectionArtifact);
});

test("rejects projection artifact schema, geometry, and triangle drift", () => {
  const wrongSchema = structuredClone(projectionArtifact);
  wrongSchema.schema_version = 2;
  assert.throws(() => validateProjectionMeshArtifact(wrongSchema), /schema version/);

  const extraField = structuredClone(projectionArtifact);
  extraField.mesh.extra = true;
  assert.throws(() => validateProjectionMeshArtifact(extraField), /unexpected fields/);

  const badTriangle = structuredClone(projectionArtifact);
  badTriangle.mesh.triangles = [[0, 0, 1]];
  assert.throws(() => validateProjectionMeshArtifact(badTriangle), /degenerate/);

  const outOfRange = structuredClone(projectionArtifact);
  outOfRange.mesh.triangles = [[0, 1, 99]];
  assert.throws(() => validateProjectionMeshArtifact(outOfRange), /invalid vertex/);

  const nonFinite = structuredClone(projectionArtifact);
  nonFinite.mesh.vertices[0][0] = Number.NaN;
  assert.throws(() => validateProjectionMeshArtifact(nonFinite), /must be finite/);
  const flatBounds = structuredClone(projectionArtifact);
  flatBounds.mesh.bounds[1][2] = 0;
  assert.throws(() => validateProjectionMeshArtifact(flatBounds), /positive extent/);
});
