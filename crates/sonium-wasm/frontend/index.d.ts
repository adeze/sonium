export interface MeshLangFrontendContract {
  api_schema: number;
  api_name: string;
  filesystem_free: boolean;
  production_meshing: boolean;
  operations: readonly string[];
  parse_schema: number;
  grammar_contract_schema: number;
  canonical_ir_preview_schema: number;
  registry_preview_schema: number;
}

export interface MeshLangWasmEditor {
  parse(source: string, filename: string): Record<string, unknown>;
  parseSubtree(
    source: string,
    filename: string,
    start: number,
    end: number,
  ): Record<string, unknown>;
  invalidate(filename: string): boolean;
  clear(): void;
  stats(): Record<string, number>;
}

export interface MeshLangFrontend {
  readonly contract: MeshLangFrontendContract;
  parse(source: string, filename: string): Record<string, unknown>;
  grammar(): Record<string, unknown>;
  preview(source: string, filename: string): Record<string, unknown>;
  meshProjection(
    source: string,
    filename: string,
    projectionName: string,
    bounds: [[number, number, number], [number, number, number]],
    cells: [number, number, number],
    tolerance: number,
  ): ProjectionMeshArtifact;
  previewWithRegistry(
    source: string,
    filename: string,
    registryJson: string,
  ): Record<string, unknown>;
  previewWithSignedRegistry(
    source: string,
    filename: string,
    registryJson: string,
  ): Record<string, unknown>;
  previewWithTrustedRegistry(
    source: string,
    filename: string,
    registryJson: string,
    trustPolicyJson: string,
  ): Record<string, unknown>;
  previewWithTrustedPackage(
    source: string,
    filename: string,
    packageJson: string,
    trustPolicyJson: string,
  ): Record<string, unknown>;
  previewWithTrustedPackageFromFetch(
    source: string,
    filename: string,
    packageUrl: string,
    trustPolicyJson: string,
    fetchImpl?: MeshLangPackageFetcher,
  ): Promise<Record<string, unknown>>;
  editor(): MeshLangWasmEditor;
}

export const EXPECTED_API_NAME: string;
export const EXPECTED_API_SCHEMA: number;
export const EXPECTED_OPERATIONS: readonly string[];
export const PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION: number;
export const MAX_PROJECTION_MESH_ARTIFACT_BYTES: number;
export const MAX_TRUSTED_PACKAGE_BYTES: number;
export interface ProjectionMeshArtifact {
  schema_version: number;
  mesh: {
    vertices: number[][];
    triangles: number[][];
    bounds: number[][];
    cells: number[];
  };
}
export function parseProjectionMeshArtifact(json: string): ProjectionMeshArtifact;
export function validateProjectionMeshArtifact(
  artifact: unknown,
): ProjectionMeshArtifact;

export interface MeshLangPackageResponse {
  ok: boolean;
  status?: number;
  headers?: { get(name: string): string | null };
  text(): Promise<string>;
}

export type MeshLangPackageFetcher = (
  url: string,
  init: {
    method: "GET";
    headers: { accept: "application/json" };
    redirect: "error";
  },
) => Promise<MeshLangPackageResponse>;

export function fetchTrustedPackageJson(
  url: string,
  fetchImpl?: MeshLangPackageFetcher,
): Promise<string>;
export function createMeshLangFrontend(
  wasm: Record<string, Function>,
): MeshLangFrontend;
export type MeshLangWasmModuleLoader = () =>
  | Promise<Record<string, Function>>
  | Record<string, Function>;
export function loadGeneratedMeshLangFrontend(
  loadModule?: MeshLangWasmModuleLoader,
): Promise<MeshLangFrontend>;
