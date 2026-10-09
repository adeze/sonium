//! Import resolver: handles `import` declarations and module loading.
//!
//! Resolves import paths to file contents, handles the standard library,
//! and detects circular imports.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::CompileError;
use meshlang_ast::*;

/// Resolved module: parsed AST + metadata.
#[derive(Debug)]
pub struct ResolvedModule {
    pub path: PathBuf,
    pub source: String,
    pub ast: SourceFile,
    pub source_hash: StructuralHash,
    pub ast_hash: StructuralHash,
}

/// Cumulative parsed-module cache activity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImportCacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// A cross-platform, source-backed module registry entry.
///
/// Registry keys are normalized MeshLang import paths (`foo/bar.mlg` or
/// `foo/bar@1.2.mlg`) rather than host filesystem paths. This gives package
/// managers and WASM/editor clients one deterministic resolution contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredModule {
    pub import_path: String,
    pub source: String,
}

/// The import resolver manages module loading and caching.
pub struct ImportResolver {
    /// Search paths for imports (in order of priority).
    search_paths: Vec<PathBuf>,
    /// Cached parsed modules.
    cache: HashMap<PathBuf, ResolvedModule>,
    /// Currently resolving (for cycle detection).
    resolving: HashSet<PathBuf>,
    /// Standard library path.
    stdlib_path: Option<PathBuf>,
    /// Optional cross-platform source registry. Registry entries are checked
    /// before filesystem resolution and never fall back across versions.
    registry: BTreeMap<String, RegisteredModule>,
    /// Explicit old-to-new module aliases installed only by a declared
    /// content-pinned package migration policy.
    registry_migrations: BTreeMap<String, String>,
    /// When enabled, every import must resolve from `registry`; filesystem
    /// and configured stdlib paths are unavailable (used by WASM previews).
    registry_only: bool,
    /// Optional canonical root that non-stdlib imports may not escape.
    sandbox_root: Option<PathBuf>,
    /// Whether absolute non-stdlib imports are accepted.
    allow_absolute_paths: bool,
    /// Deterministic parser file IDs for exact cross-module source spans.
    file_ids: HashMap<PathBuf, u32>,
    next_file_id: u32,
    cache_stats: ImportCacheStats,
    /// Directed module dependency graph retained for incremental impact
    /// queries. BTreeSet keeps diagnostics and editor invalidation lists
    /// deterministic across processes.
    dependencies: HashMap<PathBuf, BTreeSet<PathBuf>>,
    dependents: HashMap<PathBuf, BTreeSet<PathBuf>>,
}

impl Default for ImportResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportResolver {
    pub fn new() -> Self {
        Self {
            search_paths: Vec::new(),
            cache: HashMap::new(),
            resolving: HashSet::new(),
            stdlib_path: None,
            registry: BTreeMap::new(),
            registry_migrations: BTreeMap::new(),
            registry_only: false,
            sandbox_root: None,
            allow_absolute_paths: false,
            file_ids: HashMap::new(),
            next_file_id: 1,
            cache_stats: ImportCacheStats::default(),
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
        }
    }

    /// Add a search path for resolving imports.
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Set the standard library path.
    pub fn set_stdlib_path(&mut self, path: impl Into<PathBuf>) {
        self.stdlib_path = Some(path.into());
    }

    /// Register an in-memory module using its canonical import spelling.
    ///
    /// The path uses `/` separators on every host and must end in `.mlg`.
    /// Version selectors are exact (`module@2.mlg` never falls back to
    /// `module.mlg` or another version). Registering a key twice is rejected
    /// so package resolution cannot depend on insertion order.
    pub fn register_module(
        &mut self,
        import_path: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<(), CompileError> {
        let key = canonical_registry_key(&import_path.into())?;
        if self.registry.contains_key(&key) {
            return Err(CompileError::Unsupported(format!(
                "registry module '{key}' is already registered"
            )));
        }
        self.registry.insert(
            key.clone(),
            RegisteredModule {
                import_path: key,
                source: source.into(),
            },
        );
        Ok(())
    }

    /// Register every canonical module from a validated package index.
    ///
    /// The index is validated and all registry keys are preflighted before
    /// mutating this resolver, so a collision cannot leave a partially
    /// registered package behind. Package-index ordering is already
    /// canonical, but the resolver keeps its own key ordering contract.
    pub fn register_package_index(
        &mut self,
        index: &crate::registry::RegistryPackageIndex,
    ) -> Result<(), CompileError> {
        self.register_package_index_with_migration_policy(
            index,
            &crate::registry::RegistryPackageMigrationPolicy::Reject,
        )
    }

    /// Register a package index with an explicit version-migration policy.
    ///
    /// Migration aliases are never enabled implicitly. When a declared plan
    /// is allowed, source-version modules are omitted from the active registry
    /// and their exact import keys resolve to the pinned target modules.
    pub fn register_package_index_with_migration_policy(
        &mut self,
        index: &crate::registry::RegistryPackageIndex,
        policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<(), CompileError> {
        let (modules, aliases) = index.modules_with_migration_policy(policy)?;
        let mut pending = BTreeSet::new();
        let entries = modules
            .into_iter()
            .map(|module| {
                let key = canonical_registry_key(&module.path)?;
                if self.registry.contains_key(&key) || !pending.insert(key.clone()) {
                    return Err(CompileError::Unsupported(format!(
                        "registry module '{key}' already registered"
                    )));
                }
                Ok((
                    key.clone(),
                    RegisteredModule {
                        import_path: key,
                        source: module.source,
                    },
                ))
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let mut pending_aliases = BTreeSet::new();
        for (from, to) in &aliases {
            if self.registry.contains_key(from)
                || self.registry_migrations.contains_key(from)
                || !pending_aliases.insert(from.clone())
            {
                return Err(CompileError::Unsupported(format!(
                    "registry migration source '{from}' is already registered"
                )));
            }
            if !self.registry.contains_key(to) && !entries.iter().any(|(key, _)| key == to) {
                return Err(CompileError::Unsupported(format!(
                    "registry migration target '{to}' is not registered"
                )));
            }
        }

        for (key, module) in entries {
            self.registry.insert(key, module);
        }
        for (from, to) in aliases {
            self.registry_migrations.insert(from, to);
        }
        Ok(())
    }

    /// Verify a signed package index against the host trust policy, then
    /// register it atomically through [`Self::register_package_index`].
    ///
    /// Returning the authenticated canonical index hash lets callers bind
    /// cache/provenance records to the exact package set they accepted.
    pub fn register_trusted_package_index(
        &mut self,
        envelope: &crate::registry::RegistryPackageIndexEnvelope,
        policy: &crate::incremental::RegistrySignaturePolicy,
    ) -> Result<StructuralHash, CompileError> {
        self.register_trusted_package_index_with_migration_policy(
            envelope,
            policy,
            &crate::registry::RegistryPackageMigrationPolicy::Reject,
        )
    }

    /// Verify and register a signed package index with an explicit migration
    /// policy before mutating resolver state.
    pub fn register_trusted_package_index_with_migration_policy(
        &mut self,
        envelope: &crate::registry::RegistryPackageIndexEnvelope,
        policy: &crate::incremental::RegistrySignaturePolicy,
        migration_policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<StructuralHash, CompileError> {
        let index_hash = envelope.verify_with_policy(policy).map_err(|error| {
            CompileError::Unsupported(format!(
                "package-index trust policy rejected envelope: {error}"
            ))
        })?;
        self.register_package_index_with_migration_policy(&envelope.index, migration_policy)?;
        Ok(index_hash)
    }

    /// Fetch, authenticate, and atomically register one exact package version
    /// through a host-owned transport. The response may include dependency
    /// packages, but it must contain the requested `package@version`; a
    /// mismatched response is rejected before resolver state changes.
    pub fn register_trusted_package_from_transport<T>(
        &mut self,
        package: &str,
        version: &str,
        transport: &mut T,
        policy: &crate::incremental::RegistrySignaturePolicy,
    ) -> Result<StructuralHash, CompileError>
    where
        T: crate::registry::RegistryPackageTransport,
    {
        self.register_trusted_package_from_transport_with_migration_policy(
            package,
            version,
            transport,
            policy,
            &crate::registry::RegistryPackageMigrationPolicy::Reject,
        )
    }

    /// Fetch, authenticate, and register one exact package with an explicit
    /// migration policy.
    pub fn register_trusted_package_from_transport_with_migration_policy<T>(
        &mut self,
        package: &str,
        version: &str,
        transport: &mut T,
        policy: &crate::incremental::RegistrySignaturePolicy,
        migration_policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<StructuralHash, CompileError>
    where
        T: crate::registry::RegistryPackageTransport,
    {
        crate::registry::validate_package_selector(package, version).map_err(|error| {
            CompileError::Unsupported(format!(
                "invalid package transport request '{package}@{version}': {error}"
            ))
        })?;
        let envelope = transport
            .fetch_package_index(package, version)
            .map_err(|error| {
                CompileError::Unsupported(format!(
                    "package transport failed for '{package}@{version}': {error}"
                ))
            })?;
        let canonical_index = envelope.index.clone().validate_and_canonicalize()?;
        if canonical_index.resolve(package, version).is_none() {
            return Err(CompileError::Unsupported(format!(
                "package transport response does not contain exact package '{package}@{version}'"
            )));
        }
        let index_hash = envelope.verify_with_policy(policy).map_err(|error| {
            CompileError::Unsupported(format!(
                "package transport trust policy rejected '{package}@{version}': {error}"
            ))
        })?;
        self.register_package_index_with_migration_policy(&canonical_index, migration_policy)?;
        Ok(index_hash)
    }

    /// Discover, select, authenticate, and atomically register one package.
    ///
    /// Discovery candidates are treated as untrusted metadata. The selected
    /// exact version is fetched through the host transport and must still
    /// satisfy the caller's pinned signature policy before resolver mutation.
    pub fn register_trusted_package_from_discovery<D, T>(
        &mut self,
        package: &str,
        policy: &crate::registry::RegistryPackageVersionPolicy,
        discovery: &mut D,
        transport: &mut T,
        trust: &crate::incremental::RegistrySignaturePolicy,
    ) -> Result<crate::registry::RegistryPackageResolution, CompileError>
    where
        D: crate::registry::RegistryPackageDiscovery,
        T: crate::registry::RegistryPackageTransport,
    {
        self.register_trusted_package_from_discovery_with_migration_policy(
            package,
            policy,
            discovery,
            transport,
            trust,
            &crate::registry::RegistryPackageMigrationPolicy::Reject,
        )
    }

    /// Discover, select, authenticate, and register one package with an
    /// explicit content-pinned migration policy.
    pub fn register_trusted_package_from_discovery_with_migration_policy<D, T>(
        &mut self,
        package: &str,
        policy: &crate::registry::RegistryPackageVersionPolicy,
        discovery: &mut D,
        transport: &mut T,
        trust: &crate::incremental::RegistrySignaturePolicy,
        migration_policy: &crate::registry::RegistryPackageMigrationPolicy,
    ) -> Result<crate::registry::RegistryPackageResolution, CompileError>
    where
        D: crate::registry::RegistryPackageDiscovery,
        T: crate::registry::RegistryPackageTransport,
    {
        crate::registry::validate_package_name(package).map_err(|error| {
            CompileError::Unsupported(format!(
                "invalid package discovery request '{package}': {error}"
            ))
        })?;
        policy.validate()?;
        let discovered = discovery
            .discover_package_versions(package)
            .map_err(|error| CompileError::Unsupported(error.to_string()))?;
        let version =
            crate::registry::select_discovered_package_version(package, policy, &discovered)?;
        let index_hash = self.register_trusted_package_from_transport_with_migration_policy(
            package,
            &version,
            transport,
            trust,
            migration_policy,
        )?;
        Ok(crate::registry::RegistryPackageResolution {
            package: package.to_owned(),
            version,
            index_hash,
        })
    }

    /// Return registered module keys in canonical deterministic order.
    pub fn registered_modules(&self) -> Vec<String> {
        self.registry.keys().cloned().collect()
    }

    /// Return explicit migration aliases in canonical order.
    pub fn registered_migrations(&self) -> Vec<(String, String)> {
        self.registry_migrations
            .iter()
            .map(|(from, to)| (from.clone(), to.clone()))
            .collect()
    }

    /// Remove all source-backed modules and invalidate their parsed cache
    /// entries/dependents without disturbing deterministic file IDs.
    pub fn clear_registry(&mut self) {
        let paths = self
            .registry
            .keys()
            .map(|key| registry_path(key))
            .collect::<Vec<_>>();
        for path in paths {
            self.invalidate(path);
        }
        self.registry.clear();
        self.registry_migrations.clear();
    }

    /// Restrict resolution to the explicit source registry.
    pub fn set_registry_only(&mut self, registry_only: bool) {
        self.registry_only = registry_only;
    }

    /// Restrict non-stdlib imports to a canonical filesystem subtree.
    pub fn set_sandbox_root(&mut self, path: impl Into<PathBuf>) -> Result<(), CompileError> {
        let path = path.into();
        let canonical = path.canonicalize().map_err(|error| {
            CompileError::Unsupported(format!(
                "cannot canonicalize import sandbox '{}': {error}",
                path.display()
            ))
        })?;
        self.sandbox_root = Some(canonical);
        Ok(())
    }

    /// Explicitly opt into absolute non-stdlib imports.
    pub fn set_allow_absolute_paths(&mut self, allow: bool) {
        self.allow_absolute_paths = allow;
    }

    /// Configure with defaults: current directory + stdlib.
    pub fn with_defaults(base_dir: &Path) -> Self {
        let mut resolver = Self::new();
        resolver.add_search_path(base_dir);

        // Look for stdlib relative to the compiler binary or in common locations
        let stdlib_candidates = vec![
            base_dir.join("stdlib"),
            base_dir.join("../stdlib"),
            PathBuf::from("/usr/share/meshlang/stdlib"),
            PathBuf::from("/usr/local/share/meshlang/stdlib"),
        ];

        for candidate in stdlib_candidates {
            if candidate.is_dir() {
                resolver.set_stdlib_path(candidate);
                break;
            }
        }

        resolver
    }

    /// Resolve an import path to a file path.
    pub fn resolve_path(
        &self,
        import_path: &str,
        from_file: &Path,
    ) -> Result<PathBuf, CompileError> {
        if import_path.is_empty() {
            return Err(CompileError::Unsupported(
                "empty import path is not allowed".into(),
            ));
        }
        // MeshLang import paths are logical, portable module identifiers. Do
        // not let `Path` interpret a Windows separator as a literal character
        // on Unix (or vice versa), which would make the same source resolve to
        // different modules on different hosts.
        if import_path.contains('\\') {
            return Err(CompileError::Unsupported(format!(
                "import path '{import_path}' must use '/' separators"
            )));
        }
        if import_path.contains('\0') {
            return Err(CompileError::Unsupported(format!(
                "import path contains a NUL byte: '{import_path}'"
            )));
        }

        validate_versioned_import(import_path)?;

        if let Some(key) = registry_lookup_key(import_path) {
            if self.registry.contains_key(&key) {
                return Ok(registry_path(&key));
            }
            if let Some(target) = self.registry_migrations.get(&key) {
                if self.registry.contains_key(target) {
                    return Ok(registry_path(target));
                }
                return Err(CompileError::Unsupported(format!(
                    "registry migration target '{target}' is not registered"
                )));
            }
        }

        if self.registry_only {
            return Err(CompileError::Unsupported(format!(
                "filesystem import '{import_path}' is disabled; register the module source"
            )));
        }

        // Handle std library imports: "std/acoustics.mlg" or "std.acoustics"
        if import_path.starts_with("std/") || import_path.starts_with("std.") {
            return self.resolve_stdlib(import_path);
        }

        let requested = Path::new(import_path);
        if requested.is_absolute() && !self.allow_absolute_paths {
            return Err(CompileError::Unsupported(format!(
                "absolute import '{}' is disabled; add an explicit search path or opt in",
                import_path
            )));
        }

        // Relative to importing file
        let from_dir = from_file.parent().unwrap_or(Path::new("."));
        let relative = from_dir.join(import_path);
        if relative.exists() {
            return self.validate_resolved_path(relative, import_path);
        }

        // Search paths
        for search_path in &self.search_paths {
            let candidate = search_path.join(import_path);
            if candidate.exists() {
                return self.validate_resolved_path(candidate, import_path);
            }
        }

        Err(CompileError::Unsupported(format!(
            "cannot resolve import '{}' from '{}'",
            import_path,
            from_file.display()
        )))
    }

    fn validate_resolved_path(
        &self,
        path: PathBuf,
        import_path: &str,
    ) -> Result<PathBuf, CompileError> {
        let canonical = path.canonicalize().map_err(|error| {
            CompileError::Unsupported(format!(
                "cannot canonicalize import '{}' at '{}': {error}",
                import_path,
                path.display()
            ))
        })?;
        let metadata = fs::metadata(&canonical).map_err(|error| {
            CompileError::Unsupported(format!(
                "cannot inspect import '{}' at '{}': {error}",
                import_path,
                canonical.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(CompileError::Unsupported(format!(
                "import '{}' resolves to a non-file path '{}', expected a regular module or asset",
                import_path,
                canonical.display()
            )));
        }
        if let Some(root) = &self.sandbox_root {
            let is_stdlib = self
                .stdlib_path
                .as_ref()
                .and_then(|stdlib| stdlib.canonicalize().ok())
                .is_some_and(|stdlib| canonical.starts_with(stdlib));
            if !canonical.starts_with(root) && !is_stdlib {
                return Err(CompileError::Unsupported(format!(
                    "import '{}' resolves outside sandbox '{}': '{}'",
                    import_path,
                    root.display(),
                    canonical.display()
                )));
            }
        }
        Ok(canonical)
    }

    fn resolve_stdlib(&self, import_path: &str) -> Result<PathBuf, CompileError> {
        validate_versioned_import(import_path)?;
        let stdlib = self.stdlib_path.as_ref().ok_or_else(|| {
            CompileError::Unsupported(
                "standard library not found; set MESHLANG_STDLIB or install to /usr/share/meshlang/stdlib".into()
            )
        })?;

        // Convert "std/acoustics.mlg" → "acoustics.mlg"
        // Convert "std.acoustics" → "acoustics.mlg"
        let module_path = import_path
            .strip_prefix("std/")
            .or_else(|| import_path.strip_prefix("std."))
            .unwrap_or(import_path);

        let module_path_obj = Path::new(module_path);
        if module_path_obj.is_absolute()
            || module_path_obj
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(CompileError::Unsupported(format!(
                "stdlib import '{}' escapes the standard-library root",
                import_path
            )));
        }

        let file_name = if module_path.ends_with(".mlg") {
            module_path.to_string()
        } else {
            format!("{}.mlg", module_path)
        };

        let full_path = stdlib.join(&file_name);
        if full_path.exists() {
            self.validate_resolved_path(full_path, import_path)
        } else {
            Err(CompileError::Unsupported(format!(
                "stdlib module '{}' not found at '{}'",
                module_path,
                full_path.display()
            )))
        }
    }

    /// Load and parse a module, returning its AST.
    /// Handles caching and cycle detection.
    pub fn load_module(
        &mut self,
        import_path: &str,
        from_file: &Path,
    ) -> Result<&SourceFile, CompileError> {
        let resolved = self.resolve_path(import_path, from_file)?;

        // Check for circular imports
        if self.resolving.contains(&resolved) {
            return Err(CompileError::Unsupported(format!(
                "circular import detected: '{}' → '{}'",
                from_file.display(),
                resolved.display()
            )));
        }

        // Load and parse
        self.resolving.insert(resolved.clone());
        let loaded = (|| {
            let source = self.source_for_path(&resolved)?;
            let source_hash = StructuralHash::from_bytes(source.as_bytes());
            if self
                .cache
                .get(&resolved)
                .is_some_and(|module| module.source_hash == source_hash)
            {
                self.cache_stats.hits += 1;
                return Ok(None);
            }
            self.cache_stats.misses += 1;
            self.reset_dependencies_for(&resolved);

            let filename = resolved.to_string_lossy().to_string();
            let file_id = self.file_id_for(&resolved);
            let ast = meshlang_parser::parse_with_options(
                &source,
                &filename,
                meshlang_parser::ParseOptions {
                    file_id,
                    ..meshlang_parser::ParseOptions::default()
                },
            )
            .map_err(|e| CompileError::Parse(e.message))?;

            // Recursively resolve language imports in this module.
            let sub_imports: Vec<String> = ast
                .declarations
                .iter()
                .filter_map(|declaration| match declaration {
                    Declaration::Import(import) if is_meshlang_import(&import.path) => {
                        Some(import.path.clone())
                    }
                    _ => None,
                })
                .collect();

            for sub_import in sub_imports {
                let dependency = self.resolve_path(&sub_import, &resolved)?;
                self.record_dependency(&resolved, &dependency);
                self.load_module(&sub_import, &resolved)?;
            }

            let ast_hash = ast.structural_hash().map_err(|error| {
                CompileError::Internal(format!("failed to hash imported AST: {error}"))
            })?;
            Ok(Some((source, source_hash, ast, ast_hash)))
        })();
        self.resolving.remove(&resolved);

        if let Some((source, source_hash, ast, ast_hash)) = loaded? {
            self.cache.insert(
                resolved.clone(),
                ResolvedModule {
                    path: resolved.clone(),
                    source,
                    ast,
                    source_hash,
                    ast_hash,
                },
            );
        }

        Ok(&self.cache[&resolved].ast)
    }

    fn source_for_path(&self, path: &Path) -> Result<String, CompileError> {
        if let Some(key) = registry_key_for_path(path) {
            if let Some(module) = self.registry.get(&key) {
                return Ok(module.source.clone());
            }
        }
        fs::read_to_string(path).map_err(|error| {
            CompileError::Unsupported(format!("cannot read '{}': {error}", path.display()))
        })
    }

    /// Get all loaded modules.
    pub fn modules(&self) -> impl Iterator<Item = (&PathBuf, &ResolvedModule)> {
        // HashMap iteration order is intentionally unstable. Returning a
        // sorted view keeps provenance manifests and editor diagnostics
        // deterministic across processes and platforms without changing the
        // cache's O(1) lookup representation.
        let mut modules = self.cache.iter().collect::<Vec<_>>();
        modules.sort_by_key(|(path, _)| *path);
        modules.into_iter()
    }

    /// Number of parsed modules currently retained by the resolver.
    pub fn cached_module_count(&self) -> usize {
        self.cache.len()
    }

    /// Cumulative parsed-module cache hit/miss counts.
    pub fn cache_stats(&self) -> ImportCacheStats {
        self.cache_stats
    }

    /// Invalidate one cached module. The path is canonicalized when possible.
    pub fn invalidate(&mut self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let affected = self.affected_modules(&key);
        let mut removed = self.cache.remove(&key).is_some();
        for dependent in affected {
            removed |= self.cache.remove(&dependent).is_some();
        }
        removed
    }

    /// Return the transitive set of modules affected by changing `path`, in
    /// deterministic canonical-path order. The changed module itself is not
    /// included; callers can use this to decide which downstream artifacts to
    /// rebuild before invalidating the cache.
    pub fn affected_modules(&self, path: impl AsRef<Path>) -> Vec<PathBuf> {
        let key = path
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| path.as_ref().to_path_buf());
        let mut queue = VecDeque::from([key.clone()]);
        let mut seen = HashSet::new();
        let mut affected = BTreeSet::new();
        while let Some(current) = queue.pop_front() {
            if !seen.insert(current.clone()) {
                continue;
            }
            if let Some(parents) = self.dependents.get(&current) {
                for parent in parents {
                    if affected.insert(parent.clone()) {
                        queue.push_back(parent.clone());
                    }
                }
            }
        }
        affected.into_iter().collect()
    }

    /// Remove parsed modules while retaining deterministic file IDs and stats.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.resolving.clear();
        self.dependencies.clear();
        self.dependents.clear();
    }

    /// Reset hit/miss counters without discarding parsed modules.
    pub fn reset_cache_stats(&mut self) {
        self.cache_stats = ImportCacheStats::default();
    }

    /// Collect all public declarations from a module.
    pub fn public_declarations(ast: &SourceFile) -> Vec<&Declaration> {
        ast.declarations
            .iter()
            .filter(|d| is_public_decl(d))
            .collect()
    }

    /// Flatten all `import` declarations in `file` into a single [`SourceFile`]
    /// by inlining the imported files' public declarations in place.
    ///
    /// Each imported file is included at most once (diamond imports are
    /// deduplicated; cycles are errors). `from_path` is the path of the file
    /// containing `file`, used to resolve relative import strings.
    pub fn flatten(
        &mut self,
        file: &SourceFile,
        from_path: &Path,
    ) -> Result<SourceFile, CompileError> {
        let mut included = HashSet::new();
        // Mark the root file as already included so it can never be imported
        // into itself.
        if let Ok(canonical) = from_path.canonicalize() {
            included.insert(canonical);
        } else {
            included.insert(from_path.to_path_buf());
        }
        let mut active = included.iter().cloned().collect::<Vec<_>>();
        let root_identity = file.metadata.identity.clone();
        let mut comments = file.comments.clone();
        let mut source_identities = file.source_identities.clone();
        let mut provenance = file.provenance.clone();
        let mut flattened = self.flatten_inner(
            file,
            from_path,
            &mut included,
            &mut active,
            &root_identity,
            &mut comments,
            &mut source_identities,
            &mut provenance,
        )?;
        flattened.comments = comments;
        flattened.source_identities.clear();
        flattened.extend_source_identities(source_identities);
        flattened.provenance = provenance;
        Ok(flattened)
    }

    #[allow(clippy::too_many_arguments)]
    fn flatten_inner(
        &mut self,
        file: &SourceFile,
        from_path: &Path,
        included: &mut HashSet<PathBuf>,
        active: &mut Vec<PathBuf>,
        root_identity: &SourceIdentity,
        comments: &mut Vec<Comment>,
        source_identities: &mut Vec<SourceIdentity>,
        provenance: &mut ProvenanceMap,
    ) -> Result<SourceFile, CompileError> {
        let parent_key = from_path
            .canonicalize()
            .unwrap_or_else(|_| from_path.to_path_buf());
        self.reset_dependencies_for(&parent_key);
        let mut declarations = Vec::new();

        for decl in &file.declarations {
            match decl {
                Declaration::Import(imp) if is_meshlang_import(&imp.path) => {
                    let resolved = self.resolve_path(&imp.path, from_path)?;
                    let key = resolved.canonicalize().unwrap_or(resolved.clone());

                    self.record_dependency(&parent_key, &key);

                    if let Some(cycle_start) = active.iter().position(|path| path == &key) {
                        let mut cycle = active[cycle_start..]
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>();
                        cycle.push(key.display().to_string());
                        return Err(CompileError::Unsupported(format!(
                            "circular import detected: {}",
                            cycle.join(" -> ")
                        )));
                    }

                    // Skip if already included through a completed diamond edge.
                    if included.contains(&key) {
                        continue;
                    }
                    included.insert(key.clone());
                    active.push(key);

                    // Load and parse (cached after first load).
                    let imported_ast = self.load_and_parse(&resolved)?;
                    let imported_identity = imported_ast.metadata.identity.clone();
                    // Parsed doc comments target nodes in the imported module's
                    // identity space. Flattened declarations are lowered and
                    // indexed in the root identity space, so normalize the target
                    // while retaining the comment's original span/source identity.
                    // Without this join, consumers can see an imported comment but
                    // cannot resolve it to the flattened declaration.
                    comments.extend(remap_imported_comment_targets(
                        &imported_ast.comments,
                        &imported_ast.declarations,
                        &imported_identity,
                        root_identity,
                    ));
                    source_identities.extend(imported_ast.source_identities.iter().cloned());
                    // Keep the imported module's parsed provenance alongside
                    // the expansion records below. Without these records an
                    // expanded declaration points at a source node that is
                    // absent from the flattened map, so consumers cannot
                    // resolve the provenance chain back to the original
                    // declaration.
                    provenance
                        .records
                        .extend(imported_ast.provenance.records.iter().cloned());

                    // Recurse — flatten the imported file's own imports first.
                    let flat = self.flatten_inner(
                        &imported_ast,
                        &resolved,
                        included,
                        active,
                        root_identity,
                        comments,
                        source_identities,
                        provenance,
                    )?;
                    active.pop();

                    // Filter declarations by import kind.
                    let imported_decls = filter_by_kind(flat.declarations, &imp.kind);
                    for declaration in &imported_decls {
                        let target = declaration.node_id(root_identity);
                        let source_node = declaration.node_id(&imported_identity);
                        provenance.push(ProvenanceRecord {
                            node: target,
                            sources: vec![SourceReference {
                                source: imported_identity.id,
                                node: source_node,
                                span: declaration.span(),
                            }],
                            operation: ProvenanceOperation::Expanded,
                        });
                    }
                    declarations.extend(imported_decls);
                }
                other => declarations.push(other.clone()),
            }
        }

        Ok(SourceFile {
            declarations,
            span: file.span,
            metadata: file.metadata.clone(),
            comments: file.comments.clone(),
            source_identities: file.source_identities.clone(),
            provenance: file.provenance.clone(),
        })
    }

    /// Load a file and parse it, returning an owned AST (cached after first load).
    fn load_and_parse(&mut self, path: &Path) -> Result<SourceFile, CompileError> {
        if self.resolving.contains(path) {
            return Err(CompileError::Unsupported(format!(
                "circular import: '{}'",
                path.display()
            )));
        }
        self.resolving.insert(path.to_path_buf());

        // Keep the active-set invariant even when filesystem, parser, or
        // structural-hash failures return early. Otherwise a corrected file
        // can be misreported as a circular import on the next attempt.
        let loaded = (|| {
            let source = self.source_for_path(path)?;
            let source_hash = StructuralHash::from_bytes(source.as_bytes());
            if let Some(module) = self.cache.get(path) {
                if module.source_hash == source_hash {
                    self.cache_stats.hits += 1;
                    return Ok((module.ast.clone(), None));
                }
            }
            self.cache_stats.misses += 1;
            self.reset_dependencies_for(path);

            let filename = path.to_string_lossy().to_string();
            let file_id = self.file_id_for(path);
            let ast = meshlang_parser::parse_with_options(
                &source,
                &filename,
                meshlang_parser::ParseOptions {
                    file_id,
                    ..meshlang_parser::ParseOptions::default()
                },
            )
            .map_err(|e| CompileError::Parse(e.message))?;
            let ast_hash = ast.structural_hash().map_err(|error| {
                CompileError::Internal(format!("failed to hash imported AST: {error}"))
            })?;
            let module = ResolvedModule {
                path: path.to_path_buf(),
                source,
                ast: ast.clone(),
                source_hash,
                ast_hash,
            };
            Ok((ast, Some(module)))
        })();
        self.resolving.remove(path);

        let (ast, module) = loaded?;
        if let Some(module) = module {
            self.cache.insert(path.to_path_buf(), module);
        }
        Ok(ast)
    }

    fn file_id_for(&mut self, path: &Path) -> u32 {
        if let Some(file_id) = self.file_ids.get(path) {
            return *file_id;
        }
        let file_id = self.next_file_id;
        self.next_file_id = self.next_file_id.saturating_add(1).max(1);
        self.file_ids.insert(path.to_path_buf(), file_id);
        file_id
    }

    fn record_dependency(&mut self, parent: &Path, dependency: &Path) {
        let parent = parent.to_path_buf();
        let dependency = dependency.to_path_buf();
        self.dependencies
            .entry(parent.clone())
            .or_default()
            .insert(dependency.clone());
        self.dependents
            .entry(dependency)
            .or_default()
            .insert(parent);
    }

    fn reset_dependencies_for(&mut self, parent: &Path) {
        let Some(previous) = self.dependencies.remove(parent) else {
            return;
        };
        for dependency in previous {
            let remove_entry = self.dependents.get_mut(&dependency).is_some_and(|parents| {
                parents.remove(parent);
                parents.is_empty()
            });
            if remove_entry {
                self.dependents.remove(&dependency);
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(crate) fn is_meshlang_import(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mlg"))
        || path.starts_with("std.")
}

/// Returns true for declarations that are exported when a file is imported.
/// `Mesh`, `Physics`, and `Export` blocks are file-local; they are never
/// re-exported from an `import`.
fn is_public_decl(decl: &Declaration) -> bool {
    match decl {
        Declaration::Param(_) | Declaration::Const(_) => true,
        Declaration::Geometry(g) => g.is_pub,
        Declaration::Function(f) => f.is_pub,
        Declaration::Struct(s) => s.is_pub,
        Declaration::Mesh(_)
        | Declaration::Physics(_)
        | Declaration::Export(_)
        | Declaration::Import(_) => false,
        Declaration::DesignVariable(_)
        | Declaration::Uncertainty(_)
        | Declaration::Objective(_)
        | Declaration::Constraint(_)
        | Declaration::DesignRegion(_)
        | Declaration::PhysicalRegion(_)
        | Declaration::Boundary(_)
        | Declaration::Interface(_)
        | Declaration::CoordinateSystem(_)
        | Declaration::MaterialRegion(_)
        | Declaration::GeometryField(_) => true,
    }
}

/// Returns the identifier name of a declaration, if it has one.
fn decl_name(decl: &Declaration) -> Option<&str> {
    match decl {
        Declaration::Param(p) => Some(&p.name.name),
        Declaration::Const(c) => Some(&c.name.name),
        Declaration::Geometry(g) => Some(&g.name.name),
        Declaration::Function(f) => Some(&f.name.name),
        Declaration::Struct(s) => Some(&s.name.name),
        Declaration::DesignVariable(variable) => Some(&variable.name.name),
        Declaration::Uncertainty(uncertainty) => Some(&uncertainty.name.name),
        Declaration::Objective(objective) => objective.name.as_ref().map(|name| name.name.as_str()),
        Declaration::Constraint(constraint) => {
            constraint.name.as_ref().map(|name| name.name.as_str())
        }
        Declaration::DesignRegion(region) => Some(&region.name.name),
        Declaration::PhysicalRegion(region) => Some(&region.name.name),
        Declaration::Boundary(boundary) => Some(&boundary.name.name),
        Declaration::Interface(interface) => Some(&interface.name.name),
        Declaration::CoordinateSystem(system) => Some(&system.name.name),
        Declaration::MaterialRegion(region) => Some(&region.name.name),
        Declaration::GeometryField(field) => Some(&field.name.name),
        Declaration::Mesh(_)
        | Declaration::Physics(_)
        | Declaration::Export(_)
        | Declaration::Import(_) => None,
    }
}

/// Normalize parsed comment targets when an imported module is flattened.
///
/// Parser-assigned doc-comment targets use the imported module's source
/// identity. The flattened AST is consumed with the root identity, so the
/// target must be translated to the corresponding root-space node ID while
/// the comment span remains in the imported file's coordinate space.
fn remap_imported_comment_targets(
    comments: &[Comment],
    declarations: &[Declaration],
    imported_identity: &SourceIdentity,
    root_identity: &SourceIdentity,
) -> Vec<Comment> {
    comments
        .iter()
        .cloned()
        .map(|mut comment| {
            if let Some(source_target) = comment.target {
                if let Some(declaration) = declarations
                    .iter()
                    .find(|declaration| declaration.node_id(imported_identity) == source_target)
                {
                    comment.target = Some(declaration.node_id(root_identity));
                }
            }
            comment
        })
        .collect()
}

/// Filter imported declarations according to the import kind.
fn filter_by_kind(decls: Vec<Declaration>, kind: &ImportKind) -> Vec<Declaration> {
    match kind {
        // Whole and Aliased both inline all public declarations.
        // (For Aliased, namespacing is a MeshLang resolver concern; the
        // .geo / Python backends use a flat namespace anyway.)
        ImportKind::Whole | ImportKind::Aliased(_) => {
            decls.into_iter().filter(is_public_decl).collect()
        }
        ImportKind::Selective(names) => {
            let name_set: HashSet<&str> = names.iter().map(|n| n.name.as_str()).collect();
            decls
                .into_iter()
                .filter(|d| decl_name(d).is_some_and(|n| name_set.contains(n)))
                .collect()
        }
    }
}

/// Validate the explicit module-version spelling used by imports.
///
/// A version selector is attached to the final path component, for example
/// `module@2.mlg` or `std/acoustics@1.2.mlg`. Versions are deliberately
/// numeric and exact: the resolver never silently falls back to an unversioned
/// module or chooses a different version from a directory listing.
fn validate_versioned_import(import_path: &str) -> Result<(), CompileError> {
    let at_positions = import_path
        .char_indices()
        .filter_map(|(index, character)| (character == '@').then_some(index))
        .collect::<Vec<_>>();
    if at_positions.is_empty() {
        return Ok(());
    }
    if at_positions.len() != 1 {
        return Err(CompileError::Unsupported(format!(
            "versioned import '{import_path}' must contain one @ selector"
        )));
    }

    let at = at_positions[0];
    let path = Path::new(import_path);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            CompileError::Unsupported(format!(
                "versioned import '{import_path}' must name a UTF-8 module file"
            ))
        })?;
    let file_at = file_name.find('@').ok_or_else(|| {
        CompileError::Unsupported(format!(
            "version selector in import '{import_path}' must be on the final path component"
        ))
    })?;
    let global_file_start = import_path.len() - file_name.len();
    if global_file_start + file_at != at {
        return Err(CompileError::Unsupported(format!(
            "version selector in import '{import_path}' must be on the final path component"
        )));
    }

    let stem = &file_name[..file_at];
    let version_and_extension = &file_name[file_at + 1..];
    if stem.is_empty() {
        return Err(CompileError::Unsupported(format!(
            "versioned import '{import_path}' is missing a module name"
        )));
    }
    let version = version_and_extension
        .strip_suffix(".mlg")
        .unwrap_or(version_and_extension);
    if version.is_empty()
        || version
            .split('.')
            .any(|component| component.is_empty() || !component.chars().all(|c| c.is_ascii_digit()))
    {
        return Err(CompileError::Unsupported(format!(
            "versioned import '{import_path}' requires a numeric version such as @2 or @1.2"
        )));
    }
    Ok(())
}

const REGISTRY_ROOT: &str = "<meshlang-registry>";

fn registry_path(key: &str) -> PathBuf {
    PathBuf::from(REGISTRY_ROOT).join(key)
}

fn registry_key_for_path(path: &Path) -> Option<String> {
    path.strip_prefix(REGISTRY_ROOT)
        .ok()
        .and_then(|relative| relative.to_str())
        .map(|relative| relative.replace(std::path::MAIN_SEPARATOR, "/"))
}

fn registry_lookup_key(import_path: &str) -> Option<String> {
    if import_path.contains('\\') {
        return None;
    }
    let mut normalized = import_path.to_owned();
    if normalized.starts_with('/') || normalized.contains("../") || normalized == ".." {
        return None;
    }
    if normalized.starts_with("std.") {
        normalized = format!("std/{}", &normalized[4..]);
    }
    if !normalized.ends_with(".mlg") {
        normalized.push_str(".mlg");
    }
    canonical_registry_key(&normalized).ok()
}

fn canonical_registry_key(import_path: &str) -> Result<String, CompileError> {
    if import_path.is_empty() || import_path.contains('\0') {
        return Err(CompileError::Unsupported(
            "registry module path must be non-empty and NUL-free".into(),
        ));
    }
    if import_path.contains('\\') {
        return Err(CompileError::Unsupported(format!(
            "registry module '{import_path}' must use '/' separators"
        )));
    }
    let mut key = import_path.to_owned();
    if key.starts_with("std.") {
        key = format!("std/{}", &key[4..]);
    }
    let path = Path::new(&key);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
    {
        return Err(CompileError::Unsupported(format!(
            "registry module '{import_path}' must be a relative canonical path"
        )));
    }
    if !key.ends_with(".mlg") {
        return Err(CompileError::Unsupported(format!(
            "registry module '{import_path}' must end in .mlg"
        )));
    }
    validate_versioned_import(&key)?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::{RegistrySignaturePolicy, RegistryTrustAnchor};
    use crate::registry::{
        RegistryPackageIndex, RegistryPackageIndexEnvelope, RegistryPackageIndexSignature,
        RegistryPackageManifest, RegistryPackageModule, REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
    };
    use ed25519_zebra::{SigningKey, VerificationKeyBytes};
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_resolve_stdlib_path() {
        let mut resolver = ImportResolver::new();
        let tmp = std::env::temp_dir().join("meshlang_test_stdlib");
        fs::create_dir_all(&tmp).unwrap();

        // Create a dummy module
        let mut f = fs::File::create(tmp.join("acoustics.mlg")).unwrap();
        writeln!(f, "pub const C_AIR = 343.0").unwrap();

        resolver.set_stdlib_path(&tmp);

        let result = resolver.resolve_stdlib("std/acoustics.mlg");
        assert!(result.is_ok());

        let result = resolver.resolve_stdlib("std.acoustics");
        assert!(result.is_ok());

        let result = resolver.resolve_stdlib("std/nonexistent.mlg");
        assert!(result.is_err());

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_resolve_relative() {
        let resolver = ImportResolver::new();
        let from = Path::new("/tmp/project/main.mlg");

        // Can't test actual file resolution without files on disk,
        // but verify the error message is sensible
        let result = resolver.resolve_path("nonexistent.mlg", from);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent.mlg"));
    }

    #[test]
    fn import_policy_rejects_empty_and_nul_paths() {
        let resolver = ImportResolver::new();
        let from = Path::new("/tmp/project/main.mlg");

        let empty = resolver.resolve_path("", from).unwrap_err().to_string();
        assert!(empty.contains("empty import path"), "{empty}");

        let nul = resolver
            .resolve_path("part\0.mlg", from)
            .unwrap_err()
            .to_string();
        assert!(nul.contains("NUL"), "{nul}");
    }

    #[test]
    fn filesystem_import_policy_rejects_host_specific_separators() {
        let resolver = ImportResolver::new();
        let from = Path::new("/tmp/project/main.mlg");
        let error = resolver
            .resolve_path("pkg\\part.mlg", from)
            .expect_err("backslash import must be rejected on every host")
            .to_string();
        assert!(error.contains("'/' separators"), "{error}");
    }

    #[test]
    fn absolute_imports_require_explicit_opt_in_and_respect_sandbox() {
        let directory = tempdir().unwrap();
        let module = directory.path().join("part.mlg");
        let root = directory.path().join("root");
        fs::create_dir(&root).unwrap();
        fs::write(&module, "pub const VALUE = 1\n").unwrap();
        let from = root.join("main.mlg");

        let resolver = ImportResolver::new();
        let disabled = resolver
            .resolve_path(module.to_str().unwrap(), &from)
            .unwrap_err()
            .to_string();
        assert!(disabled.contains("absolute import"), "{disabled}");

        let mut allowed = ImportResolver::new();
        allowed.set_allow_absolute_paths(true);
        allowed.set_sandbox_root(directory.path()).unwrap();
        assert_eq!(
            allowed
                .resolve_path(module.to_str().unwrap(), &from)
                .unwrap(),
            module.canonicalize().unwrap()
        );
    }

    #[test]
    fn stdlib_imports_cannot_escape_with_parent_components() {
        let directory = tempdir().unwrap();
        let outside = directory.path().join("outside.mlg");
        fs::write(&outside, "pub const ESCAPED = 1\n").unwrap();
        let stdlib = directory.path().join("stdlib");
        fs::create_dir(&stdlib).unwrap();
        let mut resolver = ImportResolver::new();
        resolver.set_stdlib_path(&stdlib);

        let error = resolver
            .resolve_path("std/../outside.mlg", &directory.path().join("main.mlg"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("escapes"), "{error}");
    }

    #[test]
    fn canonical_aliases_share_one_cached_module() {
        let directory = tempdir().unwrap();
        let module = directory.path().join("part.mlg");
        fs::write(&module, "pub const VALUE = 1\n").unwrap();
        let mut resolver = ImportResolver::new();

        resolver
            .load_module("part.mlg", &directory.path().join("main.mlg"))
            .unwrap();
        resolver
            .load_module("./part.mlg", &directory.path().join("main.mlg"))
            .unwrap();

        assert_eq!(resolver.cached_module_count(), 1);
        assert_eq!(resolver.cache_stats().misses, 1);
        assert_eq!(resolver.cache_stats().hits, 1);
    }

    #[test]
    fn dependency_impact_is_transitive_and_invalidation_cascades() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("root.mlg");
        let middle = directory.path().join("middle.mlg");
        let leaf = directory.path().join("leaf.mlg");
        fs::write(&root, "import \"middle.mlg\"\n").unwrap();
        fs::write(&middle, "import \"leaf.mlg\"\nconst MIDDLE = 1\n").unwrap();
        fs::write(&leaf, "const LEAF = 1\n").unwrap();

        let root_ast =
            meshlang_parser::parse(&fs::read_to_string(&root).unwrap(), root.to_str().unwrap())
                .unwrap();
        let mut resolver = ImportResolver::with_defaults(directory.path());
        resolver.set_sandbox_root(directory.path()).unwrap();
        resolver.flatten(&root_ast, &root).unwrap();

        let affected = resolver.affected_modules(&leaf);
        assert_eq!(
            affected,
            vec![middle.canonicalize().unwrap(), root.canonicalize().unwrap()]
        );
        assert!(resolver.invalidate(&leaf));
        assert_eq!(resolver.cached_module_count(), 0);
    }

    #[test]
    fn versioned_imports_resolve_exact_numeric_modules_without_fallback() {
        let directory = tempdir().unwrap();
        let main = directory.path().join("main.mlg");
        let exact = directory.path().join("module@2.mlg");
        let dotted = directory.path().join("module@1.2.mlg");
        fs::write(&exact, "const VALUE = 2\n").unwrap();
        fs::write(&dotted, "const VALUE = 1.2\n").unwrap();
        fs::write(directory.path().join("module.mlg"), "const VALUE = 0\n").unwrap();

        let resolver = ImportResolver::new();
        assert_eq!(
            resolver.resolve_path("module@2.mlg", &main).unwrap(),
            exact.canonicalize().unwrap()
        );
        assert_eq!(
            resolver.resolve_path("module@1.2.mlg", &main).unwrap(),
            dotted.canonicalize().unwrap()
        );
        let missing = resolver
            .resolve_path("module@3.mlg", &main)
            .unwrap_err()
            .to_string();
        assert!(missing.contains("module@3.mlg"), "{missing}");
    }

    #[test]
    fn versioned_import_policy_rejects_ambiguous_or_non_numeric_selectors() {
        let resolver = ImportResolver::new();
        let from = Path::new("/tmp/project/main.mlg");
        for (path, expected) in [
            ("module@beta.mlg", "numeric version"),
            ("module@1@2.mlg", "one @ selector"),
            ("vendor@2/module.mlg", "final path component"),
            ("@2.mlg", "missing a module name"),
        ] {
            let error = resolver.resolve_path(path, from).unwrap_err().to_string();
            assert!(error.contains(expected), "{path}: {error}");
        }
    }

    #[test]
    fn versioned_stdlib_imports_use_the_same_exact_policy() {
        let directory = tempdir().unwrap();
        let stdlib = directory.path().join("stdlib");
        fs::create_dir_all(&stdlib).unwrap();
        let versioned = stdlib.join("acoustics@2.mlg");
        fs::write(&versioned, "const C_AIR = 343.0\n").unwrap();
        let mut resolver = ImportResolver::new();
        resolver.set_stdlib_path(&stdlib);

        assert_eq!(
            resolver
                .resolve_path("std/acoustics@2.mlg", &directory.path().join("main.mlg"))
                .unwrap(),
            versioned.canonicalize().unwrap()
        );
        let error = resolver
            .resolve_path("std/acoustics@beta.mlg", &directory.path().join("main.mlg"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("numeric version"), "{error}");
    }

    #[test]
    fn directories_are_not_accepted_as_import_modules() {
        let directory = tempdir().unwrap();
        let module_dir = directory.path().join("module.mlg");
        fs::create_dir(&module_dir).unwrap();
        let resolver = ImportResolver::new();
        let error = resolver
            .resolve_path("module.mlg", &directory.path().join("main.mlg"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-file path"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn sandbox_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        let project = directory.path().join("project");
        fs::create_dir(&project).unwrap();
        let outside = directory.path().join("outside.mlg");
        fs::write(&outside, "pub const ESCAPED = 1\n").unwrap();
        symlink(&outside, project.join("link.mlg")).unwrap();

        let mut resolver = ImportResolver::new();
        resolver.set_sandbox_root(&project).unwrap();
        let error = resolver
            .resolve_path("link.mlg", &project.join("main.mlg"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside sandbox"), "{error}");
    }

    #[test]
    fn flatten_reports_the_complete_import_cycle() {
        let directory = tempdir().unwrap();
        let a_path = directory.path().join("a.mlg");
        let b_path = directory.path().join("b.mlg");
        fs::write(&a_path, "import \"b.mlg\"\nconst A = 1\n").unwrap();
        fs::write(&b_path, "import \"a.mlg\"\nconst B = 2\n").unwrap();
        let ast = meshlang_parser::parse(
            &fs::read_to_string(&a_path).unwrap(),
            a_path.to_str().unwrap(),
        )
        .unwrap();
        let mut resolver = ImportResolver::with_defaults(directory.path());
        resolver.set_sandbox_root(directory.path()).unwrap();

        let error = resolver.flatten(&ast, &a_path).unwrap_err().to_string();
        assert!(error.contains("circular import detected"), "{error}");
        assert!(
            error.contains("a.mlg") && error.contains("b.mlg"),
            "{error}"
        );
        assert!(error.contains(" -> "), "{error}");
    }

    #[test]
    fn sandbox_rejects_parent_directory_escape() {
        let directory = tempdir().unwrap();
        let project = directory.path().join("project");
        fs::create_dir(&project).unwrap();
        let main_path = project.join("main.mlg");
        let outside = directory.path().join("outside.mlg");
        fs::write(&main_path, "import \"../outside.mlg\"\n").unwrap();
        fs::write(&outside, "pub const ESCAPED = 1\n").unwrap();

        let mut resolver = ImportResolver::with_defaults(&project);
        resolver.set_sandbox_root(&project).unwrap();
        let error = resolver
            .resolve_path("../outside.mlg", &main_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside sandbox"), "{error}");
    }

    #[test]
    fn cad_imports_are_preserved_for_backend_codegen() {
        let ast = meshlang_parser::parse("import \"cabinet.step\"\n", "cad.mlg").unwrap();
        let mut resolver = ImportResolver::new();
        let flattened = resolver.flatten(&ast, Path::new("cad.mlg")).unwrap();
        assert!(matches!(
            flattened.declarations.as_slice(),
            [Declaration::Import(import)] if import.path == "cabinet.step"
        ));
    }

    #[test]
    fn parsed_module_cache_invalidates_by_structural_source_hash() {
        let directory = tempdir().unwrap();
        let module = directory.path().join("module.mlg");
        fs::write(&module, "const VALUE = 1\n").unwrap();
        let mut resolver = ImportResolver::new();

        resolver.load_and_parse(&module).unwrap();
        let first = resolver.cache[&module].source_hash;
        fs::write(&module, "const VALUE = 2\n").unwrap();
        resolver.load_and_parse(&module).unwrap();
        let second = resolver.cache[&module].source_hash;

        assert_ne!(first, second);
        assert_ne!(
            resolver.cache[&module].source_hash,
            StructuralHash::default()
        );
    }

    #[test]
    fn module_listing_is_sorted_for_deterministic_manifests() {
        let directory = tempdir().unwrap();
        let first = directory.path().join("a.mlg");
        let second = directory.path().join("b.mlg");
        fs::write(&first, "const A = 1\n").unwrap();
        fs::write(&second, "const B = 2\n").unwrap();
        let mut resolver = ImportResolver::new();

        // Deliberately load in reverse order; callers must not observe the
        // HashMap's process-randomized insertion/iteration order.
        resolver.load_and_parse(&second).unwrap();
        resolver.load_and_parse(&first).unwrap();
        let paths = resolver
            .modules()
            .map(|(path, _)| path.file_name().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                std::ffi::OsString::from("a.mlg"),
                std::ffi::OsString::from("b.mlg")
            ]
        );
    }

    #[test]
    fn flattened_imports_keep_distinct_file_spans_and_source_provenance() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("root.mlg");
        let imported = directory.path().join("part.mlg");
        fs::write(&root, "import \"part.mlg\"\n").unwrap();
        fs::write(
            &imported,
            "pub geometry Part { let solid = Box(1.0, 1.0, 1.0) }\n",
        )
        .unwrap();
        let root_ast =
            meshlang_parser::parse(&fs::read_to_string(&root).unwrap(), root.to_str().unwrap())
                .unwrap();
        let root_identity = root_ast.metadata.identity.clone();
        let mut resolver = ImportResolver::with_defaults(directory.path());
        let flat = resolver.flatten(&root_ast, &root).unwrap();
        let declaration = flat.declarations.first().unwrap();
        assert_ne!(declaration.span().file_id, 0);
        let record = flat
            .provenance
            .for_node(declaration.node_id(&root_identity))
            .next()
            .unwrap();
        assert_ne!(record.sources[0].source, root_identity.id);
        assert_eq!(record.sources[0].span.file_id, declaration.span().file_id);
        let imported_source_node = record.sources[0].node;
        let parsed_record = flat
            .provenance
            .for_node(imported_source_node)
            .find(|record| record.operation == ProvenanceOperation::Parsed)
            .expect("expanded source node must retain its parsed provenance record");
        assert_eq!(parsed_record.sources[0].node, imported_source_node);
        assert_eq!(
            parsed_record.sources[0].source, record.sources[0].source,
            "the expanded record must resolve to the imported module's source identity"
        );
    }

    #[test]
    fn flatten_preserves_imported_comments_and_selects_named_metadata() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("root.mlg");
        let imported = directory.path().join("design.mlg");
        fs::write(&root, "import { width } from \"design.mlg\"\n").unwrap();
        fs::write(
            &imported,
            "/// exported width\ndesign width { type: continuous initial: 1.0 }\n",
        )
        .unwrap();
        let root_ast =
            meshlang_parser::parse(&fs::read_to_string(&root).unwrap(), root.to_str().unwrap())
                .unwrap();
        let mut resolver = ImportResolver::with_defaults(directory.path());

        let flattened = resolver.flatten(&root_ast, &root).unwrap();

        assert!(matches!(
            flattened.declarations.as_slice(),
            [Declaration::DesignVariable(variable)] if variable.name.name == "width"
        ));
        assert_eq!(flattened.comments.len(), 1);
        assert_eq!(flattened.comments[0].text, "/// exported width");
        assert_ne!(flattened.comments[0].span.file_id, 0);
    }

    #[test]
    fn registry_registers_package_index_atomically() {
        let mut package = RegistryPackageManifest {
            name: "pkg".into(),
            version: "1".into(),
            modules: vec![
                RegistryPackageModule {
                    path: "pkg/root@1.mlg".into(),
                    source: "pub const ROOT = 1\n".into(),
                },
                RegistryPackageModule {
                    path: "pkg/part@1.mlg".into(),
                    source: "pub const PART = 2\n".into(),
                },
            ],
            dependencies: Vec::new(),
            manifest_hash: String::new(),
        };
        package.manifest_hash = package.canonical_hash().unwrap().to_string();
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![package],
        };

        let mut resolver = ImportResolver::new();
        resolver.register_package_index(&index).unwrap();
        assert_eq!(
            resolver.registered_modules(),
            vec!["pkg/part@1.mlg".to_owned(), "pkg/root@1.mlg".to_owned()]
        );
        assert!(resolver
            .load_module("pkg/root@1.mlg", Path::new("/virtual/main.mlg"))
            .is_ok());

        let mut colliding = index.clone();
        colliding.packages[0].modules.push(RegistryPackageModule {
            path: "pkg/new@1.mlg".into(),
            source: "pub const NEW = 3\n".into(),
        });
        colliding.packages[0].manifest_hash =
            colliding.packages[0].canonical_hash().unwrap().to_string();
        let error = resolver.register_package_index(&colliding).unwrap_err();
        assert!(error.to_string().contains("already registered"));
        assert!(!resolver
            .registered_modules()
            .iter()
            .any(|module| module == "pkg/new@1.mlg"));
    }

    #[test]
    fn registry_registers_only_trusted_signed_package_indexes() {
        let mut package = RegistryPackageManifest {
            name: "signed".into(),
            version: "1".into(),
            modules: vec![RegistryPackageModule {
                path: "signed/main@1.mlg".into(),
                source: "pub const SIGNED = 1\n".into(),
            }],
            dependencies: Vec::new(),
            manifest_hash: String::new(),
        };
        package.manifest_hash = package.canonical_hash().unwrap().to_string();
        let index = RegistryPackageIndex {
            schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
            packages: vec![package],
        }
        .validate_and_canonicalize()
        .unwrap();
        let signing_key = SigningKey::from([12_u8; 32]);
        let key_id = "signed-package-key-v1";
        let detached = signing_key.sign(&index.signing_payload(key_id).unwrap());
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        let hex = |bytes: &[u8]| bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        let signature = RegistryPackageIndexSignature {
            algorithm: "ed25519".into(),
            key_id: key_id.into(),
            public_key: hex(&public_key),
            signature: hex(&detached.to_bytes()),
        };
        let envelope = RegistryPackageIndexEnvelope {
            index: index.clone(),
            signature: signature.clone(),
        };
        let policy = RegistrySignaturePolicy {
            trusted_keys: vec![RegistryTrustAnchor {
                key_id: key_id.into(),
                public_key: signature.public_key.clone(),
            }],
        };
        let mut resolver = ImportResolver::new();
        assert_eq!(
            resolver
                .register_trusted_package_index(&envelope, &policy)
                .unwrap(),
            index.index_hash().unwrap()
        );
        assert_eq!(resolver.registered_modules(), vec!["signed/main@1.mlg"]);

        let mut untrusted = signature;
        untrusted.key_id = "other-key".into();
        let mut rejected = ImportResolver::new();
        assert!(rejected
            .register_trusted_package_index(
                &RegistryPackageIndexEnvelope {
                    index,
                    signature: untrusted,
                },
                &policy,
            )
            .is_err());
        assert!(rejected.registered_modules().is_empty());
    }

    #[test]
    fn registry_resolves_exact_versions_without_filesystem_fallback() {
        let mut resolver = ImportResolver::new();
        resolver
            .register_module("pkg/geometry@1.mlg", "pub const VERSION = 1\n")
            .unwrap();
        resolver
            .register_module("pkg/geometry@2.mlg", "pub const VERSION = 2\n")
            .unwrap();

        let from = Path::new("/virtual/main.mlg");
        assert_eq!(
            resolver.resolve_path("pkg/geometry@2.mlg", from).unwrap(),
            registry_path("pkg/geometry@2.mlg")
        );
        assert!(resolver.resolve_path("pkg/geometry@3.mlg", from).is_err());
        assert!(resolver.resolve_path("pkg/geometry.mlg", from).is_err());
        assert_eq!(
            resolver.registered_modules(),
            vec![
                "pkg/geometry@1.mlg".to_owned(),
                "pkg/geometry@2.mlg".to_owned()
            ]
        );
    }

    #[test]
    fn registry_loads_nested_modules_and_preserves_provenance() {
        let mut resolver = ImportResolver::new();
        resolver
            .register_module(
                "pkg/root@1.mlg",
                "import \"pkg/part@1.mlg\"\n geometry Root { let body = part }\n",
            )
            .unwrap();
        resolver
            .register_module(
                "pkg/part@1.mlg",
                "/// registry part\npub geometry Part { let body = Box(1.0, 1.0, 1.0) }\n",
            )
            .unwrap();

        let root = resolver
            .load_module("pkg/root@1.mlg", Path::new("/virtual/main.mlg"))
            .unwrap()
            .clone();
        assert!(root
            .declarations
            .iter()
            .any(|declaration| matches!(declaration, Declaration::Import(_))));
        assert_eq!(
            resolver.cache_stats(),
            ImportCacheStats { hits: 0, misses: 2 }
        );

        let flattened = resolver
            .flatten(&root, &registry_path("pkg/root@1.mlg"))
            .unwrap();
        assert_eq!(flattened.comments.len(), 1);
        assert_eq!(flattened.comments[0].text, "/// registry part");
        assert!(!flattened.source_identities.is_empty());
    }

    #[test]
    fn registry_rejects_host_dependent_or_duplicate_keys() {
        let mut resolver = ImportResolver::new();
        for invalid in [
            "../escape.mlg",
            "/absolute.mlg",
            "pkg\\windows.mlg",
            "pkg/no-extension",
        ] {
            assert!(resolver.register_module(invalid, "").is_err(), "{invalid}");
        }
        resolver.register_module("pkg/value.mlg", "").unwrap();
        assert!(resolver.register_module("pkg/value.mlg", "").is_err());
        assert!(resolver.register_module("std.acoustics@1.mlg", "").is_ok());
        assert_eq!(
            resolver
                .resolve_path("std/acoustics@1.mlg", Path::new("/virtual/main.mlg"))
                .unwrap(),
            registry_path("std/acoustics@1.mlg")
        );
        resolver.clear_registry();
        assert!(resolver
            .resolve_path("pkg/value.mlg", Path::new("/virtual/main.mlg"))
            .is_err());
        resolver
            .register_module("pkg/registered.mlg", "pub const VALUE = 1\n")
            .unwrap();
        resolver.set_registry_only(true);
        assert!(resolver
            .resolve_path("pkg/registered.mlg", Path::new("/virtual/main.mlg"))
            .is_ok());
        let error = resolver
            .resolve_path("pkg/filesystem.mlg", Path::new("/virtual/main.mlg"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("filesystem import"), "{error}");
    }
}
