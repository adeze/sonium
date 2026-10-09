use meshlang_compiler::{
    imports::ImportResolver, IncrementalCompiler, RegistryPackageIndex, RegistryPackageManifest,
    RegistryPackageMigrationModule, RegistryPackageMigrationPlan, RegistryPackageMigrationPolicy,
    RegistryPackageModule, REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
    REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION,
};
use std::path::Path;

fn manifest(version: &str, source_suffix: &str) -> RegistryPackageManifest {
    let mut package = RegistryPackageManifest {
        name: "demo".into(),
        version: version.into(),
        modules: vec![
            RegistryPackageModule {
                path: format!("demo/root@{version}.mlg"),
                source: format!("pub const VERSION = {version}\n"),
            },
            RegistryPackageModule {
                path: format!("demo/util@{version}.mlg"),
                source: format!("pub const SUFFIX = {source_suffix}\n"),
            },
        ],
        dependencies: Vec::new(),
        manifest_hash: String::new(),
    };
    package.manifest_hash = package.canonical_hash().unwrap().to_string();
    package
}

fn fixture() -> (RegistryPackageIndex, RegistryPackageMigrationPlan) {
    let old = manifest("1", "10");
    let new = manifest("2", "20");
    let index = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![old.clone(), new.clone()],
    };
    let plan = RegistryPackageMigrationPlan {
        schema_version: REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION,
        package: "demo".into(),
        from_version: "1".into(),
        to_version: "2".into(),
        from_manifest_hash: old.manifest_hash,
        to_manifest_hash: new.manifest_hash,
        modules: vec![
            RegistryPackageMigrationModule {
                from: "demo/root@1.mlg".into(),
                to: "demo/root@2.mlg".into(),
            },
            RegistryPackageMigrationModule {
                from: "demo/util@1.mlg".into(),
                to: "demo/util@2.mlg".into(),
            },
        ],
    };
    (index, plan)
}

#[test]
fn migration_plan_is_content_pinned_and_deterministic() {
    let (index, mut plan) = fixture();
    let aliases = plan.validate_against(&index).unwrap();
    assert_eq!(
        aliases,
        vec![
            ("demo/root@1.mlg".into(), "demo/root@2.mlg".into()),
            ("demo/util@1.mlg".into(), "demo/util@2.mlg".into()),
        ]
    );

    plan.modules.reverse();
    assert_eq!(plan.validate_against(&index).unwrap(), aliases);

    let replacement = if plan.to_manifest_hash.starts_with('0') {
        '1'
    } else {
        '0'
    };
    plan.to_manifest_hash
        .replace_range(0..1, &replacement.to_string());
    assert!(plan.validate_against(&index).is_err());
}

#[test]
fn default_registration_preserves_exact_versions_without_aliases() {
    let (index, _) = fixture();
    let mut resolver = ImportResolver::new();
    resolver.register_package_index(&index).unwrap();

    assert_eq!(
        resolver.registered_migrations(),
        Vec::<(String, String)>::new()
    );
    assert_eq!(
        resolver
            .resolve_path("demo/root@1.mlg", Path::new("/virtual/main.mlg"))
            .unwrap()
            .to_string_lossy(),
        "<meshlang-registry>/demo/root@1.mlg"
    );
    assert!(resolver
        .resolve_path("demo/root.mlg", Path::new("/virtual/main.mlg"))
        .is_err());
}

#[test]
fn declared_migration_routes_old_exact_imports_to_new_modules() {
    let (index, plan) = fixture();
    let mut resolver = ImportResolver::new();
    resolver
        .register_package_index_with_migration_policy(
            &index,
            &RegistryPackageMigrationPolicy::AllowDeclared(plan),
        )
        .unwrap();

    assert_eq!(
        resolver.registered_modules(),
        vec!["demo/root@2.mlg".to_owned(), "demo/util@2.mlg".to_owned()]
    );
    assert_eq!(
        resolver.registered_migrations(),
        vec![
            ("demo/root@1.mlg".to_owned(), "demo/root@2.mlg".to_owned()),
            ("demo/util@1.mlg".to_owned(), "demo/util@2.mlg".to_owned()),
        ]
    );
    assert_eq!(
        resolver
            .resolve_path("demo/root@1.mlg", Path::new("/virtual/main.mlg"))
            .unwrap()
            .to_string_lossy(),
        "<meshlang-registry>/demo/root@2.mlg"
    );
}

#[test]
fn invalid_declared_migration_leaves_resolver_unmodified() {
    let (index, mut plan) = fixture();
    plan.modules[0].to = "demo/missing@2.mlg".into();
    let mut resolver = ImportResolver::new();
    assert!(resolver
        .register_package_index_with_migration_policy(
            &index,
            &RegistryPackageMigrationPolicy::AllowDeclared(plan),
        )
        .is_err());
    assert!(resolver.registered_modules().is_empty());
    assert!(resolver.registered_migrations().is_empty());
}

#[test]
fn incremental_facade_accepts_declared_migration_before_preview() {
    let directory = tempfile::tempdir().unwrap();
    let (index, plan) = fixture();
    let mut compiler = IncrementalCompiler::new(directory.path());
    compiler
        .register_package_index_with_migration_policy(
            &index,
            &RegistryPackageMigrationPolicy::AllowDeclared(plan),
        )
        .unwrap();
    let preview = compiler
        .preview_canonical_ir(
            "import \"demo/root@1.mlg\"\n geometry Main { let solid = Box(1.0, 1.0, 1.0) }",
            "main.mlg",
        )
        .unwrap();
    assert!(!preview.canonical_ir.entities.is_empty());
}
