# `flyrigloader.api` Module Inventory

This inventory captures the current structure of the public API facade after the Phase 2 refactor. It is designed to make the import surface obvious and to highlight which submodules own which behaviours so future changes can be planned without spelunking through the entire package.

## Package layout

| Module | Responsibility | Key exports |
| --- | --- | --- |
| `flyrigloader.api.__init__` | Backwards-compatible façade that re-exports the public API. Delegates to the focused helper modules and keeps orchestration logic. | High level helpers such as `discover_experiment_manifest`, `load_experiment_files`, registry utilities, plus re-exports from other submodules. |
| `flyrigloader.api.configuration` | Centralises configuration source validation, loading, and schema coercion utilities. Provides pure functions that operate on the dependency provider. | `validate_config_parameters`, `resolve_config_source`, `load_and_validate_config`, `coerce_config_for_version_validation`. |
| `flyrigloader.api.dependencies` | Defines dependency-provider protocols and the lazy `DefaultDependencyProvider`. Responsible for wiring config/discovery/io utilities and exposing `set/get/reset_dependency_provider`. | `DefaultDependencyProvider`, the provider protocols, and the provider management helpers. |
| `flyrigloader.api.kedro` | Owns the optional Kedro integration, including capability checks and dataset factory creation. Kedro-specific imports live here so plain users avoid import-time warnings. | `check_kedro_available`, `create_kedro_dataset`, `FlyRigLoaderDataSet` (when available). |

## Dependency notes

* `configuration` depends on `dependencies` for the provider interface and is logging-heavy so that misconfigurations are loud.
* `kedro` consumes both `configuration` and `dependencies`, ensuring Kedro-only code paths are gated behind `check_kedro_available()`.
* `__init__` remains the compatibility surface. When moving functionality, prefer to shift implementation into a submodule and import it here to keep historical imports working.

## Testing guardrails

* `tests/flyrigloader/test_api_surface.py` verifies that importing `flyrigloader.api` without Kedro installed does **not** emit warnings and that `check_kedro_available()` fails loudly when Kedro is missing.
* Existing manifest, registry, and Kedro integration tests continue to exercise the re-exported functions, protecting against accidental regressions while the façade is decomposed.

## Follow-up opportunities

* Split manifest and legacy loader helpers into their own modules once the dependency graph is mapped.
* Revisit documentation to link directly to these modules so contributors know where to patch logic.
