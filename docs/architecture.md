# FlyRigLoader Architecture Overview

This document summarizes how the refactored FlyRigLoader package is organized, how the
runtime boot flow progresses from configuration to data products, and where
maintainers can extend the system safely. Cross-references to topical guides are
provided throughout so feature documentation remains synchronized.

## High-Level System Layers

FlyRigLoader is organized into layered modules that separate configuration,
discovery, loading, transformation, and orchestration concerns:

- **`flyrigloader.config`** – Defines Pydantic models, builders, validators, and
  versioning helpers for configuration management.【F:src/flyrigloader/config/__init__.py†L1-L18】【F:src/flyrigloader/config/models.py†L1-L46】
- **`flyrigloader.discovery`** – Provides manifest generation utilities and file
  discovery helpers with metadata extraction support.【F:src/flyrigloader/discovery/files.py†L1-L189】
- **`flyrigloader.io`** – Houses loader implementations, transformation
  pipelines, and column configuration helpers used once files are selected.【F:src/flyrigloader/io/loaders.py†L1-L187】【F:src/flyrigloader/io/transformers.py†L1-L210】
- **`flyrigloader.registries`** – Supplies registry singletons and capability
  queries that make the loader and schema layers pluggable.【F:src/flyrigloader/registries/__init__.py†L1-L120】
- **`flyrigloader.api`** – Coordinates dependency providers, manifest discovery,
  and data loading in a backwards-compatible façade consumed by downstream
  tools.【F:src/flyrigloader/api/_core.py†L1-L120】【F:src/flyrigloader/api/dependencies.py†L1-L180】
- **`flyrigloader` package root** – Exposes logging setup and shared utilities so
  every layer can emit structured diagnostics consistently.【F:src/flyrigloader/__init__.py†L1-L120】

Additional modules such as `flyrigloader.exceptions` and `flyrigloader.utils`
provide domain-specific errors, path helpers, and test scaffolding that support
all layers.【F:src/flyrigloader/exceptions.py†L1-L147】【F:src/flyrigloader/utils/paths.py†L1-L160】

Refer to the [Configuration Guide](configuration_guide.md) and
[Extension Guide](extension_guide.md) for in-depth coverage of specific layers.

## Runtime Boot Flow

The primary runtime sequence is designed to fail fast when configuration or file
assumptions are violated while keeping data loading memory efficient.

```
┌─────────────────────┐
│ initialize_logger() │  ⇢  Sets up Loguru sinks & bridges stdlib logging.【F:src/flyrigloader/__init__.py†L122-L274】
└─────────┬───────────┘
          │
          ▼
┌────────────────────────────┐
│ create_config(...) / load │  ⇢  Build or load validated config models.【F:src/flyrigloader/config/builder.py†L1-L120】【F:src/flyrigloader/api/config.py†L40-L153】
└──────────┬─────────────────┘
           │
           ▼
┌───────────────────────────────┐
│ discover_experiment_manifest │  ⇢  Produce manifest with metadata & audits.【F:src/flyrigloader/api/_core.py†L123-L256】
└─────────────┬─────────────────┘
              │
              ▼
┌──────────────────────────┐
│ load_dataset_files(...) │  ⇢  Resolve loaders & stream files from registry.【F:src/flyrigloader/api/_core.py†L420-L610】【F:src/flyrigloader/io/loaders.py†L70-L187】
└───────────┬──────────────┘
            │
            ▼
┌────────────────────────────┐
│ transform_to_dataframe(...)│ ⇢  Apply column models & return tidy frame.【F:src/flyrigloader/io/transformers.py†L77-L210】
└────────────────────────────┘
```

Each stage records progress through `loguru` so analysts can correlate file
operations with downstream results. Exceptions derived from `FlyRigLoaderError`
propagate without silent fallbacks, aligning with the fail-fast policy.【F:src/flyrigloader/exceptions.py†L1-L147】

## Logging Flow Notes

The logging infrastructure intentionally centralizes configuration and maintains
verbosity by default:

- `initialize_logger()` installs both console and rotating file sinks, then
  bridges stdlib logging through an `InterceptHandler` so third-party modules are
  captured without extra plumbing.【F:src/flyrigloader/__init__.py†L122-L248】
- Console output defaults to `INFO` while file logs capture `DEBUG` for full
  traceability during investigations; contributors can override these levels via
  `LoggerConfig` when diagnosing issues.【F:src/flyrigloader/__init__.py†L250-L356】
- All API entry points include explicit info/debug statements describing
  parameters, injected dependencies, and selection results to keep audits
  readable.【F:src/flyrigloader/api/_core.py†L176-L213】【F:src/flyrigloader/api/_core.py†L308-L376】
- When developing new features, prefer raising contextual exceptions over
  suppressing errors so the log stream accurately reflects pipeline health.

## Extension and Injection Points

FlyRigLoader encourages extension through explicit registries and dependency
providers:

- **Loader & schema registries** – Register new loaders or validation schemas via
  `LoaderRegistry`/`SchemaRegistry`, including priority controls and capability
  queries for discovery-time diagnostics.【F:src/flyrigloader/registries/__init__.py†L1-L120】【F:src/flyrigloader/api/registry.py†L1-L98】
- **Dependency provider overrides** – Use `use_dependency_provider()` to scope
  temporary overrides or `set_dependency_provider()` for longer-lived swaps
  during specialized deployments. Both approaches reuse the same dependency
  façade while keeping overrides explicit.【F:src/flyrigloader/api/dependencies.py†L300-L332】
- **Configuration builders** – Compose validated configs programmatically through
  `create_config()` and helper functions, ensuring downstream modules always see
  typed models and consistent defaults.【F:src/flyrigloader/config/builder.py†L1-L120】
- **Manifest hooks** – `discover_experiment_manifest` exposes metadata buckets
  and audit trails that downstream tools can enrich or filter before loading
  data.【F:src/flyrigloader/api/_core.py†L123-L320】

When extending the system, update this architecture summary alongside the
feature-specific guide so future contributors can reason about new seams quickly.

## Maintenance Checklist

Use the following checklist when merging architectural changes:

1. Update module responsibilities and diagrams in this document to reflect new
   flows.
2. Verify README links point to the latest topical guides and add new sections as
   needed.
3. Document new configuration fields or defaults in the
   [Configuration Guide](configuration_guide.md) and reference them from relevant
   API docstrings.
4. Describe new extension points in the [Extension Guide](extension_guide.md) and
   add usage examples or logging expectations.
5. Ensure logging behavior is covered in release notes and `initialize_logger`
   guidance when verbosity levels or sinks change.

For contributor workflow expectations, see [CONTRIBUTING.md](../CONTRIBUTING.md).
