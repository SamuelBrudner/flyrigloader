# FlyRigLoader API Module Inventory

## Overview

The legacy ``flyrigloader.api`` module previously contained Kedro adapters, configuration
helpers, registry utilities, manifest orchestration, and legacy compatibility shims in a
single 2,600-line file. This refactor splits those responsibilities into focused
submodules to reduce import side effects and make ownership clearer:

* ``flyrigloader.api.kedro`` – optional Kedro integration, including
  ``check_kedro_available`` and ``create_kedro_dataset``.
* ``flyrigloader.api.config`` – configuration loading/validation helpers and path
  utilities shared across the facade.
* ``flyrigloader.api.manifest`` – discovery and validation logic for manifest-driven
  workflows.
* ``flyrigloader.api.registry`` – registry introspection helpers used by Kedro hooks and
  plugin discovery.
* ``flyrigloader.api.dependencies`` – dependency provider protocols and factory used for
  dependency injection.
* ``flyrigloader.api._core`` – remaining facade orchestration functions that coordinate
  dependency providers, loading, and transformation.

``flyrigloader.api.__init__`` now re-exports the public surface from these modules to
preserve the historical import contract while allowing deeper modules to be loaded on
demand.

## Natural Seams for Lazy Imports

* Kedro support is isolated behind ``api.kedro`` so importing ``flyrigloader.api`` no
  longer triggers logging when Kedro is absent. Callers explicitly invoke
  ``check_kedro_available()`` to fail fast instead of relying on import-time warnings.
* Registry and manifest helpers can be imported independently without importing Kedro.
  ``api.__init__`` uses simple attribute bindings, so submodules are only loaded when
  their exports are accessed.

## Proposed Follow-on Split Outline

The remaining ``api._core`` file groups related workflows that can be migrated in
follow-up slices roughly scoped for 30-minute development tasks:

1. **Config extraction (serial prerequisite)** – Move ``_load_and_validate_config`` and
   related helpers fully into ``api.config`` so `_core`` only orchestrates calls.
2. **Manifest orchestration (parallel)** – Relocate ``discover_experiment_manifest`` and
   ``validate_manifest`` into ``api.manifest`` with thin wrappers in `_core` for
   backwards compatibility.
3. **Loader orchestration (parallel)** – Split ``load_experiment_files``,
   ``load_dataset_files``, and ``process_experiment_data`` into an ``api.loader`` module
   that coordinates config + manifest helpers.

These steps will further shrink ``api._core`` and make it feasible to unit test each
concern independently.
