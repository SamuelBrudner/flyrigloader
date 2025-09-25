# Contributing to FlyRigLoader

FlyRigLoader emphasizes fail-fast behavior, traceable logging, and well-curated
documentation. The following guidelines help keep the README, architecture
overview, and topical guides aligned with the evolving feature set.

## Workflow Expectations

- **Plan with tests first** – adopt a red/green/refactor loop and design tests to
  confirm observable results rather than implementation details.
- **Fail loud and fast** – validate inputs aggressively and raise
  `FlyRigLoaderError` subclasses when assumptions break instead of falling back to
  ambiguous states.【F:src/flyrigloader/exceptions.py†L1-L147】
- **Prefer explicit configuration** – use helper builders from
  `flyrigloader.config` instead of ad-hoc dictionaries so validation errors appear
  early.【F:src/flyrigloader/config/builder.py†L1-L120】
- **Embrace verbose logging** – call `initialize_logger()` during manual testing
  and ensure new code paths emit contextual `logger.info`/`logger.debug` messages
  rather than muting diagnostics.【F:src/flyrigloader/__init__.py†L122-L248】【F:src/flyrigloader/api/_core.py†L176-L213】
- **Scope dependency overrides** – the dependency provider is a module-level
  singleton. Use `use_dependency_provider()` when overriding in application code
  and rely on the `dependency_provider_state_guard` pytest fixture to reset state
  in tests so concurrent workflows remain isolated.【F:src/flyrigloader/api/dependencies.py†L300-L332】【F:tests/conftest.py†L120-L140】

## Documentation Alignment Checklist

Run through this checklist before merging any change that adds behavior, alters
configuration, or adjusts logging:

1. **README** – confirm feature lists, usage examples, and logging instructions
   still match reality. Add cross-links to new topical docs when applicable.
2. **Architecture Overview** – update module responsibilities, diagrams, and the
   maintenance checklist in `docs/architecture.md` to reflect new flow control.
3. **Configuration Guide** – document new schema fields, defaults, or resolution
   rules in `docs/configuration_guide.md`.
4. **Extension Guide** – describe new registry hooks or plugin patterns in
   `docs/extension_guide.md`, including logging expectations for third-party
   integrations.
5. **Changelogs / Release notes** – call out logging level changes or new sinks
   so operators can adjust verbosity settings promptly.

## Logging Standards

- Default to `INFO` level console messages and `DEBUG` level file messages unless
  there is a compelling reason to diverge. Adjust levels temporarily for tests by
  constructing a custom `LoggerConfig`.
- Include operation identifiers and key parameters in log messages so correlating
  file activity with downstream data frames remains straightforward.【F:src/flyrigloader/api/_core.py†L176-L213】【F:src/flyrigloader/api/_core.py†L308-L376】
- Never swallow exceptions; raise with context and allow the logger to capture
  stack traces.
- When adding third-party integrations, verify they honor the Loguru intercept so
  logs stay centralized. If a library requires custom handling, document it in the
  architecture guide.

## Configuration Guidance

- Use `create_config()` or the YAML loaders from `flyrigloader.api.config` to
  ensure inputs are validated and versioned.【F:src/flyrigloader/api/config.py†L40-L153】
- Keep sensitive paths out of committed configuration files; only reference them
  through validated directory fields.
- Avoid introducing environment-variable driven logic unless dealing with secret
  material.

## Pull Request Checklist

- [ ] Tests cover observable behavior and fail before the fix is applied.
- [ ] Logging statements clearly describe the new workflow and remain consistent
      with verbosity standards.
- [ ] Documentation (README, architecture, configuration, extension guides) is
      updated or explicitly confirmed as unchanged.
- [ ] Added extension points are registered and documented with usage examples.
- [ ] No unexpected fallbacks or silent error handling paths remain.

Adhering to this checklist keeps the codebase predictable and the documentation
authoritative for both maintainers and downstream data analysts.
