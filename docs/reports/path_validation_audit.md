# Path Validation Audit

Date: 2025-09-24

This note captures the current consumers of the path traversal safeguards and
highlights the surface area affected by the restrictive defaults.

## Key Components

- `path_traversal_protection`: the core validator that enforces traversal
  checks, URL blocking, and sensitive-root restrictions.
- `path_existence_validator`: wraps the traversal checks with optional
  filesystem existence validation and the pytest-aware bypass.
- `ProjectConfig` directory validation: the only configuration model that
  invokes `path_existence_validator`, thereby inheriting the sensitive-root
  blacklist.

## Call Sites

| Location | Purpose | Notes |
| --- | --- | --- |
| `src/flyrigloader/config/models.py` (`ProjectConfig.apply_directory_security`) | Validates every configured directory on model construction | Blocks `/etc`, `/var`, `/dev`, `/proc`, `/sys`, `/root`, `/bin`, `/sbin` by default. |
| `src/flyrigloader/config/validators.py` (`path_existence_validator`) | Exported for direct use by other modules and extensions | Currently not consumed elsewhere in the repository, but part of the public API referenced by docs. |

## Observations

1. The sensitive-root guard is effectively global because `ProjectConfig`
   instantiates it for every directory field. No other configuration model
   overrides the behaviour.
2. Legitimate deployments under `/var` or `/etc` fail even in tests because the
   blacklist is evaluated before filesystem checks.
3. Third-party extensions that import `path_existence_validator` inherit the
   same hard-coded restrictions with no configuration hook.

## Recommended Adjustments

- Introduce a configuration-backed policy so that projects can explicitly allow
  certain roots without disabling traversal detection.
- Ensure logging clearly indicates when policy overrides allow a path that
  would otherwise be blocked.
- Document the manual nature of logger initialization to avoid confusion during
  policy debugging (log directories are only created when
  `initialize_logger()` runs).
