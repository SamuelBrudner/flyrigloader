# Path Validation Audit

## Overview
The audit focused on understanding how `path_traversal_protection` influences
configuration loading and where sensitive-root decisions occur. The validator is
invoked by `path_existence_validator`, which in turn is used inside
`ProjectConfig.validate_directories` during configuration loading.

## Call-Site Inventory
- `ProjectConfig.directories`: every configured directory is validated via
  `path_existence_validator`, meaning project-level configuration is the primary
  consumer of traversal protection today.
- Programmatic builders (e.g., `create_project_config`) defer to the same
  validation logic through the `ProjectConfig` model, so no additional entry
  points bypass the guardrails.

## Logging Instrumentation
- Added DEBUG logs to record the effective deny roots being evaluated for a
  given path.
- Added explicit DEBUG logs when a path is permitted by a configured allow root
  so that opt-in policies are visible during troubleshooting.
- Permission denials continue to emit ERROR-level messages with the exact root
  that triggered the block.

## Opportunities for Future Work
- Dataset-level or experiment-level overrides currently rely on the project
  policy; future work could expose dedicated policies for those scopes if
  required.
- Consider centralizing audit summaries like this under `docs/audits/` for other
  security-sensitive validators to keep the rationale close to the code.
