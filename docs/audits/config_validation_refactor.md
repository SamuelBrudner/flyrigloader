# Configuration Validator Refactor Notes

## Validator Clusters

| Cluster | Module | Responsibilities | Key Validators |
| ------- | ------ | ---------------- | -------------- |
| Storage | `config/validators/storage.py` | Path hygiene, security policy enforcement, filesystem existence checks | `PathSecurityPolicy`, `path_traversal_protection`, `path_existence_validator` |
| Discovery | `config/validators/discovery.py` | Regex compilation guardrails, pattern safety heuristics, date parsing | `pattern_validation`, `date_format_validator` |
| Versioning | `config/validators/versioning.py` | Semantic version checks, compatibility gates, structure probes | `validate_version_format`, `validate_version_compatibility`, `validate_config_version`, `validate_config_with_version` |

## Model/Validator Co-occurrence

- **ProjectConfig**
  - `schema_version` -> `validate_config_version`
  - `directories` -> `PathSecurityPolicy`, `path_existence_validator`
  - `ignore_substrings` / `extraction_patterns` -> regex compilation (local)
- **DatasetConfig**
  - `schema_version` -> `validate_config_version`
  - `metadata.extraction_patterns` -> `pattern_validation`
  - `dates_vials` -> date heuristics (still in model; candidates for Discovery cluster)
- **ExperimentConfig**
  - `schema_version` -> `validate_config_version`
  - `datasets` / `parameters` -> pure Pydantic validation (no shared validators yet)
- **YAML loader**
  - Extraction pattern normalization -> `pattern_validation`
  - Legacy adapter bootstrap -> `validate_config_with_version`

## High-churn Observations

`git log --pretty=format: --name-only src/flyrigloader/config | sort | uniq -c | sort -nr` highlights:

- `yaml_config.py` changed ~20 times.
- `validators.py` (pre-refactor) changed ~9 times.
- `models.py` changed ~8 times.

These hot spots are the coupling points the refactor splits into modules, enabling more targeted ownership.

## Follow-up Targets

- Consider extracting date parsing helpers used inside `DatasetConfig` into `discovery` cluster for symmetry.
- Formalize structured logging schema (`validator=...`, `field=...`) so downstream tooling can parse validation reports.
