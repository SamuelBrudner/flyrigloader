# FlyRigLoader Documentation Index

**Last Updated**: 2025-09-30  
**Version**: 2.0.0-dev

## Overview

This index provides a comprehensive map of all FlyRigLoader documentation, organized by topic and purpose. All documentation reflects the semantic model improvements implemented following the test suite review.

---

## 🎯 Quick Start

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](../README.md) | Project overview and installation | All users |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute tutorial | New users |
| [EXAMPLES.md](EXAMPLES.md) | Common usage patterns | All users |

---

## 📋 Core Concepts (NEW - Semantic Model Clarifications)

These documents resolve ambiguities identified in the test suite review:

### Pattern and Filter Behavior

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [PATTERN_PRECEDENCE.md](PATTERN_PRECEDENCE.md) | ✅ Pattern type hierarchy<br>✅ Filter application order<br>✅ OR vs AND logic<br>✅ Edge cases | **NEW** |
| [METADATA_MERGE.md](METADATA_MERGE.md) | ✅ Metadata source precedence<br>✅ Merge algorithm<br>✅ Type preservation<br>✅ Empty value handling | **NEW** |

### Data Transformation

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [DIMENSION_HANDLING.md](DIMENSION_HANDLING.md) | ✅ 1D/2D/3D array behavior<br>✅ Special handler semantics<br>✅ Time alignment rules<br>✅ Storage patterns | **NEW** |

---

## 🏗️ Architecture & Design (UPDATED)

### API Simplification

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [LEGACY_CONFIG_ADAPTER_REMOVAL.md](LEGACY_CONFIG_ADAPTER_REMOVAL.md) | ✅ Removal rationale<br>✅ Migration strategy<br>✅ Timeline<br>✅ User migration guide | **NEW** |
| [API_SIMPLIFICATION.md](API_SIMPLIFICATION.md) | ⏳ DiscoveryOptions dataclass<br>⏳ Parameter consolidation<br>⏳ Function signature changes | **PLANNED** |
| [ERROR_TAXONOMY.md](ERROR_TAXONOMY.md) | ⏳ Exception hierarchy<br>⏳ Error handling contracts<br>⏳ Recovery strategies | **PLANNED** |

---

## 📚 Reference Documentation

### Configuration

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [CONFIG_SCHEMA.md](CONFIG_SCHEMA.md) | YAML structure, field definitions | ⏳ UPDATE NEEDED |
| [PYDANTIC_MODELS.md](PYDANTIC_MODELS.md) | ProjectConfig, DatasetConfig, ExperimentConfig | ⏳ UPDATE NEEDED |
| [CONFIG_VALIDATION.md](CONFIG_VALIDATION.md) | Validation rules, error messages | ⏳ UPDATE NEEDED |

### API Reference

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [API_REFERENCE.md](API_REFERENCE.md) | Complete function signatures | ⏳ UPDATE NEEDED |
| [DISCOVERY_API.md](DISCOVERY_API.md) | File discovery functions | ⏳ UPDATE NEEDED |
| [TRANSFORMATION_API.md](TRANSFORMATION_API.md) | DataFrame transformation | ⏳ UPDATE NEEDED |

### Data Formats

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [PICKLE_FORMAT.md](PICKLE_FORMAT.md) | Supported pickle formats | ✅ Current |
| [DATAFRAME_SCHEMA.md](DATAFRAME_SCHEMA.md) | Output DataFrame structure | ⏳ UPDATE NEEDED |
| [COLUMN_CONFIG.md](COLUMN_CONFIG.md) | Column configuration schema | ✅ Current |

---

## 🔧 Developer Guides

### Testing

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Test structure, fixtures, best practices | ✅ Current |
| [TEST_COVERAGE.md](TEST_COVERAGE.md) | Coverage requirements, reports | ✅ Current |

### Contributing

| Document | Topics Covered | Status |
|----------|----------------|--------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution workflow | ✅ Current |
| [CODE_STYLE.md](CODE_STYLE.md) | Style guide, linting rules | ✅ Current |

---

## 📊 Migration & Upgrade Guides

### Version Migrations

| Document | From → To | Status |
|----------|-----------|--------|
| [MIGRATE_V1_TO_V2.md](migrations/MIGRATE_V1_TO_V2.md) | v1.x → v2.0 | ⏳ IN PROGRESS |
| [BREAKING_CHANGES_V2.md](migrations/BREAKING_CHANGES_V2.md) | v2.0 breaking changes | ⏳ IN PROGRESS |

### Feature-Specific Migrations

| Document | Migration Topic | Status |
|----------|-----------------|--------|
| [DICT_TO_PYDANTIC.md](migrations/DICT_TO_PYDANTIC.md) | Dictionary configs → Pydantic models | ⏳ NEEDED |
| [OLD_API_TO_NEW.md](migrations/OLD_API_TO_NEW.md) | Legacy API → Simplified API | ⏳ NEEDED |

---

## 🎓 Tutorials & Examples

### By Use Case

| Document | Use Case | Status |
|----------|----------|--------|
| [BASIC_WORKFLOW.md](tutorials/BASIC_WORKFLOW.md) | Load data, create DataFrame | ⏳ UPDATE NEEDED |
| [METADATA_EXTRACTION.md](tutorials/METADATA_EXTRACTION.md) | Extract metadata from filenames | ⏳ UPDATE NEEDED |
| [BATCH_PROCESSING.md](tutorials/BATCH_PROCESSING.md) | Process multiple experiments | ✅ Current |
| [CUSTOM_HANDLERS.md](tutorials/CUSTOM_HANDLERS.md) | Define special handlers | ✅ Current |

---

## 📝 Documentation Standards

All documentation follows these standards:

### Structure

```markdown
# Document Title

**Version**: X.Y.Z
**Last Updated**: YYYY-MM-DD

## Overview
[Brief summary]

## [Main Sections]
[Content with examples]

## Related Documentation
[Links to related docs]
```

### Status Indicators

| Status | Meaning | Action Required |
|--------|---------|------------------|
| ✅ Current | Up-to-date with code | None |
| ⏳ UPDATE NEEDED | Out of sync | Review and update |
| **NEW** | Recently created | Review for accuracy |
| **PLANNED** | Not yet written | Create document |
| 🗑️ DEPRECATED | Obsolete | Mark for removal |

---

## 🔄 Documentation Update Tracking

### Recent Changes (2025-09-30)

#### Priority 1: Clarifying Ambiguities ✅

| Change | Documentation | Status |
|--------|---------------|--------|
| Pattern precedence rules defined | [PATTERN_PRECEDENCE.md](PATTERN_PRECEDENCE.md) | ✅ Complete |
| Metadata merge semantics specified | [METADATA_MERGE.md](METADATA_MERGE.md) | ✅ Complete |
| Dimension handling clarified | [DIMENSION_HANDLING.md](DIMENSION_HANDLING.md) | ✅ Complete |

#### Priority 2: Simplifying API 🔄

| Change | Documentation | Status |
|--------|---------------|--------|
| LegacyConfigAdapter removal plan | [LEGACY_CONFIG_ADAPTER_REMOVAL.md](LEGACY_CONFIG_ADAPTER_REMOVAL.md) | ✅ Complete |
| DiscoveryOptions dataclass design | [API_SIMPLIFICATION.md](API_SIMPLIFICATION.md) | ⏳ In Progress |
| Manifest utilities relocation | [MANIFEST_UTILITIES.md](MANIFEST_UTILITIES.md) | ⏳ Pending |

#### Priority 3: Strengthening Contracts 📋

| Change | Documentation | Status |
|--------|---------------|--------|
| Error taxonomy definition | [ERROR_TAXONOMY.md](ERROR_TAXONOMY.md) | ⏳ Pending |
| Performance monitoring | [PERFORMANCE_SLA.md](PERFORMANCE_SLA.md) | ⏳ Pending |

### Pending Updates

Documents that need updates to reflect semantic model changes:

1. **API_REFERENCE.md** - Update function signatures for DiscoveryOptions
2. **CONFIG_SCHEMA.md** - Clarify pattern/filter precedence
3. **DATAFRAME_SCHEMA.md** - Document dimension handling for 2D+ arrays
4. **QUICKSTART.md** - Show Pydantic models instead of LegacyConfigAdapter
5. **EXAMPLES.md** - Update to use DiscoveryOptions

---

## 🔍 Finding Documentation

### By Topic

- **Configuration**: See "Configuration" section under Reference Documentation
- **API Usage**: See "API Reference" and "Tutorials & Examples"
- **Patterns/Filters**: See [PATTERN_PRECEDENCE.md](PATTERN_PRECEDENCE.md)
- **Metadata**: See [METADATA_MERGE.md](METADATA_MERGE.md)
- **Arrays/DataFrames**: See [DIMENSION_HANDLING.md](DIMENSION_HANDLING.md)
- **Migration**: See "Migration & Upgrade Guides"
- **Testing**: See "Developer Guides"

### By User Journey

1. **New User Getting Started**
   - README.md → QUICKSTART.md → BASIC_WORKFLOW.md

2. **Migrating from v1.x to v2.0**
   - MIGRATE_V1_TO_V2.md → BREAKING_CHANGES_V2.md → DICT_TO_PYDANTIC.md

3. **Understanding Pattern Behavior**
   - PATTERN_PRECEDENCE.md → METADATA_MERGE.md → CONFIG_SCHEMA.md

4. **Implementing Custom Handlers**
   - DIMENSION_HANDLING.md → CUSTOM_HANDLERS.md → API_REFERENCE.md

---

## 📞 Documentation Feedback

Found an error or have a suggestion? Please:

1. **File an issue**: Tag with `documentation` label
2. **Submit a PR**: Follow [CONTRIBUTING.md](../CONTRIBUTING.md)
3. **Ask in discussions**: For clarifications

---

## 📅 Documentation Roadmap

### v2.0.0 Release (Target: +6 months)

- [ ] Complete all ⏳ UPDATE NEEDED documents
- [ ] Create all ⏳ NEEDED migration guides
- [ ] Review and update all tutorials
- [ ] Add video walkthroughs for common workflows
- [ ] Create interactive examples with Jupyter notebooks

### Future Enhancements

- [ ] API documentation auto-generation from docstrings
- [ ] Search functionality across all docs
- [ ] Multi-language support (starting with Python/R)
- [ ] Integration with readthedocs.io

---

## 🏷️ Document Tags

Documents are tagged for easy filtering:

- `#core-concepts` - Fundamental behavior and semantics
- `#api-reference` - Function/class documentation
- `#tutorial` - Step-by-step guides
- `#migration` - Upgrade and migration guides
- `#developer` - For contributors
- `#new` - Recently created (2025-09-30)

---

**Next**: Start with [PATTERN_PRECEDENCE.md](PATTERN_PRECEDENCE.md) to understand file filtering behavior.
