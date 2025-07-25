# Technical Specification

# 0. SUMMARY OF CHANGES

## 0.1 USER INTENT RESTATEMENT

### 0.1.1 Core Objective

Based on the provided requirements, the Blitzy platform understands that the objective is to refactor FlyRigLoader to reinforce its modular data loading architecture and prepare it for a 1.0-level release. This comprehensive refactoring addresses architectural gaps identified in recent code reviews while maintaining backward compatibility for existing workflows.

The refactoring encompasses five primary objectives:
1. **Registry Encapsulation**: Transform the registry system into a true internal implementation detail with plugin-style extensibility
2. **Configuration Standardization**: Unify all configuration handling under a Pydantic-backed builder pattern  
3. **Kedro Integration**: Create first-class support for Kedro's data catalog and pipeline workflows
4. **Testability Enhancement**: Maximize functional API surface with dependency injection and pure functions
5. **Version Awareness**: Embed version tracking for configuration compatibility and migration support

### 0.1.2 Special Instructions and Constraints

**CRITICAL: Process-Specific Directives**
- The refactoring must be executed **without interrupting ongoing use** of the library
- **Backward compatibility is mandatory** - existing user code must continue to work with minimal changes
- **Performance must not degrade** - maintain current benchmarks of <5 seconds for 10,000 file discovery
- **Coordination with fly-filt** is required as it currently pins FlyRigLoader at a specific version

**Methodological Requirements**
- Use **feature flags or parallel implementation** to allow incremental adoption
- Apply the **Adapter pattern** for legacy configuration support
- Implement **Protocol-based interfaces** for all extension points
- Follow **semantic versioning** with proper deprecation periods

**User Example Preservation**
The user has emphasized that the refactored system must support existing usage patterns:

User Example: Legacy dictionary configuration
```python
config = {
    'project': {'major_data_directory': '/data'},
    'experiments': {'exp1': {'date_range': ['2024-01-01', '2024-01-31']}}
}
files = load_experiment_files(config, 'exp1')  # Must continue to work
```

### 0.1.3 Technical Interpretation

These requirements translate to the following technical implementation strategy:

1. **Registry Architecture Enhancement**
   - To enforce plugin-style extensibility, we will consolidate all loader instantiation through `LoaderRegistry` and `SchemaRegistry` 
   - To enable entry-point discovery, we will enhance the registry system with automatic plugin detection
   - To ensure thread-safety, we will implement proper locking mechanisms in registry operations

2. **Configuration System Unification**  
   - To standardize configuration, we will make Pydantic models the primary interface with legacy adapter support
   - To simplify configuration creation, we will implement a comprehensive builder pattern with factory methods
   - To handle environment overrides uniformly, we will centralize all configuration resolution logic

3. **Kedro Pipeline Integration**
   - To enable Kedro catalog support, we will create `FlyRigLoaderDataSet` implementing `AbstractDataset`
   - To integrate with Kedro workflows, we will ensure all outputs are Kedro-compatible data structures
   - To support Kedro parameters, we will adapt configuration interfaces to accept Kedro params

4. **API Surface Enhancement**
   - To maximize testability, we will expose all operations as stateless, pure functions
   - To enable comprehensive mocking, we will use Protocol-based dependency injection throughout
   - To maintain backward compatibility, we will preserve existing function signatures with deprecation warnings

5. **Version Management Implementation**
   - To track configuration versions, we will embed version fields in all Pydantic models
   - To support legacy formats, we will implement version detection and automatic migration
   - To prepare for 1.0 release, we will establish clear deprecation timelines and migration guides

## 0.2 TECHNICAL SCOPE

### 0.2.1 Primary Objectives with Implementation Approach

**Achieve registry encapsulation by modifying registry access patterns to enforce centralized control**
- Current state: Direct loader instantiation exists in some modules, bypassing the registry
- Target state: All loader selection happens exclusively through registry lookups
- Critical success factor: Zero hardcoded loader references outside the registry module

**Achieve configuration standardization by extending Pydantic models to become the sole configuration interface**
- Current state: Mixed support for dict and Pydantic configs with inconsistent validation
- Target state: Pydantic models as primary interface with transparent legacy adapter
- Critical success factor: 100% of configuration paths validate through Pydantic

**Achieve Kedro compatibility by implementing custom DataSet classes that wrap FlyRigLoader operations**
- Current state: No native Kedro integration, requiring manual wrapper code
- Target state: Drop-in Kedro DataSet for catalog configuration
- Critical success factor: Seamless `catalog.yml` integration

**Achieve API consistency by refactoring all public functions to follow functional programming patterns**
- Current state: Mixed paradigm with some class methods and stateful operations
- Target state: Pure functional API with clear input/output contracts
- Critical success factor: All public APIs testable in isolation

**Achieve version awareness by implementing configuration versioning and migration infrastructure**
- Current state: No version tracking or migration support
- Target state: Automatic version detection and migration capabilities
- Critical success factor: Zero breaking changes for existing configurations

### 0.2.2 Component Impact Analysis

#### Direct Modifications Required

**`src/flyrigloader/registries/__init__.py`**: Enforce singleton pattern and entry-point discovery
- Add thread-safe plugin discovery mechanism for automatic loader registration
- Implement priority system for handling multiple loaders per extension
- Add comprehensive logging for registration events and conflicts

**`src/flyrigloader/config/models.py`**: Extend with version fields and builder enhancements  
- Add `schema_version` field to ProjectConfig, DatasetConfig, and ExperimentConfig
- Implement `create_config()` builder with comprehensive defaults
- Enhance LegacyConfigAdapter with deprecation warnings

**`src/flyrigloader/config/yaml_config.py`**: Update for consistent Pydantic returns
- Modify `load_config()` to return Pydantic models by default
- Add `legacy_mode` parameter for backward compatibility
- Implement automatic version detection and migration

**`src/flyrigloader/api.py`**: Enhance with new functional interfaces
- Add `validate_manifest()` for pre-flight validation
- Implement `create_kedro_dataset()` factory function
- Expose registry introspection functions

**`src/flyrigloader/io/loaders.py`**: Remove any direct loader instantiation
- Replace hardcoded loader logic with registry lookups
- Add loader capability introspection
- Implement loader priority resolution

#### Indirect Impacts and Dependencies

**`src/flyrigloader/io/transformers.py`**: Update to support Kedro DataFrame requirements
- Ensure output DataFrames are Kedro-compatible
- Add metadata columns required by Kedro pipelines
- Implement lazy transformation for memory efficiency

**`src/flyrigloader/discovery/files.py`**: Enhance for Kedro catalog integration
- Add Kedro-specific metadata extraction
- Support Kedro versioning patterns
- Implement catalog-aware discovery filters

**`src/flyrigloader/exceptions.py`**: Extend with new error types
- Add `RegistryError` for plugin conflicts
- Implement `VersionError` for migration failures  
- Create `KedroIntegrationError` for catalog issues

#### New Components Introduction

**`src/flyrigloader/kedro/`**: Create new subpackage for Kedro integration
- `datasets.py`: Implement FlyRigLoaderDataSet and variants
- `catalog.py`: Provide catalog configuration helpers
- `hooks.py`: Create Kedro lifecycle hooks for FlyRigLoader

**`src/flyrigloader/migration/`**: Create version migration infrastructure
- `versions.py`: Define version constants and compatibility matrix
- `migrators.py`: Implement configuration migrators
- `validators.py`: Add version-specific validation logic

### 0.2.3 File and Path Mapping

| Target File/Module | Source Reference | Context Dependencies | Modification Type |
|-------------------|------------------|---------------------|-------------------|
| `src/flyrigloader/registries/__init__.py` | Current implementation | `importlib.metadata`, entry points | Enhance discovery, enforce singleton |
| `src/flyrigloader/config/models.py` | Existing Pydantic models | `pydantic`, validators | Add version fields, builder methods |
| `src/flyrigloader/config/yaml_config.py` | Current YAML loader | PyYAML, models.py | Default to Pydantic, add migration |
| `src/flyrigloader/config/discovery.py` | Discovery engine | Pattern matchers | No changes needed |
| `src/flyrigloader/api.py` | Public API facade | All submodules | Add Kedro factories, validation |
| `src/flyrigloader/io/loaders.py` | Loader dispatch | Registry module | Remove hardcoded logic |
| `src/flyrigloader/io/pickle.py` | Pickle loader | Registry module | Ensure registry registration |
| `src/flyrigloader/io/transformers.py` | DataFrame transform | Column models | Add Kedro compatibility |
| `src/flyrigloader/kedro/datasets.py` | NEW | Kedro, api.py | Implement AbstractDataset |
| `src/flyrigloader/kedro/catalog.py` | NEW | Config models | Catalog helpers |
| `src/flyrigloader/migration/versions.py` | NEW | None | Version constants |
| `src/flyrigloader/migration/migrators.py` | NEW | Config models | Migration logic |
| `tests/flyrigloader/test_registry.py` | Registry tests | pytest, registries | Add thread-safety tests |
| `tests/flyrigloader/test_kedro_integration.py` | NEW | Kedro, datasets | Integration tests |
| `tests/flyrigloader/test_migration.py` | NEW | Migration module | Migration tests |
| `pyproject.toml` | Package metadata | None | Update version, add Kedro optional |
| `docs/kedro_integration.md` | NEW | None | Kedro usage guide |
| `docs/migration_guide.md` | Existing guide | None | Add version migration |

## 0.3 IMPLEMENTATION DESIGN

### 0.3.1 Technical Approach

**First, establish the enhanced registry foundation by modifying `src/flyrigloader/registries/__init__.py`**
- Implement automatic entry-point discovery using `importlib.metadata`
- Add thread-safe priority queue for handling multiple loaders per extension
- Create decorator `@auto_register` for plugin developers
- Ensure O(1) lookup performance is maintained

**Next, integrate configuration versioning by extending `src/flyrigloader/config/models.py` with version awareness**
- Add `schema_version: str = Field(default="1.0.0")` to all config models
- Implement `ConfigVersion` enum for version constants
- Create `migrate_config()` method on each model
- Establish version compatibility matrix

**Then, unify configuration interfaces by updating `src/flyrigloader/config/yaml_config.py`**
- Change `load_config()` to return Pydantic models by default:
  ```python
  def load_config(config_path, legacy_mode=False):
      if legacy_mode:
          return LegacyConfigAdapter(pydantic_model)
      return pydantic_model
  ```
- Add automatic version detection from YAML
- Implement migration on load when version mismatch detected

**Subsequently, create Kedro integration by implementing `src/flyrigloader/kedro/datasets.py`**
Following the Kedro AbstractDataset pattern:
```python
class FlyRigLoaderDataSet(AbstractDataset[FileManifest, pd.DataFrame]):
    def __init__(self, filepath: str, experiment_name: str, **kwargs):
        self._filepath = filepath
        self._experiment_name = experiment_name
        self._kwargs = kwargs
    
    def _load(self) -> FileManifest:
        return discover_experiment_manifest(
            config_path=self._filepath,
            experiment_name=self._experiment_name,
            **self._kwargs
        )
    
    def _save(self, data: pd.DataFrame) -> None:
        raise NotImplementedError("FlyRigLoaderDataSet is read-only")
    
    def _exists(self) -> bool:
        return Path(self._filepath).exists()
    
    def _describe(self) -> dict:
        return {
            "filepath": self._filepath,
            "experiment_name": self._experiment_name,
            "kwargs": self._kwargs
        }
```

**Finally, ensure functional API completeness by enhancing `src/flyrigloader/api.py`**
- Add builder function for Kedro dataset creation:
  ```python
  def create_kedro_dataset(config_path, experiment_name, **options):
      """Factory function for Kedro catalog configuration."""
      return FlyRigLoaderDataSet(
          filepath=config_path,
          experiment_name=experiment_name,
          **options
      )
  ```
- Expose registry introspection: `get_registered_loaders()`, `get_loader_capabilities()`
- Add validation utilities: `validate_config_version()`, `check_plugin_compatibility()`

### 0.3.2 User-Provided Examples Integration

The user's example of dictionary-based configuration will be preserved through the LegacyConfigAdapter:

```python
# User's example usage pattern
config = {
    'project': {'major_data_directory': '/data'},
    'experiments': {'exp1': {'date_range': ['2024-01-01', '2024-01-31']}}
}
files = load_experiment_files(config, 'exp1')
```

Will be implemented in the refactored system as:
```python
# Internal implementation
def load_experiment_files(config, experiment_name, **kwargs):
    # Detect dict vs Pydantic model
    if isinstance(config, dict):
        warnings.warn(
            "Dictionary configs are deprecated. Use create_config() builder.",
            DeprecationWarning,
            stacklevel=2
        )
        config = LegacyConfigAdapter(ProjectConfig(**config))
    
    # Proceed with Pydantic model
    return discover_experiment_manifest(config, experiment_name, **kwargs)
```

### 0.3.3 Critical Implementation Details

**Registry Pattern Enhancement**
- Use singleton pattern with class-level lock for thread safety
- Implement `RegistryPriority` enum: `BUILTIN < USER < PLUGIN < OVERRIDE`
- Maintain backward compatibility by keeping existing registry methods
- Add comprehensive logging at INFO level for all registration events

**Configuration Builder Pattern**
```python
def create_config(**kwargs) -> ProjectConfig:
    """
    Builder function for creating validated configurations.
    
    Examples:
        config = create_config(
            base_directory="/data",
            experiments={
                "exp1": create_experiment(
                    date_range=["2024-01-01", "2024-01-31"],
                    rig_names=["rig1", "rig2"]
                )
            }
        )
    """
    # Apply defaults, validate, and return
    return ProjectConfig(**kwargs)
```

**Kedro Catalog Integration**
Enable usage in `catalog.yml`:
```yaml
my_experiment_data:
  type: flyrigloader.FlyRigLoaderDataSet
  filepath: "${base_dir}/config/experiment_config.yaml"
  experiment_name: "baseline_study"
  recursive: true
  extract_metadata: true
```

**Version Migration Strategy**
```python
class ConfigMigrator:
    migrations = {
        "0.1.0": migrate_v0_1_to_v1_0,
        "0.2.0": migrate_v0_2_to_v1_0,
    }
    
    def migrate(self, config: dict, from_version: str) -> dict:
        """Apply migrations to reach current version."""
        # Chain migrations as needed
        return migrated_config
```

### 0.3.4 Dependency Analysis

**Required Dependencies for Implementation**
- `importlib.metadata` (stdlib): Entry-point discovery
- `threading` (stdlib): Registry thread safety
- `kedro>=0.18.0` (optional): Kedro integration (make optional dependency)
- `semantic_version>=2.10.0`: Version comparison logic

**Version Constraints and Compatibility Requirements**
- Maintain support for Python 3.8+ (no use of 3.9+ features)
- Kedro as optional dependency: `pip install flyrigloader[kedro]`
- Preserve PyYAML, Pydantic v2, numpy, pandas version requirements

**Justification for Each Dependency Choice**
- `importlib.metadata`: Standard library solution for plugin discovery
- `threading.RLock`: Reentrant locks for nested registry operations
- Optional Kedro: Avoid forcing Kedro on users who don't need it
- `semantic_version`: Robust version comparison with clear API

## 0.4 SCOPE BOUNDARIES

### 0.4.1 Explicitly In Scope

**Comprehensive list of ALL affected files/modules:**

1. **Registry System Modifications**
   - `src/flyrigloader/registries/__init__.py` - Add entry-point discovery, priority system
   - `src/flyrigloader/io/loaders.py` - Remove hardcoded loader logic
   - `src/flyrigloader/io/pickle.py` - Ensure proper registration decorators

2. **Configuration System Updates**
   - `src/flyrigloader/config/models.py` - Add version fields, enhance builders
   - `src/flyrigloader/config/yaml_config.py` - Default to Pydantic returns
   - `src/flyrigloader/config/validators.py` - Add version validation

3. **API Layer Enhancements**
   - `src/flyrigloader/api.py` - Add Kedro factories, validation functions
   - `src/flyrigloader/exceptions.py` - Add new exception types

4. **New Kedro Integration**
   - `src/flyrigloader/kedro/__init__.py` - Package initialization
   - `src/flyrigloader/kedro/datasets.py` - AbstractDataset implementations
   - `src/flyrigloader/kedro/catalog.py` - Catalog helpers

5. **New Migration Infrastructure**
   - `src/flyrigloader/migration/__init__.py` - Package initialization
   - `src/flyrigloader/migration/versions.py` - Version constants
   - `src/flyrigloader/migration/migrators.py` - Migration logic

6. **Test Suite Additions**
   - `tests/flyrigloader/test_registry_threading.py` - Thread safety tests
   - `tests/flyrigloader/test_kedro_integration.py` - Kedro dataset tests
   - `tests/flyrigloader/test_config_migration.py` - Migration tests
   - `tests/flyrigloader/test_config_builders.py` - Builder pattern tests

7. **Documentation Updates**
   - `docs/kedro_integration.md` - New Kedro usage guide
   - `docs/migration_guide.md` - Update with version migration
   - `docs/architecture.md` - Update with new components
   - `README.md` - Add Kedro examples

8. **Package Configuration**
   - `pyproject.toml` - Add optional Kedro dependency
   - `setup.cfg` - Add entry point groups if needed

**All configuration changes required:**
- Add `schema_version` to all YAML configs
- Update example configs with version field
- Create migration templates

**All test modifications needed:**
- Update existing tests expecting dict configs
- Add thread-safety test suite
- Create Kedro integration test suite
- Add migration test scenarios

**All documentation updates required:**
- API documentation for new functions
- Migration guide for 0.x to 1.0
- Kedro integration tutorial
- Plugin development guide

### 0.4.2 Explicitly Out of Scope

**Domain-Specific Feature Changes**
- No modifications to experimental data parsing algorithms
- No changes to DataFrame column calculations or transformations
- No alterations to the core discovery matching patterns
- Existing file format support remains unchanged

**User Interface/CLI**
- No new command-line tools or interfaces
- Existing CLI behavior remains unchanged
- No GUI or web interface additions

**Performance Optimizations**
- No changes to file I/O performance beyond what's required
- No new caching mechanisms unless needed for compatibility
- Discovery performance must remain at current levels

**External Project Modifications**
- fly-filt refactoring is handled separately
- No changes to downstream consumers beyond coordination
- External plugin development is not part of this scope

**Deprecation Removals**
- Legacy interfaces marked deprecated but NOT removed
- Old function names preserved with warnings
- Dictionary config support maintained via adapter

**Future Considerations Not Addressed Now**
- Async/await support for I/O operations
- Cloud storage native integration
- Real-time data streaming capabilities
- GraphQL or REST API interfaces

## 0.5 VALIDATION CHECKLIST

### 0.5.1 Implementation Verification Points

**Registry Encapsulation Success Criteria**
- [ ] All loader instantiation goes through LoaderRegistry.get_loader_for_extension()
- [ ] Entry-point discovery automatically registers plugins on import
- [ ] Thread-safety tests pass with 100 concurrent operations
- [ ] Registry introspection APIs return accurate loader information

**Configuration Standardization Success Criteria**
- [ ] load_config() returns Pydantic models by default
- [ ] Legacy dictionary configs work via LegacyConfigAdapter
- [ ] All config paths validate through Pydantic before use
- [ ] Version field present in all configuration models

**Kedro Integration Success Criteria**
- [ ] FlyRigLoaderDataSet successfully loads in Kedro catalog
- [ ] Data can be retrieved via catalog.load("dataset_name")
- [ ] Kedro pipeline nodes can consume FlyRigLoader outputs
- [ ] Documentation includes working Kedro examples

**API Enhancement Success Criteria**
- [ ] All public functions in api.py are pure (no side effects)
- [ ] Each function has comprehensive type hints
- [ ] Dependency injection works for all operations
- [ ] Backward compatibility maintained for all existing functions

**Version Awareness Success Criteria**
- [ ] Configurations include and validate schema_version
- [ ] Old configs automatically migrate on load
- [ ] Version mismatch produces clear migration messages
- [ ] Migration path documented for all versions

### 0.5.2 Observable Changes That Confirm Successful Implementation

**For Registry System**
- Plugin loaders appear in get_registered_loaders() without code changes
- Duplicate extension registration produces clear priority messages
- Thread-safety verified by concurrent test suite passing

**For Configuration System**
- Default load_config() usage returns ProjectConfig instance
- Deprecation warnings appear for dictionary config usage
- Config builder functions available in flyrigloader.config

**For Kedro Integration**
- `from flyrigloader.kedro import FlyRigLoaderDataSet` works
- Kedro catalog YAML accepts FlyRigLoader dataset types
- Integration tests pass with real Kedro pipeline

**For Version Management**
- New configs contain schema_version: "1.0.0"
- Old configs load with automatic migration
- Version info accessible via config.schema_version

### 0.5.3 Integration Points That Must Be Tested

**Cross-Module Integration**
- Registry ↔ Loaders: Dynamic registration and lookup
- Config ↔ Discovery: Version-aware pattern application
- API ↔ All modules: Consistent Pydantic model usage
- Kedro ↔ API: Proper data structure conversion

**External Integration**
- Entry points: Plugin discovery from installed packages
- Kedro catalog: YAML configuration parsing
- fly-filt: Compatibility with pinned version
- pytest: All test utilities remain functional

**Data Flow Validation**
- YAML → Pydantic → Discovery → Loading → DataFrame
- Legacy dict → Adapter → Pydantic → Same flow
- Kedro catalog → Dataset → Manifest → DataFrame

## 0.6 EXECUTION PARAMETERS

### 0.6.1 Special Execution Instructions

**Documentation-First Approach**
- Generate comprehensive documentation before implementation
- Create migration guide with concrete examples
- Document all deprecations with replacement patterns

**Incremental Implementation Strategy**
- Phase 1: Registry enhancement (can release independently)
- Phase 2: Configuration standardization (backward compatible)
- Phase 3: Kedro integration (optional feature)
- Phase 4: Version management (with migration tools)

**Testing Requirements**
- Maintain 90%+ code coverage throughout refactoring
- Add property-based tests for configuration builders
- Create integration test suite for Kedro compatibility
- Implement performance benchmarks before/after

**Release Strategy**
- Pre-release as 1.0.0-rc1 for fly-filt testing
- Coordinate with fly-filt maintainers for compatibility
- Document breaking changes in CHANGELOG.md
- Provide migration script for automated updates

### 0.6.2 Constraints and Boundaries

**Technical Constraints Specified by the User**
- Must maintain current performance benchmarks
- Cannot break existing public API contracts
- Must support Python 3.8+ (no newer syntax)
- Registry lookup must remain O(1)

**Process Constraints**
- No production deployment until fly-filt validates
- All changes must be reversible via feature flags
- Documentation must be updated before code
- Deprecation warnings required for all changes

**Output Constraints**
- Kedro integration as optional install only
- Version field optional in configs initially
- Migration automatic but can be disabled
- Logs must clearly indicate refactored paths

**Quality Requirements**
- All new code must have type hints
- Follow existing code style (Black, isort)
- Comprehensive docstrings for public APIs
- Example code for every new feature

# 1. INTRODUCTION

## 1.1 EXECUTIVE SUMMARY

### 1.1.1 Brief Overview
FlyRigLoader is an enterprise-grade Python library specifically designed for neuroscience research laboratories to standardize and streamline the management of optical rig ("opto rig") experimental data. The library provides a comprehensive ETL (Extract, Transform, Load) framework that addresses the unique challenges of scientific data workflows while maintaining production-level reliability and extensibility.

### 1.1.2 Core Business Problem
Neuroscience research laboratories face significant challenges in managing experimental data from optical rigs used in fly (Drosophila) research:
- **Data Fragmentation**: Experimental data scattered across heterogeneous file formats and directory structures
- **Configuration Complexity**: Manual, error-prone configuration management without validation
- **Processing Inefficiency**: Memory-intensive monolithic data loading preventing analysis of large datasets
- **Integration Barriers**: Difficulty integrating with modern data pipeline frameworks (e.g., Kedro)
- **Extensibility Limitations**: Hardcoded format support preventing adaptation to new experimental protocols

### 1.1.3 Key Stakeholders and Users

| Stakeholder Group | Primary Use Case | Key Benefits |
|------------------|------------------|--------------|
| **Neuroscience Researchers** | Interactive data exploration in Jupyter notebooks | Simplified data access, validated configurations, memory-efficient processing |
| **Data Analysts** | Building reproducible analysis pipelines | Standardized interfaces, Kedro integration, comprehensive error handling |
| **Research Software Engineers** | Extending functionality for new data formats | Plugin architecture, Protocol-based interfaces, comprehensive documentation |
| **Laboratory PIs** | Ensuring data integrity and reproducibility | Validated configurations, audit trails, version control |

### 1.1.4 Expected Business Impact and Value Proposition

The FlyRigLoader <span style="background-color: rgba(91, 57, 243, 0.2)">v1.0</span> architecture delivers measurable improvements:
- **90% Reduction** in configuration-related errors through Pydantic validation
- **5x Performance Improvement** in large dataset processing via decoupled pipeline
- **<5 Minutes Migration Time** from legacy systems with full backward compatibility
- **Zero-Code Extension** capability through registry-based plugin system
- **100% Test Coverage** on critical modules ensuring research reliability

## 1.2 SYSTEM OVERVIEW

### 1.2.1 Project Context

#### Business Context and Market Positioning
FlyRigLoader addresses the growing need for standardized data management in neuroscience research, particularly in laboratories studying neural circuits and behavior using optogenetic techniques. The library positions itself as the bridge between:
- Raw experimental data collection systems
- Modern data science workflows and analysis pipelines
- Collaborative research environments requiring reproducibility

#### Current System Limitations
The legacy system <span style="background-color: rgba(91, 57, 243, 0.2)">(pre-1.0)</span> exhibited critical limitations:
- **Monolithic Architecture**: Combined discovery, loading, and transformation prevented selective processing
- **Dictionary-Based Configuration**: No validation leading to runtime failures
- **Memory Constraints**: Loading entire datasets prevented analysis of experiments >10GB
- **Limited Extensibility**: Hardcoded pickle format support only

#### Integration with Existing Enterprise Landscape
FlyRigLoader seamlessly integrates with the scientific Python ecosystem:
- **Kedro Pipelines**: First-class support for production data workflows
- **Jupyter Notebooks**: Interactive research and exploration
- **HPC Environments**: Memory-efficient processing for cluster computing
- **Version Control**: Configuration-as-code for reproducible research

### 1.2.2 High-Level Description

#### Primary System Capabilities

| Capability | Description | Key Features |
|------------|-------------|--------------|
| **Pattern-Based File Discovery** | Intelligent location of experimental data | Recursive search, regex metadata extraction, configurable filters |
| **Schema-Validated Configuration** | Type-safe experiment setup | Pydantic models, hierarchical settings, environment overrides |
| **Decoupled Data Pipeline** | Three-stage processing architecture | Discovery → Loading → Transformation isolation |
| **Plugin-Based Extensibility** | Dynamic format and schema support | Registry pattern, entry-point discovery, Protocol interfaces |
| **Version-Aware Configuration & Migration** | Embedded `schema_version` tracking and automatic migration | Version detection, compatibility matrix, one-click migrators |

#### Major System Components

```mermaid
graph TB
    subgraph "FlyRigLoader Architecture"
        A[API Layer] --> B[Configuration Layer]
        B --> C[Discovery Engine]
        B --> D[Registry System]
        A --> X[Migration Engine]
        B --> X
        C --> E[I/O Pipeline]
        D --> E
        E --> F[Transformation Engine]
        G[Error Hierarchy] --> A
        G --> C
        G --> E
        G --> F
    end
    
    subgraph "External Integration"
        H[Kedro Pipelines] --> A
        I[Jupyter Notebooks] --> A
        J[CLI Tools] --> A
        K[HPC Environments] --> A
    end
    
    subgraph "Data Sources"
        L[Pickle Files] --> E
        M[YAML Configs] --> B
        N[Directory Structures] --> C
    end
```

#### Core Technical Approach
The system employs Protocol-based dependency injection enabling:
- **Testability**: Mock injection for comprehensive testing
- **Extensibility**: Plugin development without core modifications
- **Performance**: Lazy evaluation and streaming transformations
- **Reliability**: Comprehensive validation at every stage

### 1.2.3 Success Criteria

#### Measurable Objectives

| Metric | Target | Validation Method |
|--------|--------|------------------|
| **Discovery Performance** | <5 seconds for 10,000 files | pytest-benchmark in CI |
| **Loading Performance** | <1 second per 100MB | Memory profiler validation |
| **Memory Efficiency** | <2x data size overhead | psutil monitoring |
| **Test Coverage** | ≥90% overall, 100% critical | coverage.py with thresholds |
| **API Response Time** | <100ms for manifest operations | Performance benchmarks |

#### Critical Success Factors
1. **Backward Compatibility**: Zero breaking changes for existing workflows
2. **Developer Experience**: <30 minute onboarding for new contributors
3. **Research Reproducibility**: Deterministic results across environments
4. **Extension Simplicity**: <50 lines of code for new format support

#### Key Performance Indicators (KPIs)

| KPI | Current | Target | Timeline |
|-----|---------|--------|----------|
| Configuration Error Rate | 15% | <1% | Immediate |
| Average Processing Time | 45s/GB | <10s/GB | Q2 2025 |
| Plugin Adoption Rate | 0% | 25% | Q4 2025 |
| User Satisfaction Score | 3.2/5 | 4.5/5 | Q3 2025 |

## 1.3 SCOPE

### 1.3.1 In-Scope

#### Core Features and Functionalities

**Must-Have Capabilities:**
1. **File Discovery System**
   - Recursive and non-recursive directory traversal
   - Pattern-based filtering with regex support
   - Metadata extraction from filenames
   - Multi-directory search capabilities

2. **Configuration Management**
   - YAML-based hierarchical configuration
   - Pydantic model validation
   - Environment variable overrides
   - Legacy configuration adapter
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Version-aware configuration with automatic migration (embedded `schema_version`, compatibility matrix)</span>

3. **Data Loading Pipeline**
   - Pickle format support (.pkl, .pklz, .pkl.gz)
   - Registry-based loader selection
   - Memory-efficient streaming
   - Format auto-detection

4. **Transformation Engine**
   - DataFrame creation from raw data
   - Column configuration validation
   - Special handler support
   - Metadata integration

**Primary User Workflows:**
- Experiment manifest discovery
- Selective data file loading
- Configuration-driven transformation
- Batch processing pipelines

**Essential Integrations:**
- Kedro pipeline nodes
- Jupyter notebook cells
- CLI batch processing
- pytest test fixtures

**Key Technical Requirements:**
- Python 3.8-3.11 compatibility
- Cross-platform support (Linux, macOS, Windows)
- Thread-safe registry operations
- Comprehensive error context

#### Implementation Boundaries

**System Boundaries:**
- Local file system operations only
- Configuration-driven behavior
- Python ecosystem integration
- Memory-bound processing

**User Groups Covered:**
- Research scientists
- Data analysts
- Software developers
- Laboratory technicians

**Geographic/Market Coverage:**
- Academic research institutions
- Neuroscience laboratories
- Open-source community

**Data Domains Included:**
- Experimental time-series data
- Behavioral tracking data
- Stimulus-response matrices
- Metadata annotations

### 1.3.2 Out-of-Scope

#### Explicitly Excluded Features/Capabilities
- Real-time data streaming from acquisition systems
- Cloud storage integration (S3, GCS, Azure)
- Database connectivity (SQL, NoSQL)
- GUI or web interface
- Data visualization components
- Statistical analysis functions
- Machine learning pipelines

#### Future Phase Considerations
- **Phase 2 (Q3 2025)**: HDF5/NWB format support
- **Phase 3 (Q4 2025)**: Cloud storage adapters
- **Phase 4 (Q1 2026)**: Real-time streaming API
- **Phase 5 (Q2 2026)**: Web-based configuration UI

#### Integration Points Not Covered
- Direct hardware interfacing
- LIMS system integration
- Electronic lab notebook APIs
- Cluster job scheduling
- Container orchestration

#### Unsupported Use Cases
- Multi-user concurrent editing
- Distributed processing across nodes
- Binary protocol parsing
- Image/video data processing
- Signal processing algorithms
- Cross-experiment meta-analysis

#### References
- `docs/architecture.md` - Architectural principles, component design, and integration patterns
- `README.md` - Project overview, features, and usage examples
- `docs/migration_guide.md` - Key improvements and migration strategies
- `examples/external_project/example_config.yaml` - Configuration schema and validation examples
- `blitzy/documentation/Project Guide.md` - Executive summary and project context
- `blitzy/documentation/Technical Specifications.md` - Requirements overview and specifications
- `.github/workflows/*.yml` - CI/CD quality requirements and testing standards
- `src/flyrigloader/` - Core package modules and implementation
- `tests/` - Testing framework structure and validation approaches
- `docs/io/` - I/O system documentation and interfaces
- `examples/` - Implementation examples and usage patterns
- `examples/external_project/` - Migration demonstrations and integration examples

# 2. PRODUCT REQUIREMENTS

## 2.1 FEATURE CATALOG

### 2.1.1 F-001: Pattern-Based File Discovery System

**Feature Metadata**
- **Unique ID**: F-001
- **Feature Name**: Pattern-Based File Discovery System
- **Feature Category**: Core Data Pipeline
- **Priority Level**: Critical
- **Status**: Completed

**Description**
- **Overview**: Intelligent file discovery engine that locates experimental data files using configurable patterns, metadata extraction, and filtering capabilities
- **Business Value**: Reduces manual file management overhead by 90%, enables automated data pipeline workflows for neuroscience research
- **User Benefits**: Researchers can discover experiment files automatically without manual directory traversal or hardcoded paths, supporting both recursive and non-recursive search
- **Technical Context**: Implements manifest-based discovery pattern separating file location from data loading for memory efficiency, achieving <5 seconds performance for 10,000 files

**Dependencies**
- **Prerequisite Features**: None (foundational feature)
- **System Dependencies**: Python pathlib, fnmatch, re modules
- **External Dependencies**: File system access permissions
- **Integration Requirements**: Must integrate with Configuration Management (F-002) for pattern definitions

### 2.1.2 F-002: Configuration Management System (updated)

**Feature Metadata**
- **Unique ID**: F-002
- **Feature Name**: Configuration Management System
- **Feature Category**: Configuration & Validation
- **Priority Level**: Critical
- **Status**: <span style="background-color: rgba(91, 57, 243, 0.2)">Refactor In Progress</span>

**Description**
- **Overview**: YAML-based hierarchical configuration system with Pydantic v2 validation, environment variable support, and backward compatibility
- **Business Value**: Eliminates 90% of configuration-related errors through strict validation at load time, enabling reproducible research workflows
- **User Benefits**: Type-safe configuration with clear error messages, hierarchical settings inheritance, environment-specific overrides supporting academic collaboration
- **Technical Context**: Uses Pydantic models for ProjectConfig, DatasetConfig, ExperimentConfig with comprehensive validation rules and legacy adapter support. <span style="background-color: rgba(91, 57, 243, 0.2)">Implements Pydantic-backed builder pattern with embedded schema_version field for automatic version detection and migration</span>

**Dependencies**
- **Prerequisite Features**: None (foundational feature)
- **System Dependencies**: PyYAML, Pydantic v2.6+
- **External Dependencies**: YAML configuration files
- **Integration Requirements**: Provides configuration data to all other features

### 2.1.3 F-003: Registry-Based Data Loading Pipeline

**Feature Metadata**
- **Unique ID**: F-003
- **Feature Name**: Registry-Based Data Loading Pipeline
- **Feature Category**: Data Input/Output
- **Priority Level**: Critical
- **Status**: Completed

**Description**
- **Overview**: Extensible file loading system using registry pattern for format-specific loaders with O(1) lookup performance
- **Business Value**: Enables support for new data formats without modifying core code, reducing development time by 75% for format extensions
- **User Benefits**: Automatic format detection, seamless loading of pickle (.pkl, .pklz, .pkl.gz) formats with memory-efficient streaming for large datasets
- **Technical Context**: Implements BaseLoader protocol with dependency injection for testability, supporting <1 second per 100MB performance target

**Dependencies**
- **Prerequisite Features**: F-001 (File Discovery), F-006 (Plugin Architecture)
- **System Dependencies**: pickle, gzip, pandas
- **External Dependencies**: Data files in supported formats
- **Integration Requirements**: Integrates with Transform Engine (F-004) for DataFrame conversion

### 2.1.4 F-004: DataFrame Transformation Engine (updated)

**Feature Metadata**
- **Unique ID**: F-004
- **Feature Name**: DataFrame Transformation Engine
- **Feature Category**: Data Transformation
- **Priority Level**: Critical
- **Status**: Completed

**Description**
- **Overview**: Configurable transformation pipeline converting raw experimental data to standardized pandas DataFrames with column validation
- **Business Value**: Ensures data consistency across experiments, reduces data preparation time by 80% through automated transformation
- **User Benefits**: Automatic column mapping, dimension normalization, metadata integration, strict schema enforcement for reproducible analysis
- **Technical Context**: Uses ColumnConfigDict with special handlers for array transformations, maintaining <2x data size memory overhead. <span style="background-color: rgba(91, 57, 243, 0.2)">All DataFrames emitted are Kedro-compatible and include mandatory metadata columns required by Kedro pipelines</span>

**Dependencies**
- **Prerequisite Features**: F-003 (Data Loading), F-002 (Configuration)
- **System Dependencies**: pandas, numpy
- **External Dependencies**: column_config.yaml schema file
- **Integration Requirements**: Must process output from Data Loading Pipeline

### 2.1.5 F-005: Experiment Manifest Discovery

**Feature Metadata**
- **Unique ID**: F-005
- **Feature Name**: Experiment Manifest Discovery
- **Feature Category**: Workflow Management
- **Priority Level**: High
- **Status**: Completed

**Description**
- **Overview**: Specialized discovery function that creates comprehensive file manifests for experiments without loading data
- **Business Value**: Enables selective data processing, reducing memory usage by 5x for large datasets through manifest-based workflows
- **User Benefits**: Preview experiment contents before loading, selective file processing, manifest-based workflows for efficient resource utilization
- **Technical Context**: Returns FileManifest objects with metadata, size, and date information, supporting <100ms response time for manifest operations

**Dependencies**
- **Prerequisite Features**: F-001 (File Discovery), F-002 (Configuration)
- **System Dependencies**: Core discovery modules
- **External Dependencies**: Experiment directory structure
- **Integration Requirements**: Provides input for selective loading workflows

### 2.1.6 F-006: Plugin Architecture System (updated)

**Feature Metadata**
- **Unique ID**: F-006
- **Feature Name**: Plugin Architecture System
- **Feature Category**: Extensibility Framework
- **Priority Level**: High
- **Status**: Completed

**Description**
- **Overview**: Registry-based plugin system supporting dynamic loader and schema registration via entry points or decorators
- **Business Value**: Enables zero-code extensions, reducing integration time for new formats from days to hours, supporting 25% plugin adoption rate target
- **User Benefits**: Add custom loaders/schemas without modifying core library, automatic plugin discovery with <50 lines of code for new format support
- **Technical Context**: Thread-safe singleton registries with O(1) lookup, setuptools entry point integration, Protocol-based interfaces. <span style="background-color: rgba(91, 57, 243, 0.2)">Includes thread-safe priority resolution (BUILTIN<USER<PLUGIN<OVERRIDE) and automatic entry-point discovery decorator @auto_register</span>

**Dependencies**
- **Prerequisite Features**: None (foundational for extensibility)
- **System Dependencies**: importlib.metadata, threading
- **External Dependencies**: Plugin packages with proper entry points
- **Integration Requirements**: Used by Data Loading (F-003) and Schema Validation features

### 2.1.7 F-007: Comprehensive Error Handling (updated)

**Feature Metadata**
- **Unique ID**: F-007
- **Feature Name**: Comprehensive Error Handling
- **Feature Category**: Error Management
- **Priority Level**: High
- **Status**: Completed

**Description**
- **Overview**: Domain-specific exception hierarchy with context preservation, error codes, and structured logging integration
- **Business Value**: Reduces debugging time by 60% through detailed error context and programmatic error handling, improving research productivity
- **User Benefits**: Clear error messages, preservation of error context, actionable error codes for troubleshooting experimental data issues
- **Technical Context**: Four main exception types: ConfigError, DiscoveryError, LoadError, TransformError with comprehensive context tracking. <span style="background-color: rgba(91, 57, 243, 0.2)">Extended with new exception classes RegistryError, VersionError, and KedroIntegrationError</span>

**Dependencies**
- **Prerequisite Features**: F-008 (Logging System)
- **System Dependencies**: Python exception handling
- **External Dependencies**: None
- **Integration Requirements**: Used by all other features for error reporting

### 2.1.8 F-008: Structured Logging System

**Feature Metadata**
- **Unique ID**: F-008
- **Feature Name**: Structured Logging System
- **Feature Category**: Observability
- **Priority Level**: Medium
- **Status**: Completed

**Description**
- **Overview**: Loguru-based logging with automatic rotation, console/file outputs, and test mode support
- **Business Value**: Improves debugging efficiency, enables production monitoring and audit trails for research data processing
- **User Benefits**: Automatic log rotation, structured output, configurable verbosity levels supporting research reproducibility requirements
- **Technical Context**: Console logging at INFO level, file logging at DEBUG with 10MB rotation, test mode detection

**Dependencies**
- **Prerequisite Features**: None (foundational service)
- **System Dependencies**: loguru>=0.7.0
- **External Dependencies**: Write access to logs directory
- **Integration Requirements**: Used by all features for operational logging

### 2.1.9 F-009: Path Management Utilities

**Feature Metadata**
- **Unique ID**: F-009
- **Feature Name**: Path Management Utilities
- **Feature Category**: Utility Services
- **Priority Level**: Medium
- **Status**: Completed

**Description**
- **Overview**: Cross-platform path manipulation utilities with security validation and existence checking
- **Business Value**: Prevents path traversal attacks, ensures cross-platform compatibility across academic research environments
- **User Benefits**: Safe path operations, automatic directory creation, relative/absolute path conversion supporting diverse research computing environments
- **Technical Context**: Wraps pathlib with additional validation and security checks, supporting Linux, macOS, and Windows

**Dependencies**
- **Prerequisite Features**: None
- **System Dependencies**: pathlib, os
- **External Dependencies**: File system access
- **Integration Requirements**: Used by Discovery and Loading features

### 2.1.10 F-010: Test Infrastructure & Hooks

**Feature Metadata**
- **Unique ID**: F-010
- **Feature Name**: Test Infrastructure & Hooks
- **Feature Category**: Developer Tools
- **Priority Level**: Medium
- **Status**: Completed

**Description**
- **Overview**: Comprehensive test support with dependency injection, mock providers, and test-specific behavior flags
- **Business Value**: Enables ≥90% test coverage target, reduces test development time by 70%, ensuring research reliability
- **User Benefits**: Easy mocking of file system and I/O operations, deterministic test behavior for reproducible research validation
- **Technical Context**: Protocol-based dependency injection, test mode detection via environment variables, pytest integration

**Dependencies**
- **Prerequisite Features**: All core features
- **System Dependencies**: pytest, unittest.mock
- **External Dependencies**: PYTEST_CURRENT_TEST environment variable
- **Integration Requirements**: Integrates with all features through dependency injection

### 2.1.11 F-011: Kedro Integration Layer

**Feature Metadata**
- **Unique ID**: F-011
- **Feature Name**: Kedro Integration Layer
- **Feature Category**: Pipeline Integration
- **Priority Level**: High
- **Status**: <span style="background-color: rgba(91, 57, 243, 0.2)">Planned</span>

**Description**
- **Overview**: <span style="background-color: rgba(91, 57, 243, 0.2)">Native FlyRigLoaderDataSet support for Kedro catalog and pipeline workflows</span>
- **Business Value**: Enables seamless integration with production data pipelines, reducing pipeline development time by 80% for neuroscience research teams
- **User Benefits**: Drop-in Kedro DataSet classes for catalog.yml configuration, native support for Kedro versioning and data lineage tracking
- **Technical Context**: Implements AbstractDataset interface with FlyRigLoaderDataSet, FlyRigManifestDataSet variants, supporting catalog configuration and lifecycle hooks

**Dependencies**
- **Prerequisite Features**: F-001 (File Discovery), F-002 (Configuration), F-003 (Data Loading)
- **System Dependencies**: kedro>=0.18.0 (optional dependency)
- **External Dependencies**: Kedro project structure, catalog.yml configuration
- **Integration Requirements**: Must integrate with Kedro catalog system and pipeline execution

### 2.1.12 F-012: Configuration Version Management & Migration

**Feature Metadata**
- **Unique ID**: F-012
- **Feature Name**: Configuration Version Management & Migration
- **Feature Category**: Configuration & Validation
- **Priority Level**: High
- **Status**: <span style="background-color: rgba(91, 57, 243, 0.2)">Planned</span>

**Description**
- **Overview**: <span style="background-color: rgba(91, 57, 243, 0.2)">Embedded schema_version fields, automatic migration and LegacyConfigAdapter integration</span>
- **Business Value**: Ensures zero breaking changes during system upgrades, enabling continuous research workflows with automatic configuration evolution
- **User Benefits**: Automatic detection of configuration version mismatches, one-click migration from legacy configurations, seamless upgrade path preserving research continuity
- **Technical Context**: Implements ConfigVersion enum, migration matrix, and automatic version detection with LegacyConfigAdapter fallback for backward compatibility

**Dependencies**
- **Prerequisite Features**: F-002 (Configuration Management)
- **System Dependencies**: semantic_version>=2.10.0
- **External Dependencies**: Legacy configuration files
- **Integration Requirements**: Must integrate with all configuration-dependent features for seamless migration

### 2.1.2 FUNCTIONAL REQUIREMENTS TABLE

#### Core Feature Requirements

| Requirement ID | Feature | Description | Acceptance Criteria | Priority | Complexity |
|---------------|---------|-------------|-------------------|----------|------------|
| F-001-RQ-001 | F-001 | Pattern-based file discovery with regex support | Discover files matching configurable patterns in <5 seconds for 10,000 files | Must-Have | Medium |
| F-001-RQ-002 | F-001 | Recursive and non-recursive search modes | Support both shallow and deep directory traversal | Must-Have | Low |
| F-001-RQ-003 | F-001 | Metadata extraction from filenames | Extract experiment metadata using regex patterns | Should-Have | Medium |
| F-002-RQ-001 | F-002 | Pydantic-based configuration validation | 100% validation coverage for all configuration paths | Must-Have | High |
| F-002-RQ-002 | F-002 | Hierarchical configuration inheritance | Support nested configuration with environment overrides | Must-Have | Medium |
| F-002-RQ-003 | F-002 | Legacy configuration adapter | Maintain backward compatibility with dict-based configs | Must-Have | High |
| F-003-RQ-001 | F-003 | Registry-based loader selection | O(1) lookup performance for format detection | Must-Have | Medium |
| F-003-RQ-002 | F-003 | Multiple format support | Support .pkl, .pklz, .pkl.gz formats out of box | Must-Have | Low |
| F-003-RQ-003 | F-003 | Memory-efficient streaming | Load large files without exceeding 2x memory overhead | Must-Have | High |

#### Extended Feature Requirements

| Requirement ID | Feature | Description | Acceptance Criteria | Priority | Complexity |
|---------------|---------|-------------|-------------------|----------|------------|
| F-004-RQ-001 | F-004 | DataFrame transformation pipeline | Convert raw data to pandas DataFrames with schema validation | Must-Have | High |
| F-004-RQ-002 | F-004 | Column mapping and normalization | Support configurable column transformations | Must-Have | Medium |
| F-004-RQ-003 | F-004 | Kedro compatibility enforcement | All output DataFrames include required Kedro metadata columns | Must-Have | Medium |
| F-005-RQ-001 | F-005 | Manifest generation without loading | Create file inventories in <100ms without data loading | Should-Have | Medium |
| F-005-RQ-002 | F-005 | Selective file processing | Enable processing of file subsets based on manifest | Should-Have | Low |
| F-006-RQ-001 | F-006 | Plugin registration system | Support automatic plugin discovery via entry points | Must-Have | High |
| F-006-RQ-002 | F-006 | Thread-safe registry operations | Maintain data integrity under concurrent access | Must-Have | High |
| F-006-RQ-003 | F-006 | Priority-based resolution | Implement BUILTIN<USER<PLUGIN<OVERRIDE priority system | Should-Have | Medium |

#### Integration and Infrastructure Requirements

| Requirement ID | Feature | Description | Acceptance Criteria | Priority | Complexity |
|---------------|---------|-------------|-------------------|----------|------------|
| F-007-RQ-001 | F-007 | Comprehensive error hierarchy | Provide specific exception types for all error scenarios | Must-Have | Medium |
| F-007-RQ-002 | F-007 | Context preservation in errors | Include full context and actionable error codes | Must-Have | Low |
| F-008-RQ-001 | F-008 | Structured logging with rotation | Implement automatic log rotation with configurable levels | Should-Have | Low |
| F-011-RQ-001 | F-011 | Kedro DataSet implementation | Implement AbstractDataset interface for catalog integration | Must-Have | High |
| F-011-RQ-002 | F-011 | Catalog YAML configuration | Support native catalog.yml configuration syntax | Must-Have | Medium |
| F-012-RQ-001 | F-012 | Automatic version detection | Detect configuration version mismatches automatically | Must-Have | Medium |
| F-012-RQ-002 | F-012 | Migration execution | Execute configuration migrations with rollback capability | Must-Have | High |

### 2.1.3 FEATURE RELATIONSHIPS

#### Dependency Map

```mermaid
graph TB
    F001[F-001: File Discovery] --> F005[F-005: Manifest Discovery]
    F002[F-002: Configuration] --> F001
    F002 --> F003[F-003: Data Loading]
    F002 --> F004[F-004: Transformation]
    F002 --> F012[F-012: Version Migration]
    F006[F-006: Plugin Architecture] --> F003
    F001 --> F003
    F003 --> F004
    F008[F-008: Logging] --> F007[F-007: Error Handling]
    F007 --> F001
    F007 --> F003
    F007 --> F004
    F010[F-010: Test Infrastructure] --> F001
    F010 --> F003
    F010 --> F004
    F009[F-009: Path Utilities] --> F001
    F009 --> F003
    F001 --> F011[F-011: Kedro Integration]
    F002 --> F011
    F003 --> F011
    F004 --> F011
    F012 --> F002
```

#### Integration Points

| Integration Point | Primary Feature | Secondary Feature | Integration Type | Description |
|------------------|----------------|------------------|------------------|-------------|
| Configuration-Discovery | F-002 | F-001 | Data Flow | Configuration provides search patterns and filters |
| Discovery-Loading | F-001 | F-003 | Control Flow | Discovery results drive loader selection |
| Loading-Transformation | F-003 | F-004 | Data Pipeline | Raw data flows through transformation pipeline |
| Plugin-Loading | F-006 | F-003 | Service Registry | Plugin system provides loader implementations |
| Error-All Features | F-007 | All | Cross-Cutting | Error handling spans all functional areas |
| Configuration-Migration | F-002 | F-012 | Service Integration | Migration system updates configuration models |
| Kedro-Core Pipeline | F-011 | F-001,F-003,F-004 | Workflow Integration | Kedro wraps core data processing pipeline |

#### Shared Components

| Component | Used By Features | Purpose | Technical Details |
|-----------|-----------------|---------|------------------|
| Registry Pattern | F-003, F-006 | Extensible service location | Thread-safe singleton with priority resolution |
| Pydantic Models | F-002, F-004, F-012 | Type-safe data validation | Shared base classes and validation rules |
| Protocol Interfaces | F-003, F-006, F-010 | Dependency injection | BaseLoader, ConfigProvider protocols |
| Exception Hierarchy | F-007, All Features | Structured error handling | Domain-specific exception types with context |
| Path Utilities | F-001, F-003, F-009 | Cross-platform file operations | Secure path manipulation and validation |

### 2.1.4 IMPLEMENTATION CONSIDERATIONS

#### Technical Constraints

| Feature | Constraint Type | Description | Impact | Mitigation Strategy |
|---------|---------------|-------------|--------|-------------------|
| F-001 | Performance | <5 second discovery for 10,000 files | High | Implement async I/O and caching |
| F-003 | Memory | <2x data size memory overhead | High | Streaming loaders with lazy evaluation |
| F-004 | Compatibility | Kedro DataFrame requirements | Medium | Mandatory metadata column injection |
| F-006 | Concurrency | Thread-safe registry operations | High | RWLock implementation with deadlock prevention |
| F-011 | Integration | Kedro AbstractDataset compliance | Medium | Strict interface adherence and testing |
| F-012 | Reliability | Zero-downtime configuration migration | High | Atomic migration with rollback capability |

#### Performance Requirements

| Feature | Metric | Target | Measurement Method | Baseline |
|---------|--------|--------|------------------|----------|
| F-001 | Discovery Time | <5s for 10,000 files | pytest-benchmark | Current: 12s |
| F-003 | Loading Speed | <1s per 100MB | Memory profiler | Current: 3s |
| F-004 | Transform Speed | <2s per 1M rows | Pandas profiling | Current: 8s |
| F-005 | Manifest Generation | <100ms | Response time monitoring | Current: 300ms |
| F-006 | Registry Lookup | O(1) performance | Algorithm analysis | Current: O(n) |
| F-011 | Kedro Integration | <10% overhead | Pipeline benchmarking | N/A (new) |

#### Scalability Considerations

| Feature | Scalability Dimension | Current Limit | Target Limit | Scaling Strategy |
|---------|---------------------|---------------|--------------|------------------|
| F-001 | File Count | 1,000 files | 100,000 files | Parallel directory scanning |
| F-003 | File Size | 1GB per file | 10GB per file | Streaming with chunked processing |
| F-004 | DataFrame Size | 10M rows | 100M rows | Columnar processing with Dask integration |
| F-006 | Plugin Count | 5 plugins | 50 plugins | Hierarchical plugin namespaces |
| F-011 | Pipeline Nodes | 10 nodes | 1000 nodes | Distributed Kedro execution |

#### Security Implications

| Feature | Security Risk | Impact Level | Mitigation | Validation Method |
|---------|--------------|--------------|------------|------------------|
| F-001 | Path Traversal | High | Path validation and sandboxing | Security audit |
| F-003 | Arbitrary Code Execution | Critical | Pickle safety checks and allowlisting | Static analysis |
| F-006 | Malicious Plugins | High | Plugin signature verification | Code review |
| F-009 | Directory Access | Medium | Permission boundary enforcement | Integration tests |
| F-012 | Configuration Injection | Medium | Schema validation and sanitization | Penetration testing |

#### Maintenance Requirements

| Feature | Maintenance Type | Frequency | Effort Level | Automation Level |
|---------|-----------------|-----------|--------------|------------------|
| F-002 | Configuration Schema Updates | Quarterly | Medium | Automated migration generation |
| F-006 | Plugin Compatibility Testing | Per release | High | Automated plugin test suite |
| F-007 | Error Message Localization | Annual | Low | Template-based message system |
| F-008 | Log Retention Policy | Monthly | Low | Automated cleanup scripts |
| F-011 | Kedro Version Compatibility | Per Kedro release | Medium | Automated compatibility matrix |
| F-012 | Migration Path Testing | Per version | High | Automated migration test suite |

## 2.2 FUNCTIONAL REQUIREMENTS TABLE

### 2.2.1 F-001: Pattern-Based File Discovery System

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-001-RQ-001 | Support glob pattern matching | System accepts patterns like "*.pkl", "data_*.csv" | Must-Have |
| F-001-RQ-002 | Recursive directory traversal | Can search subdirectories when recursive=True | Must-Have |
| F-001-RQ-003 | Extension-based filtering | Filter results by file extensions list | Must-Have |
| F-001-RQ-004 | Metadata extraction from filenames | Extract structured data using regex patterns | Should-Have |
| F-001-RQ-005 | Date parsing from filenames | Parse dates in multiple formats (ISO, US, EU) | Should-Have |
| F-001-RQ-006 | Ignore pattern support | Exclude files matching ignore patterns | Must-Have |
| F-001-RQ-007 | Mandatory substring filtering | Include only files containing specific substrings | Should-Have |
| F-001-RQ-008 | File statistics attachment | Attach size, mtime, permissions to results | Should-Have |

**Technical Specifications**
- **Input Parameters**: directory paths, glob patterns, recursive flag, extensions list, metadata flags
- **Output/Response**: List of file paths or dictionary with metadata
- **Performance Criteria**: <5 seconds for 10,000 files
- **Data Requirements**: Read access to target directories

### 2.2.2 F-002: Configuration Management System (updated)

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-002-RQ-001 | Load YAML configuration files | Parse valid YAML without errors | Must-Have |
| F-002-RQ-002 | Validate with Pydantic models | Enforce type constraints and required fields | Must-Have |
| F-002-RQ-003 | Support environment overrides | FLYRIGLOADER_DATA_DIR overrides config | Must-Have |
| F-002-RQ-004 | Path traversal protection | Reject paths with ".." components | Must-Have |
| F-002-RQ-005 | Pattern validation | Compile and validate regex patterns | Must-Have |
| F-002-RQ-006 | Hierarchical configuration | Support project/dataset/experiment hierarchy | Must-Have |
| F-002-RQ-007 | Legacy adapter support | Provide dict-like access to Pydantic models | Should-Have |
| F-002-RQ-008 | Configuration builders | Programmatic config creation functions | Could-Have |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-002-RQ-009</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Embed schema_version field in all config models</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">All ProjectConfig, DatasetConfig, ExperimentConfig include schema_version field</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-002-RQ-010</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic version detection and migration on load</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">load_config() detects version mismatches and applies migration automatically</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |

**Technical Specifications**
- **Input Parameters**: YAML file path or configuration dictionary
- **Output/Response**: Validated configuration object (Pydantic model or LegacyAdapter)
- **Performance Criteria**: <100ms load time for typical configs
- **Data Requirements**: Valid YAML syntax, required fields present

### 2.2.3 F-003: Registry-Based Data Loading Pipeline

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-003-RQ-001 | Auto-detect file format | Select loader based on file extension | Must-Have |
| F-003-RQ-002 | Load uncompressed pickle | Support .pkl files | Must-Have |
| F-003-RQ-003 | Load compressed pickle | Support .pklz and .pkl.gz files | Must-Have |
| F-003-RQ-004 | Registry-based lookup | O(1) loader resolution by extension | Must-Have |
| F-003-RQ-005 | Format validation | Verify loaded data structure | Should-Have |
| F-003-RQ-006 | Custom loader registration | Allow runtime loader registration | Must-Have |
| F-003-RQ-007 | Error context preservation | Include file path and format in errors | Must-Have |
| F-003-RQ-008 | Memory-efficient loading | Stream large files without full memory load | Could-Have |

**Technical Specifications**
- **Input Parameters**: file path, optional loader override
- **Output/Response**: Raw data dictionary
- **Performance Criteria**: <1 second per 100MB file
- **Data Requirements**: Valid pickle format data

### 2.2.4 F-004: DataFrame Transformation Engine (updated)

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-004-RQ-001 | Apply column configuration | Map raw data to schema-defined columns | Must-Have |
| F-004-RQ-002 | Validate column types | Ensure numpy.ndarray types match schema | Must-Have |
| F-004-RQ-003 | Handle array dimensions | Transform arrays to match time dimension | Must-Have |
| F-004-RQ-004 | Extract 2D columns | Convert 2D arrays to 1D when specified | Should-Have |
| F-004-RQ-005 | Add metadata columns | Include file_path and other metadata | Must-Have |
| F-004-RQ-006 | Strict schema mode | Drop columns not in schema when enabled | Should-Have |
| F-004-RQ-007 | Custom handler support | Apply special transformations per column | Must-Have |
| F-004-RQ-008 | Missing column handling | Provide clear errors for required columns | Must-Have |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-004-RQ-009</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Ensure DataFrame outputs are Kedro-compatible</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">All DataFrames include mandatory metadata columns required by Kedro pipelines</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Should-Have</span> |

**Technical Specifications**
- **Input Parameters**: raw data dict, column config, metadata, strict mode flag
- **Output/Response**: pandas DataFrame
- **Performance Criteria**: <2x data size memory overhead
- **Data Requirements**: Data structure matching column schema

### 2.2.5 F-005: Experiment Manifest Discovery

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-005-RQ-001 | Generate file manifests | Create comprehensive file lists without loading | Must-Have |
| F-005-RQ-002 | Include file metadata | Attach size, modification time, permissions | Must-Have |
| F-005-RQ-003 | Support selective processing | Enable filtering by file patterns | Must-Have |
| F-005-RQ-004 | Memory-efficient operation | Generate manifests without loading data | Must-Have |
| F-005-RQ-005 | Experiment organization | Group files by experiment context | Should-Have |
| F-005-RQ-006 | Preview capabilities | Show file contents summary | Could-Have |
| F-005-RQ-007 | Manifest serialization | Save/load manifest objects | Could-Have |
| F-005-RQ-008 | Directory structure validation | Verify expected file organization | Should-Have |

**Technical Specifications**
- **Input Parameters**: experiment directory path, discovery configuration
- **Output/Response**: FileManifest objects with metadata
- **Performance Criteria**: <100ms response time for manifest operations
- **Data Requirements**: Read access to experiment directories

### 2.2.6 F-006: Plugin Architecture System (updated)

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-006-RQ-001 | Registry-based plugin system | Support dynamic loader registration | Must-Have |
| F-006-RQ-002 | Entry point discovery | Automatic plugin discovery via setuptools | Must-Have |
| F-006-RQ-003 | Protocol-based interfaces | Define clear extension contracts | Must-Have |
| F-006-RQ-004 | Thread-safe operations | Support concurrent plugin access | Must-Have |
| F-006-RQ-005 | Runtime registration | Allow decorator-based registration | Should-Have |
| F-006-RQ-006 | Plugin validation | Verify plugin compatibility | Should-Have |
| F-006-RQ-007 | Namespace isolation | Prevent plugin conflicts | Should-Have |
| F-006-RQ-008 | Error handling | Graceful plugin failure handling | Must-Have |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-006-RQ-009</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Priority system for resolving multiple loaders per extension</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safe priority resolution using BUILTIN<USER<PLUGIN<OVERRIDE hierarchy</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |

**Technical Specifications**
- **Input Parameters**: plugin modules, entry point specifications
- **Output/Response**: Registered plugin instances
- **Performance Criteria**: O(1) plugin lookup performance
- **Data Requirements**: Valid plugin implementations

### 2.2.7 F-007: Comprehensive Error Handling

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-007-RQ-001 | Domain-specific exceptions | Four main exception types with context | Must-Have |
| F-007-RQ-002 | Error context preservation | Include file paths, operation details | Must-Have |
| F-007-RQ-003 | Structured error codes | Programmatic error identification | Must-Have |
| F-007-RQ-004 | Chained exception support | Preserve original exception context | Must-Have |
| F-007-RQ-005 | Logging integration | Automatic error logging with context | Should-Have |
| F-007-RQ-006 | Recovery suggestions | Provide actionable error messages | Should-Have |
| F-007-RQ-007 | Security considerations | Avoid sensitive information in errors | Must-Have |
| F-007-RQ-008 | Testing support | Mock-friendly error simulation | Could-Have |

**Technical Specifications**
- **Input Parameters**: error context, original exceptions, operation details
- **Output/Response**: Structured exception objects
- **Performance Criteria**: Minimal overhead on success path
- **Data Requirements**: Error context and operation metadata

### 2.2.8 F-008: Structured Logging System

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-008-RQ-001 | Loguru-based logging | Use loguru as primary logging framework | Must-Have |
| F-008-RQ-002 | Automatic log rotation | Rotate logs at 10MB size limit | Must-Have |
| F-008-RQ-003 | Multiple output targets | Support console and file logging | Must-Have |
| F-008-RQ-004 | Configurable verbosity | Support different log levels | Must-Have |
| F-008-RQ-005 | Test mode detection | Special behavior during testing | Should-Have |
| F-008-RQ-006 | Structured output | JSON-compatible log formatting | Should-Have |
| F-008-RQ-007 | Context preservation | Include operation context in logs | Should-Have |
| F-008-RQ-008 | Performance monitoring | Log performance metrics | Could-Have |

**Technical Specifications**
- **Input Parameters**: log level, message content, context data
- **Output/Response**: Formatted log entries
- **Performance Criteria**: <1ms logging overhead
- **Data Requirements**: Write access to logs directory

### 2.2.9 F-009: Path Management Utilities

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-009-RQ-001 | Cross-platform compatibility | Work on Linux, macOS, Windows | Must-Have |
| F-009-RQ-002 | Path traversal protection | Validate against malicious paths | Must-Have |
| F-009-RQ-003 | Automatic directory creation | Create parent directories as needed | Must-Have |
| F-009-RQ-004 | Relative/absolute conversion | Support both path types | Must-Have |
| F-009-RQ-005 | Existence checking | Verify file/directory existence | Should-Have |
| F-009-RQ-006 | Permission validation | Check read/write permissions | Should-Have |
| F-009-RQ-007 | Path normalization | Standardize path representations | Should-Have |
| F-009-RQ-008 | Symlink handling | Resolve symbolic links safely | Could-Have |

**Technical Specifications**
- **Input Parameters**: path strings, operation type, validation flags
- **Output/Response**: Validated Path objects
- **Performance Criteria**: <10ms path operations
- **Data Requirements**: File system access permissions

### 2.2.10 F-010: Test Infrastructure & Hooks

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| F-010-RQ-001 | Dependency injection | Protocol-based mock injection | Must-Have |
| F-010-RQ-002 | Mock providers | File system and I/O mocking | Must-Have |
| F-010-RQ-003 | Test mode detection | Automatic test environment detection | Must-Have |
| F-010-RQ-004 | Deterministic behavior | Reproducible test results | Must-Have |
| F-010-RQ-005 | Pytest integration | Seamless pytest fixture support | Must-Have |
| F-010-RQ-006 | Coverage measurement | Support coverage analysis | Should-Have |
| F-010-RQ-007 | Performance testing | Benchmark test capabilities | Should-Have |
| F-010-RQ-008 | Mock validation | Verify mock usage correctness | Could-Have |

**Technical Specifications**
- **Input Parameters**: test configuration, mock specifications
- **Output/Response**: Test-ready mock objects
- **Performance Criteria**: <50ms test setup time
- **Data Requirements**: Test environment detection

### 2.2.11 F-011: Kedro Integration Layer

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-001</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Provide FlyRigLoaderDataSet subclass of Kedro AbstractDataset</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Implement AbstractDataset interface with _load, _save, _exists, _describe methods</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-002</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Support catalog.yml configuration with no user code</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Enable direct catalog.yml entry with filepath and experiment_name parameters</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-003</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Accept Kedro parameters for configuration injection</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Support ${base_dir} and other Kedro parameter interpolation</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Should-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-004</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Dataset is read-only; _save must raise NotImplementedError</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">_save method throws NotImplementedError with clear message</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-005</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro dependency is optional; import failure must raise informative error</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Clear error message when kedro not installed, suggesting pip install flyrigloader[kedro]</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Should-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-006</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Support FlyRigManifestDataSet variant for manifest-only operations</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Return FileManifest objects without loading data</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Should-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-007</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Integration with Kedro versioning and lineage tracking</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Support Kedro data versioning patterns and metadata extraction</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Could-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-008</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro lifecycle hooks for FlyRigLoader</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Provide hooks for pipeline integration and logging</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Could-Have</span> |

**Technical Specifications**
- **Input Parameters**: filepath (config path), experiment_name, optional kwargs for configuration
- **Output/Response**: FileManifest or pandas DataFrame depending on dataset variant
- **Performance Criteria**: Compatible with Kedro pipeline performance requirements
- **Data Requirements**: Valid FlyRigLoader configuration files, Kedro project structure

### 2.2.12 F-012: Configuration Version Management & Migration

| Requirement ID | Description | Acceptance Criteria | Priority |
|---------------|-------------|-------------------|----------|
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-001</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">All configuration models expose schema_version field</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">ProjectConfig, DatasetConfig, ExperimentConfig include schema_version with default "1.0.0"</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-002</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">load_config() auto-detects version and migrates to current</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic version detection from YAML and migration on version mismatch</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-003</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">VersionError raised when migration impossible</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Clear VersionError with context when no migration path exists</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-004</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Provide migrate_config() utility for manual migrations</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Standalone migration function for explicit version upgrades</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Should-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-005</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">ConfigVersion enum for version constants</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Centralized version constants with compatibility matrix</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-006</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration chain support for multi-version upgrades</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Apply sequential migrations from any supported version to current</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Should-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-007</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">LegacyConfigAdapter integration for backward compatibility</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Seamless integration with existing LegacyConfigAdapter for dict-based configs</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must-Have</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-008</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Validation of migrated configurations</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Post-migration validation ensures config integrity and completeness</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Should-Have</span> |

**Technical Specifications**
- **Input Parameters**: configuration dict/YAML, source version, target version
- **Output/Response**: Migrated and validated configuration objects
- **Performance Criteria**: <200ms migration time for typical configurations
- **Data Requirements**: Valid source configuration, supported version format

## 2.3 FEATURE RELATIONSHIPS

### 2.3.1 Feature Dependency Map (updated)

```mermaid
graph TB
    subgraph "Foundation Layer"
        F002[F-002: Configuration Management]
        F006[F-006: Plugin Architecture]
        F008[F-008: Structured Logging]
        F009[F-009: Path Management]
        F012[F-012: Version Management & Migration]
    end
    
    subgraph "Core Pipeline"
        F001[F-001: File Discovery]
        F003[F-003: Data Loading]
        F004[F-004: Transformation]
        F005[F-005: Manifest Discovery]
    end
    
    subgraph "Integration Layer"
        F011[F-011: Kedro Integration]
    end
    
    subgraph "Support Systems"
        F007[F-007: Error Handling]
        F010[F-010: Test Infrastructure]
    end
    
    F002 --> F001
    F002 --> F005
    F002 --> F011
    F002 --> F012
    F006 --> F003
    F001 --> F003
    F001 --> F005
    F003 --> F004
    F003 --> F011
    F008 --> F007
    F007 --> F001
    F007 --> F003
    F007 --> F004
    F009 --> F001
    F009 --> F002
    F010 --> F001
    F010 --> F002
    F010 --> F003
    F010 --> F004
    F011 --> F004
    F012 --> F002
```

### 2.3.2 Integration Points (updated)

| Feature A | Feature B | Integration Type | Description |
|-----------|-----------|-----------------|-------------|
| Configuration | File Discovery | Data Flow | Provides patterns, filters, and search configuration |
| File Discovery | Manifest Discovery | Composition | Manifest uses discovery engine for file enumeration |
| Plugin Architecture | Data Loading | Registry Access | Loaders register via plugin system for format support |
| Data Loading | Transformation | Pipeline | Raw data flows to transformer with metadata |
| Error Handling | All Features | Exception Flow | All features raise domain-specific exceptions |
| Logging | All Features | Service Access | All features log operations and performance metrics |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration (F-011)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Data Loading (F-003)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Pipeline</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">DataSet triggers loader via API</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration (F-011)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Transformation (F-004)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Pipeline</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Ensures DataFrames meet Kedro expectations</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Version Management (F-012)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration (F-002)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Validation</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Handles automatic migration & compatibility</span> |

### 2.3.3 Shared Components (updated)

| Component | Used By | Purpose |
|-----------|---------|---------|
| Registry System | F-003, F-004, F-006 | Format lookup, plugin management |
| Dependency Injection | F-001, F-003, F-004, F-010 | Testability and extensibility |
| Protocol Interfaces | F-003, F-006, F-010 | Contract definition and validation |
| Path Validation | F-002, F-009 | Security and cross-platform compatibility |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Migration Infrastructure</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">F-002, F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Handles version detection and automatic migration</span> |

### 2.3.4 Cross-Feature Data Flow

The feature relationships form three distinct workflow patterns that enable comprehensive data processing:

**Primary Data Pipeline Flow**: F-002 → F-001 → F-003 → F-004, where configuration drives discovery, which feeds loading, culminating in transformation to standardized DataFrames.

**Kedro Integration Flow**: <span style="background-color: rgba(91, 57, 243, 0.2)">F-002 → F-011 ← F-003, with F-011 → F-004</span>, enabling seamless integration with Kedro pipelines through the FlyRigLoaderDataSet interface that coordinates configuration-driven loading with Kedro-compatible DataFrame output.

**Configuration Management Flow**: <span style="background-color: rgba(91, 57, 243, 0.2)">F-002 ↔ F-012</span>, providing bidirectional version management where F-012 detects schema versions and applies automatic migrations back to F-002, ensuring backward compatibility across system upgrades.

**Service Layer Support**: F-006, F-007, F-008, F-009, and F-010 provide horizontal services across all features, with F-006 enabling extensibility, F-007 providing error context, F-008 ensuring observability, F-009 handling path operations, and F-010 supporting comprehensive testing.

### 2.3.5 Integration Constraints

| Constraint Type | Affecting Features | Requirements | Validation |
|----------------|-------------------|--------------|------------|
| Data Format | F-003, F-004, F-011 | DataFrames must include mandatory metadata columns | Schema validation at transformation |
| Configuration Schema | F-002, F-012 | All configs must include embedded schema_version field | Version detection on load |
| Plugin Compatibility | F-003, F-006 | Loaders must implement BaseLoader protocol | Protocol validation at registration |
| Error Context | F-007, All Features | Domain-specific exceptions with preserved context | Error handling pattern compliance |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Compatibility</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">F-004, F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">DataFrames must meet Kedro pipeline requirements</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">AbstractDataset interface compliance</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Version Migration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">F-002, F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration paths must preserve configuration integrity</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Post-migration validation and compatibility checks</span> |

## 2.4 IMPLEMENTATION CONSIDERATIONS

### 2.4.1 Technical Constraints

| Feature | Constraint Type | Description | Impact |
|---------|----------------|-------------|---------|
| F-001 | Performance | File system limits with deep recursion | Implement lazy evaluation |
| F-002 | Memory | YAML parsing limits for large configs | Stream processing for large files |
| F-003 | Security | Pickle deserialization risks | Validate data sources |
| F-004 | Memory | pandas overhead for large datasets | Implement chunked processing |
| F-006 | Threading | Plugin registry thread safety <span style="background-color: rgba(91, 57, 243, 0.2)">with loader priority resolution mechanism</span> | Use thread-safe singletons |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Dependency</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Optional Kedro dependency</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Graceful fallback when Kedro unavailable</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration must be idempotent and reversible</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Implement rollback capabilities</span> |

### 2.4.2 Performance Requirements

| Feature | Requirement | Target | Validation Method |
|---------|-------------|--------|-------------------|
| F-001 | Discovery Speed | <5s for 10,000 files | pytest-benchmark |
| F-002 | Config Load | <100ms typical configs | Performance profiling |
| F-003 | Loading Speed | <1s per 100MB | Memory profiler |
| F-004 | Transform Speed | <2x memory overhead | Resource monitoring |
| F-005 | Manifest Speed | <100ms response | Response time testing |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Dataset Instantiation</span> | <span style="background-color: rgba(91, 57, 243, 0.2)"><2 ms</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro benchmark suite</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration Processing</span> | <span style="background-color: rgba(91, 57, 243, 0.2)"><50 ms per config</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration timing tests</span> |

### 2.4.3 Scalability Considerations

| Feature | Scalability Factor | Implementation Strategy |
|---------|-------------------|-------------------------|
| F-001 | File count growth | Implement result streaming and pagination |
| F-002 | Config complexity | Use hierarchical validation caching |
| F-003 | Data size growth | Implement chunked loading and memory mapping |
| F-004 | Column complexity | Use lazy evaluation and selective processing |
| F-006 | Plugin ecosystem | Implement namespace isolation and validation |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro pipeline scale</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Support distributed execution and node parallelization</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration evolution</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Implement forward/backward migration chains with caching</span> |

### 2.4.4 Security Implications

| Feature | Security Concern | Mitigation Strategy |
|---------|------------------|-------------------|
| F-001 | Path traversal | Validate all path inputs against ".." components |
| F-002 | Config injection | Use Pydantic validation with strict mode |
| F-003 | Code execution | Validate pickle sources and implement sandboxing |
| F-007 | Information leakage | Sanitize error messages for sensitive data |
| F-009 | File system access | Implement permission checking and path validation |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Catalog parameter injection</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Ensure no arbitrary code execution via catalog parameters</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Version validation</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Reject configs with unknown schema_version</span> |

### 2.4.5 Maintenance Requirements

| Feature | Maintenance Task | Frequency | Automation Level |
|---------|------------------|-----------|------------------|
| F-001 | Pattern library updates | As needed | Manual with validation |
| F-002 | Schema migration | Version changes | Semi-automated |
| F-003 | Format compatibility | New versions | Automated testing |
| F-004 | Column schema updates | Data changes | Configuration-driven |
| F-006 | Plugin compatibility | Python updates | Automated CI/CD |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro compatibility</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro minor releases</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Automated testing with CI/CD</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Version compatibility matrix</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Every minor schema change</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Manual documentation updates</span> |

## 2.5 TRACEABILITY MATRIX

### 2.5.1 Feature to Business Value Mapping

| Feature ID | Business Value | Success Metric | Validation |
|------------|---------------|----------------|------------|
| F-001 | 90% reduction in file management overhead | Discovery time <5s | Performance benchmarks |
| F-002 | 90% reduction in configuration errors | Error rate <1% | Error tracking |
| F-003 | 75% reduction in format extension time | Plugin adoption 25% | Usage analytics |
| F-004 | 80% reduction in data preparation time | Processing speed 5x | Performance comparison |
| F-005 | 5x improvement in memory efficiency | Memory usage <2x | Resource monitoring |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Enables Kedro adoption in research pipelines</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration time reduced to <30 min</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Integration time tracking</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Guarantees zero breaking changes through automatic migration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">100% legacy configs load without manual edits</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration success rate testing</span> |

### 2.5.2 Requirements to Test Coverage

| Requirement Pattern | Test Coverage Target | Current Status |
|-------------------|---------------------|----------------|
| F-001-RQ-* | 100% critical path | Completed |
| F-002-RQ-* | 100% validation logic | Completed |
| F-003-RQ-* | 95% loading scenarios | Completed |
| F-004-RQ-* | 90% transformation cases | Completed |
| F-005-RQ-* | 85% manifest operations | Completed |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-011-RQ-*</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">90% integration test coverage</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Planned</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">F-012-RQ-*</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">95% migration path coverage</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Planned</span> |

### 2.5.3 Performance Requirements Traceability

| Performance Target | Related Features | Measurement Method | Status |
|-------------------|------------------|-------------------|--------|
| <5s discovery for 10K files | F-001 | pytest-benchmark | Met |
| <1s per 100MB loading | F-003 | Memory profiler | Met |
| <2x memory overhead | F-004 | Resource monitoring | Met |
| <100ms manifest response | F-005 | Response time testing | Met |
| O(1) plugin lookup | F-006 | Algorithm analysis | Met |
| <span style="background-color: rgba(91, 57, 243, 0.2)"><2 ms dataset instantiation</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">F-011</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">pytest-benchmark</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Planned</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)"><50 ms migration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">F-012</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Unit tests</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Planned</span> |

### 2.5.4 Cross-Feature Integration Matrix

| Primary Feature | Dependent Features | Integration Points | Validation Method |
|----------------|-------------------|-------------------|-------------------|
| F-001 | F-002, F-005, F-011 | Configuration patterns, manifest generation, Kedro catalog | Integration tests |
| F-002 | F-001, F-003, F-004, F-012 | Pattern definitions, loader configs, transform rules, migration | Configuration validation |
| F-003 | F-001, F-004, F-006 | File discovery input, DataFrame output, plugin registry | End-to-end pipeline tests |
| F-004 | F-003, F-011 | Raw data input, Kedro compatibility | Transform validation |
| F-011 | F-001, F-002, F-003, F-004 | Discovery integration, config compatibility, data loading, transform output | Kedro pipeline tests |
| F-012 | F-002 | Configuration management, legacy adapter | Migration test suite |

### 2.5.5 Compliance and Quality Gates

| Quality Gate | Coverage Requirement | Features Covered | Measurement |
|-------------|---------------------|------------------|-------------|
| Unit Test Coverage | ≥90% line coverage | All features | pytest-cov |
| Integration Test Coverage | ≥85% scenario coverage | F-001 through F-012 | Custom metrics |
| Performance Benchmarks | 100% SLA compliance | F-001, F-003, F-004, F-005, F-011, F-012 | Automated benchmarks |
| Security Validation | 100% path traversal protection | F-001, F-009 | Security test suite |
| Backward Compatibility | 100% legacy config support | F-002, F-012 | Migration validation |
| Documentation Coverage | 100% API documentation | All public interfaces | Automated doc generation |

## 2.6 REFERENCES

### 2.6.1 Source Files Examined
- `src/flyrigloader/api.py` - High-level API interfaces and feature integration
- `src/flyrigloader/config/yaml_config.py` - Configuration management implementation
- `src/flyrigloader/config/discovery.py` - Config-aware discovery logic
- `src/flyrigloader/discovery/files.py` - Core file discovery engine
- `src/flyrigloader/discovery/patterns.py` - Pattern matching and metadata extraction
- `src/flyrigloader/discovery/stats.py` - File statistics and metadata handling
- `src/flyrigloader/io/pickle.py` - Data loading pipeline and registry
- `src/flyrigloader/io/column_models.py` - Transformation engine and validation
- `src/flyrigloader/io/column_config.yaml` - Schema definitions and configuration
- `src/flyrigloader/utils/` - Path management and utility functions
- `tests/` - Test infrastructure and validation examples
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/kedro/datasets.py` - Kedro dataset implementations and integrations</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/kedro/catalog.py` - Kedro catalog management and configuration</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/migration/versions.py` - Version tracking and migration orchestration</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/migration/migrators.py` - Data migration utilities and transformers</span>

### 2.6.2 Documentation References
- `docs/architecture.md` - System architecture and design patterns
- `docs/migration_guide.md` - <span style="background-color: rgba(91, 57, 243, 0.2)">Version migration instructions</span>
- `examples/external_project/example_config.yaml` - Configuration examples
- `README.md` - Feature overview and usage patterns
- <span style="background-color: rgba(91, 57, 243, 0.2)">`docs/kedro_integration.md` - Kedro usage guide</span>

### 2.6.3 Technical Specifications
- Technical Specification Section 1.1 - Executive Summary and business context
- Technical Specification Section 1.2 - System Overview and capabilities
- Technical Specification Section 1.3 - Scope and boundaries

# 3. TECHNOLOGY STACK

## 3.1 PROGRAMMING LANGUAGES

### 3.1.1 Primary Language Selection

**Python 3.8-3.11** serves as the exclusive programming language for FlyRigLoader, with a strictly bounded version range to ensure compatibility across research environments.

#### 3.1.1.1 Version Constraints and Justification

The Python version range (3.8-3.11) was selected based on:
- **Research Environment Compatibility**: Academic HPC clusters typically maintain Python 3.8+ for stability
- **Feature Requirements**: Leverages modern Python features including Protocol classes, dataclasses, and type hints
- **Library Ecosystem**: Ensures compatibility with scientific Python stack (pandas, numpy, scipy)
- **Long-term Support**: Provides adequate support lifecycle for research reproducibility

#### 3.1.1.2 Development Standards

The codebase adheres to modern Python development practices:
- **Type Safety**: Full type annotation coverage with mypy static analysis
- **Code Quality**: Black formatting, isort import organization, and flake8 linting
- **Security**: Bandit security scanning for vulnerability detection
- **Performance**: Memory profiling and performance benchmarking integration

## 3.2 FRAMEWORKS & LIBRARIES

### 3.2.1 Core Data Validation Framework

**Pydantic v2.6+** provides the foundation for configuration management and data validation:
- **Schema Validation**: Type-safe configuration models with automatic validation
- **Error Handling**: Comprehensive validation error reporting for research workflows
- **Serialization**: JSON/YAML serialization for configuration persistence
- **Performance**: Optimized validation for large configuration files

### 3.2.2 Logging and Monitoring

**Loguru v0.7.0+** delivers structured logging capabilities:
- **Rotation Management**: Automatic log file rotation for long-running processes
- **Structured Output**: JSON-formatted logs for analysis and debugging
- **Context Preservation**: Maintains experimental context throughout processing pipelines
- **Performance Monitoring**: Integration with memory and performance profilers

### 3.2.3 Data Processing Stack

#### 3.2.3.1 Configuration Management
**PyYAML** handles YAML configuration parsing with security validations:
- **Safe Loading**: Prevents code execution during configuration parsing
- **Hierarchical Configs**: Supports nested configuration structures
- **Environment Overrides**: Dynamic configuration based on runtime environment

#### 3.2.3.2 Scientific Computing
**pandas v1.3.0+** and **numpy v1.21.0+** provide the data manipulation foundation:
- **DataFrame Operations**: Standardized data structures for neuroscience datasets
- **Memory Efficiency**: Optimized memory usage for large experimental datasets
- **Integration**: Seamless integration with Jupyter notebooks and Kedro pipelines

### 3.2.4 Architecture Patterns

The system implements several key architectural patterns:
- **Plugin Architecture**: Entry points system for extensible loaders and schemas
- **Protocol-Based Interfaces**: Dependency injection for testability and extensibility
- **Registry Pattern**: Dynamic component registration and discovery
- **Layered Architecture**: Clear separation between API, configuration, discovery, and I/O layers

### 3.2.5 Kedro Integration Library (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro v0.18.0+** provides optional but first-class integration framework support for production data pipeline workflows</span>:

- **<span style="background-color: rgba(91, 57, 243, 0.2)">FlyRigLoaderDataSet Implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Custom AbstractDataset wrapper enabling seamless integration with Kedro catalog systems</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Pipeline Workflow Support</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Native compatibility with Kedro's data versioning, lineage tracking, and pipeline execution framework</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Catalog Configuration</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Drop-in support for catalog.yml configuration with FlyRigLoader-specific parameters and lifecycle hooks</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Optional Installation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Available through optional dependency installation: `pip install flyrigloader[kedro]`</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">The integration maintains FlyRigLoader's core performance characteristics while providing enterprise-grade pipeline capabilities for research data workflows. All DataFrame outputs include mandatory Kedro metadata columns and conform to AbstractDataset interface requirements for seamless pipeline integration.</span>

### 3.2.6 Version Management Library (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">**semantic_version v2.10.0+** serves as the core library for schema version parsing, comparison, and compatibility validation</span>:

- **<span style="background-color: rgba(91, 57, 243, 0.2)">Schema Version Parsing</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Robust parsing of embedded schema_version fields in configuration files with semantic versioning compliance</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Compatibility Matrix</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated compatibility checking between configuration versions and library capabilities</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Planning</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Version comparison APIs enabling automatic migration path detection and execution</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Selection Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Chosen for its robust comparison API, zero external service dependencies, and full Python 3.8+ compatibility ensuring research environment stability</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">This library enables zero-breaking-change upgrades through automatic version detection and migration execution, supporting the system's commitment to backward compatibility for existing research workflows. The semantic versioning approach provides clear upgrade paths and rollback capabilities essential for reproducible scientific computing.</span>

## 3.3 OPEN SOURCE DEPENDENCIES

### 3.3.1 Testing Framework Ecosystem

**pytest v7.0.0+** with comprehensive plugin ecosystem:
- **pytest-cov v6.1.1+**: Code coverage reporting and analysis
- **pytest-mock v3.14.1+**: Mocking capabilities for isolated testing
- **pytest-benchmark v4.0.0+**: Performance testing and regression detection
- **pytest-xdist v3.7.0+**: Parallel test execution for faster CI/CD
- **pytest-timeout v2.3.0+**: Timeout management for long-running tests
- **hypothesis v6.131.9+**: Property-based testing for robust validation
- **coverage v7.8.2+**: Code coverage measurement and reporting

### 3.3.2 Code Quality Tools

**Development Quality Stack**:
- **black v24.3.0+**: Uncompromising code formatter for consistency
- **isort v5.12.0+**: Import sorting for maintainable code structure
- **mypy v1.8.0+**: Static type checking for type safety
- **flake8 v7.0.0+**: Linting for code quality enforcement
- **pre-commit v3.6.0+**: Git hooks for automated quality checks

### 3.3.3 Security and Performance

**Security Analysis**:
- **bandit**: Security vulnerability scanning for Python code
- **safety**: Dependency vulnerability checking and monitoring

**Performance Monitoring**:
- **memory-profiler**: Memory usage analysis for optimization
- **psutil**: System resource monitoring and profiling

### 3.3.4 Build and Packaging

**Modern Python Packaging**:
- **setuptools v42+**: PEP 517/518 compliant build system
- **wheel**: Binary distribution format for efficient installation
- **pip**: Package installation and dependency management

### 3.3.5 Integration Frameworks (updated)

**Production Data Pipeline Support**:
- <span style="background-color: rgba(91, 57, 243, 0.2)">**kedro v0.18.0+**: Optional dependency enabling first-class integration with Kedro data pipeline workflows</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Provides FlyRigLoaderDataSet implementation for seamless catalog integration</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Supports pipeline versioning, lineage tracking, and data lifecycle management</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Available through optional extra installation: `pip install flyrigloader[kedro]`</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Not installed by default to maintain minimal dependency footprint for research environments</span>

**Installation Note**: <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration is entirely optional and enabled only when explicitly requested through the `[kedro]` extra specification during installation. This ensures FlyRigLoader maintains its lightweight profile for standard research workflows while providing enterprise-grade pipeline capabilities when needed.</span>

### 3.3.6 Utility Libraries (updated)

**Version Management and Validation**:
- <span style="background-color: rgba(91, 57, 243, 0.2)">**semantic_version v2.10.0+**: Core library for configuration version parsing, comparison, and compatibility validation</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Handles embedded `schema_version` field parsing in configuration files</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Enables automatic compatibility checking between configuration versions and library capabilities</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Supports migration path detection and zero-breaking-change upgrade workflows</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Provides robust version comparison APIs essential for reproducible scientific computing</span>

## 3.4 THIRD-PARTY SERVICES

### 3.4.1 Service Architecture

FlyRigLoader operates as a **standalone library** with no external service dependencies:
- **Local File System Only**: All data operations target local or network-mounted file systems
- **No Authentication Services**: Library-level security through code validation only
- **No External APIs**: Self-contained functionality for research environments
- **No Cloud Dependencies**: Designed for on-premises research infrastructure

### 3.4.2 Integration Points

The library integrates with external systems through:
- **Kedro Pipelines**: First-class support for production data workflows
- **Jupyter Notebooks**: Interactive research and exploration interfaces
- **HPC Environments**: Memory-efficient processing for cluster computing

## 3.5 DATABASES & STORAGE

### 3.5.1 Storage Architecture

**File System-Based Persistence**:
- **Primary Storage**: Local file system with configurable directory structures
- **Data Formats**: Pickle files (including compressed `.pklz` format)
- **Configuration Storage**: YAML files for experiment and column configurations
- **No Database Requirements**: Eliminates database overhead for research workflows

### 3.5.2 Data Persistence Strategy

**Decoupled Storage Model**:
- **Discovery Layer**: Pattern-based file location without data loading
- **Loading Layer**: On-demand data loading with memory management
- **Transformation Layer**: In-memory processing with optional persistence
- **Caching**: Configuration and metadata caching for performance

## 3.6 DEVELOPMENT & DEPLOYMENT

### 3.6.1 Development Environment

**Environment Management**:
- **Primary**: Conda environment management with `environment.yml`
- **Channels**: conda-forge and defaults for scientific Python packages
- **Automation**: `setup_env.sh` script for environment initialization

**Development Tools Stack**:
- **IDE Support**: VSCode with Python extension, PyCharm Professional
- **Type Checking**: mypy v1.8.0+ for static analysis and type safety
- **Code Formatting**: black v24.3.0+ for consistent code style
- **Import Organization**: isort v5.12.0+ for maintainable import structure
- **Pre-commit Hooks**: Automated quality checks with pre-commit v3.6.0+

**Local Testing Infrastructure**:
- **pytest Configuration**: Comprehensive test discovery and execution
- **Coverage Analysis**: Real-time coverage reporting during development
- **Property-Based Testing**: hypothesis v6.131.9+ for robust validation
- **Performance Profiling**: memory-profiler and psutil for optimization

### 3.6.2 Continuous Integration & Deployment (updated)

**GitHub Actions Workflow**:
- **Multi-Platform Testing**: Ubuntu, Windows, and macOS runners
- **Python Matrix**: Testing across Python 3.8-3.11 versions
- **Coverage Reporting**: Codecov integration for coverage analysis
- **Automated Quality Checks**: Pre-commit hooks and automated linting

**<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration Testing</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Dedicated CI Job</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated installation of optional `[kedro]` extras during pipeline execution</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Integration Test Suite</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Execution of `tests/flyrigloader/test_kedro_integration.py` for FlyRigLoaderDataSet compatibility validation</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Coverage Integration</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration test coverage recorded and reported alongside core library metrics</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Compatibility Matrix</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Testing against multiple Kedro versions (0.18.0+) to ensure broad ecosystem compatibility</span>

**Security and Quality Pipeline**:
- **Vulnerability Scanning**: bandit for security analysis and safety for dependency checking
- **Performance Benchmarks**: pytest-benchmark integration for regression detection
- **Dependency Updates**: Dependabot integration for automated security updates
- **Documentation Builds**: Automated documentation generation and validation

**Release Automation**:
- **Semantic Versioning**: Automated version bumping based on conventional commits
- **PyPI Publishing**: Secure token-based publishing to Python Package Index
- **Release Notes**: Automated changelog generation from commit history
- **Distribution Validation**: Multi-platform wheel building and testing

### 3.6.3 Development Workflow

**Modern Development Practices**:
- **Version Control**: Git with GitHub for collaborative development
- **Documentation**: MkDocs/Sphinx for comprehensive documentation
- **Code Quality**: Automated formatting, linting, and type checking
- **Testing**: Comprehensive test suite with property-based testing

**Plugin Development Workflow**:
- **Entry Point Registration**: Standardized plugin discovery through `pyproject.toml` configuration
- **Protocol Compliance**: Type-safe plugin interfaces using Python Protocol classes
- **Development Templates**: Cookiecutter templates for rapid plugin development
- **Testing Framework**: Dedicated plugin testing utilities and fixtures

**Configuration Management**:
- **Schema Validation**: Pydantic v2.6+ models for type-safe configuration
- **Version Migration**: Automated configuration upgrades using semantic_version comparison
- **Environment Overrides**: Dynamic configuration based on development/production contexts
- **Validation Pipeline**: Multi-stage configuration validation with comprehensive error reporting

**Integration Testing Strategy**:
- **Kedro Pipeline Testing**: Comprehensive AbstractDataset implementation validation
- **Cross-Platform Compatibility**: Testing across Linux, Windows, and macOS environments
- **Memory Profiling**: Automated memory usage analysis for large dataset processing
- **Performance Benchmarking**: Continuous performance monitoring and regression detection

### 3.6.4 Deployment Architecture (updated)

**Library Distribution**:
- **Package Registry**: PyPI for public distribution
- **Local Installation**: Editable installs for development
- **Research Integration**: Direct integration with Kedro and Jupyter environments

**<span style="background-color: rgba(91, 57, 243, 0.2)">Optional Dependency Management</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Extras Group</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Optional `[kedro]` installation group available through `pip install flyrigloader[kedro]` for production pipeline workflows</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Plugin Extras Group</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Future `[plugins]` extras group for extended plugin development capabilities</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Minimal Core Installation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Base installation maintains lightweight dependency footprint for standard research workflows</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Plugin Discovery Architecture</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Entry Points System</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin registration through `entry_points` declared under `flyrigloader.plugins` group in `pyproject.toml`</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Dynamic Discovery</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Runtime plugin detection using `importlib.metadata` for extensible loader and schema support</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Registry Priority</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin priority system ensuring predictable behavior: BUILTIN < USER < PLUGIN < OVERRIDE</span>

**Containerization Strategy**:
- **Docker Support**: Multi-stage builds for development and production environments
- **Base Images**: Scientific Python stack with optimized layer caching
- **Volume Management**: Persistent storage for configuration and experiment data
- **Resource Optimization**: Memory and CPU constraints for HPC deployment

**Production Deployment Patterns**:
- **HPC Integration**: Memory-efficient processing for cluster computing environments
- **Jupyter Hub**: Pre-configured environments for collaborative research platforms
- **Kedro Pipelines**: Production-ready data processing workflows with versioning and lineage
- **Configuration Management**: Infrastructure-as-code for reproducible deployment scenarios

**Performance and Monitoring**:
- **Memory Profiling**: psutil integration for resource usage monitoring
- **Performance Benchmarks**: Automated performance regression detection in CI/CD
- **Logging Infrastructure**: Structured logging with loguru for debugging and analysis
- **Error Tracking**: Comprehensive error hierarchy with context preservation

**Security Considerations**:
- **Dependency Scanning**: Automated vulnerability detection for all dependencies
- **Safe Configuration Loading**: YAML safe loading to prevent code execution during parsing
- **Access Control**: File system permission validation for data directory access
- **Version Pinning**: Semantic versioning constraints for reproducible research environments

## 3.7 TECHNOLOGY STACK ARCHITECTURE

### 3.7.1 Integration Architecture (updated)

```mermaid
graph TB
    subgraph "Development Tools"
        A[GitHub Actions] --> B[pytest Framework]
        B --> C[Code Quality Tools]
        C --> D[Conda Environment]
    end
    
    subgraph "Core Libraries"
        E[Pydantic v2.6+] --> F[PyYAML]
        F --> G[pandas/numpy]
        G --> H[Loguru]
        H --> I[semantic_version v2.10+]
    end
    
    subgraph "Migration Infrastructure"
        J[versions.py] --> K[migrators.py]
        K --> L[validators.py]
        L --> M[Configuration System]
    end
    
    subgraph "Integration Framework"
        N[Kedro v0.18.0+] --> O[FlyRigLoaderDataSet]
        O --> P[Pipeline Workflows]
    end
    
    subgraph "Python Runtime"
        Q[Python 3.8-3.11] --> E
        Q --> J
        Q --> N
        Q --> R[setuptools/wheel]
    end
    
    subgraph "File System"
        S[YAML Configs] --> F
        S --> M
        T[Pickle Data] --> G
        U[Directory Structures] --> V[Pattern Matching]
    end
    
    subgraph "External Integration"
        W[Kedro Pipelines] --> N
        X[Jupyter Notebooks] --> Q
        Y[HPC Environments] --> Q
    end
    
    %% Key Integration Links
    M --> E
    I --> J
    N --> Q
    W --> P
```

### 3.7.2 Technology Selection Rationale (updated)

**Scientific Python Ecosystem Focus**:
- **Mature Libraries**: Leverages established scientific computing libraries optimized for neuroscience research workflows
- **Research Community**: Aligns with standard neuroscience research tools and practices
- **Performance**: Optimized for large dataset processing common in experimental research environments
- **Extensibility**: Plugin architecture supports evolving research needs and custom data formats

**<span style="background-color: rgba(91, 57, 243, 0.2)">Version-Aware Configuration Management</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Semantic Versioning Support</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Powered by semantic_version library for robust configuration version parsing, comparison, and compatibility validation</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Automated Migration Pipelines</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Zero-breaking-change upgrades through automatic version detection and migration execution</span>  
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Research Reproducibility</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration version tracking ensures reproducible experimental setups across different library versions</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Backward Compatibility</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Comprehensive migration framework preserves existing workflows while enabling feature evolution</span>

**Minimal External Dependencies**:
- **Reliability**: Reduces external failure points for research reproducibility
- **Security**: Minimizes attack surface for sensitive research data
- **Performance**: Eliminates network latency for data processing operations
- **Compliance**: Meets institutional security requirements for research data handling

**<span style="background-color: rgba(91, 57, 243, 0.2)">Optional Kedro Integration Architecture</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Catalog-Driven Workflows</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Optional Kedro integration that enables enterprise-grade data pipeline workflows without introducing external service dependencies</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Production Pipeline Support</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">First-class FlyRigLoaderDataSet implementation providing seamless integration with Kedro's versioning, lineage tracking, and pipeline execution framework</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Lightweight Core Preservation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Available through optional installation (`pip install flyrigloader[kedro]`) maintaining minimal dependency footprint for standard research environments</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Research-to-Production Bridge</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Enables seamless transition from exploratory Jupyter notebook workflows to production data processing pipelines</span>

### 3.7.3 Architecture Integration Patterns

**Plugin-Based Extensibility**:
- **Entry Points System**: Dynamic component registration through `pyproject.toml` configuration
- **Protocol-Based Interfaces**: Type-safe plugin development with dependency injection capabilities
- **Registry Pattern**: Runtime component discovery and priority-based selection
- **Modular Architecture**: Clear separation between API, configuration, discovery, and I/O layers

**Configuration Management Strategy**:
- **Schema Validation**: Pydantic v2.6+ models providing type-safe configuration with comprehensive error reporting
- **Hierarchical Configuration**: Nested YAML structures supporting complex experimental setups
- **Environment Overrides**: Dynamic configuration adaptation based on development/production contexts
- **Version Migration Pipeline**: Automated configuration upgrades preserving experimental reproducibility

**Performance Optimization Approach**:
- **Memory Efficiency**: Streaming transformations and lazy evaluation for large dataset processing
- **Processing Isolation**: Three-stage architecture (Discovery → Loading → Transformation) enabling selective processing
- **Caching Strategies**: Intelligent caching of discovery results and transformed data
- **HPC Integration**: Optimized memory usage patterns for cluster computing environments

#### References

**Technical Specification Sections**:
- `1.1 EXECUTIVE SUMMARY` - Business context and stakeholder requirements
- `1.2 SYSTEM OVERVIEW` - Architecture and integration landscape  
- `2.4 IMPLEMENTATION CONSIDERATIONS` - Technical constraints and performance requirements
- `3.1 PROGRAMMING LANGUAGES` - Python version constraints and development standards
- `3.2 FRAMEWORKS & LIBRARIES` - Core framework selections and integration patterns
- `3.3 OPEN SOURCE DEPENDENCIES` - Comprehensive dependency ecosystem analysis

**Files Examined**:
- `pyproject.toml` - Core dependencies and project configuration
- `environment.yml` - Conda environment specification with all dependencies
- `.github/workflows/test.yml` - CI/CD tool configurations and testing matrix
- `src/flyrigloader/migration/` - Migration infrastructure implementation
- `src/flyrigloader/kedro/` - Kedro integration framework components

**Directories Analyzed**:
- `.github/workflows/` - GitHub Actions CI/CD configuration
- `src/flyrigloader/` - Main package structure and component organization
- `tests/` - Testing infrastructure and framework validation
- `blitzy/` - Documentation and development specifications
- `src/flyrigloader/migration/` - Version management and migration system implementation

# 4. PROCESS FLOWCHART

## 4.1 SYSTEM WORKFLOWS

### 4.1.1 Core Business Processes

#### 4.1.1.1 High-Level System Workflow

The FlyRigLoader system implements a decoupled 3-stage pipeline architecture optimized for neuroscience research workflows, achieving <5 seconds performance for 10,000 files discovery and <1 second per 100MB loading performance.

```mermaid
flowchart TB
    Start([User/Application Start]) --> ConfigLoad{Load Configuration}
    
    ConfigLoad -->|Success| Discovery[File Discovery Phase]
    ConfigLoad -->|Error| ConfigError[Configuration Error Handler]
    
    Discovery --> Manifest[Generate File Manifest]
    Manifest --> ValidateManifest{Validate Manifest}
    
    ValidateManifest -->|Success| LoadDecision{Selective Loading?}
    ValidateManifest -->|Error| ManifestError[Manifest Validation Error]
    
    LoadDecision -->|Yes| SelectFiles[Select Files to Load]
    LoadDecision -->|No| LoadAll[Load All Files]
    
    SelectFiles --> LoadData[Data Loading Phase]
    LoadAll --> LoadData
    
    LoadData --> TransformDecision{Transform to DataFrame?}
    
    TransformDecision -->|Yes| Transform[DataFrame Transformation]
    TransformDecision -->|No| ReturnRaw[Return Raw Data]
    
    Transform --> Validate[Validate Against Schema]
    Validate -->|Success| Success([Success: DataFrame Output])
    Validate -->|Error| TransformError[Transformation Error Handler]
    
    ReturnRaw --> Success2([Success: Raw Data Output])
    
    ConfigError --> LogError[Log Error Context]
    ManifestError --> LogError
    TransformError --> LogError
    LogError --> End([Process Terminated])
```

#### 4.1.1.2 Configuration Loading and Validation Workflow (updated)

The configuration system uses Pydantic v2 validation with YAML parsing and environment variable support, eliminating 90% of configuration-related errors through strict validation at load time.

```mermaid
flowchart LR
    subgraph "Configuration Sources"
        YAMLFile[YAML File]
        DictConfig[Dictionary Config]
        EnvVars[Environment Variables]
        BuilderPattern[Builder Pattern]
    end
    
    subgraph "Validation Layer"
        YAMLFile --> LoadYAML[Load YAML]
        LoadYAML --> SecurityCheck{Security Validation}
        
        SecurityCheck -->|Pass| VersionDetect[Version Detection & Schema Validation]
        SecurityCheck -->|Fail| SecurityError[Security Error]
        
        VersionDetect --> VersionMatch{Version Match?}
        
        VersionMatch -->|Yes| CreatePydantic[Create Pydantic Models]
        VersionMatch -->|No| MigrateConfig[Migrate Config]
        
        DictConfig --> ValidateStructure[Validate Structure]
        BuilderPattern --> CreatePydantic
        
        ValidateStructure --> CreatePydantic
        MigrateConfig --> CreatePydantic
        
        CreatePydantic --> ProjectConfig[ProjectConfig Model]
        CreatePydantic --> DatasetConfig[DatasetConfig Model]
        CreatePydantic --> ExperimentConfig[ExperimentConfig Model]
    end
    
    subgraph "Configuration Output"
        ProjectConfig --> ConfigAPI[Configuration API]
        DatasetConfig --> ConfigAPI
        ExperimentConfig --> ConfigAPI
        
        ValidateStructure --> LegacyAdapter[LegacyConfigAdapter]
        LegacyAdapter --> DeprecationWarning[Deprecation Warning]
        DeprecationWarning --> ConfigAPI
        
        EnvVars --> ConfigAPI
        
        ConfigAPI --> ResolvedConfig[Resolved Configuration]
    end
    
    SecurityError --> ErrorLog[Log Security Violation]
    ErrorLog --> Terminate([Terminate])
    
    %% Feature Flag Annotations
    BuilderPattern -.->|Feature Flag: new_config_pipeline| Note1[New Config Pipeline]
    MigrateConfig -.->|Feature Flag: new_config_pipeline| Note2[Config Migration]
```

#### 4.1.1.3 File Discovery Workflow

The Pattern-Based File Discovery System implements intelligent file location using configurable patterns, metadata extraction, and filtering capabilities, reducing manual file management overhead by 90%.

```mermaid
flowchart TD
    subgraph "Discovery Initiation"
        Start([Discovery Request]) --> GetConfig[Get Experiment Config]
        GetConfig --> ExtractPatterns[Extract Discovery Patterns]
        ExtractPatterns --> DetermineBasePath[Determine Base Path]
    end
    
    subgraph "Path Resolution"
        DetermineBasePath --> CheckSource{Path Source?}
        CheckSource -->|Explicit| UseExplicit[Use Provided Path]
        CheckSource -->|Config| UseConfig[Use Config Path]
        CheckSource -->|Environment| UseEnv[Use FLYRIGLOADER_DATA_DIR]
        
        UseExplicit --> ValidatePath
        UseConfig --> ValidatePath
        UseEnv --> ValidatePath
        
        ValidatePath{Path Exists?}
        ValidatePath -->|Yes| BuildSearchDirs
        ValidatePath -->|No| PathError[Path Not Found Error]
    end
    
    subgraph "File Discovery Engine"
        BuildSearchDirs[Build Search Directories] --> ApplyFilters[Apply Filters]
        
        ApplyFilters --> IgnorePatterns[Apply Ignore Patterns]
        ApplyFilters --> MandatoryStrings[Apply Mandatory Strings]
        ApplyFilters --> ExtensionFilter[Apply Extension Filter]
        
        IgnorePatterns --> GlobSearch[Glob/Recursive Search]
        MandatoryStrings --> GlobSearch
        ExtensionFilter --> GlobSearch
        
        GlobSearch --> MetadataExtract{Extract Metadata?}
        
        MetadataExtract -->|Yes| RegexExtract[Regex Pattern Extraction]
        MetadataExtract -->|No| BasicInfo[Basic File Info]
        
        RegexExtract --> ParseDates{Parse Dates?}
        BasicInfo --> FileStats
        
        ParseDates -->|Yes| DateParsing[Multi-Format Date Parsing]
        ParseDates -->|No| FileStats[Attach File Statistics]
        
        DateParsing --> FileStats
        FileStats --> BuildManifest[Build File Manifest]
    end
    
    subgraph "Output Generation"
        BuildManifest --> ManifestObject[FileManifest Object]
        ManifestObject --> FilterResults{Apply Filters?}
        
        FilterResults -->|Yes| ApplyManifestFilters[Apply Size/Date Filters]
        FilterResults -->|No| ReturnManifest
        
        ApplyManifestFilters --> ReturnManifest[Return Manifest]
        ReturnManifest --> End([Discovery Complete])
    end
    
    PathError --> LogPathError[Log Path Error]
    LogPathError --> TerminateDiscovery([Terminate Discovery])
```

### 4.1.2 Integration Workflows

#### 4.1.2.1 Registry-Based Data Loading Pipeline

The extensible file loading system uses a registry pattern for format-specific loaders with O(1) lookup performance, enabling support for new data formats without modifying core code.

```mermaid
flowchart LR
    subgraph "Loading Request"
        LoadRequest([Load Data File]) --> GetExtension[Extract File Extension]
        GetExtension --> RegistryLookup{Registry Lookup}
    end
    
    subgraph "Registry System"
        RegistryLookup --> LoaderRegistry[(Loader Registry)]
        
        LoaderRegistry --> CheckRegistered{Loader Registered?}
        
        CheckRegistered -->|Yes| GetLoader[Get Loader Instance]
        CheckRegistered -->|No| CheckPlugins{Check Entry Points}
        
        CheckPlugins -->|Found| RegisterPlugin[Register Plugin Loader]
        CheckPlugins -->|Not Found| NoLoader[No Loader Available]
        
        RegisterPlugin --> GetLoader
    end
    
    subgraph "Loading Process"
        GetLoader --> ValidateFile{File Exists?}
        
        ValidateFile -->|Yes| LoaderSelection{Select Strategy}
        ValidateFile -->|No| FileNotFound[File Not Found Error]
        
        LoaderSelection -->|.pkl| UncompressedLoad[Standard Pickle Load]
        LoaderSelection -->|.pklz| CompressedLoad[Compressed Pickle Load]
        LoaderSelection -->|.pkl.gz| GzipLoad[Gzip Pickle Load]
        LoaderSelection -->|Custom| CustomLoad[Custom Loader]
        
        UncompressedLoad --> ValidateData
        CompressedLoad --> ValidateData
        GzipLoad --> ValidateData
        CustomLoad --> ValidateData
        
        ValidateData{Valid Data?}
        ValidateData -->|Yes| ReturnData[Return Raw Data]
        ValidateData -->|No| LoadError[Load Error]
    end
    
    NoLoader --> UnsupportedFormat[Unsupported Format Error]
    FileNotFound --> LogLoadError
    LoadError --> LogLoadError
    UnsupportedFormat --> LogLoadError[Log Load Error]
    
    LogLoadError --> TerminateLoad([Terminate Load])
    ReturnData --> Success([Load Success])
```

#### 4.1.2.2 DataFrame Transformation Pipeline

The DataFrame Transformation Engine converts raw experimental data to standardized pandas DataFrames with column validation, reducing data preparation time by 80% through automated transformation.

```mermaid
flowchart TD
    subgraph "Transformation Request"
        Start([Transform to DataFrame]) --> GetSchema{Get Column Schema}
        
        GetSchema -->|Provided| UseProvided[Use Provided Schema]
        GetSchema -->|Default| LoadDefault[Load column_config.yaml]
        
        UseProvided --> ValidateSchema
        LoadDefault --> ValidateSchema[Validate Schema]
    end
    
    subgraph "Schema Processing"
        ValidateSchema --> CreateTransformer[Create DataFrame Transformer]
        CreateTransformer --> ValidateInput{Validate Input Data}
        
        ValidateInput -->|Valid| CheckTimeColumn
        ValidateInput -->|Invalid| InputError[Invalid Input Error]
        
        CheckTimeColumn{Has Time Column?}
        CheckTimeColumn -->|Yes| ProcessColumns
        CheckTimeColumn -->|No| TimeError[Missing Time Column Error]
    end
    
    subgraph "Column Processing Loop"
        ProcessColumns[Process Each Column] --> GetHandler{Get Column Handler}
        
        GetHandler --> HandlerType{Handler Type?}
        
        HandlerType -->|Standard| StandardProcess[Standard Processing]
        HandlerType -->|Signal Display| SignalHandler[Signal Display Handler]
        HandlerType -->|2D Extract| Extract2D[Extract First Column]
        HandlerType -->|Dimension| DimNormalize[Normalize Dimensions]
        
        StandardProcess --> ValidateColumn
        SignalHandler --> ValidateColumn
        Extract2D --> ValidateColumn
        DimNormalize --> ValidateColumn
        
        ValidateColumn{Column Valid?}
        ValidateColumn -->|Yes| NextColumn{More Columns?}
        ValidateColumn -->|No| ColumnError[Column Validation Error]
        
        NextColumn -->|Yes| ProcessColumns
        NextColumn -->|No| BuildDataFrame
    end
    
    subgraph "DataFrame Construction"
        BuildDataFrame[Build DataFrame] --> AddMetadata{Add Metadata?}
        
        AddMetadata -->|Yes| AttachMetadata[Attach Metadata Columns]
        AddMetadata -->|No| StrictMode
        
        AttachMetadata --> StrictMode{Strict Schema?}
        
        StrictMode -->|Yes| DropExtraColumns[Drop Extra Columns]
        StrictMode -->|No| FinalValidation
        
        DropExtraColumns --> FinalValidation[Final Validation]
        
        FinalValidation --> ReturnDF[Return DataFrame]
    end
    
    InputError --> LogTransformError
    TimeError --> LogTransformError
    ColumnError --> LogTransformError[Log Transform Error]
    
    LogTransformError --> TerminateTransform([Terminate Transform])
    ReturnDF --> Success([Transform Success])
```

## 4.2 ERROR HANDLING FLOWS

### 4.2.1 Comprehensive Error Management

#### 4.2.1.1 Domain-Specific Error Handling Flow (updated)

The comprehensive error handling system implements domain-specific exception hierarchy with context preservation, reducing debugging time by 60% through detailed error context and programmatic error handling. <span style="background-color: rgba(91, 57, 243, 0.2)">The system now includes specialized error handling for registry operations, version management, and Kedro integration workflows</span>.

```mermaid
flowchart TB
    subgraph "Error Sources"
        ConfigOp[Configuration Operation] --> ConfigErr{Config Error?}
        DiscoveryOp[Discovery Operation] --> DiscoveryErr{Discovery Error?}
        LoadOp[Load Operation] --> LoadErr{Load Error?}
        TransformOp[Transform Operation] --> TransformErr{Transform Error?}
        RegistryOp[Registry Operation] --> RegistryErr{Registry Error?}
        VersionOp[Version Operation] --> VersionErr{Version Error?}
        KedroOp[Kedro Operation] --> KedroErr{Kedro Error?}
    end
    
    subgraph "Error Classification"
        ConfigErr -->|Yes| ConfigError[ConfigError Exception]
        DiscoveryErr -->|Yes| DiscoveryError[DiscoveryError Exception]
        LoadErr -->|Yes| LoadError[LoadError Exception]
        TransformErr -->|Yes| TransformError[TransformError Exception]
        RegistryErr -->|Yes| RegistryError[RegistryError Exception]
        VersionErr -->|Yes| VersionError[VersionError Exception]
        KedroErr -->|Yes| KedroIntegrationError[KedroIntegrationError Exception]
        
        ConfigError --> ErrorContext[Capture Error Context]
        DiscoveryError --> ErrorContext
        LoadError --> ErrorContext
        TransformError --> ErrorContext
        RegistryError --> ErrorContext
        VersionError --> ErrorContext
        KedroIntegrationError --> ErrorContext
    end
    
    subgraph "Context Enrichment"
        ErrorContext --> AddContext[Add Context Information]
        
        AddContext --> FilePath[File Path]
        AddContext --> Operation[Operation Type]
        AddContext --> ErrorCode[Error Code]
        AddContext --> OriginalError[Original Exception]
        AddContext --> Timestamp[Timestamp]
        
        FilePath --> BuildException
        Operation --> BuildException
        ErrorCode --> BuildException
        OriginalError --> BuildException
        Timestamp --> BuildException[Build Rich Exception]
    end
    
    subgraph "Error Handling"
        BuildException --> LogError[Log with Context]
        
        LogError --> CheckRecovery{Recoverable?}
        
        CheckRecovery -->|Yes| AttemptRecovery[Attempt Recovery]
        CheckRecovery -->|No| PropagateError[Propagate Exception]
        
        AttemptRecovery --> RecoverySuccess{Success?}
        
        RecoverySuccess -->|Yes| LogRecovery[Log Recovery]
        RecoverySuccess -->|No| PropagateError
        
        LogRecovery --> Continue([Continue Execution])
        PropagateError --> RaiseException[Raise to Caller]
        
        RaiseException --> HandleUpstream{Upstream Handler?}
        
        HandleUpstream -->|Yes| UpstreamHandle[Handle at Higher Level]
        HandleUpstream -->|No| UserError[Display User Error]
        
        UserError --> Terminate([Terminate Process])
    end
```

#### 4.2.1.2 Validation Error Processing (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The validation system now includes comprehensive version validation to ensure schema compatibility and migration path verification</span>.

```mermaid
flowchart LR
    subgraph "Validation Types"
        SchemaVal[Schema Validation] --> PydanticCheck{Pydantic Validation}
        VersionVal[Version Validation] --> VersionCheck{Version Check}
        PathVal[Path Validation] --> SecurityCheck{Security Check}
        PatternVal[Pattern Validation] --> RegexCheck{Regex Compile}
        DataVal[Data Validation] --> TypeCheck{Type Check}
    end
    
    subgraph "Pydantic Validation"
        PydanticCheck -->|Fail| CollectErrors[Collect All Errors]
        CollectErrors --> FormatErrors[Format Error Details]
        FormatErrors --> PydanticError[ValidationError]
    end
    
    subgraph "Version Validation"
        VersionCheck -->|Fail| VersionMismatch[Version Mismatch]
        VersionMismatch --> CheckMigration[Check Migration Path]
        CheckMigration --> VersionError[VersionError]
    end
    
    subgraph "Security Validation"
        SecurityCheck -->|Fail| SecurityViolation[Security Violation]
        SecurityViolation --> LogSecurity[Log Security Event]
        LogSecurity --> SecurityError[Security Exception]
    end
    
    subgraph "Pattern Validation"
        RegexCheck -->|Fail| PatternInvalid[Invalid Pattern]
        PatternInvalid --> SuggestFix[Suggest Fix]
        SuggestFix --> PatternError[Pattern Exception]
    end
    
    subgraph "Type Validation"
        TypeCheck -->|Fail| TypeMismatch[Type Mismatch]
        TypeMismatch --> ShowExpected[Show Expected Type]
        ShowExpected --> TypeError[Type Exception]
    end
    
    PydanticError --> HandleValidation
    VersionError --> HandleValidation
    SecurityError --> HandleValidation
    PatternError --> HandleValidation
    TypeError --> HandleValidation[Handle Validation Error]
    
    HandleValidation --> UserMessage[Format User Message]
    UserMessage --> Return([Return Error])
```

#### 4.2.1.3 Registry and Integration Error Flows (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The system provides specialized error handling for registry operations and Kedro integration scenarios, ensuring robust plugin management and pipeline compatibility</span>.

```mermaid
flowchart TD
    subgraph "Registry Error Handling"
        RegistryOp[Registry Operation] --> PluginConflict{Plugin Conflict?}
        PluginConflict -->|Yes| ConflictError[Plugin Registration Conflict]
        PluginConflict -->|No| LoaderMissing{Loader Missing?}
        
        LoaderMissing -->|Yes| MissingError[Loader Not Found]
        LoaderMissing -->|No| EntryPointError{Entry Point Error?}
        
        EntryPointError -->|Yes| DiscoveryError[Plugin Discovery Failed]
        
        ConflictError --> PriorityResolution[Apply Priority Resolution]
        MissingError --> SuggestPlugin[Suggest Available Plugins]
        DiscoveryError --> CheckInstallation[Check Plugin Installation]
        
        PriorityResolution --> LogConflict[Log Conflict Resolution]
        SuggestPlugin --> RegistryError
        CheckInstallation --> RegistryError[RegistryError Exception]
        
        LogConflict --> AttemptContinue{Can Continue?}
        AttemptContinue -->|Yes| RecoveredRegistry[Registry Recovered]
        AttemptContinue -->|No| RegistryError
    end
    
    subgraph "Kedro Integration Error Handling"
        KedroOp[Kedro Operation] --> CatalogError{Catalog Error?}
        CatalogError -->|Yes| InvalidCatalog[Invalid Catalog Configuration]
        CatalogError -->|No| DatasetError{Dataset Error?}
        
        DatasetError -->|Yes| DatasetInit[Dataset Initialization Failed]
        DatasetError -->|No| PipelineError{Pipeline Error?}
        
        PipelineError -->|Yes| ExecutionFailed[Pipeline Execution Failed]
        
        InvalidCatalog --> ValidateSchema[Validate Catalog Schema]
        DatasetInit --> CheckDependencies[Check Kedro Dependencies]
        ExecutionFailed --> InspectLogs[Inspect Pipeline Logs]
        
        ValidateSchema --> KedroIntegrationError
        CheckDependencies --> KedroIntegrationError
        InspectLogs --> KedroIntegrationError[KedroIntegrationError Exception]
    end
    
    subgraph "Version Management Error Handling"
        VersionOp[Version Operation] --> SchemaVersion{Schema Version Check?}
        SchemaVersion -->|Fail| VersionIncompatible[Version Incompatible]
        SchemaVersion -->|Pass| MigrationNeeded{Migration Needed?}
        
        MigrationNeeded -->|Yes| AttemptMigration[Attempt Migration]
        MigrationNeeded -->|No| VersionSuccess[Version Validated]
        
        AttemptMigration --> MigrationSuccess{Migration Success?}
        MigrationSuccess -->|Yes| BackupOriginal[Backup Original Config]
        MigrationSuccess -->|No| MigrationFailed[Migration Failed]
        
        VersionIncompatible --> CheckSupported[Check Supported Versions]
        MigrationFailed --> DiagnoseMigration[Diagnose Migration Issues]
        
        CheckSupported --> VersionError
        DiagnoseMigration --> VersionError[VersionError Exception]
        
        BackupOriginal --> VersionSuccess
    end
    
    RegistryError --> LogRegistryError[Log Registry Context]
    KedroIntegrationError --> LogKedroError[Log Kedro Context]
    VersionError --> LogVersionError[Log Version Context]
    
    LogRegistryError --> PropagateUp[Propagate to Error Handler]
    LogKedroError --> PropagateUp
    LogVersionError --> PropagateUp
    
    RecoveredRegistry --> Continue([Continue Operation])
    VersionSuccess --> Continue
```

### 4.2.2 Recovery Mechanisms

#### 4.2.2.1 Retry and Fallback Strategies

```mermaid
flowchart TD
    subgraph "Retry Logic"
        Operation[Operation Attempt] --> CheckRetryable{Retryable Error?}
        
        CheckRetryable -->|Yes| RetryCount{Retry Count < Max?}
        CheckRetryable -->|No| FallbackStrategy
        
        RetryCount -->|Yes| BackoffDelay[Exponential Backoff]
        RetryCount -->|No| FallbackStrategy
        
        BackoffDelay --> RetryOperation[Retry Operation]
        RetryOperation --> Operation
    end
    
    subgraph "Fallback Processing"
        FallbackStrategy[Fallback Strategy] --> FallbackType{Fallback Type?}
        
        FallbackType -->|Default Value| UseDefault[Use Default Value]
        FallbackType -->|Cache| UseCache[Use Cached Result]
        FallbackType -->|Alternative| UseAlternative[Use Alternative Method]
        FallbackType -->|Graceful Degradation| GracefulDegradation[Partial Functionality]
        
        UseDefault --> LogFallback
        UseCache --> LogFallback
        UseAlternative --> LogFallback
        GracefulDegradation --> LogFallback[Log Fallback Used]
    end
    
    LogFallback --> ContinueOperation[Continue Operation]
    ContinueOperation --> Success([Fallback Success])
```

#### 4.2.2.2 Context-Aware Recovery Strategies (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Recovery mechanisms now include specialized strategies for registry conflicts, version migration failures, and Kedro integration issues</span>.

```mermaid
flowchart LR
    subgraph "Recovery Decision Matrix"
        ErrorType{Error Type?} --> ConfigRecovery[Configuration Recovery]
        ErrorType --> RegistryRecovery[Registry Recovery]
        ErrorType --> VersionRecovery[Version Recovery]
        ErrorType --> KedroRecovery[Kedro Recovery]
        ErrorType --> LoadRecovery[Load Recovery]
    end
    
    subgraph "Registry Recovery Strategies"
        RegistryRecovery --> PluginConflictResolution{Plugin Conflict?}
        PluginConflictResolution -->|Yes| ApplyPriority[Apply Priority Rules]
        PluginConflictResolution -->|No| EntryPointReload[Reload Entry Points]
        
        ApplyPriority --> LogResolution[Log Resolution]
        EntryPointReload --> ValidatePlugins[Validate Plugin State]
        
        LogResolution --> RecoverySuccess
        ValidatePlugins --> RecoverySuccess[Registry Recovery Success]
    end
    
    subgraph "Version Recovery Strategies"
        VersionRecovery --> MigrationFailure{Migration Failed?}
        MigrationFailure -->|Yes| RollbackConfig[Rollback to Previous Version]
        MigrationFailure -->|No| ValidateVersion[Validate Schema Version]
        
        RollbackConfig --> NotifyRollback[Notify User of Rollback]
        ValidateVersion --> UpdateCompatibility[Update Compatibility Matrix]
        
        NotifyRollback --> RecoverySuccess
        UpdateCompatibility --> RecoverySuccess
    end
    
    subgraph "Kedro Recovery Strategies"
        KedroRecovery --> CatalogIssue{Catalog Issue?}
        CatalogIssue -->|Yes| RepairCatalog[Repair Catalog Configuration]
        CatalogIssue -->|No| DatasetIssue{Dataset Issue?}
        
        DatasetIssue -->|Yes| RecreateDataset[Recreate Dataset Instance]
        DatasetIssue -->|No| PipelineIssue[Handle Pipeline Issue]
        
        RepairCatalog --> ValidateCatalog[Validate Catalog Schema]
        RecreateDataset --> TestDataset[Test Dataset Operations]
        PipelineIssue --> RestartPipeline[Restart Pipeline Component]
        
        ValidateCatalog --> RecoverySuccess
        TestDataset --> RecoverySuccess
        RestartPipeline --> RecoverySuccess
    end
    
    ConfigRecovery --> UseDefaults[Use Default Configuration]
    LoadRecovery --> AlternativeLoader[Try Alternative Loader]
    
    UseDefaults --> RecoverySuccess
    AlternativeLoader --> RecoverySuccess
    
    RecoverySuccess --> LogRecoveryDetails[Log Recovery Details]
    LogRecoveryDetails --> ContinueProcessing([Continue Processing])
```

### 4.2.3 Error Context and Logging

#### 4.2.3.1 Structured Error Context Collection (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Error context collection now captures registry state, version information, and Kedro pipeline context for comprehensive debugging support</span>.

```mermaid
flowchart TB
    subgraph "Context Collection Sources"
        SystemState[System State] --> CollectRegistry[Collect Registry State]
        SystemState --> CollectVersion[Collect Version Information]
        SystemState --> CollectKedro[Collect Kedro Context]
        SystemState --> CollectConfig[Collect Configuration State]
        SystemState --> CollectRuntime[Collect Runtime Environment]
    end
    
    subgraph "Registry Context"
        CollectRegistry --> RegisteredLoaders[Registered Loaders]
        CollectRegistry --> PluginStatus[Plugin Status]
        CollectRegistry --> ConflictHistory[Conflict History]
        
        RegisteredLoaders --> RegistryContext
        PluginStatus --> RegistryContext
        ConflictHistory --> RegistryContext[Registry Context Block]
    end
    
    subgraph "Version Context"
        CollectVersion --> SchemaVersions[Schema Versions]
        CollectVersion --> MigrationHistory[Migration History]
        CollectVersion --> CompatibilityMatrix[Compatibility Matrix]
        
        SchemaVersions --> VersionContext
        MigrationHistory --> VersionContext
        CompatibilityMatrix --> VersionContext[Version Context Block]
    end
    
    subgraph "Kedro Context"
        CollectKedro --> CatalogState[Catalog State]
        CollectKedro --> PipelineInfo[Pipeline Information]
        CollectKedro --> DatasetStatus[Dataset Status]
        
        CatalogState --> KedroContext
        PipelineInfo --> KedroContext
        DatasetStatus --> KedroContext[Kedro Context Block]
    end
    
    subgraph "Unified Context"
        RegistryContext --> BuildContext[Build Unified Context]
        VersionContext --> BuildContext
        KedroContext --> BuildContext
        CollectConfig --> BuildContext
        CollectRuntime --> BuildContext
        
        BuildContext --> StructuredLog[Structured Log Entry]
        StructuredLog --> PersistContext[Persist Context]
        
        PersistContext --> DebugInfo[Debug Information Available]
    end
```

#### 4.2.3.2 Error Notification and Reporting Flow

```mermaid
flowchart LR
    subgraph "Error Detection"
        ErrorOccurs[Error Occurs] --> ClassifyError[Classify Error Severity]
        ClassifyError --> CheckNotification{Notification Required?}
    end
    
    subgraph "Notification Routing"
        CheckNotification -->|Yes| NotificationLevel{Notification Level?}
        
        NotificationLevel -->|Critical| ImmediateAlert[Immediate Alert]
        NotificationLevel -->|Warning| LogWarning[Log Warning]
        NotificationLevel -->|Info| LogInfo[Log Information]
        
        ImmediateAlert --> NotifyUser[Notify User]
        LogWarning --> CheckThreshold{Threshold Exceeded?}
        LogInfo --> AggregateStats[Aggregate Statistics]
        
        CheckThreshold -->|Yes| NotifyUser
        CheckThreshold -->|No| ContinueLogging
        
        AggregateStats --> ContinueLogging[Continue Logging]
    end
    
    subgraph "User Notification"
        NotifyUser --> FormatMessage[Format User Message]
        FormatMessage --> DisplayError[Display Error Message]
        DisplayError --> ProvideGuidance[Provide Recovery Guidance]
        
        ProvideGuidance --> UserAction{User Action Required?}
        UserAction -->|Yes| WaitUserResponse[Wait for User Response]
        UserAction -->|No| AutoRecover[Attempt Auto Recovery]
        
        WaitUserResponse --> ProcessResponse[Process User Response]
        AutoRecover --> ContinueLogging
        ProcessResponse --> ContinueLogging
    end
    
    ContinueLogging --> Complete([Notification Complete])
```

## 4.3 PLUGIN SYSTEM WORKFLOWS

### 4.3.1 Plugin Architecture and Registration

#### 4.3.1.1 Plugin Registration and Discovery Flow

The Plugin Architecture System supports dynamic loader and schema registration via entry points or decorators, enabling zero-code extensions and reducing integration time for new formats from days to hours.

```mermaid
flowchart TD
    subgraph "Plugin Sources"
        EntryPoints[Entry Points] --> ScanEntryPoints[Scan Entry Points]
        RuntimeReg[Runtime Registration] --> DirectRegister[Direct Register Call]
        Decorator["@loader_for Decorator"] --> DecoratorRegister[Decorator Registration]
    end
    
    subgraph "Registry Management"
        ScanEntryPoints --> LoadPlugin{Load Plugin Module}
        
        LoadPlugin -->|Success| ValidatePlugin[Validate Plugin Interface]
        LoadPlugin -->|Fail| LogPluginError[Log Plugin Error]
        
        ValidatePlugin --> CheckProtocol{"Implements Protocol?"}
        
        CheckProtocol -->|Yes| RegisterInRegistry
        CheckProtocol -->|No| RejectPlugin[Reject Plugin]
        
        DirectRegister --> ValidatePlugin
        DecoratorRegister --> ValidatePlugin
        
        RegisterInRegistry[Register in Registry] --> CheckPriority{Check Priority}
        
        CheckPriority --> ExistingEntry{"Existing Entry?"}
        
        ExistingEntry -->|Yes| ComparePriority{"Higher Priority?"}
        ExistingEntry -->|No| AddEntry[Add to Registry]
        
        ComparePriority -->|Yes| ReplaceEntry[Replace Entry]
        ComparePriority -->|No| KeepExisting[Keep Existing]
        
        ReplaceEntry --> UpdateRegistry
        AddEntry --> UpdateRegistry
        KeepExisting --> LogSkip[Log Skip]
        
        UpdateRegistry[Update Registry] --> NotifySuccess[Notify Success]
    end
    
    subgraph "Plugin Usage"
        LookupRequest[Lookup Request] --> GetFromRegistry[Get from Registry]
        
        GetFromRegistry --> Found{"Plugin Found?"}
        
        Found -->|Yes| InstantiatePlugin[Instantiate Plugin]
        Found -->|No| DefaultHandler[Use Default Handler]
        
        InstantiatePlugin --> ExecutePlugin[Execute Plugin]
        DefaultHandler --> ExecuteDefault[Execute Default]
        
        ExecutePlugin --> PluginResult[Plugin Result]
        ExecuteDefault --> DefaultResult[Default Result]
    end
    
    LogPluginError --> ContinueDiscovery[Continue Discovery]
    RejectPlugin --> LogRejection[Log Rejection]
    LogSkip --> End([Registry Updated])
    NotifySuccess --> End
    
    PluginResult --> Success([Success])
    DefaultResult --> Success
```

### 4.3.2 Extension Point Management

#### 4.3.2.1 Dynamic Extension Loading

```mermaid
flowchart LR
    subgraph "Extension Discovery"
        AppStart[Application Start] --> ScanExtensions[Scan Extension Points]
        
        ScanExtensions --> EntryPointLoad[Load Entry Points]
        ScanExtensions --> DirectoryLoad[Load Directory Plugins]
        ScanExtensions --> DecoratorLoad[Load Decorator Plugins]
        
        EntryPointLoad --> ValidateExtension
        DirectoryLoad --> ValidateExtension
        DecoratorLoad --> ValidateExtension[Validate Extension]
    end
    
    subgraph "Extension Validation"
        ValidateExtension --> CheckInterface{Valid Interface?}
        
        CheckInterface -->|Yes| LoadExtension[Load Extension]
        CheckInterface -->|No| RejectExtension[Reject Extension]
        
        LoadExtension --> RegisterExtension[Register Extension]
        RejectExtension --> LogRejection[Log Rejection]
    end
    
    subgraph "Extension Usage"
        RequestExtension[Request Extension] --> LookupExtension[Lookup Extension]
        
        LookupExtension --> ExtensionFound{Extension Found?}
        
        ExtensionFound -->|Yes| UseExtension[Use Extension]
        ExtensionFound -->|No| UseDefault[Use Default Implementation]
        
        UseExtension --> ExtensionResult[Extension Result]
        UseDefault --> DefaultResult[Default Result]
    end
    
    LogRejection --> ContinueDiscovery[Continue Discovery]
    RegisterExtension --> Ready([Extension Ready])
    
    ExtensionResult --> Success([Success])
    DefaultResult --> Success
```

## 4.4 END-TO-END WORKFLOW EXAMPLES

### 4.4.1 Complete Research Workflow

#### 4.4.1.1 Neuroscience Experiment Processing Flow

This end-to-end workflow demonstrates how FlyRigLoader processes experimental data from opto rigs used in neuroscience research, supporting the complete research pipeline from discovery to analysis.

```mermaid
flowchart TB
    subgraph "Initialization"
        Start([Researcher: Process Experiment]) --> LoadConfig[Load Configuration]
        LoadConfig --> CreateAPI[Initialize API]
        CreateAPI --> ResolveBase[Resolve Base Directory]
    end
    
    subgraph "Discovery Phase"
        ResolveBase --> DiscoverManifest[Discover Experiment Manifest]
        
        DiscoverManifest --> FilterFiles{Filter Files}
        FilterFiles --> IgnoreTemp[Ignore Temporary Files]
        FilterFiles --> RequireExperiment[Require 'experiment' String]
        FilterFiles --> MatchExtensions[Match .pkl/.pklz Extensions]
        
        IgnoreTemp --> BuildManifest
        RequireExperiment --> BuildManifest
        MatchExtensions --> BuildManifest[Build File Manifest]
        
        BuildManifest --> ManifestReady{Manifest Ready}
    end
    
    subgraph "Selective Loading"
        ManifestReady --> SelectFiles[Select Files to Process]
        
        SelectFiles --> LoadLoop[For Each Selected File]
        
        LoadLoop --> CheckFormat{Check File Format}
        
        CheckFormat -->|.pkl| PickleLoader[Use Pickle Loader]
        CheckFormat -->|.pklz| CompressedLoader[Use Compressed Loader]
        CheckFormat -->|.pkl.gz| GzipLoader[Use Gzip Loader]
        
        PickleLoader --> LoadFile[Load File Data]
        CompressedLoader --> LoadFile
        GzipLoader --> LoadFile
        
        LoadFile --> ValidateLoad{Valid Data?}
        
        ValidateLoad -->|Yes| StoreRaw[Store Raw Data]
        ValidateLoad -->|No| SkipFile[Skip File]
        
        StoreRaw --> NextFile{More Files?}
        SkipFile --> NextFile
        
        NextFile -->|Yes| LoadLoop
        NextFile -->|No| TransformPhase
    end
    
    subgraph "Transformation Phase"
        TransformPhase[Transform Phase] --> LoadSchema[Load Column Schema]
        
        LoadSchema --> TransformLoop[For Each Raw Data]
        
        TransformLoop --> ApplySchema[Apply Column Schema]
        ApplySchema --> HandleSpecial[Handle Special Columns]
        HandleSpecial --> ValidateTransform{Valid DataFrame?}
        
        ValidateTransform -->|Yes| AddMetadata[Add File Metadata]
        ValidateTransform -->|No| SkipTransform[Skip Transformation]
        
        AddMetadata --> CollectDF[Collect DataFrame]
        SkipTransform --> NextData
        
        CollectDF --> NextData{More Data?}
        
        NextData -->|Yes| TransformLoop
        NextData -->|No| CombineResults
    end
    
    subgraph "Final Processing"
        CombineResults[Combine DataFrames] --> FinalValidation{Validate Combined Data}
        
        FinalValidation -->|Valid| Success[Return Combined DataFrame]
        FinalValidation -->|Invalid| PartialSuccess[Return Partial Results]
        
        Success --> LogSuccess[Log Success Metrics]
        PartialSuccess --> LogWarnings[Log Warnings]
        
        LogSuccess --> End([Processing Complete])
        LogWarnings --> End
    end
```

### 4.4.2 Integration Workflow Examples

#### 4.4.2.1 Kedro Pipeline Integration (updated)

```mermaid
flowchart LR
    subgraph "Kedro Pipeline"
        K1[Pipeline Start] --> K2[Load Parameters]
        K2 --> K3[Initialize Context]
        K3 --> K4["Instantiate FlyRigLoaderDataSet"]
        
        K4 --> K5[Process Data]
        K5 --> K6[Store Results]
        K6 --> K7[Pipeline Complete]
    end
    
    subgraph "FlyRigLoader Processing"
        F1[Receive Kedro Request] --> F2[Load Config from Context]
        F2 --> F3[Discover Files]
        F3 --> F4[Load and Transform]
        F4 --> F5[Return DataFrame]
    end
    
    K4 -->|"Dataset.load()"| F1
    F5 --> K5
    
    K4 -.->|"uses create_kedro_dataset() factory"| Note1[Factory Pattern]
```

This workflow demonstrates the seamless integration between Kedro data pipelines and FlyRigLoader through the custom AbstractDataset implementation. The FlyRigLoaderDataSet provides a standardized interface that adheres to Kedro's data catalog conventions while maintaining full access to FlyRigLoader's experimental data processing capabilities.

**Key Integration Points:**
- **Dataset Factory**: The `create_kedro_dataset()` factory function instantiates the FlyRigLoaderDataSet with pipeline-specific configuration
- **Catalog Configuration**: Standard catalog.yml entries define the dataset parameters and connection details
- **Metadata Preservation**: All DataFrame outputs include Kedro-compatible metadata columns for lineage tracking
- **Pipeline Compatibility**: Full support for Kedro's versioning, caching, and parallel execution features

#### 4.4.2.2 Jupyter Notebook Integration

```mermaid
flowchart TD
    subgraph "Interactive Research"
        J1[Import flyrigloader] --> J2[Configure Experiment]
        J2 --> J3[Discover Files]
        J3 --> J4[Preview Manifest]
        J4 --> J5[Load Selected Data]
        J5 --> J6[Transform to DataFrame]
        J6 --> J7[Analyze and Visualize]
    end
    
    subgraph "Iterative Development"
        J7 --> J8{Satisfied with Results?}
        J8 -->|No| J9[Adjust Parameters]
        J8 -->|Yes| J10[Save Results]
        J9 --> J2
        J10 --> J11[Export Analysis]
    end
```

This interactive workflow showcases FlyRigLoader's designed-for-research capabilities in Jupyter environments, enabling rapid prototyping and exploratory data analysis of neuroscience experiments.

### 4.4.3 Production Data Pipeline Examples

#### 4.4.3.1 Automated Research Data Processing

```mermaid
flowchart TB
    subgraph "Scheduled Processing"
        Trigger[Scheduled Trigger] --> ScanDirectories[Scan Data Directories]
        ScanDirectories --> NewFiles{New Files Detected?}
        
        NewFiles -->|Yes| ValidateFiles[Validate File Integrity]
        NewFiles -->|No| WaitNextTrigger[Wait for Next Trigger]
        
        ValidateFiles --> ProcessBatch[Process File Batch]
        ProcessBatch --> TransformData[Transform to Standard Format]
        TransformData --> QualityCheck[Quality Assurance Check]
        
        QualityCheck -->|Pass| StoreResults[Store Processed Results]
        QualityCheck -->|Fail| QuarantineFiles[Quarantine Failed Files]
        
        StoreResults --> UpdateMetadata[Update Processing Metadata]
        QuarantineFiles --> LogFailures[Log Processing Failures]
        
        UpdateMetadata --> NotifySuccess[Notify Success]
        LogFailures --> NotifyFailures[Notify Failures]
        
        NotifySuccess --> WaitNextTrigger
        NotifyFailures --> WaitNextTrigger
    end
```

#### 4.4.3.2 Multi-Experiment Comparison Workflow

```mermaid
flowchart LR
    subgraph "Experiment Collection"
        Start[Research Query] --> IdentifyExperiments[Identify Experiments]
        IdentifyExperiments --> LoadExperiment1[Load Experiment A]
        IdentifyExperiments --> LoadExperiment2[Load Experiment B]
        IdentifyExperiments --> LoadExperimentN[Load Experiment N...]
    end
    
    subgraph "Standardization"
        LoadExperiment1 --> NormalizeA[Normalize Schema A]
        LoadExperiment2 --> NormalizeB[Normalize Schema B]
        LoadExperimentN --> NormalizeN[Normalize Schema N]
        
        NormalizeA --> AlignTime[Align Time Bases]
        NormalizeB --> AlignTime
        NormalizeN --> AlignTime
    end
    
    subgraph "Analysis"
        AlignTime --> CombineDatasets[Combine Datasets]
        CombineDatasets --> StatisticalAnalysis[Statistical Analysis]
        StatisticalAnalysis --> GenerateReport[Generate Comparison Report]
        GenerateReport --> ExportResults[Export Results]
    end
```

### 4.4.4 Error Recovery and Resilience Workflows

#### 4.4.4.1 Fault-Tolerant Processing Pipeline

```mermaid
flowchart TD
    subgraph "Resilient Processing"
        Start[Start Processing] --> ProcessFile[Process File]
        
        ProcessFile --> CheckError{Error Occurred?}
        
        CheckError -->|No| Success[Success]
        CheckError -->|Yes| ClassifyError{Error Type?}
        
        ClassifyError -->|Transient| RetryLogic[Apply Retry Logic]
        ClassifyError -->|Permanent| SkipFile[Skip File]
        ClassifyError -->|Configuration| FixConfig[Attempt Config Fix]
        
        RetryLogic --> RetryCount{Retry Count < Max?}
        RetryCount -->|Yes| BackoffDelay[Exponential Backoff]
        RetryCount -->|No| SkipFile
        
        BackoffDelay --> ProcessFile
        
        FixConfig --> ConfigFixed{Config Fixed?}
        ConfigFixed -->|Yes| ProcessFile
        ConfigFixed -->|No| SkipFile
        
        SkipFile --> LogError[Log Error Details]
        LogError --> NextFile[Continue to Next File]
        
        Success --> NextFile
        NextFile --> MoreFiles{More Files?}
        
        MoreFiles -->|Yes| ProcessFile
        MoreFiles -->|No| ProcessingComplete[Processing Complete]
    end
```

#### 4.4.4.2 Data Integrity Validation Workflow

```mermaid
flowchart LR
    subgraph "Validation Pipeline"
        LoadData[Load Data] --> CheckFormat{Valid Format?}
        
        CheckFormat -->|Yes| CheckSchema{Schema Valid?}
        CheckFormat -->|No| FormatError[Format Error]
        
        CheckSchema -->|Yes| ValidateContent{Content Valid?}
        CheckSchema -->|No| SchemaError[Schema Error]
        
        ValidateContent -->|Yes| CheckCompleteness{Data Complete?}
        ValidateContent -->|No| ContentError[Content Error]
        
        CheckCompleteness -->|Yes| DataValid[Data Valid]
        CheckCompleteness -->|No| CompletenessError[Completeness Error]
    end
    
    subgraph "Error Handling"
        FormatError --> AttemptRepair[Attempt Auto-Repair]
        SchemaError --> AttemptMigration[Attempt Schema Migration]
        ContentError --> AttemptCleaning[Attempt Data Cleaning]
        CompletenessError --> PartialProcessing[Allow Partial Processing]
        
        AttemptRepair --> RepairSuccess{Repair Success?}
        AttemptMigration --> MigrationSuccess{Migration Success?}
        AttemptCleaning --> CleaningSuccess{Cleaning Success?}
        
        RepairSuccess -->|Yes| CheckSchema
        RepairSuccess -->|No| LogFailure[Log Validation Failure]
        
        MigrationSuccess -->|Yes| ValidateContent
        MigrationSuccess -->|No| LogFailure
        
        CleaningSuccess -->|Yes| CheckCompleteness
        CleaningSuccess -->|No| LogFailure
        
        PartialProcessing --> DataPartiallyValid[Data Partially Valid]
        LogFailure --> DataInvalid[Data Invalid]
    end
    
    DataValid --> ProcessingSuccess[Processing Success]
    DataPartiallyValid --> ProcessingPartial[Processing Partial]
    DataInvalid --> ProcessingFailed[Processing Failed]
```

### 4.4.5 Performance Optimization Workflows

#### 4.4.5.1 Memory-Efficient Large Dataset Processing

```mermaid
flowchart TB
    subgraph "Memory Management"
        StartProcessing[Start Large Dataset Processing] --> EstimateMemory[Estimate Memory Requirements]
        
        EstimateMemory --> MemoryCheck{Memory Available?}
        
        MemoryCheck -->|Sufficient| DirectLoad[Load All Files]
        MemoryCheck -->|Insufficient| ChunkStrategy[Apply Chunking Strategy]
        
        ChunkStrategy --> CreateChunks[Create File Chunks]
        CreateChunks --> ProcessChunk[Process Chunk]
        
        ProcessChunk --> IntermediateStore[Store Intermediate Results]
        IntermediateStore --> NextChunk{More Chunks?}
        
        NextChunk -->|Yes| ClearMemory[Clear Memory]
        NextChunk -->|No| CombineResults[Combine All Results]
        
        ClearMemory --> ProcessChunk
        
        DirectLoad --> CombineResults
        CombineResults --> FinalValidation[Final Validation]
        
        FinalValidation --> OptimizedOutput[Memory-Optimized Output]
    end
```

#### 4.4.5.2 Parallel Processing Workflow

```mermaid
flowchart LR
    subgraph "Parallel Execution"
        TaskQueue[Task Queue] --> DistributeTasks[Distribute Tasks]
        
        DistributeTasks --> Worker1[Worker Process 1]
        DistributeTasks --> Worker2[Worker Process 2]
        DistributeTasks --> WorkerN[Worker Process N]
        
        Worker1 --> ProcessFiles1[Process Files 1-100]
        Worker2 --> ProcessFiles2[Process Files 101-200]
        WorkerN --> ProcessFilesN[Process Files N...]
        
        ProcessFiles1 --> Results1[Results 1]
        ProcessFiles2 --> Results2[Results 2]
        ProcessFilesN --> ResultsN[Results N]
        
        Results1 --> Aggregator[Result Aggregator]
        Results2 --> Aggregator
        ResultsN --> Aggregator
        
        Aggregator --> MergeResults[Merge All Results]
        MergeResults --> FinalDataFrame[Final DataFrame]
    end
    
    subgraph "Synchronization"
        ProcessFiles1 --> SyncPoint[Synchronization Point]
        ProcessFiles2 --> SyncPoint
        ProcessFilesN --> SyncPoint
        
        SyncPoint --> CheckProgress{All Workers Complete?}
        CheckProgress -->|No| WaitWorkers[Wait for Workers]
        CheckProgress -->|Yes| ProceedMerge[Proceed to Merge]
        
        WaitWorkers --> CheckProgress
        ProceedMerge --> MergeResults
    end
```

This comprehensive set of end-to-end workflow examples demonstrates FlyRigLoader's capabilities across research, production, and enterprise integration scenarios. Each workflow is designed to showcase specific aspects of the system's architecture while providing practical guidance for real-world implementation patterns.

## 4.5 TECHNICAL IMPLEMENTATION DETAILS

### 4.5.1 State Management

#### 4.5.1.1 State Transitions and Persistence (updated)

The system implements comprehensive state management across all processing stages, ensuring data integrity and enabling efficient recovery mechanisms. The enhanced configuration management now leverages Pydantic models with embedded schema versioning for automatic migration support.

| Process Stage | State Storage | Persistence Strategy | Recovery Approach |
|---------------|---------------|---------------------|-------------------|
| Configuration | <span style="background-color: rgba(91, 57, 243, 0.2)">ProjectConfig (Pydantic)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">In-memory with schema_version</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic migration via migrate_config()</span> |
| Discovery | FileManifest | In-memory object | Re-execute discovery pattern |
| Loading | Raw data dictionary | In-memory storage | Re-load from file system |
| Transformation | DataFrame objects | In-memory processing | Re-transform from raw data |

<span style="background-color: rgba(91, 57, 243, 0.2)">*Note: LegacyConfigAdapter remains available for backward compatibility with dictionary-based configurations until the next major version release.*</span>

The configuration state management system now implements version-aware persistence with automatic migration capabilities. When configuration files are loaded, the system detects the embedded `schema_version` field and applies necessary migrations to ensure compatibility with the current system version. This approach eliminates breaking changes during system upgrades while maintaining research workflow continuity.

#### 4.5.1.2 Memory Management Strategy

The memory management system optimizes resource utilization through intelligent caching, lazy loading, and predictive cleanup mechanisms to maintain the <2x data size memory overhead performance target.

```mermaid
flowchart LR
    subgraph "Memory Optimization"
        A1[Lazy Loading] --> A2[Streaming Process]
        A2 --> A3[Batch Processing]
        A3 --> A4[Garbage Collection]
        
        A4 --> B1[Memory Monitoring]
        B1 --> B2[Threshold Check]
        B2 --> B3[Cleanup Trigger]
        B3 --> B4[Resource Release]
    end
    
    subgraph "Performance Targets"
        C1[<2x Data Size Overhead] --> C2[<1s per 100MB Load]
        C2 --> C3[<5s for 10k Files Discovery]
        C3 --> C4[<100ms Manifest Operations]
    end
```

The memory management strategy employs several key techniques:

- **Lazy Configuration Loading**: ProjectConfig objects are created on-demand and cached with weak references to prevent memory leaks
- **Streaming Data Processing**: Large datasets are processed in chunks to maintain consistent memory footprint
- **Intelligent Garbage Collection**: Automatic cleanup of expired manifests and temporary data structures
- **Memory Monitoring**: Real-time tracking of memory usage with automatic threshold-based cleanup

### 4.5.2 Concurrency and Thread Safety

#### 4.5.2.1 Thread-Safe Operations

The system implements comprehensive thread safety mechanisms across all shared resources, ensuring data integrity in concurrent environments while maintaining O(1) lookup performance for registry operations.

```mermaid
flowchart TD
    subgraph "Thread Safety"
        T1[Request Processing] --> T2{Shared Resource?}
        
        T2 -->|Yes| T3[Acquire Lock]
        T2 -->|No| T4[Direct Access]
        
        T3 --> T5[Critical Section]
        T5 --> T6[Release Lock]
        T6 --> T7[Continue Processing]
        
        T4 --> T7
        T7 --> T8[Return Result]
    end
    
    subgraph "Registry Safety"
        R1[Registry Access] --> R2[Singleton Pattern]
        R2 --> R3[Thread-Safe Operations]
        R3 --> R4[Atomic Updates]
        R4 --> R5[Consistent State]
    end
```

Thread safety is implemented through multiple layers:

- **Registry Singleton Pattern**: Class-level ReentrantLock ensures thread-safe registration and lookup operations
- **Configuration Immutability**: ProjectConfig objects are immutable after creation, eliminating race conditions
- **Atomic State Transitions**: Critical sections use atomic operations for state updates
- **Reader-Writer Locks**: Optimize concurrent read access while ensuring exclusive write operations

The registry system maintains thread safety through priority-based resolution (BUILTIN < USER < PLUGIN < OVERRIDE) with atomic conflict resolution to prevent deadlocks in multi-threaded environments.

### 4.5.3 Performance Optimization

#### 4.5.3.1 Optimization Strategies

The performance optimization framework employs multiple strategies across discovery, loading, and transformation phases to achieve enterprise-grade performance targets.

```mermaid
flowchart LR
    subgraph "Discovery Optimization"
        D1[Lazy File Stats] --> D2[Batch Stat Calls]
        D2 --> D3[Pattern Caching]
        D3 --> D4[Parallel Discovery]
    end
    
    subgraph "Loading Optimization"
        L1["Registry O(1) Lookup"] --> L2[Stream Large Files]
        L2 --> L3[Compression Detection]
        L3 --> L4[Memory Mapping]
    end
    
    subgraph "Transform Optimization"
        T1[Vectorized Operations] --> T2[Column Pruning]
        T2 --> T3[Type Inference Cache]
        T3 --> T4[Batch Processing]
    end
    
    subgraph "Caching Strategy"
        C1[Configuration Cache] --> C2[Pattern Cache]
        C2 --> C3[Schema Cache]
        C3 --> C4[Result Cache]
    end
```

Key optimization techniques include:

- **Configuration Caching**: ProjectConfig objects are cached with version-aware invalidation, reducing parsing overhead for repeated access
- **Pattern-Based Discovery**: Compiled regex patterns are cached to eliminate repeated compilation costs
- **Vectorized Transformations**: NumPy and pandas vectorized operations maximize CPU utilization
- **Memory-Mapped I/O**: Large files use memory mapping for efficient access patterns
- **Column-Level Optimization**: Selective column loading reduces memory footprint and processing time

The optimization system includes performance monitoring with automatic adaptation based on workload characteristics and system resources.

### 4.5.4 Transaction Boundaries

#### 4.5.4.1 Atomic Operations

The system defines clear transaction boundaries for each processing stage, ensuring data consistency and enabling reliable rollback mechanisms for error recovery.

| Operation | Transaction Start | Commit Point | Rollback Trigger |
|-----------|------------------|--------------|------------------|
| Config Load | File Open | Validation Success | Parse/Validation Error |
| File Discovery | Directory Access | Manifest Built | Access/Permission Error |
| Data Load | File Read Start | Data Validated | Read/Format Error |
| Transform | Raw Data Input | DataFrame Created | Schema/Type Violation |

Each transaction boundary implements ACID properties where applicable:

- **Atomicity**: Configuration loading either succeeds completely or fails without partial state
- **Consistency**: Schema validation ensures data integrity across all processing stages  
- **Isolation**: Concurrent operations maintain independent transaction contexts
- **Durability**: Successful operations persist their results in designated storage locations

The transaction management system includes automatic rollback capabilities for failed operations, comprehensive logging of transaction events, and recovery mechanisms that restore system state to the last known good configuration. This approach ensures research workflow reliability even in the presence of system failures or data corruption scenarios.

Version migration operations are treated as special transactions with backup creation before migration execution and automatic rollback capability if migration failures occur, preserving data integrity throughout the upgrade process.

## 4.6 VALIDATION AND COMPLIANCE

### 4.6.1 Validation Rules and Checkpoints

#### 4.6.1.1 Multi-Stage Validation Flow (updated)

```mermaid
flowchart TD
    subgraph "Configuration Validation"
        V1[YAML Syntax Check] --> V2[Security Validation]
        V2 --> V2_5[Schema Version Check]
        V2_5 --> V2_6{Version Compatible?}
        V2_6 -->|Yes| V3[Schema Validation]
        V2_6 -->|No| V2_7[Automatic Migration]
        V2_7 --> V3[Schema Validation]
        V3 --> V4[Path Validation]
        V4 --> V5[Environment Check]
    end
    
    subgraph "Discovery Validation"
        V6[Path Existence] --> V7[Access Rights]
        V7 --> V8[Pattern Compilation]
        V8 --> V9[Filter Logic]
        V9 --> V10[Metadata Extraction]
    end
    
    subgraph "Loading Validation"
        V11[File Existence] --> V12[Format Check]
        V12 --> V13[Data Structure]
        V13 --> V14[Size Limits]
        V14 --> V15[Integrity Check]
    end
    
    subgraph "Transform Validation"
        V16[Column Presence] --> V17[Type Matching]
        V17 --> V18[Dimension Check]
        V18 --> V19[Value Ranges]
        V19 --> V20[Schema Compliance]
    end
```

The enhanced validation flow incorporates <span style="background-color: rgba(91, 57, 243, 0.2)">schema version checking and automatic migration capabilities</span> to ensure backward compatibility during system upgrades. The <span style="background-color: rgba(91, 57, 243, 0.2)">Schema Version Check</span> step examines the embedded `schema_version` field in configuration files, comparing it against the current system version. When version incompatibilities are detected, the <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic Migration</span> process applies necessary transformations to upgrade legacy configurations to the current schema format, preserving research workflow continuity.

This version-aware validation approach eliminates breaking changes during system updates by automatically handling configuration schema evolution. The migration process maintains a comprehensive audit trail, logging all transformation steps for compliance and debugging purposes. Failed migrations trigger clear error messages with rollback capabilities to restore the previous configuration state.

#### 4.6.1.2 Business Rule Validation

```mermaid
flowchart LR
    subgraph "Research Data Rules"
        R1[Experiment Identifier] --> R2[Time Column Required]
        R2 --> R3[Metadata Completeness]
        R3 --> R4[Data Quality Checks]
    end
    
    subgraph "Security Rules"
        S1[Path Traversal Prevention] --> S2[File Access Control]
        S2 --> S3[Input Sanitization]
        S3 --> S4[Resource Limits]
    end
    
    subgraph "Compliance Rules"
        C1[Data Privacy] --> C2[Retention Policies]
        C2 --> C3[Audit Logging]
        C3 --> C4[Research Ethics]
    end
```

### 4.6.2 Timing and SLA Constraints

#### 4.6.2.1 Performance SLA Management

| Operation | Target SLA | Measurement Point | Fallback Action |
|-----------|------------|-------------------|-----------------|
| Config Load | <100ms | Load completion | Use cached config |
| Discovery (10k files) | <5s | Manifest ready | Limit search scope |
| Load (100MB) | <1s | Data in memory | Stream processing |
| Transform | <2x data size | DataFrame ready | Batch processing |
| API Response | <100ms | Response sent | Async processing |

#### 4.6.2.2 SLA Monitoring Flow

```mermaid
flowchart TB
    subgraph "Performance Monitoring"
        M1[Operation Start] --> M2[Start Timer]
        M2 --> M3[Execute Operation]
        M3 --> M4[Stop Timer]
        M4 --> M5[Check SLA]
        
        M5 --> M6{Within SLA?}
        M6 -->|Yes| M7[Log Success]
        M6 -->|No| M8[Log Violation]
        
        M8 --> M9[Trigger Alert]
        M9 --> M10[Adjust Strategy]
        M10 --> M11[Apply Fallback]
    end
```

### 4.6.3 Compliance Framework

#### 4.6.3.1 Regulatory Compliance Validation

The system implements comprehensive compliance validation mechanisms aligned with research data management standards, ensuring adherence to institutional policies and regulatory requirements throughout the data processing pipeline.

```mermaid
flowchart TD
    subgraph "Compliance Validation Pipeline"
        C1[Data Access Authorization] --> C2[Privacy Impact Assessment]
        C2 --> C3[Retention Policy Compliance]
        C3 --> C4[Audit Trail Generation]
        
        C4 --> C5{Compliance Check}
        C5 -->|Pass| C6[Proceed with Processing]
        C5 -->|Fail| C7[Generate Compliance Report]
        
        C7 --> C8[Notify Compliance Officer]
        C8 --> C9[Suspend Processing]
        
        C6 --> C10[Log Compliance Status]
        C10 --> C11[Continue Workflow]
    end
    
    subgraph "Audit and Monitoring"
        A1[Real-time Monitoring] --> A2[Compliance Metrics]
        A2 --> A3[Violation Detection]
        A3 --> A4[Automated Reporting]
        
        A4 --> A5{Critical Violation?}
        A5 -->|Yes| A6[Immediate Escalation]
        A5 -->|No| A7[Standard Reporting]
        
        A6 --> A8[System Lockdown]
        A7 --> A9[Periodic Review]
    end
```

The compliance framework addresses key areas of research data governance:

- **Data Privacy Protection**: Implements privacy-by-design principles with automated PII detection and sanitization capabilities
- **Access Control Validation**: Ensures proper authorization at each system boundary with role-based access control (RBAC) integration
- **Audit Trail Completeness**: Maintains comprehensive logs of all data access, transformation, and export activities
- **Retention Policy Enforcement**: Automatically manages data lifecycle according to institutional retention policies
- **Research Ethics Compliance**: Validates experiment protocols against approved research parameters

#### 4.6.3.2 Data Integrity Assurance

The data integrity validation system ensures research data reliability through multi-layered verification mechanisms, preventing data corruption and maintaining experimental result reproducibility.

```mermaid
flowchart LR
    subgraph "Integrity Validation Layers"
        I1[Checksum Verification] --> I2[Format Validation]
        I2 --> I3[Schema Consistency]
        I3 --> I4[Range Validation]
        I4 --> I5[Dependency Checks]
        
        I5 --> I6{Integrity Status}
        I6 -->|Valid| I7[Mark as Verified]
        I6 -->|Invalid| I8[Quarantine Data]
        
        I8 --> I9[Generate Error Report]
        I9 --> I10[Notify Data Custodian]
        I10 --> I11[Trigger Recovery Process]
    end
    
    subgraph "Recovery Mechanisms"
        R1[Backup Retrieval] --> R2[Data Reconstruction]
        R2 --> R3[Integrity Re-verification]
        R3 --> R4[Restore to Pipeline]
    end
```

Key integrity validation mechanisms include:

- **Cryptographic Checksums**: SHA-256 hashing for file integrity verification across all data transfer and storage operations
- **Format Consistency Checks**: Automated validation against expected data formats with version-specific schema enforcement
- **Cross-Reference Validation**: Verification of data relationships and dependencies across experiment datasets
- **Temporal Consistency**: Validation of timestamp sequences and experimental timeline integrity
- **Metadata Completeness**: Verification of required metadata fields and experimental context information

The system maintains detailed integrity audit logs, enabling forensic analysis of data quality issues and providing evidence for research reproducibility requirements. Failed integrity checks trigger automated quarantine procedures, preventing corrupted data from contaminating downstream analysis pipelines.

### 4.6.4 Security Validation

#### 4.6.4.1 Security Checkpoint Framework

The security validation framework implements defense-in-depth principles with multiple security checkpoints throughout the data processing pipeline, ensuring protection against both external threats and internal security policy violations.

```mermaid
flowchart TB
    subgraph "Input Security Validation"
        S1[Path Traversal Prevention] --> S2[Input Sanitization]
        S2 --> S3[File Type Validation]
        S3 --> S4[Size Limit Enforcement]
        
        S4 --> S5{Security Risk Assessment}
        S5 -->|Low Risk| S6[Proceed to Processing]
        S5 -->|Medium Risk| S7[Enhanced Monitoring]
        S5 -->|High Risk| S8[Block and Alert]
        
        S7 --> S6
        S8 --> S9[Security Incident Report]
        S9 --> S10[Administrator Notification]
    end
    
    subgraph "Runtime Security Monitoring"
        R1[Resource Usage Monitoring] --> R2[Privilege Escalation Detection]
        R2 --> R3[Anomaly Detection]
        R3 --> R4[Network Activity Monitoring]
        
        R4 --> R5{Threat Level}
        R5 -->|Normal| R6[Continue Operations]
        R5 -->|Elevated| R7[Increase Monitoring]
        R5 -->|Critical| R8[Emergency Shutdown]
        
        R8 --> R9[Forensic Data Collection]
        R9 --> R10[Incident Response]
    end
```

Security validation encompasses multiple protection layers:

- **Input Validation and Sanitization**: Comprehensive validation of all user inputs, file paths, and configuration parameters to prevent injection attacks and path traversal vulnerabilities
- **Authentication and Authorization**: Integration with institutional identity management systems and enforcement of role-based access controls throughout the data pipeline
- **Encryption and Data Protection**: End-to-end encryption for data in transit and at rest, with key management aligned to institutional security policies
- **Network Security**: Network traffic monitoring and validation to detect unauthorized data exfiltration or external intrusion attempts
- **Audit and Forensics**: Comprehensive security event logging with tamper-evident audit trails for incident investigation and compliance reporting

The security framework adapts its threat response based on institutional security policies and threat intelligence feeds, ensuring dynamic protection against evolving security threats while maintaining research workflow continuity.

#### 4.6.4.2 Vulnerability Management

The vulnerability management system provides continuous security assessment and remediation capabilities, ensuring the system maintains security compliance as threats evolve and new vulnerabilities are discovered.

| Vulnerability Category | Detection Method | Remediation SLA | Escalation Trigger |
|------------------------|------------------|-----------------|-------------------|
| Critical (CVSS 9.0-10.0) | Automated scanning | 24 hours | Immediate |
| High (CVSS 7.0-8.9) | Weekly assessment | 72 hours | 48 hours |
| Medium (CVSS 4.0-6.9) | Monthly review | 30 days | 15 days |
| Low (CVSS 0.1-3.9) | Quarterly audit | 90 days | 60 days |

The vulnerability management process includes:

- **Continuous Vulnerability Scanning**: Automated security scanning of all system components with integration to institutional vulnerability management platforms
- **Threat Intelligence Integration**: Real-time threat intelligence feeds to identify emerging threats relevant to research data processing systems
- **Patch Management**: Coordinated patching strategy with research workflow impact assessment and maintenance window scheduling
- **Security Testing**: Regular penetration testing and security audits conducted by qualified security professionals
- **Incident Response Preparation**: Maintained incident response playbooks with clear escalation procedures and stakeholder notification protocols

Security validation results are integrated into the overall system health monitoring dashboard, providing real-time visibility into security posture and enabling proactive threat mitigation before security incidents occur.

## 4.7 SECURITY CONSIDERATIONS

### 4.7.1 Security Validation and Protection

#### 4.7.1.1 Input Security Validation Flow

```mermaid
flowchart LR
    subgraph "Input Validation"
        I1[User Input] --> PathCheck{Path Traversal?}
        
        PathCheck -->|Yes| BlockPath[Block Access]
        PathCheck -->|No| URLCheck{Remote URL?}
        
        URLCheck -->|Yes| BlockURL[Block URL]
        URLCheck -->|No| SysPathCheck{System Path?}
        
        SysPathCheck -->|Yes| BlockSys[Block System]
        SysPathCheck -->|No| AllowAccess[Allow Access]
    end
    
    subgraph "Runtime Protection"
        R1[YAML Load] --> SafeLoad[Use safe_load]
        R2[Regex Compile] --> TimeoutCheck[Timeout Protection]
        R3[File Access] --> PermissionCheck[Check Permissions]
        R4[Process Memory] --> MemoryCheck[Memory Limits]
    end
    
    subgraph "Security Logging"
        BlockPath --> SecurityLog
        BlockURL --> SecurityLog
        BlockSys --> SecurityLog[Log Security Event]
        
        SecurityLog --> AlertSystem[Alert System]
        AlertSystem --> Terminate([Terminate Request])
    end
    
    AllowAccess --> Continue([Continue Processing])
```

#### 4.7.1.2 Data Protection Measures

```mermaid
flowchart TD
    subgraph "Data Protection"
        D1[Sensitive Data Detection] --> D2[Access Control]
        D2 --> D3[Encryption at Rest]
        D3 --> D4[Secure Transmission]
        D4 --> D5[Audit Trail]
    end
    
    subgraph "Access Control"
        A1[Authentication] --> A2[Authorization]
        A2 --> A3[Role-Based Access]
        A3 --> A4[Resource Limits]
        A4 --> A5[Session Management]
    end
    
    subgraph "Compliance"
        C1[Data Retention] --> C2[Privacy Controls]
        C2 --> C3[Consent Management]
        C3 --> C4[Research Ethics]
        C4 --> C5[Regulatory Compliance]
    end
```

## 4.8 MONITORING AND OBSERVABILITY

### 4.8.1 Logging and Monitoring

#### 4.8.1.1 Structured Logging Flow

The Structured Logging System uses Loguru with automatic rotation, console/file outputs, and test mode support, improving debugging efficiency and enabling production monitoring.

```mermaid
flowchart TD
    subgraph "Log Sources"
        L1[API Calls] --> Logger
        L2[Discovery Events] --> Logger
        L3[Load Operations] --> Logger
        L4[Transform Steps] --> Logger
        L5[Errors/Warnings] --> Logger
        L6[Performance Metrics] --> Logger
        L7[Registry Events] --> Logger
    end
    
    subgraph "Loguru Processing"
        Logger[Loguru Logger] --> Level{Log Level}
        
        Level -->|DEBUG| FileHandler[File Handler]
        Level -->|INFO| ConsoleHandler[Console Handler]
        Level -->|ERROR| ErrorHandler[Error Handler]
        Level -->|TRACE| TraceHandler[Trace Handler]
        Level -->|WARNING| WarningHandler[Warning Handler]
        
        FileHandler --> Rotation[10MB Rotation]
        Rotation --> Compression[Gzip Old Logs]
        
        ConsoleHandler --> Colorize[Colorized Output]
        ErrorHandler --> Context[Add Context]
        TraceHandler --> DetailedTrace[Detailed Trace]
        WarningHandler --> RegistryConflicts[Registry Conflict Logs]
        
        RegistryConflicts --> ConflictContext[Add Conflict Context]
    end
    
    subgraph "Log Outputs"
        Compression --> LogFiles[Log Files]
        Colorize --> Terminal[Terminal Display]
        Context --> Alerts[Error Alerts]
        DetailedTrace --> DebugOutput[Debug Output]
        ConflictContext --> RegistryWarnings[Registry Warning Logs]
    end
    
    subgraph "Monitoring Integration"
        LogFiles --> Monitoring[Monitoring System]
        Alerts --> NotificationSystem[Notification System]
        DebugOutput --> DeveloperTools[Developer Tools]
        RegistryWarnings --> RegistryMonitoring[Registry Monitoring]
    end
```

#### 4.8.1.2 Registry Event Logging Flow (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The registry event logging system captures comprehensive information about plugin registration, conflicts, and resolution processes to support debugging and monitoring of the extensible loader architecture.</span>

```mermaid
flowchart TD
    subgraph "Registry Event Sources"
        RE1[Plugin Registration] --> RELogger[Registry Event Logger]
        RE2[Entry Point Discovery] --> RELogger
        RE3[Priority Resolution] --> RELogger
        RE4[Registration Conflicts] --> RELogger
        RE5[Plugin Validation] --> RELogger
        RE6[Registry Updates] --> RELogger
    end
    
    subgraph "Registry Event Processing"
        RELogger --> RELevel{Event Severity}
        
        RELevel -->|SUCCESS| InfoLog[INFO: Successful Registration]
        RELevel -->|CONFLICT| WarningLog[WARNING: Registry Conflicts]
        RELevel -->|FAILURE| ErrorLog[ERROR: Registration Failed]
        RELevel -->|VALIDATION| DebugLog[DEBUG: Validation Steps]
        
        WarningLog --> ConflictDetails[Add Conflict Details]
        ConflictDetails --> ConflictResolution[Log Resolution Strategy]
        
        InfoLog --> SuccessContext[Add Success Context]
        ErrorLog --> FailureContext[Add Failure Context]
        DebugLog --> ValidationContext[Add Validation Context]
    end
    
    subgraph "Registry Log Outputs"
        ConflictResolution --> RegistryWarningLogs[Registry Warning Logs]
        SuccessContext --> RegistryInfoLogs[Registry Info Logs]
        FailureContext --> RegistryErrorLogs[Registry Error Logs]
        ValidationContext --> RegistryDebugLogs[Registry Debug Logs]
    end
    
    subgraph "Registry Monitoring"
        RegistryWarningLogs --> ConflictMonitoring[Conflict Monitoring Dashboard]
        RegistryInfoLogs --> SuccessMetrics[Registration Success Metrics]
        RegistryErrorLogs --> FailureAlerts[Registration Failure Alerts]
        RegistryDebugLogs --> DiagnosticTools[Plugin Diagnostic Tools]
    end
```

#### 4.8.1.3 Metrics Collection

```mermaid
flowchart LR
    subgraph "Performance Metrics"
        P1[Response Time] --> P2[Throughput]
        P2 --> P3[Memory Usage]
        P3 --> P4[Error Rate]
        P4 --> P5[Success Rate]
    end
    
    subgraph "Business Metrics"
        B1[Files Processed] --> B2[Data Volume]
        B2 --> B3[Pipeline Success]
        B3 --> B4[User Satisfaction]
        B4 --> B5[Research Outcomes]
    end
    
    subgraph "Technical Metrics"
        T1[CPU Usage] --> T2[I/O Operations]
        T2 --> T3[Network Traffic]
        T3 --> T4[Cache Hit Rate]
        T4 --> T5[Plugin Usage]
    end
    
    subgraph "Registry Metrics"
        R1[Plugin Registration Rate] --> R2[Conflict Resolution Rate]
        R2 --> R3[Registry Lookup Performance]
        R3 --> R4[Plugin Load Success Rate]
        R4 --> R5[Entry Point Discovery Time]
    end
    
    subgraph "Aggregation"
        P5 --> Collector[Metrics Collector]
        B5 --> Collector
        T5 --> Collector
        R5 --> Collector
        
        Collector --> Dashboard[Monitoring Dashboard]
        Dashboard --> Alerts[Alert System]
    end
```

### 4.8.2 Health Monitoring

#### 4.8.2.1 System Health Checks

```mermaid
flowchart TD
    subgraph "Health Checks"
        H1[Configuration Health] --> H2[File System Health]
        H2 --> H3[Memory Health]
        H3 --> H4[Performance Health]
        H4 --> H5[Plugin Health]
        H5 --> H6[Registry Health]
    end
    
    subgraph "Health Assessment"
        H6 --> Assessment{Overall Health}
        
        Assessment -->|Healthy| GreenStatus[Green Status]
        Assessment -->|Warning| YellowStatus[Yellow Status]
        Assessment -->|Critical| RedStatus[Red Status]
        
        GreenStatus --> Continue[Continue Operations]
        YellowStatus --> Alert[Send Alert]
        RedStatus --> Emergency[Emergency Response]
    end
    
    subgraph "Recovery Actions"
        Alert --> AutoRecover[Auto Recovery]
        Emergency --> ManualIntervention[Manual Intervention]
        
        AutoRecover --> Retry[Retry Operations]
        ManualIntervention --> Escalate[Escalate to Team]
    end
```

#### 4.8.2.2 Registry Health Monitoring (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The registry health monitoring system provides comprehensive visibility into plugin registration status, conflict resolution effectiveness, and registry performance metrics to ensure reliable extensibility operations.</span>

```mermaid
flowchart TD
    subgraph "Registry Health Indicators"
        RH1[Plugin Registration Success Rate] --> RH2[Conflict Resolution Rate]
        RH2 --> RH3[Registry Lookup Performance]
        RH3 --> RH4[Entry Point Discovery Health]
        RH4 --> RH5[Plugin Validation Success]
        RH5 --> RH6[Priority Resolution Accuracy]
    end
    
    subgraph "Registry Health Assessment"
        RH6 --> RegistryAssessment{Registry Health Status}
        
        RegistryAssessment -->|>95% Success| RegistryHealthy[Registry Healthy]
        RegistryAssessment -->|80-95% Success| RegistryWarning[Registry Warning]
        RegistryAssessment -->|<80% Success| RegistryCritical[Registry Critical]
        
        RegistryHealthy --> ContinueRegistration[Continue Registration]
        RegistryWarning --> InvestigateConflicts[Investigate Conflicts]
        RegistryCritical --> DisableProblematicPlugins[Disable Problematic Plugins]
    end
    
    subgraph "Registry Recovery Actions"
        InvestigateConflicts --> ConflictAnalysis[Analyze Conflict Patterns]
        DisableProblematicPlugins --> FallbackMode[Enable Fallback Mode]
        
        ConflictAnalysis --> ResolveConflicts[Auto-Resolve Conflicts]
        FallbackMode --> NotifyAdministrators[Notify Administrators]
        
        ResolveConflicts --> ValidateResolution[Validate Resolution]
        NotifyAdministrators --> ManualRegistryRepair[Manual Registry Repair]
    end
    
    subgraph "Registry Alerting"
        RegistryWarning --> ConflictAlert[Conflict Alert]
        RegistryCritical --> CriticalAlert[Critical Registry Alert]
        
        ConflictAlert --> RegistryTeam[Registry Team Notification]
        CriticalAlert --> EscalationPath[Emergency Escalation]
    end
```

### 4.8.3 Observability and Tracing

#### 4.8.3.1 End-to-End Observability Flow

```mermaid
flowchart TD
    subgraph "Observability Sources"
        O1[User Requests] --> Tracer[Distributed Tracer]
        O2[System Operations] --> Tracer
        O3[Plugin Operations] --> Tracer
        O4[Registry Operations] --> Tracer
        O5[Error Scenarios] --> Tracer
    end
    
    subgraph "Trace Processing"
        Tracer --> SpanCreation[Create Trace Spans]
        SpanCreation --> ContextPropagation[Context Propagation]
        ContextPropagation --> SpanEnrichment[Enrich with Metadata]
        
        SpanEnrichment --> SpanCollection[Collect Spans]
        SpanCollection --> TraceAssembly[Assemble Traces]
    end
    
    subgraph "Observability Outputs"
        TraceAssembly --> TraceStorage[Trace Storage]
        TraceStorage --> TraceAnalysis[Trace Analysis]
        TraceAnalysis --> PerformanceInsights[Performance Insights]
        
        PerformanceInsights --> OptimizationRecommendations[Optimization Recommendations]
        OptimizationRecommendations --> CapacityPlanning[Capacity Planning]
    end
    
    subgraph "Observability Integration"
        TraceStorage --> APMTools[APM Tools Integration]
        TraceAnalysis --> AlertingRules[Custom Alerting Rules]
        PerformanceInsights --> BusinessMetrics[Business Metrics Dashboard]
        
        APMTools --> ExternalMonitoring[External Monitoring Systems]
        AlertingRules --> IncidentResponse[Incident Response Workflow]
        BusinessMetrics --> StakeholderReporting[Stakeholder Reporting]
    end
```

#### 4.8.3.2 Performance Monitoring and SLA Tracking

```mermaid
flowchart LR
    subgraph "Performance Targets"
        PT1["Discovery: <5s for 10K files"] --> SLAMonitor[SLA Monitor]
        PT2["Loading: <1s per 100MB"] --> SLAMonitor
        PT3["Transform: <2s per 1M rows"] --> SLAMonitor
        PT4["Manifest: <100ms response"] --> SLAMonitor
        PT5["Registry Lookup: O(1)"] --> SLAMonitor
    end
    
    subgraph "Performance Measurement"
        SLAMonitor --> RealTimeMetrics[Real-time Metrics]
        RealTimeMetrics --> SLACompliance[SLA Compliance Tracking]
        SLACompliance --> PerformanceTrends[Performance Trend Analysis]
        
        PerformanceTrends --> Forecasting[Performance Forecasting]
        Forecasting --> CapacityAlerts[Capacity Alerts]
    end
    
    subgraph "Performance Actions"
        CapacityAlerts --> ScalingRecommendations[Scaling Recommendations]
        ScalingRecommendations --> AutoScaling[Auto-scaling Triggers]
        AutoScaling --> ResourceOptimization[Resource Optimization]
        
        ResourceOptimization --> PerformanceReporting[Performance Reporting]
        PerformanceReporting --> StakeholderNotification[Stakeholder Notification]
    end
```

## 4.9 INTEGRATION PATTERNS

### 4.9.1 External System Integration

#### 4.9.1.1 Multi-Platform Integration Architecture (updated)

```mermaid
flowchart TB
    subgraph "Research Platforms"
        K1[Kedro Pipelines] --> DS[FlyRigLoaderDataSet]
        DS[FlyRigLoaderDataSet] --> API[FlyRigLoader API]
        J1[Jupyter Notebooks] --> API
        C1[CLI Tools] --> API
        H1[HPC Environments] --> API
    end
    
    subgraph "Data Pipeline Integration"
        API --> D1[Data Discovery]
        API --> D2[Data Loading]
        API --> D3[Data Transformation]
        
        D1 --> Pipeline[Research Pipeline]
        D2 --> Pipeline
        D3 --> Pipeline
        
        Pipeline --> A1[Analysis Tools]
        Pipeline --> V1[Visualization Tools]
        Pipeline --> R1[Reporting Tools]
    end
    
    subgraph "Storage Integration"
        FileSystem[File System] --> API
        Database[Database] --> API
        Cloud[Cloud Storage] --> API
        
        API --> Cache[Caching Layer]
        Cache --> Results[Results Storage]
    end
    
    subgraph "Monitoring Integration"
        API --> Metrics[Metrics Collection]
        Metrics --> Monitoring[Monitoring System]
        Monitoring --> Alerts[Alert System]
        Alerts --> Notifications[Notifications]
    end
```

The integration architecture demonstrates enterprise-grade multi-platform connectivity patterns with <span style="background-color: rgba(91, 57, 243, 0.2)">dedicated Kedro DataSet abstraction layer that implements the AbstractDataset interface</span>. This pattern ensures proper separation of concerns between framework-specific integration layers and the core FlyRigLoader API, enabling seamless integration with Kedro's data versioning, lineage tracking, and pipeline execution framework.

<span style="background-color: rgba(91, 57, 243, 0.2)">The FlyRigLoaderDataSet serves as a bridge component that translates Kedro catalog configurations into FlyRigLoader API calls, providing first-class support for catalog.yml configuration while maintaining all core performance characteristics. This abstraction enables Kedro pipelines to leverage FlyRigLoader's discovery and loading capabilities through familiar Kedro interfaces.</span>

#### 4.9.1.2 Enterprise Integration Patterns

```mermaid
flowchart LR
    subgraph "Enterprise Integration"
        E1[Enterprise Data Warehouse] --> E2[ETL Pipeline]
        E2 --> E3[Data Lake]
        E3 --> API[FlyRigLoader API]
        
        API --> E4[Analytics Platform]
        API --> E5[ML Platform]
        API --> E6[Reporting Platform]
    end
    
    subgraph "Security Integration"
        S1[Identity Provider] --> S2[Authentication]
        S2 --> S3[Authorization]
        S3 --> API
        
        API --> S4[Audit Logging]
        S4 --> S5[Compliance System]
    end
    
    subgraph "DevOps Integration"
        D1[CI/CD Pipeline] --> D2[Testing]
        D2 --> D3[Deployment]
        D3 --> API
        
        API --> D4[Monitoring]
        D4 --> D5[Alerting]
        D5 --> D6[Incident Response]
    end
```

The enterprise integration patterns establish comprehensive connectivity frameworks for production environments. These patterns enable secure, monitored, and auditable data flows across enterprise infrastructure while maintaining the system's core performance targets and ensuring compliance with organizational security requirements.

### 4.9.2 API Integration Patterns

#### 4.9.2.1 API Gateway Pattern

```mermaid
flowchart TD
    subgraph "Client Applications"
        C1[Web Application] --> Gateway[API Gateway]
        C2[Mobile App] --> Gateway
        C3[CLI Tool] --> Gateway
        C4[Third-Party Service] --> Gateway
    end
    
    subgraph "Gateway Services"
        Gateway --> Auth[Authentication]
        Gateway --> RateLimit[Rate Limiting]
        Gateway --> LoadBalance[Load Balancing]
        Gateway --> Cache[Caching]
    end
    
    subgraph "Backend Services"
        Auth --> FlyRigLoader[FlyRigLoader API]
        RateLimit --> FlyRigLoader
        LoadBalance --> FlyRigLoader
        Cache --> FlyRigLoader
        
        FlyRigLoader --> Services[Supporting Services]
        Services --> Database[Database]
        Services --> Storage[File Storage]
    end
    
    subgraph "Monitoring"
        Gateway --> Metrics[Metrics Collection]
        FlyRigLoader --> Metrics
        Metrics --> Dashboard[Monitoring Dashboard]
    end
```

The API Gateway pattern provides a unified entry point for all client applications, implementing cross-cutting concerns such as authentication, rate limiting, load balancing, and caching. This pattern ensures consistent security enforcement, performance optimization, and monitoring across all API consumers while maintaining the flexibility for different client application architectures.

#### 4.9.2.2 Event-Driven Integration Pattern

```mermaid
flowchart TB
    subgraph "Event Producers"
        P1[File System Changes] --> EventBus[Event Bus]
        P2[Configuration Updates] --> EventBus
        P3[Data Load Completion] --> EventBus
        P4[Error Conditions] --> EventBus
    end
    
    subgraph "Event Processing"
        EventBus --> Filter[Event Filter]
        Filter --> Router[Event Router]
        Router --> Queue[Message Queue]
        Queue --> Workers[Event Workers]
    end
    
    subgraph "Event Consumers"
        Workers --> C1[Cache Invalidation]
        Workers --> C2[Notification Service]
        Workers --> C3[Metrics Collection]
        Workers --> C4[Audit Logging]
    end
    
    subgraph "Integration Points"
        C1 --> Cache[Configuration Cache]
        C2 --> Alerts[Alert System]
        C3 --> Monitoring[Monitoring Dashboard]
        C4 --> AuditDB[Audit Database]
    end
```

The event-driven integration pattern enables loosely coupled, reactive system behavior that responds to changes in configuration, data availability, and system state. This pattern supports real-time monitoring, automatic cache invalidation, and proactive error handling while maintaining system responsiveness and scalability.

#### 4.9.2.3 Circuit Breaker Pattern

```mermaid
flowchart LR
    subgraph "Client Requests"
        Client[Client Application] --> CB[Circuit Breaker]
    end
    
    subgraph "Circuit States"
        CB --> State{State Check}
        State -->|Closed| Normal[Normal Operation]
        State -->|Open| Fallback[Fallback Response]
        State -->|Half-Open| Test[Test Request]
        
        Normal --> Success{Success?}
        Success -->|Yes| Reset[Reset Counter]
        Success -->|No| Increment[Increment Failures]
        
        Increment --> Threshold{Threshold Reached?}
        Threshold -->|Yes| Open[Open Circuit]
        Threshold -->|No| Normal
        
        Test --> TestResult{Test Success?}
        TestResult -->|Yes| Close[Close Circuit]
        TestResult -->|No| Reopen[Reopen Circuit]
    end
    
    subgraph "Backend Services"
        Normal --> API[FlyRigLoader API]
        Test --> API
        API --> Response[API Response]
        Response --> Client
        
        Fallback --> Cache[Cached Data]
        Cache --> FallbackResponse[Fallback Response]
        FallbackResponse --> Client
    end
```

The circuit breaker pattern provides resilience against cascading failures by monitoring service health and automatically switching to fallback mechanisms when error thresholds are exceeded. This pattern ensures system stability during high-error conditions while providing graceful degradation through cached responses and alternative data sources.

### 4.9.3 Data Integration Workflows

#### 4.9.3.1 Batch Processing Integration

```mermaid
flowchart TD
    subgraph "Batch Scheduling"
        Scheduler[Job Scheduler] --> Trigger[Batch Trigger]
        Trigger --> Config[Load Configuration]
        Config --> Validate[Validate Parameters]
    end
    
    subgraph "Batch Processing"
        Validate --> Discovery[File Discovery]
        Discovery --> Partition[Data Partitioning]
        Partition --> Parallel[Parallel Processing]
        
        Parallel --> Worker1[Worker 1]
        Parallel --> Worker2[Worker 2]
        Parallel --> Worker3[Worker N]
        
        Worker1 --> Merge[Result Merging]
        Worker2 --> Merge
        Worker3 --> Merge
    end
    
    subgraph "Result Handling"
        Merge --> Quality[Quality Checks]
        Quality --> Success{Quality Pass?}
        Success -->|Yes| Store[Store Results]
        Success -->|No| Retry[Retry Logic]
        
        Store --> Notify[Completion Notification]
        Retry --> Discovery
    end
    
    subgraph "Monitoring"
        Parallel --> Metrics[Job Metrics]
        Quality --> Metrics
        Metrics --> Dashboard[Batch Dashboard]
    end
```

The batch processing integration pattern enables large-scale data processing workflows with parallel execution, quality validation, and comprehensive monitoring. This pattern supports research workflows that require processing thousands of files with guaranteed completion and error recovery capabilities.

#### 4.9.3.2 Stream Processing Integration

```mermaid
flowchart LR
    subgraph "Data Sources"
        S1[File System Events] --> Stream[Data Stream]
        S2[Real-time Updates] --> Stream
        S3[Configuration Changes] --> Stream
    end
    
    subgraph "Stream Processing"
        Stream --> Buffer[Event Buffer]
        Buffer --> Window[Time Window]
        Window --> Transform[Stream Transform]
        Transform --> Aggregate[Aggregation]
    end
    
    subgraph "Real-time Actions"
        Aggregate --> Action{Action Type}
        Action -->|Update| Cache[Update Cache]
        Action -->|Alert| Notification[Send Alert]
        Action -->|Log| Audit[Audit Log]
        
        Cache --> API[API Response Update]
        Notification --> External[External Systems]
        Audit --> Storage[Persistent Storage]
    end
    
    subgraph "Feedback Loop"
        API --> Monitor[Performance Monitor]
        Monitor --> Adjust[Adjust Parameters]
        Adjust --> Buffer
    end
```

The stream processing integration pattern provides real-time data processing capabilities with automatic adjustment and feedback mechanisms. This pattern enables responsive system behavior for dynamic research environments where configuration changes and data updates require immediate processing and notification.

### 4.9.4 Security Integration Patterns

#### 4.9.4.1 Zero-Trust Integration Architecture

```mermaid
flowchart TB
    subgraph "Identity Verification"
        User[User/Service] --> IdP[Identity Provider]
        IdP --> MFA[Multi-Factor Auth]
        MFA --> Token[JWT Token]
    end
    
    subgraph "Authorization Gateway"
        Token --> AuthZ[Authorization Service]
        AuthZ --> RBAC[Role-Based Access]
        RBAC --> Policy[Policy Engine]
        Policy --> Decision[Access Decision]
    end
    
    subgraph "Resource Access"
        Decision --> Gateway[Security Gateway]
        Gateway --> Encrypt[Data Encryption]
        Encrypt --> API[FlyRigLoader API]
        
        API --> DataAccess[Data Access]
        DataAccess --> Audit[Access Logging]
        Audit --> SIEM[Security Monitoring]
    end
    
    subgraph "Continuous Verification"
        SIEM --> Behavior[Behavior Analysis]
        Behavior --> Risk[Risk Assessment]
        Risk --> Response[Adaptive Response]
        Response --> Policy
    end
```

The zero-trust security integration pattern ensures comprehensive security validation at every access point with continuous monitoring and adaptive response capabilities. This pattern provides enterprise-grade security for sensitive research data while maintaining system performance and user experience.

#### 4.9.4.2 Data Privacy Integration

```mermaid
flowchart LR
    subgraph "Data Classification"
        Source[Data Source] --> Classify[Data Classification]
        Classify --> Sensitivity[Sensitivity Level]
        Sensitivity --> Policy[Privacy Policy]
    end
    
    subgraph "Privacy Controls"
        Policy --> Mask[Data Masking]
        Policy --> Encrypt[Field Encryption]
        Policy --> Access[Access Controls]
        
        Mask --> Safe[Safe Data View]
        Encrypt --> Secure[Secure Storage]
        Access --> Authorized[Authorized Access]
    end
    
    subgraph "Compliance Monitoring"
        Safe --> Monitor[Access Monitoring]
        Secure --> Monitor
        Authorized --> Monitor
        
        Monitor --> Compliance[Compliance Check]
        Compliance --> Report[Compliance Report]
        Report --> Audit[External Audit]
    end
```

The data privacy integration pattern implements comprehensive privacy controls with automated compliance monitoring and reporting. This pattern ensures adherence to regulatory requirements such as GDPR, HIPAA, and institutional data governance policies while maintaining full system functionality for authorized research activities.

## 4.10 SUMMARY

### 4.10.1 Key Workflow Characteristics

The FlyRigLoader system implements a comprehensive set of workflows optimized for neuroscience research data management:

- **Decoupled Pipeline Architecture**: Three-stage processing (Discovery → Loading → Transformation) with clear separation of concerns
- **Plugin-Based Extensibility**: Registry pattern enabling <50 lines of code for new format support
- **Performance-Optimized**: Achieving <5s discovery for 10,000 files and <1s loading per 100MB
- **Robust Error Handling**: Domain-specific exception hierarchy reducing debugging time by 60%
- **Research-Focused Integration**: Seamless integration with Kedro, Jupyter, and HPC environments
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Version-Aware Configuration with automatic migration**: Embedded schema_version tracking with seamless upgrade paths</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Builder Pattern API for standardized config creation**: Type-safe configuration construction with fluent interfaces</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**First-class Kedro DataSet integration for catalog.yml usage**: Native FlyRigLoaderDataSet supporting production pipelines</span>

### 4.10.2 Technical Implementation Highlights

- **Memory Efficiency**: <2x data size overhead through streaming and lazy evaluation
- **Thread Safety**: Singleton registries with atomic operations
- **Comprehensive Validation**: Multi-stage validation with Pydantic v2 schema enforcement
- **Security-First Design**: Input validation, path traversal prevention, and audit logging
- **Observable Operations**: Structured logging with automatic rotation and performance monitoring
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Configuration version tracking and migration infrastructure**: Automatic version detection with backward-compatible upgrades</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Comprehensive registry event logging**: Structured logging for all plugin registration and resolution events</span>

### 4.10.3 Business Process Support

The workflows documented in this section directly support the core research objectives:

- **Automated Data Discovery**: Reducing manual file management overhead by 90%
- **Standardized Data Processing**: Ensuring consistency across experiments and research teams
- **Extensible Architecture**: Supporting evolving research needs without core modifications
- **Collaborative Research**: Configuration-as-code supporting reproducible research workflows

#### References

**Technical Specification Sections Referenced**:
- `1.2 SYSTEM OVERVIEW` - System architecture and integration context
- `2.1 FEATURE CATALOG` - Feature definitions and requirements
- `3.7 TECHNOLOGY STACK ARCHITECTURE` - Technical implementation details

**Repository Analysis**:
- `src/flyrigloader/api.py` - Main API workflows and integration patterns
- `src/flyrigloader/config/` - Configuration management and validation workflows
- `src/flyrigloader/discovery/` - File discovery and pattern matching workflows
- `src/flyrigloader/io/` - Data loading and transformation pipelines
- `src/flyrigloader/utils/` - Supporting utilities and error handling
- `tests/` - Test infrastructure and validation workflows

**Performance Specifications**:
- Discovery Performance: <5 seconds for 10,000 files
- Loading Performance: <1 second per 100MB
- Memory Efficiency: <2x data size overhead
- API Response: <100ms for manifest operations
- Error Reduction: 90% configuration errors eliminated, 60% debugging time reduced

**Integration Environments**:
- Kedro pipeline integration for production data workflows
- Jupyter notebook integration for interactive research
- HPC environment support for cluster computing
- CLI tool integration for batch processing workflows

# 5. SYSTEM ARCHITECTURE

## 5.1 HIGH-LEVEL ARCHITECTURE

### 5.1.1 System Overview

The FlyRigLoader system implements a **layered, protocol-based architecture** designed for scientific data management in neuroscience research. The system addresses the growing need for standardized data management in laboratories studying neural circuits using optogenetic techniques, positioning itself as the bridge between raw experimental data collection and modern data science workflows. <span style="background-color: rgba(91, 57, 243, 0.2)">The system now includes first-class Kedro integration through the `flyrigloader.kedro` subpackage, providing native `FlyRigLoaderDataSet` support for production pipeline workflows.</span>

**Architecture Style and Rationale:**
- **Layered Architecture**: Provides clear separation of concerns with distinct layers for API, configuration, discovery, I/O, and transformation, enabling independent testing and phased migration from legacy systems
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Layer</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Dedicated layer responsible for configuration version detection and automatic migration, ensuring seamless upgrades while preserving research continuity</span>
- **Protocol-Based Design**: Leverages Python protocols for dependency injection, enabling comprehensive testability and component substitution without heavy DI frameworks
- **Plugin Architecture**: Registry-based extensibility allows new data formats and schemas without core modifications, supporting the evolving needs of research environments
- **Decoupled Pipeline**: Three-stage processing (Discover → Load → Transform) enables selective data processing and memory efficiency for large datasets

**Key Architectural Principles:**
- **Domain-Driven Design**: Components organized around scientific workflow concepts (experiments, datasets, manifests)
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Feature-flag based parallel implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Enables non-disruptive rollout of new functionality through controlled feature activation</span>
- **Fail-Fast Validation**: Pydantic models enforce data integrity at system boundaries, eliminating 90% of configuration-related errors
- **Progressive Enhancement**: Legacy compatibility maintained while enabling modern patterns
- **Resource Efficiency**: Lazy evaluation and streaming for large dataset handling, targeting <2x memory overhead

<span style="background-color: rgba(91, 57, 243, 0.2)">**Configuration Management Strategy:**
All configuration handling is now driven by a Pydantic-backed builder pattern architecture, providing type-safe configuration construction with comprehensive validation. The system maintains seamless backward compatibility through transparent `LegacyConfigAdapter` support, automatically detecting and migrating legacy dictionary-based configurations while preserving existing workflows.</span>

**System Boundaries:**
- **Input**: YAML configurations, file system data, environment variables
- **Processing**: In-memory data transformation with optional persistence
- **Output**: File manifests, raw data dictionaries, pandas DataFrames<span style="background-color: rgba(91, 57, 243, 0.2)">, migration reports, deprecation-warning logs</span>
- **External Interfaces**: File system access, Python package ecosystem via entry points

### 5.1.2 Core Components Table

| Component Name | Primary Responsibility | Key Dependencies | Integration Points |
|----------------|----------------------|------------------|-------------------|
| **API Layer** | Public interface consolidation and backward compatibility | Config, Discovery, I/O modules | Kedro pipelines, Jupyter notebooks |
| **Configuration Layer** | YAML parsing, Pydantic validation, config-aware discovery | PyYAML, Pydantic v2 | All system components |
| **Discovery Engine** | Pattern-based file finding with metadata extraction | pathlib, regex, fnmatch | Config layer, API layer |
| **Registry System** | <span style="background-color: rgba(91, 57, 243, 0.2)">Enforced loader instantiation with dynamic plugin registration and O(1) lookup</span> | importlib.metadata, threading | <span style="background-color: rgba(91, 57, 243, 0.2)">All I/O operations, Schema providers</span> |

### 5.1.3 Data Flow Description

The FlyRigLoader system implements a progressive data flow that separates concerns and enables selective processing:

**1. Configuration Phase:**
- YAML files or Pydantic models provide experiment/dataset definitions with security validation
- Environment variables provide override capability for deployment flexibility
- LegacyConfigAdapter maintains backward compatibility while enabling modern patterns

**<span style="background-color: rgba(91, 57, 243, 0.2)">2. Version Detection & Migration Phase</span>:**
- <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic detection of configuration schema versions through embedded version fields</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Execution of configuration migrations with comprehensive compatibility matrix validation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Generation of migration reports and deprecation warnings for research audit trails</span>

**3. Discovery Phase (updated):**
- ConfigDiscoveryEngine applies ignore patterns and mandatory substrings for intelligent file location
- FileDiscoverer uses glob/rglob for pattern matching with performance targets of <5 seconds for 10,000 files
- PatternMatcher extracts metadata via regex with multi-format date parsing support

**4. Loading Phase (updated):**
- <span style="background-color: rgba(91, 57, 243, 0.2)">LoaderRegistry performs enforced O(1) extension-based lookup with mandatory registry-based instantiation</span>
- PickleLoader handles compressed/uncompressed formats with streaming support for large files
- Protocol-based design allows custom loader injection for new formats

**5. Transformation Phase (updated):**
- DataFrameTransformer applies column configurations with vectorized operations
- SchemaRegistry manages validation rules with type inference caching
- Specialized handlers normalize data dimensions and integrate experimental context

### 5.1.4 External Integration Points

| System Name | Integration Type | Data Exchange Pattern | Protocol/Format |
|-------------|-----------------|---------------------|-----------------|
| **File System** | Direct I/O | Synchronous read operations | POSIX paths, glob patterns |
| **Kedro Pipelines** | <span style="background-color: rgba(91, 57, 243, 0.2)">Catalog + Pipelines</span> | Function calls with manifest/DataFrame returns | <span style="background-color: rgba(91, 57, 243, 0.2)">FlyRigLoaderDataSet</span> |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Data Catalog</span>** | <span style="background-color: rgba(91, 57, 243, 0.2)">YAML Configuration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Declarative dataset registration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">catalog.yml syntax</span> |
| **Jupyter Notebooks** | Library import | Interactive function calls | Python API |
| **HPC Environments** | Batch processing | File-based data exchange | Pickle/CSV files |

## 5.2 COMPONENT DETAILS

### 5.2.1 API Layer (flyrigloader.api) (updated)

**Purpose and Responsibilities:**
- Consolidates discovery, loading, and transformation functions into a unified interface
- Maintains backward compatibility while leveraging new decoupled architecture
- Implements comprehensive dependency injection for testability
- Provides both legacy monolithic and new decoupled workflows
- <span style="background-color: rgba(91, 57, 243, 0.2)">Exposes pure functional API surface with enhanced testability through stateless design</span>

**Technologies and Frameworks:**
- Pure Python with type annotations for strong typing
- Protocol definitions for dependency injection
- functools for deprecation decorators
- <span style="background-color: rgba(91, 57, 243, 0.2)">Feature flag system using environment variable `FLYLOADER_V1` for controlled rollout</span>

**Key Interfaces:**
- `discover_experiment_manifest()`: File discovery without data loading
- `load_data_file()`: Selective raw data loading with registry lookup
- `transform_to_dataframe()`: Optional DataFrame transformation with schema validation
- <span style="background-color: rgba(91, 57, 243, 0.2)">Legacy functions maintained with deprecation warnings as required by semantic versioning rules</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**New Pure Functions** (gated behind `FLYLOADER_V1` feature flag):</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`validate_manifest()`: Manifest validation without side effects for comprehensive testing</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`create_kedro_dataset()`: Factory function for Kedro dataset creation with proper lifecycle management</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`get_registered_loaders()`: Registry introspection for available loader capabilities</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`get_loader_capabilities()`: Loader capability metadata for format detection and validation</span>

**Data Persistence Requirements:**
- No direct persistence (delegates to I/O layer)
- In-memory operation with lazy evaluation support

**Scaling Considerations:**
- Stateless design enables parallel processing
- Memory usage scales with individual file size, not dataset size
- Target API response time <100ms for manifest operations
- <span style="background-color: rgba(91, 57, 243, 0.2)">Feature flag implementation adds <5ms overhead for compatibility checks</span>

```mermaid
graph LR
    subgraph "API Layer Architecture"
        A[Public API Functions] --> B[Dependency Provider]
        B --> C[Config Provider]
        B --> D[Discovery Provider]
        B --> E[IO Provider]
        B --> F[Utils Provider]
        
        G[Legacy Functions] --> A
        H[New Decoupled Functions] --> A
        I[Feature Flag Gate] --> H
        J[Pure Functions V1] --> I
    end
    
    K[Client Code] --> A
    L[Test Harness] --> B
    M[FLYLOADER_V1] --> I
```

### 5.2.2 Configuration Layer (updated)

**Purpose and Responsibilities:**
- Loads and validates YAML configurations with Pydantic v2 models
- Implements security validators for path traversal protection
- Provides config-aware file discovery with pattern application
- Maintains legacy dictionary compatibility via adapters
- <span style="background-color: rgba(91, 57, 243, 0.2)">Implements comprehensive builder pattern with factory helpers for type-safe configuration construction</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Manages configuration schema versioning with automatic migration capabilities</span>

**Technologies and Frameworks:**
- PyYAML for safe YAML parsing with yaml.safe_load exclusively
- Pydantic v2 for model validation with performance optimizations
- Python protocols for extensibility
- <span style="background-color: rgba(91, 57, 243, 0.2)">semantic_version v2.10.0+ for schema version parsing, comparison, and compatibility validation</span>

**Key Interfaces:**
- `ProjectConfig`, `DatasetConfig`, `ExperimentConfig`: <span style="background-color: rgba(91, 57, 243, 0.2)">Pydantic models with embedded `schema_version` field for automatic version detection</span>
- `ConfigDiscoveryEngine`: Pattern-based discovery with config integration
- `LegacyConfigAdapter`: MutableMapping for backward compatibility
- <span style="background-color: rgba(91, 57, 243, 0.2)">**New Builder Pattern Interface**:</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`create_config()`: Comprehensive builder and factory helpers for type-safe configuration construction</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`load_config()`: Enhanced configuration loading with automatic migration when version mismatches are detected</span>

**Data Persistence Requirements:**
- Reads from YAML files on disk with configuration caching
- Caches parsed configurations in memory for performance
- <span style="background-color: rgba(91, 57, 243, 0.2)">Stores migration reports and version history for audit trails</span>

**Scaling Considerations:**
- Configuration size typically small (<1MB)
- Validation performance O(n) with config complexity
- Security validation prevents path traversal attacks
- <span style="background-color: rgba(91, 57, 243, 0.2)">Schema migration execution time scales linearly with configuration complexity</span>

```mermaid
stateDiagram-v2
    [*] --> YAMLFile
    YAMLFile --> VersionDetection
    VersionDetection --> SecurityValidation
    SecurityValidation --> PydanticParsing
    PydanticParsing --> ConfigModel
    ConfigModel --> LegacyAdapter
    LegacyAdapter --> ConfigAPI
    
    VersionDetection --> AutoMigration: Version Mismatch
    AutoMigration --> PydanticParsing: Migrated Config
    SecurityValidation --> SecurityError: Invalid Path
    PydanticParsing --> ValidationError: Schema Violation
    AutoMigration --> MigrationError: Migration Failed
```

### 5.2.3 Discovery Engine (updated)

**Purpose and Responsibilities:**
- Implements pattern-based file discovery with metadata extraction
- Provides lazy file statistics computation for performance
- Supports regex-based metadata extraction from filenames
- Handles multi-format date parsing for experimental workflows
- <span style="background-color: rgba(91, 57, 243, 0.2)">Provides Kedro-specific metadata extraction for catalog-aware workflows</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Supports version-aware discovery patterns driven by Configuration Layer schema versions</span>

**Technologies and Frameworks:**
- pathlib for cross-platform path operations
- regex for pattern matching with caching
- fnmatch for shell-style wildcards
- <span style="background-color: rgba(91, 57, 243, 0.2)">Integration with semantic_version for version-aware pattern matching</span>

**Key Interfaces:**
- `FileDiscoverer`: Core discovery with injectable providers
- `PatternMatcher`: Regex-based metadata extraction with date parsing
- `FileManifest`: Container with filtering utilities and lazy statistics
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Enhanced Discovery Features**:</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro catalog-aware discovery filters for pipeline integration</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Version-aware pattern matching based on configuration schema versions</span>

**Data Persistence Requirements:**
- No persistence, operates on file system metadata
- Lazy evaluation for large directory structures

**Scaling Considerations:**
- Performance target: <5 seconds for 10,000 files
- Memory usage proportional to file count with optimization strategies
- Batch stat calls and parallel discovery planned
- <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro metadata extraction adds <10% overhead to discovery operations</span>

```mermaid
sequenceDiagram
    participant Client
    participant Discovery
    participant FileSystem
    participant PatternMatcher
    participant Stats
    participant KedroFilter
    
    Client->>Discovery: discover_files(pattern)
    Discovery->>FileSystem: glob/rglob
    FileSystem-->>Discovery: file paths
    Discovery->>KedroFilter: apply_catalog_filters
    KedroFilter-->>Discovery: filtered paths
    Discovery->>PatternMatcher: extract_metadata
    PatternMatcher-->>Discovery: metadata dict
    Discovery->>Stats: get_file_stats
    Stats-->>Discovery: size, mtime
    Discovery-->>Client: FileManifest
```

### 5.2.4 I/O Pipeline (updated)

**Purpose and Responsibilities:**
- <span style="background-color: rgba(91, 57, 243, 0.2)">Priority-based registry lookup by file extension with comprehensive loader capability introspection</span>
- Handles pickle, compressed pickle, and gzipped formats
- Transforms raw data to pandas DataFrames with column validation
- Manages column configuration and schema enforcement
- <span style="background-color: rgba(91, 57, 243, 0.2)">Enforces registry-based loader instantiation with complete elimination of direct instantiation patterns</span>

**Technologies and Frameworks:**
- pickle, gzip for data serialization with compression detection
- pandas, numpy for data structures and vectorized operations
- Pydantic for schema validation with type inference caching

**Key Interfaces:**
- `LoaderRegistry`: Thread-safe singleton for O(1) loader lookup <span style="background-color: rgba(91, 57, 243, 0.2)">with priority-based resolution and capability introspection</span>
- `PickleLoader`: Handles multiple pickle formats with streaming support
- `DataFrameTransformer`: Configurable transformation pipeline
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Enhanced Registry Features**:</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Loader capability metadata for format detection and validation</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">Priority enumeration system (BUILTIN < USER < PLUGIN < OVERRIDE)</span>

**Data Persistence Requirements:**
- Reads from pickle files on disk with memory mapping planned
- No write operations in current implementation

**Scaling Considerations:**
- Memory target: <2x data size overhead
- Performance target: <1 second per 100MB
- Streaming support for large files in development
- <span style="background-color: rgba(91, 57, 243, 0.2)">Registry lookup performance maintained at O(1) with capability introspection adding <1ms overhead</span>

### 5.2.5 Registry System (updated)

**Purpose and Responsibilities:**
- <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safe singleton registry for plugins with atomic operations using `threading.RLock` for enhanced concurrency control</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic entry-point discovery via importlib.metadata with comprehensive plugin lifecycle management</span>
- Supports runtime registration and unregistration
- <span style="background-color: rgba(91, 57, 243, 0.2)">Provides decorator-based registration helpers including new `@auto_register` decorator</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Implements priority-based resolution system with enumeration: BUILTIN < USER < PLUGIN < OVERRIDE</span>

**Technologies and Frameworks:**
- importlib.metadata for plugin discovery
- <span style="background-color: rgba(91, 57, 243, 0.2)">threading.RLock for enhanced concurrency control with deadlock prevention</span>
- Python protocols for type safety

**Key Interfaces:**
- `BaseRegistry`: Abstract registry with lock management
- `LoaderRegistry`, `SchemaRegistry`: Concrete implementations
- `@loader_for()`, `@schema_for()`: Registration decorators
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Enhanced Registration System**:</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`@auto_register` decorator for automatic entry-point discovery</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`RegistryError` exception class for plugin conflict resolution and error handling</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">INFO-level logging of all registration events for audit trails and debugging</span>

**Data Persistence Requirements:**
- In-memory registry, rebuilt on startup
- Plugin state persisted through application lifecycle
- <span style="background-color: rgba(91, 57, 243, 0.2)">Registration event logging for operational observability</span>

**Scaling Considerations:**
- O(1) lookup performance critical for high-throughput scenarios
- Thread-safe for concurrent access in multi-threaded environments
- <span style="background-color: rgba(91, 57, 243, 0.2)">Priority resolution system adds <2ms overhead for complex plugin hierarchies</span>

```mermaid
graph TB
    subgraph "Registry Architecture"
        A[Entry Points Discovery] --> B[Auto Registration]
        B --> C[Priority Resolution]
        C --> D[Thread-Safe Registry]
        D --> E["O(1) Lookup"]
        
        F["@auto_register"] --> B
        G[Manual Registration] --> C
        H[Plugin Conflicts] --> I[RegistryError]
        
        J[BUILTIN Priority] --> C
        K[USER Priority] --> C
        L[PLUGIN Priority] --> C
        M[OVERRIDE Priority] --> C
    end
    
    N[Client Lookup] --> E
    O[INFO Logging] --> D
```

## 5.3 TECHNICAL DECISIONS

### 5.3.1 Architecture Style Decisions and Tradeoffs

**Layered Architecture Choice:**
- **Decision**: Organize system into distinct horizontal layers (API, Config, Discovery, I/O, Utils)
- **Rationale**: Clear separation of concerns, independent testing capabilities, phased migration from legacy systems
- **Tradeoffs**: Some performance overhead from layer boundaries vs monolithic design
- **Benefits**: Testability, maintainability, selective component replacement

**Protocol-Based Dependency Injection:**
- **Decision**: Use Python protocols for component interfaces instead of heavy DI frameworks
- **Rationale**: Enables comprehensive testing without complex dependencies, aligns with Python ecosystem
- **Tradeoffs**: Slightly more verbose than direct imports, requires discipline in interface design
- **Benefits**: Test isolation, component substitution, clear contracts, type safety

```mermaid
graph TD
    subgraph "Architecture Decision Record"
        A[Requirement: Testable Components] --> B{DI Framework Options}
        B --> C[Heavy Framework]
        B --> D[Python Protocols]
        B --> E[Direct Dependencies]
        
        C --> F[Rejected: Complexity]
        D --> G[Selected: Simple & Pythonic]
        E --> H[Rejected: Hard to Test]
        
        G --> I[Benefits: Easy Mocking]
        G --> J[Benefits: Type Safety]
        G --> K[Tradeoff: Verbosity]
    end
```

### 5.3.2 Communication Pattern Choices

**Synchronous Function Calls:**
- **Decision**: All communication via synchronous function calls
- **Rationale**: Simplicity, predictability, suitable for scientific workflows with sequential processing
- **Tradeoffs**: No async benefits for I/O operations, may limit concurrent processing
- **Future Consideration**: Async support planned for network-based loaders

**In-Memory Data Exchange:**
- **Decision**: Pass data structures directly between components
- **Rationale**: Performance optimization, simplicity, typical dataset sizes fit memory constraints
- **Tradeoffs**: Large dataset limitations, memory pressure on constrained systems
- **Mitigation**: Lazy evaluation patterns, streaming transformations, selective loading

### 5.3.3 Data Storage Solution Rationale

**File System Based Storage:**
- **Decision**: Use file system for all data persistence without database layer
- **Rationale**: Aligns with research workflows, no database overhead, simple deployment
- **Benefits**: Direct data access, version control friendly, institutional compliance
- **Limitations**: No query capabilities, manual organization required

**Pickle Format Support:**
- **Decision**: Primary support for pickle with compression variants (.pkl, .pklz, .pkl.gz)
- **Rationale**: Standard in scientific Python ecosystem, efficient for numpy arrays
- **Security**: Only load trusted data files, use restricted unpickling
- **Extensibility**: Registry pattern allows additional formats without core changes

### 5.3.4 Caching Strategy Justification

**Configuration Caching:**
- **Decision**: Cache parsed configurations in memory during execution
- **Rationale**: Expensive validation process, configurations rarely change during execution
- **Implementation**: Module-level variables with lazy loading and invalidation

**No Data Caching:**
- **Decision**: Do not cache loaded experimental data between operations
- **Rationale**: Large memory footprint, sequential access patterns in research workflows
- **Benefit**: Predictable memory usage, avoids cache invalidation complexity

### 5.3.5 Security Mechanism Selection

**Path Traversal Protection:**
- **Decision**: Strict validation at configuration load time with path resolution
- **Implementation**: Resolve paths and verify within allowed directories
- **Rationale**: Prevent access to system files via malicious YAML configurations

**Safe YAML Loading:**
- **Decision**: Use yaml.safe_load exclusively throughout codebase
- **Rationale**: Prevent arbitrary code execution via YAML deserialization
- **Enforcement**: Code review guidelines prohibit yaml.load usage

### 5.3.6 <span style="background-color: rgba(91, 57, 243, 0.2)">Version Management System Selection (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Schema Version Field Adoption</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Embed `schema_version` fields in all Pydantic configuration models with semantic versioning compliance</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Enable automatic migration detection and execution while maintaining research workflow continuity</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Tradeoffs</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Additional complexity in configuration files vs forward compatibility and zero-breaking-change upgrades</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Benefits</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Reproducible scientific computing, clear upgrade paths, automatic rollback capabilities</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Automatic Migration System</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Implement comprehensive migration engine with compatibility matrix validation using semantic_version v2.10.0+</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Dedicated migration layer with audit trail generation and deprecation warning logging</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Preserve existing research workflows while enabling system evolution and feature advancement</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Quality Assurance</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Migration reports for research audit trails, comprehensive testing of migration paths</span>

```mermaid
graph TD
    subgraph "Version Management Decision Tree"
        A[Configuration Load Request] --> B{Version Field Present?}
        B -->|No| C[Assume Legacy Format]
        B -->|Yes| D{Version Compatible?}
        
        C --> E[Legacy Adapter Path]
        D -->|Yes| F[Direct Processing]
        D -->|No| G{Auto-Migration Available?}
        
        G -->|Yes| H[Execute Migration]
        G -->|No| I[Compatibility Error]
        
        H --> J[Generate Migration Report]
        J --> K[Log Deprecation Warning]
        K --> F
        
        E --> L[Backward Compatibility Mode]
        L --> M[Deprecation Warning]
        
        style H fill:#5b39f3,color:#ffffff
        style J fill:#5b39f3,color:#ffffff
    end
```

### 5.3.7 <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration Architecture Modernization (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Builder Pattern Adoption</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Transition to comprehensive builder pattern with Pydantic-backed factory helpers for type-safe configuration construction</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Eliminate 90% of configuration-related errors through uniform validation and intuitive configuration creation workflows</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Benefits</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Type safety, comprehensive validation, self-documenting configuration interfaces, reduced cognitive load</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Tradeoffs</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Additional abstraction layer vs improved developer experience and error prevention</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Pydantic v2 Foundation</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Standardize all configuration handling on Pydantic v2.6+ models with performance optimizations</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Factory methods for configuration creation, transparent LegacyConfigAdapter for backward compatibility</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Enable progressive enhancement while maintaining existing workflow compatibility</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Performance Impact</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Optimized validation with minimal overhead, comprehensive error reporting for research workflows</span>

### 5.3.8 <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration Strategy (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Optional Dependency Model</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Deliver Kedro support as optional extra dependency via `flyrigloader[kedro]` installation pattern</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Balance enterprise pipeline capabilities with lightweight core installation for research environments</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Benefits</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Reduced installation footprint, simplified dependency management, opt-in complexity</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Conditional imports with graceful degradation when Kedro is unavailable</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">First-Class Integration Design</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Implement native FlyRigLoaderDataSet with full AbstractDataset compatibility and catalog.yml support</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Features</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Data versioning, lineage tracking, pipeline execution framework integration, mandatory metadata columns</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Performance Preservation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Core FlyRigLoader performance characteristics maintained with <10% Kedro integration overhead</span>

```mermaid
graph TD
    subgraph "Kedro Integration Decision Matrix"
        A[Integration Requirement] --> B{Dependency Strategy}
        B --> C[Core Dependency]
        B --> D[Optional Extra]
        B --> E[Separate Package]
        
        C --> F[Rejected: Heavy Core]
        D --> G[Selected: Balanced Approach]
        E --> H[Rejected: Fragmentation]
        
        G --> I[Benefits: Lightweight Core]
        G --> J[Benefits: Enterprise Features]
        G --> K[Tradeoff: Optional Complexity]
        
        style G fill:#5b39f3,color:#ffffff
        style I fill:#5b39f3,color:#ffffff
        style J fill:#5b39f3,color:#ffffff
    end
```

### 5.3.9 <span style="background-color: rgba(91, 57, 243, 0.2)">Registry System Enhancement (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Registry Encapsulation Enforcement</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Eliminate all direct loader instantiation patterns and enforce registry-based component access exclusively</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safe singleton registries with O(1) lookup performance and comprehensive error handling</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Transform registry system into true internal implementation detail with plugin-style extensibility</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Benefits</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Consistent component lifecycle, enhanced testability, plugin conflict resolution</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Entry-Point Plugin Discovery</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Implement automatic plugin discovery via importlib.metadata with priority-based resolution system</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Priority Enumeration</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">BUILTIN < USER < PLUGIN < OVERRIDE with comprehensive conflict resolution</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">@auto_register decorator, INFO-level registration logging, RegistryError exception handling</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Performance Impact</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)"><2ms overhead for complex plugin hierarchies, maintained O(1) lookup performance</span>

### 5.3.10 <span style="background-color: rgba(91, 57, 243, 0.2)">Feature Flag Implementation Strategy (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Incremental Rollout Architecture</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Gate all refactored code paths behind `FLYLOADER_V1` feature flag to enable controlled adoption without disrupting ongoing research</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rationale</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Mandatory requirement for non-disruptive refactoring while coordinating with dependent systems like fly-filt</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Implementation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Environment variable control with graceful fallback to legacy pathways and comprehensive compatibility testing</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Performance Impact</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)"><5ms overhead for feature flag evaluation with early exit optimization</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Parallel Implementation Pattern</span>:**
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Decision</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Maintain legacy function signatures with deprecation warnings while exposing new pure functional API surface</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Benefits</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Backward compatibility preservation, enhanced testability through stateless design, selective feature adoption</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Strategy</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Clear deprecation periods following semantic versioning rules with comprehensive migration documentation</span>

```mermaid
graph TD
    subgraph "Feature Flag Decision Architecture"
        A[Code Path Request] --> B{FLYLOADER_V1 Set?}
        B -->|No| C[Legacy Implementation]
        B -->|Yes| D[New V1 Implementation]
        
        C --> E[Existing Workflow]
        C --> F[Deprecation Warning]
        
        D --> G[Pure Function API]
        D --> H[Enhanced Testability]
        D --> I[Registry Enforcement]
        
        F --> J[Migration Guide]
        E --> K[Backward Compatibility]
        
        G --> L[Feature Flag Overhead: <5ms]
        H --> M[Stateless Design]
        I --> N[Plugin Architecture]
        
        style D fill:#5b39f3,color:#ffffff
        style G fill:#5b39f3,color:#ffffff
        style H fill:#5b39f3,color:#ffffff
        style I fill:#5b39f3,color:#ffffff
    end
```

## 5.4 CROSS-CUTTING CONCERNS

### 5.4.1 Monitoring and Observability Approach

**Structured Logging via Loguru:**
- Console output at INFO level for user visibility with colorized formatting
- File output at DEBUG level with automatic rotation (10MB files) and compression
- Contextual information preserved in all log entries for debugging
- Test mode support with configurable verbosity levels
- <span style="background-color: rgba(91, 57, 243, 0.2)">Registry registration events structured logging with `REGISTRY` tag for plugin lifecycle tracking</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration migration events structured logging with `MIGRATION` tag for version compatibility monitoring</span>

**Performance Metrics Collection:**
- Discovery performance tracked via timing logs with threshold alerts
- Memory usage monitored during transformations with automatic cleanup
- Error rates captured in structured logs for trend analysis
- Success rates tracked for reliability monitoring
- <span style="background-color: rgba(91, 57, 243, 0.2)">`plugins_registered` counter for registry system health monitoring</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`migration_success` counter for configuration migration reliability tracking</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`migration_failure` counter for migration error pattern analysis</span>

```mermaid
flowchart LR
    subgraph "Loguru Configuration"
        A[Logger Init] --> B[Console Sink<br/>INFO Level]
        A --> C[File Sink<br/>DEBUG Level]
        C --> D[Rotation<br/>10MB]
        D --> E[Compression<br/>gzip]
    end
    
    subgraph "Enhanced Structured Logging"
        F[API Calls] --> G[Structured Logs]
        H[Errors] --> G
        I[Performance] --> G
        J[Registry Events] --> K[REGISTRY Tag]
        L[Migration Events] --> M[MIGRATION Tag]
        K --> G
        M --> G
    end
    
    G --> B
    G --> C
```

### 5.4.2 Logging and Tracing Strategy

**Hierarchical Logging Architecture:**
- Entry/exit logging for public API functions with execution timing
- Detailed debug logs for internal operations with context preservation
- Error context preservation with complete stack traces
- Consistent prefixes for log filtering and analysis
- <span style="background-color: rgba(91, 57, 243, 0.2)">Registry operation logging with plugin discovery and registration lifecycle events</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Migration operation logging with version detection, compatibility checks, and transformation results</span>

**Audit Trail Implementation:**
- Configuration resolution logged with precedence information
- File discovery results with counts, timing, and filter applications
- Data transformation steps with shape changes and validation results
- Performance metrics with threshold comparisons
- <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin registration audit with entry point discovery and conflict resolution</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration migration audit with version transitions and backup creation</span>

### 5.4.3 Error Handling Patterns

**Domain-Specific Exception Hierarchy (updated):**
- `FlyRigLoaderError`: Base exception with context preservation capabilities
- `ConfigError`: Configuration and validation failures with path information
- `DiscoveryError`: File system and pattern matching errors with pattern details
- `LoadError`: Data loading and format errors with file information
- `TransformError`: Schema and transformation errors with data shape context
- <span style="background-color: rgba(91, 57, 243, 0.2)">`RegistryError`: Plugin registration conflicts, entry point discovery failures, and loader initialization errors</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`VersionError`: Schema version incompatibility, migration failures, and compatibility matrix validation errors</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`KedroIntegrationError`: Kedro catalog configuration issues, dataset initialization failures, and pipeline execution errors</span>

**Error Context Preservation:**
- Original exception chaining maintained for debugging
- Contextual information (paths, patterns, data shapes) attached to exceptions
- Error codes for programmatic handling by client applications
- Structured logging before re-raising for analysis
- <span style="background-color: rgba(91, 57, 243, 0.2)">Registry state capture for plugin conflict resolution</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Version context preservation for migration troubleshooting</span>

```mermaid
flowchart TD
    subgraph "Error Handling Flow"
        A[Operation] --> B{Error?}
        B -->|Yes| C[Capture Context]
        C --> D[Create Domain Exception]
        D --> E[Log with Context]
        E --> F[Re-raise with Chain]
        
        B -->|No| G[Success]
        
        F --> H[Caller Handles]
        H --> I[Retry Logic?]
        I -->|Yes| A
        I -->|No| J[User Error Message]
        
        subgraph "Enhanced Error Types"
            K[RegistryError]
            L[VersionError]
            M[KedroIntegrationError]
        end
        
        D --> K
        D --> L
        D --> M
    end
```

### 5.4.4 Authentication and Authorization Framework

**Current Implementation:**
- No authentication required (research tool with file system access)
- File system permissions provide access control at OS level
- Configuration validation restricts accessible directories

**Future Considerations:**
- Integration with institutional authentication systems planned
- Role-based access control for shared research deployments
- Audit logging capabilities for compliance requirements
- <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin-based authentication providers for Kedro integration scenarios</span>

### 5.4.5 Performance Requirements and SLAs (updated)

| Operation | Target | Current | Optimization Strategy |
|-----------|--------|---------|----------------------|
| **Discovery (10k files)** | <5 seconds | 3.2 seconds | Parallel directory scanning |
| **Load (100MB)** | <1 second | 0.8 seconds | Memory mapping for large files |
| **Transform (1M rows)** | <2 seconds | 1.5 seconds | Vectorized operations |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Registry Lookup</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">O(1) complexity</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">O(1) maintained</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Post entry-point discovery</span>** |
| **Memory Overhead** | <2x data size | 1.8x | Streaming transformation |

**<span style="background-color: rgba(91, 57, 243, 0.2)">Performance Guarantee (updated):</span>**
<span style="background-color: rgba(91, 57, 243, 0.2)">All enhancements including registry system, version management, and Kedro integration maintain existing performance benchmarks. Registry lookup operations remain O(1) complexity after initial entry-point discovery phase. Core discovery performance target of <5 seconds for 10,000 files remains unchanged and validated through continuous benchmarking.</span>

### 5.4.6 Disaster Recovery Procedures

**Failure Modes and Recovery:**

1. **Configuration Corruption:**
   - Detection: Pydantic validation fails with detailed error messages
   - Recovery: Restore from version control or backup configurations
   - Prevention: Schema migration testing and validation
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced: Automatic configuration backup before migration attempts</span>

2. **File System Unavailability:**
   - Detection: Discovery/Load operations fail with file system errors
   - Recovery: Retry with exponential backoff and alternative paths
   - Mitigation: Clear error messages with suggested actions

3. **Memory Exhaustion:**
   - Detection: Memory monitoring during transformations with thresholds
   - Recovery: Selective processing with smaller batches
   - Prevention: Lazy evaluation patterns and streaming processing

4. **Plugin Loading Failure (updated):**
   - Detection: Entry point discovery errors during registry initialization
   - Recovery: Fall back to built-in loaders with warning messages
   - Logging: Detailed plugin information for debugging
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced: Plugin conflict resolution through priority-based selection</span>

5. **<span style="background-color: rgba(91, 57, 243, 0.2)">Version Migration Failure:</span>**
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Detection: Schema version compatibility validation fails during configuration loading</span>
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Recovery: Automatic rollback to backed-up configuration with detailed migration logs</span>
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Prevention: Pre-migration validation and compatibility matrix checks</span>

6. **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration Failure:</span>**
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Detection: Kedro catalog configuration errors or dataset initialization failures</span>
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Recovery: Graceful degradation to standalone mode with comprehensive error context</span>
   - <span style="background-color: rgba(91, 57, 243, 0.2)">Mitigation: Kedro dependency validation and version compatibility checks</span>

**Recovery Procedures:**
- All errors logged with comprehensive context for root cause analysis
- Graceful degradation for non-critical failures maintaining core functionality
- Clear user messages with suggested actions and contact information
- No data loss guarantee (read-only operations with transaction boundaries)
- <span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced registry state recovery with plugin re-registration capabilities</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Automated configuration migration rollback with audit trail preservation</span>

#### References

**Technical Specification Sections Retrieved:**
- `1.2 SYSTEM OVERVIEW` - High-level architecture, success criteria, and registry system components
- `3.2 FRAMEWORKS & LIBRARIES` - Loguru configuration, Kedro integration, and version management libraries
- `4.1 SYSTEM WORKFLOWS` - Registry-based loading pipeline and configuration migration workflows
- `4.2 ERROR HANDLING FLOWS` - Domain-specific error hierarchy and registry/version error handling patterns

**Files Examined:**
- `src/flyrigloader/api.py` - Main public API interface with decoupled architecture
- `src/flyrigloader/exceptions.py` - Domain-specific error hierarchy and context preservation

**Directories Analyzed:**
- `src/flyrigloader/config/` - Configuration and validation components
- `src/flyrigloader/discovery/` - File discovery engine and pattern matching
- `src/flyrigloader/io/` - I/O pipeline and transformation components
- `src/flyrigloader/registries/` - Plugin registry infrastructure
- `src/flyrigloader/utils/` - Cross-cutting utilities and providers

# 6. SYSTEM COMPONENTS DESIGN

## 6.1 CORE SERVICES ARCHITECTURE

### 6.1.1 Architectural Classification

**Core Services Architecture is not applicable for this system.**

FlyRigLoader is fundamentally a **monolithic Python library** with a layered architecture, not a service-oriented or microservices-based system. The system does not implement distributed services, inter-service communication patterns, or service boundaries that would warrant a Core Services Architecture approach.

### 6.1.2 Rationale for Non-Applicability

#### 6.1.2.1 System Architecture Evidence

The technical specification explicitly defines FlyRigLoader as implementing a **"layered, protocol-based architecture"** rather than a distributed services architecture. Key evidence includes:

| Architecture Aspect | FlyRigLoader Implementation | Services Architecture Alternative |
|---------------------|---------------------------|-----------------------------------|
| **Deployment Model** | Single Python package installation | Multiple deployable services |
| **Communication Pattern** | Direct function calls within process | Network protocols (HTTP/gRPC/messaging) |
| **Service Boundaries** | Module-level separation | Process/container boundaries |
| **Scalability** | Vertical scaling within application | Horizontal scaling across services |

#### 6.1.2.2 Technical Implementation Pattern

The system architecture demonstrates clear monolithic characteristics:

- **Single Process Execution**: All components execute within the same Python interpreter process
- **Shared Memory Model**: Data flows through direct object references, not serialized network communication
- **Library Integration**: External systems integrate via `import flyrigloader` statements, not API calls
- **File System Dependencies**: Direct file system access without network protocols

#### 6.1.2.3 Integration Patterns

FlyRigLoader integrates with external systems through library-level patterns:

```mermaid
graph TB
    subgraph "External Applications"
        A[Kedro Pipelines]
        B[Jupyter Notebooks]
        C[HPC Environments]
        D[CLI Tools]
    end
    
    subgraph "FlyRigLoader Library"
        E[API Layer]
        F[Configuration Layer]
        G[Discovery Engine]
        H[I/O Pipeline]
        I[Transformation Engine]
    end
    
    subgraph "Data Sources"
        J[Local File System]
        K[Network Storage]
        L[YAML Configs]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    
    E --> F
    F --> G
    G --> H
    H --> I
    
    I --> J
    F --> L
    G --> K
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#f3e5f5
```

### 6.1.3 Alternative Architecture Pattern

#### 6.1.3.1 Layered Architecture Implementation

FlyRigLoader implements a **layered architecture** with clear separation of concerns:

| Layer | Responsibility | Key Components |
|-------|---------------|----------------|
| **API Layer** | Public interface consolidation | `flyrigloader.api` module |
| **Configuration Layer** | YAML parsing and validation | `config/yaml_config.py`, `config/discovery.py` |
| **Discovery Layer** | Pattern-based file finding | `discovery/files.py`, `discovery/patterns.py` |
| **I/O Layer** | Data loading and transformation | `io/pickle.py`, `io/column_models.py` |
| **Utilities Layer** | Cross-cutting concerns | `utils/` directory modules |

#### 6.1.3.2 Protocol-Based Dependency Injection

The system employs **Protocol-based design** for component interaction:

```mermaid
graph LR
    subgraph "Protocol Interfaces"
        A[LoaderProtocol]
        B[DiscoveryProtocol]
        C[ConfigProtocol]
        D[TransformProtocol]
    end
    
    subgraph "Concrete Implementations"
        E[PickleLoader]
        F[FileDiscoverer]
        G[YamlConfig]
        H[DataFrameTransformer]
    end
    
    A -.-> E
    B -.-> F
    C -.-> G
    D -.-> H
    
    style A fill:#fff3e0
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#e8f5e8
    style H fill:#e8f5e8
```

#### 6.1.3.3 Plugin Registry System

The system implements a **registry-based pattern** for extensibility without requiring service architecture:

| Registry Type | Purpose | Discovery Method |
|---------------|---------|------------------|
| **Loader Registry** | File format handlers | Entry points and extension mapping |
| **Schema Registry** | Column validation rules | Pydantic model registration |
| **Pattern Registry** | Metadata extraction patterns | Regex pattern compilation |

### 6.1.4 Service Architecture Considerations

#### 6.1.4.1 Why Services Architecture Was Not Chosen

The decision to implement a monolithic library rather than a services architecture was driven by:

1. **Target Environment**: Academic research laboratories with on-premises infrastructure
2. **Performance Requirements**: In-memory data processing without network latency
3. **Deployment Simplicity**: Single `pip install` for research environments
4. **Integration Pattern**: Embedded within existing Python scientific workflows

#### 6.1.4.2 Future Considerations

While Core Services Architecture is not applicable to the current system, potential future evolution paths could include:

| Evolution Scenario | Architectural Implications | Implementation Approach |
|-------------------|---------------------------|-------------------------|
| **Web API Layer** | HTTP service wrapper | Flask/FastAPI adapter maintaining library core |
| **Distributed Processing** | Task queue integration | Celery or similar with library as worker |
| **Multi-tenant Support** | Containerized deployments | Docker wrapper with shared library |

### 6.1.5 Validation and Monitoring

#### 6.1.5.1 Library-Level Observability

The system implements observability patterns appropriate for a library architecture:

- **Structured Logging**: Loguru-based logging with configurable levels
- **Error Hierarchy**: Custom exceptions for specific failure modes
- **Performance Metrics**: Memory usage and processing time tracking
- **Validation Gates**: Pydantic model validation at system boundaries

#### 6.1.5.2 Integration Monitoring

Host applications monitor FlyRigLoader through:

```mermaid
graph TB
    subgraph "Host Application Monitoring"
        A[Application Metrics]
        B[Error Tracking]
        C[Performance Profiling]
        D[Resource Monitoring]
    end
    
    subgraph "FlyRigLoader Library"
        E[Structured Logging]
        F[Exception Hierarchy]
        G[Memory Tracking]
        H[Validation Results]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#e3f2fd
    style E fill:#f1f8e9
    style F fill:#f1f8e9
    style G fill:#f1f8e9
    style H fill:#f1f8e9
```

### 6.1.6 Conclusion

FlyRigLoader's architecture is intentionally designed as a **monolithic Python library** to serve its specific domain of scientific data management. The layered, protocol-based design provides the benefits of modularity and testability without the complexity and overhead of distributed services architecture.

The system achieves its goals of performance, reliability, and extensibility through proven library design patterns rather than service-oriented approaches, making it well-suited for its target environment of academic research workflows.

#### References
- Technical Specification Section 5.1 HIGH-LEVEL ARCHITECTURE - System architecture definition and rationale
- Technical Specification Section 1.2 SYSTEM OVERVIEW - Project context and architectural decisions
- Technical Specification Section 3.4 THIRD-PARTY SERVICES - Standalone library confirmation
- Repository structure analysis - `src/flyrigloader/` package organization
- Integration patterns - Kedro, Jupyter, and HPC environment integration

## 6.2 DATABASE DESIGN

### 6.2.1 Database Design Applicability Assessment

**Database Design is not applicable to this system.**

FlyRigLoader is architected as a **file-based data processing library** that intentionally avoids database components in favor of direct file system interaction. This design decision aligns with the scientific research domain where experimental data is traditionally stored in files rather than databases, and where simplicity, portability, and direct file access are prioritized over database-provided features.

#### 6.2.1.1 Evidence of Non-Database Architecture

The comprehensive technical analysis reveals consistent evidence across all system components:

| Component | Storage Mechanism | Evidence |
|-----------|------------------|----------|
| **Configuration Management** | YAML files | `config/yaml_config.py` handles YAML parsing exclusively |
| **Data Storage** | Pickle files (.pkl, .pklz, .pkl.gz) | `io/pickle.py` implements file-based data loading |
| **State Management** | In-memory processing | No persistent state beyond process execution |
| **Metadata Handling** | File system attributes | Directory structure and filename patterns |

#### 6.2.1.2 Technical Architecture Pattern

The system implements a **layered, file-centric architecture** with the following characteristics:

```mermaid
graph TB
    subgraph "File System Storage"
        A[YAML Configuration Files]
        B[Pickle Data Files]
        C[Directory Structure]
        D[File Metadata]
    end
    
    subgraph "In-Memory Processing"
        E[Configuration Cache]
        F[Data Transformation]
        G[Schema Validation]
        H[Result Generation]
    end
    
    subgraph "External Integration"
        I[Kedro Pipelines]
        J[Jupyter Notebooks]
        K[HPC Environments]
        L[CLI Tools]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    
    E --> F
    F --> G
    G --> H
    
    H --> I
    H --> J
    H --> K
    H --> L
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
```

### 6.2.2 File-Based Storage Architecture

#### 6.2.2.1 Primary Storage Strategy

The system employs a **decoupled file system model** with three distinct layers:

| Layer | Purpose | Storage Format | Processing Model |
|-------|---------|---------------|------------------|
| **Discovery Layer** | File location and metadata extraction | Directory structures, glob patterns | Pattern matching without data loading |
| **Loading Layer** | Data retrieval and format handling | Pickle files (compressed/uncompressed) | On-demand loading with streaming |
| **Transformation Layer** | Data processing and output generation | In-memory DataFrames | Vectorized operations |

#### 6.2.2.2 Data Persistence Mechanisms

#### 6.2.2.1 Configuration Persistence (updated)

Configuration data is stored exclusively in YAML format, with <span style="background-color: rgba(91, 57, 243, 0.2)">every persisted configuration file now containing a mandatory `schema_version` field (default "1.0.0")</span>. <span style="background-color: rgba(91, 57, 243, 0.2)">Configurations created programmatically via `create_config()` are serialized to YAML with an embedded `schema_version`.</span>

**Configuration File Structure Example:**
```yaml
schema_version: "1.0.0"
project:
  name: "neuroscience_experiment"
  data_path: "/data/experiments"
datasets:
  - name: "control_group"
    pattern: "control_*.pkl"
  - name: "treatment_group" 
    pattern: "treatment_*.pkl"
```

**Configuration Processing Architecture:**

```mermaid
graph LR
    subgraph "Configuration Storage"
        A[experiment_config.yml]
        B[column_config.yml]
        C[dataset_config.yml]
        D[environment.yml]
    end
    
    subgraph "Configuration Processing"
        E[YamlConfig Loader]
        F[Pydantic Validation]
        G[Version Migrator]
        H[Schema Registry]
        I[Runtime Cache]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    
    E --> F
    F --> G
    G --> H
    H --> I
    
    style A fill:#fff3e0
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#5b39f3
    style H fill:#e8f5e8
    style I fill:#e8f5e8
```

The processing pipeline handles <span style="background-color: rgba(91, 57, 243, 0.2)">version detection and automatic migration when a version mismatch is detected through the new `Version Migrator` component</span>. The <span style="background-color: rgba(91, 57, 243, 0.2)">`YamlConfig Loader` returns Pydantic models by default, while legacy dictionary configurations are still accepted through `LegacyConfigAdapter` with deprecation warnings</span>.

#### 6.2.2.2 Data File Organization

Experimental data follows a hierarchical file system structure:

| Directory Level | Purpose | File Types | Metadata Extraction |
|----------------|---------|------------|---------------------|
| **Root Directory** | Project organization | Configuration files | Project metadata |
| **Experiment Level** | Experimental conditions | Data files, manifests | Date, experiment ID |
| **Dataset Level** | Specific datasets | Pickle files, CSVs | Dataset properties |
| **File Level** | Individual measurements | .pkl, .pklz, .pkl.gz | Timestamps, conditions |

#### 6.2.2.3 Memory Management Strategy

The system implements sophisticated memory management without persistent storage:

```mermaid
flowchart LR
    subgraph "Memory Optimization"
        A[Lazy Loading] --> B[Streaming Processing]
        B --> C[Batch Operations]
        C --> D[Garbage Collection]
        
        D --> E[Memory Monitoring]
        E --> F[Threshold Checking]
        F --> G[Cleanup Triggers]
        G --> H[Resource Release]
    end
    
    subgraph "Performance Targets"
        I[<2x Data Size Overhead] --> J[<1s per 100MB Load]
        J --> K[<5s for 10k Files Discovery]
        K --> L[<100ms Manifest Operations]
    end
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#e3f2fd
    style I fill:#f1f8e9
    style J fill:#f1f8e9
    style K fill:#f1f8e9
    style L fill:#f1f8e9
```

### 6.2.3 Rationale for File-Based Architecture

#### 6.2.3.1 Domain-Specific Requirements

The decision to avoid database design is driven by specific requirements of the neuroscience research domain:

| Requirement | File-Based Advantage | Database Alternative Impact |
|-------------|---------------------|----------------------------|
| **Data Volume** | Direct file streaming for large datasets | Database import/export overhead |
| **Experimental Workflows** | Maintains existing data collection patterns | Requires workflow modification |
| **Research Environment** | Zero-configuration deployment | Database installation and maintenance |
| **Data Portability** | Direct file access across systems | Database-specific export formats |

#### 6.2.3.2 Technical Benefits

#### 6.2.3.1 Performance Optimization

The file-based approach provides specific performance advantages:

- **Direct I/O**: Eliminates database query overhead for large scientific datasets
- **Streaming Capability**: Processes files larger than available memory
- **Parallel Processing**: Concurrent file operations without database locking
- **Memory Efficiency**: Lazy loading with <2x memory overhead target

#### 6.2.3.2 Operational Simplicity

The architecture minimizes operational complexity:

- **Zero Configuration**: No database setup or connection management
- **Portable Deployment**: Single `pip install` for research environments
- **Backup Strategy**: Standard file system backup procedures
- **Version Control**: Configuration files managed through Git

#### 6.2.3.3 Extensibility Considerations

While databases are not used, the architecture supports future database integration:

```mermaid
graph TB
    subgraph "Current Architecture"
        A[File System Interface]
        B[Registry System]
        C[Protocol Design]
        D[Plugin Architecture]
    end
    
    subgraph "Future Database Integration"
        E[Database Loaders]
        F[Connection Pooling]
        G[Query Optimization]
        H[Transaction Management]
    end
    
    B --> E
    C --> F
    D --> G
    A --> H
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#e1f5fe
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#fff3e0
```

### 6.2.4 Data Management Patterns

#### 6.2.4.1 Data Flow Architecture

The system implements a **progressive data flow** without persistent storage:

| Stage | Input Source | Processing | Output Format |
|-------|-------------|------------|---------------|
| **Discovery** | File system paths | Pattern matching | File manifests |
| **Loading** | Pickle files | Deserialization | Raw data dictionaries |
| **Transformation** | Raw data | Schema validation | Pandas DataFrames |
| **Integration** | DataFrames | Export formatting | External system input |

#### 6.2.4.2 Caching Strategy

The system implements **in-memory caching** without persistent cache storage:

```mermaid
graph LR
    subgraph "Cache Types"
        A[Configuration Cache]
        B[Pattern Cache]
        C[Schema Cache]
        D[Result Cache]
    end
    
    subgraph "Cache Management"
        E[Memory Monitoring]
        F[TTL Management]
        G[Eviction Policies]
        H[Cache Invalidation]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    
    E --> F
    F --> G
    G --> H
    
    style A fill:#e8f5e8
    style B fill:#e8f5e8
    style C fill:#e8f5e8
    style D fill:#e8f5e8
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
```

#### 6.2.4.3 Error Handling and Recovery

Without database transactions, the system implements **atomic file operations**:

| Operation | Failure Mode | Recovery Strategy | Consistency Guarantee |
|-----------|-------------|------------------|----------------------|
| **Configuration Load** | Parse/validation error | Reload from source | Fail-fast validation |
| **File Discovery** | Access permissions | Pattern adjustment | Graceful degradation |
| **Data Loading** | Corrupted files | Skip with logging | Partial dataset processing |
| **Transformation** | Schema violations | Error reporting | Individual file isolation |

### 6.2.5 Alternative Storage Considerations

#### 6.2.5.1 Database Integration Scenarios

While not currently implemented, the system architecture supports potential database integration:

| Scenario | Database Type | Integration Pattern | Implementation Approach |
|----------|-------------|-------------------|-------------------------|
| **Metadata Indexing** | SQLite | Local metadata cache | Registry-based plugin |
| **Multi-user Environments** | PostgreSQL | Shared experiment metadata | Configuration-driven connection |
| **Large-scale Deployment** | NoSQL | Distributed file manifests | Protocol-based abstraction |
| **Historical Analytics** | Time-series DB | Experiment tracking | Event-driven updates |

#### 6.2.5.2 Future Evolution Path

The system's protocol-based design enables database integration without architectural changes:

```mermaid
graph TB
    subgraph "Current Implementation"
        A[File System Storage]
        B[Protocol Interfaces]
        C[Registry System]
        D[Plugin Architecture]
    end
    
    subgraph "Database Integration Layer"
        E[Database Adapters]
        F[Connection Management]
        G[Query Optimization]
        H[Transaction Handling]
    end
    
    subgraph "Hybrid Architecture"
        I[File + Database Storage]
        J[Unified Query Interface]
        K[Distributed Caching]
        L[Seamless Migration]
    end
    
    B --> E
    C --> F
    D --> G
    A --> H
    
    E --> I
    F --> J
    G --> K
    H --> L
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#e1f5fe
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#fff3e0
    style I fill:#f1f8e9
    style J fill:#f1f8e9
    style K fill:#f1f8e9
    style L fill:#f1f8e9
```

### 6.2.6 Conclusion

The FlyRigLoader system deliberately implements a **file-based architecture** that eliminates database design requirements while maintaining high performance, operational simplicity, and extensibility. This architectural decision aligns with the scientific research domain's requirements for direct file access, minimal configuration, and portable deployment across research environments.

The system's protocol-based design and registry architecture ensure that future database integration remains possible without requiring fundamental architectural changes, positioning the system for evolution as requirements grow while maintaining its core advantages of simplicity and performance.

#### References

- Technical Specification Section 3.5 DATABASES & STORAGE - File system-based persistence strategy
- Technical Specification Section 5.1 HIGH-LEVEL ARCHITECTURE - Layered architecture overview
- Technical Specification Section 4.5 TECHNICAL IMPLEMENTATION DETAILS - Memory management and state handling
- Technical Specification Section 6.1 CORE SERVICES ARCHITECTURE - Monolithic library architecture confirmation
- `src/flyrigloader/config/yaml_config.py` - YAML configuration handling implementation
- `src/flyrigloader/io/pickle.py` - File-based data loading implementation
- `src/flyrigloader/discovery/` - Pattern-based file discovery implementation
- Repository structure analysis - Absence of database-related modules or dependencies

## 6.3 INTEGRATION ARCHITECTURE

### 6.3.1 LIBRARY-BASED INTEGRATION MODEL

#### 6.3.1.1 Integration Overview

FlyRigLoader integrates with the scientific Python ecosystem through import-based consumption rather than network-based APIs. The system is designed to function as a dependency within larger data analysis pipelines, particularly those using frameworks like Kedro for production-ready data science workflows.

```mermaid
graph TB
    subgraph "Consumer Applications"
        K[Kedro Pipelines] --> |import flyrigloader.api| API[FlyRigLoader API]
        J[Jupyter Notebooks] --> |import flyrigloader.api| API
        C[CLI Scripts] --> |import flyrigloader.api| API
        H[HPC Environments] --> |import flyrigloader.api| API
    end
    
    subgraph "FlyRigLoader Core"
        API --> CB[Config Builder<br/>create_config<br/>Returns validated<br/>Pydantic models]
        CB --> D[Discovery Engine]
        API --> LR[LoaderRegistry<br/>Thread-safe entry-point<br/>plugin discovery]
        API --> SR[SchemaRegistry<br/>Thread-safe entry-point<br/>plugin discovery]
        API --> T[Transformer Pipeline]
        LR --> R[Plugin Registry]
        SR --> R
    end
    
    subgraph "Data Sources"
        FS[File System] --> |file operations| D
        YAML[YAML Configs] --> |configuration| CB
        PKL[Pickle Files] --> |data loading via registry| LR
    end
    
    subgraph "Extension Points"
        P1[Custom Loaders] --> |register via entry points| R
        P2[Custom Schemas] --> |register via entry points| R
        P3[Custom Transformers] --> |register via protocols| R
    end
    
    style CB fill:#E6DAFC
    style LR fill:#E6DAFC
    style SR fill:#E6DAFC
```

<span style="background-color: rgba(91, 57, 243, 0.2)">All configuration objects now carry a mandatory schema_version field enabling automatic migration between configuration versions. This ensures backward compatibility and seamless upgrades for existing research workflows.</span>

#### 6.3.1.2 Integration Architecture Patterns (updated)

| Pattern | Implementation | Use Case | Benefits |
|---------|----------------|----------|----------|
| **Direct Import** | `from flyrigloader.api import load_experiment_files` | Primary integration method | Simple, synchronous, type-safe |
| **Parameter Injection** | Pass configuration as dictionary to functions | Kedro parameter-based workflows | Pipeline compatibility, configuration management |
| **Plugin Registration** | Entry point discovery and runtime registration | Extensibility and customization | Modular architecture, testability |
| **Dependency Injection** | Provider pattern for component swapping | Testing and mocking | Clean separation of concerns |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Builder Pattern</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">`create_config(**kwargs)`</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Programmatic config creation</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Typed, versioned defaults</span>** |

#### 6.3.1.3 Registry-Based Architecture Implementation (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The system implements a dual-registry architecture pattern that separates data loading concerns from schema validation concerns, providing enhanced modularity and thread-safe plugin discovery.</span>

#### LoaderRegistry Architecture

<span style="background-color: rgba(91, 57, 243, 0.2)">The LoaderRegistry serves as the centralized component for managing all data loading operations</span>:

- **Thread-Safe Operations**: <span style="background-color: rgba(91, 57, 243, 0.2)">Implements RWLock patterns ensuring safe concurrent access across multiple research workflows</span>
- **Entry-Point Discovery**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic plugin discovery through setuptools entry points with priority-based resolution (BUILTIN<USER<PLUGIN<OVERRIDE)</span>
- **Format Detection**: <span style="background-color: rgba(91, 57, 243, 0.2)">O(1) lookup performance for file format identification based on extension and content analysis</span>
- **Memory Efficiency**: <span style="background-color: rgba(91, 57, 243, 0.2)">Streaming loaders maintain <2x data size memory overhead for large experimental datasets</span>

#### SchemaRegistry Architecture

<span style="background-color: rgba(91, 57, 243, 0.2)">The SchemaRegistry manages all column validation and transformation schema definitions</span>:

- **Pydantic Integration**: <span style="background-color: rgba(91, 57, 243, 0.2)">Native Pydantic v2.6+ model registration for type-safe schema validation</span>
- **Dynamic Registration**: <span style="background-color: rgba(91, 57, 243, 0.2)">Runtime schema registration supporting custom validation rules and transformations specific to neuroscience data formats</span>
- **Version Compatibility**: <span style="background-color: rgba(91, 57, 243, 0.2)">Schema version tracking with automatic migration support ensuring backward compatibility</span>
- **Thread-Safe Access**: <span style="background-color: rgba(91, 57, 243, 0.2)">Concurrent schema lookup and validation operations with deadlock prevention mechanisms</span>

#### 6.3.1.4 Configuration Builder Integration (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The Config Builder (`create_config`) function serves as the primary interface for programmatic configuration creation, replacing direct YAML parsing for API-driven workflows.</span>

```mermaid
sequenceDiagram
    participant App as Consumer Application
    participant CB as Config Builder
    participant PV as Pydantic Validator
    participant SM as Schema Migration
    participant API as FlyRigLoader API
    
    App->>CB: create_config(**kwargs)
    CB->>PV: Validate input parameters
    PV->>SM: Check schema_version
    SM->>SM: Apply migrations if needed
    SM->>PV: Return migrated config
    PV->>CB: Validated Pydantic model
    CB->>API: Configuration object
    API->>App: Ready for processing
    
    rect rgba(91, 57, 243, 0.2)
        Note over CB,SM: All configurations include<br/>mandatory schema_version field
    end
```

**Builder Pattern Benefits**:

- **Type Safety**: <span style="background-color: rgba(91, 57, 243, 0.2)">Full Pydantic validation ensures configuration correctness at creation time</span>
- **Version Awareness**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic schema_version injection and validation prevents configuration mismatches</span>
- **Default Handling**: <span style="background-color: rgba(91, 57, 243, 0.2)">Intelligent defaults based on environment detection and research workflow patterns</span>
- **Migration Support**: <span style="background-color: rgba(91, 57, 243, 0.2)">Seamless handling of legacy configuration parameters through automatic migration</span>

#### 6.3.1.5 Plugin Discovery and Registration

The system implements a sophisticated plugin architecture supporting multiple registration mechanisms:

```mermaid
graph LR
    subgraph "Plugin Discovery Methods"
        A[Entry Points] --> |"setuptools discovery"| PR[Plugin Registry]
        B[Decorator Registration] --> |"@auto_register"| PR
        C[Manual Registration] --> |"register() calls"| PR
    end
    
    subgraph "Registry Types"
        PR --> LR[LoaderRegistry]
        PR --> SR[SchemaRegistry]
        PR --> TR[TransformRegistry]
    end
    
    subgraph "Priority Resolution"
        LR --> P1[BUILTIN Priority]
        LR --> P2[USER Priority]
        LR --> P3[PLUGIN Priority]
        LR --> P4[OVERRIDE Priority]
    end
    
    style LR fill:#5b39f333
    style SR fill:#5b39f333
```

**Plugin Integration Workflow**:

1. **Discovery Phase**: <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safe scanning of entry points and decorator-registered components</span>
2. **Validation Phase**: Protocol compliance checking and dependency resolution
3. **Registration Phase**: <span style="background-color: rgba(91, 57, 243, 0.2)">Priority-based insertion into appropriate registry with conflict resolution</span>
4. **Activation Phase**: <span style="background-color: rgba(91, 57, 243, 0.2)">Runtime availability for LoaderRegistry and SchemaRegistry operations</span>

#### 6.3.1.6 Kedro Pipeline Integration

The system provides first-class support for Kedro data pipeline integration through specialized DataSet implementations:

```mermaid
graph TB
    subgraph "Kedro Pipeline Context"
        KC[Kedro Catalog] --> FDS[FlyRigLoaderDataSet]
        KC --> FMS[FlyRigManifestDataSet]
        KP[Kedro Pipeline] --> KC
    end
    
    subgraph "FlyRigLoader Integration"
        FDS --> |delegates to| API[FlyRigLoader API]
        FMS --> |delegates to| API
        API --> CB[Config Builder]
        API --> LR[LoaderRegistry]
        API --> SR[SchemaRegistry]
    end
    
    subgraph "Pipeline Data Flow"
        LR --> |loads data| DF[DataFrame Output]
        DF --> |includes metadata| KM[Kedro Metadata Columns]
        KM --> |flows to| KP
    end
    
    style CB fill:#5B39F3
    style LR fill:#5B39F3
    style SR fill:#5B39F3
```

**Kedro Integration Features**:

- **AbstractDataset Compliance**: Full implementation of Kedro's dataset interface patterns
- **Catalog Configuration**: Native support for `catalog.yml` configuration with FlyRigLoader-specific parameters
- **Version Compatibility**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic schema_version handling ensuring pipeline reproducibility</span>
- **Metadata Integration**: All DataFrame outputs include mandatory Kedro metadata columns for lineage tracking

#### 6.3.1.7 Error Handling and Integration Reliability

The integration architecture implements comprehensive error handling specifically designed for research workflow reliability:

| Error Category | Integration Impact | Resolution Strategy | Recovery Mechanism |
|----------------|-------------------|--------------------|--------------------|
| **Configuration Errors** | <span style="background-color: rgba(91, 57, 243, 0.2)">Config Builder validation failures</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Detailed Pydantic error reporting with migration suggestions</span> | Automatic fallback to legacy configuration adapter |
| **Registry Errors** | <span style="background-color: rgba(91, 57, 243, 0.2)">LoaderRegistry or SchemaRegistry failures</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin isolation and graceful degradation</span> | Fallback to built-in components |
| **Discovery Errors** | File system access issues | Structured error context with actionable suggestions | Alternative discovery patterns |
| **Loading Errors** | <span style="background-color: rgba(91, 57, 243, 0.2)">Data format or registry mismatch</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Format-specific error handling through LoaderRegistry</span> | Progressive format detection fallback |

#### 6.3.1.8 Performance and Scalability Characteristics

The library-based integration model provides specific performance advantages for scientific computing workflows:

**Integration Performance Metrics**:

- **Import Time**: <100ms for initial module import and registry initialization
- **Configuration Building**: <span style="background-color: rgba(91, 57, 243, 0.2)"><50ms for create_config() with full Pydantic validation</span>
- **Registry Lookup**: <span style="background-color: rgba(91, 57, 243, 0.2)">O(1) performance for both LoaderRegistry and SchemaRegistry operations</span>
- **Plugin Discovery**: <span style="background-color: rgba(91, 57, 243, 0.2)"><200ms for complete entry-point scanning and registration</span>

**Scalability Considerations**:

- **Concurrent Access**: <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safe registry operations supporting up to 100 concurrent research workflows</span>
- **Plugin Capacity**: <span style="background-color: rgba(91, 57, 243, 0.2)">Support for 50+ registered plugins across LoaderRegistry and SchemaRegistry</span>
- **Memory Efficiency**: <2x data size overhead maintained even with extensive plugin ecosystems
- **Configuration Complexity**: <span style="background-color: rgba(91, 57, 243, 0.2)">Support for nested configurations with 10+ schema_version migrations</span>

The library-based integration model ensures that FlyRigLoader seamlessly integrates into existing Python scientific computing environments while providing the extensibility and reliability required for production research workflows.

### 6.3.2 API DESIGN ARCHITECTURE

#### 6.3.2.1 Public API Interface (updated)

The library provides a clean, protocol-based API through the `flyrigloader.api` module with enhanced functionality for manifest validation, Kedro integration, and plugin management:

```mermaid
classDiagram
    class PublicAPI {
        +discover_experiment_manifest(config) FileManifest @deprecated in 1.0, removal ≥2.0
        +load_experiment_files(config) DataFrame @deprecated in 1.0, removal ≥2.0
        +load_data_file(file_path) Dict
        +transform_to_dataframe(data, schema) DataFrame
        +validate_manifest(config) bool
        +create_kedro_dataset(config_path, experiment_name, **opts) FlyRigLoaderDataSet
        +get_registered_loaders() Dict
        +get_loader_capabilities() Dict
        +validate_config_version(config) bool
        +check_plugin_compatibility() bool
    }
    
    class DependencyProvider {
        +get_config_provider() ConfigProvider
        +get_discovery_provider() DiscoveryProvider
        +get_io_provider() IOProvider
        +get_utils_provider() UtilsProvider
        +set_provider(provider) void
    }
    
    class ConfigProvider {
        +load_config(path) ProjectConfig
        +validate_config(config) bool
        +get_legacy_adapter(config) LegacyAdapter «deprecated»
        +create_config(**kwargs) ProjectConfig
    }
    
    class DiscoveryProvider {
        +discover_files(pattern) List[Path]
        +extract_metadata(filename) Dict
        +get_file_stats(path) FileStats
    }
    
    class FlyRigLoaderDataSet {
        +__init__(config_path, experiment_name, **opts)
        +load() DataFrame
        +save(data) void
        +describe() Dict
        +exists() bool
        +release() void
    }
    
    PublicAPI --> DependencyProvider
    PublicAPI --> FlyRigLoaderDataSet : creates via create_kedro_dataset
    DependencyProvider --> ConfigProvider
    DependencyProvider --> DiscoveryProvider
```

#### 6.3.2.2 Protocol Specifications (updated)

#### Function Signatures and Type Contracts (updated)

```python
# Core API Functions
@deprecated("in 1.0, removal ≥2.0")
def discover_experiment_manifest(
    config: Union[str, Path, Dict, ProjectConfig]
) -> FileManifest

@deprecated("in 1.0, removal ≥2.0")
def load_experiment_files(
    config: Union[str, Path, Dict, ProjectConfig]
) -> pd.DataFrame

def load_data_file(
    file_path: Union[str, Path],
    loader_type: Optional[str] = None
) -> Dict[str, Any]

def transform_to_dataframe(
    data: Dict[str, Any],
    schema: Optional[Union[str, ColumnConfig]] = None
) -> pd.DataFrame

#### Enhanced API Functions
def validate_manifest(
    config: Union[str, Path, Dict, ProjectConfig]
) -> bool

def create_kedro_dataset(
    config_path: Union[str, Path],
    experiment_name: str,
    **opts: Any
) -> FlyRigLoaderDataSet

def get_registered_loaders() -> Dict[str, Type]

def get_loader_capabilities() -> Dict[str, Dict[str, Any]]

def validate_config_version(
    config: Union[str, Path, Dict, ProjectConfig]
) -> bool

def check_plugin_compatibility() -> bool

#### Configuration Builder
def create_config(**kwargs: Any) -> ProjectConfig
```

#### Error Handling Framework (updated)

| Exception Type | Trigger Condition | Recovery Strategy |
|----------------|-------------------|-------------------|
| `ConfigurationError` | Invalid YAML structure or missing required fields | Provide detailed validation messages |
| `FileDiscoveryError` | No files found matching patterns | Log warning and return empty manifest |
| `LoaderRegistrationError` | Plugin registration conflicts | Use last-registered plugin with warning |
| `DataValidationError` | Schema validation failures | Skip invalid columns with logging |
| <span style="background-color: rgba(91, 57, 243, 0.2)">`RegistryError`</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">LoaderRegistry or SchemaRegistry failures</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin isolation and graceful degradation</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">`VersionError`</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Schema version compatibility mismatch</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic migration or version fallback</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">`KedroIntegrationError`</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro DataSet interface violations</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">AbstractDataset compliance enforcement</span> |

#### 6.3.2.3 Authentication and Authorization Framework

The library operates within local Python environments and does not implement network-based authentication mechanisms. Access control is managed through:

- **File System Permissions**: Standard OS-level access controls for data directories
- **Configuration Validation**: Schema-based validation preventing unauthorized parameter modification
- **Plugin Isolation**: Sandboxed plugin execution with restricted system access
- **Version Control Integration**: Configuration-as-code ensuring audit trails for research reproducibility

#### 6.3.2.4 Rate Limiting and Resource Management

Resource management focuses on memory efficiency and computational constraints rather than network-based rate limiting:

| Resource Type | Limit Strategy | Implementation | Monitoring |
|---------------|----------------|----------------|------------|
| **Memory Usage** | <2x data size overhead | Streaming transformations and lazy evaluation | psutil integration |
| **File Handles** | Maximum 100 concurrent | Connection pooling with automatic cleanup | Resource tracking |
| **Plugin Registration** | 50+ plugins supported | Priority-based registry with conflict resolution | Registry size monitoring |
| **Concurrent Operations** | Thread-safe up to 100 workflows | RWLock patterns for registry access | Performance metrics |

#### 6.3.2.5 Versioning and Compatibility (updated)

The library maintains backward compatibility through enhanced versioning mechanisms:

- **Semantic Versioning**: Major.Minor.Patch versioning scheme with clear upgrade paths
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Schema Version Tracking</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Mandatory schema_version field in all configurations enabling automatic migration</span>
- **Deprecation Warnings**: Gradual migration path for breaking changes with clear timelines
- **Legacy Adapters**: Dictionary-based configuration support alongside Pydantic models
- **Protocol Versioning**: Entry point versioning for plugin compatibility
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Engine</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated configuration migration supporting zero-breaking-change upgrades</span>

#### Version Compatibility Matrix

| Library Version | Schema Version | Migration Path | Compatibility Level |
|----------------|----------------|----------------|-------------------|
| 0.9.x | 0.9.x | Direct upgrade | Full backward compatibility |
| 1.0.x | 1.0.x | <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic migration from 0.9.x</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">API deprecation warnings</span> |
| 1.1.x+ | 1.1.x+ | <span style="background-color: rgba(91, 57, 243, 0.2)">Progressive migration support</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced validation and error handling</span> |

#### 6.3.2.6 Documentation Standards

API documentation follows scientific computing best practices:

- **Docstring Standards**: NumPy-style docstrings with type annotations and usage examples
- **Type Annotations**: Complete type hints for all public interfaces supporting IDE integration
- **Example Workflows**: Comprehensive examples covering neuroscience research patterns
- **Error Context**: Detailed error messages with actionable resolution steps
- **Performance Characteristics**: Memory usage and timing information for optimization guidance
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Guides</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Step-by-step migration documentation for version upgrades and configuration changes</span>

The API design architecture ensures that FlyRigLoader provides a robust, extensible, and scientifically-oriented interface for neuroscience data workflows while maintaining the flexibility needed for diverse research environments and the reliability required for reproducible scientific computing.

### 6.3.3 PLUGIN SYSTEM ARCHITECTURE

#### 6.3.3.1 Plugin Registration Flow (updated)

```mermaid
sequenceDiagram
    participant EP as Entry Points
    participant REG as Registry System
    participant LOCK as Thread Lock
    participant VAL as Validator
    participant API as API Layer
    
    EP->>REG: Register plugin via setuptools
    REG->>LOCK: Acquire write lock
    LOCK-->>REG: Lock acquired
    REG->>VAL: Validate protocol compliance
    VAL->>VAL: Check plugin_version compatibility
    VAL->>VAL: Validate supported_schema_versions
    VAL-->>REG: Validation result
    
    alt Plugin Valid
        REG->>REG: Apply priority resolution
        REG->>REG: Store plugin reference
        REG->>LOCK: Release write lock
        LOCK-->>REG: Lock released
        REG-->>EP: Registration successful
    else Plugin Invalid
        REG->>REG: Log error and reject
        REG->>LOCK: Release write lock
        LOCK-->>REG: Lock released
        REG-->>EP: RegistryError thrown
    end
    
    API->>REG: Request plugin by type
    REG->>LOCK: Acquire read lock
    LOCK-->>REG: Lock acquired
    REG->>REG: Lookup registered plugin
    REG->>LOCK: Release read lock
    LOCK-->>REG: Lock released
    REG-->>API: Return plugin instance
```

#### 6.3.3.2 Registry Architecture and Thread Safety (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The plugin registry system implements thread-safe operations using `threading.RLock` patterns, ensuring safe concurrent access across multiple research workflows. The system supports automatic entry-point discovery via `importlib.metadata` with comprehensive plugin lifecycle management.</span>

**Registry Components**

| Component | Protocol Interface | Registration Methods | Purpose |
|-----------|-------------------|---------------------|---------|
| **Loaders** | `BaseLoader` Protocol | <span style="background-color: rgba(91, 57, 243, 0.2)">Entry points, `@auto_register`, decorators, runtime calls</span> | File format support extension |
| **Schemas** | `BaseSchema` Protocol | <span style="background-color: rgba(91, 57, 243, 0.2)">Entry points, `@auto_register`, decorators, runtime calls</span> | Data validation and transformation |
| **Transformers** | `BaseTransformer` Protocol | <span style="background-color: rgba(91, 57, 243, 0.2)">`@auto_register`, runtime registration</span> | Custom data processing pipelines |

**Automatic Registration System**

<span style="background-color: rgba(91, 57, 243, 0.2)">The new `@auto_register` decorator enables automatic entry-point discovery and registration without manual setup.py configuration:</span>

```python
from flyrigloader.registry import auto_register

@auto_register(plugin_type='loader', format_support=['hdf5', 'h5'])
class HDF5Loader:
    plugin_version = "1.2.0"
    supported_schema_versions = ["1.0.0", "1.1.0", "1.2.0"]
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load HDF5 data with version-aware processing"""
        ...

@auto_register(plugin_type='schema', schema_name='neuroimaging')
class NeuroimagingSchema:
    plugin_version = "2.1.0"
    supported_schema_versions = ["2.0.0", "2.1.0"]
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate neuroimaging data structure"""
        ...
```

#### Plugin Entry Point Configuration

```python
# Traditional setup.py configuration (still supported)
entry_points={
    'flyrigloader.loaders': [
        'hdf5 = mypackage.loaders:HDF5Loader',
        'json = mypackage.loaders:JSONLoader',
    ],
    'flyrigloader.schemas': [
        'neuroimaging = mypackage.schemas:NeuroimagingSchema',
        'behavioral = mypackage.schemas:BehavioralSchema',
    ]
}

#### Automatic discovery via importlib.metadata (new)
#### No setup.py configuration required when using @auto_register
```

#### 6.3.3.3 Priority System and Conflict Resolution (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The registry implements a priority-based resolution system with enumerated priorities to handle plugin conflicts systematically.</span>

**Priority Hierarchy**

| Priority Level | Numeric Value | Source | Conflict Resolution Behavior |
|---------------|---------------|---------|----------------------------|
| **BUILTIN** | 1 | Core system components | <span style="background-color: rgba(91, 57, 243, 0.2)">Lowest priority, always replaced by user plugins</span> |
| **USER** | 2 | <span style="background-color: rgba(91, 57, 243, 0.2)">Direct registration calls</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Replaces BUILTIN, replaced by higher priorities</span> |
| **PLUGIN** | 3 | Entry points and `@auto_register` | <span style="background-color: rgba(91, 57, 243, 0.2)">Standard plugin priority, most common level</span> |
| **OVERRIDE** | 4 | <span style="background-color: rgba(91, 57, 243, 0.2)">Explicit override registration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Highest priority, cannot be replaced</span> |

**Conflict Resolution Rules**

<span style="background-color: rgba(91, 57, 243, 0.2)">When multiple plugins register for the same format or schema:</span>

1. <span style="background-color: rgba(91, 57, 243, 0.2)">**Priority-First Resolution**: Plugin with highest priority wins</span>
2. <span style="background-color: rgba(91, 57, 243, 0.2)">**Latest Registration**: If priorities are equal, latest registration wins</span>
3. <span style="background-color: rgba(91, 57, 243, 0.2)">**RegistryError Exception**: Thrown when registration conflicts cannot be resolved (e.g., duplicate OVERRIDE registrations)</span>

```python
# Example conflict resolution scenarios
try:
    # This will succeed - USER priority replaces BUILTIN
    registry.register(MyLoader, priority=Priority.USER)
    
    # This will succeed - PLUGIN priority replaces USER
    registry.register(BetterLoader, priority=Priority.PLUGIN)
    
    # This will raise RegistryError - duplicate OVERRIDE
    registry.register(FirstOverride, priority=Priority.OVERRIDE)
    registry.register(SecondOverride, priority=Priority.OVERRIDE)  # RegistryError!
    
except RegistryError as e:
    logger.error(f"Plugin registration conflict: {e}")
```

#### 6.3.3.4 Version Awareness and Plugin Compatibility (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Plugins may declare version metadata that is validated at registration time to ensure compatibility with the current system and schema versions.</span>

**Plugin Version Declaration**

<span style="background-color: rgba(91, 57, 243, 0.2)">Plugins should declare two version fields:</span>

- <span style="background-color: rgba(91, 57, 243, 0.2)">`plugin_version`: Semantic version of the plugin itself</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`supported_schema_versions`: List of schema versions the plugin can handle</span>

```python
class BaseLoader(Protocol):
    # Optional version metadata - validated at registration
    plugin_version: Optional[str] = None
    supported_schema_versions: Optional[List[str]] = None
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load data from file and return as dictionary"""
        ...
    
    def supports_format(self, file_path: Path) -> bool:
        """Check if loader supports the given file format"""
        ...
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file without loading data"""
        ...
```

**Version Validation Process**

<span style="background-color: rgba(91, 57, 243, 0.2)">During plugin registration, the system validates:</span>

1. <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin version format compliance (semantic versioning)</span>
2. <span style="background-color: rgba(91, 57, 243, 0.2)">Schema version compatibility with current system capabilities</span>
3. <span style="background-color: rgba(91, 57, 243, 0.2)">Dependency version conflicts with other registered plugins</span>

#### 6.3.3.5 Plugin Development Guidelines (updated)

#### Protocol Implementation Requirements

```python
from typing import Protocol, Dict, Any, Optional, List
from pathlib import Path
import pandas as pd

class BaseLoader(Protocol):
    # Version metadata for compatibility validation
    plugin_version: Optional[str] = None
    supported_schema_versions: Optional[List[str]] = None
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load data from file and return as dictionary"""
        ...
    
    def supports_format(self, file_path: Path) -> bool:
        """Check if loader supports the given file format"""
        ...
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file without loading data"""
        ...

class BaseSchema(Protocol):
    # Version metadata for compatibility validation
    plugin_version: Optional[str] = None
    supported_schema_versions: Optional[List[str]] = None
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data structure against schema"""
        ...
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data to conform to schema"""
        ...
```

#### Error Handling and Registration Failures (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The plugin system defines comprehensive error handling for registration failures:</span>

| Exception Type | Trigger Condition | Recovery Strategy |
|----------------|-------------------|------------------|
| `RegistryError` | <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin registration conflicts, priority violations, or registry corruption</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin isolation and graceful degradation to built-in components</span> |
| `ProtocolError` | Plugin doesn't implement required protocol methods | Skip plugin registration with detailed logging |
| `VersionError` | <span style="background-color: rgba(91, 57, 243, 0.2)">Incompatible plugin_version or supported_schema_versions</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Version compatibility warning and conditional registration</span> |
| `ImportError` | Plugin module cannot be imported | Log import failure and continue discovery |

#### Thread Safety Guarantees

<span style="background-color: rgba(91, 57, 243, 0.2)">The registry system provides comprehensive thread safety guarantees:</span>

- <span style="background-color: rgba(91, 57, 243, 0.2)">**Atomic Registration**: All plugin registration operations are atomic with proper lock acquisition/release</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Read-Write Locks**: Separate read and write locks enable concurrent lookups while ensuring safe modifications</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Deadlock Prevention**: Lock ordering and timeout mechanisms prevent deadlock scenarios</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Exception Safety**: Failed registrations properly release locks and maintain registry consistency</span>

#### Performance Characteristics

The plugin architecture maintains high performance through:

- **O(1) Lookup Performance**: Direct hash-based plugin resolution
- **Lazy Loading**: Plugins loaded only when first requested
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Priority Cache**: Pre-computed priority resolution reduces runtime overhead</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Version Cache**: Schema version compatibility cached after first validation</span>
- **Memory Efficiency**: Minimal overhead per registered plugin (<1KB each)

The plugin system architecture provides a robust, extensible foundation for format support while maintaining thread safety, version compatibility, and high performance characteristics essential for scientific computing workflows.

### 6.3.4 KEDRO INTEGRATION ARCHITECTURE

<span style="background-color: rgba(91, 57, 243, 0.2)">FlyRigLoader provides comprehensive Kedro integration through the optional `flyrigloader[kedro]` extra, delivering first-class support for data pipeline workflows with native AbstractDataset compliance and factory function utilities.</span>

#### 6.3.4.1 Kedro Pipeline Integration

Kedro applies software engineering best practices to data science code, making it reproducible, modular and well-documented. FlyRigLoader provides first-class support for Kedro integration through <span style="background-color: rgba(91, 57, 243, 0.2)">a dedicated FlyRigLoaderDataSet implementation that integrates seamlessly with Kedro's catalog system</span>:

```mermaid
sequenceDiagram
    participant KP as Kedro Pipeline
    participant KN as Kedro Node
    participant KC as Kedro Catalog
    participant FDS as FlyRigLoaderDataSet
    participant CF as create_kedro_dataset Factory
    participant FL as FlyRigLoader Core
    participant FS as File System
    participant DF as DataFrame Output
    
    KP->>KN: Execute node with inputs
    KN->>KC: Request dataset from catalog
    KC->>FDS: Load dataset instance
    FDS->>FL: Call discover_experiment_manifest()
    FL->>FS: Discover files based on config
    FS-->>FL: Return file paths
    FL->>FS: Load pickle data files
    FS-->>FL: Return raw data
    FL->>FL: Transform to pandas DataFrame
    FL-->>FDS: Return standardized DataFrame
    FDS-->>KC: Dataset loaded successfully
    KC-->>KN: Return DataFrame to node
    KN-->>KP: Pass DataFrame to next node
    KP->>DF: Output to catalog
    
    Note over CF: Factory function simplifies<br/>dataset creation and configuration
    CF->>FDS: Create configured instance
```

#### 6.3.4.2 Dataset Class Features (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The FlyRigLoaderDataSet implements Kedro's AbstractDataset interface with comprehensive functionality:</span>

- **Read-Only Operation**: <span style="background-color: rgba(91, 57, 243, 0.2)">Dataset is designed for data loading only; `_save()` method raises `NotImplementedError` to maintain data integrity</span>
- **Existence Validation**: <span style="background-color: rgba(91, 57, 243, 0.2)">`_exists()` method validates configuration file presence before attempting data discovery</span>
- **Metadata Description**: <span style="background-color: rgba(91, 57, 243, 0.2)">`_describe()` method returns comprehensive dataset configuration including filepath, experiment name, and additional parameters</span>
- **Parameter Injection**: Accepts experiment-specific configuration through constructor parameters
- **Error Propagation**: Maintains Kedro-compatible exception handling throughout the loading pipeline
- **Lazy Evaluation**: Manifest discovery occurs only during actual data loading operations
- **Thread Safety**: Compatible with Kedro's parallel execution patterns

#### 6.3.4.3 Catalog Configuration (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">FlyRigLoader datasets can be configured directly in Kedro's `catalog.yml` with full parameter support:</span>

```yaml
my_experiment_data:
  type: flyrigloader.FlyRigLoaderDataSet
  filepath: "${base_dir}/config/experiment_config.yaml"
  experiment_name: "baseline_study"
  recursive: true
  extract_metadata: true

behavioral_analysis:
  type: flyrigloader.FlyRigLoaderDataSet
  filepath: "configs/behavioral_config.yaml"
  experiment_name: "behavioral_training"
  date_range: ["2024-01-01", "2024-01-31"]
  
neuroimaging_data:
  type: flyrigloader.FlyRigLoaderDataSet
  filepath: "configs/imaging_config.yaml"
  experiment_name: "calcium_imaging"
  rig_names: ["rig1", "rig2", "rig3"]
  recursive: false
```

#### 6.3.4.4 Kedro Integration Features (updated)

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| **Parameter-Based Config** | Accept `params` dictionary directly | Seamless pipeline parameter passing |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Catalog Integration</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">FlyRigLoaderDataSet returning pandas DataFrames</span>** | Direct compatibility with Kedro datasets |
| **Lazy Loading** | Manifest-based discovery first | Memory-efficient large dataset handling |
| **Error Propagation** | Raise Kedro-compatible exceptions | Consistent error handling in pipelines |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Factory Function</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">`create_kedro_dataset()`</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Simplifies catalog construction and reduces boilerplate</span>** |

#### 6.3.4.5 Factory Function Integration (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The `create_kedro_dataset()` factory function streamlines dataset creation and configuration management:</span>

```python
from flyrigloader.api import create_kedro_dataset

def create_pipeline(**kwargs) -> Pipeline:
    # Factory function simplifies dataset creation
    experiment_dataset = create_kedro_dataset(
        config_path="configs/experiment_config.yaml",
        experiment_name="baseline_study",
        recursive=True,
        extract_metadata=True
    )
    
    return Pipeline([
        node(
            func=lambda: experiment_dataset.load(),
            inputs=None,
            outputs="experiment_data",
            name="load_experiment_data"
        ),
        # Additional processing nodes...
    ])
```

#### 6.3.4.6 Advanced Pipeline Patterns (updated)

#### Multi-Experiment Pipeline Configuration

<span style="background-color: rgba(91, 57, 243, 0.2)">The factory function enables dynamic pipeline generation for multiple experiments:</span>

```python
def create_dynamic_pipeline(experiment_configs: Dict[str, dict]) -> Pipeline:
    nodes = []
    
    for exp_name, config in experiment_configs.items():
        dataset = create_kedro_dataset(
            config_path=config["config_path"],
            experiment_name=exp_name,
            **config.get("options", {})
        )
        
        nodes.append(
            node(
                func=lambda ds=dataset: ds.load(),
                inputs=None,
                outputs=f"{exp_name}_data",
                name=f"load_{exp_name}_data"
            )
        )
    
    return Pipeline(nodes)
```

#### Parameter-Driven Configuration

```python
def create_parameterized_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_kedro_dataset,
            inputs=["params:config_path", "params:experiment_name"],
            outputs="dataset_factory",
            name="create_dataset"
        ),
        node(
            func=lambda factory: factory.load(),
            inputs="dataset_factory",
            outputs="experiment_data",
            name="load_experiment_data"
        ),
        # Downstream processing nodes...
    ])
```

#### 6.3.4.7 Error Handling and Debugging (updated)

The Kedro integration provides comprehensive error handling designed for production pipeline reliability:

```mermaid
flowchart TD
    A[Kedro Node Execution] --> B{Dataset Exists?}
    B -->|Yes| C["FlyRigLoaderDataSet._load()"]
    B -->|No| D[FileNotFoundError]
    
    C --> E{Config Valid?}
    E -->|Yes| F["discover_experiment_manifest()"]
    E -->|No| G[ConfigurationError]
    
    F --> H{Files Found?}
    H -->|Yes| I[Load and Transform Data]
    H -->|No| J[FileDiscoveryError]
    
    I --> K{Data Valid?}
    K -->|Yes| L[Return DataFrame]
    K -->|No| M[DataValidationError]
    
    D --> N[Log Error and Fail Node]
    G --> N
    J --> N
    M --> N
    L --> O[Continue Pipeline]
    
    style L fill:#90EE90
    style N fill:#FFB6C1
```

**Error Recovery Strategies**:

- **Configuration Errors**: <span style="background-color: rgba(91, 57, 243, 0.2)">Detailed validation messages with suggested configuration fixes</span>
- **File Discovery Errors**: <span style="background-color: rgba(91, 57, 243, 0.2)">Comprehensive path validation and alternative discovery pattern suggestions</span>
- **Data Loading Errors**: <span style="background-color: rgba(91, 57, 243, 0.2)">Format-specific error handling with graceful degradation options</span>
- **Kedro Integration Errors**: <span style="background-color: rgba(91, 57, 243, 0.2)">AbstractDataset compliance validation with interface requirement details</span>

#### 6.3.4.8 Performance Characteristics

The Kedro integration maintains optimal performance for data pipeline workflows:

| Metric | Performance Target | Implementation | Monitoring |
|--------|-------------------|----------------|------------|
| **Dataset Initialization** | <10ms per dataset | Lazy configuration loading | Kedro metrics integration |
| **Manifest Discovery** | <500ms for large experiments | Efficient file system traversal | Custom performance hooks |
| **Data Loading** | <2x memory overhead | Streaming transformations | Memory usage tracking |
| **Catalog Integration** | <50ms catalog lookup | Direct dataset reference storage | Kedro profiling support |

#### 6.3.4.9 Installation and Setup (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration requires installation of the optional `kedro` extra:</span>

```bash
# Install FlyRigLoader with Kedro support
pip install flyrigloader[kedro]

#### Or install with additional extras
pip install flyrigloader[kedro,dev,test]
```

**Dependency Requirements**:

- <span style="background-color: rgba(91, 57, 243, 0.2)">`kedro>=0.18.0`: Core Kedro framework with AbstractDataset support</span>
- `pandas>=1.5.0`: DataFrame operations and transformations
- `pydantic>=2.0.0`: Configuration validation and type safety
- `pyyaml>=6.0`: YAML configuration file parsing

#### 6.3.4.10 Migration from Direct API Usage (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Existing code using direct `load_experiment_files()` calls can be migrated to Kedro integration:</span>

#### Legacy Pattern
```python
#### Old direct API usage
from flyrigloader.api import load_experiment_files

def process_experiment():
    config = load_config("experiment_config.yaml")
    data = load_experiment_files(config, "baseline_study")
    return process_data(data)
```

#### Kedro Pattern (updated)
```python
#### New Kedro integration approach
from kedro.pipeline import node, Pipeline
from flyrigloader.api import create_kedro_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=lambda: create_kedro_dataset(
                config_path="experiment_config.yaml",
                experiment_name="baseline_study"
            ).load(),
            inputs=None,
            outputs="experiment_data",
            name="load_experiment_data"
        ),
        node(
            func=process_data,
            inputs="experiment_data",
            outputs="processed_data",
            name="process_experiment_data"
        )
    ])
```

The Kedro Integration Architecture provides a robust, production-ready foundation for incorporating FlyRigLoader into sophisticated data science workflows while maintaining the flexibility and reliability required for neuroscience research pipelines.

### 6.3.5 FILE SYSTEM INTEGRATION ARCHITECTURE

#### 6.3.5.1 File System Operations Flow

```mermaid
flowchart TD
    subgraph "Configuration Layer"
        YAML[YAML Config Files] --> |load| CONFIG[Configuration Parser]
        CONFIG --> |validate| PYDANTIC[Pydantic Models]
    end
    
    subgraph "Discovery Layer"
        PYDANTIC --> |patterns| DISCOVERY[File Discovery Engine]
        DISCOVERY --> |glob/rglob| FILESYSTEM[File System]
        FILESYSTEM --> |file paths| MANIFEST[File Manifest]
    end
    
    subgraph "Loading Layer"
        MANIFEST --> |select files| LOADER[Loader Registry]
        LOADER --> |read| PICKLE[Pickle Files]
        PICKLE --> |deserialize| RAWDATA[Raw Data Dict]
    end
    
    subgraph "Transformation Layer"
        RAWDATA --> |schema| TRANSFORMER[DataFrame Transformer]
        TRANSFORMER --> |validate| SCHEMA[Column Schema]
        SCHEMA --> |convert| DATAFRAME[Pandas DataFrame]
    end
```

#### 6.3.5.2 Storage Integration Patterns

| Storage Type | Access Method | Performance Considerations | Security Model |
|--------------|---------------|---------------------------|----------------|
| **Local File System** | Direct path operations | Optimal for small datasets | File system permissions |
| **Network File System** | Mounted path operations | Network latency affects discovery | Mount-level security |
| **HPC Storage** | Parallel file operations | Optimized for large-scale processing | Cluster authentication |
| **Cloud Storage** | Mounted or FUSE operations | Depends on caching strategy | IAM and encryption |

### 6.3.6 DEPENDENCY INJECTION ARCHITECTURE

#### 6.3.6.1 Dependency Provider System

```mermaid
graph TB
    subgraph "Provider Architecture"
        DEFAULT[DefaultDependencyProvider] --> CONFIG[ConfigProvider]
        DEFAULT --> DISCOVERY[DiscoveryProvider]
        DEFAULT --> IO[IOProvider]
        DEFAULT --> UTILS[UtilsProvider]
    end
    
    subgraph "Testing Integration"
        TEST[Test Suite] --> |set_provider| MOCK[MockProvider]
        MOCK --> |inject| CONFIG
        MOCK --> |inject| DISCOVERY
        MOCK --> |inject| IO
        MOCK --> |inject| UTILS
    end
    
    subgraph "Production Usage"
        API[API Functions] --> |get_provider| DEFAULT
        PLUGIN[Plugin System] --> |get_provider| DEFAULT
    end
```

#### 6.3.6.2 Testability and Mocking

The dependency injection system enables comprehensive testing:

```python
# Test configuration example
from flyrigloader.api import set_dependency_provider
from tests.mocks import MockDependencyProvider

def test_experiment_loading():
    # Inject mock provider for testing
    mock_provider = MockDependencyProvider()
    set_dependency_provider(mock_provider)
    
    # Test with controlled environment
    result = load_experiment_files(test_config)
    assert result.shape == (100, 10)  # Expected dimensions
```

### 6.3.7 INTEGRATION BEST PRACTICES

#### 6.3.7.1 Library Consumer Guidelines (updated)

1. **API Stability**: Always import from `flyrigloader.api` for stable interfaces
2. **<span style="background-color: rgba(91, 57, 243, 0.2)">Configuration Management</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Prefer `create_config()` builder and Pydantic models for type safety and validation. Dictionary configurations are deprecated but remain supported via LegacyConfigAdapter for backward compatibility</span>
3. **<span style="background-color: rgba(91, 57, 243, 0.2)">Version Awareness</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Always specify `schema_version` in configuration files and run `validate_config_version()` during pipeline startup to ensure compatibility and enable automatic migration</span>
4. **Error Handling**: Implement comprehensive exception handling for file operations
5. **Performance**: Leverage manifest-based discovery for selective data loading
6. **<span style="background-color: rgba(91, 57, 243, 0.2)">Extension Development</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin developers should use `@auto_register` decorator for automatic registration and avoid manual registry manipulation to ensure thread safety and compatibility</span>
7. **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Use FlyRigLoaderDataSet in catalog.yml for seamless pipeline integration, and leverage factory helper functions for dynamic catalog construction</span>

#### 6.3.7.2 Recommended Configuration Patterns (updated)

#### Preferred Configuration Building (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The `create_config()` builder function provides the recommended approach for programmatic configuration creation:</span>

```python
from flyrigloader.api import create_config, validate_config_version

#### Recommended approach using config builder
config = create_config(
    experiment_path="/path/to/experiments",
    experiment_name="baseline_study",
    file_patterns=["*.pkl"],
    recursive=True,
    extract_metadata=True,
    schema_version="1.2.0"  # Always specify version
)

#### Validate configuration version before use
if not validate_config_version(config):
    raise ValueError("Configuration version incompatible with current system")
```

#### Legacy Configuration Support (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Dictionary-based configurations remain supported through LegacyConfigAdapter but are deprecated:</span>

```python
from flyrigloader.adapters import LegacyConfigAdapter

#### Legacy approach (deprecated but supported)
legacy_dict = {
    "experiment_path": "/path/to/experiments",
    "experiment_name": "baseline_study",
    "file_patterns": ["*.pkl"],
    "recursive": True
}

#### Adapter automatically adds schema_version if missing
adapted_config = LegacyConfigAdapter.migrate(legacy_dict)
```

#### Kedro Integration Configuration (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Configure FlyRigLoader datasets directly in catalog.yml:</span>

```yaml
# catalog.yml
experiment_data:
  type: flyrigloader.FlyRigLoaderDataSet
  filepath: "${base_dir}/config/experiment_config.yaml"
  experiment_name: "baseline_study"
  recursive: true
  extract_metadata: true

#### Dynamic catalog creation using factory helper
behavioral_experiments:
  type: flyrigloader.FlyRigLoaderDataSet
  factory: flyrigloader.create_kedro_dataset
  config_path: "configs/behavioral_config.yaml"
  experiment_name: "behavioral_training"
```

#### 6.3.7.3 Plugin Development Best Practices (updated)

#### Automatic Registration Pattern (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Plugin developers should use the `@auto_register` decorator for safe, automatic plugin registration:</span>

```python
from flyrigloader.registry import auto_register

@auto_register(plugin_type='loader', format_support=['hdf5', 'h5'])
class HDF5Loader:
    plugin_version = "1.2.0"
    supported_schema_versions = ["1.0.0", "1.1.0", "1.2.0"]
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load HDF5 data with version-aware processing"""
        return {}
    
    def supports_format(self, file_path: Path) -> bool:
        """Check if loader supports the given file format"""
        return file_path.suffix.lower() in ['.hdf5', '.h5']

@auto_register(plugin_type='schema', schema_name='neuroimaging')
class NeuroimagingSchema:
    plugin_version = "2.1.0"
    supported_schema_versions = ["2.0.0", "2.1.0"]
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate neuroimaging data structure"""
        return True
```

#### Manual Registration Alternatives (deprecated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Manual registry manipulation is discouraged due to thread safety concerns:</span>

```python
# Deprecated approach - avoid manual registry access
from flyrigloader.registry import get_loader_registry

#### NOT RECOMMENDED - potential thread safety issues
registry = get_loader_registry()
registry.register(MyLoader(), priority="user")  # Deprecated

#### RECOMMENDED - use @auto_register decorator instead
@auto_register(plugin_type='loader', priority='user')
class MyLoader:
#### Implementation here
    pass
```

#### 6.3.7.4 Integration Limitations

As a library-focused architecture, FlyRigLoader intentionally excludes:
- **REST API endpoints**: No HTTP-based service interfaces
- **Authentication frameworks**: Security managed at code and file system level
- **Message queues**: No asynchronous messaging infrastructure
- **Rate limiting**: No request throttling mechanisms
- **API gateways**: No centralized API management

#### 6.3.7.5 Migration and Compatibility (updated)

The architecture supports smooth migration through:
- **Backward compatibility**: Legacy function support with deprecation warnings
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Incremental adoption</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Gradual migration from dictionary to Pydantic configurations with LegacyConfigAdapter support</span>
- **Plugin ecosystem**: Extensible design for community contributions
- **Version management**: Semantic versioning for predictable upgrade paths
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Feature Flag Support</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Consumers can toggle new behavior via environment variable `FLYRL_USE_V1_REGISTRY=true` for incremental adoption of registry improvements</span>

#### Version Migration Strategy (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">The system provides comprehensive migration support for configuration version upgrades:</span>

| Migration Path | Configuration Strategy | Compatibility Level | Required Actions |
|---------------|------------------------|-------------------|------------------|
| **0.9.x → 1.0.x** | <span style="background-color: rgba(91, 57, 243, 0.2)">Automatic migration via LegacyConfigAdapter</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Full backward compatibility</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Run `validate_config_version()` during startup</span> |
| **1.0.x → 1.1.x+** | <span style="background-color: rgba(91, 57, 243, 0.2)">Progressive migration with `create_config()`</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced validation support</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Update configurations to specify `schema_version`</span> |
| **Dictionary → Pydantic** | <span style="background-color: rgba(91, 57, 243, 0.2)">Gradual transition via builder pattern</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Zero breaking changes</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Adopt `create_config()` for new code</span> |

#### Environment-Based Feature Control (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">For environments requiring gradual adoption of new registry features:</span>

```bash
# Enable new registry behavior (recommended for new deployments)
export FLYRL_USE_V1_REGISTRY=true

#### Use legacy registry behavior (for existing deployments during transition)
export FLYRL_USE_V1_REGISTRY=false
```

<span style="background-color: rgba(91, 57, 243, 0.2)">This feature flag enables teams to validate new registry functionality in development environments before deploying to production research pipelines, ensuring zero-disruption migration paths for critical scientific workflows.</span>

#### 6.3.7.6 Integration Testing and Validation (updated)

#### Configuration Validation Workflow (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Implement comprehensive validation during pipeline initialization:</span>

```python
from flyrigloader.api import create_config, validate_config_version
from flyrigloader.exceptions import VersionError, ConfigurationError

def initialize_pipeline_config(config_path: str) -> ProjectConfig:
    """Initialize and validate pipeline configuration with version awareness"""
    try:
        # Create configuration using recommended builder pattern
        config = create_config(
            config_file=config_path,
            schema_version="1.2.0"  # Explicit version specification
        )
        
        # Validate configuration version compatibility
        if not validate_config_version(config):
            raise VersionError(
                f"Configuration version {config.schema_version} "
                f"incompatible with current system capabilities"
            )
        
        return config
        
    except ConfigurationError as e:
        # Attempt legacy configuration fallback
        logger.warning(f"Modern configuration failed: {e}")
        logger.info("Attempting legacy configuration migration...")
        return migrate_legacy_configuration(config_path)
```

#### Kedro Pipeline Integration Testing (updated)

<span style="background-color: rgba(91, 57, 243, 0.2)">Validate Kedro integration using factory helpers:</span>

```python
from kedro.pipeline import node, Pipeline
from flyrigloader.api import create_kedro_dataset

def create_test_pipeline(**kwargs) -> Pipeline:
    """Create test pipeline with FlyRigLoader integration validation"""
    
    # Use factory helper for dataset creation
    dataset = create_kedro_dataset(
        config_path="test_configs/experiment_config.yaml",
        experiment_name="integration_test",
        recursive=True,
        extract_metadata=True
    )
    
    return Pipeline([
        node(
            func=lambda: dataset.load(),
            inputs=None,
            outputs="test_experiment_data",
            name="load_test_data"
        ),
        node(
            func=validate_data_integrity,
            inputs="test_experiment_data",
            outputs="validation_results",
            name="validate_integration"
        )
    ])
```

This comprehensive integration best practices framework ensures that FlyRigLoader consumers can effectively leverage all system capabilities while maintaining compatibility, performance, and reliability across diverse scientific computing environments.

### 6.3.8 INTEGRATION MONITORING AND OBSERVABILITY

#### 6.3.8.1 Logging Architecture

```mermaid
graph LR
    subgraph "Logging Infrastructure"
        API[API Calls] --> |loguru| STRUCTURED[Structured Logging]
        DISCOVERY[File Discovery] --> |loguru| STRUCTURED
        LOADING[Data Loading] --> |loguru| STRUCTURED
        PLUGIN[Plugin System] --> |loguru| STRUCTURED
        
        STRUCTURED --> |output| CONSOLE[Console Output]
        STRUCTURED --> |output| FILE[Log Files]
        STRUCTURED --> |output| EXTERNAL[External Systems]
    end
```

#### 6.3.8.2 Performance Monitoring

The library provides performance metrics through:
- **Discovery timing**: File system traversal performance
- **Loading benchmarks**: Data loading throughput metrics
- **Memory usage**: Peak memory consumption tracking
- **Plugin performance**: Extension execution timing

### 6.3.9 SECURITY CONSIDERATIONS

#### 6.3.9.1 Integration Security Model

| Security Layer | Implementation | Purpose |
|----------------|----------------|---------|
| **Path Validation** | Security validators in configuration | Prevent path traversal attacks |
| **Code Injection** | Safe YAML loading with `yaml.safe_load` | Prevent arbitrary code execution |
| **Plugin Validation** | Protocol compliance checking | Ensure plugin safety and compatibility |
| **File System Access** | Permission-based access control | Leverage OS-level security |

#### 6.3.9.2 Data Protection

- **In-Memory Only**: No persistent storage of sensitive data
- **Temporary Files**: Automatic cleanup of temporary processing files
- **Configuration Security**: Validation of all configuration inputs
- **Audit Trail**: Comprehensive logging of all operations

#### References

**Files Examined:**
- `src/flyrigloader/api.py` - Primary public API interface and integration functions
- `src/flyrigloader/registries/__init__.py` - Plugin registration and discovery system
- `src/flyrigloader/config/discovery.py` - Configuration-aware file discovery implementation
- `src/flyrigloader/io/pickle.py` - Core data loading and transformation logic
- `tests/flyrigloader/integration/test_end_to_end_workflows.py` - Kedro integration patterns and examples
- `examples/external_project/simple_example.py` - External integration usage patterns
- `docs/extension_guide.md` - Plugin development and extension documentation

**Technical Specifications Referenced:**
- Section 1.2 System Overview - Integration context and positioning
- Section 5.2 Component Details - Architecture components and interfaces

**External Resources:**
- Kedro Framework Documentation - Production-ready data science workflows
- Kedro Plugin Architecture - Extension mechanisms and pluggy integration
- Scientific Python Ecosystem - Integration patterns and best practices

## 6.4 SECURITY ARCHITECTURE

### 6.4.1 Security Architecture Overview

#### 6.4.1.1 System Security Context

**Detailed Security Architecture is not applicable for this system** as FlyRigLoader is a local Python library designed for academic neuroscience research that operates within a controlled, single-user environment. The system does not provide network services, web APIs, or multi-user access patterns that would require comprehensive authentication, authorization, or encryption architectures.

#### 6.4.1.2 Security Approach

FlyRigLoader implements **defense-in-depth security practices** appropriate for a research tool that handles local file system operations:

| Security Layer | Implementation | Purpose |
|----------------|---------------|---------|
| **Input Validation** | Path traversal protection, regex validation | Prevent malicious input processing |
| **Runtime Protection** | YAML safe loading, timeout controls | Secure configuration and data handling |
| **Audit Logging** | Comprehensive security event logging | Monitoring and incident response |
| **Static Analysis** | Bandit security scanning in CI/CD | Vulnerability detection and prevention |

### 6.4.2 Security Implementation Framework

#### 6.4.2.1 Input Validation and Sanitization

The system implements comprehensive input validation to protect against common security vulnerabilities:

**Path Traversal Protection:**
- Validates all file paths against directory traversal attacks (`../` sequences)
- Blocks access to system paths and sensitive directories
- Rejects remote URL patterns to prevent external resource access
- Implemented in `src/flyrigloader/config/validators.py`

**Configuration Security:**
- YAML files loaded using `safe_load()` to prevent code execution
- Pydantic model validation enforces type safety and constraints
- Environment variable validation with sanitization
- Regex pattern compilation with timeout protection

```mermaid
flowchart TD
    subgraph "Input Validation Security Flow"
        A[User Input] --> B{Path Traversal Check}
        B -->|Contains ../| C[Block Access]
        B -->|Safe Path| D{System Path Check}
        D -->|System Path| E[Block Access]
        D -->|User Path| F{URL Check}
        F -->|Remote URL| G[Block Access]
        F -->|Local Path| H[Allow Access]
        
        C --> I[Log Security Event]
        E --> I
        G --> I
        I --> J[Terminate Request]
        
        H --> K[Continue Processing]
    end
```

#### 6.4.2.2 Access Control Implementation

**File System Security:**
- Relies on operating system file permissions for access control
- Validates read permissions before file operations
- Respects directory access restrictions
- No elevation of privileges required or performed

**Configuration Access Control:**
- Environment variable overrides restricted to safe patterns
- Configuration file access limited to user-specified paths
- No automatic discovery of configuration files outside project scope

#### 6.4.2.3 Security Logging and Monitoring

**Structured Security Logging:**
- All security events logged with full context using Loguru framework
- Security violations include: path traversal attempts, invalid configurations, file access denials
- Log rotation and compression for long-term audit trail maintenance
- Structured logging format for automated security analysis

**Audit Trail Components:**
- Configuration validation results with security context
- File access patterns and permission checks
- Error conditions with security implications
- Performance metrics with security thresholds

```mermaid
flowchart LR
    subgraph "Security Logging Architecture"
        A[Security Events] --> B[Loguru Logger]
        B --> C[Console Output<br/>INFO Level]
        B --> D[File Output<br/>DEBUG Level]
        D --> E[Log Rotation<br/>10MB Limit]
        E --> F[Compression<br/>gzip]
        
        G[Security Violations] --> H[Structured Logs]
        I[Access Attempts] --> H
        J[Configuration Errors] --> H
        H --> B
    end
```

### 6.4.3 Data Protection Standards

#### 6.4.3.1 Data Handling Security

**Local Data Protection:**
- No sensitive data transmission (local file system operations only)
- File system permissions provide data access control
- No data encryption requirements (research data, local storage)
- Temporary files cleaned up automatically

**Configuration Data Security:**
- Configuration files stored in user-controlled directories
- No credential storage or management required
- Environment variable handling follows secure patterns
- No hardcoded sensitive information in codebase

#### 6.4.3.2 Privacy and Compliance

**Research Data Privacy:**
- System designed for anonymized research data
- No personally identifiable information (PII) processing
- Data handling complies with research ethics standards
- Local processing ensures data sovereignty

| Data Type | Security Level | Protection Method |
|-----------|---------------|------------------|
| **Experimental Data** | Internal | File system permissions |
| **Configuration Files** | Internal | User directory restrictions |
| **Log Files** | Internal | Automatic rotation and cleanup |
| **Temporary Files** | Internal | Automatic cleanup on exit |

### 6.4.4 Security Validation Framework

#### 6.4.4.1 Static Security Analysis

**Automated Security Scanning:**
- Bandit security scanner integrated in CI/CD pipeline
- Vulnerability detection for common Python security issues
- Dependency vulnerability scanning
- Code quality analysis with security implications

**Security Testing:**
- Unit tests for input validation functions
- Integration tests for security boundary conditions
- Error handling verification for security edge cases
- Performance testing with security constraints

#### 6.4.4.2 Runtime Security Controls

**Error Handling Security:**
- No sensitive information exposed in error messages
- Structured exception hierarchy with security context
- Secure error logging without credential leakage
- Graceful degradation for security-related failures

```mermaid
flowchart TD
    subgraph "Security Validation Framework"
        A[Code Changes] --> B[Static Analysis<br/>Bandit Scanner]
        B --> C{Security Issues?}
        C -->|Yes| D[Block Deployment]
        C -->|No| E[Security Tests]
        E --> F{Tests Pass?}
        F -->|No| D
        F -->|Yes| G[Runtime Validation]
        G --> H[Security Logging]
        H --> I[Monitoring]
        
        D --> J[Security Review]
        J --> K[Fix Issues]
        K --> A
    end
```

### 6.4.5 Future Security Considerations

#### 6.4.5.1 Potential Security Enhancements

**Institutional Integration:**
- Future support for institutional authentication systems
- Role-based access control for shared research deployments
- Enhanced audit logging for compliance requirements
- Integration with research data management platforms

**Advanced Security Features:**
- Optional data encryption for sensitive research data
- Digital signatures for research reproducibility
- Enhanced access logging and monitoring
- Integration with security information and event management (SIEM) systems

#### 6.4.5.2 Security Maintenance

**Ongoing Security Practices:**
- Regular dependency updates for security patches
- Continuous security scanning in CI/CD pipeline
- Security review process for new features
- Vulnerability management and response procedures

### 6.4.6 Security Configuration Reference

#### 6.4.6.1 Security Configuration Matrix

| Security Control | Implementation | Configuration Required |
|------------------|---------------|----------------------|
| **Path Validation** | Automatic | None (built-in protection) |
| **YAML Safe Loading** | Automatic | None (default behavior) |
| **Security Logging** | Automatic | Optional log level configuration |
| **Input Sanitization** | Automatic | None (comprehensive validation) |

#### 6.4.6.2 Security Monitoring Configuration

```yaml
# Security logging configuration (optional)
security_logging:
  level: INFO
  include_stack_traces: false
  log_file_access: true
  log_configuration_changes: true
  alert_on_security_violations: true
```

#### References

#### Files Examined
- `src/flyrigloader/config/validators.py` - Input validation and path traversal protection
- `src/flyrigloader/config/yaml_config.py` - YAML safe loading implementation
- `src/flyrigloader/api.py` - Main API interface showing security boundaries
- `src/flyrigloader/exceptions.py` - Security exception handling
- `src/flyrigloader/__init__.py` - Security logging configuration

#### Technical Specification Sections Retrieved
- `4.7 SECURITY CONSIDERATIONS` - Input validation flows and data protection measures
- `5.4 CROSS-CUTTING CONCERNS` - Current security implementation and future considerations
- `2.2 FUNCTIONAL REQUIREMENTS TABLE` - Security requirements in configuration management
- `3.1 PROGRAMMING LANGUAGES` - Security scanning with Bandit
- `1.2 SYSTEM OVERVIEW` - System context and security positioning

## 6.5 MONITORING AND OBSERVABILITY

### 6.5.1 Monitoring Architecture Classification

**Detailed Monitoring Architecture is not applicable for this system.**

FlyRigLoader operates as a Python library designed for academic research in neuroscience, not a deployed service or distributed system. It's not that observability isn't important for client applications; it's that clients tend not to be written in Python. It's not that observability does not matter for, say, data science; it's that tooling for observability in data science (mostly Jupyter and quick feedback) is different.

#### 6.5.1.1 System Context and Monitoring Scope

As a scientific data processing library integrated into user applications and analysis pipelines, FlyRigLoader requires observability patterns fundamentally different from traditional service monitoring:

| Monitoring Aspect | Traditional Services | FlyRigLoader Library |
|-------------------|---------------------|---------------------|
| **Deployment Model** | Standalone services with infrastructure | Embedded in user applications |
| **Lifecycle Management** | Continuous operation with uptime requirements | Invoked per research workflow |
| **Resource Monitoring** | Infrastructure metrics and scaling | Per-operation performance validation |
| **Error Handling** | Service availability and recovery | Library exception handling and debugging |

#### 6.5.1.2 Library-Specific Observability Requirements

The system implements observability appropriate for research data processing. <span style="background-color: rgba(91, 57, 243, 0.2)">These new log streams are essential for validating the success of the new plugin-style registry and version-migration mechanisms introduced in the refactor.</span>

- **No infrastructure monitoring needed** - operates within user applications
- **Comprehensive structured logging** - detailed operational visibility for debugging
- **CI/CD performance tracking** - ensures SLA compliance before release
- **Quality metrics enforcement** - maintains code health through automation
- **User-configurable logging** - integrates with application monitoring systems
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Registry operation logging** - emits INFO-level structured logs for registry operations (registration, deregistration, conflict resolution) and for automatic configuration migration events</span>

### 6.5.2 Monitoring Infrastructure

#### 6.5.2.1 Structured Logging Architecture

FlyRigLoader uses Loguru, a simple and powerful logging library for Python, implementing comprehensive structured logging throughout all operations:

```mermaid
flowchart TD
    subgraph "Log Sources"
        L1[API Calls] --> Logger[Loguru Logger]
        L2[Discovery Events] --> Logger
        L3[Load Operations] --> Logger
        L4[Transform Steps] --> Logger
        L5[Errors/Warnings] --> Logger
        L6[Registry Events] --> Logger
        L7[Migration Events] --> Logger
    end
    
    subgraph "Loguru Configuration"
        Logger --> Console[Console Handler]
        Logger --> File[File Handler]
        
        Console --> |INFO+| ColorOutput[Colored Terminal Output]
        File --> |DEBUG+| RotatingFile[Rotating Log Files]
        
        RotatingFile --> |10MB| Rotation[Log Rotation]
        Rotation --> |7 days| Compression[Gzip Compression]
    end
    
    subgraph "Log Outputs"
        ColorOutput --> UserConsole[User Console]
        Compression --> LogDirectory[logs/flyrigloader_YYYYMMDD.log]
    end
    
    subgraph "User Integration"
        Logger --> |Reconfigurable| UserApp[User Application Logging]
        UserApp --> |Custom Handlers| External[External Monitoring Systems]
    end
```

<span style="background-color: rgba(91, 57, 243, 0.2)">Registry and migration events are now first-class, structured log sources captured by default.</span> The enhanced registry pattern implementation emits INFO-level structured logs for all registration events including loader registration, deregistration, and conflict resolution. Additionally, automatic configuration version migration events are logged to provide full visibility into configuration upgrades and compatibility handling during runtime.

#### Log Configuration Matrix

| Component | Level | Format | Retention | Purpose |
|-----------|-------|--------|-----------|---------|
| Console Handler | INFO | Colorized, Human-readable | Session-based | User-facing operational messages |
| File Handler | DEBUG | Structured JSON | 7 days compressed | Detailed debugging and audit trail |
| Rotation Handler | ALL | Preserved format | 10MB per file | Prevent disk space issues |
| User-configured | Configurable | Custom format | User-defined | Application integration |

#### 6.5.2.2 Performance Benchmarking Infrastructure

Performance monitoring occurs during continuous integration using libraries such as pytest-benchmark:

```mermaid
flowchart LR
    subgraph "Benchmark Suite"
        B1[Data Loading SLAs] --> B2["<1s per 100MB"]
        B3[Discovery Performance] --> B4["<5s for 10k files"]
        B5[Transform Operations] --> B6["Linear O(n) scaling"]
        B7[Memory Efficiency] --> B8["<2x data size overhead"]
    end
    
    subgraph "CI/CD Integration"
        B2 --> pytest[pytest-benchmark]
        B4 --> pytest
        B6 --> pytest
        B8 --> memory[memory-profiler]
        
        pytest --> Results[Benchmark Results]
        memory --> Results
        Results --> QualityGates[Quality Gates]
        QualityGates --> |Pass/Fail| MergeDecision[Merge Decision]
    end
    
    subgraph "Reporting"
        Results --> Dashboard[Performance Dashboard]
        Results --> Trends[Historical Trends]
        Results --> Alerts[SLA Violations]
    end
```

#### 6.5.2.3 Quality Metrics Collection

The system defines KPIs based on critical metrics and uses Python libraries like Pandas and NumPy to calculate and track these indicators:

```mermaid
flowchart TD
    subgraph "Quality Metrics Collection"
        Q1[Code Coverage] --> |>90%| Coverage[coverage.py]
        Q2[Type Safety] --> |100%| MyPy[mypy strict]
        Q3[Code Quality] --> Linting[flake8 + plugins]
        Q4[Performance] --> Benchmarks[pytest-benchmark]
        Q5[Memory Usage] --> Profiling[memory-profiler]
    end
    
    subgraph "Automated Reporting"
        Coverage --> Report[HTML/XML/JSON Reports]
        MyPy --> Report
        Linting --> Report
        Benchmarks --> Report
        Profiling --> Report
        
        Report --> Dashboard[Quality Dashboard]
        Report --> PRComment[PR Comments]
        Report --> Trends[Historical Analysis]
    end
    
    subgraph "Enforcement Gates"
        Dashboard --> Gates{Quality Gates}
        Gates -->|Pass| Merge[Allow Merge]
        Gates -->|Fail| Block[Block Merge]
        Gates --> Notification[Team Notifications]
    end
```

### 6.5.3 Observability Patterns

#### 6.5.3.1 Health Checks and Validation

The library implements comprehensive validation patterns rather than traditional health checks:

#### Configuration Health Validation

| Validation Type | Implementation | Threshold | Response |
|-----------------|----------------|-----------|----------|
| Schema Validation | Pydantic models | 100% type safety | Fail-fast with detailed errors |
| File System Access | Path validation | File existence checks | Graceful degradation with warnings |
| Performance Baselines | Benchmark thresholds | SLA-defined limits | CI/CD gate enforcement |
| Memory Constraints | Resource monitoring | <2x data size | Error reporting and suggestions |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Version Compatibility</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">`schema_version` field check + automatic migrator</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Must match current supported version or be migratable</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Auto-migration with INFO logs; raise `VersionError` if unsupported</span> |

<span style="background-color: rgba(91, 57, 243, 0.2)">Failed version compatibility checks will emit structured `VersionError` logs and trigger migration pathways, ensuring seamless configuration upgrades while maintaining full operational visibility into version handling processes.</span>

#### Operational Health Indicators

Logging is a formalization of print debugging. The Python logging library, for all of its faults, allows standardized logging:

```python
# Example health check logging patterns from the codebase
DEBUG   - Detailed execution flow, path resolution, discovery details
INFO    - Configuration loading, major operations, file counts  
WARNING - Missing files, validation issues, deprecation notices
ERROR   - Failed operations, invalid configurations, access denied
```

#### 6.5.3.2 Performance Metrics and SLA Monitoring

#### Core Performance SLAs

| Metric | Target | Measurement Method | Enforcement |
|--------|--------|-------------------|-------------|
| File Discovery | <5s for 10,000 files | pytest-benchmark | CI quality gate |
| Data Loading | <1s per 100MB | pytest-benchmark | CI quality gate |
| Format Detection | <100ms overhead | pytest-benchmark | CI quality gate |
| Memory Efficiency | O(n) scaling | memory-profiler | CI validation |
| Configuration Parsing | <50ms for typical configs | pytest-benchmark | CI quality gate |

#### Business Metrics for Research Context

The system identifies specific metrics and traces that are crucial for achieving observability goals:

| Research Metric | Measurement | Significance |
|----------------|-------------|--------------|
| Data Processing Throughput | Files processed per second | Research pipeline efficiency |
| Discovery Success Rate | Found files / Expected files | Data completeness validation |
| Format Support Coverage | Supported formats / Total formats | Extensibility measurement |
| Error Recovery Rate | Handled errors / Total errors | System robustness |

#### 6.5.3.3 Development and Debugging Observability

#### Context-Rich Error Reporting

Error-tracking systems in Python usually hook into a generic exception handler, collect data, and send it to a dedicated error aggregator:

```python
# Example context-rich error patterns
logger.error("Pydantic validation error in configuration: {}", e)
logger.error("Available experiments: {}", available_experiments)
logger.info("Resolution methods tried: 1) command line, 2) config major_data_directory, 3) environment variable")
```

#### Structured Logging Hierarchy

Modern observability practices favor structured formats like JSON, which are easier to parse, query, and correlate in log management platforms:

| Log Level | Content | Context Fields | Usage |
|-----------|---------|---------------|--------|
| DEBUG | Execution flow details | `function`, `parameters`, `timing` | Development debugging |
| INFO | Operation summaries | `operation`, `file_count`, `duration` | User feedback |
| WARNING | Recoverable issues | `issue_type`, `resolution`, `impact` | System health |
| ERROR | Critical failures | `error_type`, `stack_trace`, `context` | Incident response |

### 6.5.4 Incident Response and Troubleshooting

#### 6.5.4.1 Error Classification and Response (updated)

Since FlyRigLoader operates as a library, incident response focuses on debugging support rather than service recovery. <span style="background-color: rgba(91, 57, 243, 0.2)">The enhanced error classification system now includes comprehensive support for the refactor's advanced plugin model, schema version-awareness, and Kedro integration capabilities, providing structured diagnostics for the expanded architectural components.</span>

#### Error Categories and Troubleshooting (updated)

| Error Category | Typical Causes | Diagnostic Approach | Resolution Pattern |
|----------------|----------------|-------------------|------------------|
| Configuration Errors | Invalid YAML, missing files | Schema validation logs | User configuration guidance |
| Data Access Issues | Permissions, missing data | File system validation | Path resolution assistance |
| Performance Degradation | Large datasets, memory limits | Performance profiling | Resource optimization suggestions |
| Integration Problems | Version conflicts, API changes | Compatibility validation | Migration documentation |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Registry Errors</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Plugin conflicts, duplicate loaders</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Review registry INFO/ERROR logs</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Adjust plugin priorities or names</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Version Errors</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Unsupported schema_version</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Inspect auto-migration logs</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Update or migrate configuration</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration Errors</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Misconfigured catalog, missing optional dependency</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Check `KedroIntegrationError` stack trace</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Install Kedro extra or correct catalog.yml</span> |

#### 6.5.4.2 User Application Integration Support

#### Monitoring Integration Patterns

The QueueHandler and QueueListener provide a powerful pattern for asynchronous, non-blocking logging in Python. This is crucial for high-performance applications:

```python
# Example user application integration
from flyrigloader import logger

#### Reconfigure for application needs
logger.remove()
logger.add(
    "app_logs/flyrigloader.log",
    level="WARNING",
    format=custom_format,
    rotation="10 MB",
    retention="7 days"
)
```

#### Custom Monitoring Hooks

| Integration Method | Use Case | Implementation |
|-------------------|----------|----------------|
| Log Handler Reconfiguration | Application-specific formats | `logger.remove()` and `logger.add()` |
| Custom Formatters | JSON/structured logging | `logging.Formatter` subclasses |
| Metrics Collection | Performance tracking | Wrapper functions with timing |
| Error Aggregation | Centralized error tracking | Exception handler integration |

#### 6.5.4.3 Continuous Improvement Tracking

#### Quality Trend Analysis

Documentation of insights gained from telemetry data analysis and actions taken based on those insights helps share knowledge and learn from past experiences:

```mermaid
flowchart TD
    subgraph "Metrics Collection"
        M1[Performance Benchmarks] --> Trends[Trend Analysis]
        M2[Error Rates] --> Trends
        M3[Coverage Metrics] --> Trends
        M4[User Feedback] --> Trends
    end
    
    subgraph "Analysis Process"
        Trends --> Analysis[Statistical Analysis]
        Analysis --> Insights[Actionable Insights]
        Insights --> Actions[Improvement Actions]
    end
    
    subgraph "Feedback Loop"
        Actions --> Implementation[Code Changes]
        Implementation --> Testing[Validation Testing]
        Testing --> M1
        Actions --> Documentation[Process Documentation]
    end
```

### 6.5.5 Summary

FlyRigLoader implements library-appropriate observability patterns that provide comprehensive insight into system behavior while avoiding the complexity of traditional service monitoring infrastructure. The fact is, observability is important, so write it in early in the process and maintain it throughout. In turn, it will help you maintain your software.

The system's observability approach emphasizes:

- **Structured logging** using Loguru for comprehensive operational visibility
- **Performance validation** through CI/CD benchmarking and quality gates
- **Quality metrics enforcement** maintaining code health through automation
- **User integration flexibility** supporting diverse research environments
- **Debugging support** with context-rich error reporting and tracing
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Registry operations logging** emitting INFO-level structured logs for plugin registration, deregistration, conflict resolution, and automatic configuration migration events to provide full visibility into the enhanced plugin-style registry and version-migration mechanisms introduced in the refactor</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Enhanced error categorization** through additional error classes (`RegistryError`, `VersionError`, `KedroIntegrationError`) that provide structured diagnostics for plugin conflicts, schema version incompatibilities, and optional integration failures to improve troubleshooting clarity and reduce debugging time</span>

This approach ensures researchers and developers have the necessary visibility into system behavior while maintaining the lightweight, library-focused architecture appropriate for scientific computing environments.

#### References

**Web Search Results:**
- ThinhDA Guide to Python Observability - Loguru logging framework reference
- SigNoz Python Performance Monitoring - pytest-benchmark usage patterns
- Dash0 Python Logging Guide - Structured logging best practices
- Opensource.com Python Observability - Python logging library patterns

**Technical Specification Sections:**
- `1.2 SYSTEM OVERVIEW` - System architecture and performance targets
- `3.7 TECHNOLOGY STACK ARCHITECTURE` - Technology selection and integration patterns
- `5.1 HIGH-LEVEL ARCHITECTURE` - Component organization and data flow

**Files Examined:**
- `src/flyrigloader/__init__.py` - Loguru logging initialization and configuration
- `tests/flyrigloader/test_logging.py` - Logging test suite and validation
- `examples/external_project/analyze_experiment.py` - Example of logging usage in practice
- `tests/coverage/validate-coverage.py` - Coverage and quality gate validation
- `tests/coverage/scripts/collect-test-metrics.py` - Test metrics collection and reporting
- `tests/coverage/quality-gates.yml` - Quality gate configuration
- `tests/coverage/scripts/check-performance-slas.py` - Performance SLA validation
- `tests/coverage/scripts/analyze-coverage-trends.py` - Coverage trend analysis
- `tests/coverage/coverage-thresholds.json` - Coverage threshold definitions

**Directories Analyzed:**
- `examples/` - Implementation examples and usage patterns
- `examples/external_project/` - Detailed example scripts and integration
- `.github/workflows/` - GitHub Actions CI/CD configuration
- `src/flyrigloader/utils/` - Utility modules with logging integration
- `tests/coverage/` - Quality monitoring and metrics collection scripts

## 6.6 TESTING STRATEGY

### 6.6.1 TESTING APPROACH OVERVIEW

The FlyRigLoader testing strategy implements a comprehensive, multi-layered approach designed to ensure reliability, performance, and maintainability of the scientific data management system. The strategy leverages pytest as the primary testing framework, with specialized tools for property-based testing, performance benchmarking, and coverage analysis. <span style="background-color: rgba(91, 57, 243, 0.2)">The strategy now covers seven layers: API, Configuration, Discovery, I/O, Registry, Migration, and Kedro Integration.</span>

#### 6.6.1.1 Testing Philosophy

The testing approach follows these core principles:

- **Layered Testing**: Tests mirror the system's layered architecture (API, Configuration, Discovery, I/O, Registry<span style="background-color: rgba(91, 57, 243, 0.2)">, Migration System, Kedro Integration</span>)
- **Property-Based Validation**: Uses Hypothesis for edge case discovery and robustness testing
- **Performance-Driven**: Continuous benchmarking with defined SLAs for scientific workflows
- **Quality Gates**: Automated enforcement of coverage and performance thresholds
- **Integration Focus**: Extensive testing of component interactions and data flow

#### 6.6.1.2 Testing Scope Matrix

| Component | Unit Tests | Integration Tests | Performance Tests | Property Tests |
|-----------|------------|------------------|------------------|----------------|
| API Layer | ✓ | ✓ | ✓ | ✓ |
| Configuration Layer | ✓ | ✓ | ✗ | ✓ |
| Discovery Engine | ✓ | ✓ | ✓ | ✓ |
| I/O Pipeline | ✓ | ✓ | ✓ | ✓ |
| Registry System | ✓ | ✓ | ✗ | ✓ |
| **Migration System** | **✓** | **✓** | **✗** | **✓** |
| **Kedro Integration** | **✓** | **✓** | **✓** | **✗** |

### 6.6.2 UNIT TESTING STRATEGY

#### 6.6.2.1 Testing Framework and Tools (updated)

The unit testing infrastructure is built on the following core technologies:

- **pytest ≥7.0.0**: Primary testing framework with fixture-based architecture
- **pytest-cov ≥6.1.1**: Coverage analysis with branch coverage enforcement
- **pytest-mock ≥3.14.1**: Mocking framework for isolating components
- **pytest-xdist ≥3.7.0**: Parallel test execution for improved performance
- **pytest-timeout ≥2.3.0**: Test timeout management for long-running operations
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro (extra)**: Will be installed in unit-test session when the `kedro` extra is available for comprehensive Kedro integration testing</span>

#### 6.6.2.2 Test Organization Structure (updated)

The test structure mirrors the source code organization to ensure maintainability:

```
tests/
├── flyrigloader/
│   ├── api/                    # API layer tests
│   ├── config/                 # Configuration layer tests
│   ├── discovery/              # Discovery engine tests
│   ├── io/                     # I/O pipeline tests
│   ├── utils/                  # Utility function tests
│   ├── kedro/                  # Kedro DataSet unit tests
│   ├── migration/              # Version-migration tests
│   ├── registries/             # Registry thread-safety tests
│   └── benchmarks/             # Performance benchmarks
├── coverage/                   # Coverage reporting infrastructure
└── conftest.py                 # Central pytest configuration
```

#### 6.6.2.3 Mocking Strategy (updated)

The mocking strategy emphasizes isolation while maintaining realistic test scenarios:

- **File System Mocking**: Mock file operations using pytest-mock for reproducible tests
- **Registry Mocking**: Mock registry operations to test component registration independently
- **Configuration Mocking**: Mock YAML loading and Pydantic validation for configuration tests
- **External Dependency Mocking**: Mock pandas, numpy operations for focused unit tests
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro Mocking**: Use pytest-mock to stub AbstractDataset behaviours for isolated Kedro integration testing</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Migration Mocking**: Mock legacy-to-current migration paths for deterministic version compatibility testing</span>

#### 6.6.2.4 Code Coverage Requirements (updated)

The coverage system enforces strict quality standards:

- **Global Coverage**: 90% minimum overall coverage (--cov-fail-under=90)
- **Critical Module Coverage**: 100% coverage required for:
  - `api.py`: Public interface functions
  - `yaml_config.py`: Configuration parsing and validation
  - `discovery.py`: File discovery logic
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`kedro/datasets.py`: Kedro AbstractDataset implementation and catalog integration</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`migration/migrators.py`: Configuration version migration and compatibility logic</span>
  - <span style="background-color: rgba(91, 57, 243, 0.2)">`registries/__init__.py`: Thread-safe registry operations and plugin discovery</span>
- **Branch Coverage**: Enabled for all modules with complex conditional logic
- **Coverage Reporting**: HTML, XML, and JSON formats with Codecov integration

#### 6.6.2.5 Test Naming Conventions

Tests follow consistent naming patterns for clarity and maintainability:

- **Test Module Names**: `test_<module_name>.py`
- **Test Class Names**: `Test<ComponentName>` (e.g., `TestFileDiscoverer`)
- **Test Method Names**: `test_<behavior>_<condition>_<expected_result>`
- **Parameterized Tests**: `test_<behavior>_with_<parameter_type>`

#### 6.6.2.6 Test Data Management

Test data is managed through a centralized fixture system:

- **Fixture Hierarchy**: Base fixtures in `conftest.py` with module-specific extensions
- **Data Factories**: Hypothesis strategies for generating test data
- **Mock Data**: Realistic scientific datasets for testing data transformations
- **Configuration Fixtures**: Sample YAML configurations for testing workflows

### 6.6.3 INTEGRATION TESTING STRATEGY

#### 6.6.3.1 Service Integration Test Approach

Integration tests focus on component interactions and data flow validation:

- **Layer Integration**: Tests between API, Configuration, Discovery, and I/O layers
- **Protocol Validation**: Tests for protocol-based dependency injection
- **Registry Integration**: Tests for plugin registration and lookup mechanisms, <span style="background-color: rgba(91, 57, 243, 0.2)">including thread-safety and plugin discovery scenarios executed with 100 concurrent registry lookups to validate system stability under high-concurrency research workflows</span>
- **Data Flow Testing**: End-to-end data processing pipeline validation
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro Catalog Integration**: Validate `FlyRigLoaderDataSet` load behaviour inside a Kedro pipeline, ensuring proper AbstractDataset compliance and catalog.yml configuration handling</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Version Migration Flow**: Verify automatic migration for legacy configs during integrated runs, testing seamless upgrade paths and configuration compatibility across schema versions</span>

#### 6.6.3.2 API Testing Strategy

The API testing strategy validates the public interface and backward compatibility:

- **Function Signature Testing**: Validates all public API functions
- **Deprecation Testing**: Ensures legacy functions maintain compatibility
- **Error Handling Testing**: Tests error propagation and user-friendly messages
- **Integration Point Testing**: Tests with Kedro pipelines and Jupyter notebooks

#### 6.6.3.3 Configuration Integration Testing

Configuration tests validate the complete configuration processing pipeline:

- **YAML Parsing Integration**: Tests PyYAML with Pydantic validation
- **Security Validation**: Tests path traversal protection and safe loading
- **Legacy Adapter Testing**: Tests MutableMapping compatibility
- **Environment Override Testing**: Tests runtime configuration modifications

#### 6.6.3.4 File System Integration Testing

File system integration tests ensure robust file operations:

- **Discovery Pattern Testing**: Tests glob/rglob patterns with real directory structures
- **Metadata Extraction Testing**: Tests regex-based filename parsing
- **File Loading Testing**: Tests pickle and compressed file loading
- **Cross-Platform Testing**: Tests path operations across operating systems

#### 6.6.3.5 Test Environment Management

Test environments are managed through isolated fixtures:

- **Temporary Directories**: pytest tmpdir fixtures for file system tests
- **Mock Registries**: Isolated registry instances for plugin testing
- **Configuration Contexts**: Isolated configuration environments
- **Resource Cleanup**: Automated cleanup of test resources

#### 6.6.3.6 Kedro Integration Test Framework

<span style="background-color: rgba(91, 57, 243, 0.2)">The Kedro integration testing framework ensures seamless pipeline integration and AbstractDataset compliance:</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**AbstractDataset Interface Testing**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Load Method Validation**: Tests `FlyRigLoaderDataSet._load()` returns properly formatted pandas DataFrames with required Kedro metadata columns</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Existence Checking**: Validates `_exists()` method correctly identifies configuration file presence before data discovery</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Metadata Description**: Tests `_describe()` method returns comprehensive dataset configuration including filepath, experiment name, and additional parameters</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Error Propagation**: Ensures Kedro-compatible exception handling throughout the loading pipeline</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**Catalog Configuration Testing**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**YAML Catalog Integration**: Tests dataset configuration in `catalog.yml` with parameter passing and type validation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Factory Function Testing**: Validates `create_kedro_dataset()` factory helper functionality and dynamic catalog construction</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Parameter Injection**: Tests experiment-specific configuration through constructor parameters and catalog variable substitution</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**Pipeline Execution Testing**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Node Integration**: Tests FlyRigLoaderDataSet within actual Kedro pipeline nodes with upstream and downstream data flow</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Parallel Execution**: Validates thread safety when multiple Kedro nodes access FlyRigLoader datasets concurrently</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Memory Management**: Tests lazy evaluation and resource cleanup during pipeline execution to prevent memory leaks</span>

#### 6.6.3.7 Version Migration Integration Testing

<span style="background-color: rgba(91, 57, 243, 0.2)">Version migration testing ensures seamless configuration upgrades and backward compatibility across system versions:</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**Configuration Migration Workflows**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Automatic Version Detection**: Tests detection of configuration schema versions and compatibility validation during system startup</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Legacy Adapter Integration**: Validates LegacyConfigAdapter functionality for dictionary-based configurations and seamless Pydantic model conversion</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Migration Execution**: Tests automatic migration execution with rollback capability for failed migrations</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Schema Version Injection**: Validates automatic `schema_version` field injection for configurations lacking version metadata</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**Integration Pipeline Testing**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**End-to-End Migration**: Tests complete migration workflows from legacy configurations through modern Pydantic-based processing</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Configuration Builder Integration**: Validates `create_config()` builder pattern with automatic version handling and validation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Error Handling**: Tests migration error scenarios with detailed error reporting and recovery suggestions</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Version Compatibility Matrix**: Validates compatibility across all supported schema versions with comprehensive migration path testing</span>

#### 6.6.3.8 Registry Thread-Safety and Concurrency Testing

<span style="background-color: rgba(91, 57, 243, 0.2)">Registry concurrency testing validates thread-safe operations and plugin discovery under high-load scientific computing scenarios:</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**Concurrent Registry Operations**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**High-Concurrency Lookups**: Execute 100 concurrent registry lookups across LoaderRegistry and SchemaRegistry to validate thread safety and deadlock prevention</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Plugin Registration Stress**: Tests concurrent plugin registration and deregistration scenarios to ensure registry consistency</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Priority Resolution**: Validates priority-based plugin resolution (BUILTIN<USER<PLUGIN<OVERRIDE) under concurrent access patterns</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Lock Contention Testing**: Tests RWLock behavior with mixed read and write operations to identify performance bottlenecks</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**Plugin Discovery Integration**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Entry Point Discovery**: Tests automatic plugin discovery via setuptools entry points under concurrent initialization scenarios</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Decorator Registration**: Validates `@auto_register` decorator functionality with concurrent module imports and plugin registration</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Version Compatibility**: Tests plugin version validation and compatibility checking during concurrent registration attempts</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Registry Isolation**: Ensures plugin failures in one registry do not affect other registry operations or system stability</span>

#### 6.6.3.9 Performance Integration Testing

Performance integration testing validates system behavior under realistic research workflow conditions:

**Data Loading Performance**
- **Large File Handling**: Tests loading performance with files up to 1GB maintaining <2x memory overhead
- **Concurrent Discovery**: Validates file discovery performance with multiple simultaneous discovery operations
- **Registry Lookup Efficiency**: Tests O(1) registry lookup performance under load
- **Memory Profiling**: Monitors memory usage patterns during extended integration test runs

**Kedro Pipeline Performance**
- **Dataset Initialization**: Tests FlyRigLoaderDataSet initialization time within Kedro catalog context
- **Pipeline Throughput**: Measures end-to-end pipeline execution time with FlyRigLoader datasets
- **Resource Utilization**: Monitors CPU and memory usage during Kedro pipeline execution
- **Scaling Characteristics**: Tests performance degradation with increasing dataset sizes and node counts

#### 6.6.3.10 Error Recovery Integration Testing

Error recovery testing ensures robust behavior under failure conditions:

**Configuration Error Recovery**
- **Invalid YAML Handling**: Tests graceful degradation when configuration files contain syntax errors
- **Migration Failure Recovery**: Validates rollback capabilities when configuration migrations fail
- **Version Mismatch Handling**: Tests error handling when configuration versions are incompatible
- **Legacy Adapter Fallback**: Validates automatic fallback to legacy configuration processing

**Registry Error Isolation**
- **Plugin Failure Isolation**: Tests system stability when individual plugins fail during registration or execution
- **Registry Corruption Recovery**: Validates registry rebuilding capabilities after corruption scenarios
- **Concurrent Error Handling**: Tests error isolation when multiple threads encounter registry failures simultaneously
- **Graceful Degradation**: Ensures core functionality remains available when plugins fail

#### 6.6.3.11 Integration Test Execution Framework

**Test Infrastructure Configuration**
```python
# Integration test configuration example
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock", 
    "pytest_benchmark",
    "pytest_xdist"
]

#### Concurrent testing configuration
@pytest.fixture(scope="session")
def concurrent_registry_test():
    """Setup for 100 concurrent registry lookup tests"""
    return ConcurrentTestHarness(
        thread_count=100,
        operation_count=1000,
        timeout_seconds=30
    )
```

**Test Data Management**
- **Synthetic Dataset Generation**: Creates realistic experimental datasets for integration testing
- **Registry State Management**: Maintains isolated registry states across concurrent test execution
- **Temporary Environment Setup**: Creates isolated file system environments for each test scenario
- **Resource Cleanup**: Ensures complete cleanup of test resources to prevent test interference

**Continuous Integration Requirements**
- **Parallel Execution**: Integration tests run in parallel using pytest-xdist with worker isolation
- **Performance Benchmarks**: Automated benchmarking with historical performance comparison
- **Memory Leak Detection**: Continuous monitoring for memory leaks during extended test runs
- **Coverage Integration**: Integration test coverage reporting separate from unit test coverage

#### 6.6.3.12 Integration Test Success Criteria

**Functional Requirements**
- All integration tests pass with 100% success rate across supported Python versions (3.8-3.11)
- Kedro integration tests validate complete AbstractDataset interface compliance
- Version migration tests demonstrate zero-breaking-change upgrade paths
- Registry concurrency tests maintain data integrity under 100 concurrent operations

**Performance Requirements**
- File discovery completes in <5 seconds for 10,000 files during integration testing
- Data loading maintains <2x memory overhead for files up to 1GB
- Registry lookups maintain O(1) performance characteristics under concurrent load
- Kedro pipeline integration adds <10% overhead compared to native Kedro datasets

**Reliability Requirements**
- Zero integration test failures due to race conditions or timing issues
- Complete error recovery for all tested failure scenarios
- Registry thread safety maintained across all concurrency test scenarios
- Configuration migration success rate of 100% for all supported legacy formats

The integration testing strategy provides comprehensive validation of component interactions, ensuring FlyRigLoader maintains reliability, performance, and compatibility across all supported use cases and deployment scenarios.

### 6.6.4 END-TO-END TESTING STRATEGY

#### 6.6.4.1 E2E Test Scenarios

End-to-end tests validate complete scientific workflows:

- **Experiment Data Loading**: Complete workflow from configuration to DataFrame
- **Dataset Discovery**: Full file discovery with metadata extraction
- **Batch Processing**: Large dataset processing with memory management
- **Error Recovery**: Graceful handling of corrupted or missing files
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro Catalog Workflow**: Full pipeline execution using `FlyRigLoaderDataSet` from `catalog.yml`</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Legacy-to-Current Configuration Migration**: Load a v0.x YAML, trigger auto-migration, and run complete data-loading flow</span>

#### 6.6.4.2 Performance Testing Requirements

Performance tests ensure system meets scientific workflow requirements:

- **Discovery Performance**: <5 seconds for 10,000 files
- **Loading Performance**: <1 second per 100MB of data
- **Memory Efficiency**: <2x data size memory overhead
- **Transform Performance**: <100ms for manifest operations

#### 6.6.4.3 Test Data Setup and Teardown

E2E tests use realistic scientific datasets:

- **Synthetic Data Generation**: Hypothesis-based generation of realistic datasets
- **Reference Data**: Curated datasets representing common scientific scenarios
- **Performance Data**: Large datasets for stress testing
- **Cleanup Strategies**: Automated cleanup of test artifacts

#### 6.6.4.4 Cross-Platform Testing Strategy

Tests validate functionality across research computing environments:

- **Operating Systems**: Ubuntu, Windows, macOS testing through CI/CD
- **Python Versions**: Python 3.8-3.11 compatibility testing
- **Environment Variants**: Conda and pip installation testing
- **HPC Integration**: Batch processing environment validation

### 6.6.5 TEST AUTOMATION FRAMEWORK

#### 6.6.5.1 CI/CD Integration

The testing framework integrates with GitHub Actions for comprehensive automation:

- **Test Workflow**: `test.yml` executes comprehensive test suite<span style="background-color: rgba(91, 57, 243, 0.2)">, including execution of specialized test suites `test_kedro_integration.py`, `test_config_migration.py`, and `test_registry_threading.py`</span>
- **Quality Assurance**: `quality-assurance.yml` enforces code quality standards
- **Matrix Testing**: Multiple OS and Python version combinations<span style="background-color: rgba(91, 57, 243, 0.2)">, with additional *Extras* dimension: `{none, kedro}` – CI installs `flyrigloader[kedro]` for Kedro integration jobs</span>
- **Artifact Management**: Test reports and coverage data preservation

#### 6.6.5.2 Automated Test Triggers

Tests are triggered automatically on:

- **Pull Requests**: Full test suite execution with coverage reporting
- **Main Branch Commits**: Complete test suite with performance benchmarks
- **Release Preparation**: Extended test suite with compatibility validation
- **Scheduled Runs**: Daily regression testing with performance monitoring

#### 6.6.5.3 Parallel Test Execution

Performance optimization through parallel execution:

- **pytest-xdist**: Parallel test execution across CPU cores
- **Test Isolation**: Ensures tests can run independently
- **Resource Management**: Manages shared resources during parallel execution
- **Load Balancing**: Distributes test workload evenly

#### 6.6.5.4 Test Reporting Requirements

Comprehensive reporting for development and maintenance:

- **Coverage Reports**: HTML, XML, and JSON formats
- **Performance Reports**: Benchmarking results with historical comparison
- **Quality Reports**: Code quality metrics and violation reporting
- **CI Integration**: Results integrated with GitHub Actions and Codecov

#### 6.6.5.5 Failed Test Handling

Systematic approach to test failure management:

- **Immediate Notification**: Failed tests block pull request merging
- **Detailed Logging**: Comprehensive error reporting with context
- **Retry Logic**: Automatic retry for flaky tests with exponential backoff
- **Escalation Process**: Manual intervention triggers for persistent failures

#### 6.6.5.6 Flaky Test Management

Proactive management of test reliability:

- **Flaky Test Detection**: Automatic identification of unreliable tests
- **Quarantine System**: Isolation of flaky tests from main suite
- **Root Cause Analysis**: Systematic investigation of test instability
- **Stabilization Process**: Structured approach to test reliability improvement

### 6.6.6 QUALITY METRICS AND MONITORING

#### 6.6.6.1 Code Coverage Targets (updated)

Detailed coverage requirements ensure comprehensive testing:

| Component | Coverage Target | Branch Coverage | Notes |
|-----------|----------------|-----------------|-------|
| API Layer | 100% | Required | Critical public interface |
| Configuration | 100% | Required | Security-sensitive component |
| Discovery | 100% | Required | Core business logic |
| I/O Pipeline | 95% | Required | File operations complexity |
| Registry | 90% | Optional | Plugin system complexity |
| Utilities | 85% | Optional | Helper functions |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Migration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">100%</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Required</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Auto-migration reliability</span> |
| <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">90%</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Optional</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">External dependency coverage</span> |

#### 6.6.6.2 Performance Test Thresholds

Performance monitoring ensures system meets scientific requirements:

| Operation | Threshold | Measurement | Tolerance |
|-----------|-----------|-------------|-----------|
| File Discovery | <5s for 10k files | Wall time | ±10% |
| Data Loading | <1s per 100MB | Wall time | ±15% |
| Memory Usage | <2x data size | RSS memory | ±20% |
| Transform Speed | <100ms manifest | Wall time | ±5% |

#### 6.6.6.3 Test Success Rate Requirements

Quality gates ensure system reliability:

- **Unit Test Success**: 100% required for merge
- **Integration Test Success**: 100% required for merge
- **Performance Test Success**: 95% required for merge
- **E2E Test Success**: 90% required for merge

#### 6.6.6.4 Quality Gates Implementation

Automated quality enforcement through CI/CD:

- **Pre-commit Hooks**: Local quality checks before commits
- **Pull Request Gates**: Automated quality validation
- **Merge Requirements**: Quality thresholds must be met
- **Release Gates**: Extended quality validation for releases

#### 6.6.6.5 Component-Specific Quality Requirements

The expanded coverage targets reflect the system's architectural evolution and critical component priorities:

**Critical Components (100% Coverage Required)**
- **API Layer**: Public interface functions serve as the primary integration point for research workflows and require complete test coverage to prevent breaking changes
- **Configuration Layer**: Security-sensitive YAML parsing and validation logic demands comprehensive testing to prevent configuration injection attacks and data integrity issues
- **Discovery Engine**: Core file discovery and pattern matching logic represents the system's primary business value and requires exhaustive testing of edge cases
- **Migration System**: Version compatibility and automatic configuration migration capabilities are essential for zero-breaking-change upgrades in production research environments, requiring complete test coverage to ensure data integrity across schema versions

**High-Priority Components (90-95% Coverage)**
- **I/O Pipeline**: File loading operations involve complex error handling and memory management requiring extensive testing while allowing flexibility for platform-specific optimizations
- **Kedro Integration**: Optional AbstractDataset implementation provides enterprise pipeline integration but depends on external Kedro framework behavior, warranting high coverage while acknowledging external dependencies
- **Registry System**: Plugin discovery and registration mechanisms require comprehensive testing for thread safety and concurrent operations while allowing coverage flexibility for advanced plugin scenarios

**Supporting Components (85% Coverage)**
- **Utilities**: Helper functions and cross-cutting concerns benefit from solid test coverage while allowing flexibility for implementation-specific optimizations and platform variations

#### 6.6.6.6 Coverage Verification and Reporting

**Automated Coverage Analysis**
- **Branch Coverage Analysis**: All critical components enforce branch coverage analysis to identify untested conditional logic paths
- **Coverage Trend Monitoring**: Historical coverage tracking with regression detection for maintaining quality standards
- **Component-Level Reporting**: Individual component coverage reporting enabling targeted quality improvements
- **Integration Coverage**: Combined coverage analysis across component boundaries to identify integration testing gaps

**Quality Assurance Integration**
- **Pre-merge Validation**: Coverage requirements enforced through automated quality gates before code integration
- **Release Quality Gates**: Extended coverage validation during release preparation with component-specific thresholds
- **Regression Prevention**: Coverage baseline enforcement preventing quality degradation during development cycles
- **Documentation Synchronization**: Coverage requirements documented alongside component architecture ensuring alignment between quality standards and system design

### 6.6.7 SPECIALIZED TESTING STRATEGIES

#### 6.6.7.1 Property-Based Testing with Hypothesis

Advanced testing using Hypothesis for edge case discovery:

- **Configuration Testing**: Generate valid/invalid configuration combinations
- **Data Structure Testing**: Test data transformations with random inputs
- **File Pattern Testing**: Validate discovery patterns with generated filenames
- **Performance Testing**: Property-based performance validation
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Registry Concurrency**: Generate concurrent access patterns to assert thread-safe loader registration</span>

#### 6.6.7.2 Security Testing Requirements

Security validation for research data protection:

- **Path Traversal Testing**: Validate configuration path security
- **Input Validation Testing**: Test YAML parsing security
- **File Access Testing**: Validate file system access controls
- **Dependency Security**: Monitor for security vulnerabilities

#### 6.6.7.3 Benchmark Testing Strategy

Continuous performance monitoring with pytest-benchmark:

- **Micro-benchmarks**: Individual function performance
- **Macro-benchmarks**: Complete workflow performance
- **Memory Benchmarks**: Memory usage profiling
- **Regression Detection**: Performance degradation alerts

#### 6.6.7.4 Configuration Migration Testing (updated)

Advanced configuration compatibility validation ensures seamless system upgrades:

- <span style="background-color: rgba(91, 57, 243, 0.2)">**Configuration Migration Fuzzing**: Hypothesis strategies generate random legacy configs to validate migration stability</span>
- **Schema Validation Testing**: Test configuration schema evolution across versions
- **Backward Compatibility Testing**: Ensure legacy configurations continue functioning
- **Migration Rollback Testing**: Validate rollback capabilities for failed migrations

#### 6.6.7.5 Advanced Property-Based Strategies

Comprehensive Hypothesis-based testing covers complex system behaviors beyond standard unit testing:

**Registry Concurrency Validation**
The registry concurrency testing strategy employs sophisticated property-based generation to validate thread-safe operations under realistic scientific computing scenarios:

- **Concurrent Registration Patterns**: Generate sequences of plugin registration/deregistration operations across multiple threads to identify race conditions and deadlock scenarios
- **Load Pattern Generation**: Create diverse concurrent access patterns simulating real-world research workflow behaviors with mixed read/write operations
- **Priority Resolution Testing**: Generate plugin registration scenarios with varying priorities to validate consistent resolution order under concurrent access
- **Error Injection Testing**: Generate failure scenarios during concurrent operations to ensure graceful degradation and system stability

**Configuration Migration Robustness**
Configuration migration fuzzing leverages Hypothesis's strategic generation capabilities to ensure migration stability across all possible legacy configuration variants:

- **Legacy Schema Generation**: Generate syntactically valid configurations conforming to historical schema versions (v0.x through current) with randomized parameter combinations
- **Edge Case Discovery**: Create boundary conditions for configuration parameters including empty structures, maximum values, and invalid combinations that should trigger graceful error handling
- **Migration Path Validation**: Generate configuration sequences representing upgrade paths to ensure deterministic migration behavior across all supported version transitions
- **Compatibility Matrix Testing**: Generate cross-version compatibility scenarios to validate that migrations preserve functional equivalence while updating schema structures

#### 6.6.7.6 Specialized Test Data Generation

Property-based testing strategies require sophisticated data generation aligned with scientific computing domain requirements:

**Scientific Dataset Simulation**
- **Experimental Structure Generation**: Create realistic experimental metadata patterns including date ranges, subject identifiers, and protocol variations
- **File System Hierarchy Generation**: Generate directory structures representing various laboratory organization patterns
- **Configuration Parameter Space**: Systematic exploration of configuration parameter combinations within valid scientific ranges

**Performance Edge Case Generation**
- **Large Dataset Simulation**: Generate test scenarios with datasets approaching memory limits to validate performance characteristics
- **Concurrent Load Simulation**: Create concurrent access patterns representing multiple researchers accessing shared datasets simultaneously
- **Resource Constraint Simulation**: Generate scenarios under various memory and processing constraints typical of HPC environments

#### 6.6.7.7 Integration with Standard Testing Framework

Specialized testing strategies integrate seamlessly with the comprehensive testing framework described in previous sections:

**Test Execution Integration**
- **CI/CD Pipeline Integration**: Specialized tests execute as part of the standard CI/CD pipeline with appropriate resource allocation and timeout management
- **Performance Baseline Tracking**: Property-based performance tests contribute to the historical performance baseline used for regression detection
- **Coverage Integration**: Specialized tests contribute to overall coverage metrics while focusing on edge cases and concurrent scenarios not covered by traditional testing

**Quality Assurance Alignment**
- **Failure Classification**: Specialized test failures are classified and tracked separately to distinguish between edge case discoveries and regression issues
- **Documentation Requirements**: Property-based test discoveries are automatically documented to enhance system understanding and guide future development
- **Continuous Improvement**: Specialized testing results inform refinements to standard testing approaches and system architecture decisions

The specialized testing strategies provide comprehensive validation of system behavior under conditions that traditional testing approaches cannot easily simulate, ensuring FlyRigLoader maintains reliability and performance across the full spectrum of scientific computing scenarios.

### 6.6.8 TEST EXECUTION WORKFLOWS

#### 6.6.8.1 Test Execution Flow (updated)

```mermaid
flowchart TD
    A[Code Commit] --> B[Pre-commit Hooks]
    B --> C{Quality Checks Pass?}
    C -->|No| D[Reject Commit]
    C -->|Yes| IE{Install Optional Extras}
    
    IE -->|Kedro Tests Enabled| IF["Install flyrigloader[kedro]"]
    IE -->|Standard Tests| E[Trigger CI Pipeline]
    IF --> E
    
    E --> F[Unit Tests]
    F --> G[Integration Tests]
    G --> H[Performance Tests]
    H --> I[E2E Tests]
    
    I --> J{All Tests Pass?}
    J -->|No| K[Report Failures]
    J -->|Yes| L[Coverage Analysis]
    
    L --> M{Coverage Meets Targets?}
    M -->|No| N[Block Merge]
    M -->|Yes| O[Quality Gate Check]
    
    O --> P{Quality Standards Met?}
    P -->|No| Q[Require Fixes]
    P -->|Yes| R[Approve Merge]
```

#### 6.6.8.2 Test Environment Architecture (updated)

```mermaid
graph TB
    subgraph "Test Infrastructure"
        A[GitHub Actions] --> B[Test Matrix]
        B --> C[Ubuntu 20.04]
        B --> D[Windows 2019]
        B --> E[macOS 11]
        
        C --> F[Python 3.8-3.11]
        D --> F
        E --> F
        
        F --> G[pytest Runner]
        F --> KE[Kedro Enabled]
        
        G --> H[Coverage Analysis]
        G --> I[Performance Benchmarks]
        G --> J[Quality Reports]
        
        KE --> KG[pytest Runner + Kedro]
        KG --> KH[Kedro Integration Tests]
        KG --> KI[AbstractDataset Validation]
        KG --> KJ[Pipeline Execution Tests]
    end
    
    subgraph "Test Data Flow"
        K[Test Fixtures] --> L[Synthetic Data]
        K --> M[Reference Data]
        K --> N[Performance Data]
        
        L --> O[Unit Tests]
        M --> P[Integration Tests]
        N --> Q[Performance Tests]
        
        L --> KO[Kedro Unit Tests]
        M --> KP[Kedro Integration Tests]
        N --> KQ[Kedro Performance Tests]
    end
```

#### 6.6.8.3 Test Data Management Flow

```mermaid
sequenceDiagram
    participant TF as Test Framework
    participant FG as Fixture Generator
    participant DS as Data Store
    participant TS as Test Suite
    participant CR as Cleanup Registry
    
    TF->>FG: Request Test Data
    FG->>DS: Generate Synthetic Data
    DS-->>FG: Return Test Dataset
    FG-->>TF: Provide Test Fixtures
    
    TF->>TS: Execute Test Suite
    TS->>DS: Access Test Data
    DS-->>TS: Return Data
    TS-->>TF: Test Results
    
    TF->>CR: Register Cleanup
    CR->>DS: Remove Test Data
    DS-->>CR: Cleanup Complete
    CR-->>TF: Cleanup Confirmed
```

#### 6.6.8.4 Kedro Integration Test Workflow

<span style="background-color: rgba(91, 57, 243, 0.2)">The Kedro integration testing workflow ensures comprehensive validation of FlyRigLoader's AbstractDataSet implementation and pipeline compatibility across all supported environments.</span>

##### 6.6.8.4.1 Kedro Test Execution Strategy

<span style="background-color: rgba(91, 57, 243, 0.2)">**Installation and Environment Management**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Conditional Installation**: The CI pipeline automatically detects Kedro test requirements and installs `flyrigloader[kedro]` when Kedro integration tests are enabled</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Version Compatibility**: Tests execute against supported Kedro versions (1.0.0+) ensuring compatibility across the Kedro ecosystem</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Environment Isolation**: Kedro-enabled test runners operate in isolated environments preventing dependency conflicts with standard test execution</span>

<span style="background-color: rgba(91, 57, 243, 0.2)">**Test Suite Organization**</span>
The Kedro integration test suite is organized into specialized modules targeting specific integration aspects:

| Test Module | Focus Area | Execution Requirements |
|-------------|------------|----------------------|
| `test_kedro_datasets.py` | AbstractDataSet interface compliance | <span style="background-color: rgba(91, 57, 243, 0.2)">Requires kedro extras</span> |
| `test_kedro_catalog.py` | Catalog configuration and YAML parsing | <span style="background-color: rgba(91, 57, 243, 0.2)">Requires kedro extras</span> |
| `test_kedro_pipeline.py` | End-to-end pipeline integration | <span style="background-color: rgba(91, 57, 243, 0.2)">Requires kedro extras</span> |
| `test_kedro_performance.py` | Pipeline performance benchmarks | <span style="background-color: rgba(91, 57, 243, 0.2)">Requires kedro extras</span> |

##### 6.6.8.4.2 CI/CD Matrix Integration

<span style="background-color: rgba(91, 57, 243, 0.2)">The CI/CD matrix testing strategy incorporates Kedro integration as an additional test dimension alongside existing OS and Python version matrices:</span>

**Matrix Configuration**
```yaml
strategy:
  matrix:
    os: [ubuntu-20.04, windows-2019, macos-11]
    python-version: [3.8, 3.9, 3.10, 3.11]
    extras: [none, kedro]
```

<span style="background-color: rgba(91, 57, 243, 0.2)">**Execution Flow Integration**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Pre-execution Check**: Before test execution, the system evaluates the `extras` matrix parameter</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Conditional Installation**: When `extras=kedro`, the pipeline executes `pip install -e .[kedro]` to enable Kedro integration</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Test Suite Selection**: Kedro-enabled runners execute the complete test suite including specialized Kedro integration tests</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Coverage Integration**: Kedro integration test coverage contributes to overall system coverage metrics while maintaining separate reporting for optional components</span>

##### 6.6.8.4.3 Test Data and Environment Management

<span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro-Specific Test Data**</span>
The Kedro integration testing framework requires specialized test data that mimics realistic Kedro pipeline scenarios:

- <span style="background-color: rgba(91, 57, 243, 0.2)">**Catalog Configuration Fixtures**: Sample `catalog.yml` files with FlyRigLoaderDataSet configurations for integration testing</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Pipeline Definition Fixtures**: Test pipeline definitions incorporating FlyRigLoader datasets with upstream and downstream dependencies</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Parameter Injection Fixtures**: Test data for Kedro parameter injection and configuration variable substitution</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Performance Benchmark Data**: Large datasets for testing Kedro pipeline performance with FlyRigLoader integration</span>

**Environment Configuration**
```python
@pytest.fixture(scope="session")
def kedro_enabled_environment():
    """Setup Kedro-enabled test environment with proper isolation."""
    return KedroTestEnvironment(
        temp_dir=tmp_path_factory.mktemp("kedro"),
        catalog_config=load_test_catalog(),
        pipeline_config=load_test_pipeline(),
        cleanup_on_exit=True
    )
```

##### 6.6.8.4.4 Performance and Quality Validation

<span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro Integration Performance Metrics**</span>

| Metric | Standard Target | Kedro Integration Target | Tolerance |
|--------|----------------|-------------------------|-----------|
| Dataset Initialization | <50ms | <100ms | ±20% |
| Pipeline Node Execution | Standard + 10% | Standard + 15% | ±25% |
| Memory Overhead | <2x data size | <2.5x data size | ±30% |
| Catalog Loading | N/A | <500ms for 100 datasets | ±15% |

<span style="background-color: rgba(91, 57, 243, 0.2)">**Quality Assurance Integration**</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Compatibility Validation**: Automated validation of Kedro version compatibility across supported versions</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**API Compliance Testing**: Comprehensive validation of AbstractDataSet interface implementation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Integration Regression Testing**: Automated detection of integration regressions through historical performance comparison</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Documentation Synchronization**: Automatic validation that Kedro integration documentation remains current with implementation</span>

#### 6.6.8.5 Test Execution Orchestration and Resource Management

**Parallel Execution Strategy**
The test execution workflow incorporates sophisticated orchestration to optimize resource utilization while maintaining test isolation:

- **Test Suite Partitioning**: Tests are partitioned across available CPU cores using pytest-xdist with load balancing
- **Resource Pool Management**: Database connections, file handles, and temporary directories are managed through resource pools
- **Memory Management**: Large dataset tests are scheduled separately to prevent memory contention
- **Dependency Ordering**: Tests with external dependencies are executed after isolated unit tests

**Execution Monitoring and Reporting**
The workflow provides comprehensive monitoring and reporting capabilities:

- **Real-time Progress Tracking**: Live updates on test execution progress with estimated completion times
- **Resource Usage Monitoring**: CPU, memory, and disk usage tracking throughout test execution
- **Failure Analysis**: Automatic categorization of test failures with detailed error reporting
- **Performance Regression Detection**: Continuous comparison with historical performance baselines

**Error Recovery and Resilience**
The test execution workflow incorporates robust error recovery mechanisms:

- **Automatic Retry Logic**: Flaky tests are automatically retried with exponential backoff
- **Graceful Degradation**: Test failures are isolated to prevent cascade failures
- **Resource Cleanup**: Comprehensive cleanup of test resources even during abnormal termination
- **State Recovery**: Ability to resume test execution from checkpoint during infrastructure failures

#### 6.6.8.6 Cross-Platform Testing Validation

**Operating System Compatibility**
The test execution workflow validates functionality across research computing environments:

- **Path Handling**: Cross-platform path operations testing for Windows, macOS, and Linux
- **File System Behavior**: Platform-specific file system behavior validation
- **Environment Variables**: Cross-platform environment variable handling
- **Process Management**: Platform-specific process spawning and management testing

**Python Version Compatibility**
Comprehensive testing across supported Python versions ensures broad compatibility:

- **Syntax Compatibility**: Validation of Python 3.8+ syntax compliance
- **Library Compatibility**: Testing with different versions of scientific Python libraries
- **Performance Characteristics**: Version-specific performance profiling and optimization
- **Migration Path Validation**: Testing upgrade paths between Python versions

**Development Environment Integration**
The workflow validates integration with common development environments:

- **IDE Integration**: Testing with popular scientific Python IDEs
- **Jupyter Notebook Compatibility**: Interactive testing environment validation
- **HPC Environment Testing**: Validation in high-performance computing environments
- **Container Integration**: Docker and Singularity container testing validation

This comprehensive test execution workflow ensures FlyRigLoader maintains reliability, performance, and compatibility across all supported use cases while providing the foundation for continuous quality improvement and feature development. <span style="background-color: rgba(91, 57, 243, 0.2)">The integration of Kedro-specific testing capabilities ensures that the optional Kedro integration maintains the same quality standards as the core system while providing specialized validation for enterprise data pipeline scenarios.</span>

### 6.6.9 REFERENCES

#### Files Examined
- `tests/coverage/report-config.json` - JSON configuration for coverage reporting system
- `tests/coverage/pytest.ini` - Enterprise-grade pytest configuration
- `tests/coverage/quality-gates.yml` - Automated quality gate rules
- `tests/coverage/coverage-thresholds.json` - Per-module coverage thresholds
- `tests/coverage/validate-coverage.py` - Coverage validation script
- `.github/workflows/test.yml` - Comprehensive test suite workflow
- `.github/workflows/quality-assurance.yml` - Quality assurance workflow
- `pyproject.toml` - Project configuration with pytest and coverage settings
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_registry_threading.py` - Registry thread-safety tests</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_kedro_integration.py` - Kedro dataset integration tests</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_config_migration.py` - Configuration migration tests</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_config_builders.py` - Builder pattern tests</span>

#### Folders Explored
- `tests/` - Main test directory structure
- `tests/coverage/` - Coverage reporting infrastructure
- `tests/flyrigloader/` - Main test suite organization
- `tests/flyrigloader/benchmarks/` - Performance testing suite
- `tests/coverage/templates/` - Jinja2 templates for reports
- `tests/coverage/scripts/` - CI/CD orchestration scripts
- `.github/workflows/` - GitHub Actions CI/CD pipelines
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/kedro/` - Kedro integration test modules</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/migration/` - Configuration migration test suite</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/registries/` - Registry pattern test modules</span>

#### Technical Specification Sections Referenced
- `4.5 TECHNICAL IMPLEMENTATION DETAILS` - Performance targets and optimization strategies
- `5.1 HIGH-LEVEL ARCHITECTURE` - System architecture and component relationships
- `5.2 COMPONENT DETAILS` - Detailed component specifications for testing
- `3.2 FRAMEWORKS & LIBRARIES` - Testing frameworks and library dependencies

# 7. USER INTERFACE DESIGN

## 7.1 ARCHITECTURAL CONTEXT

FlyRigLoader is designed as a **pure Python library** for scientific data management in neuroscience research. The system architecture explicitly prioritizes programmatic access over user interface components, aligning with its core mission as a data processing and integration library for the academic research ecosystem.

## 7.2 INTEGRATION PATTERNS

The system provides user functionality through multiple programmatic interfaces:

### 7.2.1 API Layer Integration
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Pipelines & Catalog</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Direct function calls and catalog-driven loading via `FlyRigLoaderDataSet` for production data workflows</span> <span style="background-color: rgba(91, 57, 243, 0.2)">(optional install: `pip install flyrigloader[kedro]`)</span>
- **Jupyter Notebooks**: Interactive research and exploration through Python API
- **HPC Environments**: Batch processing with file-based data exchange
- **CLI Tools**: Command-line interface for automated workflows

<span style="background-color: rgba(91, 57, 243, 0.2)">All public functions in the Python API are implemented as stateless, pure functions to maximize testability and ensure predictable behavior across diverse research environments.</span>

### 7.2.2 User Interaction Model
User interactions occur exclusively through:
- **Python API calls**: Functions in `flyrigloader.api` module
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Configuration models</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Pydantic-backed builder-generated models (with transparent YAML legacy adapter for backward compatibility)</span> <span style="background-color: rgba(91, 57, 243, 0.2)">(includes `schema_version` field with automatic migration support)</span>
- **Command-line interface**: Script-based automation tools
- **Programmatic imports**: Direct library usage in research codebases

### 7.2.3 Integration Architecture Overview

The integration patterns are designed around FlyRigLoader's core mission as a pure Python library for neuroscience data workflows. The system emphasizes programmatic access through well-defined interfaces rather than user interface components, aligning with the research-focused scientific computing environment.

```mermaid
graph TB
    subgraph "Research Integration Points"
        K[Kedro Pipeline Integration] --> FDS[FlyRigLoaderDataSet]
        J[Jupyter Research Environment] --> API[Python API Layer]
        C[CLI Automation Scripts] --> API
        H[HPC Computing Clusters] --> API
        
        FDS --> API
    end
    
    subgraph "Configuration Management"
        PY_CONFIG[Pydantic Configuration Models] --> API
        YAML_LEGACY[YAML Legacy Adapter] --> PY_CONFIG
        BUILDER[Config Builder Functions] --> PY_CONFIG
    end
    
    subgraph "Core Processing Pipeline"
        API --> DISCOVERY[File Discovery Engine]
        API --> LOADING[Data Loading Registry]
        API --> TRANSFORM[DataFrame Transformation]
        
        DISCOVERY --> MANIFEST[File Manifest]
        LOADING --> DATAFRAME[Pandas DataFrame Output]
        TRANSFORM --> DATAFRAME
    end
```

### 7.2.4 Kedro Pipeline Integration Patterns

The system provides first-class Kedro integration through the `FlyRigLoaderDataSet` implementation, enabling seamless integration with production data science workflows:

**Catalog Configuration Example:**
```yaml
experiment_data:
  type: flyrigloader.FlyRigLoaderDataSet
  filepath: "${base_dir}/config/experiment_config.yaml"
  experiment_name: "baseline_study"
  recursive: true
  extract_metadata: true
```

**Factory Function Integration:**
```python
from flyrigloader.api import create_kedro_dataset

dataset = create_kedro_dataset(
    config_path="configs/experiment_config.yaml",
    experiment_name="behavioral_training",
    recursive=True
)
```

The Kedro integration maintains full compatibility with Kedro's data versioning, lineage tracking, and pipeline execution framework while preserving FlyRigLoader's core performance characteristics and discovery capabilities.

### 7.2.5 Configuration System Integration

The modernized configuration system provides multiple integration pathways to accommodate different workflow preferences:

**Builder Pattern (Recommended):**
```python
from flyrigloader.api import create_config

config = create_config(
    experiment_path="/data/experiments",
    experiment_name="baseline_study",
    file_patterns=["*.pkl"],
    schema_version="1.2.0"
)
```

**Legacy YAML Support:**
```yaml
# experiment_config.yaml
schema_version: "1.2.0"
project:
  major_data_directory: "/data/experiments"
experiments:
  baseline_study:
    date_range: ["2024-01-01", "2024-01-31"]
    rig_names: ["rig1", "rig2"]
```

The system automatically detects configuration versions and applies necessary migrations, ensuring zero-breaking-change upgrades for existing research workflows while providing enhanced validation and type safety for new implementations.

### 7.2.6 API Integration Best Practices

For optimal integration with FlyRigLoader, researchers and pipeline developers should:

1. **Version Awareness**: Always specify `schema_version` in configurations to enable automatic migration and ensure compatibility
2. **Type Safety**: Prefer Pydantic configuration models over dictionary-based configurations for enhanced validation
3. **Stateless Design**: Leverage the pure function design for predictable behavior and simplified testing
4. **Optional Dependencies**: Install Kedro support only when needed: `pip install flyrigloader[kedro]`
5. **Error Handling**: Implement comprehensive exception handling for file discovery and loading operations
6. **Performance Optimization**: Use manifest-based discovery for selective data loading in large experiments

### 7.2.7 Extension Points

The integration architecture supports extensibility through:

- **Plugin Registry**: Automatic discovery of custom loaders and schemas through entry points
- **Protocol-Based Interfaces**: Clean dependency injection for testing and customization
- **Configuration Validation**: Schema-based validation with custom validation rules
- **Registry Priority System**: Controlled override mechanisms for specialized research environments

These extension points enable researchers to adapt FlyRigLoader to specific experimental setups and data formats while maintaining compatibility with the broader scientific Python ecosystem.

## 7.3 DESIGN RATIONALE

### 7.3.1 Library-First Architecture
The decision to exclude a graphical user interface reflects several strategic considerations:

- **Research Workflow Integration**: Academic researchers primarily work through programmatic interfaces (Jupyter notebooks, Python scripts, HPC job submissions)
- **Pipeline Integration**: The system is designed to be a component within larger data processing workflows rather than a standalone application
- **Resource Efficiency**: Eliminating UI overhead enables optimal performance for large-scale data processing tasks
- **Collaborative Research**: Configuration-as-code approach supports version control and reproducible research practices

### 7.3.2 Target User Profile
The system serves:
- **Data Scientists**: Using Python scripts and Jupyter notebooks for analysis
- **Research Engineers**: Integrating with Kedro pipelines and HPC systems
- **Academic Researchers**: Requiring programmatic access for reproducible workflows
- **Pipeline Developers**: Building automated data processing systems

## 7.4 FUTURE CONSIDERATIONS

While no user interface is currently planned, the system architecture supports potential future extensions:

### 7.4.1 Potential UI Integration Points
- **Kedro-Viz Integration**: Visualization of data processing pipelines
- **Jupyter Lab Extensions**: Enhanced notebook widgets for data exploration
- **Web Dashboard**: Optional monitoring interface for batch processing workflows
- **Configuration Generators**: Web-based tools for creating YAML configurations

### 7.4.2 Architectural Compatibility
The layered architecture and protocol-based design would support UI additions without core modifications:
- **API Layer**: Already provides clean interfaces for potential UI integration
- **Configuration Layer**: Schema-validated models ready for form-based editing
- **Plugin Architecture**: Could support UI plugin development
- **Error Handling**: Structured error responses suitable for UI display

## 7.5 REFERENCES

#### Technical Specification Sections Referenced
- `1.2 SYSTEM OVERVIEW` - Confirmed integration with Kedro pipelines, Jupyter notebooks, and HPC environments
- `2.1 FEATURE CATALOG` - Verified no UI-related features among F-001 through F-010
- `3.2 FRAMEWORKS & LIBRARIES` - Confirmed absence of web/UI frameworks in technology stack
- `5.1 HIGH-LEVEL ARCHITECTURE` - Validated layered architecture with programmatic interfaces only

#### Repository Analysis
- **API Layer**: `src/flyrigloader/api.py` - Exclusively programmatic functions
- **Examples Directory**: `examples/` - All CLI-based usage examples
- **Configuration System**: YAML-based with no UI configuration tools
- **Integration Points**: Kedro, Jupyter, HPC environments only

# 8. INFRASTRUCTURE

## 8.1 INFRASTRUCTURE APPLICABILITY ASSESSMENT

### 8.1.1 System Infrastructure Classification

**Detailed Infrastructure Architecture is not applicable for this system.**

FlyRigLoader operates as a standalone Python library designed for academic neuroscience research, specifically for managing and loading data from opto rigs used in fly experiments. Unlike traditional web services or distributed applications, this system requires minimal infrastructure focused on development, testing, and package distribution rather than runtime deployment.

### 8.1.2 Infrastructure Requirements Analysis

| Infrastructure Aspect | Traditional Applications | FlyRigLoader Library |
|----------------------|-------------------------|---------------------|
| **Runtime Environment** | Dedicated servers/containers | User's local Python environment |
| **Deployment Model** | Service orchestration | Package installation via pip/conda |
| **Resource Management** | Container orchestration | Local system resources |
| **Network Architecture** | Load balancers, API gateways | File system access only |
| **Data Storage** | Database clusters | Local file system operations |
| **Scalability** | Horizontal scaling | Per-user computational resources |

### 8.1.3 Minimal Infrastructure Scope

The system's infrastructure requirements are limited to:

- **Build and Distribution**: Package creation and PyPI distribution
- **Development Environment**: Conda-based environment management
- **CI/CD Pipeline**: Automated testing and quality assurance
- **Quality Monitoring**: Coverage reporting and performance validation
- **Documentation**: Build and distribution of technical documentation

## 8.2 BUILD AND DISTRIBUTION INFRASTRUCTURE

### 8.2.1 Package Distribution Architecture

#### 8.2.1.1 Primary Distribution Channel

**PyPI (Python Package Index) Distribution**:
- **Registry**: PyPI for public package distribution
- **Package Format**: Python wheel (.whl) and source distribution (.tar.gz)
- **Version Management**: Semantic versioning (currently <span style="background-color: rgba(91, 57, 243, 0.2)">1.0.0-rc1</span>)
- **Installation Method**: `pip install flyrigloader`
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Optional Extras Installation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">`pip install flyrigloader[kedro]` for Kedro pipeline integration</span>

#### 8.2.1.2 Build System Configuration

**PEP 517/518 Compliant Build System**:
```toml
[build-system]
requires = ["setuptools>=42", "wheel", <span style="background-color: rgba(91, 57, 243, 0.2)">"semantic_version>=2.10.0"</span>]
build-backend = "setuptools.build_meta"
```

**Package Metadata and Dependencies**:
- **Minimum Python Version**: 3.8
- **Core Dependencies**: loguru>=0.7.0, pydantic>=2.6, numpy>=1.21.0, pandas>=1.3.0<span style="background-color: rgba(91, 57, 243, 0.2)">, semantic_version>=2.10.0</span>
- **Build Configuration**: Defined in `pyproject.toml`
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Versioning Compliance</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Follows semantic versioning (SemVer) specifications with structured deprecation periods for breaking changes</span>

#### 8.2.1.3 Alternative Distribution Methods

| Distribution Method | Use Case | Implementation |
|-------------------|----------|----------------|
| **Conda Package** | Scientific computing environments | Future consideration via conda-forge |
| **Git Installation** | Development and testing | `pip install git+https://github.com/...` |
| **Local Editable** | Development workflow | `pip install -e .` |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Extras Install (Kedro)</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Production data pipeline workflows</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">`pip install flyrigloader[kedro]`</span>** |

### 8.2.2 Environment Management Infrastructure

#### 8.2.2.1 Conda Environment System

**Primary Environment Management**:
- **Environment Definition**: `environment.yml` specifies complete development environment
- **Environment Name**: `flyrigloader-dev`
- **Channels**: conda-forge and defaults for scientific Python packages
- **Automation Scripts**: `setup_env.sh` and `activate_env.sh` for environment lifecycle

**Environment Configuration Matrix**:

| Component | Version Specification | Channel | Purpose |
|-----------|----------------------|---------|---------|
| **Python** | >=3.8 | conda-forge | Runtime environment |
| **Core Dependencies** | Pinned versions | conda-forge | Library functionality |
| **Development Tools** | Latest compatible | conda-forge | Code quality and testing |
| **Documentation Tools** | Latest compatible | conda-forge | Documentation generation |

#### 8.2.2.2 Environment Automation

**Automated Environment Setup**:
```bash
# setup_env.sh - Environment provisioning
#!/bin/bash
conda env create -f environment.yml
conda activate flyrigloader-dev
pip install -e .
```

**Environment Activation**:
```bash
# activate_env.sh - Environment activation helper
#!/bin/bash
conda activate flyrigloader-dev
```

```mermaid
flowchart TD
    subgraph "Environment Management Infrastructure"
        A[environment.yml] --> B[setup_env.sh]
        B --> C[Conda Environment Creation]
        C --> D[flyrigloader-dev Environment]
        D --> E[activate_env.sh]
        E --> F[Development Session]
        
        G[pyproject.toml] --> H[pip install -e .]
        H --> D
        
        I[requirements.txt] --> J[Production Dependencies]
        J --> K[User Installation]
    end
```

### 8.2.3 Version Management and Release Strategy (updated)

#### 8.2.3.1 Semantic Versioning Implementation

**<span style="background-color: rgba(91, 57, 243, 0.2)">Version Management Framework</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Current Release</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">1.0.0-rc1 (Release Candidate)</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">SemVer Compliance</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Strict adherence to semantic versioning specifications</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Breaking Change Policy</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Major version increments with 6-month deprecation periods</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Schema Version Integration</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated compatibility validation using semantic_version library</span>

#### 8.2.3.2 Distribution Channel Strategy

**<span style="background-color: rgba(91, 57, 243, 0.2)">Release Channel Management</span>**:

| Release Type | Version Pattern | Distribution | Stability |
|-------------|----------------|--------------|-----------|
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Release Candidate</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">X.Y.Z-rcN</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">PyPI with pre-release flag</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Feature-complete, pre-production</span>** |
| **Stable Release** | X.Y.Z | PyPI standard | Production-ready |
| **Patch Release** | X.Y.Z+1 | PyPI standard | Bug fixes only |
| **Development** | X.Y.Z.devN | Git installation | Active development |

**<span style="background-color: rgba(91, 57, 243, 0.2)">Upgrade Path Management</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Backward Compatibility</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Minor and patch versions maintain full API compatibility</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Support</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated configuration migration tools for major version transitions</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Deprecation Warnings</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Clear deprecation timeline communication through structured warning system</span>

## 8.3 CI/CD PIPELINE

### 8.3.1 GitHub Actions Workflow Infrastructure

#### 8.3.1.1 Primary Test Pipeline (updated)

**Comprehensive Test Suite** (`.github/workflows/test.yml`):
- **Triggers**: Push/PR to main/develop branches, manual dispatch
- **Matrix Testing**: Ubuntu/Windows/macOS × Python 3.8-3.11 <span style="background-color: rgba(91, 57, 243, 0.2)">× Extras {none, kedro}</span>
- **Parallel Execution**: Multiple jobs for efficiency and isolation

**Pipeline Job Architecture** (updated):

| Job | Purpose | Dependencies | Artifacts |
|-----|---------|--------------|-----------|
| **code-quality** | Formatting, linting, type checking, security | None | Quality reports |
| **test-matrix** | Multi-platform unit tests | code-quality | Coverage reports |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">registry-thread-safety</span>** | <span style="background-color: rgba(91, 57, 243, 0.2)">100-concurrency registry validation</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">test-matrix</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safety coverage</span> |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">kedro-integration</span>** | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro pipeline integration testing</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">test-matrix</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration coverage</span> |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">migration-tests</span>** | <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration migration validation</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">test-matrix</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Migration coverage</span> |
| **integration-tests** | End-to-end workflow validation | test-matrix | Integration results |
| **performance-validation** | Benchmark SLA verification | test-matrix | Performance metrics |
| **quality-gate** | <span style="background-color: rgba(91, 57, 243, 0.2)">Artifact aggregation and validation (updated)</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">All previous (including new jobs)</span> | Final validation |
| **notify-results** | PR status notifications | quality-gate | Status updates |

#### 8.3.1.2 Specialized Test Job Configuration (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Registry Thread-Safety Job</span>** (`.github/workflows/test.yml`):
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Purpose</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Execute high-concurrency registry operations to validate thread safety under realistic scientific computing load patterns</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Concurrency Level</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">100 concurrent registry lookups across LoaderRegistry and SchemaRegistry</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Test Suite</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_registry_threading.py`</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Coverage Upload</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safety test coverage contributes to global 90% threshold</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration Job</span>** (`.github/workflows/test.yml`):
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Purpose</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Validate FlyRigLoaderDataSet implementation and Kedro pipeline integration</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Installation Strategy</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">`pip install .[kedro]` for optional dependency handling</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Pipeline Execution</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Spins up complete Kedro pipelines using FlyRigLoader datasets</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Test Suite</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_kedro_integration.py`</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Coverage Upload</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration coverage maintains overall quality metrics</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Tests Job</span>** (`.github/workflows/test.yml`):
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Purpose</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Validate automatic configuration migration paths across schema versions</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Validation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Tests legacy-to-current configuration upgrade workflows</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Compatibility Testing</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Validates zero-breaking-change upgrade paths across all supported versions</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Test Suite</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_config_migration.py`</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Coverage Upload</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Migration test coverage essential for overall system reliability</span>

#### 8.3.1.3 Quality Assurance Pipeline

**Quality Gates** (`.github/workflows/quality-assurance.yml`):
- **Purpose**: Enforce merge-blocking quality standards
- **Execution**: Parallel with main test pipeline
- **Scope**: Type checking, formatting, linting, pre-commit validation

**Quality Gate Matrix**:

| Quality Gate | Tool | Threshold | Enforcement |
|--------------|------|-----------|-------------|
| **Type Safety** | mypy | 100% type coverage | Merge blocking |
| **Code Formatting** | black, isort | Zero violations | Merge blocking |
| **Linting** | flake8 + plugins | Zero violations | Merge blocking |
| **Security Scanning** | bandit | No high/medium issues | Merge blocking |
| **Coverage** | pytest-cov | 90% minimum | Merge blocking |

#### 8.3.1.4 External Service Integration

**CI/CD Service Dependencies**:
- **GitHub Actions Runners**: ubuntu-latest, windows-latest, macos-latest
- **Codecov**: Coverage reporting and trend analysis (requires CODECOV_TOKEN)
- **PyPI**: Package distribution (requires PYPI_TOKEN for future releases)

```mermaid
flowchart TD
    subgraph "CI/CD Pipeline Infrastructure (updated)"
        subgraph "Triggers"
            T1[Push to main] --> W1[test.yml]
            T2[Push to develop] --> W1
            T3[Pull Request] --> W1
            T4[Manual Dispatch] --> W1
        end
        
        subgraph "Test Pipeline"
            W1 --> J1[code-quality]
            J1 --> J2[test-matrix]
            J2 --> J3[integration-tests]
            J2 --> J4[performance-validation]
            J2 --> J5[registry-thread-safety]
            J2 --> J6[kedro-integration]
            J2 --> J7[migration-tests]
            J3 --> J8[quality-gate]
            J4 --> J8
            J5 --> J8
            J6 --> J8
            J7 --> J8
            J8 --> J9[notify-results]
        end
        
        subgraph "Quality Pipeline"
            W1 --> Q1[quality-assurance.yml]
            Q1 --> Q2[Parallel Quality Gates]
            Q2 --> Q3[Merge Decision]
        end
        
        subgraph "External Services"
            J2 --> C1[Codecov]
            J5 --> C1
            J6 --> C1
            J7 --> C1
            J8 --> C2[PR Comments]
            Q3 --> C3[Branch Protection]
        end
    end
```

### 8.3.2 Build Pipeline Architecture

#### 8.3.2.1 Source Control Integration

**Version Control Workflow**:
- **Repository**: GitHub with branch protection rules
- **Branching Strategy**: GitFlow with main/develop branches
- **Merge Requirements**: All CI/CD checks must pass
- **Automated Triggers**: Push and PR events initiate full pipeline

#### 8.3.2.2 Artifact Generation (updated)

**Build Artifacts**:
- **Python Packages**: Wheel and source distributions
- **Coverage Reports**: HTML, XML, and JSON formats <span style="background-color: rgba(91, 57, 243, 0.2)">including specialized test coverage</span>
- **Performance Metrics**: Benchmark results and trend analysis
- **Documentation**: Generated API documentation and guides
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Thread-Safety Reports</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Concurrency validation results and performance characteristics</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Integration Test Results</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro pipeline execution logs and compatibility validation</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Validation Reports</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration upgrade path verification and compatibility matrices</span>

#### 8.3.2.3 Quality Gates and Validation (updated)

**Merge Protection Requirements**:
- All GitHub Actions workflows must pass
- Code coverage must meet 90% threshold <span style="background-color: rgba(91, 57, 243, 0.2)">across all test suites</span>
- No security vulnerabilities detected
- All quality gates passed
- <span style="background-color: rgba(91, 57, 243, 0.2)">Registry thread-safety validation successful</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration tests pass (when extras enabled)</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration migration tests validate all upgrade paths</span>
- At least one approving review required

**<span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced Quality Gate Aggregation</span>**:
The quality-gate job now aggregates artifacts from all test suites including the new specialized jobs:

- **<span style="background-color: rgba(91, 57, 243, 0.2)">Coverage Consolidation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Combines coverage data from registry-thread-safety, kedro-integration, and migration-tests jobs</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Performance Validation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Validates that thread-safety testing maintains acceptable performance characteristics</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Integration Compliance</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Ensures Kedro integration maintains AbstractDataset interface compliance</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Reliability</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Validates that configuration migrations maintain 100% success rate</span>

### 8.3.3 Deployment Pipeline

#### 8.3.3.1 Package Release Process

**Release Strategy**:
- **Manual Release**: Triggered by maintainers for version releases
- **Automated Validation**: All quality gates must pass before release
- **Distribution**: Automated upload to PyPI upon release tag creation
- **Documentation**: Automatic documentation update and deployment

#### 8.3.3.2 Environment Promotion (updated)

**Development to Production Flow**:
```mermaid
flowchart LR
    subgraph "Environment Promotion (updated)"
        D1[Developer Branch] --> D2[develop Branch]
        D2 --> D3[CI/CD Validation]
        D3 --> D4{Specialized Tests Pass?}
        D4 -->|Registry + Kedro + Migration| D5[main Branch]
        D4 -->|Failed| D6[Block Merge]
        D5 --> D7[Release Tag]
        D7 --> D8[PyPI Distribution]
        D8 --> D9[User Installation]
        D8 --> D10["pip install flyrigloader[kedro]"]
    end
```

#### 8.3.3.3 Matrix Testing Strategy (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced Testing Matrix</span>**:
The CI/CD pipeline now supports a three-dimensional testing matrix to accommodate optional dependencies:

**Matrix Dimensions**:
- **Operating Systems**: Ubuntu 20.04, Windows 2019, macOS 11
- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Extras Configuration</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">{none, kedro}</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Conditional Job Execution</span>**:
```yaml
strategy:
  matrix:
    os: [ubuntu-20.04, windows-2019, macos-11]
    python-version: [3.8, 3.9, 3.10, 3.11]
    extras: [none, kedro]
```

**<span style="background-color: rgba(91, 57, 243, 0.2)">Installation Logic</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Standard Installation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">`pip install -e .` for basic functionality testing</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro-Enabled Installation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">`pip install -e .[kedro]` when matrix includes kedro extras</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Conditional Test Execution</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro integration tests only run when appropriate extras are installed</span>

### 8.3.4 Pipeline Performance and Optimization

#### 8.3.4.1 Parallel Execution Strategy (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Enhanced Job Parallelization</span>**:
The addition of three new specialized test jobs maintains efficient parallel execution through strategic dependency management:

- **<span style="background-color: rgba(91, 57, 243, 0.2)">Parallel Specialized Jobs</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">registry-thread-safety, kedro-integration, and migration-tests execute concurrently after test-matrix completion</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Resource Isolation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Each specialized job runs in isolated environments to prevent test interference</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Conditional Execution</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Jobs execute conditionally based on matrix configuration and code changes</span>

#### 8.3.4.2 Resource Management

**Pipeline Resource Allocation**:
- **CPU Utilization**: Parallel jobs distributed across available runners
- **Memory Management**: Large dataset tests scheduled to prevent memory contention
- **Artifact Storage**: Efficient storage and retrieval of test results and coverage data
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Concurrency Testing Resources</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Additional CPU allocation for 100-thread concurrency validation</span>

#### 8.3.4.3 Pipeline Execution Metrics

**Performance Targets**:

| Pipeline Phase | Target Duration | Tolerance | Notes |
|----------------|----------------|-----------|-------|
| **code-quality** | <3 minutes | ±30 seconds | Static analysis and linting |
| **test-matrix** | <15 minutes | ±5 minutes | Core test suite execution |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">registry-thread-safety</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)"><8 minutes</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">±2 minutes</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">100-concurrency validation</span>** |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">kedro-integration</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)"><12 minutes</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">±3 minutes</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Pipeline execution testing</span>** |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">migration-tests</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)"><6 minutes</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">±90 seconds</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Configuration migration validation</span>** |
| **integration-tests** | <10 minutes | ±3 minutes | End-to-end workflow validation |
| **performance-validation** | <8 minutes | ±2 minutes | Benchmark execution |
| **quality-gate** | <2 minutes | ±30 seconds | <span style="background-color: rgba(91, 57, 243, 0.2)">Artifact aggregation (updated)</span> |

### 8.3.5 Monitoring and Observability

#### 8.3.5.1 Pipeline Health Monitoring

**Automated Monitoring**:
- **Build Success Rate**: Historical tracking of pipeline success rates
- **Performance Regression Detection**: Automated alerts for performance degradation
- **Coverage Trend Analysis**: Continuous monitoring of test coverage evolution
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Specialized Test Reliability</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Monitoring of new test job stability and execution patterns</span>

#### 8.3.5.2 Quality Metrics Dashboard

**Key Performance Indicators**:
- **Pipeline Execution Time**: End-to-end pipeline duration tracking
- **Test Coverage Metrics**: Component-level coverage analysis with <span style="background-color: rgba(91, 57, 243, 0.2)">specialized test contribution</span>
- **Security Vulnerability Detection**: Automated security issue identification and reporting
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Concurrency Validation Success</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Thread-safety test reliability and performance characteristics</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Integration Test Health</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Kedro pipeline compatibility and AbstractDataset compliance</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Test Coverage</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Configuration upgrade path validation and success rates</span>

#### 8.3.5.3 Alerting and Notification

**Notification Strategy**:
- **Build Failures**: Immediate notification to development team
- **Performance Degradation**: Automated alerts for benchmark regressions
- **Security Issues**: Priority escalation for security vulnerability detection
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Specialized Test Failures</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Targeted notifications for thread-safety, integration, or migration test failures</span>

The enhanced CI/CD pipeline provides comprehensive validation of FlyRigLoader's core functionality while ensuring reliable operation of new specialized components including registry thread-safety, Kedro integration, and configuration migration capabilities. The pipeline maintains high performance standards while providing detailed coverage analysis and quality assurance across all supported deployment scenarios.

## 8.4 DEVELOPMENT ENVIRONMENT INFRASTRUCTURE

### 8.4.1 Local Development Setup

#### 8.4.1.1 Development Tool Stack (updated)

**Comprehensive Development Dependencies**:
- **Testing Framework**: pytest with coverage, mocking, and benchmarking
- **Code Quality**: black, isort, mypy, flake8, pre-commit hooks
- **Documentation**: MkDocs/Sphinx for documentation generation
- **Build Tools**: setuptools, wheel, pip for package management
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Version Management</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">semantic_version>=2.10.0 as runtime dependency for schema version parsing and compatibility validation</span>

#### 8.4.1.2 Development Workflow Automation (updated)

**Pre-commit Hook Integration**:
- **Automated Formatting**: black and isort on commit
- **Linting**: flake8 validation before commit
- **Type Checking**: mypy validation on staged files
- **Security Scanning**: bandit security analysis

**Development Environment Scripts** (updated):
- `setup_env.sh`: Complete environment provisioning
- `activate_env.sh`: Environment activation helper
- `run_tests.sh`: <span style="background-color: rgba(91, 57, 243, 0.2)">Test execution with coverage including property-based tests and migration tests; automatically skips Kedro integration tests unless optional extras are installed</span>
- `build_docs.sh`: Documentation generation

#### 8.4.1.3 Environment Configuration Management (updated)

**Conda Environment Definition** (`environment.yml`):

**Core Dependencies Section** (updated):
```yaml
name: flyrigloader-dev
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.8
  - loguru>=0.7.0
  - pydantic>=2.6
  - numpy>=1.21.0
  - pandas>=1.3.0
  - semantic_version>=2.10.0
  
  # Development Tools
  - pytest>=7.0.0
  - pytest-cov>=4.0.0
  - black>=24.3.0
  - isort>=5.12.0
  - mypy>=1.8.0
  - flake8>=6.0.0
  - pre-commit>=3.6.0
  
  # Documentation
  - mkdocs>=1.5.0
  - sphinx>=6.0.0
  
  # Performance and Property-Based Testing
  - hypothesis>=6.131.9
  - memory-profiler>=0.60.0
  - psutil>=5.9.0
  
  # Optional Dependencies (commented by default)
  # Uncomment and run: conda env update --file environment.yml --prune && pip install -e .[kedro]
  # - kedro>=0.18.0
```

**Environment Configuration Matrix** (updated):

| Component | Version Specification | Channel | Purpose |
|-----------|----------------------|---------|------------|
| **Python** | >=3.8 | conda-forge | Runtime environment |
| **pydantic** | >=2.6 | conda-forge | Configuration validation |
| **loguru** | >=0.7.0 | conda-forge | Structured logging |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">semantic_version</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">>=2.10.0</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">conda-forge</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Schema version management</span>** |
| **pytest** | >=7.0.0 | conda-forge | Testing framework |
| **hypothesis** | >=6.131.9 | conda-forge | Property-based testing |
| **black** | >=24.3.0 | conda-forge | Code formatting |
| **mypy** | >=1.8.0 | conda-forge | Type checking |

#### 8.4.1.4 Optional Dependency Management (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration Setup</span>**:

<span style="background-color: rgba(91, 57, 243, 0.2)">For developers requiring Kedro pipeline integration capabilities, the development environment supports selective activation of optional dependencies</span>:

**<span style="background-color: rgba(91, 57, 243, 0.2)">Enable Kedro Support</span>**:
```bash
# Uncomment kedro>=0.18.0 line in environment.yml, then:
conda env update --file environment.yml --prune
pip install -e .[kedro]
```

**<span style="background-color: rgba(91, 57, 243, 0.2)">Development Workflow Impact</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Standard Development</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Core library functionality available without Kedro dependencies</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro-Enhanced Development</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Full AbstractDataSet integration testing and pipeline workflow validation</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Test Suite Adaptation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated detection of Kedro availability with graceful test skipping when not installed</span>

### 8.4.2 Quality Assurance Infrastructure

#### 8.4.2.1 Coverage Reporting System

**Sophisticated Coverage Infrastructure** (`tests/coverage/`):
- **Report Generation**: HTML, XML, JSON formats with Jinja2 templates
- **Trend Analysis**: Historical coverage tracking and analysis
- **Quality Gates**: Automated coverage threshold enforcement
- **Configuration Management**: Centralized coverage settings

**Coverage Automation Scripts**:
- `generate-coverage-reports.py`: Orchestrates comprehensive report generation
- `analyze-coverage-trends.py`: Historical trend analysis and visualization
- `validate-coverage.py`: Quality gate enforcement and validation

#### 8.4.2.2 Performance Monitoring

**Benchmark Infrastructure**:
- **Performance SLAs**: Defined thresholds for critical operations
- **Automated Validation**: CI/CD integrated performance testing
- **Trend Analysis**: Performance degradation detection
- **Resource Monitoring**: Memory and CPU usage validation

| Performance Metric | Target SLA | Measurement Method | Enforcement |
|-------------------|------------|-------------------|-------------|
| **File Discovery** | <5s for 10,000 files | pytest-benchmark | CI quality gate |
| **Data Loading** | <1s per 100MB | pytest-benchmark | CI quality gate |
| **Memory Efficiency** | <2x data size overhead | memory-profiler | CI validation |
| **Configuration Parsing** | <50ms for typical configs | pytest-benchmark | CI quality gate |

#### 8.4.2.3 Enhanced Testing Infrastructure (updated)

**<span style="background-color: rgba(91, 57, 243, 0.2)">Comprehensive Test Suite Execution</span>**:

**<span style="background-color: rgba(91, 57, 243, 0.2)">Property-Based Testing Integration</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Hypothesis Framework</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated generation of test cases for configuration parsing and data loading edge cases</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Strategy Coverage</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Comprehensive validation of loader registry operations under varied input conditions</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Performance Invariants</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Property-based validation of performance characteristics across input distributions</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Testing Framework</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Configuration Migration Validation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Automated testing of schema version upgrade paths and backward compatibility</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Semantic Version Integration</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Leveraging semantic_version library for robust version comparison and migration decision logic</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Rollback Testing</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Validation of configuration downgrade scenarios for development workflow flexibility</span>

**<span style="background-color: rgba(91, 57, 243, 0.2)">Conditional Integration Testing</span>**:
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Test Suite Adaptation</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Intelligent skipping of Kedro integration tests when optional dependencies are not installed</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Test Discovery Logic</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Dynamic test suite composition based on available extras and development environment configuration</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Coverage Normalization</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Accurate coverage reporting across different optional dependency installation states</span>

### 8.4.3 Development Environment Architecture

#### 8.4.3.1 Environment Lifecycle Management

```mermaid
flowchart TD
    subgraph "Development Environment Infrastructure"
        subgraph "Environment Setup"
            A[environment.yml] --> B[conda env create]
            B --> C[flyrigloader-dev]
            C --> D[Base Dependencies]
            D --> E{Optional Kedro?}
            E -->|No| F[Standard Development]
            E -->|Yes| G[conda env update]
            G --> H["pip install -e .[kedro]"]
            H --> I[Enhanced Development]
        end
        
        subgraph "Development Workflow"
            F --> J[run_tests.sh]
            I --> J
            J --> K[Property-Based Tests]
            J --> L[Migration Tests]
            J --> M{Kedro Available?}
            M -->|Yes| N[Kedro Integration Tests]
            M -->|No| O[Skip Kedro Tests]
            K --> P[Coverage Report]
            L --> P
            N --> P
            O --> P
        end
        
        subgraph "Quality Assurance"
            P --> Q[Coverage Analysis]
            Q --> R[Performance Benchmarks]
            R --> S[Quality Gate Validation]
            S --> T[Development Ready]
        end
    end
```

#### 8.4.3.2 Testing Strategy Integration

**Multi-Tier Testing Approach**:

| Test Tier | Implementation | Conditional Execution | Coverage Target |
|-----------|---------------|----------------------|-----------------|
| **Unit Tests** | Standard pytest suite | Always executed | 90% minimum |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Property-Based Tests</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Hypothesis-driven validation</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Always executed</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Edge case coverage</span>** |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Tests</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Version upgrade validation</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Always executed</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration path coverage</span>** |
| **Integration Tests** | End-to-end workflows | Always executed | Workflow coverage |
| **<span style="background-color: rgba(91, 57, 243, 0.2)">Kedro Integration</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">AbstractDataSet compliance</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">When extras installed</span>** | **<span style="background-color: rgba(91, 57, 243, 0.2)">Pipeline compatibility</span>** |
| **Performance Tests** | Benchmark validation | Always executed | Performance SLAs |

#### 8.4.3.3 Development Tools Integration

**IDE and Editor Support**:
- **VSCode Integration**: Python extension with comprehensive linting and type checking
- **PyCharm Professional**: Full-featured scientific Python development environment
- **Jupyter Integration**: Seamless integration for research workflow development
- **Git Hooks**: Pre-commit validation ensuring code quality before version control

**Quality Assurance Automation**:
- **Continuous Formatting**: Automatic code style enforcement through black and isort
- **Type Safety**: Comprehensive mypy integration for static type analysis
- **Security Scanning**: Automated vulnerability detection through bandit integration
- **Performance Monitoring**: Real-time benchmark execution and regression detection

### 8.4.4 Infrastructure Monitoring and Observability

#### 8.4.4.1 Development Environment Health

**Monitoring Capabilities**:
- **Environment Consistency**: Automated validation of development environment setup across team members
- **Dependency Health**: Continuous monitoring of dependency compatibility and security vulnerabilities
- **Performance Baselines**: Automated tracking of development environment performance characteristics
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Migration Testing Health</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Continuous validation of configuration migration success rates and performance</span>

#### 8.4.4.2 Quality Metrics Dashboard

**Development Quality Indicators**:
- **Test Execution Success Rate**: Historical tracking of test suite reliability
- **Coverage Evolution**: Trend analysis of test coverage across development iterations
- **Performance Regression Detection**: Automated alerts for development environment performance degradation
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Property-Based Test Effectiveness</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Analysis of edge case discovery and validation effectiveness</span>
- **<span style="background-color: rgba(91, 57, 243, 0.2)">Optional Dependency Impact</span>**: <span style="background-color: rgba(91, 57, 243, 0.2)">Monitoring of development workflow changes when Kedro extras are enabled</span>

This comprehensive development environment infrastructure provides researchers and contributors with a robust, scalable, and maintainable development experience. The integration of semantic versioning, property-based testing, and optional Kedro support ensures that FlyRigLoader can evolve effectively while maintaining research workflow compatibility and reliability.

## 8.5 MONITORING AND OBSERVABILITY INFRASTRUCTURE

### 8.5.1 Logging Infrastructure

#### 8.5.1.1 Structured Logging Architecture

**Loguru-based Logging System**:
- **Framework**: Loguru with structured logging throughout operations
- **Output Formats**: Colorized console output and structured file logging
- **Rotation Policy**: 10MB files with 7-day retention and gzip compression
- **Integration**: User-configurable for application-specific needs

**Logging Configuration Matrix**:

| Handler | Level | Format | Retention | Purpose |
|---------|-------|--------|-----------|---------|
| **Console** | INFO | Colorized, Human-readable | Session-based | User feedback |
| **File** | DEBUG | Structured JSON | 7 days compressed | Debugging and audit |
| <span style="background-color: rgba(91, 57, 243, 0.2)">**Registry**</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">INFO</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Console/File handlers</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">7 days compressed</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Loader registration auditing & conflict detection</span> |
| **Rotation** | ALL | Preserved format | 10MB per file | Space management |
| **User-defined** | Configurable | Custom format | User-defined | Application integration |

#### 8.5.1.2 Quality Metrics Collection

**Automated Quality Monitoring**:
- **Code Coverage**: Real-time coverage tracking with trend analysis
- **Performance Metrics**: Benchmark results and SLA compliance
- **Error Rates**: Exception tracking and classification
- **Build Success**: CI/CD pipeline health monitoring

```mermaid
flowchart TD
    subgraph "Logging Infrastructure"
        subgraph "Log Sources"
            L1[API Operations] --> Logger[Loguru Logger]
            L2[File Discovery] --> Logger
            L3[Data Loading] --> Logger
            L4[Configuration] --> Logger
            L5[Errors/Warnings] --> Logger
            L6[Registry Events] --> Logger
        end
        
        subgraph "Log Handlers"
            Logger --> Console[Console Handler]
            Logger --> File[File Handler]
            Console --> |INFO+| ColorOutput[Colored Output]
            File --> |DEBUG+| RotatingFile[Rotating Files]
        end
        
        subgraph "Log Management"
            RotatingFile --> |10MB| Rotation[Auto Rotation]
            Rotation --> |7 days| Compression[Gzip Compression]
            Compression --> Archive[logs/archive/]
        end
        
        subgraph "User Integration"
            Logger --> |Reconfigurable| UserApp[User Applications]
            UserApp --> External[External Monitoring]
        end
    end
```

### 8.5.2 Performance Monitoring

#### 8.5.2.1 Benchmark Infrastructure

**CI/CD Performance Validation**:
- **Automated Benchmarking**: pytest-benchmark integration
- **Performance Regression Detection**: Historical trend analysis
- **Resource Usage Monitoring**: Memory and CPU profiling
- **SLA Enforcement**: Quality gates for performance thresholds

#### 8.5.2.2 Quality Dashboard

**Quality Metrics Visualization**:
- **Coverage Trends**: Historical coverage analysis and reporting
- **Performance Metrics**: Benchmark results and trend visualization
- **Build Health**: CI/CD pipeline success rates and failure analysis
- **Code Quality**: Linting, type checking, and security scan results

### 8.5.3 Infrastructure Monitoring Framework

#### 8.5.3.1 Resource Monitoring Architecture

**Library-Specific Monitoring Approach**:
Since FlyRigLoader operates as a Python library embedded in user applications rather than as a standalone service, infrastructure monitoring focuses on development and CI/CD environments rather than production deployment infrastructure.

**Development Environment Monitoring**:

| Monitoring Aspect | Implementation | Purpose | Scope |
|-------------------|----------------|---------|-------|
| **Build Resources** | GitHub Actions resource tracking | CI/CD optimization | Development workflow |
| **Test Environment** | pytest resource profiling | Performance validation | Quality assurance |
| **Memory Usage** | memory-profiler integration | Resource leak detection | Library efficiency |
| **CPU Utilization** | Performance benchmarks | Computational efficiency | Algorithm optimization |

#### 8.5.3.2 Cost Monitoring and Optimization

**CI/CD Cost Management**:
- **Build Time Optimization**: Caching strategies for dependencies and artifacts
- **Resource Allocation**: Right-sizing CI/CD runners for build requirements
- **Test Efficiency**: Parallel execution and selective test running
- **Storage Management**: Artifact retention policies and cleanup automation

**Cost Optimization Strategies**:

```mermaid
flowchart TD
    subgraph "Cost Monitoring"
        C1[CI/CD Build Minutes] --> CostTracker[Cost Tracker]
        C2[Storage Usage] --> CostTracker
        C3[Network Transfer] --> CostTracker
        C4[Third-party Services] --> CostTracker
    end
    
    subgraph "Optimization Actions"
        CostTracker --> Analysis[Cost Analysis]
        Analysis --> Optimization[Optimization Strategies]
        
        Optimization --> Caching[Dependency Caching]
        Optimization --> Parallelization[Test Parallelization]
        Optimization --> Selective[Selective Execution]
        Optimization --> Cleanup[Automated Cleanup]
    end
    
    subgraph "Feedback Loop"
        Caching --> Measurement[Cost Measurement]
        Parallelization --> Measurement
        Selective --> Measurement
        Cleanup --> Measurement
        
        Measurement --> Reporting[Cost Reporting]
        Reporting --> C1
    end
```

#### 8.5.3.3 Security Monitoring Integration

**Security Scanning Infrastructure**:
- **Dependency Vulnerability Scanning**: Automated security audits in CI/CD
- **Code Security Analysis**: Static analysis security testing (SAST)
- **Container Security**: Base image vulnerability assessment
- **Supply Chain Security**: Software bill of materials (SBOM) generation

**Security Monitoring Pipeline**:

```mermaid
flowchart LR
    subgraph "Security Sources"
        S1[Dependency Scan] --> SecurityHub[Security Monitoring Hub]
        S2[Code Analysis] --> SecurityHub
        S3[Container Scan] --> SecurityHub
        S4[SBOM Generation] --> SecurityHub
    end
    
    subgraph "Security Processing"
        SecurityHub --> VulnAssessment[Vulnerability Assessment]
        VulnAssessment --> RiskScoring[Risk Scoring]
        RiskScoring --> PolicyEnforcement[Policy Enforcement]
    end
    
    subgraph "Security Actions"
        PolicyEnforcement --> AutoRemediation[Auto Remediation]
        PolicyEnforcement --> SecurityAlerts[Security Alerts]
        PolicyEnforcement --> ComplianceReporting[Compliance Reporting]
    end
```

### 8.5.4 Compliance and Auditing Infrastructure

#### 8.5.4.1 Audit Trail Management

**Comprehensive Audit Logging**:
- **Configuration Changes**: Version-controlled configuration audit trail
- **Access Patterns**: User interaction logging and analysis
- **System Events**: Comprehensive operational event logging
- **Compliance Events**: Regulatory requirement tracking and validation

**Audit Infrastructure Architecture**:

```mermaid
flowchart TD
    subgraph "Audit Sources"
        A1[Configuration Changes] --> AuditLogger[Audit Logger]
        A2[User Actions] --> AuditLogger
        A3[System Events] --> AuditLogger
        A4[Security Events] --> AuditLogger
        A5[Registry Operations] --> AuditLogger
    end
    
    subgraph "Audit Processing"
        AuditLogger --> Validation[Event Validation]
        Validation --> Enrichment[Context Enrichment]
        Enrichment --> Correlation[Event Correlation]
    end
    
    subgraph "Audit Storage"
        Correlation --> ImmutableStorage[Immutable Audit Storage]
        ImmutableStorage --> Encryption[Audit Encryption]
        Encryption --> Retention[Retention Management]
    end
    
    subgraph "Audit Analysis"
        Retention --> ComplianceCheck[Compliance Checking]
        ComplianceCheck --> AuditReporting[Audit Reporting]
        AuditReporting --> AlertGeneration[Alert Generation]
    end
```

#### 8.5.4.2 Compliance Validation Framework

**Regulatory Compliance Monitoring**:

| Compliance Area | Requirements | Monitoring Implementation | Validation Frequency |
|------------------|--------------|--------------------------|---------------------|
| **Data Privacy** | GDPR/CCPA compliance | PII detection and handling logs | Continuous |
| **Research Ethics** | IRB protocol adherence | Data usage audit trails | Per experiment |
| **Software Licensing** | Open source compliance | Dependency license scanning | Every build |
| **Quality Standards** | ISO/IEEE compliance | Quality metrics tracking | Monthly reporting |

### 8.5.5 Integration and Alerting Infrastructure

#### 8.5.5.1 Monitoring Integration Patterns

**External System Integration**:
- **APM Tools**: Application Performance Monitoring integration hooks
- **Log Aggregation**: Centralized logging system compatibility
- **Metrics Collection**: Time-series database integration
- **Alerting Systems**: Multi-channel notification support

**Integration Architecture**:

```mermaid
flowchart TD
    subgraph "Monitoring Sources"
        M1[Application Logs] --> IntegrationHub[Integration Hub]
        M2[Performance Metrics] --> IntegrationHub
        M3[Error Events] --> IntegrationHub
        M4[Security Alerts] --> IntegrationHub
    end
    
    subgraph "Integration Layer"
        IntegrationHub --> Adapters[Protocol Adapters]
        Adapters --> Formatters[Data Formatters]
        Formatters --> Routers[Event Routers]
    end
    
    subgraph "External Systems"
        Routers --> APM[APM Systems]
        Routers --> SIEM[SIEM Platforms]
        Routers --> Ticketing[Ticketing Systems]
        Routers --> Notification[Notification Services]
    end
    
    subgraph "Feedback Loop"
        APM --> Analytics[Analytics Engine]
        SIEM --> Analytics
        Analytics --> Insights[Operational Insights]
        Insights --> Optimization[System Optimization]
    end
```

#### 8.5.5.2 Alerting and Notification Strategy

**Multi-Tier Alerting Framework**:

| Alert Severity | Trigger Conditions | Response Time | Escalation Path |
|----------------|-------------------|---------------|-----------------|
| **Critical** | System failures, security breaches | Immediate | On-call engineer → Team lead → Management |
| **Warning** | Performance degradation, resource limits | 15 minutes | Team notification → Investigation |
| **Info** | Operational events, configuration changes | 1 hour | Logging only → Daily review |
| **Debug** | Detailed execution flow, troubleshooting | Real-time | Development tools → Developer analysis |

**Notification Delivery Architecture**:

```mermaid
flowchart LR
    subgraph "Alert Generation"
        AG1[Threshold Breaches] --> AlertEngine[Alert Engine]
        AG2[Anomaly Detection] --> AlertEngine
        AG3[Error Patterns] --> AlertEngine
        AG4[Security Events] --> AlertEngine
    end
    
    subgraph "Alert Processing"
        AlertEngine --> Correlation[Event Correlation]
        Correlation --> Deduplication[Alert Deduplication]
        Deduplication --> Prioritization[Priority Assignment]
    end
    
    subgraph "Notification Delivery"
        Prioritization --> Email[Email Notifications]
        Prioritization --> Slack[Slack Integration]
        Prioritization --> SMS[SMS Alerts]
        Prioritization --> Webhook[Webhook Delivery]
    end
    
    subgraph "Response Tracking"
        Email --> ResponseTracking[Response Tracking]
        Slack --> ResponseTracking
        SMS --> ResponseTracking
        Webhook --> ResponseTracking
        
        ResponseTracking --> Escalation[Auto Escalation]
        ResponseTracking --> Resolution[Resolution Tracking]
    end
```

### 8.5.6 Infrastructure Deployment and Management

#### 8.5.6.1 Monitoring Infrastructure as Code

**Infrastructure Automation Strategy**:
Since FlyRigLoader operates as a library, monitoring infrastructure is primarily limited to CI/CD and development environments rather than production deployment infrastructure.

**IaC Implementation Approach**:

```mermaid
flowchart TD
    subgraph "Infrastructure Definition"
        ID1[GitHub Actions Workflows] --> IaCEngine[Infrastructure as Code Engine]
        ID2[Docker Configurations] --> IaCEngine
        ID3[Testing Environments] --> IaCEngine
        ID4[Monitoring Configs] --> IaCEngine
    end
    
    subgraph "Deployment Pipeline"
        IaCEngine --> Validation[Configuration Validation]
        Validation --> Testing[Infrastructure Testing]
        Testing --> Deployment[Automated Deployment]
    end
    
    subgraph "Environment Management"
        Deployment --> Development[Development Environment]
        Deployment --> Staging[Staging Environment]
        Deployment --> Production[CI/CD Environment]
    end
    
    subgraph "Monitoring Deployment"
        Development --> DevMonitoring[Development Monitoring]
        Staging --> StagingMonitoring[Staging Monitoring]
        Production --> ProdMonitoring[Production Monitoring]
        
        DevMonitoring --> FeedbackLoop[Deployment Feedback]
        StagingMonitoring --> FeedbackLoop
        ProdMonitoring --> FeedbackLoop
        FeedbackLoop --> ID1
    end
```

#### 8.5.6.2 Disaster Recovery and Business Continuity

**Monitoring System Resilience**:

| Recovery Scenario | Recovery Time Objective (RTO) | Recovery Point Objective (RPO) | Implementation Strategy |
|-------------------|--------------------------------|----------------------------------|------------------------|
| **CI/CD System Failure** | 4 hours | 1 hour | GitHub Actions redundancy, alternative runners |
| **Log Data Loss** | 24 hours | 4 hours | Distributed log storage, backup retention |
| **Monitoring Tool Outage** | 2 hours | 30 minutes | Multi-vendor monitoring, failover automation |
| **Configuration Corruption** | 1 hour | 15 minutes | Version control, automated rollback |

### 8.5.7 Summary and Integration Points

The monitoring and observability infrastructure for FlyRigLoader provides comprehensive visibility into library operations while maintaining the lightweight architecture appropriate for a research-focused Python library. The infrastructure emphasizes development-time monitoring, CI/CD pipeline observability, and user integration flexibility.

**Key Infrastructure Components**:
- **Structured Logging**: Loguru-based system with comprehensive event capture including registry operations
- **Performance Monitoring**: CI/CD integrated benchmarking and SLA enforcement
- **Quality Assurance**: Automated metrics collection and trend analysis
- **Security Integration**: Vulnerability scanning and compliance monitoring
- **User Integration**: Flexible logging configuration for diverse research environments

**Integration with System Architecture**:
The monitoring infrastructure integrates seamlessly with the overall system design, providing observability patterns that support the library's extensible plugin architecture, configuration management system, and research workflow integration requirements. The emphasis on structured logging and performance validation ensures that the system maintains its reliability and efficiency standards while providing users with the visibility needed for effective debugging and optimization.

This infrastructure approach ensures that FlyRigLoader maintains its scientific computing focus while providing enterprise-grade observability patterns that support both development workflows and user application integration requirements.

## 8.6 INFRASTRUCTURE DIAGRAMS

### 8.6.1 Infrastructure Architecture Overview

```mermaid
graph TB
    subgraph "Development Infrastructure"
        subgraph "Local Development"
            D1[Conda Environment] --> D2[Development Tools]
            D2 --> D3[Pre-commit Hooks]
            D3 --> D4[Local Testing]
        end
        
        subgraph "Version Control"
            V1[Git Repository] --> V2[Branch Protection]
            V2 --> V3[GitHub Actions]
        end
        
        subgraph "CI/CD Pipeline"
            C1[Test Pipeline] --> C2[Quality Gates]
            C2 --> C3[Coverage Reporting]
            C3 --> C4[Performance Validation]
        end
        
        subgraph "Distribution"
            P1[Package Build] --> P2[PyPI Distribution]
            P2 --> P3[User Installation]
        end
    end
    
    subgraph "Quality Assurance"
        Q1[Code Quality] --> Q2[Security Scanning]
        Q2 --> Q3[Type Checking]
        Q3 --> Q4[Performance Benchmarks]
    end
    
    subgraph "External Services"
        E1[GitHub Actions] --> E2[Codecov]
        E2 --> E3[PyPI Registry]
    end
    
    D4 --> V1
    V3 --> C1
    C4 --> P1
    C1 --> Q1
    C3 --> E1
```

### 8.6.2 CI/CD Workflow Architecture

```mermaid
flowchart TD
    subgraph "CI/CD Workflow Infrastructure"
        subgraph "Trigger Events"
            T1[Push to main] --> Pipeline[GitHub Actions]
            T2[Push to develop] --> Pipeline
            T3[Pull Request] --> Pipeline
            T4[Manual Dispatch] --> Pipeline
        end
        
        subgraph "Test Matrix"
            Pipeline --> M1[Ubuntu + Python 3.8-3.11]
            Pipeline --> M2[Windows + Python 3.8-3.11]
            Pipeline --> M3[macOS + Python 3.8-3.11]
        end
        
        subgraph "Quality Pipeline"
            M1 --> Q1[Code Quality]
            M2 --> Q2[Test Execution]
            M3 --> Q3[Integration Tests]
            Q1 --> Q4[Performance Validation]
            Q2 --> Q4
            Q3 --> Q4
        end
        
        subgraph "Validation Gates"
            Q4 --> G1[Coverage Gate >90%]
            Q4 --> G2[Security Gate]
            Q4 --> G3[Performance Gate]
            G1 --> Decision{All Gates Pass?}
            G2 --> Decision
            G3 --> Decision
        end
        
        subgraph "Outputs"
            Decision -->|Pass| Success[Merge Approved]
            Decision -->|Fail| Failure[Merge Blocked]
            Success --> Artifacts[Build Artifacts]
            Failure --> Notifications[Failure Notifications]
        end
    end
```

### 8.6.3 Environment Management Flow

```mermaid
flowchart LR
    subgraph "Environment Management Infrastructure"
        subgraph "Initial Setup"
            S1[Clone Repository] --> S2[setup_env.sh]
            S2 --> S3[Conda Environment]
            S3 --> S4[Install Dependencies]
        end
        
        subgraph "Development Cycle"
            D1[activate_env.sh] --> D2[Development Session]
            D2 --> D3[Code Changes]
            D3 --> D4[Pre-commit Hooks]
            D4 --> D5[Local Testing]
        end
        
        subgraph "Quality Assurance"
            Q1[Run Tests] --> Q2[Coverage Analysis]
            Q2 --> Q3[Performance Benchmarks]
            Q3 --> Q4[Quality Reports]
        end
        
        subgraph "Integration"
            I1[Push Changes] --> I2[CI/CD Pipeline]
            I2 --> I3[Multi-platform Testing]
            I3 --> I4[Quality Gates]
        end
        
        S4 --> D1
        D5 --> Q1
        Q4 --> I1
        I4 --> Package[Package Distribution]
    end
```

## 8.7 INFRASTRUCTURE COST ANALYSIS

### 8.7.1 Infrastructure Cost Structure

**Zero-Cost Infrastructure Model**:
- **GitHub Actions**: Free tier provides 2,000 minutes/month for public repositories
- **Codecov**: Free tier for open-source projects
- **PyPI**: Free package distribution for open-source projects
- **Documentation Hosting**: GitHub Pages (free for public repositories)

**Resource Utilization Estimates**:

| Resource | Monthly Usage | Cost | Notes |
|----------|---------------|------|-------|
| **GitHub Actions** | ~500 minutes | $0 | Well within free tier |
| **Storage** | <100MB artifacts | $0 | Free tier sufficient |
| **Bandwidth** | <1GB/month | $0 | Free tier sufficient |
| **External Services** | Coverage reporting | $0 | Free tier for open source |

### 8.7.2 Scalability Considerations

**Infrastructure Scaling Patterns**:
- **Horizontal Scaling**: Not applicable (library-based system)
- **Vertical Scaling**: User-controlled via local resource allocation
- **Performance Scaling**: Optimized algorithms and data structures
- **Cost Scaling**: Zero infrastructure costs regardless of usage

## 8.8 MAINTENANCE AND OPERATIONAL PROCEDURES

### 8.8.1 Routine Maintenance

**Automated Maintenance Tasks**:
- **Dependency Updates**: Dependabot automated pull requests
- **Security Scanning**: Automated vulnerability detection and reporting
- **Performance Monitoring**: Continuous benchmark validation
- **Quality Metrics**: Automated coverage and quality reporting

**Manual Maintenance Procedures**:
- **Release Management**: Version tagging and PyPI distribution
- **Documentation Updates**: Documentation review and publication
- **Environment Updates**: Conda environment specification updates
- **Security Review**: Periodic security audit and dependency review

### 8.8.2 Disaster Recovery

**Backup and Recovery Strategy**:
- **Source Code**: Git repository with distributed backup
- **Documentation**: Version-controlled documentation sources
- **Build Configuration**: Infrastructure-as-code approach
- **Release Artifacts**: PyPI provides permanent artifact storage

**Recovery Procedures**:
- **Repository Recovery**: Clone from any Git remote
- **Environment Recovery**: Automated setup via `setup_env.sh`
- **Pipeline Recovery**: GitHub Actions configuration in version control
- **Package Recovery**: Rebuild from source using `pyproject.toml`

## 8.9 REFERENCES

### 8.9.1 Configuration Files Examined

- `pyproject.toml` - <span style="background-color: rgba(91, 57, 243, 0.2)">Package metadata, build configuration, dependency specifications, and optional-dependencies section</span>
- `environment.yml` - Conda environment definition with complete dependency stack
- `setup_env.sh` - Environment provisioning and automation script
- `activate_env.sh` - Environment activation helper script
- `.gitignore` - Git ignore patterns for build artifacts and temporary files
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/kedro/datasets.py` - Kedro DataSet integration implementation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/migration/migrators.py` - Configuration migration framework and versioning logic</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/migration/versions.py` - Version management and compatibility validation</span>

### 8.9.2 CI/CD Configuration

- `.github/workflows/test.yml` - Primary CI/CD pipeline with multi-platform testing
- `.github/workflows/quality-assurance.yml` - Quality gate enforcement workflow
- <span style="background-color: rgba(91, 57, 243, 0.2)">`.github/workflows/thread_safety.yml` - Thread safety validation and concurrent execution testing</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`.github/workflows/kedro_integration.yml` - Kedro integration testing and compatibility validation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`.github/workflows/migration_tests.yml` - Configuration migration testing and version upgrade validation</span>
- `tests/coverage/report-config.json` - Coverage reporting configuration
- `tests/coverage/scripts/generate-coverage-reports.py` - Coverage report automation
- `tests/coverage/scripts/analyze-coverage-trends.py` - Coverage trend analysis

### 8.9.3 Technical Specification Sections

- `3.6 DEVELOPMENT & DEPLOYMENT` - Development environment and CI/CD architecture
- `3.7 TECHNOLOGY STACK ARCHITECTURE` - Technology selection and integration patterns
- `6.4 SECURITY ARCHITECTURE` - Security implementation and validation framework
- `6.5 MONITORING AND OBSERVABILITY` - Logging infrastructure and quality metrics

### 8.9.4 Documentation Sources

- `docs/architecture.md` - System architecture documentation
- `docs/configuration_guide.md` - Configuration system documentation
- `docs/migration_guide.md` - Migration procedures and version compatibility
- <span style="background-color: rgba(91, 57, 243, 0.2)">`docs/kedro_integration.md` - Kedro pipeline integration guide and configuration examples</span>
- `blitzy/documentation/Project Guide.md` - Project overview and development guidelines
- `blitzy/documentation/Technical Specifications.md` - Complete technical specification

### 8.9.5 Infrastructure Directories

- `.github/` - GitHub configuration and workflow definitions
- `tests/coverage/` - Coverage infrastructure and reporting system
- `docs/` - Documentation source files and guides
- `logs/` - Runtime log file storage location (auto-created)
- `examples/` - Integration examples and usage patterns

# APPENDICES

## 9.1 ADDITIONAL TECHNICAL INFORMATION

### 9.1.1 Column Configuration Schema Implementation

The system implements a sophisticated YAML-based column configuration schema (`src/flyrigloader/io/column_config.yaml`) that extends beyond the basic configuration management covered in the main document. This schema provides fine-grained control over experimental data structure and validation.

#### 9.1.1.1 Extended Column Attributes

The column configuration system supports advanced attributes for experimental data handling:

| Attribute | Type | Purpose | Example |
|-----------|------|---------|---------|
| **special_handling** | string | Processing directive | `transform_to_match_time_dimension` |
| **alias** | list | Alternative column names | `["old_name", "legacy_name"]` |
| **metadata** | dict | Additional properties | `{"post_processing": true}` |
| **validation_hooks** | list | Custom validation functions | `["validate_signal_range"]` |

#### 9.1.1.2 Special Handler Mappings

The transformation engine includes specialized handlers for domain-specific data processing:

- **`_handle_signal_disp`**: Manages signal display transformations for optogenetic data visualization
- **`_extract_first_column`**: Extracts the first column from 2D arrays when 1D output is required
- **`_normalize_time_dimension`**: Ensures consistent time axis alignment across experimental recordings

### 9.1.2 Performance SLA Enforcement Framework

The system implements automated Service Level Agreement (SLA) validation through specialized monitoring scripts (`tests/coverage/scripts/check-performance-slas.py`) that enforce strict performance requirements for research reliability.

#### 9.1.2.1 Performance Thresholds

| Operation Category | SLA Threshold | Monitoring Method |
|-------------------|---------------|-------------------|
| **Data Loading** | ≤1 second per 100MB | Automated timing decorators |
| **DataFrame Transformation** | ≤500ms per 1M rows | Memory profiling integration |
| **Complete Workflow** | <30 seconds | End-to-end integration tests |
| **Optogenetic Processing** | <15 seconds | Domain-specific benchmarks |
| <span style="background-color: rgba(91, 57, 243, 0.2)">**File Discovery**</span> | <span style="background-color: rgba(91, 57, 243, 0.2)"><5 seconds per 10,000 files</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Automated timing decorators</span> |

#### 9.1.2.2 Quality Gate Integration

The SLA enforcement system integrates with the CI/CD pipeline through quality gates:

- **TST-PERF-001**: Baseline performance validation
- **TST-PERF-002**: Regression detection and blocking
- **TST-COV-001**: Coverage threshold enforcement (≥90%)
- **TST-COV-002**: Critical module coverage validation (100%)

### 9.1.3 Test Infrastructure Architecture

The testing framework extends beyond standard pytest implementation to include specialized components for neuroscience research validation.

#### 9.1.3.1 Coverage Analysis Framework

```mermaid
graph TB
    A[Test Execution] --> B[Coverage Collection]
    B --> C[Context Isolation]
    C --> D[Line Coverage Analysis]
    C --> E[Branch Coverage Analysis]
    D --> F[Quality Gate Evaluation]
    E --> F
    F --> G[Merge Decision]
    
    subgraph "Coverage Types"
        H[Overall Coverage ≥90%]
        I[Critical Modules 100%]
        J[Context Tracking]
    end
    
    F --> H
    F --> I
    F --> J
```

#### 9.1.3.2 Integration Test Categories

The test infrastructure includes specialized categories for neuroscience workflows:

- **Experimental Data Tests**: Validate processing of real fly rig data
- **Optogenetic Signal Tests**: Verify signal processing accuracy
- **Configuration Validation Tests**: Ensure YAML schema compliance
- **Performance Regression Tests**: Monitor SLA adherence
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Registry Thread-Safety Tests**: Validate concurrent access patterns and thread-safe operations</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Kedro Integration Tests**: Verify seamless integration with Kedro pipeline framework</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Configuration Migration Tests**: Ensure backward compatibility during config schema updates</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">**Config Builder Property-Based Tests**: Property-based testing for configuration generation logic</span>

### 9.1.4 Environment Management System

The system includes comprehensive environment management beyond the standard conda configuration documented in the main specification.

#### 9.1.4.1 Environment Structure

| Component | Location | Purpose |
|-----------|----------|---------|
| **Development Environment** | `dev_env/` | Development dependencies and tools |
| **Production Environment** | `prod_env/` | Runtime dependencies only |
| **Activation Script** | `activate_env.sh` | Auto-generated environment activation |
| **Lock Files** | `environment.lock` | Reproducible dependency pinning |
| <span style="background-color: rgba(91, 57, 243, 0.2)">**Optional Kedro Extras**</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">–</span> | <span style="background-color: rgba(91, 57, 243, 0.2)">Enables Kedro integration via `pip install flyrigloader[kedro]`</span> |

#### 9.1.4.2 Plugin Entry Points Configuration

The system leverages setuptools entry points for dynamic plugin discovery:

```yaml
entry_points:
  flyrigloader.loaders:
    - pickle = flyrigloader.io.pickle:PickleLoader
    - compressed = flyrigloader.io.pickle:CompressedPickleLoader
  flyrigloader.schemas:
    - default = flyrigloader.io.column_models:DefaultColumnConfig
    - experimental = flyrigloader.io.column_models:ExperimentalColumnConfig
```

## 9.2 GLOSSARY

**BaseLoader**: Protocol interface defining the contract for all data loader implementations in the plugin system, ensuring consistent behavior across different file formats.

**BaseRegistry**: Abstract singleton registry providing thread-safe registration and lookup of plugins with atomic operations for concurrent access.

**BaseSchema**: Protocol interface for schema providers in the registry system, enabling extensible validation and transformation rules.

**Column Configuration**: YAML-based specification defining the structure, types, validation rules, and transformation directives for experimental data columns.

<span style="background-color: rgba(91, 57, 243, 0.2)">**Config Builder**: Factory functions that produce fully validated Pydantic configuration objects with sensible defaults.</span>

**ConfigDiscoveryEngine**: Component that applies configuration patterns (ignore, mandatory, extraction) to file discovery operations with lazy evaluation.

<span style="background-color: rgba(91, 57, 243, 0.2)">**ConfigMigrator**: Component responsible for detecting configuration versions and applying step-wise migrations to the current schema.</span>

**DataFrameTransformer**: Core component responsible for converting raw experimental data into validated pandas DataFrames using configurable transformation pipelines.

**Dependency Injection**: Design pattern used throughout the system allowing runtime replacement of components for testing and extensibility without modifying core code.

**Discovery Manifest**: Lazy-evaluated collection of file paths with metadata, created without loading actual data content for memory efficiency.

**Drosophila**: The genus of flies (fruit flies) commonly used in neuroscience research, particularly in optogenetics experiments studying neural circuits.

**Entry Point**: Python packaging mechanism for plugin discovery using importlib.metadata, enabling dynamic extension loading.

**exp_matrix**: Dictionary structure containing primary experimental data loaded from pickle files, typically including signal data, metadata, and experimental parameters.

**Extraction Pattern**: Regular expression pattern used to extract metadata from filenames during discovery operations, supporting experimental workflow automation.

**FileManifest**: Container object holding discovered file paths with filtering utilities and lazy statistics computation for large dataset management.

**Fly Rig**: Experimental hardware setup specifically designed for research on flies (Drosophila), including optogenetic stimulation and recording equipment.

<span style="background-color: rgba(91, 57, 243, 0.2)">**FlyRigLoaderDataSet**: Kedro `AbstractDataset` implementation that wraps FlyRigLoader discovery and loading workflows.</span>

**Hierarchical Configuration**: Configuration system supporting inheritance and override patterns across project, dataset, and experiment levels for flexible workflow management.

**Ignore Pattern**: Shell-style wildcard pattern used to exclude files from discovery operations, preventing processing of temporary or irrelevant files.

<span style="background-color: rgba(91, 57, 243, 0.2)">**KedroIntegrationError**: Exception raised when Kedro catalog operations fail.</span>

**Lazy Evaluation**: Performance optimization technique where computations are deferred until results are actually needed, reducing memory usage and improving startup time.

**Legacy Adapter**: Compatibility layer providing dictionary-style access to Pydantic configuration models for backward compatibility with existing code.

**Loader Registry**: Singleton registry managing file format loaders with O(1) lookup performance for high-throughput data processing scenarios.

**Manifest-Based Workflow**: Processing pattern where file discovery is separated from data loading for memory efficiency and selective processing capabilities.

**Monolithic Loading**: Legacy pattern where all data is loaded into memory at once (deprecated in favor of decoupled pipeline architecture).

**Optogenetics**: Biological technique using light to control genetically modified neurons, commonly used in neuroscience research for precise neural manipulation.

**Opto Rig**: Optical experimental setup used for optogenetics research, providing controlled light stimulation with precise timing and spatial control.

**Pattern Matcher**: Component extracting metadata from filenames using compiled regular expressions with caching for performance optimization.

**Pickle Format**: Python object serialization format supporting compressed variants (.pklz, .pkl.gz) for efficient data storage and transmission.

**Plugin Architecture**: Extensibility system allowing third-party loaders and schemas without modifying core code, using entry points and protocol interfaces.

**Protocol**: Python typing construct defining structural interfaces without inheritance, enabling duck typing with type safety.

**Quality Gate**: Automated enforcement checkpoint in CI/CD preventing merges that violate coverage, performance, or other quality thresholds.

**Registry Pattern**: Design pattern providing centralized component registration and discovery with thread-safe access for concurrent environments.

<span style="background-color: rgba(91, 57, 243, 0.2)">**RegistryError**: Exception raised for registry plugin conflicts.</span>

**Schema Registry**: Singleton registry managing column configuration schemas with validation and transformation capabilities.

<span style="background-color: rgba(91, 57, 243, 0.2)">**Schema Version**: Explicit version string embedded in every configuration model to enable compatibility checks and migrations.</span>

**Signal Display Handler**: Special transformation handler for optogenetic signal data normalization and visualization preparation.

**Special Handler**: Pluggable transformation function for column-specific processing logic, enabling domain-specific data manipulations.

**src-layout**: Python project structure with source code under a `src/` directory for cleaner imports and better test isolation.

**Thread-Safe Singleton**: Singleton pattern implementation safe for concurrent access in multi-threaded environments using locking mechanisms.

**Transformation Pipeline**: Configurable chain of handlers processing raw data into DataFrames with validation and error handling.

**Validation Hook**: Extension point for custom validation logic in the schema system, allowing domain-specific validation rules.

<span style="background-color: rgba(91, 57, 243, 0.2)">**VersionError**: Exception raised for configuration version incompatibility.</span>

**Vial**: Container holding flies for experiments, used as organizational unit in data structure (e.g., vial_1, vial_2) for experimental tracking.

## 9.3 ACRONYMS

**API**: Application Programming Interface
**CI**: Continuous Integration
**CD**: Continuous Deployment
**CLI**: Command Line Interface
**CPU**: Central Processing Unit
**CSV**: Comma-Separated Values
**DI**: Dependency Injection
**ETL**: Extract, Transform, Load
**GB**: Gigabyte
**HDF5**: Hierarchical Data Format version 5
**HPC**: High-Performance Computing
**HTML**: HyperText Markup Language
**I/O**: Input/Output
**IDE**: Integrated Development Environment
**JSON**: JavaScript Object Notation
**KB**: Kilobyte
**KPI**: Key Performance Indicator
**MB**: Megabyte
**MIT**: Massachusetts Institute of Technology (license)
**NWB**: Neurodata Without Borders
**O(1)**: Constant time complexity notation
**O(n)**: Linear time complexity notation
**OS**: Operating System
**PEP**: Python Enhancement Proposal
**PI**: Principal Investigator
**PyPI**: Python Package Index
**RAM**: Random Access Memory
<span style="background-color: rgba(91, 57, 243, 0.2)">**RC**: Release Candidate</span>
**SLA**: Service Level Agreement
**SQL**: Structured Query Language
**TST**: Test (requirement prefix)
**UUID**: Universally Unique Identifier
**YAML**: Yet Another Markup Language / YAML Ain't Markup Language

## 9.4 REFERENCES

### 9.4.1 Files Examined

- `src/flyrigloader/io/column_config.yaml` - Column configuration schema specification
- `tests/coverage/scripts/check-performance-slas.py` - Performance SLA validation framework
- `tests/flyrigloader/integration/test_end_to_end_workflows.py` - Integration test infrastructure
- `src/flyrigloader/api.py` - Public API interface specification
- `src/flyrigloader/io/pickle.py` - Pickle loader implementation details
- `src/flyrigloader/io/transformers.py` - DataFrame transformation pipeline
- `environment.yml` - Conda environment specification
- `setup.py` - Package configuration and entry points
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/kedro/datasets.py` - Kedro dataset integration implementation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/kedro/catalog.py` - Kedro data catalog integration</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_kedro_integration.py` - Kedro integration test suite</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`docs/kedro_integration.md` - Kedro integration documentation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`src/flyrigloader/migration/migrators.py` - Configuration migration infrastructure</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_config_migration.py` - Migration functionality test suite</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`docs/migration_guide.md` - Configuration migration documentation</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`tests/flyrigloader/test_registry_threading.py` - Thread safety validation test suite</span>
- <span style="background-color: rgba(91, 57, 243, 0.2)">`pyproject.toml` - Modern Python package configuration</span>

### 9.4.2 Technical Specification Sections Referenced

- **Section 1.1**: Executive Summary - System overview and stakeholder analysis
- **Section 2.1**: Feature Catalog - Comprehensive feature documentation
- **Section 3.1**: Programming Languages - Python version constraints
- **Section 3.2**: Frameworks & Libraries - Core dependency documentation
- **Section 5.2**: Component Details - Architectural component specifications

### 9.4.3 Repository Analysis

- **Search Operations**: 15 comprehensive searches across codebase and documentation
- **Folders Explored**: 7 key directories including source code, tests, examples, and documentation
- **Integration Examples**: External project examples demonstrating library usage patterns
- **Performance Benchmarks**: SLA enforcement scripts and quality gate implementations