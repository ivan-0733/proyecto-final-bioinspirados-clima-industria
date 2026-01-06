## GitHub Copilot Instructions for TSAB Project

### What This Is
- **MOEA/D for association rule mining** on diabetes dataset using `pymoo`
- **Diploid genome encoding** (roles + values) with repair-based validation
- **5 mutation strategies** benchmarked (Fallback 🏆 best diversity, Mixed best quality)
- **Dual scenarios**: Casual ARM (supp/conf/maxConf) vs Correlation (jaccard/cosine/phi/kappa)
- **MIGRATION COMPLETE**: Clean architecture with SOLID, 152/152 tests passing, >80% coverage on refactored modules
- **Key Discovery**: Random n-point crossover + pop=50 → **3-5x more diversity** than fixed 2-point + pop=100

### Quick Start Commands
```bash
# Modern CLI (recommended)
python main.py list                    # Show configs
python main.py validate escenario_1    # Validate config (accepts short names)
python main.py run                     # Interactive mode
python main.py run --config escenario_1.json --no-report

# Testing & Validation
make test         # pytest with coverage (152 tests)
make validate     # Quick refactoring checks (runs validate_refactoring.py)
make check        # Full suite (validate + lint + test)
make clean        # Remove generated files

# Windows PowerShell (if make unavailable)
pytest tests/ -v --cov=src --cov-report=html --cov-report=term  # Run tests
python validate_refactoring.py                                   # Quick validation
python temp/validate_phase4.py                                   # Phase 4 checks

# Benchmarking mutations (in temp/ folder)
python temp/compare_quick.py              # Fast comparison (30 gens, 4 strategies, ~5min)
python temp/compare_mutations_full.py     # Exhaustive (150 gens, 5 strategies, ~40min)

# Diagnostic scripts (temp/ folder - for debugging)
python temp/diagnose_reproducibility.py   # Check seed consistency, file hashes, determinism
python temp/test_quick_reproducibility.py # Fast reproducibility test (3 runs)
python temp/check_moead_attrs.py          # Verify MOEAD algorithm attributes
```

### Architecture: Coexistence Model
- **Legacy** (`src/*.py`): individual.py, MOEAD.py, mutation.py, metrics.py, validator.py, callback.py, visualization.py, loggers.py
- **Refactored** (`src/core/`, `src/representation/`, `src/operators/`, `src/optimization/`, `src/metrics/`, `src/statistics/`, `src/cli/`): Pydantic config, structlog, SHA256 rules, SOLID validators, metrics factory, CLI
- **Import pattern**: `from src.representation import Rule` (new) vs `from src.individual import Individual` (legacy)
- **Data pipeline**: `src/preprocessing.py` → `src/calculate_supports.py` → `src/sampling.py` (optional pregeneration)
- **Temp utilities** (`temp/`): Diagnostic scripts, benchmark comparisons, migration docs (MIGRATION.md, PHASE*_SUMMARY.md, TROUBLESHOOTING.md)

### Execution Flow (Orchestrator Pattern)
```
main.py → src/cli/main_cli.py → Orchestrator
  ├─ Load config/*.json + merge config/general/base_config.json
  ├─ Choose full vs sample data (data/processed/*.csv + supports.json)
  ├─ Allocate results/<experiment>/exp_###/
  ├─ Setup structlog → logs/moead.log
  ├─ Snapshot config → config_snapshot.json
  └─ Orchestrator.run()
      ├─ Build MOEAD_ARM (wires metrics factory, validator, mutation factory)
      ├─ Attach ARMCallback (per-interval stats, HV tracking, pareto saves)
      ├─ Execute AdaptiveMOEAD (1/5 rule, stuck detection, dedup guard)
      ├─ Save discarded/reasons.json (aggregated by frequency)
      ├─ Dedupe final_pareto.csv → final_pareto_historical.csv
      └─ Generate plots/ via VisualizationManager (if enabled)
```

**Output structure** (`results/<experiment>/exp_###/`):
- `config_snapshot.json`: frozen config at run time
- `logs/moead.log`: structured JSON logs with generation context
- `populations/pop_gen_NNN.csv`: full pop every logging_interval
- `pareto/pareto_gen_NNN.csv`: non-dominated solutions with decoded rules
- `discarded/gen_NNN.json`: differential logs per generation
- `discarded/reasons.json`: aggregated validation failures sorted by count
- `stats/evolution_stats.csv`: min/mean/max objectives, HV, diversity, duplicates, adaptive probs
- `final_pareto.csv`: deduped by genome hash
- `final_pareto_historical.csv`: unique solutions across all gens
- `plots/`: metric_evolution, hypervolume, discarded_reasons, pareto_2d/3d/parallel

### Diploid Genome & Operators
**Genome layout** (`X` length `2*num_genes`):
```python
# Roles (first half): 0=ignore, 1=antecedent, 2=consequent
# Values (second half): feature indices
X = [role_0, role_1, ..., role_n, val_0, val_1, ..., val_n]
```

**Critical pattern**: ALWAYS call `Individual.repair()` after ANY genome modification
- Enforces role/value consistency (ignored genes have value=-1)
- Enforces cardinality bounds (min/max items per side from config)
- Empty antecedent/consequent or cross-side duplicates invalidate rule

**Operators**:
- `ARMSampling`: Validated initialization with Bloom filter for O(1) duplicate detection. Falls back to duplicates after `initialization.max_attempts` (default 5000) to meet pop_size.
- `DiploidNPointCrossover`: **Random n-point** (1 to n_points) with random positions each time; probability adapted via 1/5 rule. **KEY**: This randomness creates 3-5x more diversity than fixed 2-point.
- `ARMMutation` (5 strategies via `mutation.method` in config):
  - `"fallback"` 🏆: Fast timeout (2s) → pool sampling, **best diversity** (30.0 avg, 54 max), 17.4s avg
  - `"mixed"`: All ops enabled, **best quality** (HV: 0.5385) but slowest (191s avg)
  - `"conservative"` ⭐: Minimal changes (extension/contraction/replacement), **balanced** (13.5 diversity, 7.4s)
  - `"template"`: Predefined patterns (50 templates), **fastest** (6.5s, 15.0 diversity)
  - `"guided"`: Recombine from valid rule pool, **consistent** (13.5 avg, 11.7s)
- `mutation_factory.py`: Factory selects strategy based on `config["mutation"]["method"]`

### Validation & Metrics (Strategy Pattern)
**ARMValidator** enforces (in order):
1. Non-empty sides (min 1 item each by default from `rule_validity`)
2. Disjoint sides (no variable appears in both)
3. Cardinality bounds (`min_antecedent_items`, `max_antecedent_items`, etc.)
4. Business rules (`fixed_consequents`, `forbidden_pairs` from config)
5. Metric thresholds (`constraints.metric_thresholds` with alias support: `kappa`→`k_measure`)

**Metrics Factory** (`src/metrics/factory.py`):
```python
# Hot-swap scenarios via config["experiment"]["scenario"]
MetricsFactory.create_metrics(
    scenario_name="scenario_1",  # or "scenario_2"
    dataframe=df, supports_dict=supports, metadata=metadata
)
```

**Scenario 1** (Casual ARM): `casual-supp`, `casual-conf`, `maxConf`
**Scenario 2** (Correlation): `jaccard`, `cosine`, `phi`, `kappa` (aliased as `k_measure`)

**Critical**: Metrics returning `None` or invalid structure → `F=2.0` penalty in `ARMProblem` (objectives negated for pymoo minimization)

### Configuration Hotspots
**Experiment configs** (`config/escenario_*.json`):
- `experiment.scenario`: `"scenario_1"` or `"scenario_2"` → drives MetricsFactory
- `objectives.selected`: metric names (must match scenario's available metrics)
- `operators.mutation.method`: `"fallback"` (max diversity), `"mixed"` (max quality), `"conservative"` (balanced), `"template"` (fastest)
- `operators.mutation.probability`: `{initial, min, max}` → adapted via 1/5 rule
- `operators.crossover.probability`: `{initial, min, max}` → **0.7 initial recommended** for high diversity
- `algorithm.population_size`: **50 recommended** (vs old 100) → less crowding = more diversity
- `algorithm.generations`: **150 for production** (vs old 300) → captures peak diversity before convergence
- `algorithm.neighborhood`: `{size, replacement_size}` → **size=3** works well (30 was overkill in old config)
- `algorithm.logging_interval`: Save populations/pareto every N gens
- `algorithm.stuck_detection`: `{window, min_new, hv_window, hv_tol}` → early stopping at ~60 gens is normal
- `termination.ftol/period`: Converge if fitness variance < ftol for period gens
- `constraints.metric_thresholds`: Keys must use canonical names or aliases (e.g., `"kappa"` or `"k_measure"`)

**Base config** (`config/general/base_config.json`): Merged automatically, provides dataset paths

**CRITICAL INSIGHT**: Random n-point crossover + small population (50) + early stopping = **3-5x more diversity**

### Outputs / Debugging
**ARMCallback** (`src/callback.py`):
- Saves per-gen stats: min/mean/max objectives, hypervolume, diversity, duplicates, adaptive probs
- Writes `populations/pop_gen_NNN.csv` with decoded rules + genome every logging_interval
- Writes `pareto/pareto_gen_NNN.csv` with non-dominated solutions
- Logs differential `discarded/gen_NNN.json` (new failures per gen)
- Aggregates `discarded/reasons.json` (sorted by frequency) via `DiscardedRulesLogger.save()`

**VisualizationManager** (`src/visualization.py`):
- Reads `stats/evolution_stats.csv` and `pareto/pareto_gen_NNN.csv`
- Generates `plots/`: metric_evolution, hypervolume, discarded_reasons, pareto_2d/3d/parallel
- Parallel coordinates plot shows trade-offs across all objectives

**Structlog** (`logs/moead.log`):
- JSON-structured logs with context binding: `bind_context(generation=N, individual_id=X)`
- Search for errors: `grep '"level":"error"' logs/moead.log | jq` (Linux/Mac) or `Select-String -Path logs/moead.log -Pattern '"level":"error"'` (PowerShell)

**Diagnostic tools** (`temp/` folder):
- `diagnose_reproducibility.py`: Comprehensive check of seeds, file hashes, sampling determinism
- `test_quick_reproducibility.py`: Fast 3-run reproducibility test
- `check_moead_attrs.py`: Verify MOEAD algorithm has required attributes for adaptive operators
- `compare_quick.py` / `compare_mutations_full.py`: Benchmark mutation strategies side-by-side
- Migration docs: `MIGRATION.md`, `PHASE*_SUMMARY.md`, `TROUBLESHOOTING.md` for historical context

### When Modifying
**Always call `repair()` after genome edits**:
```python
# BAD - forgot repair
individual.X[0] = 1  # Change role
# Genome now inconsistent!

# GOOD - repair enforces constraints
individual.X[0] = 1
individual.repair(metadata, config)
```

**Metadata alignment is critical**:
- `metadata.feature_order` must match CSV column order
- Misalignment breaks decoding/support lookup and invalidates caches
- Example: `["age", "gender", "diabetes"]` → age=0, gender=1, diabetes=2

**Hypervolume reference point**:
- Assumes objectives scaled to [-1, 0] (negated for minimization)
- Adjust if adding metrics with different ranges
- See `ARMCallback._calculate_hypervolume()` for ref point logic

**Testing pattern** (from `tests/conftest.py`):
```python
# Use fixtures for consistency
def test_something(sample_dataframe, sample_supports, sample_metadata):
    metrics = MetricsFactory.create_metrics(
        "scenario_1", sample_dataframe, sample_supports, sample_metadata
    )
    # Test logic here
```

### New Architecture (Phase 1 Complete)
- **Config**: Use `src.core.Config.from_json()` with Pydantic validation instead of raw dict loading
- **Logging**: Use `src.core.setup_logging()` for structured JSON logs; bind context with `bind_context(generation=N)`
- **Rules**: Use `src.representation.Rule` with SHA256 hashing for O(1) deduplication (order-independent equality)
- **Validators**: Compose `RuleStructureValidator` + `BusinessRuleValidator` via `CompositeValidator` (SOLID)
- **Exceptions**: Raise `MOEADDeadlockError`, `RuleValidationError`, etc. with rich context instead of generic exceptions
- **Testing**: Run `python validate_refactoring.py` to verify new components; aim for >90% coverage with pytest

### Key Patterns
- **Single Responsibility**: Each validator checks ONE concern (structure vs business logic vs metrics)
- **Open/Closed**: Extend via composition (CompositeValidator) not modification
- **Dependency Inversion**: All new classes accept interfaces, not concrete implementations
- **Immutability**: Rule is frozen dataclass; use `Rule.from_items()` factory
- **Fail-Fast**: Pydantic validates configs at load time; no runtime surprises