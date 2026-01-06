"""
Configuration management with Pydantic validation.
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
import json


class DatasetConfig(BaseModel):
    """Dataset paths and metadata."""
    name: str
    raw_path: Path
    processed_path: Path
    metadata_path: Path
    supports_path: Path
    sampling_path: Optional[Path] = None
    sample_metadata_path: Optional[Path] = None
    sample_supports_path: Optional[Path] = None

    @field_validator('raw_path', 'processed_path', 'metadata_path', 'supports_path')
    @classmethod
    def validate_required_paths(cls, v: Path) -> Path:
        """Ensure required paths exist."""
        if not v.exists():
            raise ValueError(f"Required path does not exist: {v}")
        return v


class ProbabilityRange(BaseModel):
    """Probability range with bounds validation."""
    initial: float = Field(ge=0.0, le=1.0)
    min: float = Field(ge=0.0, le=1.0)
    max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_bounds(self):
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        if not (self.min <= self.initial <= self.max):
            raise ValueError(
                f"initial ({self.initial}) must be in range [{self.min}, {self.max}]"
            )
        return self


class MutationConfig(BaseModel):
    """Mutation operator configuration."""
    method: Literal["mixed", "extension", "contraction", "replacement"] = "mixed"
    active_ops: List[str] = ["extension", "contraction", "replacement"]
    probability: ProbabilityRange
    repair_attempts: int = Field(ge=1, default=5)
    duplicate_attempts: int = Field(ge=0, default=5)
    timeout_seconds: float = Field(ge=0.1, default=10.0)


class CrossoverConfig(BaseModel):
    """Crossover operator configuration."""
    probability: ProbabilityRange
    n_points: int = Field(ge=1, le=10, default=2)


class OperatorsConfig(BaseModel):
    """All genetic operators."""
    crossover: CrossoverConfig
    mutation: MutationConfig


class NeighborhoodConfig(BaseModel):
    """MOEA/D neighborhood settings."""
    size: int = Field(ge=2, default=20)
    replacement_size: int = Field(ge=1, default=2)
    selection_probability: float = Field(ge=0.0, le=1.0, default=0.9)


class TerminationConfig(BaseModel):
    """Termination criteria."""
    enabled: bool = False
    ftol: float = Field(ge=0.0, default=0.0001)
    period: int = Field(ge=1, default=30)


class StuckDetectionConfig(BaseModel):
    """Deadlock detection configuration."""
    enabled: bool = True
    window: int = Field(ge=1, default=10)
    min_new: int = Field(ge=0, default=1)
    hv_window: int = Field(ge=1, default=10)
    hv_tol: float = Field(ge=0.0, default=1e-4)
    max_runtime_minutes: Optional[float] = Field(ge=1.0, default=None)


class DecompositionConfig(BaseModel):
    """Decomposition method for MOEA/D."""
    method: Literal["pbi", "tchebycheff", "tcheb", "weighted_sum", "ws"] = "tchebycheff"
    params: Dict[str, Any] = Field(default_factory=dict)


class InitializationConfig(BaseModel):
    """Population initialization settings."""
    max_attempts: int = Field(ge=100, default=10000)
    timeout_seconds: float = Field(ge=1.0, default=300.0)
    use_pregenerated: bool = Field(default=True, description="Use pregenerated pool for initialization (True) or random generation (False)")


class AlgorithmConfig(BaseModel):
    """Complete algorithm configuration."""
    generations: int = Field(ge=1, default=300)
    population_size: int = Field(ge=4, default=100)
    logging_interval: int = Field(ge=1, default=50)
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    stuck_detection: StuckDetectionConfig = Field(default_factory=StuckDetectionConfig)
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    neighborhood: NeighborhoodConfig = Field(default_factory=NeighborhoodConfig)
    operators: OperatorsConfig


class MetricThreshold(BaseModel):
    """Threshold for a single metric."""
    min: Optional[float] = None
    max: Optional[float] = None


class RuleValidityConfig(BaseModel):
    """Rule structure constraints."""
    min_antecedent_items: int = Field(ge=1, default=1)
    min_consequent_items: int = Field(ge=1, default=1)
    max_antecedent_items: int = Field(ge=1, default=10)
    max_consequent_items: int = Field(ge=1, default=10)


class ExclusionsConfig(BaseModel):
    """Business logic exclusions."""
    fixed_antecedents: List[str] = Field(default_factory=list)
    fixed_consequents: List[str] = Field(default_factory=list)
    forbidden_pairs: List[List[str]] = Field(default_factory=list)


class ConstraintsConfig(BaseModel):
    """All validation constraints."""
    rule_validity: RuleValidityConfig = Field(default_factory=RuleValidityConfig)
    metric_thresholds: Dict[str, MetricThreshold] = Field(default_factory=dict)
    exclusions: ExclusionsConfig = Field(default_factory=ExclusionsConfig)


class ExperimentConfig(BaseModel):
    """Experiment metadata."""
    name: str
    description: str = ""
    random_seed: int = Field(ge=0, default=42)
    output_root: Path = Path("results")


class Config(BaseModel):
    """Root configuration model."""
    experiment: ExperimentConfig
    dataset: DatasetConfig
    use_sampling: bool = False
    algorithm: AlgorithmConfig
    objectives: Dict[str, List[str]]  # {"selected": ["support", "confidence", ...]}
    constraints: ConstraintsConfig

    @classmethod
    def from_json(cls, path: Path, base_config_path: Optional[Path] = None) -> "Config":
        """Load configuration from JSON file with optional base config merge."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # Merge with base config if provided
        if base_config_path and base_config_path.exists():
            with open(base_config_path, 'r', encoding='utf-8') as f:
                base_dict = json.load(f)
            
            # Deep merge dataset section
            if 'dataset' in base_dict:
                if 'dataset' not in config_dict:
                    config_dict['dataset'] = {}
                for k, v in base_dict['dataset'].items():
                    if k not in config_dict['dataset']:
                        config_dict['dataset'][k] = v

        return cls(**config_dict)

    def to_json(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
