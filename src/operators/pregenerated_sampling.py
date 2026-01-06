"""
Sampling from pre-generated valid rules to avoid initialization timeouts.

This module loads validated rules from CSV and samples them for initialization,
bypassing the expensive validation process during population creation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from pymoo.core.sampling import Sampling

from src.core.logging_config import get_logger
from src.representation import RuleIndividual


class PregeneratedSampling(Sampling):
    """
    Sample from pre-validated rules stored in CSV.
    
    This approach:
    - Avoids timeout issues during initialization
    - Guarantees all individuals are valid
    - Provides deterministic initialization (with seed)
    - Much faster than random generation + validation
    
    CSV Format Expected:
        encoded_rule: "[role1, role2, ..., val1, val2, ...]"
        Other columns (antecedent, consequent, metrics) are informational
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        csv_path: str = "data/processed/pregenerated/valid_rules_1m.csv",
        allow_duplicates: bool = False,
        **kwargs
    ):
        """
        Initialize sampling from pre-generated rules.
        
        Args:
            metadata: Feature metadata (required for RuleIndividual)
            csv_path: Path to CSV with pre-validated rules
            allow_duplicates: If True, allow sampling same rule multiple times
        """
        super().__init__()
        self.metadata = metadata
        self.csv_path = Path(csv_path)
        self.allow_duplicates = allow_duplicates
        self.logger = get_logger(__name__)
        
        # Load rules on init
        self._load_rules()
        
        self.logger.info(
            "pregenerated_sampling_initialized",
            csv_path=str(self.csv_path),
            num_rules=len(self.rules),
            allow_duplicates=allow_duplicates
        )
    
    def _load_rules(self) -> None:
        """Load encoded rules from CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Pregenerated rules CSV not found: {self.csv_path}\n"
                f"Generate it first using brute-force or evolutionary sampling."
            )
        
        self.logger.info("loading_pregenerated_rules", path=str(self.csv_path))
        
        df = pd.read_csv(self.csv_path)
        
        if 'encoded_rule' not in df.columns:
            raise ValueError(
                f"CSV must have 'encoded_rule' column. Found: {df.columns.tolist()}"
            )
        
        # Parse encoded rules from string "[1,2,3,...]" to numpy arrays
        self.rules = []
        for idx, row in df.iterrows():
            try:
                # Parse string representation to list
                genome_str = row['encoded_rule'].strip('[]')
                genome = np.array([int(x.strip()) for x in genome_str.split(',')], dtype=int)
                self.rules.append(genome)
            except Exception as e:
                self.logger.warning(
                    "failed_to_parse_rule",
                    row_index=idx,
                    encoded_rule=row['encoded_rule'],
                    error=str(e)
                )
        
        self.rules = np.array(self.rules)
        self.logger.info(
            "rules_loaded",
            total_rows=len(df),
            parsed_rules=len(self.rules)
        )
    
    def _do(self, problem, n_samples, **kwargs):
        """
        Sample n_samples individuals from pre-generated rules.
        
        Args:
            problem: MOEA/D problem instance
            n_samples: Number of individuals to sample
            
        Returns:
            X: Array of sampled genomes (n_samples x n_var)
        """
        if n_samples > len(self.rules) and not self.allow_duplicates:
            self.logger.warning(
                "insufficient_pregenerated_rules",
                requested=n_samples,
                available=len(self.rules),
                will_sample_with_replacement=True
            )
            # Allow duplicates if not enough unique rules
            allow_replacement = True
        else:
            allow_replacement = self.allow_duplicates
        
        # Sample indices
        if allow_replacement:
            indices = np.random.choice(len(self.rules), size=n_samples, replace=True)
        else:
            indices = np.random.choice(len(self.rules), size=n_samples, replace=False)
        
        # Get genomes
        X = self.rules[indices].copy()
        
        self.logger.info(
            "population_sampled",
            n_samples=n_samples,
            unique_rules=len(np.unique(X, axis=0)),
            with_replacement=allow_replacement
        )
        
        return X
