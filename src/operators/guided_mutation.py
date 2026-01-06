"""
Guided Mutation: Mutate by exchanging fragments from valid rules pool.

This strategy has high success rate because it recombines validated components.
"""
import numpy as np
from typing import Dict, Any, Optional
from pymoo.core.mutation import Mutation

from src.core.logging_config import get_logger
from src.representation import RuleIndividual


class GuidedMutation(Mutation):
    """
    Mutation guided by valid rules pool.
    
    Strategy:
    1. Load valid rules from pregenerated pool
    2. For mutation: take random valid rule
    3. Exchange antecedent or consequent with parent
    4. High probability of producing valid offspring
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        validator,
        pool_path: str = "data/processed/pregenerated/valid_rules_1m.csv",
        prob: float = 0.7,
        **kwargs
    ):
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.pool_path = pool_path
        self.prob = prob
        self.logger = get_logger(__name__)
        
        # Load pool
        self._load_pool()
        
    def _load_pool(self):
        """Load valid rules pool for guided mutation."""
        import pandas as pd
        from pathlib import Path
        
        pool_file = Path(self.pool_path)
        if not pool_file.exists():
            self.logger.warning(
                "guided_mutation_pool_not_found",
                path=self.pool_path,
                fallback="will use random mutation"
            )
            self.pool = None
            return
        
        df = pd.read_csv(pool_file)
        self.pool = []
        
        for _, row in df.iterrows():
            try:
                genome_str = row['encoded_rule'].strip('[]')
                genome = np.array([int(x.strip()) for x in genome_str.split(',')], dtype=int)
                self.pool.append(genome)
            except:
                pass
        
        self.pool = np.array(self.pool)
        self.logger.info("guided_mutation_pool_loaded", pool_size=len(self.pool))
    
    def _do(self, problem, X, **kwargs):
        """Apply guided mutation."""
        if self.pool is None or len(self.pool) == 0:
            return X  # Fallback: no mutation
        
        Y = np.copy(X)
        num_genes = problem.n_var // 2
        
        for i in range(len(X)):
            if np.random.random() < self.prob:
                # Select random rule from pool
                donor = self.pool[np.random.randint(len(self.pool))].copy()
                
                # Choose operation: swap antecedent or consequent
                if np.random.random() < 0.5:
                    # Swap antecedent
                    donor_roles = donor[:num_genes]
                    donor_values = donor[num_genes:]
                    
                    # Copy antecedent from donor
                    for j in range(num_genes):
                        if donor_roles[j] == 1:  # Antecedent
                            Y[i, j] = 1
                            Y[i, j + num_genes] = donor_values[j]
                else:
                    # Swap consequent
                    donor_roles = donor[:num_genes]
                    donor_values = donor[num_genes:]
                    
                    # Copy consequent from donor
                    for j in range(num_genes):
                        if donor_roles[j] == 2:  # Consequent
                            Y[i, j] = 2
                            Y[i, j + num_genes] = donor_values[j]
                
                # Repair to ensure consistency
                ind = RuleIndividual(self.metadata)
                ind.X = Y[i]
                ind.repair()
                Y[i] = ind.X
        
        return Y
