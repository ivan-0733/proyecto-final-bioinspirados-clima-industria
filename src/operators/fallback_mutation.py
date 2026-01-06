"""
Fallback Mutation: Fast timeout with pool fallback.

If mutation fails quickly, take a random rule from pool instead.
"""
import numpy as np
from typing import Dict, Any, Optional
from pymoo.core.mutation import Mutation
import signal

from src.core.logging_config import get_logger
from src.representation import RuleIndividual


def _get_cardinality(metadata: Dict[str, Any], feature_idx: int) -> int:
    """Get cardinality for a feature by index."""
    if feature_idx >= len(metadata['feature_order']):
        return 2  # Default for out-of-bound indices
    feature_name = metadata['feature_order'][feature_idx]
    return metadata['variables'][feature_name]['cardinality']


class FallbackMutation(Mutation):
    """
    Fast mutation with pool fallback.
    
    Strategy:
    1. Try mutation with aggressive timeout (2-3s)
    2. If fails: take random rule from pool
    3. Always produces output (no deadlock)
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        validator,
        pool_path: str = "data/processed/pregenerated/valid_rules_1m.csv",
        prob: float = 0.7,
        timeout: float = 5.0,  # Usado solo si reproducible_mode=False
        max_operations: int = 500,  # Presupuesto de operaciones para modo reproducible
        reproducible_mode: bool = True,  # Por defecto reproducible
        **kwargs
    ):
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.pool_path = pool_path
        self.prob = prob
        self.timeout = timeout
        self.max_operations = max_operations
        self.reproducible_mode = reproducible_mode
        self.logger = get_logger(__name__)
        
        # Load pool
        self._load_pool()
        
        # Stats
        self.mutations_attempted = 0
        self.mutations_succeeded = 0
        self.fallbacks_used = 0
        self.operations_used = 0  # Contador para modo reproducible
        
        mode = "reproducible (operations-based)" if reproducible_mode else f"fast (timeout={timeout}s)"
        self.logger.info("fallback_mutation_initialized", mode=mode, max_operations=max_operations)
    
    def _load_pool(self):
        """Load valid rules pool for fallback."""
        import pandas as pd
        from pathlib import Path
        
        pool_file = Path(self.pool_path)
        if not pool_file.exists():
            self.logger.warning(
                "fallback_pool_not_found",
                path=self.pool_path
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
        self.logger.info("fallback_pool_loaded", pool_size=len(self.pool))
    
    def _mutate_with_budget(self, X_i: np.ndarray, num_genes: int) -> Optional[np.ndarray]:
        """
        Try mutation with operation budget (reproducible mode).
        
        Args:
            X_i: Input genome
            num_genes: Number of genes
            
        Returns:
            Mutated genome or None if budget exhausted
        """
        operations_used = 0
        
        while operations_used < self.max_operations:
            # Simple mutation: change 1-2 items
            Y_i = X_i.copy()
            ind = RuleIndividual(self.metadata)
            ind.X = Y_i
            
            # Random operation
            roles = ind.X[:num_genes]
            active = np.where(roles > 0)[0]
            operations_used += 1
            
            if len(active) > 0 and np.random.random() < 0.5:
                # Change value
                pos = np.random.choice(active)
                ind.X[pos + num_genes] = np.random.randint(
                    0, _get_cardinality(self.metadata, pos)
                )
                operations_used += 1
            else:
                # Add/remove item
                if np.random.random() < 0.5 and len(active) < num_genes:
                    # Add
                    ignored = np.where(roles == 0)[0]
                    if len(ignored) > 0:
                        pos = np.random.choice(ignored)
                        ind.X[pos] = np.random.choice([1, 2])
                        ind.X[pos + num_genes] = np.random.randint(
                            0, _get_cardinality(self.metadata, pos)
                        )
                        operations_used += 2
                elif len(active) > 1:
                    # Remove
                    pos = np.random.choice(active)
                    ind.X[pos] = 0
                    ind.X[pos + num_genes] = 0
                    operations_used += 1
            
            ind.repair()
            operations_used += 1  # Contar repair
            
            # Validate
            try:
                rule = ind.decode()
                operations_used += 1  # Contar decode
                if self.validator.is_valid(rule):
                    operations_used += 1  # Contar validación
                    self.operations_used += operations_used
                    return ind.X
                operations_used += 1  # Contar validación fallida
            except:
                operations_used += 1
                pass
        
        # Budget agotado
        self.operations_used += operations_used
        return None
    
    def _mutate_with_timeout(self, X_i: np.ndarray, num_genes: int) -> Optional[np.ndarray]:
        """Try mutation with timeout."""
        import time
        
        start = time.time()
        max_attempts = 10  # More attempts before fallback
        
        for attempt in range(max_attempts):
            if time.time() - start > self.timeout:
                return None  # Timeout
            
            # Simple mutation: change 1-2 items
            Y_i = X_i.copy()
            ind = RuleIndividual(self.metadata)
            ind.X = Y_i
            
            # Random operation
            roles = ind.X[:num_genes]
            active = np.where(roles > 0)[0]
            
            if len(active) > 0 and np.random.random() < 0.5:
                # Change value
                pos = np.random.choice(active)
                ind.X[pos + num_genes] = np.random.randint(
                    0, _get_cardinality(self.metadata, pos)
                )
            else:
                # Add/remove item
                if np.random.random() < 0.5 and len(active) < num_genes:
                    # Add
                    ignored = np.where(roles == 0)[0]
                    if len(ignored) > 0:
                        pos = np.random.choice(ignored)
                        ind.X[pos] = np.random.choice([1, 2])
                        ind.X[pos + num_genes] = np.random.randint(
                            0, _get_cardinality(self.metadata, pos)
                        )
                elif len(active) > 1:
                    # Remove
                    pos = np.random.choice(active)
                    ind.X[pos] = 0
                    ind.X[pos + num_genes] = 0
            
            ind.repair()
            
            # Validate
            try:
                rule = ind.decode()
                if self.validator.is_valid(rule):
                    return ind.X
            except:
                pass
        
        return None
    
    def _do(self, problem, X, **kwargs):
        """Apply mutation with fallback."""
        Y = np.copy(X)
        num_genes = problem.n_var // 2
        
        for i in range(len(X)):
            if np.random.random() < self.prob:
                self.mutations_attempted += 1
                
                # Try mutation según modo
                if self.reproducible_mode:
                    result = self._mutate_with_budget(X[i], num_genes)
                else:
                    result = self._mutate_with_timeout(X[i], num_genes)
                
                if result is not None:
                    Y[i] = result
                    self.mutations_succeeded += 1
                elif self.pool is not None and len(self.pool) > 0:
                    # Fallback: take from pool
                    Y[i] = self.pool[np.random.randint(len(self.pool))].copy()
                    self.fallbacks_used += 1
                    
                    self.logger.debug(
                        "fallback_used",
                        individual=i,
                        fallback_rate=self.fallbacks_used / self.mutations_attempted
                    )
        
        return Y
