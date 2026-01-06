"""
Conservative Mutation: Minimal changes guaranteed to be fast.

Only adds/removes 1 item or changes 1 value at a time.
"""
import numpy as np
from typing import Dict, Any
from pymoo.core.mutation import Mutation

from src.core.logging_config import get_logger
from src.representation import RuleIndividual


def _get_cardinality(metadata: Dict[str, Any], feature_idx: int) -> int:
    """Get cardinality for a feature by index."""
    if feature_idx >= len(metadata['feature_order']):
        return 2  # Default for out-of-bound indices
    feature_name = metadata['feature_order'][feature_idx]
    return metadata['variables'][feature_name]['cardinality']


class ConservativeMutation(Mutation):
    """
    Conservative mutation with minimal changes.
    
    Operations:
    - add_one: Add 1 item to antecedent or consequent
    - remove_one: Remove 1 item from antecedent or consequent
    - change_value: Change value of 1 existing item
    
    All operations are guaranteed fast (no complex validation loops).
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        validator,
        prob: float = 0.7,
        **kwargs
    ):
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.prob = prob
        self.logger = get_logger(__name__)
    
    def _do(self, problem, X, **kwargs):
        """Apply conservative mutation."""
        Y = np.copy(X)
        num_genes = problem.n_var // 2
        num_features = len(self.metadata['feature_order'])
        
        for i in range(len(X)):
            if np.random.random() < self.prob:
                ind = RuleIndividual(self.metadata)
                ind.X = Y[i].copy()
                
                # Choose operation randomly
                op = np.random.choice(['add_one', 'remove_one', 'change_value'])
                
                if op == 'add_one':
                    # Find ignored positions
                    roles = ind.X[:num_genes]
                    ignored = np.where(roles == 0)[0]
                    
                    if len(ignored) > 0:
                        pos = np.random.choice(ignored)
                        new_role = np.random.choice([1, 2])  # Antecedent or consequent
                        new_value = np.random.randint(0, _get_cardinality(self.metadata, pos))
                        
                        ind.X[pos] = new_role
                        ind.X[pos + num_genes] = new_value
                
                elif op == 'remove_one':
                    # Find active positions
                    roles = ind.X[:num_genes]
                    active = np.where(roles > 0)[0]
                    
                    if len(active) > 1:  # Keep at least 1 item
                        pos = np.random.choice(active)
                        ind.X[pos] = 0
                        ind.X[pos + num_genes] = 0
                
                else:  # change_value
                    # Find active positions
                    roles = ind.X[:num_genes]
                    active = np.where(roles > 0)[0]
                    
                    if len(active) > 0:
                        pos = np.random.choice(active)
                        new_value = np.random.randint(0, _get_cardinality(self.metadata, pos))
                        ind.X[pos + num_genes] = new_value
                
                # Repair and validate
                ind.repair()
                
                # Only accept if valid
                try:
                    rule = ind.decode()
                    if self.validator.is_valid(rule):
                        Y[i] = ind.X
                except:
                    pass  # Keep original if invalid
        
        return Y
