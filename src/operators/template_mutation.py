"""
Template Mutation: Use predefined valid patterns.

Mutates only values within known-valid structures.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pymoo.core.mutation import Mutation

from src.core.logging_config import get_logger
from src.representation import RuleIndividual


def _get_cardinality(metadata: Dict[str, Any], feature_idx: int) -> int:
    """Get cardinality for a feature by index."""
    if feature_idx >= len(metadata['feature_order']):
        return 2  # Default for out-of-bound indices  
    feature_name = metadata['feature_order'][feature_idx]
    return metadata['variables'][feature_name]['cardinality']


class TemplateMutation(Mutation):
    """
    Mutation using predefined templates.
    
    Templates define valid rule structures (which variables in ant/cons).
    Mutation only changes values, not structure.
    
    Example templates:
    - [gender] => [diabetes]
    - [age, glucose] => [HbA1c]
    - [BMI, smoking] => [diabetes, glucose]
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        validator,
        templates: Optional[List[Tuple[List[str], List[str]]]] = None,
        prob: float = 0.7,
        **kwargs
    ):
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.prob = prob
        self.logger = get_logger(__name__)
        
        # Default templates if none provided
        if templates is None:
            self.templates = self._create_default_templates()
        else:
            self.templates = templates
        
        self.logger.info("template_mutation_initialized", num_templates=len(self.templates))
    
    def _create_default_templates(self) -> List[Tuple[List[int], List[int]]]:
        """Create default templates based on metadata."""
        feature_order = self.metadata['feature_order']
        templates = []
        
        # Common patterns (by feature index)
        # Single antecedent => single consequent
        for i in range(len(feature_order)):
            for j in range(len(feature_order)):
                if i != j:
                    templates.append(([i], [j]))
        
        # Two antecedents => single consequent
        for i in range(len(feature_order)):
            for j in range(i+1, len(feature_order)):
                for k in range(len(feature_order)):
                    if k != i and k != j:
                        templates.append(([i, j], [k]))
        
        return templates[:50]  # Limit to 50 templates
    
    def _do(self, problem, X, **kwargs):
        """Apply template-based mutation."""
        Y = np.copy(X)
        num_genes = problem.n_var // 2
        
        for i in range(len(X)):
            if np.random.random() < self.prob:
                # Select random template
                ant_indices, cons_indices = self.templates[np.random.randint(len(self.templates))]
                
                # Build new individual from template
                ind = RuleIndividual(self.metadata)
                ind.X = np.zeros(problem.n_var, dtype=int)
                
                # Set antecedent
                for idx in ant_indices:
                    ind.X[idx] = 1  # Role: antecedent
                    ind.X[idx + num_genes] = np.random.randint(
                        0, _get_cardinality(self.metadata, idx)
                    )
                
                # Set consequent
                for idx in cons_indices:
                    ind.X[idx] = 2  # Role: consequent
                    ind.X[idx + num_genes] = np.random.randint(
                        0, _get_cardinality(self.metadata, idx)
                    )
                
                # Repair and validate
                ind.repair()
                
                try:
                    rule = ind.decode()
                    if self.validator.is_valid(rule):
                        Y[i] = ind.X
                except:
                    pass  # Keep original if invalid
        
        return Y
