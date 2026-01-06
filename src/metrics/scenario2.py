"""
Scenario 2: Correlation-based metrics (Jaccard, Cosine, Phi, K-measure).
"""
from typing import List, Tuple, Dict
import numpy as np
from .base import BaseMetrics


class Scenario2Metrics(BaseMetrics):
    """
    Correlation-based Association Rule Mining metrics.
    
    Metrics:
    - jaccard: |A ∩ C| / |A ∪ C|
    - cosine: P(A ∩ C) / sqrt(P(A) * P(C))
    - phi_coefficient: (P(AC) - P(A)P(C)) / sqrt(P(A)P(A')P(C)P(C'))
    - k_measure: (P(AC) - P(A)P(C)) / max(P(A)(1-P(C)), P(C)(1-P(A)))
    """
    
    ALIAS_MAP = {
        'kappa': 'k_measure',
        'phi': 'phi_coefficient'
    }
    
    def get_canonical_name(self, metric_name: str) -> str:
        """Handle aliases like 'kappa' -> 'k_measure'."""
        return self.ALIAS_MAP.get(metric_name, metric_name)
    
    def get_available_metrics(self) -> List[str]:
        """Available metrics for Scenario 2."""
        return ['jaccard', 'cosine', 'phi_coefficient', 'k_measure']
    
    def _calculate_all_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]]
    ) -> dict:
        """
        Calculate all Scenario 2 metrics.
        
        Returns dict with metric values and error reasons.
        """
        metrics = {}
        
        # Calculate probabilities
        p_a = self._get_probability(antecedent)
        p_c = self._get_probability(consequent)
        p_ac = self._get_probability(antecedent + consequent)
        
        # Jaccard: |A ∩ C| / |A ∪ C|
        # P(A ∪ C) = P(A) + P(C) - P(A ∩ C)
        p_union = p_a + p_c - p_ac
        if p_union > 0:
            metrics['jaccard'] = p_ac / p_union
        else:
            metrics['jaccard'] = None
            metrics['jaccard_error'] = 'zero_union'
        
        # Cosine: P(A ∩ C) / sqrt(P(A) * P(C))
        if p_a > 0 and p_c > 0:
            metrics['cosine'] = p_ac / np.sqrt(p_a * p_c)
        else:
            metrics['cosine'] = None
            metrics['cosine_error'] = 'zero_support_X_or_Y'
        
        # Phi Coefficient: (P(AC) - P(A)P(C)) / sqrt(P(A)P(A')P(C)P(C'))
        p_a_comp = 1 - p_a  # P(A')
        p_c_comp = 1 - p_c  # P(C')
        
        numerator = p_ac - (p_a * p_c)
        denominator = np.sqrt(p_a * p_a_comp * p_c * p_c_comp)
        
        if denominator > 0:
            metrics['phi_coefficient'] = numerator / denominator
        else:
            metrics['phi_coefficient'] = None
            metrics['phi_coefficient_error'] = 'zero_denominator'
        
        # K-measure: (P(AC) - P(A)P(C)) / max(P(A)(1-P(C)), P(C)(1-P(A)))
        numerator_k = p_ac - (p_a * p_c)
        term1 = p_a * (1 - p_c)
        term2 = p_c * (1 - p_a)
        denominator_k = max(term1, term2)
        
        if denominator_k > 0:
            metrics['k_measure'] = numerator_k / denominator_k
        else:
            metrics['k_measure'] = None
            metrics['k_measure_error'] = 'zero_denominator'
        
        return metrics
