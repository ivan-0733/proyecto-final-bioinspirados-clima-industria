"""
Scenario 1: Casual ARM metrics (casual-supp, casual-conf, maxConf).
"""
from typing import List, Tuple, Dict
import numpy as np
from .base import BaseMetrics


class Scenario1Metrics(BaseMetrics):
    """
    Casual Association Rule Mining metrics.
    
    Metrics:
    - casual_support: P(A ∩ C) + P(¬A ∩ ¬C)
    - casual_confidence: 0.5 * [P(C|A) + P(¬C|¬A)]
    - max_conf: max(P(C|A), P(A|C))
    """
    
    ALIAS_MAP = {
        'casual-supp': 'casual_support',
        'casual-conf': 'casual_confidence',
        'maxConf': 'max_conf'
    }
    
    def get_canonical_name(self, metric_name: str) -> str:
        """Handle aliases like 'conf' -> 'confidence'."""
        return self.ALIAS_MAP.get(metric_name, metric_name)
    
    def get_available_metrics(self) -> List[str]:
        """Available metrics for Scenario 1."""
        return ['casual_support', 'casual_confidence', 'max_conf']
    
    def _calculate_all_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]]
    ) -> dict:
        """
        Calculate all Scenario 1 metrics.
        
        Returns dict with metric values and error reasons.
        """
        metrics = {}
        
        # Calculate probabilities
        p_a = self._get_probability(antecedent)
        p_c = self._get_probability(consequent)
        p_ac = self._get_probability(antecedent + consequent)
        
        # Derived probabilities
        p_not_a = 1.0 - p_a
        p_not_c = 1.0 - p_c
        p_not_a_not_c = 1.0 - (p_a + p_c - p_ac)
        p_not_a_not_c = max(0.0, min(1.0, p_not_a_not_c))
        
        # Casual Support: P(A ∩ C) + P(¬A ∩ ¬C)
        metrics['casual_support'] = p_ac + p_not_a_not_c
        
        # Casual Confidence: 0.5 * [P(C|A) + P(¬C|¬A)]
        if p_a == 0:
            metrics['casual_confidence'] = None
            metrics['casual_confidence_error'] = 'zero_antecedent_support'
        elif p_not_a == 0:  # p_a == 1
            metrics['casual_confidence'] = None
            metrics['casual_confidence_error'] = 'full_antecedent_support'
        else:
            conf_a_c = p_ac / p_a
            conf_not_a_not_c = p_not_a_not_c / p_not_a
            metrics['casual_confidence'] = 0.5 * (conf_a_c + conf_not_a_not_c)
        
        # Max Confidence: max(P(C|A), P(A|C))
        if p_a == 0:
            metrics['max_conf'] = None
            metrics['max_conf_error'] = 'zero_antecedent_support'
        elif p_c == 0:
            metrics['max_conf'] = None
            metrics['max_conf_error'] = 'zero_consequent_support'
        else:
            conf_a_c = p_ac / p_a
            conf_c_a = p_ac / p_c
            metrics['max_conf'] = max(conf_a_c, conf_c_a)
        
        return metrics
