"""
Base metrics interface for ARM scenarios.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.core.logging_config import get_logger


class BaseMetrics(ABC):
    """
    Abstract base class for ARM metrics.
    
    Defines interface for metric calculation, caching, and validation.
    Subclasses implement scenario-specific metrics (casual, correlation, etc.).
    """
    
    def __init__(self, dataframe: pd.DataFrame, supports_dict: dict, metadata: dict):
        """
        Initialize metrics calculator.
        
        Args:
            dataframe: Dataset for joint probability calculation
            supports_dict: Pre-calculated support values
            metadata: Dataset metadata
        """
        self.df = dataframe
        self.supports = supports_dict
        self.metadata = metadata
        self.total_rows = len(dataframe)
        
        # Parse variable order
        self.var_names = self._get_variable_order()
        
        # Cache for expensive calculations
        # Key: (frozenset(antecedent), frozenset(consequent))
        self._cache = {}
        
        self.log = get_logger(__name__)
        self.log.info(
            "metrics_initialized",
            scenario=self.__class__.__name__,
            dataset_rows=self.total_rows,
            num_variables=len(self.var_names)
        )
    
    def _get_variable_order(self) -> List[str]:
        """Get ordered list of variable names."""
        order = list(self.metadata.get('feature_order', []))
        target_name = self.metadata.get('target_variable')
        if isinstance(target_name, dict):
            target_name = target_name.get('name')
        
        if target_name and target_name not in order:
            order.append(target_name)
        return order
    
    def _get_probability(self, items: List[Tuple[int, int]]) -> float:
        """
        Calculate P(items).
        
        Args:
            items: List of (variable_index, value_index) tuples
        
        Returns:
            Probability value [0, 1]
        """
        if not items:
            return 0.0
        
        # Single item: use pre-calculated supports
        if len(items) == 1:
            var_idx, val_idx = items[0]
            var_name = self.var_names[var_idx]
            try:
                return self.supports['variables'][var_name][str(val_idx)]
            except KeyError:
                return 0.0
        
        # Multiple items: query DataFrame
        mask = np.ones(self.total_rows, dtype=bool)
        for var_idx, val_idx in items:
            var_name = self.var_names[var_idx]
            mask &= (self.df[var_name] == val_idx)
        
        count = mask.sum()
        return count / self.total_rows
    
    def get_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]],
        objectives: List[str]
    ) -> Tuple[List[Optional[float]], Dict[str, str]]:
        """
        Calculate metrics for a rule.
        
        Args:
            antecedent: List of (var_idx, val_idx) tuples
            consequent: List of (var_idx, val_idx) tuples
            objectives: List of metric names to calculate
        
        Returns:
            Tuple of (metric_values, error_details)
            - metric_values: List with same length as objectives (None if invalid)
            - error_details: Dict mapping objective to error reason
        """
        # Check cache
        cache_key = (frozenset(antecedent), frozenset(consequent))
        
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return self._extract_objectives(cached, objectives)
        
        # Calculate all metrics
        all_metrics = self._calculate_all_metrics(antecedent, consequent)
        
        # Cache result
        self._cache[cache_key] = all_metrics
        
        return self._extract_objectives(all_metrics, objectives)
    
    def _extract_objectives(
        self,
        all_metrics: dict,
        objectives: List[str]
    ) -> Tuple[List[Optional[float]], Dict[str, str]]:
        """Extract requested objectives from full metrics dict."""
        values = []
        errors = {}
        
        for obj in objectives:
            # Handle aliases
            canonical = self.get_canonical_name(obj)
            
            if canonical in all_metrics:
                val = all_metrics[canonical]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    values.append(None)
                    errors[obj] = all_metrics.get(f'{canonical}_error', 'unknown')
                else:
                    values.append(val)
            else:
                values.append(None)
                errors[obj] = 'metric_not_found'
        
        return values, errors
    
    @abstractmethod
    def _calculate_all_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]]
    ) -> dict:
        """
        Calculate all metrics for this scenario.
        
        Subclasses must implement this to return dict with all metric values.
        Should include error reasons as '{metric}_error' keys when applicable.
        
        Args:
            antecedent: Rule antecedent items
            consequent: Rule consequent items
        
        Returns:
            Dictionary with all metric values and error reasons
        """
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metric names.
        
        Returns:
            List of metric names this scenario supports
        """
        pass
    
    @abstractmethod
    def get_canonical_name(self, metric_name: str) -> str:
        """
        Get canonical metric name (handles aliases).
        
        Args:
            metric_name: Potentially aliased name
        
        Returns:
            Canonical metric name
        """
        pass
    
    def clear_cache(self) -> None:
        """Clear the metrics cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        self.log.info("cache_cleared", entries_removed=cache_size)
    
    def get_cache_statistics(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "cached_rules": len(self._cache),
            "cache_hit_rate": "not_tracked"  # Can be enhanced
        }
