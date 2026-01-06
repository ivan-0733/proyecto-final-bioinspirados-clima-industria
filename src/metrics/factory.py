"""
Factory for creating scenario-specific metrics instances.
"""
from typing import Dict
import pandas as pd

from src.metrics.base import BaseMetrics
from src.metrics.scenario1 import Scenario1Metrics
from src.metrics.scenario2 import Scenario2Metrics
from src.core.exceptions import ConfigurationError


class MetricsFactory:
    """
    Factory for creating metrics instances based on scenario.
    
    Enables hot-swapping between metric calculation strategies via config.
    """
    
    _registry: Dict[str, type] = {
        'scenario_1': Scenario1Metrics,
        'scenario_2': Scenario2Metrics,
    }
    
    @classmethod
    def create_metrics(
        cls,
        scenario_name: str,
        dataframe: pd.DataFrame,
        supports_dict: dict,
        metadata: dict
    ) -> BaseMetrics:
        """
        Create metrics instance for specified scenario.
        
        Args:
            scenario_name: Scenario identifier ('scenario_1', 'scenario_2')
            dataframe: Processed dataset DataFrame
            supports_dict: Single-item supports from supports.json
            metadata: Dataset metadata with feature_order
        
        Returns:
            Metrics instance implementing BaseMetrics
        
        Raises:
            ConfigurationError: If scenario_name not registered
        """
        if scenario_name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ConfigurationError(
                f"Unknown scenario '{scenario_name}'. Available: {available}"
            )
        
        metrics_class = cls._registry[scenario_name]
        return metrics_class(
            dataframe=dataframe,
            supports_dict=supports_dict,
            metadata=metadata
        )
    
    @classmethod
    def register_scenario(cls, scenario_name: str, metrics_class: type) -> None:
        """
        Register custom metrics class for scenario.
        
        Args:
            scenario_name: Unique scenario identifier
            metrics_class: Class implementing BaseMetrics
        """
        if not issubclass(metrics_class, BaseMetrics):
            raise TypeError(
                f"{metrics_class.__name__} must inherit from BaseMetrics"
            )
        
        cls._registry[scenario_name] = metrics_class
    
    @classmethod
    def available_scenarios(cls) -> list:
        """Get list of registered scenario names."""
        return list(cls._registry.keys())
