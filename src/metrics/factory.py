"""
Factory for creating scenario-specific metrics instances.
"""
from typing import Dict, Optional
import pandas as pd
from pathlib import Path

from src.metrics.base import BaseMetrics
from src.metrics.scenario1 import Scenario1Metrics
from src.metrics.scenario2 import Scenario2Metrics
from src.core.exceptions import ConfigurationError

from .climate_metrics import ClimateMetrics


class MetricsFactory:
    """
    Factory for creating metrics instances based on scenario.
    
    Enables hot-swapping between metric calculation strategies via config.
    """
    
    _registry: Dict[str, type] = {
        'scenario_1': Scenario1Metrics,
        'scenario_2': Scenario2Metrics,
        'climate_5_obj': ClimateMetrics,
    }
    
    @classmethod
    def create_metrics(
        cls,
        scenario_name: str,
        dataframe: pd.DataFrame,
        supports_dict: dict,
        metadata: dict,
        raw_dataframe: Optional[pd.DataFrame] = None,
        config: Optional[dict] = None
    ) -> BaseMetrics:
        """
        Create metrics instance for specified scenario.
        
        Args:
            scenario_name: Scenario identifier ('scenario_1', 'scenario_2', 'climate_5_obj')
            dataframe: Processed dataset DataFrame (discretized)
            supports_dict: Single-item supports from supports.json
            metadata: Dataset metadata with feature_order
            raw_dataframe: Raw dataset with continuous values (required for climate_5_obj)
            config: Full experiment config (to get raw_path if needed)
        
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
        
        # Caso especial: ClimateMetrics necesita el dataset RAW
        if scenario_name == 'climate_5_obj':
            # Intentar obtener raw_dataframe si no se proporciona
            if raw_dataframe is None and config is not None:
                raw_path = config.get('dataset', {}).get('raw_path')
                if raw_path:
                    raw_path = Path(raw_path)
                    if raw_path.exists():
                        raw_dataframe = pd.read_csv(raw_path)
                        # Eliminar columna date si existe
                        if 'date' in raw_dataframe.columns:
                            raw_dataframe = raw_dataframe.drop(columns=['date'])
            
            # Añadir raw_path al metadata para que ClimateMetrics pueda cargarlo
            if raw_dataframe is None and config is not None:
                metadata = metadata.copy()
                metadata['raw_path'] = config.get('dataset', {}).get('raw_path')
            
            return metrics_class(
                dataframe=dataframe,
                supports_dict=supports_dict,
                metadata=metadata,
                raw_dataframe=raw_dataframe
            )
        
        # Otros escenarios
        return metrics_class(
            dataframe=dataframe,
            supports_dict=supports_dict,
            metadata=metadata
        )
    
    @classmethod
    def register_scenario(cls, scenario_name: str, metrics_class: type) -> None:
        """Register custom metrics class for scenario."""
        if not issubclass(metrics_class, BaseMetrics):
            raise TypeError(
                f"{metrics_class.__name__} must inherit from BaseMetrics"
            )
        cls._registry[scenario_name] = metrics_class
    
    @classmethod
    def available_scenarios(cls) -> list:
        """Get list of registered scenario names."""
        return list(cls._registry.keys())