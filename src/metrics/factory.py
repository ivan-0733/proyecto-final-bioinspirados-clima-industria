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
    Se define la fábrica encargada de crear las herramientas de medición según el escenario.
    Esta clase permite cambiar la estrategia de cálculo automáticamente leyendo la configuración,
    asegurando que se use la herramienta correcta para cada tipo de problema.
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
        Se fabrica y entrega la instancia de métricas correcta para el problema actual.
        
        Args:
            scenario_name: El nombre clave del escenario que indica qué métricas usar (ej. 'climate_5_obj').
            dataframe: El dataset procesado (discretizado) que usa el algoritmo para navegar y buscar reglas.
            supports_dict: Diccionario con conteos rápidos de cuántas veces aparece cada dato.
            metadata: El manual que contiene la información sobre la estructura y significado de los datos.
            raw_dataframe: El dataset crudo con valores continuos. Es indispensable para la precisión en Clima.
            config: La configuración completa del experimento, útil para buscar rutas de archivos si hacen falta.
        
        Returns:
            Una instancia lista para realizar los cálculos de evaluación.
        
        Raises:
            ConfigurationError: Si se pide un escenario que no existe en el registro.
        """
        if scenario_name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ConfigurationError(
                f"Unknown scenario '{scenario_name}'. Available: {available}"
            )
        
        metrics_class = cls._registry[scenario_name]
        
        # Se identifica si el escenario solicitado es el de Clima.
        # Este caso es especial y se trata de manera diferente a los demás porque requiere una precisión matemática absoluta.
        # Mientras que otros escenarios pueden funcionar con datos simplificados (categorías), la evaluación climática
        # necesita acceder a los números reales (valores continuos) para medir el impacto ambiental verdadero.
        if scenario_name == 'climate_5_obj':
            # Se revisa si los datos reales (raw_dataframe) ya fueron entregados a la fábrica.
            # Si la variable está vacía (None), significa que necesitamos ir a buscar esos datos al archivo original
            # en el disco, usando la dirección guardada en la configuración del experimento.
            if raw_dataframe is None and config is not None:
                # Se extrae la ruta donde se encuentra el archivo de datos original.
                raw_path = config.get('dataset', {}).get('raw_path')
                
                # Si la ruta existe y es válida, se procede a intentar leer el archivo.
                if raw_path:
                    raw_path = Path(raw_path)
                    if raw_path.exists():
                        # Se lee el archivo CSV completo para cargar los valores numéricos reales en memoria.
                        raw_dataframe = pd.read_csv(raw_path)
                        
                        # Se realiza una limpieza preventiva: si existe una columna de fechas ('date'), se elimina.
                        # Esto se hace porque las fechas no son valores con los que se puedan hacer operaciones matemáticas
                        # como promedios o desviaciones estándar, y su presencia podría causar errores en los cálculos climáticos.
                        if 'date' in raw_dataframe.columns:
                            raw_dataframe = raw_dataframe.drop(columns=['date'])
            
            # Si después del intento anterior aún no tenemos los datos cargados en memoria (quizás por un error de lectura),
            # se asegura que al menos la dirección del archivo (raw_path) quede guardada en la metadata.
            # De esta forma, se le pasa la responsabilidad a la clase de métricas (ClimateMetrics) para que ella misma
            # intente buscar y cargar los datos por su cuenta cuando sea el momento de evaluar.
            if raw_dataframe is None and config is not None:
                metadata = metadata.copy()
                metadata['raw_path'] = config.get('dataset', {}).get('raw_path')
            
            # Se entrega la instancia de métricas climáticas completamente equipada.
            # Se le pasan tanto los datos discretos (para que el algoritmo pueda buscar patrones) como los datos raw
            # (para que pueda evaluar el impacto real), garantizando que pueda medir el desempeño ambiental con total precisión.
            return metrics_class(
                dataframe=dataframe,
                supports_dict=supports_dict,
                metadata=metadata,
                raw_dataframe=raw_dataframe
            )
        
        # Para otros escenarios (como Diabetes o pruebas genéricas), se utiliza la inicialización estándar.
        # Estos casos no requieren la carga adicional de datos continuos, por lo que se crean solo con
        # los datos procesados y los soportes básicos.
        return metrics_class(
            dataframe=dataframe,
            supports_dict=supports_dict,
            metadata=metadata
        )
    
    @classmethod
    def register_scenario(cls, scenario_name: str, metrics_class: type) -> None:
        """Se registra una nueva clase de métricas para permitir escenarios personalizados."""
        if not issubclass(metrics_class, BaseMetrics):
            raise TypeError(
                f"{metrics_class.__name__} must inherit from BaseMetrics"
            )
        cls._registry[scenario_name] = metrics_class
    
    @classmethod
    def available_scenarios(cls) -> list:
        """Se obtiene la lista de nombres de todos los escenarios disponibles en el sistema."""
        return list(cls._registry.keys())