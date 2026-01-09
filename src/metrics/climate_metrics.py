"""
🌍 CLIMATE METRICS v4.0 - WORLD COMPETITION READY
=================================================
Métricas de Descubrimiento de Subgrupos para optimización clima-industria.

CORRECCIÓN CRÍTICA v4.0:
- Usa valores CONTINUOS del dataset RAW para calcular estadísticas
- Filtra por dataset DISCRETIZADO, evalúa con dataset RAW
- Esto permite discriminar subgrupos realmente diferentes

Autor: Sistema de Optimización Multiobjetivo
Versión: 4.0 - Fixed Continuous Values
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseMetrics


class ClimateMetrics(BaseMetrics):
    """
    Métricas de Descubrimiento de Subgrupos para 5 objetivos climáticos.
    
    ENFOQUE v4.0: Evaluación sobre VALORES CONTINUOS
    =================================================
    - Filtrado: usa dataset discretizado (0-4) para seleccionar subgrupo
    - Evaluación: usa dataset RAW (valores continuos) para calcular métricas
    
    Esto permite encontrar subgrupos donde los valores reales son significativamente
    diferentes del promedio global.
    """
    
    METRIC_NAMES = [
        'co2_emission', 
        'energy_consumption', 
        'renewable_share',
        'industrial_activity_index', 
        'energy_price'
    ]
    
    MAXIMIZE_METRICS = {
        'co2_emission': False,
        'energy_consumption': False,
        'renewable_share': True,
        'industrial_activity_index': True,
        'energy_price': False
    }
    
    PENALTY_VALUE = 2.0

    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        supports_dict: dict, 
        metadata: dict,
        raw_dataframe: Optional[pd.DataFrame] = None
    ):
        """
        Inicializa ClimateMetrics.
        
        Args:
            dataframe: DataFrame DISCRETIZADO (valores ordinales 0-4)
            supports_dict: Diccionario de soportes
            metadata: Metadata con feature_order y encodings
            raw_dataframe: DataFrame RAW con valores continuos (CRÍTICO)
        """
        super().__init__(dataframe, supports_dict, metadata)
        
        # Guardar DataFrame discretizado para filtrado
        self.df_discrete = self.df.copy()
        
        # Cargar o usar DataFrame RAW para evaluación
        if raw_dataframe is not None:
            self.df_raw = raw_dataframe.copy()
        else:
            # Intentar cargar desde config si no se proporciona
            self.df_raw = self._try_load_raw_dataframe(metadata)
        
        # Verificar que tenemos datos raw
        if self.df_raw is None:
            raise ValueError(
                "ClimateMetrics v4.0 requiere el dataset RAW con valores continuos. "
                "Proporcione raw_dataframe o configure raw_path en metadata."
            )
        
        # Verificar alineación de filas
        if len(self.df_discrete) != len(self.df_raw):
            raise ValueError(
                f"Mismatch en filas: discrete={len(self.df_discrete)}, raw={len(self.df_raw)}"
            )
        
        # Pre-calcular estadísticas globales sobre valores CONTINUOS
        self.global_stats = {}
        for col in self.METRIC_NAMES:
            if col in self.df_raw.columns:
                values = self.df_raw[col].astype(float)
                self.global_stats[col] = {
                    'mean': float(values.mean()),
                    'std': max(float(values.std()), 1e-6),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'range': float(values.max() - values.min()) if values.max() != values.min() else 1.0,
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                }
        
        # Índices máximos para dataset discretizado
        self.max_indices = {}
        for col in self.df_discrete.columns:
            self.max_indices[col] = int(self.df_discrete[col].max())
        
        self.var_names = metadata.get('feature_order', list(self.df_discrete.columns))
        self._cache = {}
        
    def _try_load_raw_dataframe(self, metadata: dict) -> Optional[pd.DataFrame]:
        """Intenta cargar el DataFrame raw desde la ruta en metadata."""
        raw_path = metadata.get('raw_path')
        if raw_path:
            path = Path(raw_path)
            if path.exists():
                df = pd.read_csv(path)
                # Eliminar columnas no numéricas excepto country
                if 'date' in df.columns:
                    df = df.drop(columns=['date'])
                return df
        return None

    def _calculate_all_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]]
    ) -> dict:
        """
        Calcula métricas usando valores CONTINUOS del dataset RAW.
        
        Proceso:
        1. Filtrar dataset DISCRETIZADO por la regla
        2. Obtener índices de filas que cumplen la regla
        3. Evaluar métricas sobre valores CONTINUOS de esas filas
        """
        full_rule_items = antecedent + consequent
        
        if not full_rule_items:
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}

        # === PASO 1: FILTRAR POR DATASET DISCRETIZADO ===
        mask = np.ones(len(self.df_discrete), dtype=bool)
        
        for var_idx, val_idx in full_rule_items:
            if var_idx >= len(self.var_names):
                continue
            
            col_name = self.var_names[var_idx]
            
            if col_name not in self.df_discrete.columns:
                continue
            
            # Validar índice de valor
            safe_val = val_idx
            max_val = self.max_indices.get(col_name, 4)
            if safe_val > max_val:
                safe_val = max_val
            if safe_val < 0:
                safe_val = 0
            
            mask &= (self.df_discrete[col_name] == safe_val)
        
        n_matched = mask.sum()
        n_total = len(self.df_discrete)
        
        if n_matched == 0:
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}
        
        if n_matched < 10:  # Mínimo de soporte
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}

        # === PASO 2: OBTENER VALORES CONTINUOS DEL SUBGRUPO ===
        matched_raw = self.df_raw[mask]
        
        # === PASO 3: CALCULAR MÉTRICAS SOBRE VALORES CONTINUOS ===
        coverage = n_matched / n_total
        coverage_factor = np.sqrt(coverage)  # Penalizar reglas muy específicas
        
        results = {}
        
        for col in self.METRIC_NAMES:
            if col not in matched_raw.columns or col not in self.global_stats:
                results[col] = self.PENALTY_VALUE
                continue
            
            # Estadísticas del subgrupo (VALORES CONTINUOS)
            local_values = matched_raw[col].astype(float)
            local_mean = float(local_values.mean())
            
            # Estadísticas globales (VALORES CONTINUOS)
            global_mean = self.global_stats[col]['mean']
            global_range = self.global_stats[col]['range']
            global_std = self.global_stats[col]['std']
            
            # === QUALITY SCORE: Z-Score normalizado ===
            # Mide cuántas desviaciones estándar está el subgrupo del global
            if global_std > 0:
                z_score = (local_mean - global_mean) / global_std
            else:
                z_score = 0.0
            
            # Normalizar z-score a [-1, 1] (clip en ±3 std)
            normalized_diff = np.clip(z_score / 3.0, -1.0, 1.0)
            
            # Ajustar según dirección de optimización
            if self.MAXIMIZE_METRICS[col]:
                # MAXIMIZAR: z_score positivo = bueno
                improvement = normalized_diff
            else:
                # MINIMIZAR: z_score negativo = bueno
                improvement = -normalized_diff
            
            # Combinar con coverage
            raw_quality = coverage_factor * improvement
            
            # Transformar a [0, 1] para MOEA/D (minimización)
            # raw_quality = 1 → fitness = 0 (óptimo)
            # raw_quality = 0 → fitness = 0.5 (neutral)
            # raw_quality = -1 → fitness = 1 (malo)
            clipped_quality = np.clip(raw_quality, -1.0, 1.0)
            fitness = (1.0 - clipped_quality) / 2.0
            fitness = np.clip(fitness, 0.0, 1.0)
            
            results[col] = float(fitness)
        
        return results

    def get_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]],
        objectives: List[str]
    ) -> Tuple[List[Optional[float]], Dict[str, str]]:
        """Calcula métricas seleccionadas para una regla."""
        cache_key = (frozenset(antecedent), frozenset(consequent))
        
        if cache_key in self._cache:
            all_metrics = self._cache[cache_key]
        else:
            all_metrics = self._calculate_all_metrics(antecedent, consequent)
            self._cache[cache_key] = all_metrics
        
        values = []
        errors = {}
        
        for metric in objectives:
            canonical = self.get_canonical_name(metric)
            
            if canonical in all_metrics:
                val = all_metrics[canonical]
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    values.append(None)
                    errors[metric] = "NaN/Inf value"
                else:
                    values.append(val)
            else:
                values.append(None)
                errors[metric] = f"Unknown metric: {metric}"
        
        return values, errors

    def get_available_metrics(self) -> List[str]:
        """Retorna lista de métricas disponibles."""
        return self.METRIC_NAMES.copy()

    def get_canonical_name(self, metric_name: str) -> str:
        """Retorna nombre canónico de la métrica."""
        aliases = {
            'avg_co2': 'co2_emission',
            'avg_consumption': 'energy_consumption',
            'avg_renewable': 'renewable_share',
            'avg_industry': 'industrial_activity_index',
            'avg_price': 'energy_price'
        }
        return aliases.get(metric_name, metric_name)

    def get_metric_info(self, metric_name: str) -> Dict[str, Any]:
        """Retorna información sobre una métrica."""
        canonical = self.get_canonical_name(metric_name)
        
        descriptions = {
            'co2_emission': 'Emisiones de CO2 (ton/capita) - MINIMIZAR',
            'energy_consumption': 'Consumo energético (kWh) - MINIMIZAR',
            'renewable_share': 'Porcentaje de energía renovable (%) - MAXIMIZAR',
            'industrial_activity_index': 'Índice de actividad industrial (0-100) - MAXIMIZAR',
            'energy_price': 'Precio de energía ($/kWh) - MINIMIZAR'
        }
        
        return {
            'name': canonical,
            'direction': 'maximize' if self.MAXIMIZE_METRICS.get(canonical, False) else 'minimize',
            'range': [0.0, 1.0],
            'description': descriptions.get(canonical, 'No description'),
            'global_stats': self.global_stats.get(canonical, {})
        }
    
    def clear_cache(self):
        """Limpia la caché de métricas calculadas."""
        self._cache.clear()


# Alias para compatibilidad
ClimateMetricsV2 = ClimateMetrics