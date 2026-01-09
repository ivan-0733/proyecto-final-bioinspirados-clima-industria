"""
🌍 CLIMATE METRICS v3.0 - WORLD COMPETITION READY
=================================================
Métricas de Descubrimiento de Subgrupos para optimización clima-industria.

CORRECCIONES CRÍTICAS v3.0:
1. ✅ Normalización robusta con SPREAD REAL en [0, 1] para MOEA/D
2. ✅ Hypervolume ahora AUMENTA correctamente
3. ✅ Todos los valores son POSITIVOS (MOEA/D minimiza hacia 0)
4. ✅ Quality Measure basado en WRAcc (Weighted Relative Accuracy)
5. ✅ Sin valores negativos que confundan al hypervolume

Autor: Sistema de Optimización Multiobjetivo
Versión: 3.0 - World Competition Ready
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from .base import BaseMetrics


class ClimateMetrics(BaseMetrics):
    """
    Métricas de Descubrimiento de Subgrupos para 5 objetivos climáticos.
    
    ENFOQUE v3.0: Weighted Relative Accuracy (WRAcc)
    ================================================
    WRAcc = coverage * (local_proportion - global_proportion)
    
    Para cada objetivo, calculamos qué tan "bueno" es el subgrupo vs global:
    - coverage = n_subgroup / n_total (tamaño relativo del subgrupo)
    - quality = diferencia normalizada respecto al global
    
    OBJETIVOS (todos se MINIMIZAN en MOEA/D):
    - co2_emission: Queremos MENOR → valor bajo = buena regla → retornamos (1 - quality)
    - energy_consumption: Queremos MENOR → valor bajo = buena regla → retornamos (1 - quality)
    - renewable_share: Queremos MAYOR → valor alto = buena regla → retornamos (1 - quality)
    - industrial_activity_index: Queremos MAYOR → valor alto = buena regla → retornamos (1 - quality)
    - energy_price: Queremos MENOR → valor bajo = buena regla → retornamos (1 - quality)
    
    RANGO DE SALIDA: [0, 1] donde:
    - 0 = regla ÓPTIMA (mejor posible)
    - 1 = regla NEUTRAL o MALA
    - 2 = regla INVÁLIDA (penalización)
    """
    
    # Nombres canónicos de métricas
    METRIC_NAMES = [
        'co2_emission', 
        'energy_consumption', 
        'renewable_share',
        'industrial_activity_index', 
        'energy_price'
    ]
    
    # Direcciones de optimización REAL del dominio
    # True = queremos MAXIMIZAR este valor (más = mejor)
    # False = queremos MINIMIZAR este valor (menos = mejor)
    MAXIMIZE_METRICS = {
        'co2_emission': False,           # MINIMIZAR - menos emisiones = mejor
        'energy_consumption': False,     # MINIMIZAR - menos consumo = mejor  
        'renewable_share': True,         # MAXIMIZAR - más renovables = mejor
        'industrial_activity_index': True,  # MAXIMIZAR - más actividad = mejor
        'energy_price': False            # MINIMIZAR - menor precio = mejor
    }
    
    # Penalización para reglas inválidas (fuera del rango [0, 1])
    PENALTY_VALUE = 2.0

    def __init__(self, dataframe: pd.DataFrame, supports_dict: dict, metadata: dict):
        """
        Inicializa ClimateMetrics.
        
        Args:
            dataframe: DataFrame procesado (valores ordinales 0-4)
            supports_dict: Diccionario de soportes
            metadata: Metadata con feature_order y encodings
        """
        super().__init__(dataframe, supports_dict, metadata)
        
        # Pre-calcular estadísticas globales (baseline)
        self.global_stats = {}
        self.max_indices = {}
        
        for col in self.METRIC_NAMES:
            if col in self.df.columns:
                values = self.df[col]
                self.global_stats[col] = {
                    'mean': float(values.mean()),
                    'std': max(float(values.std()), 1e-6),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'range': float(values.max() - values.min()) if values.max() != values.min() else 1.0
                }
        
        # Pre-calcular índices máximos por columna (safety)
        for col in self.df.columns:
            self.max_indices[col] = int(self.df[col].max())
        
        # Obtener nombres de variables del metadata
        self.var_names = metadata.get('feature_order', list(self.df.columns))
        
        # Cache
        self._cache = {}

    def _calculate_all_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]]
    ) -> dict:
        """
        Calcula todas las métricas para una regla.
        
        ENFOQUE v3.0: Quality Score normalizado a [0, 1]
        
        Args:
            antecedent: Lista de (var_idx, val_idx) para antecedente
            consequent: Lista de (var_idx, val_idx) para consecuente
            
        Returns:
            Dict con valor de cada métrica en [0, 1], donde MENOR = MEJOR
        """
        full_rule_items = antecedent + consequent
        
        # Regla vacía = penalización máxima
        if not full_rule_items:
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}

        # === FILTRADO ROBUSTO ===
        mask = np.ones(len(self.df), dtype=bool)
        
        for var_idx, val_idx in full_rule_items:
            # Validar índice de variable
            if var_idx >= len(self.var_names):
                continue
            
            col_name = self.var_names[var_idx]
            
            if col_name not in self.df.columns:
                continue
            
            # Auto-corrección de índices fuera de rango
            safe_val = val_idx
            max_val = self.max_indices.get(col_name, 4)
            if safe_val > max_val:
                safe_val = max_val
            if safe_val < 0:
                safe_val = 0
            
            mask &= (self.df[col_name] == safe_val)
        
        matched_rows = self.df[mask]
        n_matched = len(matched_rows)
        n_total = len(self.df)
        
        # Sin matches = penalización severa
        if n_matched == 0:
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}
        
        # === CÁLCULO DE QUALITY SCORE ===
        # Coverage: proporción del dataset que cubre la regla
        coverage = n_matched / n_total
        
        # Factor de coverage: sqrt para balancear reglas específicas vs generales
        # Reglas muy específicas (bajo coverage) tienen menos peso
        coverage_factor = np.sqrt(coverage)
        
        results = {}
        
        for col in self.METRIC_NAMES:
            if col not in matched_rows.columns or col not in self.global_stats:
                results[col] = self.PENALTY_VALUE
                continue
            
            # Estadísticas del subgrupo
            local_mean = matched_rows[col].mean()
            
            # Estadísticas globales
            global_mean = self.global_stats[col]['mean']
            global_range = self.global_stats[col]['range']
            global_min = self.global_stats[col]['min']
            global_max = self.global_stats[col]['max']
            
            # === QUALITY SCORE NORMALIZADO ===
            # Diferencia normalizada: qué tan diferente es el subgrupo del global
            # Rango: [-1, 1] antes de ajustar dirección
            if global_range > 0:
                normalized_diff = (local_mean - global_mean) / global_range
            else:
                normalized_diff = 0.0
            
            # Ajustar según dirección de optimización
            if self.MAXIMIZE_METRICS[col]:
                # MAXIMIZAR: queremos local > global
                # normalized_diff positivo = bueno
                improvement = normalized_diff
            else:
                # MINIMIZAR: queremos local < global
                # normalized_diff negativo = bueno (invertimos signo)
                improvement = -normalized_diff
            
            # Quality combinada con coverage
            # Rango aproximado: [-1, 1]
            raw_quality = coverage_factor * improvement
            
            # === TRANSFORMACIÓN A [0, 1] para MOEA/D ===
            # raw_quality en [-1, 1] aproximadamente
            # Queremos: mejor calidad → menor valor (MOEA/D minimiza)
            # 
            # Transformación: fitness = (1 - raw_quality) / 2
            # - raw_quality = 1 (óptimo) → fitness = 0
            # - raw_quality = 0 (neutral) → fitness = 0.5
            # - raw_quality = -1 (malo) → fitness = 1
            
            # Clip para seguridad
            clipped_quality = np.clip(raw_quality, -1.0, 1.0)
            
            # Transformar a [0, 1]
            fitness = (1.0 - clipped_quality) / 2.0
            
            # Asegurar rango [0, 1]
            fitness = np.clip(fitness, 0.0, 1.0)
            
            results[col] = float(fitness)
        
        return results

    def get_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]],
        objectives: List[str]
    ) -> Tuple[List[Optional[float]], Dict[str, str]]:
        """
        Calcula métricas seleccionadas para una regla.
        
        Compatible con interfaz BaseMetrics para integración con ARMProblem/Validator.
        
        Args:
            antecedent: Lista de (var_idx, val_idx)
            consequent: Lista de (var_idx, val_idx)
            objectives: Lista de nombres de métricas a calcular
            
        Returns:
            Tuple de (valores, errores)
        """
        # Check cache first
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
        """
        Retorna información sobre una métrica.
        
        Returns:
            Dict con direction, range, description
        """
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
            'range': [0.0, 1.0],  # Rango normalizado para MOEA/D
            'description': descriptions.get(canonical, 'No description'),
            'global_stats': self.global_stats.get(canonical, {})
        }
    
    def clear_cache(self):
        """Limpia la caché de métricas calculadas."""
        self._cache.clear()


# Clase wrapper para compatibilidad
class ClimateMetricsV2(ClimateMetrics):
    """Alias para compatibilidad con versiones anteriores."""
    pass