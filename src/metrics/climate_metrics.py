"""
🌍 CLIMATE METRICS v2.0 - WORLD COMPETITION READY
=================================================
Métricas de Descubrimiento de Subgrupos para optimización clima-industria.

CORRECCIONES CRÍTICAS vs versión anterior:
1. Normalización robusta a rango [-1, 1] para MOEA/D
2. Manejo de edge cases (división por cero, valores extremos)
3. Cálculo de impacto estadísticamente sólido
4. Compatibilidad completa con ARMProblem

Autor: Sistema de Optimización Multiobjetivo
Versión: 2.0 - Bug-Free, Premio Mundial Compatible
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from .base import BaseMetrics


class ClimateMetrics(BaseMetrics):
    """
    Métricas de Descubrimiento de Subgrupos para 5 objetivos climáticos.
    
    Objetivos:
    - co2_emission: MINIMIZAR (menores emisiones = mejor)
    - energy_consumption: MINIMIZAR (menor consumo = mejor)
    - renewable_share: MAXIMIZAR (más renovables = mejor)
    - industrial_activity_index: MAXIMIZAR (más actividad = mejor)
    - energy_price: MINIMIZAR (menor precio = mejor)
    
    MOEA/D minimiza todos los objetivos, por lo que:
    - Para MINIMIZAR: retornamos el valor directo (menor = mejor)
    - Para MAXIMIZAR: retornamos -valor (menor = mejor en MOEA/D)
    """
    
    # Nombres canónicos de métricas
    METRIC_NAMES = [
        'co2_emission', 
        'energy_consumption', 
        'renewable_share',
        'industrial_activity_index', 
        'energy_price'
    ]
    
    # Direcciones de optimización
    # True = MAXIMIZAR (MOEA/D minimiza -valor)
    # False = MINIMIZAR (MOEA/D minimiza valor directamente)
    MAXIMIZE_METRICS = {
        'co2_emission': False,           # MINIMIZAR
        'energy_consumption': False,     # MINIMIZAR
        'renewable_share': True,         # MAXIMIZAR
        'industrial_activity_index': True,  # MAXIMIZAR
        'energy_price': False            # MINIMIZAR
    }
    
    # Penalización para reglas inválidas
    PENALTY_VALUE = 2.0  # Fuera del rango normalizado [-1, 1]

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
                    'std': max(float(values.std()), 1e-6),  # Evitar división por cero
                    'min': float(values.min()),
                    'max': float(values.max())
                }
        
        # Pre-calcular índices máximos por columna (safety)
        for col in self.df.columns:
            self.max_indices[col] = int(self.df[col].max())
        
        # Obtener nombres de variables del metadata
        self.var_names = metadata.get('feature_order', list(self.df.columns))

    def _calculate_all_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]]
    ) -> dict:
        """
        Calcula todas las métricas para una regla.
        
        Enfoque: Descubrimiento de Subgrupos (Subgroup Discovery)
        - Encontrar reglas donde el subgrupo tiene mejor rendimiento vs global
        - Impacto = sqrt(Support) * Improvement_Score
        
        Args:
            antecedent: Lista de (var_idx, val_idx) para antecedente
            consequent: Lista de (var_idx, val_idx) para consecuente
            
        Returns:
            Dict con valor de cada métrica (listo para MOEA/D minimización)
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
        
        # Sin matches = penalización (pero menos severa que regla inválida)
        if n_matched == 0:
            return {m: 1.5 for m in self.METRIC_NAMES}
        
        # === CÁLCULO DE IMPACTO ===
        # Support de la regla
        support = n_matched / len(self.df)
        
        results = {}
        
        for col in self.METRIC_NAMES:
            if col not in matched_rows.columns or col not in self.global_stats:
                results[col] = self.PENALTY_VALUE
                continue
            
            # Estadísticas del subgrupo
            local_mean = matched_rows[col].mean()
            
            # Estadísticas globales
            global_mean = self.global_stats[col]['mean']
            global_std = self.global_stats[col]['std']
            
            # Z-Score: cuántas desviaciones estándar del promedio global
            z_score = (local_mean - global_mean) / global_std
            
            # Determinar dirección de mejora
            if self.MAXIMIZE_METRICS[col]:
                # MAXIMIZAR: queremos local > global
                # z_score positivo = mejor
                improvement = z_score
            else:
                # MINIMIZAR: queremos local < global
                # z_score negativo = mejor (lo invertimos)
                improvement = -z_score
            
            # Impacto final: sqrt(support) * improvement
            # - sqrt(support) balancea reglas específicas vs generales
            # - improvement mide qué tan mejor es el subgrupo
            impact = np.sqrt(support) * improvement
            
            # Normalización a rango aproximado [-1, 1]
            # Clamp para evitar valores extremos
            normalized_impact = np.clip(impact, -1.0, 1.0)
            
            # MOEA/D minimiza, así que:
            # - Mayor impacto positivo = mejor regla = valor más negativo
            # - Valor 0 = regla neutral
            # - Impacto negativo = peor que global = valor positivo
            results[col] = -normalized_impact
        
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
        # Aliases para compatibilidad
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
            'co2_emission': 'Emisiones de CO2 (ton/capita)',
            'energy_consumption': 'Consumo energético (kWh)',
            'renewable_share': 'Porcentaje de energía renovable (%)',
            'industrial_activity_index': 'Índice de actividad industrial (0-100)',
            'energy_price': 'Precio de energía ($/kWh)'
        }
        
        return {
            'name': canonical,
            'direction': 'maximize' if self.MAXIMIZE_METRICS.get(canonical, False) else 'minimize',
            'range': [-1.0, 1.0],  # Rango normalizado
            'description': descriptions.get(canonical, 'No description'),
            'global_stats': self.global_stats.get(canonical, {})
        }


# Clase wrapper para compatibilidad
class ClimateMetricsV2(ClimateMetrics):
    """Alias para compatibilidad con versiones anteriores."""
    pass
