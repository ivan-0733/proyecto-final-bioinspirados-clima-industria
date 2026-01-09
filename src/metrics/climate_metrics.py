from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from .base import BaseMetrics

class ClimateMetrics(BaseMetrics):
    """
    Métricas de Descubrimiento de Subgrupos (Enfoque Híbrido Diabetes-Clima).
    Optimiza el 'Impacto' de la regla: Soporte * Mejora Estandarizada.
    """
    
    METRIC_NAMES = [
        'co2_emission', 'energy_consumption', 
        'renewable_share', 'industrial_activity_index', 
        'energy_price'
    ]

    def __init__(self, dataframe: pd.DataFrame, supports_dict: dict, metadata: dict):
        super().__init__(dataframe, supports_dict, metadata)
        
        # 1. Pre-calcular estadísticas globales (El "Baseline")
        # Esto nos permite saber si un subgrupo es realmente "mejor" que el promedio
        self.global_stats = {}
        self.max_indices = {}
        
        for col in self.METRIC_NAMES:
            if col in self.df.columns:
                self.global_stats[col] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std() if self.df[col].std() > 0 else 1.0
                }
        
        # Pre-calcular límites para corrección de índices (Safety Patch)
        for col in self.df.columns:
            self.max_indices[col] = self.df[col].max()

    def _calculate_all_metrics(
        self,
        antecedent: List[Tuple[int, int]],
        consequent: List[Tuple[int, int]]
    ) -> dict:
        
        full_rule_items = antecedent + consequent
        
        # Penalización máxima si la regla está vacía
        if not full_rule_items:
            return {m: 0.0 for m in self.METRIC_NAMES} # 0.0 es malo aquí (sin impacto)

        # --- FILTRADO ROBUSTO (Mantiene el parche de seguridad) ---
        mask = np.ones(len(self.df), dtype=bool)
        
        for var_idx, val_idx in full_rule_items:
            if var_idx >= len(self.var_names): continue
            
            col_name = self.var_names[var_idx]
            
            # Auto-corrección de índices fuera de rango (como en tu versión anterior)
            safe_val = val_idx
            max_val = self.max_indices.get(col_name, 0)
            if safe_val > max_val: safe_val = max_val
            
            mask &= (self.df[col_name] == safe_val)
            
        matched_rows = self.df[mask]
        n_matched = len(matched_rows)
        
        # Si no hay datos, retornamos "Costo Máximo" (El algoritmo minimiza)
        if n_matched == 0:
            return {m: 10.0 for m in self.METRIC_NAMES} # Valor alto positivo = Malo

        # --- CÁLCULO ESTILO "DIABETES" (Probabilístico / Estadístico) ---
        # Calculamos el soporte (Probabilidad de la regla)
        support = n_matched / len(self.df)
        
        results = {}
        
        # Para cada objetivo, calculamos el "Z-Score de Impacto"
        # Impacto = Soporte * (Diferencia Estandarizada vs Global)
        # MOEA/D minimiza, así que retornamos: -1 * Impacto
        
        for col in self.METRIC_NAMES:
            if col not in matched_rows.columns:
                results[col] = 10.0
                continue
                
            local_mean = matched_rows[col].mean()
            global_mean = self.global_stats[col]['mean']
            global_std = self.global_stats[col]['std']
            
            # Calcular la mejora (Improvement)
            # Z-Score positivo significa que el valor local es MAYOR que el global
            z_score = (local_mean - global_mean) / global_std
            
            # Definir dirección de la mejora
            if col in ['renewable_share', 'industrial_activity_index']:
                # MAXIMIZAR: Queremos que Local > Global. 
                # Improvement es positivo si z_score es positivo.
                improvement = z_score
            else:
                # MINIMIZAR (CO2, Energía, Precio): Queremos Local < Global.
                # Improvement es positivo si z_score es negativo.
                improvement = -z_score
            
            # IMPACTO FINAL:
            # Ponderamos la mejora por la raíz cuadrada del soporte (estándar en minería de datos)
            # Esto equilibra "Reglas muy específicas pero perfectas" vs "Reglas generales pero buenas"
            impact = np.sqrt(support) * improvement
            
            # Invertimos el signo para MOEA/D (que busca minimizar)
            # Si encontramos una regla excelente, impact será alto (ej. 2.0), devolvemos -2.0
            # Si la regla es igual al promedio global, impact es 0.0
            results[col] = -impact

        return results

    def get_available_metrics(self) -> List[str]:
        return self.METRIC_NAMES

    def get_canonical_name(self, metric_name: str) -> str:
        return metric_name