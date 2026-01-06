import numpy as np
import pandas as pd
from .base import BaseMetrics

class ClimateMetrics(BaseMetrics):
    """
    Métricas robustas para Clima y Energía (5 Objetivos).
    Maneja dualidad entre reglas discretas y evaluación continua.
    """
    
    def __init__(self, dataframe, supports_dict, metadata):
        super().__init__(dataframe, supports_dict, metadata)
        
        # [CRÍTICO] Cargamos los datos RAW para cálculos precisos
        # Buscamos la ruta en la configuración o asumimos la ruta estándar
        # Nota: Ajusta la ruta si tu CSV está en otro lado
        try:
            self.raw_df = pd.read_csv("data/raw/climate_dataset/global_climate_energy_2020_2024.csv")
        except FileNotFoundError:
            # Fallback por si acaso se ejecuta desde otro directorio
            self.raw_df = pd.read_csv("global_climate_energy_2020_2024.csv")
            
        self.log.info("climate_metrics_loaded", raw_rows=len(self.raw_df))

    def get_available_metrics(self):
        return ['avg_co2', 'avg_consumption', 'avg_renewable', 'avg_industry', 'avg_price']
    
    def get_canonical_name(self, metric_name):
        return metric_name

    def _calculate_all_metrics(self, antecedent, consequent):
        metrics = {}
        
        # 1. Obtenemos la máscara lógica usando el DF discretizado (self.df)
        # Esto nos dice QUÉ filas cumplen la regla "Temperatura=Alta"
        covered_mask = self._get_rule_mask(antecedent + consequent)
        
        # Validación de seguridad: Si la regla no cubre nada
        if np.sum(covered_mask) == 0:
            return {
                'avg_co2': 1e9,        # Penalización infinita
                'avg_consumption': 1e9,
                'avg_renewable': 1e9,  
                'avg_industry': 1e9,   
                'avg_price': 1e9
            }

        # 2. Usamos esa máscara para extraer los datos REALES (self.raw_df)
        # Esto nos da los valores numéricos precisos (ej. 245.5 toneladas)
        subset_raw = self.raw_df[covered_mask]

        # --- CÁLCULO DE OBJETIVOS (Minimización) ---
        
        # Obj 1: Minimizar CO2 (Directo)
        metrics['avg_co2'] = subset_raw['co2_emission'].mean()
        
        # Obj 2: Minimizar Consumo (Directo)
        metrics['avg_consumption'] = subset_raw['energy_consumption'].mean()
        
        # Obj 3: Maximizar Renovables (Multiplicamos por -1 para minimizar)
        metrics['avg_renewable'] = -1.0 * subset_raw['renewable_share'].mean()
        
        # Obj 4: Maximizar Industria (Multiplicamos por -1)
        metrics['avg_industry'] = -1.0 * subset_raw['industrial_activity_index'].mean()
        
        # Obj 5: Minimizar Precio (Directo)
        metrics['avg_price'] = subset_raw['energy_price'].mean()

        return metrics

    def _get_rule_mask(self, items):
        """Construye la máscara booleana exacta para el DF discretizado."""
        mask = np.ones(len(self.df), dtype=bool)
        
        for var_idx, val_idx in items:
            var_name = self.var_names[var_idx]
            
            # Verificación robusta de columnas
            if var_name not in self.df.columns:
                continue
                
            # Filtramos donde la columna (discretizada) coincide con el valor (alelo)
            mask &= (self.df[var_name] == val_idx)
            
        return mask