from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseMetrics


class ClimateMetrics(BaseMetrics):
    """
    Se define la clase responsable de evaluar la calidad climática de las reglas generadas.
    
    El funcionamiento se basa en un sistema dual de datos para garantizar la precisión:
    1. Se utiliza el dataset discretizado (simplificado) para la búsqueda y filtrado de registros.
    2. Se utiliza el dataset original (continuo) para los cálculos matemáticos y estadísticos.
    
    El objetivo es identificar subgrupos de datos que presenten comportamientos estadísticos
    significativamente distintos al promedio global, lo que indica el descubrimiento de un
    patrón climático o industrial relevante.
    """
    
    # Se listan los identificadores de las 5 métricas climáticas que se evaluarán.
    METRIC_NAMES = [
        'co2_emission', 
        'energy_consumption', 
        'renewable_share',
        'industrial_activity_index', 
        'energy_price'
    ]
    
    # Se configura la dirección de optimización deseada para cada métrica.
    # True indica que se busca aumentar el valor (maximizar).
    # False indica que se busca reducir el valor (minimizar).
    MAXIMIZE_METRICS = {
        'co2_emission': False,              # Se busca reducir las emisiones
        'energy_consumption': False,        # Se busca reducir el consumo
        'renewable_share': True,            # Se busca aumentar la participación renovable
        'industrial_activity_index': True,  # Se busca aumentar la actividad industrial eficiente
        'energy_price': False               # Se busca reducir el costo energético
    }
    
    # Se establece un valor numérico alto para penalizar las reglas inválidas o vacías.
    PENALTY_VALUE = 2.0

    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        supports_dict: dict, 
        metadata: dict,
        raw_dataframe: Optional[pd.DataFrame] = None
    ):
        """
        Se inicializa el evaluador preparando los conjuntos de datos y calculando estadísticas base.
        
        Args:
            dataframe: Conjunto de datos con valores discretos, utilizado exclusivamente para el filtrado lógico.
            supports_dict: Estructura de datos que contiene los conteos de frecuencia de las variables.
            metadata: Diccionario con la información estructural y definiciones del dataset.
            raw_dataframe: Conjunto de datos con valores continuos reales, esencial para la evaluación precisa.
        """
        super().__init__(dataframe, supports_dict, metadata)
        
        # Se crea una copia independiente del dataset discretizado para evitar modificaciones accidentales durante el filtrado.
        self.df_discrete = self.df.copy()
        
        # Se establece el dataset de valores continuos.
        # Es indispensable contar con estos datos reales para medir el impacto climático con exactitud.
        if raw_dataframe is not None:
            self.df_raw = raw_dataframe.copy()
        else:
            # Se intenta cargar el archivo desde la ruta especificada en la configuración si no se proporciona directamente.
            self.df_raw = self._try_load_raw_dataframe(metadata)
        
        # Se valida la existencia del dataset continuo; si no existe, se detiene el proceso por imposibilidad de cálculo.
        if self.df_raw is None:
            raise ValueError(
                "Se requiere obligatoriamente el dataset con valores continuos para realizar la evaluación climática."
            )
        
        # Se verifica la integridad de los datos asegurando que ambos datasets tengan el mismo número de registros.
        if len(self.df_discrete) != len(self.df_raw):
            raise ValueError(
                f"Existe una discrepancia en el tamaño de los datasets: discretizado={len(self.df_discrete)}, continuo={len(self.df_raw)}"
            )
        
        # Se calculan y almacenan las estadísticas globales de referencia.
        # Estos valores sirven como línea base para comparar el desempeño de cualquier subgrupo encontrado.
        self.global_stats = {}
        for col in self.METRIC_NAMES:
            if col in self.df_raw.columns:
                values = self.df_raw[col].astype(float)
                self.global_stats[col] = {
                    'mean': float(values.mean()),       # Promedio global de la variable
                    'std': max(float(values.std()), 1e-6), # Desviación estándar global (variabilidad)
                    'min': float(values.min()),         # Valor mínimo registrado
                    'max': float(values.max()),         # Valor máximo registrado
                    'range': float(values.max() - values.min()) if values.max() != values.min() else 1.0,
                    'q25': float(values.quantile(0.25)), # Percentil 25
                    'q75': float(values.quantile(0.75)), # Percentil 75
                }
        
        # Se pre-calculan los límites superiores de los valores discretos para optimizar las validaciones de rango.
        self.max_indices = {}
        for col in self.df_discrete.columns:
            self.max_indices[col] = int(self.df_discrete[col].max())
        
        # Se almacena el orden de las variables para mapear correctamente los índices genéticos a columnas del dataset.
        self.var_names = metadata.get('feature_order', list(self.df_discrete.columns))
        
        # Se inicializa un diccionario de almacenamiento temporal para guardar resultados y evitar recálculos innecesarios.
        self._cache = {}
        
    def _try_load_raw_dataframe(self, metadata: dict) -> Optional[pd.DataFrame]:
        """
        Se ejecuta el procedimiento de carga del archivo de datos continuos desde el disco.
        Se eliminan columnas no numéricas que no aportan al cálculo estadístico.
        """
        raw_path = metadata.get('raw_path')
        if raw_path:
            path = Path(raw_path)
            if path.exists():
                df = pd.read_csv(path)
                # Se descarta la columna de fecha ya que el análisis es atemporal y estadístico.
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
        Se realiza el cálculo matemático de las métricas de evaluación para una regla específica.
        
        El procedimiento sigue una secuencia lógica estricta:
        1. Se identifican los registros que cumplen las condiciones en el dataset discretizado.
        2. Se extraen los valores reales correspondientes a esos registros del dataset continuo.
        3. Se computa la desviación estadística del subgrupo respecto al promedio global.
        """
        # Se combinan las condiciones del antecedente y consecuente para formar la regla completa.
        full_rule_items = antecedent + consequent
        
        # Se retorna el valor de penalización si la regla no contiene condiciones.
        if not full_rule_items:
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}

        # Fase 1: Filtrado lógico.
        # Se inicializa una máscara booleana asumiendo que todos los registros son válidos.
        mask = np.ones(len(self.df_discrete), dtype=bool)
        
        # Se iteran las condiciones de la regla para refinar la máscara.
        for var_idx, val_idx in full_rule_items:
            if var_idx >= len(self.var_names):
                continue
            
            col_name = self.var_names[var_idx]
            
            # Se omite la condición si la columna no existe en el dataset.
            if col_name not in self.df_discrete.columns:
                continue
            
            # Se restringe el índice del valor para asegurar que esté dentro de los límites permitidos.
            safe_val = val_idx
            max_val = self.max_indices.get(col_name, 4)
            if safe_val > max_val:
                safe_val = max_val
            if safe_val < 0:
                safe_val = 0
            
            # Se actualiza la máscara conservando solo los registros que coinciden con el valor especificado.
            mask &= (self.df_discrete[col_name] == safe_val)
        
        # Se calcula la cantidad de registros que cumplen todas las condiciones.
        n_matched = mask.sum()
        n_total = len(self.df_discrete)
        
        # Se penaliza si ningún registro cumple la regla.
        if n_matched == 0:
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}
        
        # Se penaliza si la cantidad de registros es insuficiente para tener validez estadística.
        if n_matched < 10:
            return {m: self.PENALTY_VALUE for m in self.METRIC_NAMES}

        # Fase 2: Extracción de datos continuos.
        # Se aplica la máscara resultante sobre el dataset continuo para obtener los valores reales.
        matched_raw = self.df_raw[mask]
        
        # Fase 3: Cálculo estadístico.
        # Se calcula la proporción de cobertura del subgrupo.
        coverage = n_matched / n_total
        # Se aplica una transformación de raíz cuadrada para balancear la importancia del tamaño del grupo.
        coverage_factor = np.sqrt(coverage)
        
        results = {}
        
        # Se procesa cada métrica climática individualmente.
        for col in self.METRIC_NAMES:
            # Se asigna penalización si faltan datos para la métrica.
            if col not in matched_raw.columns or col not in self.global_stats:
                results[col] = self.PENALTY_VALUE
                continue
            
            # Se calcula el promedio aritmético de los valores continuos del subgrupo.
            local_values = matched_raw[col].astype(float)
            local_mean = float(local_values.mean())
            
            # Se recuperan los parámetros estadísticos globales.
            global_mean = self.global_stats[col]['mean']
            global_std = self.global_stats[col]['std']
            
            # Se calcula el puntaje Z (Z-Score) para cuantificar la desviación del subgrupo respecto a la media global.
            if global_std > 0:
                z_score = (local_mean - global_mean) / global_std
            else:
                z_score = 0.0
            
            # Se normaliza la desviación a un rango estandarizado para evitar valores extremos.
            normalized_diff = np.clip(z_score / 3.0, -1.0, 1.0)
            
            # Se ajusta el signo de la desviación según el objetivo de optimización (maximizar o minimizar).
            if self.MAXIMIZE_METRICS[col]:
                # Se mantiene el signo si el objetivo es aumentar el valor.
                improvement = normalized_diff
            else:
                # Se invierte el signo si el objetivo es reducir el valor.
                improvement = -normalized_diff
            
            # Se pondera la mejora obtenida por el factor de cobertura del subgrupo.
            raw_quality = coverage_factor * improvement
            
            # Se transforma el indicador de calidad a un valor de aptitud (fitness) para el algoritmo de optimización.
            # El algoritmo busca minimizar este valor, donde 0.0 es óptimo y 1.0 es pésimo.
            clipped_quality = np.clip(raw_quality, -1.0, 1.0)
            fitness = (1.0 - clipped_quality) / 2.0
            
            # Se asegura que el valor final esté estrictamente dentro del rango permitido.
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
        Se gestiona la obtención de métricas, verificando primero el almacenamiento temporal (caché).
        """
        # Se genera una clave única inmutable basada en los componentes de la regla.
        cache_key = (frozenset(antecedent), frozenset(consequent))
        
        # Se retorna el resultado almacenado si la regla ya fue evaluada previamente.
        if cache_key in self._cache:
            all_metrics = self._cache[cache_key]
        else:
            # Se ejecuta el cálculo completo si la regla es nueva.
            all_metrics = self._calculate_all_metrics(antecedent, consequent)
            # Se guarda el resultado en el almacenamiento temporal.
            self._cache[cache_key] = all_metrics
        
        values = []
        errors = {}
        
        # Se filtran y ordenan los resultados según la lista de objetivos solicitados.
        for metric in objectives:
            canonical = self.get_canonical_name(metric)
            
            if canonical in all_metrics:
                val = all_metrics[canonical]
                # Se valida que el valor numérico sea finito y correcto.
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    values.append(None)
                    errors[metric] = "Valor numérico inválido detectado"
                else:
                    values.append(val)
            else:
                values.append(None)
                errors[metric] = f"La métrica solicitada no se encuentra disponible: {metric}"
        
        return values, errors

    def get_available_metrics(self) -> List[str]:
        """
        Se retorna la lista completa de métricas climáticas que el sistema puede evaluar.
        """
        return self.METRIC_NAMES.copy()

    def get_canonical_name(self, metric_name: str) -> str:
        """
        Se normaliza el nombre de la métrica, convirtiendo alias o variantes al identificador interno estándar.
        """
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
        Se proporciona un diccionario con la metadata técnica y estadística de la métrica solicitada.
        """
        canonical = self.get_canonical_name(metric_name)
        
        descriptions = {
            'co2_emission': 'Emisiones de CO2 (ton/capita) - Objetivo: MINIMIZAR',
            'energy_consumption': 'Consumo energético (kWh) - Objetivo: MINIMIZAR',
            'renewable_share': 'Porcentaje de energía renovable (%) - Objetivo: MAXIMIZAR',
            'industrial_activity_index': 'Índice de actividad industrial (0-100) - Objetivo: MAXIMIZAR',
            'energy_price': 'Precio de energía ($/kWh) - Objetivo: MINIMIZAR'
        }
        
        return {
            'name': canonical,
            'direction': 'maximize' if self.MAXIMIZE_METRICS.get(canonical, False) else 'minimize',
            'range': [0.0, 1.0],
            'description': descriptions.get(canonical, 'Descripción no disponible'),
            'global_stats': self.global_stats.get(canonical, {})
        }
    
    def clear_cache(self):
        """
        Se vacía el almacenamiento temporal para liberar recursos de memoria del sistema.
        """
        self._cache.clear()