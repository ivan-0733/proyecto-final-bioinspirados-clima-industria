"""
Este módulo implementa el motor central de optimización (MOEA/D) aplicado a la Minería de Reglas de Asociación.

El objetivo principal aquí es tomar decisiones inteligentes para encontrar reglas que equilibren múltiples metas conflictivas,
como reducir emisiones de CO2 y mantener la actividad industrial.

Integra tres componentes clave que trabajan en conjunto:
- AdaptiveMOEAD: El cerebro evolutivo que aprende y ajusta sus estrategias (mutación/cruce) mientras busca soluciones.
- ARMProblem: La definición del examen; aquí es donde se evalúa qué tan buena es una regla comparándola con los datos climáticos reales.
- Operadores Personalizados: Las herramientas que modifican las reglas (cortar, pegar, cambiar) para crear nuevas variantes.
"""

import numpy as np
import math
import logging
import time
from pathlib import Path
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff as Tcheb
from pymoo.decomposition.weighted_sum import WeightedSum as WS
from pymoo.termination.default import DefaultMultiObjectiveTermination

from src.representation import RuleIndividual
from src.operators import DiploidNPointCrossover, ARMSampling, PregeneratedSampling
from src.operators.mutation_factory import create_mutation
from src.validator import ARMValidator
from src.metrics import MetricsFactory
from src.loggers import DiscardedRulesLogger
from src.optimization import AdaptiveControl, ProbabilityConfig, StuckDetector
from src.core.exceptions import MOEADDeadlockError

def get_H_from_N(N, M):
    """
    Se calcula matemáticamente cuántas divisiones o 'direcciones de referencia' se necesitan 
    para cubrir uniformemente el espacio de objetivos.
    
    Imagina que tenemos un pastel (el espacio de soluciones) y queremos cortarlo en rebanadas iguales 
    para asegurarnos de buscar en todas partes (bajas emisiones, alta energía, etc.).
    Esta función encuentra el número de cortes (H) necesario para tener aproximadamente N rebanadas 
    cuando tenemos M objetivos distintos.
    """
    H = 1
    while True:
        count = math.comb(H + M - 1, M - 1)
        if count >= N:
            # Se verifica si el número anterior estaba más cerca del objetivo deseado
            prev_count = math.comb((H - 1) + M - 1, M - 1)
            if abs(prev_count - N) < abs(count - N):
                return H - 1
            return H
        H += 1

from pymoo.core.population import Population

import sys
from types import SimpleNamespace


logger = logging.getLogger(__name__)


class AdaptiveMOEAD(MOEAD):
    """
    Se define una versión personalizada del algoritmo MOEA/D que tiene la capacidad de adaptarse.
    A diferencia de un algoritmo estático, este observa su propio éxito y ajusta qué tanto 
    cambia las reglas (mutación) o qué tanto las mezcla (cruce) para evitar quedarse atascado.
    """
    def __init__(self, mutation_adapter_config, crossover_adapter_config, n_replace=2, prob_neighbor_mating=0.9, stuck_config=None, **kwargs):
        super().__init__(prob_neighbor_mating=prob_neighbor_mating, **kwargs)
        
        # Se configura el "cerebro adaptativo" (AdaptiveControl).
        # Este componente recibe instrucciones sobre los límites permitidos para las probabilidades de mutación y cruce.
        # Su trabajo es monitorear si las nuevas reglas son mejores que las anteriores y ajustar las probabilidades en consecuencia.
        self.adaptive_control = AdaptiveControl(
            mutation_config=ProbabilityConfig(
                initial=mutation_adapter_config['initial'],
                min_val=mutation_adapter_config['min'],
                max_val=mutation_adapter_config['max']
            ),
            crossover_config=ProbabilityConfig(
                initial=crossover_adapter_config['initial'],
                min_val=crossover_adapter_config['min'],
                max_val=crossover_adapter_config['max']
            )
        )
        
        # Se configura el "detector de estancamiento" (StuckDetector).
        # Este es un mecanismo de seguridad. Si el algoritmo pasa mucho tiempo (definido en stuck_config)
        # sin encontrar ninguna regla nueva valiosa para el clima, este detector levanta una alerta para detener o reiniciar el proceso.
        stuck_cfg = stuck_config or {"enabled": False}
        if stuck_cfg.get("enabled", False):
            self.stuck_detector = StuckDetector(
                max_runtime_minutes=stuck_cfg.get("max_runtime_minutes"),
                window_size=stuck_cfg.get("window", 10),
                min_new_per_window=stuck_cfg.get("min_new", 5),
                hv_tolerance=stuck_cfg.get("hv_tol", 1e-6),
                hv_period=stuck_cfg.get("hv_period", 20)
            )
        else:
            self.stuck_detector = None
        
        self.n_replace = n_replace
        self.prob_neighbor_mating = prob_neighbor_mating
        self.current_gen = 0

    def _infill(self):
        # Este método se deja vacío intencionalmente porque controlamos el flujo manualmente en _next.
        return None

    def _advance(self, infills=None):
        # Se avanza al siguiente paso del ciclo evolutivo.
        self._next()

    def _next(self):
        """
        Se ejecuta el ciclo principal de una generación evolutiva.
        Aquí es donde ocurre la magia de la optimización paso a paso.
        """
        # 1. Se toma una fotografía del estado actual.
        # Guardamos la población actual (old_X) para poder comparar al final si logramos algún cambio.
        # También creamos un registro de 'firmas' (sig_set) para evitar crear reglas duplicadas.
        old_X = self.pop.get("X").copy()
        sig_set = {tuple(x.flatten()) for x in old_X}
        dup_skips = 0
        curpop_dup_skips = 0
        replacements_accepted = 0
        self.last_counters = {"new": 0, "skip_new": 0, "skip_pop": 0}
        
        # 2. Se inicia el proceso manual de evolución.
        pop = self.pop
        
        # Si aún no tenemos un punto ideal (los mejores valores teóricos alcanzados hasta ahora), lo inicializamos.
        if not hasattr(self, 'ideal_point') or self.ideal_point is None:
            self.ideal_point = np.min(pop.get("F"), axis=0)

        # Se crea un orden aleatorio para visitar a cada individuo de la población.
        perm = np.random.permutation(len(pop))
        
        for i in perm:
            # a) Selección de Vecinos.
            # En MOEA/D, no cruzamos a cualquiera con cualquiera. Cruzamos a individuos que tienen objetivos similares
            # (sus 'vecinos'). Esto ayuda a refinar soluciones locales antes de buscar globalmente.
            nbs = self.neighbors[i]
            
            # b) Selección para el Apareamiento.
            # Decidimos si cruzamos con un vecino cercano (alta probabilidad) o con alguien aleatorio de toda la población (baja probabilidad).
            if np.random.random() < self.prob_neighbor_mating:
                parent_indices = np.random.choice(nbs, 2, replace=False)
            else:
                parent_indices = np.random.choice(len(pop), 2, replace=False)
            
            parents = pop[parent_indices]
            
            # c) Reproducción (Cruce y Mutación).
            # Se extrae el ADN (genoma) de los padres seleccionados.
            p_X = np.array([parents[0].X, parents[1].X]) # (2, n_var)
            p_X_input = p_X[None, :, :] # (1, 2, n_var)
            
            # Se realiza el CRUCE: Mezclamos la información de los dos padres para crear hijos.
            off_X_pair = self.mating.crossover._do(self.problem, p_X_input)[0] # (2, n_var)
            
            # Procesamos ambos hijos resultantes.
            for off_X_raw in off_X_pair:
                # Se realiza la MUTACIÓN: Hacemos pequeños cambios aleatorios en el hijo.
                # Esto es vital para descubrir cosas nuevas que no estaban en los padres.
                off_X_mut = self.mating.mutation._do(self.problem, off_X_raw[None, :])[0] # (n_var,)
                
                # d) Evaluación.
                # Ahora que tenemos una nueva regla candidata (el hijo mutado), preguntamos: "¿Qué tan buena es?".
                # Creamos una población temporal de uno solo y llamamos al evaluador (Problem).
                off_pop = Population.new(X=np.array([off_X_mut]))
                self.evaluator.eval(self.problem, off_pop)
                off = off_pop[0]

                # Se verifica que no sea una regla duplicada que ya exista o acabemos de crear.
                off_sig = tuple(off.X.flatten())
                if off_sig in sig_set:
                    dup_skips += 1
                    continue

                # Se verifica también contra la población actual para asegurar diversidad.
                current_X = pop.get("X")
                if np.any(np.all(current_X == off.X, axis=1)):
                    curpop_dup_skips += 1
                    continue
                
                # e) Actualización del Punto Ideal.
                # Si este nuevo hijo logró un valor récord en algún objetivo (ej. el CO2 más bajo visto), actualizamos el punto ideal.
                self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)
                
                # f) Reemplazo de Vecinos (Descomposición).
                # Comparamos al hijo con sus vecinos usando una fórmula matemática (Tchebycheff o PBI) que combina los objetivos.
                weights = self.ref_dirs[nbs]
                
                off_fv = self.decomposition.do(off.F, weights, ideal_point=self.ideal_point)
                nbs_F = pop[nbs].get("F")
                nbs_fv = self.decomposition.do(nbs_F, weights, ideal_point=self.ideal_point)
                
                # Identificamos a qué vecinos supera este nuevo hijo.
                improved_idx = np.where(off_fv < nbs_fv)[0]
                
                # Si supera a muchos, elegimos solo a unos pocos para reemplazar (para no eliminar demasiada diversidad de golpe).
                if len(improved_idx) > self.n_replace:
                    np.random.shuffle(improved_idx)
                    improved_idx = improved_idx[:self.n_replace]
                
                # Realizamos el reemplazo: La regla vieja se descarta y la nueva toma su lugar.
                for idx in improved_idx:
                    pop_idx = nbs[idx]
                    
                    replace_sig = tuple(off.X.flatten())
                    if replace_sig in sig_set:
                        continue
                    pop[pop_idx] = off
                    sig_set.add(replace_sig)
                    replacements_accepted += 1

        # Se guardan contadores para saber qué pasó en esta generación (cuántos entraron, cuántos se rechazaron por duplicados).
        self.last_counters = {
            "new": replacements_accepted,
            "skip_new": dup_skips,
            "skip_pop": curpop_dup_skips,
        }
        
        self.current_gen += 1
        
        # Se muestra una barra de progreso en la consola para que el usuario sepa que el sistema está trabajando.
        total_gen = self.termination.n_max_gen if hasattr(self.termination, 'n_max_gen') else 300 
        percent = (self.current_gen / total_gen) * 100
        bar_length = 30
        filled_length = int(bar_length * self.current_gen // total_gen)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% (Gen {self.current_gen}/{total_gen})')
        sys.stdout.flush()
        if self.current_gen >= total_gen:
            print() 
        sys.stdout.write(f' | new:{replacements_accepted} skip_new:{dup_skips} skip_pop:{curpop_dup_skips}')
        sys.stdout.flush()

        # 3. Cálculo de Tasa de Éxito y Adaptación.
        # Aquí el "cerebro adaptativo" revisa el trabajo hecho.
        new_X = self.pop.get("X")
        
        # Contamos cuántos individuos son diferentes respecto al inicio de la generación.
        diffs = (old_X != new_X)
        changes_per_ind = np.sum(diffs, axis=1)
        n_replacements = np.sum(changes_per_ind > 0)
        
        # Le informamos al controlador adaptativo para que calcule nuevas probabilidades.
        self.adaptive_control.record_generation(n_replacements, len(self.pop))
        new_mut_prob, new_cx_prob = self.adaptive_control.update_probabilities()
        
        # Se actualizan las herramientas (operadores) con las nuevas probabilidades.
        self.mating.mutation.prob = new_mut_prob
        self.mating.crossover.prob = new_cx_prob
        
        # Se verifica si estamos estancados.
        if self.stuck_detector:
            self.stuck_detector.record_generation(replacements_accepted, None)
            
            try:
                self.stuck_detector.raise_if_stuck(self.current_gen)
            except Exception as e:
                logger.warning(f"Stuck detected: {e}")
                if hasattr(self, "termination") and hasattr(self.termination, "force_termination"):
                    self.termination.force_termination = True
                else:
                    raise


import numpy as np
import logging
from pymoo.core.problem import Problem
from src.representation import RuleIndividual

logger = logging.getLogger(__name__)


class ARMProblem(Problem):
    """
    Se define el Problema de Optimización para las Reglas de Asociación.
    
    Esta clase es el puente entre los números que maneja el algoritmo (genotipos) y 
    el significado real de esos números (reglas climáticas).
    
    Sus responsabilidades son:
    - Definir cuántas variables tiene el ADN de una regla.
    - Definir cuántos objetivos estamos persiguiendo (ej. 5 objetivos climáticos).
    - Evaluar cada solución candidata para asignarle una calificación.
    """
    
    # Se definen los rangos para normalizar métricas comunes si es necesario.
    METRIC_RANGES = {
        # Scenario 1 - Casual ARM
        'casual-supp': (0.0, 1.0),
        'casual-conf': (0.0, 1.0),
        'maxConf': (0.0, 1.0),
        'casual_support': (0.0, 1.0),
        'casual_confidence': (0.0, 1.0),
        'max_conf': (0.0, 1.0),
        
        # Scenario 2 - Correlation
        'jaccard': (0.0, 1.0),
        'cosine': (0.0, 1.0),
        'phi': (-1.0, 1.0),
        'phi_coefficient': (-1.0, 1.0),
        'kappa': (-1.0, 1.0),
        'k_measure': (-1.0, 1.0),
        
        # Escenario Climático - 5 Objetivos
        # ClimateMetrics ya entrega valores listos entre 0 y 1, por lo que la normalización externa es mínima.
        'co2_emission': (0.0, 1.0),
        'energy_consumption': (0.0, 1.0),
        'renewable_share': (0.0, 1.0),
        'industrial_activity_index': (0.0, 1.0),
        'energy_price': (0.0, 1.0)
    }
    
    def __init__(
        self, 
        metadata: dict,
        supports: dict, 
        df, 
        config: dict,
        validator,
        metrics,
        logger_instance
    ):
        """
        Se configura el problema recibiendo todos los datos necesarios.
        
        Args:
            metadata: La estructura que nos dice qué significa cada gen (país, año, etc.).
            supports: Datos precalculados para rapidez.
            df: El dataset principal.
            config: La configuración que nos dice qué objetivos importan hoy.
            validator: El juez que decide si una regla tiene sentido lógico.
            metrics: El experto (ej. ClimateMetrics) que calcula los valores reales de CO2, etc.
            logger_instance: Donde anotamos las reglas que salieron mal.
        """
        self.metadata = metadata
        self.supports = supports
        self.config = config
        self.objectives = config['objectives']['selected']
        self.n_obj = len(self.objectives)
        self.logger = logger_instance
        
        # Se calcula el tamaño del genoma. Cada característica del dataset necesita espacio en el ADN.
        dummy = RuleIndividual(metadata=metadata)
        self.n_var = 2 * dummy.num_genes
        
        self.validator = validator
        self.metrics = metrics
        
        # Se detecta si estamos trabajando en el escenario climático para aplicar su lógica específica.
        self.is_climate_scenario = hasattr(metrics, 'METRIC_NAMES')
        
        # Se inicializa la clase base de pymoo con las dimensiones calculadas.
        super().__init__(
            n_var=self.n_var, 
            n_obj=self.n_obj, 
            n_ieq_constr=0,
            xl=0, 
            xu=1
        )
        
        logger.info(
            f"Problema inicializado con {self.n_var} variables y {self.n_obj} objetivos. Modo Clima: {self.is_climate_scenario}"
        )
    
    def _normalize_metric(self, value: float, metric_name: str) -> float:
        """
        Se transforma un valor crudo a una escala estándar de 0 a 1.
        Esto es necesario porque el algoritmo trabaja mejor cuando todos los objetivos tienen escalas similares.
        """
        if metric_name not in self.METRIC_RANGES:
            return value
        
        min_val, max_val = self.METRIC_RANGES[metric_name]
        
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Se realiza la evaluación de un grupo de individuos (reglas).
        
        El proceso es:
        1. Recibir una matriz de números (genotipos).
        2. Convertir esos números en reglas legibles (País X, Industria Y).
        3. Validar si la regla tiene sentido (no vacía, no contradictoria).
        4. Calcular sus métricas climáticas usando el dataset raw.
        5. Guardar los resultados en la matriz de fitness 'F'.
        """
        n_pop = x.shape[0]
        F = np.zeros((n_pop, self.n_obj))
        
        for i in range(n_pop):
            ind_genome = x[i]
            
            # Se reconstruye la regla a partir de los genes.
            temp_ind = RuleIndividual(self.metadata)
            temp_ind.X = ind_genome
            ant, con = temp_ind.get_rule_items()
            
            # 1. Validación Estructural.
            # Preguntamos: "¿Es esta una regla válida?".
            is_valid, reason, _ = self.validator.validate(ant, con)
            
            if not is_valid:
                # Si la regla está rota (ej. consecuente vacío), la registramos y la penalizamos severamente.
                self.logger.log(temp_ind, f"invalid_structure:{reason}")
                
                # Se asigna un valor de 2.0 (muy malo, ya que buscamos minimizar hacia 0).
                F[i, :] = 2.0
                continue

            # 2. Cálculo de Métricas Reales.
            # Delegamos al experto (metrics engine) el cálculo sobre los datos reales.
            vals, errors = self.metrics.get_metrics(ant, con, self.objectives)
            
            # 3. Procesamiento de Objetivos.
            obj_values = []
            
            for metric_name, val in zip(self.objectives, vals):
                if val is None:
                    # Si el cálculo falló (ej. división por cero), penalizamos.
                    obj_values.append(2.0)
                else:
                    if self.is_climate_scenario:
                        # ===== LÓGICA CLIMÁTICA =====
                        # En el escenario de clima, las métricas ya vienen preparadas para minimización.
                        # El cálculo interno de ClimateMetrics ya invirtió los valores que se maximizan
                        # y normalizó todo al rango [0, 1], donde 0 es el ideal.
                        obj_values.append(val)
                    else:
                        # ===== OTROS ESCENARIOS (Diabetes, etc.) =====
                        # Aquí seguimos la lógica estándar: normalizamos y negamos si es necesario.
                        normalized = self._normalize_metric(val, metric_name)
                        obj_values.append(-normalized)
            
            # Se asignan los valores calculados al individuo correspondiente.
            F[i, :] = obj_values
        
        # Se guarda la matriz de resultados en la salida estándar de pymoo.
        out["F"] = F

    def decode_individual(self, genome: np.ndarray) -> dict:
        """
        Se traduce un genoma numérico a un formato legible por humanos.
        Útil para reportes y logs, convirtiendo [0, 1, 5] en "País=Alemania".
        """
        temp_ind = RuleIndividual(self.metadata)
        temp_ind.X = genome
        ant, con = temp_ind.get_rule_items()
        
        # Se recuperan los nombres reales desde la metadata.
        feature_order = self.metadata.get('feature_order', [])
        variables = self.metadata.get('variables', {})
        
        def items_to_str(items):
            parts = []
            for var_idx, val_idx in items:
                if var_idx < len(feature_order):
                    var_name = feature_order[var_idx]
                    var_meta = variables.get(var_name, {})
                    labels = var_meta.get('labels', [])
                    
                    if val_idx < len(labels):
                        val_str = labels[val_idx]
                    else:
                        val_str = str(val_idx)
                    
                    parts.append(f"{var_name}={val_str}")
                else:
                    parts.append(f"var{var_idx}={val_idx}")
            return parts
        
        ant_str = items_to_str(ant)
        con_str = items_to_str(con)
        
        return {
            'antecedent': ant,
            'consequent': con,
            'antecedent_str': ant_str,
            'consequent_str': con_str,
            'rule_str': f"{' AND '.join(ant_str)} => {' AND '.join(con_str)}"
        }


class ARMProblemV2(ARMProblem):
    """
    Se define una variante del problema que lleva contabilidad extra de validaciones.
    Útil para diagnóstico si queremos saber cuántas reglas inválidas estamos generando.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_count = 0
        self.valid_count = 0
        self.invalid_count = 0
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Se evalúa igual que el padre, pero contando éxitos y fallos."""
        self.eval_count += x.shape[0]
        
        # Se llama a la evaluación original.
        super()._evaluate(x, out, *args, **kwargs)
        
        # Se cuentan cuántos tuvieron penalización (inválidos) y cuántos no.
        F = out["F"]
        valid_mask = F[:, 0] < 2.0
        self.valid_count += np.sum(valid_mask)
        self.invalid_count += np.sum(~valid_mask)
    
    def get_stats(self) -> dict:
        """Se retornan las estadísticas acumuladas."""
        return {
            'total_evaluations': self.eval_count,
            'valid': self.valid_count,
            'invalid': self.invalid_count,
            'validity_rate': self.valid_count / max(1, self.eval_count)
        }

class MOEAD_ARM:
    """
    Se define la clase orquestadora principal.
    Esta clase actúa como el director de orquesta que prepara todo antes del concierto.
    
    Sus tareas son:
    1. Preparar los datos (Discretos y Continuos).
    2. Configurar el motor de métricas (ClimateMetrics).
    3. Inicializar el algoritmo genético y sus operadores.
    4. Ejecutar la optimización y manejar errores.
    """
    def __init__(self, config, data_context):
        self.config = config
        self.data = data_context # Diccionario con df, supports, metadata
        
        # Se determina qué escenario estamos corriendo (ej. clima con 5 objetivos).
        scenario = self.config['experiment'].get('scenario', 'scenario_1')
        
        # Se inicializan los componentes usando la Fábrica de Métricas.
        # Es vital pasar 'raw_df' para el escenario climático, ya que ahí residen los valores reales
        # necesarios para calcular el impacto ambiental verdadero.
        raw_df = self.data.get('raw_df', None)
        
        self.metrics = MetricsFactory.create_metrics(
            scenario_name=scenario,
            dataframe=self.data['df'],
            supports_dict=self.data['supports'],
            metadata=self.data['metadata'],
            raw_dataframe=raw_df,
            config=self.config
        )
        
        self.validator = ARMValidator(
            config=self.config,
            metrics_engine=self.metrics,
            metadata=self.data['metadata']
        )
        
        self.logger = DiscardedRulesLogger()
        
        self.problem = ARMProblem(
            metadata=self.data['metadata'],
            supports=self.data['supports'],
            df=self.data['df'],
            config=self.config,
            validator=self.validator,
            metrics=self.metrics,
            logger_instance=self.logger
        )
        
    def run(self, callback=None):
        # Se leen los parámetros del algoritmo (tamaño de población, generaciones).
        alg_config = self.config['algorithm']
        target_pop_size = alg_config['population_size']
        n_gen = alg_config['generations']
        
        # Se configuran las Direcciones de Referencia.
        # Estas actúan como guías para descomponer el problema multiobjetivo en subproblemas simples.
        # Calculamos H para distribuir los pesos uniformemente según Das-Dennis.
        H = get_H_from_N(target_pop_size, self.problem.n_obj)
        ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=H)
        actual_pop_size = ref_dirs.shape[0]
        
        print(f"MOEA/D inicializado con {actual_pop_size} direcciones de referencia (Objetivo: {target_pop_size})")
        
        # Se configuran los Operadores Genéticos.
        
        # 1. Inicialización (Sampling):
        # Intentamos usar reglas pre-generadas si existen para empezar con buen pie.
        init_config = alg_config.get('initialization', {})
        use_pregenerated = init_config.get('use_pregenerated', True)
        pregenerated_path = Path("data/processed/pregenerated/valid_rules_1m.csv")
        
        if use_pregenerated and pregenerated_path.exists():
            print(f"✓ Usando reglas pre-generadas desde {pregenerated_path}")
            sampling = PregeneratedSampling(
                metadata=self.data['metadata'],
                csv_path=str(pregenerated_path),
                allow_duplicates=True
            )
        else:
            if use_pregenerated:
                print(f"⚠ No se encontraron reglas pre-generadas en {pregenerated_path}")
            print(f"→ Usando inicialización aleatoria (más lento pero explora más)")
            max_init_attempts = init_config.get('max_attempts', 10000)
            sampling = ARMSampling(
                metadata=self.data['metadata'],
                validator=self.validator,
                logger=self.logger,
                max_attempts=max_init_attempts
            )
        
        # 2. Cruce (Crossover):
        # Configuramos cómo se combinan dos reglas padres.
        crossover = DiploidNPointCrossover(
            prob=alg_config['operators']['crossover']['probability']['initial']
        )
        
        # 3. Mutación (Mutation):
        # Configuramos cómo se alteran las reglas aleatoriamente.
        mutation_type = alg_config['operators']['mutation'].get('method', 'mixed')
        mutation = create_mutation(
            mutation_type=mutation_type,
            metadata=self.data['metadata'],
            validator=self.validator,
            logger=self.logger,
            config=alg_config['operators']['mutation']
        )
        
        # Se configura el Método de Descomposición (ej. PBI).
        # Esto decide matemáticamente cómo agregamos los 5 objetivos en un solo valor escalar para cada dirección.
        decomp_config = alg_config['decomposition']
        method = decomp_config['method'].lower()
        params = decomp_config.get('params', {})
        
        if method == 'pbi':
            decomposition = PBI(theta=params.get('theta', 5.0))
        elif method == 'tchebycheff' or method == 'tcheb':
            decomposition = Tcheb()
        elif method == 'weighted_sum' or method == 'ws':
            decomposition = WS()
        else:
            print(f"Advertencia: Método de descomposición desconocido '{method}'. Usando PBI por defecto.")
            decomposition = PBI()

        # Se instancia el Algoritmo Completo.
        algorithm = AdaptiveMOEAD(
            mutation_adapter_config=alg_config['operators']['mutation']['probability'],
            crossover_adapter_config=alg_config['operators']['crossover']['probability'],
            ref_dirs=ref_dirs,
            n_neighbors=alg_config['neighborhood']['size'],
            n_replace=alg_config['neighborhood'].get('replacement_size', 2),
            decomposition=decomposition,
            prob_neighbor_mating=alg_config['neighborhood']['selection_probability'],
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            stuck_config=alg_config.get('stuck_detection')
        )
        
        # Se configura el Criterio de Terminación.
        # Definimos cuándo debe detenerse el algoritmo (por número de generaciones o si deja de mejorar).
        termination_config = alg_config.get('termination', {})
        
        if termination_config.get('enabled', False):
            print(f"Detención temprana habilitada (ftol={termination_config.get('ftol')}, period={termination_config.get('period')})")
            termination = DefaultMultiObjectiveTermination(
                xtol=1e-8,
                cvtol=1e-6,
                ftol=termination_config.get('ftol', 0.0001),
                period=termination_config.get('period', 30),
                n_max_gen=n_gen,
                n_max_evals=1000000
            )
        else:
            termination = ('n_gen', n_gen)

        # Ejecución de la Optimización.
        try:
            res = minimize(
                self.problem,
                algorithm,
                termination,
                seed=self.config['experiment']['random_seed'],
                callback=callback,
                verbose=False
            )
        except MOEADDeadlockError as e:
            # Si el detector de estancamiento detiene el proceso, capturamos el error para guardar lo que llevamos.
            logger.warning("Optimización detenida por el detector de estancamiento: %s", e)
            print(f"\n[!] Optimización detenida: {e}")
            # Se recupera la mejor población encontrada hasta el momento del fallo.
            current_opt = algorithm.opt if hasattr(algorithm, 'opt') and algorithm.opt is not None else algorithm.pop
            res = SimpleNamespace(opt=current_opt)
        except Exception as e:
            logger.exception("Optimización abortada por error inesperado: %s", e)
            raise
        
        return res