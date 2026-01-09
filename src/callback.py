import os
import json
import logging
import numpy as np
import pandas as pd
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from src.representation import RuleIndividual
from src.MOEAD import StuckRunDetected

from src.metrics.scenario1 import Scenario1Metrics
from src.metrics.scenario2 import Scenario2Metrics

class ARMCallback(Callback):
    def __init__(self, config, logger, output_dir):
        super().__init__()
        self.config = config
        self.rule_logger = logger
        self.output_dir = output_dir
        self.interval = config['algorithm'].get('logging_interval', 10)
        self.objectives = config['objectives']['selected']
        stuck_cfg = config['algorithm'].get('stuck_detection', {})
        self.stuck_enabled = stuck_cfg.get('enabled', False)
        self.stuck_window = stuck_cfg.get('window', 10)
        self.stuck_min_new = stuck_cfg.get('min_new', 1)
        self.stuck_hv_window = stuck_cfg.get('hv_window', self.stuck_window)
        self.stuck_hv_tol = stuck_cfg.get('hv_tol', 1e-4)
        self.hv_history = []
        self.no_new_streak = 0
        self.stuck_reason = None
        
        # Setup directories
        self.dirs = {
            'stats': os.path.join(output_dir, 'stats'),
            'populations': os.path.join(output_dir, 'populations'),
            'pareto': os.path.join(output_dir, 'pareto'),
            'discarded': os.path.join(output_dir, 'discarded')
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        # Initialize stats file
        self.stats_file = os.path.join(self.dirs['stats'], 'evolution_stats.csv')
        self.stats_history = []
        
        # Hypervolume Indicator
        # ClimateMetrics retorna valores en [0, 1] donde 0 = óptimo (MINIMIZACIÓN)
        # El ref_point debe ser MAYOR que todos los puntos posibles (nadir point)
        # Usamos 1.1 para dar margen al rango [0, 1]
        self.hv_indicator = HV(ref_point=np.ones(len(self.objectives)) * 1.1)
        
        # Track cumulative discarded count
        self.cumulative_discarded = 0
        self.log = logging.getLogger(__name__)

    def _calculate_arm_metrics(self, antecedent, consequent, problem):
        """
        Calcula métricas ARM adicionales (informativas, no objetivos).
        
        Returns:
            dict con métricas: casual_support, casual_confidence, max_conf,
                              jaccard, cosine, phi, kappa
        """
        arm_metrics = {}
        
        try:
            # Obtener probabilidades básicas usando el DataFrame del problema
            df = problem.metrics.df
            var_names = problem.metrics.var_names
            supports = problem.metrics.supports
            total_rows = len(df)
            
            # Helper para calcular P(items)
            def get_prob(items):
                if not items:
                    return 0.0
                if len(items) == 1:
                    var_idx, val_idx = items[0]
                    if var_idx < len(var_names):
                        var_name = var_names[var_idx]
                        try:
                            return supports['variables'][var_name][str(val_idx)]
                        except KeyError:
                            return 0.0
                    return 0.0
                # Multiple items
                import numpy as np
                mask = np.ones(total_rows, dtype=bool)
                for var_idx, val_idx in items:
                    if var_idx < len(var_names):
                        var_name = var_names[var_idx]
                        if var_name in df.columns:
                            mask &= (df[var_name] == val_idx)
                return mask.sum() / total_rows
            
            # Probabilidades básicas
            p_a = get_prob(antecedent)
            p_c = get_prob(consequent)
            p_ac = get_prob(antecedent + consequent)
            
            p_not_a = 1.0 - p_a
            p_not_c = 1.0 - p_c
            p_not_a_not_c = max(0.0, min(1.0, 1.0 - (p_a + p_c - p_ac)))
            
            # === SCENARIO 1: Casual ARM ===
            # casual_support: P(A ∩ C) + P(¬A ∩ ¬C)
            arm_metrics['casual_support'] = p_ac + p_not_a_not_c
            
            # casual_confidence: 0.5 * [P(C|A) + P(¬C|¬A)]
            if p_a > 0 and p_not_a > 0:
                conf_a_c = p_ac / p_a
                conf_not_a_not_c = p_not_a_not_c / p_not_a
                arm_metrics['casual_confidence'] = 0.5 * (conf_a_c + conf_not_a_not_c)
            else:
                arm_metrics['casual_confidence'] = None
            
            # max_conf: max(P(C|A), P(A|C))
            if p_a > 0 and p_c > 0:
                arm_metrics['max_conf'] = max(p_ac / p_a, p_ac / p_c)
            else:
                arm_metrics['max_conf'] = None
            
            # === SCENARIO 2: Correlation ===
            # jaccard: P(A ∩ C) / P(A ∪ C)
            p_union = p_a + p_c - p_ac
            if p_union > 0:
                arm_metrics['jaccard'] = p_ac / p_union
            else:
                arm_metrics['jaccard'] = None
            
            # cosine: P(A ∩ C) / sqrt(P(A) * P(C))
            import numpy as np
            if p_a > 0 and p_c > 0:
                arm_metrics['cosine'] = p_ac / np.sqrt(p_a * p_c)
            else:
                arm_metrics['cosine'] = None
            
            # phi: (P(AC) - P(A)P(C)) / sqrt(P(A)P(A')P(C)P(C'))
            denom_phi = np.sqrt(p_a * p_not_a * p_c * p_not_c)
            if denom_phi > 0:
                arm_metrics['phi'] = (p_ac - p_a * p_c) / denom_phi
            else:
                arm_metrics['phi'] = None
            
            # kappa: (P(AC) - P(A)P(C)) / max(P(A)(1-P(C)), P(C)(1-P(A)))
            denom_kappa = max(p_a * (1 - p_c), p_c * (1 - p_a))
            if denom_kappa > 0:
                arm_metrics['kappa'] = (p_ac - p_a * p_c) / denom_kappa
            else:
                arm_metrics['kappa'] = None
                
        except Exception as e:
            # Si falla, retornar None para todas las métricas
            arm_metrics = {
                'casual_support': None, 'casual_confidence': None, 'max_conf': None,
                'jaccard': None, 'cosine': None, 'phi': None, 'kappa': None
            }
        
        return arm_metrics

    def notify(self, algorithm):
        # Only run every 'interval' generations or at the last one
        if algorithm.n_gen % self.interval != 0:
            return

        self.log.info("Callback at gen %s | pop=%s", algorithm.n_gen, len(algorithm.pop))
            
        gen = algorithm.n_gen
        pop = algorithm.pop
        
        # 1. Calculate Statistics (on REAL values)
        # Para Climate scenario: ClimateMetrics ya retorna [0, 1] donde 0=óptimo
        # Para otros scenarios: valores fueron negados, hay que revertir
        # Detectamos por el rango de valores
        raw_F = pop.get("F")
        if np.all(raw_F >= 0):
            # Climate scenario: valores ya están en [0, 1], no negar
            real_F = raw_F.copy()
        else:
            # Otros scenarios: valores negativos, negar para obtener positivos
            real_F = -raw_F
        
        stats = {'generation': gen}
        
        # Objective Stats
        for i, obj_name in enumerate(self.objectives):
            values = real_F[:, i]
            stats[f'{obj_name}_min'] = np.min(values)
            stats[f'{obj_name}_max'] = np.max(values)
            stats[f'{obj_name}_mean'] = np.mean(values)
            
        # Hypervolume (Population)
        hv_value = 0.0
        try:
            hv_value = self.hv_indicator(pop.get("F"))
            stats['hypervolume'] = hv_value
        except Exception as e:
            stats['hypervolume'] = 0.0
            print(f"Warning: HV calculation failed at gen {gen}: {e}")
            
        # Track Operator Probabilities
        if hasattr(algorithm, 'mating') and hasattr(algorithm.mating, 'mutation'):
            stats['mutation_prob'] = algorithm.mating.mutation.prob
        if hasattr(algorithm, 'mating') and hasattr(algorithm.mating, 'crossover'):
            stats['crossover_prob'] = algorithm.mating.crossover.prob
            
        # Track Cumulative Discarded
        # We get the count from the current logger state before clearing
        current_discarded_count = sum(entry['total_count'] for entry in self.rule_logger.storage.values())
        self.cumulative_discarded += current_discarded_count
        stats['total_discarded_cumulative'] = self.cumulative_discarded
        stats['discarded_this_interval'] = current_discarded_count

        # Counters from algorithm (dedup diagnostics)
        counters = getattr(algorithm, 'last_counters', {}) or {}
        stats['new_replacements'] = counters.get('new', 0)
        stats['skip_new_duplicates'] = counters.get('skip_new', 0)
        stats['skip_pop_duplicates'] = counters.get('skip_pop', 0)

        # Stuck detection (callback-level) combining replacements and HV plateau
        if self.stuck_enabled:
            new_count = stats['new_replacements']
            if new_count < self.stuck_min_new:
                self.no_new_streak += 1
            else:
                self.no_new_streak = 0

            self.hv_history.append(hv_value)
            hv_plateau = False
            if len(self.hv_history) >= self.stuck_hv_window:
                window_vals = self.hv_history[-self.stuck_hv_window:]
                hv_plateau = (max(window_vals) - min(window_vals)) < self.stuck_hv_tol

            if self.no_new_streak >= self.stuck_window or hv_plateau:
                reason_parts = []
                if self.no_new_streak >= self.stuck_window:
                    reason_parts.append(f"{self.no_new_streak} gens sin nuevos (<{self.stuck_min_new})")
                if hv_plateau:
                    reason_parts.append(
                        f"HV plano en ventana {self.stuck_hv_window} (Δ<{self.stuck_hv_tol})"
                    )
                self.stuck_reason = " | ".join(reason_parts)
                msg = f"Stuck detection: {self.stuck_reason} @gen {gen}"
                self.log.warning(msg)
                print(msg)
                if hasattr(algorithm, 'termination') and hasattr(algorithm.termination, 'force_termination'):
                    algorithm.termination.force_termination = True
                else:
                    raise StuckRunDetected(msg)
            
        self.stats_history.append(stats)
        
        # Save Stats to CSV incrementally
        pd.DataFrame(self.stats_history).to_csv(self.stats_file, index=False)
        
        # 2. Save Population (All Individuals)
        self._save_population(pop, gen, 'population', algorithm, hv_value)
        
        # 3. Save Pareto Front (Non-dominated)
        # Pymoo's algorithm.opt holds the current best non-dominated solutions found so far
        opt = algorithm.opt
        if opt is not None and len(opt) > 0:
            self._save_population(opt, gen, 'pareto', algorithm, hv_value)
            
        # 4. Save Discarded Rules (Differential Snapshot)
        self._save_discarded(gen)
        
        # Clear logger for next interval (Option 3: Hybrid)
        self.rule_logger.storage = {}

    def _save_population(self, pop, gen, type_name, algorithm, hv_value=None):
        """
        Saves a population to CSV.
        Decodes individuals and includes metrics + ARM metrics adicionales.
        """
        data = []
        raw_F = pop.get("F")
        # Detectar si es Climate scenario (valores positivos) o tradicional (negativos)
        if np.all(raw_F >= 0):
            real_F = raw_F.copy()
        else:
            real_F = -raw_F
        X = pop.get("X")
        
        # Accessing metadata from problem
        problem = algorithm.problem if hasattr(algorithm, 'problem') else None
        
        for i, ind in enumerate(pop):
            row = {'generation': gen, 'id': i}
            
            if hv_value is not None:
                row['hypervolume'] = hv_value
            
            # Metrics (objetivos optimizados)
            for j, obj_name in enumerate(self.objectives):
                row[obj_name] = real_F[i, j]
                
            # Decode Rule
            if problem:
                temp_ind = RuleIndividual(problem.metadata)
                temp_ind.X = X[i]
                ant_str, con_str = temp_ind.decode_parts()
                ant_items, con_items = temp_ind.get_rule_items()
                
                row['antecedent'] = ant_str
                row['consequent'] = con_str
                row['rule'] = f"{ant_str} => {con_str}"
                
                # === MÉTRICAS ARM ADICIONALES (informativas) ===
                arm_metrics = self._calculate_arm_metrics(ant_items, con_items, problem)
                for metric_name, metric_val in arm_metrics.items():
                    row[f'arm_{metric_name}'] = metric_val
                
                # Save raw genes
                row['encoded_rule'] = json.dumps(X[i].tolist())
            
            data.append(row)
            
        df = pd.DataFrame(data)
        filename = os.path.join(self.dirs['populations' if type_name == 'population' else 'pareto'], 
                                f'{type_name}_gen_{gen}.csv')
        df.to_csv(filename, index=False)

    def _save_discarded(self, gen):
        """
        Flushes the DiscardedRulesLogger to disk.
        """
        # The logger is passed in __init__
        # We access its storage
        storage = self.rule_logger.storage
        
        if not storage:
            return
            
        data = []
        for key, entry in storage.items():
            row = {
                'rule': entry['rule_decoded'],
                'total_count': entry['total_count'],
                'reasons': json.dumps(entry['reasons'])
            }
            # Add metrics if available (from the last discarded instance of this rule)
            if entry.get('metrics'):
                for m, v in entry['metrics'].items():
                    row[f"metric_{m}"] = v
            
            data.append(row)
            
        df = pd.DataFrame(data)
        filename = os.path.join(self.dirs['discarded'], f'discarded_gen_{gen}.csv')
        df.to_csv(filename, index=False)
        
        # Logger is cleared in notify() for Option 3 (Hybrid)
