""" 
This module implements the MOEA/D algorithm for multi-objective optimization
applied to Association Rule Mining (ARM).

It integrates custom components:
- AdaptiveMOEAD: Extends pymoo's MOEA/D with 1/5 Rule for adaptive mutation.
- ARMProblem: Defines the optimization problem (objectives, evaluation).
- Custom Operators: Mutation, Crossover, Sampling.
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

def get_H_from_N(N, M):
    """
    Calculates the number of partitions H for Das-Dennis weights 
    given a target population size N and number of objectives M.
    Returns H such that nCr(H+M-1, M-1) is closest to N.
    """
    H = 1
    while True:
        count = math.comb(H + M - 1, M - 1)
        if count >= N:
            # Check if previous H was closer
            prev_count = math.comb((H - 1) + M - 1, M - 1)
            if abs(prev_count - N) < abs(count - N):
                return H - 1
            return H
        H += 1

from pymoo.core.population import Population

import sys
from types import SimpleNamespace


logger = logging.getLogger(__name__)


class StuckRunDetected(Exception):
    """Raised when the stuck detector decides to stop the run early."""
    pass

class AdaptiveMOEAD(MOEAD):
    """
    Custom MOEA/D with adaptive probability control and stuck detection.
    """
    def __init__(self, mutation_adapter_config, crossover_adapter_config, n_replace=2, prob_neighbor_mating=0.9, stuck_config=None, **kwargs):
        super().__init__(prob_neighbor_mating=prob_neighbor_mating, **kwargs)
        
        # Adaptive control for mutation/crossover probabilities
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
        
        # Stuck detection
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
        return None

    def _advance(self, infills=None):
        self._next()

    def _next(self):
        # 1. Snapshot current population (X) to detect changes
        old_X = self.pop.get("X").copy()
        # Track signatures to block duplicates during this generation
        sig_set = {tuple(x.flatten()) for x in old_X}
        dup_skips = 0
        curpop_dup_skips = 0
        replacements_accepted = 0
        self.last_counters = {"new": 0, "skip_new": 0, "skip_pop": 0}
        
        # 2. Manual MOEA/D Step (Bypassing pymoo generator)
        pop = self.pop
        
        # Initialize ideal point if needed
        if not hasattr(self, 'ideal_point') or self.ideal_point is None:
            self.ideal_point = np.min(pop.get("F"), axis=0)

        # Random permutation
        perm = np.random.permutation(len(pop))
        
        for i in perm:
            # a) Select Neighbors
            nbs = self.neighbors[i]
            
            # b) Mating Selection
            if np.random.random() < self.prob_neighbor_mating:
                parent_indices = np.random.choice(nbs, 2, replace=False)
            else:
                parent_indices = np.random.choice(len(pop), 2, replace=False)
            
            parents = pop[parent_indices]
            
            # c) Mating (Crossover + Mutation)
            # Crossover
            p_X = np.array([parents[0].X, parents[1].X]) # (2, n_var)
            p_X_input = p_X[None, :, :] # (1, 2, n_var)
            
            # Call _do directly to bypass pymoo's object handling
            # Crossover returns (n_matings, n_parents, n_var) -> (1, 2, n_var)
            off_X_pair = self.mating.crossover._do(self.problem, p_X_input)[0] # (2, n_var)
            
            # Process BOTH offspring (User Requirement)
            for off_X_raw in off_X_pair:
                # Mutation
                # Call _do directly
                off_X_mut = self.mating.mutation._do(self.problem, off_X_raw[None, :])[0] # (n_var,)
                
                # d) Evaluation
                # Create temp population for evaluation
                off_pop = Population.new(X=np.array([off_X_mut]))
                self.evaluator.eval(self.problem, off_pop)
                off = off_pop[0]

                # Deduplication guard: skip if already in current or new signatures
                off_sig = tuple(off.X.flatten())
                if off_sig in sig_set:
                    dup_skips += 1
                    continue

                # Check for duplicates in current population
                # This prevents the population from collapsing to a single individual
                current_X = pop.get("X")
                if np.any(np.all(current_X == off.X, axis=1)):
                    curpop_dup_skips += 1
                    continue
                
                # e) Update Ideal Point
                self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)
                
                # f) Update Neighbors (Decomposition)
                weights = self.ref_dirs[nbs]
                
                off_fv = self.decomposition.do(off.F, weights, ideal_point=self.ideal_point)
                nbs_F = pop[nbs].get("F")
                nbs_fv = self.decomposition.do(nbs_F, weights, ideal_point=self.ideal_point)
                
                improved_idx = np.where(off_fv < nbs_fv)[0]
                
                if len(improved_idx) > self.n_replace:
                    np.random.shuffle(improved_idx)
                    improved_idx = improved_idx[:self.n_replace]
                
                for idx in improved_idx:
                    pop_idx = nbs[idx]
                    # Final deduplication check against signature set
                    replace_sig = tuple(off.X.flatten())
                    if replace_sig in sig_set:
                        continue
                    pop[pop_idx] = off
                    sig_set.add(replace_sig)
                    replacements_accepted += 1

        # Expose per-generation counters for callbacks/diagnostics
        self.last_counters = {
            "new": replacements_accepted,
            "skip_new": dup_skips,
            "skip_pop": curpop_dup_skips,
        }
        
        self.current_gen += 1
        
        # Progress Bar
        total_gen = self.termination.n_max_gen if hasattr(self.termination, 'n_max_gen') else 300 # Fallback
        percent = (self.current_gen / total_gen) * 100
        bar_length = 30
        filled_length = int(bar_length * self.current_gen // total_gen)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% (Gen {self.current_gen}/{total_gen})')
        sys.stdout.flush()
        if self.current_gen >= total_gen:
            print() # Newline at end
        # Debug counters (lightweight)
        sys.stdout.write(f' | new:{replacements_accepted} skip_new:{dup_skips} skip_pop:{curpop_dup_skips}')
        sys.stdout.flush()

        # 3. Calculate Success Rate and Adapt
        new_X = self.pop.get("X")
        
        # Count how many individuals changed
        diffs = (old_X != new_X)
        changes_per_ind = np.sum(diffs, axis=1)
        n_replacements = np.sum(changes_per_ind > 0)
        
        # Record and adapt probabilities
        self.adaptive_control.record_generation(n_replacements, len(self.pop))
        new_mut_prob, new_cx_prob = self.adaptive_control.update_probabilities()
        
        # Update operator probabilities
        self.mating.mutation.prob = new_mut_prob
        self.mating.crossover.prob = new_cx_prob
        
        # Record for stuck detection
        if self.stuck_detector:
            self.stuck_detector.record_generation(replacements_accepted, None)
            
            # Check if stuck
            try:
                self.stuck_detector.raise_if_stuck(self.current_gen)
            except Exception as e:
                logger.warning(f"Stuck detected: {e}")
                if hasattr(self, "termination") and hasattr(self.termination, "force_termination"):
                    self.termination.force_termination = True
                else:
                    raise


class ARMProblem(Problem):
    def __init__(self, metadata, supports, df, config, validator, metrics, logger):
        self.metadata = metadata
        self.supports = supports
        self.config = config
        self.objectives = config['objectives']['selected']
        self.n_obj = len(self.objectives)
        self.logger = logger
        
        # Determine number of genes from metadata
        dummy = RuleIndividual(metadata=metadata)
        self.n_var = 2 * dummy.num_genes
        
        self.validator = validator
        self.metrics = metrics
        
        # Metric normalization ranges (min, max) for each metric
        # This ensures all objectives are in [0, 1] scale before negation
        self.metric_ranges = {
            # Scenario 1 - Casual ARM
            'casual-supp': (0.0, 1.0),
            'casual-conf': (0.0, 1.0),
            'maxConf': (0.0, 1.0),
            # Scenario 2 - Correlation
            'jaccard': (0.0, 1.0),
            'cosine': (0.0, 1.0),
            'phi': (-1.0, 1.0),  # Range is [-1, 1], needs normalization
            'kappa': (-1.0, 1.0),  # Range is [-1, 1], needs normalization
            'k_measure': (-1.0, 1.0),  # Alias for kappa
            'phi_coefficient': (-1.0, 1.0),  # Alias for phi
        }

        super().__init__(n_var=self.n_var, 
                         n_obj=self.n_obj, 
                         n_ieq_constr=0, 
                         xl=0, 
                         xu=1) # Bounds not strictly used for custom sampling/mutation
    
    def _normalize_metric(self, value: float, metric_name: str) -> float:
        """
        Normalize metric value to [0, 1] range.
        
        Args:
            value: Raw metric value
            metric_name: Name of the metric
        
        Returns:
            Normalized value in [0, 1]
        """
        if metric_name not in self.metric_ranges:
            # Unknown metric, assume [0, 1]
            return value
        
        min_val, max_val = self.metric_ranges[metric_name]
        
        # Normalize to [0, 1]
        if max_val == min_val:
            return 0.5  # Avoid division by zero
        
        normalized = (value - min_val) / (max_val - min_val)
        
        # Clamp to [0, 1] to handle edge cases
        return max(0.0, min(1.0, normalized))

    def _evaluate(self, x, out, *args, **kwargs):
        # x shape: (n_pop, n_var)
        n_pop = x.shape[0]
        F = np.zeros((n_pop, self.n_obj))
        
        for i in range(n_pop):
            ind_genome = x[i]
            
            # Extract rule items using temp individual
            temp_ind = RuleIndividual(self.metadata)
            temp_ind.X = ind_genome
            ant, con = temp_ind.get_rule_items()
            
            # 1. Validate Structure & Constraints
            is_valid, reason, _ = self.validator.validate(ant, con)
            
            if not is_valid:
                # Log invalid individual
                self.logger.log(temp_ind, f"invalid_structure:{reason}")
                
                # Penalty for invalid individuals
                # Assign a value worse than any possible valid metric
                # Metrics are [-1, 1] or [0, 1]. We minimize -Metric.
                # So valid range is [-1, 1].
                # We assign 2.0 to ensure it's dominated by everything.
                F[i, :] = 2.0
                continue

            # 2. Calculate Metrics
            # get_metrics returns (values_list, errors_dict)
            vals, errors = self.metrics.get_metrics(ant, con, self.objectives)
            
            # Extract and normalize objectives
            obj_values = []
            for metric_name, val in zip(self.objectives, vals):
                if val is None:
                    # Penalty for undefined metric
                    obj_values.append(2.0) 
                else:
                    # Normalize to [0, 1] to ensure all metrics are on the same scale
                    # This is CRITICAL for PBI decomposition to work correctly
                    normalized = self._normalize_metric(val, metric_name)
                    
                    # Negate for minimization (pymoo minimizes by default)
                    # Higher metric value → lower objective value → better solution
                    obj_values.append(-normalized)
                
            F[i, :] = obj_values
        
        out["F"] = F
class MOEAD_ARM:
    """
    Wrapper class for MOEA/D algorithm applied to Association Rule Mining.
    Orchestrates the components.
    """
    def __init__(self, config, data_context):
        self.config = config
        self.data = data_context # Dict with df, supports, metadata
        
        # Determine scenario from config
        scenario = self.config['experiment'].get('scenario', 'scenario_1')
        
        # Initialize Components with MetricsFactory
        self.metrics = MetricsFactory.create_metrics(
            scenario_name=scenario,
            dataframe=self.data['df'],
            supports_dict=self.data['supports'],
            metadata=self.data['metadata']
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
            logger=self.logger
        )
        
    def run(self, callback=None):
        # Algorithm Parameters
        alg_config = self.config['algorithm']
        target_pop_size = alg_config['population_size']
        n_gen = alg_config['generations']
        
        # Reference Directions (Decomposition)
        # Calculate H for Das-Dennis
        H = get_H_from_N(target_pop_size, self.problem.n_obj)
        ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=H)
        actual_pop_size = ref_dirs.shape[0]
        
        print(f"Initialized MOEA/D with {actual_pop_size} reference directions (Target: {target_pop_size})")
        
        # Operators
        # Check configuration for initialization method
        init_config = alg_config.get('initialization', {})
        use_pregenerated = init_config.get('use_pregenerated', True)  # Default: True
        pregenerated_path = Path("data/processed/pregenerated/valid_rules_1m.csv")
        
        if use_pregenerated and pregenerated_path.exists():
            print(f"✓ Using pregenerated rules from {pregenerated_path}")
            sampling = PregeneratedSampling(
                metadata=self.data['metadata'],
                csv_path=str(pregenerated_path),
                allow_duplicates=True  # Allow if needed for large populations
            )
        else:
            if use_pregenerated:
                print(f"⚠ Pregenerated rules not found at {pregenerated_path}")
            print(f"→ Using random initialization (slower but more exploratory)")
            max_init_attempts = init_config.get('max_attempts', 10000)
            sampling = ARMSampling(
                metadata=self.data['metadata'],
                validator=self.validator,
                logger=self.logger,
                max_attempts=max_init_attempts
            )
        
        crossover = DiploidNPointCrossover(
            prob=alg_config['operators']['crossover']['probability']['initial']
        )
        
        # Create mutation using factory
        mutation_type = alg_config['operators']['mutation'].get('method', 'mixed')
        mutation = create_mutation(
            mutation_type=mutation_type,
            metadata=self.data['metadata'],
            validator=self.validator,
            logger=self.logger,
            config=alg_config['operators']['mutation']
        )
        
        # Decomposition Method
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
            print(f"Warning: Unknown decomposition method '{method}'. Defaulting to PBI.")
            decomposition = PBI()

        # Algorithm Instance
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
        
        # Termination Criterion
        # Check for early stopping config
        termination_config = alg_config.get('termination', {})
        
        if termination_config.get('enabled', False):
            print(f"Early stopping enabled (ftol={termination_config.get('ftol')}, period={termination_config.get('period')})")
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

        # Execution
        try:
            res = minimize(
                self.problem,
                algorithm,
                termination,
                seed=self.config['experiment']['random_seed'],
                callback=callback,
                verbose=False
            )
        except StuckRunDetected as e:
            logger.warning("Optimization stopped by stuck detector: %s", e)
            print(f"Optimization stopped by stuck detector: {e}")
            current_opt = algorithm.opt if hasattr(algorithm, 'opt') and algorithm.opt is not None else algorithm.pop
            res = SimpleNamespace(opt=current_opt)
        except Exception as e:
            logger.exception("Optimization aborted by unexpected error: %s", e)
            print(f"Optimization aborted by unexpected error: {e}")
            raise
        
        return res

