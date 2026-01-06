"""
Refactored mutation with circuit breaker pattern and timeout.
"""
import time
import logging
import numpy as np
from typing import Dict, Any, List
from pymoo.core.mutation import Mutation

from src.core.logging_config import get_logger
from src.representation import RuleIndividual


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker opens due to repeated failures."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent infinite retry loops.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, stop trying
    - HALF_OPEN: Testing if system recovered
    """
    
    def __init__(self, failure_threshold: int = 10, timeout: float = 5.0):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening
            timeout: Seconds before trying again
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        
        self.log = get_logger(__name__)
    
    def call(self, func, *args, **kwargs):
        """
        Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        if self.state == "OPEN":
            # Check if timeout elapsed
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                self.log.info("circuit_breaker_half_open")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker open after {self.failures} failures"
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset or close
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
                self.log.info("circuit_breaker_closed")
            
            return result
            
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                self.log.error(
                    "circuit_breaker_opened",
                    failures=self.failures,
                    error=str(e)
                )
            
            raise
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.failures = 0
        self.state = "CLOSED"


class ARMMutation(Mutation):
    """
    Refactored mutation with circuit breaker and timeout protection.
    
    Improvements over legacy:
    - Circuit breaker prevents infinite loops
    - Per-attempt timeout
    - Better retry strategy
    - Structured logging
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        validator: Any,
        logger: Any,
        prob: float = 0.1,
        repair_attempts: int = 5,
        duplicate_attempts: int = 5,
        timeout_per_attempt: float = 10.0,
        active_ops: List[str] = None
    ):
        """
        Initialize mutation operator.
        
        Args:
            metadata: Dataset metadata
            validator: Rule validator
            logger: Discarded rules logger
            prob: Mutation probability
            repair_attempts: Max validation retries
            duplicate_attempts: Extra retries for duplicates
            timeout_per_attempt: Timeout per mutation attempt
            active_ops: Active operations (extension/contraction/replacement)
        """
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.discard_logger = logger
        self.prob = prob
        self.repair_attempts = repair_attempts
        self.duplicate_attempts = duplicate_attempts
        self.timeout_per_attempt = timeout_per_attempt
        
        # Active operations
        self.available_ops = ['extension', 'contraction', 'replacement']
        if active_ops is None:
            self.active_ops = self.available_ops
        else:
            self.active_ops = [op for op in active_ops if op in self.available_ops]
            if not self.active_ops:
                raise ValueError("At least one valid mutation operator must be active.")
        
        # Circuit breaker per operation type
        self.circuit_breakers = {
            op: CircuitBreaker(failure_threshold=20, timeout=5.0)
            for op in self.active_ops
        }
        
        # Helper data
        dummy_ind = RuleIndividual(metadata)
        self.variables_info = dummy_ind.variables_info
        self.num_genes = dummy_ind.num_genes
        
        self.log = get_logger(__name__)
    
    def _do(self, problem, X, **kwargs):
        """
        Apply mutation with timeout and circuit breaker protection.
        
        Args:
            problem: Optimization problem
            X: Genomes to mutate (n_ind, n_var)
            **kwargs: Additional arguments
        
        Returns:
            Mutated genomes
        """
        n_ind, n_var = X.shape
        n_genes = n_var // 2
        
        # Reshape to (n_ind, 2, n_genes) for easier manipulation
        Y = X.reshape(n_ind, 2, n_genes).copy()
        
        # Track batch signatures for duplicate detection
        batch_signatures = set()
        for i in range(n_ind):
            ind = RuleIndividual(self.metadata)
            ind.X = Y[i].flatten()
            try:
                rule = ind.to_rule()
                batch_signatures.add(rule.hash)
            except Exception:
                pass
        
        for i in range(n_ind):
            if np.random.random() < self.prob:
                original = Y[i].copy()
                success = False
                start_time = time.time()
                
                validation_left = self.repair_attempts
                duplicate_left = self.duplicate_attempts
                
                while (validation_left > 0 or duplicate_left > 0):
                    # Timeout check
                    if time.time() - start_time >= self.timeout_per_attempt:
                        self.log.warning(
                            "mutation_timeout",
                            individual=i,
                            elapsed=f"{time.time() - start_time:.2f}s"
                        )
                        break
                    
                    mutant = original.copy()
                    
                    # Select operation
                    op = np.random.choice(self.active_ops)
                    
                    try:
                        # Apply with circuit breaker protection
                        self.circuit_breakers[op].call(
                            self._apply_operation,
                            mutant,
                            op
                        )
                        
                        # Repair structure
                        self._repair_structure(mutant)
                        
                        # Extract and validate
                        ind = RuleIndividual(self.metadata)
                        ind.X = mutant.flatten()
                        
                        try:
                            rule = ind.to_rule()
                            rule_hash = rule.hash
                        except Exception:
                            validation_left -= 1
                            continue
                        
                        # Check duplicates in batch
                        if rule_hash in batch_signatures:
                            if duplicate_left > 0:
                                duplicate_left -= 1
                                continue
                            else:
                                validation_left -= 1
                                continue
                        
                        # Validate
                        ant, con = ind.get_rule_items()
                        is_valid, reason, metrics = self.validator.validate(ant, con)
                        
                        if is_valid:
                            Y[i] = mutant
                            batch_signatures.add(rule_hash)
                            success = True
                            break
                        else:
                            validation_left -= 1
                            # Log failure
                            self.discard_logger.log(ind, f"mutation_fail_{op}:{reason}", metrics)
                    
                    except CircuitBreakerOpen:
                        # Circuit breaker opened, try different operation
                        self.log.warning("circuit_breaker_open", operation=op)
                        # Remove this op temporarily
                        remaining_ops = [o for o in self.active_ops if o != op]
                        if remaining_ops:
                            op = np.random.choice(remaining_ops)
                        else:
                            break
                        validation_left -= 1
                    
                    except Exception as e:
                        self.log.warning("mutation_error", error=str(e), operation=op)
                        validation_left -= 1
                
                if not success:
                    Y[i] = original
                    self.log.debug(
                        "mutation_failed_all_attempts",
                        individual=i,
                        operations=self.active_ops,
                        repair_attempts=self.repair_attempts,
                        duplicate_attempts=self.duplicate_attempts
                    )
        
        return Y.reshape(n_ind, n_var)
    
    def _apply_operation(self, genome: np.ndarray, operation: str) -> None:
        """
        Apply mutation operation to genome.
        
        Args:
            genome: Genome array (2, n_genes)
            operation: Operation name
        """
        if operation == 'extension':
            self._apply_extension(genome)
        elif operation == 'contraction':
            self._apply_contraction(genome)
        elif operation == 'replacement':
            self._apply_replacement(genome)
    
    def _apply_extension(self, genome: np.ndarray) -> None:
        """Activate a random inactive gene."""
        inactive_indices = np.where(genome[0] == 0)[0]
        if len(inactive_indices) > 0:
            idx = np.random.choice(inactive_indices)
            genome[0, idx] = np.random.choice([1, 2])
            card = self.variables_info[idx]['cardinality']
            genome[1, idx] = np.random.randint(0, card)
    
    def _apply_contraction(self, genome: np.ndarray) -> None:
        """Deactivate a random active gene."""
        active_indices = np.where(genome[0] != 0)[0]
        if len(active_indices) > 0:
            idx = np.random.choice(active_indices)
            genome[0, idx] = 0
            genome[1, idx] = 0
    
    def _apply_replacement(self, genome: np.ndarray) -> None:
        """Change value of a random active gene."""
        active_indices = np.where(genome[0] != 0)[0]
        if len(active_indices) > 0:
            idx = np.random.choice(active_indices)
            card = self.variables_info[idx]['cardinality']
            if card > 1:
                current_val = genome[1, idx]
                possible_vals = list(range(card))
                if current_val in possible_vals:
                    possible_vals.remove(current_val)
                if possible_vals:
                    genome[1, idx] = np.random.choice(possible_vals)
    
    def _repair_structure(self, genome: np.ndarray) -> None:
        """Ensure at least one antecedent and consequent."""
        roles = genome[0]
        has_ant = np.any(roles == 1)
        has_con = np.any(roles == 2)
        
        if not has_ant:
            zeros = np.where(roles == 0)[0]
            if len(zeros) > 0:
                idx = np.random.choice(zeros)
                genome[0, idx] = 1
                card = self.variables_info[idx]['cardinality']
                genome[1, idx] = np.random.randint(0, card)
            else:
                cons = np.where(roles == 2)[0]
                if len(cons) > 0:
                    idx = np.random.choice(cons)
                    genome[0, idx] = 1
        
        if not has_con:
            roles = genome[0]
            zeros = np.where(roles == 0)[0]
            if len(zeros) > 0:
                idx = np.random.choice(zeros)
                genome[0, idx] = 2
                card = self.variables_info[idx]['cardinality']
                genome[1, idx] = np.random.randint(0, card)
            else:
                ants = np.where(roles == 1)[0]
                if len(ants) > 0:
                    idx = np.random.choice(ants)
                    genome[0, idx] = 2
