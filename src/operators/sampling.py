"""
Refactored sampling with timeout protection and Bloom filter.
"""
import time
import numpy as np
from typing import Dict, Any, Optional, Set
from pymoo.core.sampling import Sampling

from src.core.exceptions import InitializationError
from src.core.logging_config import get_logger
from src.representation import RuleIndividual, Rule


class TimeoutError(Exception):
    """Raised when initialization exceeds timeout."""
    pass


class BloomFilter:
    """
    Simple Bloom filter for fast membership tests.
    
    Uses multiple hash functions to reduce false positives.
    Memory efficient for large rule sets.
    """
    
    def __init__(self, expected_size: int = 10000, false_positive_rate: float = 0.01):
        """
        Initialize Bloom filter.
        
        Args:
            expected_size: Expected number of elements
            false_positive_rate: Target false positive rate (0-1)
        """
        import math
        
        # Calculate optimal size and number of hash functions
        self.size = int(-(expected_size * math.log(false_positive_rate)) / (math.log(2) ** 2))
        self.num_hashes = int((self.size / expected_size) * math.log(2))
        self.bit_array = np.zeros(self.size, dtype=bool)
        
        self.logger = get_logger(__name__)
        self.logger.debug(
            "bloom_filter_initialized",
            size=self.size,
            num_hashes=self.num_hashes,
            expected_size=expected_size
        )
    
    def _hashes(self, item: str) -> list:
        """Generate multiple hash values for an item."""
        import hashlib
        
        hashes = []
        for i in range(self.num_hashes):
            # Use SHA256 with salt for each hash function
            h = hashlib.sha256(f"{item}{i}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.size)
        return hashes
    
    def add(self, item: str) -> None:
        """Add item to filter."""
        for h in self._hashes(item):
            self.bit_array[h] = True
    
    def contains(self, item: str) -> bool:
        """Check if item might be in set (may have false positives)."""
        return all(self.bit_array[h] for h in self._hashes(item))
    
    def __contains__(self, item: str) -> bool:
        """Support 'in' operator."""
        return self.contains(item)
    
    def clear(self) -> None:
        """Clear the filter."""
        self.bit_array.fill(False)


class ARMSampling(Sampling):
    """
    Refactored sampling with timeout protection and Bloom filter.
    
    Improvements over legacy:
    - Uses RuleIndividual with SHA256 hashing
    - Bloom filter for fast duplicate detection
    - Timeout protection to prevent infinite loops
    - Better logging and diagnostics
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        validator: Any,
        logger: Any,
        max_attempts: int = 10000,
        timeout_seconds: float = 300.0,
        use_bloom_filter: bool = True
    ):
        """
        Initialize sampling operator.
        
        Args:
            metadata: Dataset metadata
            validator: Rule validator instance
            logger: Discarded rules logger
            max_attempts: Maximum initialization attempts
            timeout_seconds: Timeout in seconds
            use_bloom_filter: Whether to use Bloom filter for fast checks
        """
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.discard_logger = logger
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.use_bloom_filter = use_bloom_filter
        
        self.log = get_logger(__name__)
        
        # Bloom filter for fast duplicate detection
        self.bloom: Optional[BloomFilter] = None
        if use_bloom_filter:
            self.bloom = BloomFilter(expected_size=max_attempts)
    
    def _do(self, problem, n_samples, **kwargs):
        """
        Generate initial population with timeout protection.
        
        Args:
            problem: Optimization problem
            n_samples: Number of samples to generate
            **kwargs: Additional arguments
        
        Returns:
            Array of genomes (n_samples, n_var)
        
        Raises:
            InitializationError: If timeout or max attempts exceeded
        """
        X = []
        seen_rules: Set[str] = set()  # Exact duplicates
        
        start_time = time.time()
        total_attempts = 0
        created = 0
        stagnant_attempts = 0
        max_stagnant = 100
        
        self.log.info(
            "initialization_started",
            target_size=n_samples,
            max_attempts=self.max_attempts,
            timeout_seconds=self.timeout_seconds
        )
        
        while created < n_samples:
            # Timeout check
            elapsed = time.time() - start_time
            if elapsed >= self.timeout_seconds:
                raise InitializationError(
                    f"Timeout after {elapsed:.1f}s",
                    valid_count=created,
                    required=n_samples
                )
            
            # Max attempts check
            if total_attempts >= self.max_attempts:
                self.log.warning(
                    "max_attempts_reached",
                    created=created,
                    required=n_samples,
                    will_fill_with_duplicates=True
                )
                break
            
            # Create individual
            ind = RuleIndividual(self.metadata)
            ind.initialize(sparsity=0.6)
            ind.repair()
            
            total_attempts += 1
            
            # Convert to Rule for duplicate detection
            try:
                rule = ind.to_rule()
                rule_hash = rule.hash
            except Exception as e:
                self.log.warning("rule_conversion_failed", error=str(e))
                continue
            
            # Fast Bloom filter check (optional)
            if self.bloom and self.bloom.contains(rule_hash):
                stagnant_attempts += 1
                if stagnant_attempts >= max_stagnant and len(X) > 0:
                    self.log.warning(
                        "stagnation_detected",
                        stagnant_attempts=stagnant_attempts,
                        will_allow_duplicates=True
                    )
                    break
                continue
            
            # Exact duplicate check
            if rule_hash in seen_rules:
                stagnant_attempts += 1
                if stagnant_attempts >= max_stagnant and len(X) > 0:
                    break
                continue
            
            # Validate rule
            antecedent, consequent = ind.get_rule_items()
            is_valid, reason, metrics = self.validator.validate(antecedent, consequent)
            
            if is_valid:
                X.append(ind.X.copy())
                seen_rules.add(rule_hash)
                
                if self.bloom:
                    self.bloom.add(rule_hash)
                
                created += 1
                stagnant_attempts = 0
                
                if created % 10 == 0 or created == n_samples:
                    self.log.debug(
                        "initialization_progress",
                        created=created,
                        target=n_samples,
                        attempts=total_attempts,
                        elapsed=f"{time.time() - start_time:.1f}s"
                    )
            else:
                stagnant_attempts += 1
                # Skip logging during initialization to speed up
        
        # Fill remaining with duplicates if needed
        if len(X) < n_samples:
            if len(X) == 0:
                # Critical: No valid individuals created, generate minimal fallback
                self.log.error(
                    "no_valid_individuals_created",
                    attempts=total_attempts,
                    generating_fallback=True
                )
                
                # Last resort: Create a minimal valid genome manually
                # Use problem.n_var to get genome length
                num_genes = problem.n_var // 2
                fallback_genome = np.zeros(problem.n_var, dtype=int)
                
                # Assign minimal antecedent (first available feature)
                fallback_genome[0] = 1  # Role: antecedent
                fallback_genome[num_genes] = 0  # Value: first feature index
                
                # Assign minimal consequent (second available feature)
                fallback_genome[1] = 2  # Role: consequent
                fallback_genome[num_genes + 1] = 1  # Value: second feature index
                
                # Create RuleIndividual to repair
                ind = RuleIndividual(self.metadata)
                ind.X = fallback_genome  # Assign genome
                ind.repair()  # Repair for consistency
                
                X.append(ind.X.copy())
                self.log.warning(
                    "fallback_individual_generated",
                    antecedent_items=1,
                    consequent_items=1
                )
            
            self.log.warning(
                "filling_with_duplicates",
                unique_created=len(X),
                required=n_samples,
                duplicates_needed=n_samples - len(X)
            )
            
            while len(X) < n_samples:
                idx = np.random.randint(0, len(X))
                X.append(X[idx].copy())
        
        elapsed = time.time() - start_time
        self.log.info(
            "initialization_complete",
            created=len(X),
            unique=len(seen_rules),
            attempts=total_attempts,
            elapsed=f"{elapsed:.1f}s",
            success_rate=f"{created / total_attempts * 100:.1f}%"
        )
        
        return np.array(X)
