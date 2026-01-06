"""
Population Statistics for MOEA/D ARM evolution.

Computes diversity metrics for population analysis:
- Genotypic diversity: Unique rules, Hamming distance
- Phenotypic diversity: Objective space spread
- Shannon entropy: Distribution of genotype patterns
"""
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
from src.core.logging_config import get_logger


class PopulationStats:
    """
    Computes diversity metrics for ARM populations.
    
    Tracks both genotypic (genome) and phenotypic (objective) diversity
    to assess population health and premature convergence.
    
    Metrics:
    - unique_count: Number of unique genotypes
    - duplicate_rate: Fraction of duplicate individuals
    - avg_hamming: Average Hamming distance between genotypes
    - shannon_entropy: Entropy of genotype distribution
    - objective_spread: Standard deviation in objective space
    """
    
    def __init__(self):
        """Initialize population statistics tracker."""
        self.log = get_logger(__name__)
        self.log.info("population_stats_initialized")
    
    def compute_unique_count(self, genotypes: List[str]) -> int:
        """
        Count unique genotypes in population.
        
        Args:
            genotypes: List of genotype identifiers (e.g., rule hashes)
        
        Returns:
            Number of unique genotypes
        """
        return len(set(genotypes))
    
    def compute_duplicate_rate(self, genotypes: List[str]) -> float:
        """
        Compute fraction of duplicate individuals.
        
        Args:
            genotypes: List of genotype identifiers
        
        Returns:
            Duplicate rate in [0, 1] (0 = all unique, 1 = all duplicates)
        """
        if len(genotypes) == 0:
            return 0.0
        
        unique = len(set(genotypes))
        total = len(genotypes)
        
        # duplicate_rate = 1 - (unique / total)
        return 1.0 - (unique / total)
    
    def compute_hamming_distance(
        self,
        genome1: np.ndarray,
        genome2: np.ndarray
    ) -> int:
        """
        Compute Hamming distance between two genomes.
        
        Args:
            genome1: First genome (1D array)
            genome2: Second genome (1D array)
        
        Returns:
            Number of differing positions
        
        Raises:
            ValueError: If genomes have different lengths
        """
        if len(genome1) != len(genome2):
            raise ValueError(
                f"Genomes must have same length: {len(genome1)} != {len(genome2)}"
            )
        
        return int(np.sum(genome1 != genome2))
    
    def compute_avg_hamming(self, genomes: np.ndarray) -> float:
        """
        Compute average pairwise Hamming distance.
        
        Args:
            genomes: Population genomes (shape: n_individuals × genome_length)
        
        Returns:
            Average Hamming distance across all pairs
        """
        n = len(genomes)
        
        if n < 2:
            return 0.0
        
        total_distance = 0
        pair_count = 0
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                total_distance += self.compute_hamming_distance(genomes[i], genomes[j])
                pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0
    
    def compute_shannon_entropy(self, genotypes: List[str]) -> float:
        """
        Compute Shannon entropy of genotype distribution.
        
        Higher entropy indicates more diverse population.
        H = -Σ p_i * log2(p_i)
        
        Args:
            genotypes: List of genotype identifiers
        
        Returns:
            Shannon entropy (non-negative, max = log2(n_unique))
        """
        if len(genotypes) == 0:
            return 0.0
        
        # Count frequencies
        counts = Counter(genotypes)
        total = len(genotypes)
        
        # Compute entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return float(entropy)
    
    def compute_objective_spread(self, objectives: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation in each objective dimension.
        
        Measures phenotypic diversity (spread in objective space).
        
        Args:
            objectives: Objective values (shape: n_individuals × n_objectives)
        
        Returns:
            Standard deviation per objective (shape: n_objectives)
        """
        if len(objectives) == 0:
            return np.array([])
        
        return np.std(objectives, axis=0)
    
    def compute_all_metrics(
        self,
        genotypes: Optional[List[str]] = None,
        genomes: Optional[np.ndarray] = None,
        objectives: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute all diversity metrics.
        
        Args:
            genotypes: List of genotype identifiers (for unique count, entropy)
            genomes: Population genomes (for Hamming distance)
            objectives: Objective values (for spread)
        
        Returns:
            Dictionary with computed metrics
        """
        metrics = {}
        
        if genotypes is not None:
            metrics['unique_count'] = self.compute_unique_count(genotypes)
            metrics['duplicate_rate'] = self.compute_duplicate_rate(genotypes)
            metrics['shannon_entropy'] = self.compute_shannon_entropy(genotypes)
            metrics['population_size'] = len(genotypes)
        
        if genomes is not None:
            metrics['avg_hamming_distance'] = self.compute_avg_hamming(genomes)
        
        if objectives is not None:
            spread = self.compute_objective_spread(objectives)
            for i, std in enumerate(spread):
                metrics[f'objective_{i}_spread'] = float(std)
            metrics['avg_objective_spread'] = float(np.mean(spread)) if len(spread) > 0 else 0.0
        
        self.log.debug("population_metrics_computed", **metrics)
        
        return metrics
    
    def is_converged(
        self,
        genotypes: List[str],
        unique_threshold: float = 0.1
    ) -> bool:
        """
        Check if population has converged (low diversity).
        
        Args:
            genotypes: List of genotype identifiers
            unique_threshold: Minimum fraction of unique individuals (default 0.1)
        
        Returns:
            True if unique_rate < threshold (converged)
        """
        if len(genotypes) == 0:
            return False
        
        unique_rate = self.compute_unique_count(genotypes) / len(genotypes)
        
        return unique_rate < unique_threshold
    
    def get_diversity_summary(
        self,
        genotypes: List[str],
        genomes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get human-readable diversity summary.
        
        Args:
            genotypes: List of genotype identifiers
            genomes: Optional genomes for Hamming distance
        
        Returns:
            Dictionary with summary statistics
        """
        unique = self.compute_unique_count(genotypes)
        total = len(genotypes)
        
        summary = {
            'population_size': total,
            'unique_individuals': unique,
            'unique_rate': unique / total if total > 0 else 0.0,
            'duplicate_rate': self.compute_duplicate_rate(genotypes),
            'shannon_entropy': self.compute_shannon_entropy(genotypes),
            'is_diverse': unique / total > 0.5 if total > 0 else False
        }
        
        if genomes is not None:
            summary['avg_hamming_distance'] = self.compute_avg_hamming(genomes)
        
        return summary
