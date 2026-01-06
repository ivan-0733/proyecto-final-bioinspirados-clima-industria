"""
Pareto Front Analyzer for MOEA/D ARM evolution.

Analyzes distribution quality of Pareto front approximations:
- Spacing: Uniformity of distribution
- Spread: Extent of coverage
- Crowding Distance: Density estimation per solution
"""
from typing import Optional
import numpy as np
from src.core.logging_config import get_logger


class ParetoFrontAnalyzer:
    """
    Analyzes distribution quality of Pareto front approximations.
    
    Computes metrics that assess how well solutions are distributed
    across the Pareto front:
    
    Metrics:
    - Spacing: Uniformity metric (lower = more uniform)
    - Spread: Extent indicator (ratio of coverage)
    - Crowding Distance: Density per solution (for diversity preservation)
    """
    
    def __init__(self):
        """Initialize Pareto front analyzer."""
        self.log = get_logger(__name__)
        self.log.info("pareto_front_analyzer_initialized")
    
    def compute_spacing(self, objectives: np.ndarray) -> float:
        """
        Compute spacing metric (S).
        
        Measures uniformity of distribution. Lower values indicate
        more uniform spacing between neighboring solutions.
        
        S = sqrt( (1/(n-1)) * Σ(d_i - d_mean)^2 )
        where d_i = min distance to other solutions
        
        Args:
            objectives: Objective values (shape: n_solutions × n_objectives)
        
        Returns:
            Spacing metric (non-negative, 0 = perfect uniform spacing)
        """
        n = len(objectives)
        
        if n < 2:
            return 0.0
        
        # Compute minimum distances
        min_distances = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    distances.append(dist)
            min_distances.append(min(distances))
        
        # Compute spacing
        d_mean = np.mean(min_distances)
        spacing = np.sqrt(np.sum((np.array(min_distances) - d_mean) ** 2) / (n - 1))
        
        return float(spacing)
    
    def compute_spread(
        self,
        objectives: np.ndarray,
        reference_extremes: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute spread metric (Δ).
        
        Measures extent of coverage along Pareto front.
        Lower values indicate better coverage.
        
        Δ = (d_f + d_l + Σ|d_i - d_mean|) / (d_f + d_l + (n-1)*d_mean)
        
        Args:
            objectives: Objective values (shape: n_solutions × n_objectives)
            reference_extremes: Optional extreme points for normalization
                               (shape: 2 × n_objectives, [min_point, max_point])
        
        Returns:
            Spread metric (≥0, 0 = ideal spread and distribution)
        """
        n = len(objectives)
        
        if n < 2:
            return 0.0
        
        # If no reference extremes, use objectives' own extremes
        if reference_extremes is None:
            extreme_points = self._find_extreme_points(objectives)
        else:
            extreme_points = reference_extremes
        
        # Compute minimum distances (same as spacing)
        min_distances = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    distances.append(dist)
            min_distances.append(min(distances))
        
        d_mean = np.mean(min_distances)
        
        # Distance to extreme points
        d_f = np.linalg.norm(objectives[0] - extreme_points[0])
        d_l = np.linalg.norm(objectives[-1] - extreme_points[1])
        
        # Spread calculation
        numerator = d_f + d_l + np.sum(np.abs(np.array(min_distances) - d_mean))
        denominator = d_f + d_l + (n - 1) * d_mean
        
        if denominator == 0:
            return 0.0
        
        spread = numerator / denominator
        
        return float(spread)
    
    def _find_extreme_points(self, objectives: np.ndarray) -> np.ndarray:
        """
        Find extreme points (min/max in each objective).
        
        Args:
            objectives: Objective values
        
        Returns:
            Extreme points (shape: 2 × n_objectives)
        """
        min_point = np.min(objectives, axis=0)
        max_point = np.max(objectives, axis=0)
        
        return np.array([min_point, max_point])
    
    def compute_crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        """
        Compute crowding distance for each solution.
        
        Crowding distance estimates density around each solution.
        Higher values indicate more isolated solutions (good for diversity).
        
        Args:
            objectives: Objective values (shape: n_solutions × n_objectives)
        
        Returns:
            Crowding distances (shape: n_solutions,)
        """
        n, m = objectives.shape
        
        if n < 3:
            # Boundary solutions get infinite distance
            return np.full(n, float('inf'))
        
        # Initialize distances
        distances = np.zeros(n)
        
        # For each objective
        for obj_idx in range(m):
            # Sort by this objective
            sorted_indices = np.argsort(objectives[:, obj_idx])
            sorted_values = objectives[sorted_indices, obj_idx]
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Normalize by objective range
            obj_range = sorted_values[-1] - sorted_values[0]
            
            if obj_range == 0:
                continue
            
            # Compute distances for interior solutions
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    (sorted_values[i + 1] - sorted_values[i - 1]) / obj_range
                )
        
        return distances
    
    def compute_avg_crowding_distance(self, objectives: np.ndarray) -> float:
        """
        Compute average crowding distance (excluding infinities).
        
        Args:
            objectives: Objective values
        
        Returns:
            Average crowding distance
        """
        distances = self.compute_crowding_distance(objectives)
        finite_distances = distances[np.isfinite(distances)]
        
        if len(finite_distances) == 0:
            return 0.0
        
        return float(np.mean(finite_distances))
    
    def compute_all_metrics(
        self,
        objectives: np.ndarray,
        reference_extremes: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compute all Pareto front distribution metrics.
        
        Args:
            objectives: Objective values
            reference_extremes: Optional extreme points for spread
        
        Returns:
            Dictionary with spacing, spread, avg_crowding_distance
        """
        metrics = {
            'spacing': self.compute_spacing(objectives),
            'spread': self.compute_spread(objectives, reference_extremes),
            'avg_crowding_distance': self.compute_avg_crowding_distance(objectives)
        }
        
        self.log.debug("pareto_distribution_metrics_computed", **metrics)
        
        return metrics
    
    def is_well_distributed(
        self,
        objectives: np.ndarray,
        spacing_threshold: float = 0.1,
        spread_threshold: float = 0.5
    ) -> bool:
        """
        Check if Pareto front is well distributed.
        
        Args:
            objectives: Objective values
            spacing_threshold: Maximum acceptable spacing
            spread_threshold: Maximum acceptable spread
        
        Returns:
            True if well distributed
        """
        spacing = self.compute_spacing(objectives)
        spread = self.compute_spread(objectives)
        
        return spacing < spacing_threshold and spread < spread_threshold
