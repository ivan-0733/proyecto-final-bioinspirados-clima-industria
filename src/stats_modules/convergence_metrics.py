"""
Convergence Metrics for MOEA/D ARM evolution.

Computes distance-based metrics to assess convergence quality:
- GD (Generational Distance): Average distance from approximation to reference
- IGD (Inverted Generational Distance): Average distance from reference to approximation
- Max Distance: Worst-case distance metric
"""
from typing import Optional
import numpy as np
from src.core.logging_config import get_logger


class ConvergenceMetrics:
    """
    Computes convergence metrics against reference Pareto front.
    
    Uses distance-based metrics to measure how close the approximation
    set is to a known or estimated reference front.
    
    Metrics:
    - GD: Average distance from approximation points to reference
    - IGD: Average distance from reference points to approximation
    - Max GD: Maximum distance (worst-case convergence)
    - Max IGD: Maximum distance (worst-case coverage)
    
    Lower values indicate better convergence/coverage.
    """
    
    def __init__(self, reference_front: Optional[np.ndarray] = None):
        """
        Initialize convergence metrics calculator.
        
        Args:
            reference_front: Reference Pareto front (shape: n_points × n_objectives)
                            Can be None and set later via set_reference()
        """
        self.reference_front = reference_front
        self.log = get_logger(__name__)
        
        if reference_front is not None:
            self._validate_reference(reference_front)
            self.log.info(
                "convergence_metrics_initialized",
                reference_size=len(reference_front),
                n_objectives=reference_front.shape[1]
            )
        else:
            self.log.info("convergence_metrics_initialized", reference="not_set")
    
    def _validate_reference(self, reference: np.ndarray) -> None:
        """
        Validate reference front format.
        
        Args:
            reference: Reference front array
        
        Raises:
            ValueError: If reference format is invalid
        """
        if reference.ndim != 2:
            raise ValueError(
                f"Reference front must be 2D array, got shape {reference.shape}"
            )
        
        if reference.shape[1] < 2:
            raise ValueError(
                f"Reference front must have ≥2 objectives, got {reference.shape[1]}"
            )
        
        if len(reference) == 0:
            raise ValueError("Reference front cannot be empty")
    
    def set_reference(self, reference_front: np.ndarray) -> None:
        """
        Set or update reference Pareto front.
        
        Args:
            reference_front: Reference front (shape: n_points × n_objectives)
        
        Raises:
            ValueError: If reference format is invalid
        """
        self._validate_reference(reference_front)
        self.reference_front = reference_front
        
        self.log.info(
            "reference_front_updated",
            reference_size=len(reference_front),
            n_objectives=reference_front.shape[1]
        )
    
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two points.
        
        Args:
            point1: First point (1D array)
            point2: Second point (1D array)
        
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(point1 - point2))
    
    def _min_distance_to_set(self, point: np.ndarray, point_set: np.ndarray) -> float:
        """
        Find minimum Euclidean distance from point to any point in set.
        
        Args:
            point: Single point (1D array)
            point_set: Set of points (2D array)
        
        Returns:
            Minimum distance to set
        """
        distances = np.linalg.norm(point_set - point, axis=1)
        return float(np.min(distances))
    
    def compute_gd(self, approximation: np.ndarray, p: int = 2) -> float:
        """
        Compute Generational Distance (GD).
        
        GD measures average distance from approximation set to reference front.
        Lower is better (0 = perfect convergence).
        
        GD = (1/|A|) * (Σ d_i^p)^(1/p)
        where d_i = min distance from point i in A to reference
        
        Args:
            approximation: Approximation set (shape: n_points × n_objectives)
            p: Distance exponent (default 2 for Euclidean)
        
        Returns:
            GD value (non-negative, 0 = perfect)
        
        Raises:
            ValueError: If reference not set or shapes incompatible
        """
        if self.reference_front is None:
            raise ValueError("Reference front not set. Call set_reference() first.")
        
        if approximation.ndim != 2:
            raise ValueError(
                f"Approximation must be 2D array, got shape {approximation.shape}"
            )
        
        if approximation.shape[1] != self.reference_front.shape[1]:
            raise ValueError(
                f"Approximation has {approximation.shape[1]} objectives but "
                f"reference has {self.reference_front.shape[1]}"
            )
        
        if len(approximation) == 0:
            return 0.0
        
        # For each point in approximation, find min distance to reference
        distances = []
        for point in approximation:
            min_dist = self._min_distance_to_set(point, self.reference_front)
            distances.append(min_dist ** p)
        
        gd = (sum(distances) / len(approximation)) ** (1.0 / p)
        
        return float(gd)
    
    def compute_igd(self, approximation: np.ndarray, p: int = 2) -> float:
        """
        Compute Inverted Generational Distance (IGD).
        
        IGD measures average distance from reference front to approximation set.
        Lower is better (0 = perfect coverage).
        
        IGD = (1/|R|) * (Σ d_i^p)^(1/p)
        where d_i = min distance from point i in R to approximation
        
        Args:
            approximation: Approximation set (shape: n_points × n_objectives)
            p: Distance exponent (default 2 for Euclidean)
        
        Returns:
            IGD value (non-negative, 0 = perfect)
        
        Raises:
            ValueError: If reference not set or shapes incompatible
        """
        if self.reference_front is None:
            raise ValueError("Reference front not set. Call set_reference() first.")
        
        if approximation.ndim != 2:
            raise ValueError(
                f"Approximation must be 2D array, got shape {approximation.shape}"
            )
        
        if approximation.shape[1] != self.reference_front.shape[1]:
            raise ValueError(
                f"Approximation has {approximation.shape[1]} objectives but "
                f"reference has {self.reference_front.shape[1]}"
            )
        
        if len(approximation) == 0:
            # No approximation → infinite IGD (worst case)
            return float('inf')
        
        # For each point in reference, find min distance to approximation
        distances = []
        for point in self.reference_front:
            min_dist = self._min_distance_to_set(point, approximation)
            distances.append(min_dist ** p)
        
        igd = (sum(distances) / len(self.reference_front)) ** (1.0 / p)
        
        return float(igd)
    
    def compute_max_gd(self, approximation: np.ndarray) -> float:
        """
        Compute maximum GD (worst-case distance).
        
        Returns the maximum distance from any approximation point
        to the reference front.
        
        Args:
            approximation: Approximation set
        
        Returns:
            Maximum distance
        """
        if self.reference_front is None:
            raise ValueError("Reference front not set.")
        
        if len(approximation) == 0:
            return 0.0
        
        max_dist = 0.0
        for point in approximation:
            min_dist = self._min_distance_to_set(point, self.reference_front)
            max_dist = max(max_dist, min_dist)
        
        return float(max_dist)
    
    def compute_max_igd(self, approximation: np.ndarray) -> float:
        """
        Compute maximum IGD (worst-case coverage gap).
        
        Returns the maximum distance from any reference point
        to the approximation set.
        
        Args:
            approximation: Approximation set
        
        Returns:
            Maximum distance
        """
        if self.reference_front is None:
            raise ValueError("Reference front not set.")
        
        if len(approximation) == 0:
            return float('inf')
        
        max_dist = 0.0
        for point in self.reference_front:
            min_dist = self._min_distance_to_set(point, approximation)
            max_dist = max(max_dist, min_dist)
        
        return float(max_dist)
    
    def compute_all_metrics(self, approximation: np.ndarray) -> dict:
        """
        Compute all convergence metrics.
        
        Args:
            approximation: Approximation set
        
        Returns:
            Dictionary with GD, IGD, Max GD, Max IGD
        """
        metrics = {
            'gd': self.compute_gd(approximation),
            'igd': self.compute_igd(approximation),
            'max_gd': self.compute_max_gd(approximation),
            'max_igd': self.compute_max_igd(approximation)
        }
        
        self.log.debug("convergence_metrics_computed", **metrics)
        
        return metrics
