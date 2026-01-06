"""
Hypervolume Tracker for MOEA/D ARM evolution.

Tracks hypervolume indicator over generations using pymoo's HV implementation.
Supports plateau detection for stuck run diagnostics.
"""
from typing import List, Optional, Dict, Any
import numpy as np
from pymoo.indicators.hv import HV
from src.core.logging_config import get_logger


class HypervolumeTracker:
    """
    Tracks hypervolume evolution over generations.
    
    Uses pymoo's HV indicator to compute dominated hypervolume
    relative to a reference point. Maintains history for:
    - Trend analysis (improvement/stagnation)
    - Plateau detection (stuck runs)
    - Progress reporting
    
    Attributes:
        ref_point: Reference point for HV calculation (nadir point)
        history: List of (generation, hv_value) tuples
        window_size: Window for plateau detection
        tolerance: Threshold for plateau detection (ΔHV < tol)
    """
    
    def __init__(
        self,
        ref_point: np.ndarray,
        window_size: int = 10,
        tolerance: float = 1e-4
    ):
        """
        Initialize hypervolume tracker.
        
        Args:
            ref_point: Reference point for HV calculation (shape: n_objectives)
            window_size: Sliding window size for plateau detection
            tolerance: Plateau threshold (max - min < tol in window)
        
        Raises:
            ValueError: If ref_point is invalid or window_size < 2
        """
        if not isinstance(ref_point, np.ndarray):
            ref_point = np.array(ref_point)
        
        if ref_point.ndim != 1 or len(ref_point) < 2:
            raise ValueError(
                f"ref_point must be 1D array with ≥2 objectives, got shape {ref_point.shape}"
            )
        
        if window_size < 2:
            raise ValueError(f"window_size must be ≥2, got {window_size}")
        
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        
        self.ref_point = ref_point
        self.window_size = window_size
        self.tolerance = tolerance
        
        # Initialize HV indicator
        self._hv_indicator = HV(ref_point=ref_point)
        
        # History: [(generation, hv_value), ...]
        self.history: List[tuple[int, float]] = []
        
        self.log = get_logger(__name__)
        self.log.info(
            "hypervolume_tracker_initialized",
            ref_point=ref_point.tolist(),
            window_size=window_size,
            tolerance=tolerance
        )
    
    def compute(self, objectives: np.ndarray) -> float:
        """
        Compute hypervolume for given objective values.
        
        Args:
            objectives: Objective values (shape: n_solutions × n_objectives)
        
        Returns:
            Hypervolume value (non-negative)
        
        Raises:
            ValueError: If objectives shape is invalid
        """
        if objectives.ndim != 2:
            raise ValueError(
                f"objectives must be 2D array, got shape {objectives.shape}"
            )
        
        if objectives.shape[1] != len(self.ref_point):
            raise ValueError(
                f"objectives has {objectives.shape[1]} columns but ref_point "
                f"has {len(self.ref_point)} dimensions"
            )
        
        if len(objectives) == 0:
            return 0.0
        
        try:
            hv_value = self._hv_indicator(objectives)
            return float(hv_value)
        except Exception as e:
            self.log.error(
                "hypervolume_computation_failed",
                error=str(e),
                objectives_shape=objectives.shape
            )
            return 0.0
    
    def record(self, generation: int, objectives: np.ndarray) -> float:
        """
        Compute and record hypervolume for a generation.
        
        Args:
            generation: Current generation number
            objectives: Objective values to evaluate
        
        Returns:
            Computed hypervolume value
        """
        hv_value = self.compute(objectives)
        self.history.append((generation, hv_value))
        
        self.log.debug(
            "hypervolume_recorded",
            generation=generation,
            hv_value=hv_value,
            n_solutions=len(objectives)
        )
        
        return hv_value
    
    def is_plateau(self, window: Optional[int] = None) -> bool:
        """
        Check if HV has plateaued in recent window.
        
        A plateau is detected when max - min < tolerance in the window.
        
        Args:
            window: Window size (uses self.window_size if None)
        
        Returns:
            True if plateau detected, False otherwise
        """
        if window is None:
            window = self.window_size
        
        if len(self.history) < window:
            return False
        
        recent_values = [hv for _, hv in self.history[-window:]]
        delta = max(recent_values) - min(recent_values)
        
        return delta < self.tolerance
    
    def get_trend(self, window: Optional[int] = None) -> Optional[float]:
        """
        Compute HV trend (average improvement per generation).
        
        Uses linear regression slope on recent window.
        
        Args:
            window: Window size (uses self.window_size if None)
        
        Returns:
            Slope (HV improvement per generation), or None if insufficient data
        """
        if window is None:
            window = self.window_size
        
        if len(self.history) < 2:
            return None
        
        recent = self.history[-window:]
        if len(recent) < 2:
            return None
        
        # Simple linear regression: slope = Δy / Δx
        generations = np.array([gen for gen, _ in recent])
        hv_values = np.array([hv for _, hv in recent])
        
        # Use numpy's polyfit for robust slope estimation
        slope, _ = np.polyfit(generations, hv_values, deg=1)
        
        return float(slope)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of HV evolution.
        
        Returns:
            Dictionary with keys:
            - total_generations: Number of recorded generations
            - initial_hv: First HV value (or None)
            - final_hv: Last HV value (or None)
            - max_hv: Maximum HV achieved
            - improvement: Total improvement (final - initial)
            - trend: Recent trend (slope)
            - is_plateau: Whether currently plateaued
        """
        if not self.history:
            return {
                'total_generations': 0,
                'initial_hv': None,
                'final_hv': None,
                'max_hv': None,
                'improvement': None,
                'trend': None,
                'is_plateau': False
            }
        
        hv_values = [hv for _, hv in self.history]
        
        return {
            'total_generations': len(self.history),
            'initial_hv': hv_values[0],
            'final_hv': hv_values[-1],
            'max_hv': max(hv_values),
            'improvement': hv_values[-1] - hv_values[0],
            'trend': self.get_trend(),
            'is_plateau': self.is_plateau()
        }
    
    def clear(self) -> None:
        """Clear all recorded history."""
        self.history.clear()
        self.log.info("hypervolume_history_cleared")
    
    def get_history(self) -> List[tuple[int, float]]:
        """
        Get full HV history.
        
        Returns:
            List of (generation, hv_value) tuples
        """
        return self.history.copy()
