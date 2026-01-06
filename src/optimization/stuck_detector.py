"""
Stuck detection for early stopping.
"""
import time
from typing import Optional
from src.core.logging_config import get_logger
from src.core.exceptions import MOEADDeadlockError


class StuckDetector:
    """
    Detects when evolution is stuck and should stop early.
    
    Monitors:
    - Runtime limit (max minutes)
    - Window-based new rule generation (min new rules per window)
    - Hypervolume stagnation (tolerance over period)
    """
    
    def __init__(
        self,
        max_runtime_minutes: Optional[float] = None,
        window_size: int = 10,
        min_new_per_window: int = 5,
        hv_tolerance: float = 1e-6,
        hv_period: int = 20
    ):
        """
        Initialize stuck detector.
        
        Args:
            max_runtime_minutes: Maximum runtime in minutes (None = no limit)
            window_size: Number of generations for new rule check
            min_new_per_window: Minimum new rules in window to continue
            hv_tolerance: Minimum hypervolume improvement
            hv_period: Generations to check HV improvement
        """
        self.max_runtime_minutes = max_runtime_minutes
        self.window_size = window_size
        self.min_new_per_window = min_new_per_window
        self.hv_tolerance = hv_tolerance
        self.hv_period = hv_period
        
        self.start_time = time.time()
        self.new_rules_window = []
        self.hv_history = []
        self.stuck_streak = 0
        
        self.log = get_logger(__name__)
        
        self.log.info(
            "stuck_detector_initialized",
            max_runtime_minutes=max_runtime_minutes,
            window_size=window_size,
            min_new_per_window=min_new_per_window,
            hv_tolerance=hv_tolerance,
            hv_period=hv_period
        )
    
    def record_generation(self, num_new: int, hypervolume: Optional[float] = None) -> None:
        """
        Record generation statistics.
        
        Args:
            num_new: Number of new unique rules this generation
            hypervolume: Current hypervolume (optional)
        """
        self.new_rules_window.append(num_new)
        if len(self.new_rules_window) > self.window_size:
            self.new_rules_window.pop(0)
        
        if hypervolume is not None:
            self.hv_history.append(hypervolume)
    
    def check_stuck(self, generation: int) -> tuple[bool, Optional[str]]:
        """
        Check if evolution is stuck.
        
        Args:
            generation: Current generation number
        
        Returns:
            Tuple of (is_stuck, reason)
        """
        # Check runtime limit
        if self.max_runtime_minutes is not None:
            elapsed_minutes = (time.time() - self.start_time) / 60.0
            if elapsed_minutes >= self.max_runtime_minutes:
                reason = f"Runtime limit ({self.max_runtime_minutes:.1f} min) exceeded"
                self.log.warning(
                    "stuck_detected_runtime",
                    elapsed_minutes=f"{elapsed_minutes:.2f}",
                    limit_minutes=self.max_runtime_minutes,
                    generation=generation
                )
                return True, reason
        
        # Check new rules window
        if len(self.new_rules_window) >= self.window_size:
            total_new = sum(self.new_rules_window)
            if total_new < self.min_new_per_window:
                self.stuck_streak += 1
                reason = f"Only {total_new} new rules in last {self.window_size} generations (min: {self.min_new_per_window})"
                self.log.warning(
                    "stuck_detected_new_rules",
                    total_new=total_new,
                    window_size=self.window_size,
                    min_required=self.min_new_per_window,
                    generation=generation,
                    stuck_streak=self.stuck_streak
                )
                # Allow some streak before stopping (transient plateaus are OK)
                if self.stuck_streak >= 3:
                    return True, reason
            else:
                self.stuck_streak = 0
        
        # Check hypervolume stagnation
        if len(self.hv_history) >= self.hv_period:
            recent_hv = self.hv_history[-self.hv_period:]
            hv_improvement = max(recent_hv) - min(recent_hv)
            
            if hv_improvement < self.hv_tolerance:
                reason = f"Hypervolume stagnant (improvement: {hv_improvement:.2e} < {self.hv_tolerance:.2e})"
                self.log.warning(
                    "stuck_detected_hypervolume",
                    hv_improvement=f"{hv_improvement:.2e}",
                    tolerance=f"{self.hv_tolerance:.2e}",
                    period=self.hv_period,
                    generation=generation
                )
                return True, reason
        
        return False, None
    
    def raise_if_stuck(self, generation: int) -> None:
        """
        Raise exception if stuck.
        
        Args:
            generation: Current generation number
        
        Raises:
            MOEADDeadlockError: If stuck condition detected
        """
        is_stuck, reason = self.check_stuck(generation)
        if is_stuck:
            self.log.error(
                "evolution_stuck",
                generation=generation,
                reason=reason,
                elapsed_minutes=f"{(time.time() - self.start_time) / 60.0:.2f}"
            )
            raise MOEADDeadlockError(
                message=reason or "Unknown stuck condition",
                generation=generation,
                stuck_streak=self.stuck_streak
            )
    
    def get_statistics(self) -> dict:
        """
        Get stuck detection statistics.
        
        Returns:
            Dictionary with statistics
        """
        elapsed_minutes = (time.time() - self.start_time) / 60.0
        
        stats = {
            "elapsed_minutes": elapsed_minutes,
            "stuck_streak": self.stuck_streak,
            "new_rules_window": list(self.new_rules_window),
            "window_total_new": sum(self.new_rules_window) if self.new_rules_window else 0
        }
        
        if self.hv_history:
            stats["hypervolume_current"] = self.hv_history[-1]
            if len(self.hv_history) >= 2:
                stats["hypervolume_improvement"] = self.hv_history[-1] - self.hv_history[0]
        
        return stats
