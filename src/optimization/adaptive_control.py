"""
Adaptive probability control using 1/5 success rule (Rechenberg).
"""
from dataclasses import dataclass
from typing import Tuple
from src.core.logging_config import get_logger


@dataclass
class ProbabilityConfig:
    """Configuration for adaptive probability."""
    initial: float
    min_val: float
    max_val: float
    increase_factor: float = 1.1
    decrease_factor: float = 0.9


class AdaptiveControl:
    """
    Adaptive probability control using 1/5 success rule.
    
    Increases probability when success rate > 1/5, decreases when < 1/5.
    This balances exploration (mutation) vs exploitation (crossover).
    """
    
    def __init__(
        self,
        mutation_config: ProbabilityConfig,
        crossover_config: ProbabilityConfig
    ):
        """
        Initialize adaptive control.
        
        Args:
            mutation_config: Mutation probability configuration
            crossover_config: Crossover probability configuration
        """
        self.mut_config = mutation_config
        self.cx_config = crossover_config
        
        self.mut_prob = mutation_config.initial
        self.cx_prob = crossover_config.initial
        
        self.success_history = []
        
        self.log = get_logger(__name__)
        
        self.log.info(
            "adaptive_control_initialized",
            mutation_initial=self.mut_prob,
            mutation_range=(mutation_config.min_val, mutation_config.max_val),
            crossover_initial=self.cx_prob,
            crossover_range=(crossover_config.min_val, crossover_config.max_val)
        )
    
    def record_generation(self, num_success: int, num_attempts: int) -> None:
        """
        Record generation results.
        
        Args:
            num_success: Number of successful offspring (new unique valid rules)
            num_attempts: Total attempts this generation
        """
        success_rate = num_success / num_attempts if num_attempts > 0 else 0.0
        self.success_history.append(success_rate)
        
        self.log.debug(
            "generation_recorded",
            success=num_success,
            attempts=num_attempts,
            success_rate=f"{success_rate:.3f}"
        )
    
    def update_probabilities(self) -> Tuple[float, float]:
        """
        Update mutation and crossover probabilities based on 1/5 rule.
        
        Returns:
            Tuple of (new_mutation_prob, new_crossover_prob)
        """
        if len(self.success_history) == 0:
            return self.mut_prob, self.cx_prob
        
        recent_rate = self.success_history[-1]
        target_rate = 0.2  # 1/5 rule
        
        old_mut = self.mut_prob
        old_cx = self.cx_prob
        
        if recent_rate > target_rate:
            # Too many successes → increase mutation (more exploration)
            self.mut_prob = min(
                self.mut_config.max_val,
                self.mut_prob * self.mut_config.increase_factor
            )
            # Decrease crossover (less exploitation)
            self.cx_prob = max(
                self.cx_config.min_val,
                self.cx_prob * self.cx_config.decrease_factor
            )
        elif recent_rate < target_rate:
            # Too few successes → decrease mutation (less exploration)
            self.mut_prob = max(
                self.mut_config.min_val,
                self.mut_prob * self.mut_config.decrease_factor
            )
            # Increase crossover (more exploitation)
            self.cx_prob = min(
                self.cx_config.max_val,
                self.cx_prob * self.cx_config.increase_factor
            )
        
        if old_mut != self.mut_prob or old_cx != self.cx_prob:
            self.log.info(
                "probabilities_adapted",
                success_rate=f"{recent_rate:.3f}",
                mutation={"old": f"{old_mut:.3f}", "new": f"{self.mut_prob:.3f}"},
                crossover={"old": f"{old_cx:.3f}", "new": f"{self.cx_prob:.3f}"}
            )
        
        return self.mut_prob, self.cx_prob
    
    def get_current_probabilities(self) -> Tuple[float, float]:
        """
        Get current probabilities without updating.
        
        Returns:
            Tuple of (mutation_prob, crossover_prob)
        """
        return self.mut_prob, self.cx_prob
    
    def get_statistics(self) -> dict:
        """
        Get statistics about adaptation.
        
        Returns:
            Dictionary with adaptation stats
        """
        if not self.success_history:
            return {
                "generations": 0,
                "avg_success_rate": 0.0,
                "current_mutation_prob": self.mut_prob,
                "current_crossover_prob": self.cx_prob
            }
        
        return {
            "generations": len(self.success_history),
            "avg_success_rate": sum(self.success_history) / len(self.success_history),
            "min_success_rate": min(self.success_history),
            "max_success_rate": max(self.success_history),
            "current_mutation_prob": self.mut_prob,
            "current_crossover_prob": self.cx_prob
        }
