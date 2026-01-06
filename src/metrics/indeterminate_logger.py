"""
Logger for indeterminate metrics (division by zero, None values).
"""
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from src.core.logging_config import get_logger


class IndeterminateMetricsLogger:
    """
    Tracks and logs indeterminate metric calculations.
    
    Useful for debugging why certain metrics return None
    (e.g., zero support, division by zero).
    """
    
    def __init__(self):
        """Initialize indeterminate logger."""
        self.log = get_logger(__name__)
        
        # Counter for each error type per metric
        # {metric_name: {error_reason: count}}
        self.error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Sample rules with errors (limit to avoid memory explosion)
        # {metric_name: [(rule_signature, error_reason)]}
        self.sample_errors: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.max_samples = 100
        
        self.log.info("indeterminate_metrics_logger_initialized")
    
    def log_indeterminate(
        self,
        metric_name: str,
        error_reason: str,
        rule_signature: Optional[str] = None
    ) -> None:
        """
        Log indeterminate metric calculation.
        
        Args:
            metric_name: Name of the metric
            error_reason: Reason for indeterminate value
            rule_signature: Optional rule identifier
        """
        self.error_counts[metric_name][error_reason] += 1
        
        # Store sample
        if len(self.sample_errors[metric_name]) < self.max_samples:
            if rule_signature:
                self.sample_errors[metric_name].append((rule_signature, error_reason))
    
    def get_summary(self) -> dict:
        """
        Get summary of indeterminate metrics.
        
        Returns:
            Dictionary with error counts and samples
        """
        summary = {
            "total_metrics_with_errors": len(self.error_counts),
            "metrics": {}
        }
        
        for metric, errors in self.error_counts.items():
            total_errors = sum(errors.values())
            summary["metrics"][metric] = {
                "total_errors": total_errors,
                "error_breakdown": dict(errors),
                "sample_count": len(self.sample_errors[metric])
            }
        
        return summary
    
    def log_summary(self) -> None:
        """Log summary to structured logger."""
        summary = self.get_summary()
        
        if summary["total_metrics_with_errors"] > 0:
            self.log.warning(
                "indeterminate_metrics_summary",
                **summary
            )
        else:
            self.log.info("no_indeterminate_metrics_detected")
    
    def get_top_errors(self, metric_name: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Get top N error reasons for a metric.
        
        Args:
            metric_name: Metric to query
            top_n: Number of top errors to return
        
        Returns:
            List of (error_reason, count) tuples sorted by count
        """
        if metric_name not in self.error_counts:
            return []
        
        errors = self.error_counts[metric_name]
        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
        return sorted_errors[:top_n]
    
    def clear(self) -> None:
        """Clear all logged data."""
        self.error_counts.clear()
        self.sample_errors.clear()
        self.log.info("indeterminate_logger_cleared")
