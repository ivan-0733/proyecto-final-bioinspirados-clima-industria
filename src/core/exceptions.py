"""
Custom exception hierarchy for MOEA/D ARM system.
"""


class MOEADError(Exception):
    """Base exception for all MOEA/D ARM errors."""
    pass


class MOEADDeadlockError(MOEADError):
    """Raised when the algorithm appears stuck in an infinite loop."""
    def __init__(self, message: str, generation: int, stuck_streak: int):
        self.generation = generation
        self.stuck_streak = stuck_streak
        super().__init__(f"Deadlock detected at gen {generation}: {message}")


class RuleValidationError(MOEADError):
    """Raised when a rule fails validation."""
    def __init__(self, message: str, rule: str = None, reason: str = None):
        self.rule = rule
        self.reason = reason
        super().__init__(message)


class ConfigurationError(MOEADError):
    """Raised when configuration is invalid or incomplete."""
    pass


class IndeterminateMetricError(MOEADError):
    """Raised when a metric calculation results in undefined values."""
    def __init__(self, metric_name: str, reason: str):
        self.metric_name = metric_name
        self.reason = reason
        super().__init__(f"Metric {metric_name} is indeterminate: {reason}")


class DuplicateRuleError(MOEADError):
    """Raised when duplicate rule is detected during generation."""
    def __init__(self, rule_hash: str, attempts: int):
        self.rule_hash = rule_hash
        self.attempts = attempts
        super().__init__(
            f"Duplicate rule detected after {attempts} attempts (hash: {rule_hash[:16]}...)"
        )


class InitializationError(MOEADError):
    """Raised when population initialization fails."""
    def __init__(self, message: str, valid_count: int, required: int):
        self.valid_count = valid_count
        self.required = required
        super().__init__(
            f"{message} (Generated {valid_count}/{required} valid individuals)"
        )
