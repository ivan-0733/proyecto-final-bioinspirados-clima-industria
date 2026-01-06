"""
Rule structure validators with single responsibility.
"""
from typing import List, Tuple, Set, Optional, Dict, Any
from dataclasses import dataclass

from src.core.exceptions import RuleValidationError
from src.representation.rule import Rule


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class RuleStructureValidator:
    """
    Validates rule structure (non-empty, disjoint, cardinality bounds).
    
    Follows Single Responsibility Principle: only structural validation,
    no metric-based validation.
    """
    
    def __init__(
        self,
        min_antecedent_items: int = 1,
        max_antecedent_items: int = 10,
        min_consequent_items: int = 1,
        max_consequent_items: int = 10
    ):
        """
        Initialize validator with cardinality constraints.
        
        Args:
            min_antecedent_items: Minimum items in antecedent
            max_antecedent_items: Maximum items in antecedent
            min_consequent_items: Minimum items in consequent
            max_consequent_items: Maximum items in consequent
        """
        self.min_ant = min_antecedent_items
        self.max_ant = max_antecedent_items
        self.min_con = min_consequent_items
        self.max_con = max_consequent_items
    
    def validate(self, rule: Rule) -> ValidationResult:
        """
        Validate rule structure.
        
        Checks:
        1. Non-empty antecedent and consequent
        2. Disjoint variable sets
        3. Cardinality bounds
        
        Args:
            rule: Rule to validate
        
        Returns:
            ValidationResult with is_valid flag and optional reason
        """
        # Check 1: Non-empty
        if not rule.antecedent:
            return ValidationResult(
                is_valid=False,
                reason="empty_antecedent",
                details={"antecedent_size": 0}
            )
        
        if not rule.consequent:
            return ValidationResult(
                is_valid=False,
                reason="empty_consequent",
                details={"consequent_size": 0}
            )
        
        # Check 2: Cardinality bounds
        ant_size = len(rule.antecedent)
        con_size = len(rule.consequent)
        
        if not (self.min_ant <= ant_size <= self.max_ant):
            return ValidationResult(
                is_valid=False,
                reason="antecedent_cardinality_violation",
                details={
                    "actual": ant_size,
                    "min": self.min_ant,
                    "max": self.max_ant
                }
            )
        
        if not (self.min_con <= con_size <= self.max_con):
            return ValidationResult(
                is_valid=False,
                reason="consequent_cardinality_violation",
                details={
                    "actual": con_size,
                    "min": self.min_con,
                    "max": self.max_con
                }
            )
        
        # Check 3: Disjoint variables
        ant_vars = {var_idx for var_idx, _ in rule.antecedent}
        con_vars = {var_idx for var_idx, _ in rule.consequent}
        
        if not ant_vars.isdisjoint(con_vars):
            overlap = ant_vars & con_vars
            return ValidationResult(
                is_valid=False,
                reason="non_disjoint_variables",
                details={"overlapping_variables": sorted(overlap)}
            )
        
        return ValidationResult(is_valid=True)


class BusinessRuleValidator:
    """
    Validates business logic constraints (domain-specific).
    
    Examples:
    - Fixed consequents (e.g., demographic variables shouldn't be predicted)
    - Forbidden variable pairs (e.g., "pregnant" + "male")
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        fixed_antecedents: Optional[List[str]] = None,
        fixed_consequents: Optional[List[str]] = None,
        forbidden_pairs: Optional[List[List[str]]] = None
    ):
        """
        Initialize business rule validator.
        
        Args:
            metadata: Dataset metadata with variable mapping
            fixed_antecedents: Variables that cannot be in antecedent
            fixed_consequents: Variables that cannot be in consequent
            forbidden_pairs: Pairs of variables that cannot coexist
        """
        self.metadata = metadata
        self.fixed_antecedents = set(fixed_antecedents or [])
        self.fixed_consequents = set(fixed_consequents or [])
        self.forbidden_pairs = [set(pair) for pair in (forbidden_pairs or [])]
        
        # Build variable name lookup
        self.var_names = self._get_variable_names()
    
    def _get_variable_names(self) -> List[str]:
        """Extract ordered variable names from metadata."""
        order = list(self.metadata.get('feature_order', []))
        target_name = self.metadata.get('target_variable')
        
        if isinstance(target_name, dict):
            target_name = target_name.get('name')
        if target_name and target_name not in order:
            order.append(target_name)
        
        return order
    
    def validate(self, rule: Rule) -> ValidationResult:
        """
        Validate business logic constraints.
        
        Args:
            rule: Rule to validate
        
        Returns:
            ValidationResult
        """
        # Extract variable names
        ant_var_indices = {var_idx for var_idx, _ in rule.antecedent}
        con_var_indices = {var_idx for var_idx, _ in rule.consequent}
        
        ant_var_names = {self.var_names[idx] for idx in ant_var_indices if idx < len(self.var_names)}
        con_var_names = {self.var_names[idx] for idx in con_var_indices if idx < len(self.var_names)}
        
        # Check 1: Fixed antecedents (variables that cannot be in antecedent)
        if not ant_var_names.isdisjoint(self.fixed_antecedents):
            forbidden = ant_var_names & self.fixed_antecedents
            return ValidationResult(
                is_valid=False,
                reason="fixed_antecedent_violation",
                details={"forbidden_variables": sorted(forbidden)}
            )
        
        # Check 2: Fixed consequents (variables that cannot be in consequent)
        if not con_var_names.isdisjoint(self.fixed_consequents):
            forbidden = con_var_names & self.fixed_consequents
            return ValidationResult(
                is_valid=False,
                reason="fixed_consequent_violation",
                details={"forbidden_variables": sorted(forbidden)}
            )
        
        # Check 3: Forbidden pairs
        all_var_names = ant_var_names | con_var_names
        for forbidden_pair in self.forbidden_pairs:
            if forbidden_pair.issubset(all_var_names):
                return ValidationResult(
                    is_valid=False,
                    reason="forbidden_pair_violation",
                    details={"forbidden_pair": sorted(forbidden_pair)}
                )
        
        return ValidationResult(is_valid=True)


class CompositeValidator:
    """
    Composite validator that chains multiple validators.
    
    Follows Open/Closed Principle: extensible without modification.
    """
    
    def __init__(self, validators: List[Any]):
        """
        Initialize composite validator.
        
        Args:
            validators: List of validator instances with validate() method
        """
        self.validators = validators
    
    def validate(self, rule: Rule) -> ValidationResult:
        """
        Run all validators in sequence, short-circuit on first failure.
        
        Args:
            rule: Rule to validate
        
        Returns:
            First failing ValidationResult, or success if all pass
        """
        for validator in self.validators:
            result = validator.validate(rule)
            if not result.is_valid:
                return result
        
        return ValidationResult(is_valid=True)
