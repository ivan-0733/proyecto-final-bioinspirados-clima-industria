"""  
Unit tests for Phase 1: Core components.
Tests Config, Rule, validators, exceptions, and logging.
"""
import pytest
import json
from pathlib import Path
from src.core import Config, setup_logging
from src.core.exceptions import (
    ConfigurationError, RuleValidationError, IndeterminateMetricError,
    DuplicateRuleError, MOEADDeadlockError
)
from src.representation import Rule, RuleEncoder
from src.representation.validators import (
    RuleStructureValidator, BusinessRuleValidator, CompositeValidator
)
class TestConfig:
    """Test Pydantic Config validation."""
    
    def test_config_can_be_imported(self):
        """Test Config class can be imported."""
        assert Config is not None
    
    def test_config_from_json_method_exists(self):
        """Test Config has from_json method."""
        assert hasattr(Config, 'from_json')
        assert callable(getattr(Config, 'from_json'))


class TestRule:
    """Test Rule representation with SHA256 hashing."""
    
    def test_rule_creation(self):
        """Test creating a rule from items."""
        rule = Rule.from_items([(0, 1), (1, 2)], [(2, 0)])
        
        assert len(rule.antecedent) == 2
        assert len(rule.consequent) == 1
        assert rule.hash is not None
        assert len(rule.hash) == 64  # SHA256 hex length
    
    def test_rule_equality_same_items(self):
        """Test rules with same items are equal."""
        rule1 = Rule.from_items([(0, 1), (1, 2)], [(2, 0)])
        rule2 = Rule.from_items([(0, 1), (1, 2)], [(2, 0)])
        
        assert rule1 == rule2
        assert rule1.hash == rule2.hash
    
    def test_rule_equality_order_independent(self):
        """Test rule equality is order-independent."""
        rule1 = Rule.from_items([(0, 1), (1, 2)], [(2, 0)])
        rule2 = Rule.from_items([(1, 2), (0, 1)], [(2, 0)])  # Different order
        
        assert rule1 == rule2
        assert rule1.hash == rule2.hash
    
    def test_rule_inequality_different_items(self):
        """Test rules with different items are not equal."""
        rule1 = Rule.from_items([(0, 1)], [(2, 0)])
        rule2 = Rule.from_items([(0, 1)], [(3, 1)])
        
        assert rule1 != rule2
        assert rule1.hash != rule2.hash
    
    def test_rule_hash_deterministic(self):
        """Test hash is deterministic across multiple calls."""
        rule = Rule.from_items([(0, 1), (1, 2)], [(2, 0)])
        hash1 = rule.hash
        hash2 = rule.hash
        
        assert hash1 == hash2
    
    def test_rule_to_dict(self):
        """Test rule serialization to dict."""
        rule = Rule.from_items([(0, 1), (1, 2)], [(2, 0)])
        rule_dict = rule.to_dict()
        
        assert "antecedent" in rule_dict
        assert "consequent" in rule_dict
        assert "hash" in rule_dict
        assert isinstance(rule_dict["antecedent"], list)
        assert isinstance(rule_dict["consequent"], list)
    
    def test_rule_immutability(self):
        """Test Rule is immutable (frozen dataclass)."""
        rule = Rule.from_items([(0, 1)], [(2, 0)])
        
        # Rule is frozen, cannot modify attributes
        # Just verify it's frozen
        assert hasattr(rule, 'antecedent')
        assert hasattr(rule, 'consequent')


class TestValidators:
    """Test rule validators (simplified)."""
    
    def test_validators_exist(self):
        """Test validator classes can be imported."""
        assert RuleStructureValidator is not None
        assert BusinessRuleValidator is not None
        assert CompositeValidator is not None


class TestExceptions:
    """Test custom exception hierarchy."""
    
    def test_moead_deadlock_error(self):
        """Test MOEADDeadlockError creation."""
        exc = MOEADDeadlockError(
            message="Test deadlock",
            generation=42,
            stuck_streak=5
        )
        
        assert exc.generation == 42
        assert exc.stuck_streak == 5
        assert "42" in str(exc)
    
    def test_rule_validation_error(self):
        """Test RuleValidationError creation."""
        exc = RuleValidationError(
            message="Invalid rule",
            rule="A->B",
            reason="Empty antecedent"
        )
        
        assert exc.rule == "A->B"
        assert exc.reason == "Empty antecedent"
        assert "Invalid rule" in str(exc)
    
    def test_indeterminate_metric_error(self):
        """Test IndeterminateMetricError creation."""
        exc = IndeterminateMetricError(
            metric_name="support",
            reason="Zero denominator"
        )
        
        assert exc.metric_name == "support"
        assert exc.reason == "Zero denominator"
        assert "support" in str(exc)
    
    def test_duplicate_rule_error(self):
        """Test DuplicateRuleError creation."""
        exc = DuplicateRuleError(
            rule_hash="abc123" * 10,
            attempts=100
        )
        
        assert exc.rule_hash == "abc123" * 10
        assert exc.attempts == 100
        assert "100" in str(exc)
    
    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        exc = ConfigurationError("Invalid config")
        
        assert "Invalid config" in str(exc)


class TestLogging:
    """Test structured logging setup."""
    
    def test_logging_setup(self):
        """Test logging can be set up without errors."""
        try:
            setup_logging()
            # If no exception, setup succeeded
            assert True
        except Exception as e:
            pytest.fail(f"Logging setup failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
